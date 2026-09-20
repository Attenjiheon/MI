"""Replay PCG64 sampling without expensive metadata to recover per-target attempts.

Matches stored origins, target IDs, total draws/rejections and final RNG states.
Does not regenerate or overwrite any corpus. Numba only accelerates CPU sampling.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from corpus.v1_4 import cells, rng, sample_pair


@njit
def candidate(gen, wanted_op, wanted_pattern, wanted_depth):
    state=np.empty(4,np.int64)
    for i in range(4): state[i]=gen.integers(0,2)
    nblocks=gen.integers(8,25)
    tokens=np.zeros(302,np.int64); tokens[0]=1; pos=1
    for d in range(4):
        tokens[pos]=3;tokens[pos+1]=9+d;tokens[pos+2]=13+state[d];pos+=3
    depths=np.zeros(4,np.int64); lastop=np.full(4,-1,np.int64)
    patterns=np.zeros(4,np.int64); seen=np.zeros(4,np.int64)
    lastblock=np.full(4,-1,np.int64); lastoffset=np.zeros(4,np.int64)
    splitok=np.zeros(4,np.bool_); matches=np.zeros(24,np.int64); nm=0; holdout=False
    for bid in range(nblocks):
        u=gen.random(); g=1 if u<.8 else 2 if u<.95 else 3
        ops=np.zeros(3,np.int64)
        for offset in range(g):
            v=gen.random(); op=0 if v<.25 else 1 if v<.5 else 2 if v<2/3 else 3 if v<5/6 else 4
            d=gen.integers(0,4); x=0
            if op==0: x=gen.integers(0,2)
            elif op!=1:
                x=gen.integers(0,3)
                if x>=d: x+=1
            ops[offset]=op
            tokens[pos]=3+op;tokens[pos+1]=9+d;pos+=2
            if op!=1:
                tokens[pos]=13+x if op==0 else 9+x;pos+=1
            patterns[d]=state[d] if op==1 else state[d]*2+state[x] if op>=2 else 0
            if op==0: state[d]=x;depths[d]=0
            elif op==1: state[d]=1-state[d];depths[d]+=1
            else:
                depths[d]=1+max(depths[d],depths[x])
                if op==2: state[d]=state[d]&state[x]
                elif op==3: state[d]=state[d]|state[x]
                else: state[d]=state[d]^state[x]
            lastop[d]=op;seen[d]=0;lastblock[d]=bid;lastoffset[d]=offset
        if gen.random()<.5: q=d
        else:
            q=gen.integers(0,3)
            if q>=d:q+=1
        if g==3 and ((ops[0]==4 and ops[1]==2 and ops[2]==3) or (ops[0]==3 and ops[1]==4 and ops[2]==2)):
            holdout=True
        for var in range(4):
            if lastblock[var]==bid:splitok[var]=lastoffset[var]+1<g and q!=var
        tokens[pos]=8;tokens[pos+1]=9+q;tokens[pos+2]=13+state[q];pos+=3
        db=1 if depths[q]==1 else 2 if depths[q]<=3 else 3
        if lastop[q]==wanted_op and patterns[q]==wanted_pattern and db==wanted_depth and seen[q]==0 and splitok[q]:
            matches[nm]=bid;nm+=1
        seen[q]+=1
    tokens[pos]=2;pos+=1
    if holdout:return -1,tokens,pos,-1
    if nblocks>=24 or nm==0:return 0,tokens,pos,-1
    rid=matches[gen.integers(0,nm)]
    return 1,tokens,pos,rid


@njit
def replay_cell(gen,op,pattern,depth,expected,lengths,rids,limit):
    accepted=0; total=0; last=0; holds=0; noeligible=0; collisions=0
    gaps=np.zeros(len(lengths),np.int64)
    while accepted<len(lengths) and total<limit:
        status,tokens,n,rid=candidate(gen,op,pattern,depth);total+=1
        if status==-1:holds+=1;continue
        if status==0:noeligible+=1;continue
        same=n==lengths[accepted] and rid==rids[accepted]
        if same:
            for i in range(n):
                if tokens[i]!=expected[accepted,i]:same=False;break
        if same:
            gaps[accepted]=total-last;last=total;accepted+=1
        else:collisions+=1
    return accepted,total,gaps,holds,noeligible,collisions


def args(cell):
    return ('SET','NOT','AND','OR','XOR').index(cell.operator),int(cell.input_pattern,2),{'1':1,'2-3':2,'4+':3}[cell.depth]


def selfcheck():
    # For all cells, compare to the unchanged production sampler before trusting JIT.
    for cell in cells():
        a=rng('attempt_audit_selfcheck',cell.index);b=rng('attempt_audit_selfcheck',cell.index)
        pair,used,rejects=sample_pair(a,'select',cell)
        for count in range(1,100001):
            status,tokens,n,rid=candidate(b,*args(cell))
            if status==1:break
        assert count==used
        assert tokens[:n].tolist()==pair['origin']['token_ids']
        assert rid==pair['origin_target_read_id']
        assert a.bit_generator.state==b.bit_generator.state
    print('42-cell sampler/RNG equivalence passed',flush=True)


def run(output):
    output.mkdir(parents=True,exist_ok=False); selfcheck();start=time.monotonic()
    root=ROOT/json.loads((ROOT/'experiment_v1_4/corpus_rebuild.json').read_text())['active_data_root']
    manifest=json.loads((root/'manifest.json').read_text()); results={}
    for split in ('select','gate','test'):
        pairs=defaultdict(list)
        with gzip.open(root/f'first_repeat/{split}.jsonl.gz','rt') as f:
            for line in f:
                p=json.loads(line);pairs[p['cell_id']].append(p)
        for cell in cells():
            group=pairs[cell.cell_id];expected=np.zeros((len(group),302),np.int64)
            lengths=np.array([len(p['origin']['token_ids']) for p in group]);rids=np.array([p['origin_target_read_id'] for p in group])
            for i,p in enumerate(group):expected[i,:lengths[i]]=p['origin']['token_ids']
            info=manifest['splits']['first_repeat/'+split]['cells'][cell.cell_id]
            gen=rng(split+'_first_repeat',cell.index)
            n,total,gaps,holds,noeligible,collisions=replay_cell(gen,*args(cell),expected,lengths,rids,info['attempts'])
            assert n==len(group) and total==info['attempts'],(split,cell.cell_id,n,total)
            assert gen.bit_generator.state==info['rng_final_state'],(split,cell.cell_id,'rng')
            rejects=info['rejections']
            assert holds==rejects.get('holdout',0) and noeligible==rejects.get('no_eligible_target',0)
            assert collisions==rejects.get('read_prefix_collision',0)+rejects.get('duplicate_sequence',0)
            record=dict(split=split,cell=cell.cell_id,accepted=n,attempts=total,attempts_per_target=gaps.tolist(),max_attempts=int(gaps.max()),within_cap=bool(gaps.max()<=100000),final_rng_matches=True,rejections_match=True)
            results[split+'/'+cell.cell_id]=record
            (output/f'{split}_{cell.index:02d}.json').write_text(json.dumps(record,indent=2)+'\n')
            print(split,cell.cell_id,'max attempts',int(gaps.max()),'elapsed',round(time.monotonic()-start,1),flush=True)
    report=dict(status='passed' if all(r['within_cap'] for r in results.values()) else 'failed',max_attempts=max(r['max_attempts'] for r in results.values()),cap=100000,targets=sum(r['accepted'] for r in results.values()),cells=len(results),rng_and_rejections_match=True,sampler_selfcheck_cells=42,elapsed_seconds=time.monotonic()-start)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)
