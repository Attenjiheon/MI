"""Independent token-level holdout/coverage audit and strict pair-depth validation."""
import argparse,gzip,json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.generate_v1_4_corpus import validate_stored_pair
from corpus.replay import replay

def check(tokens,composition=None,length=False):
    state=[tokens[3+3*i]-13 for i in range(4)]; states={tuple(state)}; commands=set(); truths=set(); adjacent=set()
    pos=13; block=[]; blocks=[]; dsts=[]; queries=[]
    while tokens[pos]!=2:
        op=tokens[pos];d=tokens[pos+1]-9
        if op==8:
            blocks.append(tuple(block));queries.append(d);block=[];pos+=3;continue
        before=state.copy();block.append(op);dsts.append(d)
        if op==3:x=tokens[pos+2]-13;state[d]=x
        elif op==4:x=None;state[d]=1-state[d]
        else:
            x=tokens[pos+2]-9;truths.add((op,state[d],state[x]))
            state[d]=state[d]&state[x] if op==5 else state[d]|state[x] if op==6 else state[d]^state[x]
        commands.add((op,d,x));states.add(tuple(state));pos+=2 if op==4 else 3
    assert 33<=len(blocks)<=48 if length else 8<=len(blocks)<=24
    patterns=((7,5,6),(6,7,5));found=[i for i,b in enumerate(blocks) if b in patterns]
    if composition is None:assert not found
    else:
        assert len(found)==1 and blocks[found[0]]==patterns[composition]
        offset=sum(len(b) for b in blocks[:found[0]])
        assert len(set(dsts[offset:offset+3]))==1 and queries[found[0]]==dsts[offset]
    for b in blocks:adjacent.update(zip(b,b[1:]))
    return states,commands,truths,adjacent


def run(output):
    assert not output.exists()
    data=ROOT/json.loads((ROOT/'experiment_v1_4/corpus_rebuild.json').read_text())['active_data_root']
    coverage=[set() for _ in range(4)];counts={}
    for p in sorted(data.rglob('*.tokens.jsonl')):
        if 'provenance' in p.parts:continue
        comp=int(p.name.split('_')[1].split('.')[0]) if p.parent.name=='composition' else None
        length=p.name=='length.tokens.jsonl';n=0
        with p.open() as f:
            for line in f:
                values=check(json.loads(line),comp,length);n+=1
                if p.parent.name=='train_shards':
                    for a,b in zip(coverage,values):a.update(b)
        counts[str(p.relative_to(data))]=n
    assert list(map(len,coverage))==[16,48,12,25]
    pair_counts={}
    for split in ('select','gate','test'):
        n=0
        with gzip.open(data/f'first_repeat/{split}.jsonl.gz','rt') as f:
            for line in f:
                pair=json.loads(line);validate_stored_pair(pair,split)
                check(pair['origin']['token_ids']);check(pair['repeat']['token_ids']);n+=1
        pair_counts[split]=n
    causal=0
    for p in sorted((data/'causal_pairs').glob('*.jsonl.gz')):
        if p.name.endswith('.origins.jsonl.gz'):
            with gzip.open(p,'rt') as f:
                for line in f:check(json.loads(line)['token_ids'])
            continue
        with gzip.open(p,'rt') as f:
            for line in f:
                pair=json.loads(line);a=pair['original_prefix_ids'];b=pair['counterfactual_prefix_ids']
                diffs=[i for i in range(len(a)) if a[i]!=b[i]]
                assert len(a)==len(b) and diffs==[pair['modified_token_index']]
                pos=diffs[0];assert {a[pos],b[pos]}=={13,14} and a[pos-2]==3
                pa,pb=replay(a,partial=True),replay(b,partial=True)
                assert pa['reads']==pb['reads'] or [r['answer'] for r in pa['reads']]==[r['answer'] for r in pb['reads']]
                assert pa['answer']==pair['original_answer'] and pb['answer']==pair['counterfactual_answer']
                assert (pa['answer']!=pb['answer'])==pair['changed_target'];causal+=1
    output.write_text(json.dumps(dict(status='passed',holdout_and_length_checked_sequences=sum(counts.values()),train_coverage=dict(zip(('states','commands','truth_table','adjacent_operators'),map(len,coverage))),first_repeat_raw_token_depth_and_insertion=pair_counts,causal_single_bit_change_pairs=causal),indent=2)+'\n')
    print(output.read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
