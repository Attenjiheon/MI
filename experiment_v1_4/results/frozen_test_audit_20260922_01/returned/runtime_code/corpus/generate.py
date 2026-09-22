"""Run: python -m corpus.generate --output data/language_v1"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import platform
import numpy as np
from .language import (VOCAB, OPS, PATTERNS, MASTER, seed, digest, execute, sample,
                       render, pattern_ok, select_target, all_commands)
from .replay import replay, validate, TABLES

MAX_ATTEMPTS = 100_000


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def self_test():
    n = 0
    for state in itertools.product((0,1), repeat=4):
        for op,d,x in all_commands():
            before = list(state); expected = before.copy()
            if op == 'SET': expected[d] = x
            elif op == 'NOT': expected[d] = (1,0)[before[d]]
            else: expected[d] = TABLES[VOCAB.index(op)][before[d]*2+before[x]]
            assert execute(before,(op,d,x)) == expected
            assert before == list(state)
            if op in ('NOT','XOR'): assert execute(execute(before,(op,d,x)),(op,d,x)) == before
            n += 1
    samples = []
    for _ in range(2):
        rng = np.random.Generator(np.random.PCG64(seed('val_iid')))
        records = []
        for _ in range(8):
            initial,blocks,_ = sample(rng)
            e = render(initial,blocks,'self_test',seed('val_iid'),True)
            validate(e); records.append(e)
        samples.append(hashlib.sha256(dump(records).encode()).hexdigest())
    assert samples[0] == samples[1]
    return {'exhaustive_transitions':n, 'not_xor_involution':'passed',
            'fixed_seed_token_and_metadata_sha256':samples[0], 'reproducibility':'passed'}


class Stats:
    def __init__(self):
        self.c = {k:Counter() for k in ('length','blocks','operators','commands','dst','query',
            'answers','target_answers','truth_table','states','structural_depth','distance','variable_values',
            'adjacent_operators','updates_per_block','set_literals','read_same_dst','initial_bits')}
        self.n = self.tokens = self.answers = self.first = self.updates = 0
    def add(self,e):
        self.n += 1; self.tokens += len(e['token_ids']); self.answers += len(e['read_events'])
        self.updates += len(e['update_events'])
        self.c['length'][len(e['token_ids'])] += 1; self.c['blocks'][len(e['blocks'])] += 1
        for v,b in zip('ABCD',e['initial_state']): self.c['initial_bits'][f'{v}:{b}'] += 1
        self.c['states'][''.join(map(str,e['initial_state']))] += 1
        for b in e['blocks']:
            ops = [e['update_events'][i]['op'] for i in b['update_ids']]
            self.c['updates_per_block'][len(ops)] += 1
            for a,z in zip(ops,ops[1:]): self.c['adjacent_operators'][f'{a},{z}'] += 1
        for u in e['update_events']:
            self.c['operators'][u['op']] += 1; self.c['dst'][u['dst']] += 1
            operand = u['literal_or_null'] if u['op']=='SET' else u['src_or_null']
            command = f"{u['op']} {u['dst']}" + ('' if operand is None else f' {operand}')
            self.c['commands'][command] += 1
            if u['op']=='SET': self.c['set_literals'][operand] += 1
            if u['input_truth_pattern_or_null'] is not None:
                self.c['truth_table'][f"{u['op']}:{u['input_truth_pattern_or_null']}"] += 1
            self.c['states'][''.join(map(str,u['state_after']))] += 1
        for r in e['read_events']:
            self.c['answers'][r['answer']] += 1; self.c['query'][r['query_var']] += 1
            self.c['structural_depth'][r['structural_depth']] += 1
            self.c['distance'][r['updates_since_last_update']] += 1
            self.c['read_same_dst'][int(r['same_as_last_dst'])] += 1
            for v,b in zip('ABCD',r['state_at_read']): self.c['variable_values'][f'{v}:{b}'] += 1
            self.first += int(r['reads_of_query_var_since_latest_set']==0)
            if r['answer_is_target']: self.c['target_answers'][r['answer']] += 1
    def result(self):
        return dict(sequences=self.n,tokens=self.tokens,prediction_tokens=self.tokens-self.n,
            reads=self.answers,updates=self.updates,first_read_fraction=self.first/max(1,self.answers),
            answer_token_fraction=self.answers/max(1,self.tokens),
            majority_class_baseline=max(self.c['answers'].values(),default=0)/max(1,self.answers),
            histograms={k:dict(v) for k,v in self.c.items()},
            coverage={k:len(self.c[k]) for k in ('commands','states','truth_table','adjacent_operators')})


class Builder:
    def __init__(self,out,smoke=False):
        self.out=out; self.smoke=smoke; self.hashes=set(); self.prefixes=set(); self.causal_prefixes=set()
        self.manifest={}; self.statistics={}; self.validated=0; self.pair_count=0
        self.started=datetime.now(timezone.utc).isoformat()
    def count(self,n): return min(n,8) if self.smoke else n
    def rng(self,purpose,index): return np.random.Generator(np.random.PCG64(seed(purpose,index)))
    def accept(self,e,causal=False):
        if e['canonical_hash'] in self.hashes: return 'duplicate_sequence'
        hashes=[digest(e['token_ids'][:r['answer_token_index']]) for r in e['read_events']]
        if any(h in self.causal_prefixes for h in hashes): return 'causal_prefix_collision'
        validate(e); self.validated+=1
        self.hashes.add(e['canonical_hash']); self.prefixes.update(hashes)
        return None
    def paths(self,name):
        p=self.out/name; p.parent.mkdir(parents=True,exist_ok=True)
        return Path(str(p)+'.jsonl.gz'),Path(str(p)+'.tokens.jsonl')
    def produce(self,name,purpose,index,count=None,predicate=None,pattern=None,length=False,position_quota=None,prediction_budget=None):
        rng=self.rng(purpose,index); derived=seed(purpose,index); stats=Stats(); reject=Counter()
        read_positions=[]; update_positions=[]; attempts=0; misses=0
        meta_path,token_path=self.paths(name)
        with gzip.open(meta_path,'wt',encoding='utf-8',compresslevel=3) as metadata, token_path.open('w') as token_file:
            while (stats.n<count if count is not None else
                   len(read_positions)<position_quota or len(update_positions)<position_quota):
                attempts+=1; misses+=1
                if misses>MAX_ATTEMPTS: raise RuntimeError(f'{name}: MAX_ATTEMPTS exceeded: {reject}')
                initial,blocks,injected=sample(rng,length,pattern)
                if not pattern_ok(blocks,injected,pattern): reject['holdout']+=1; continue
                e=render(initial,blocks,name,derived,purpose!='lm_train',injected,pattern)
                if len(e['token_ids'])>(590 if length else 302): reject['length']+=1; continue
                if predicate is not None:
                    eligible=[r['read_id'] for r in e['read_events'] if predicate(e,r)]
                    if not eligible: reject['no_eligible_target']+=1; continue
                    select_target(e,eligible[int(rng.integers(len(eligible)))])
                if injected is not None:
                    select_target(e,injected)
                    r=e['read_events'][injected]
                    e['holdout_start_dst_flip_survives']=bool(r['local_sensitivity']['ABCD'.index(r['query_var'])])
                reason=self.accept(e)
                if reason: reject[reason]+=1; continue
                misses=0; stats.add(e)
                metadata.write(dump(e)+'\n'); token_file.write(dump(e['token_ids'])+'\n')
                if position_quota:
                    read_positions.extend((e['sequence_id'],r['query_token_index'],r['read_id']) for r in e['read_events'])
                    update_positions.extend((e['sequence_id'],u['end_token_index'],u['update_id']) for u in e['update_events'])
                if prediction_budget is not None and stats.n%64==0 and stats.tokens-stats.n>=prediction_budget:
                    break
        if position_quota:
            selection_seeds={}
            for k,positions in enumerate((read_positions,update_positions),start=1):
                kind='read' if k==1 else 'update'; selection_seeds[kind]=seed(purpose,k)
                chosen=self.rng(purpose,k).choice(len(positions),size=position_quota,replace=False)
                selected=sorted(positions[int(i)] for i in chosen)
                write_json(self.out/f'{name}.{kind}_positions.json',[
                    dict(sequence_id=s,token_index=p,event_id=eid) for s,p,eid in selected])
        info=dict(purpose=purpose,index=index,rng_seed=derived,rng_final_state=rng.bit_generator.state,
            attempts=attempts,accepted=stats.n,rejections=dict(reject),acceptance_rate=stats.n/attempts,
            requested_sequences=count,position_quota=position_quota,prediction_budget=prediction_budget)
        if position_quota: info['position_selection_seeds']=selection_seeds
        self.manifest[name]=info; self.statistics[name]=stats.result()
        print(f'{name}: {stats.n} sequences, {stats.tokens-stats.n:,} prediction tokens',flush=True)
        return stats.tokens-stats.n,stats.n

    def pairs(self,name,purpose,index,count,changed,condition):
        rng=self.rng(purpose,index); derived=seed(purpose,index); reject=Counter(); stats=Stats()
        accepted=attempts=misses=0
        pair_path=self.out/f'{name}.jsonl.gz'; pair_path.parent.mkdir(parents=True,exist_ok=True)
        origins_path=self.out/f'{name}.origins.jsonl.gz'
        with gzip.open(pair_path,'wt',compresslevel=3) as f,gzip.open(origins_path,'wt',compresslevel=3) as origins:
            while accepted<count:
                attempts+=1; misses+=1
                if misses>MAX_ATTEMPTS: raise RuntimeError(f'{name}: infeasible {reject}')
                initial,blocks,_=sample(rng)
                if not pattern_ok(blocks): reject['holdout']+=1; continue
                e=render(initial,blocks,name,derived,True)
                rid=int(rng.integers(len(e['read_events']))); target=e['read_events'][rid]
                end=target['answer_token_index']; q='ABCD'.index(target['query_var'])
                sites=[(3+3*d,d) for d in range(4)] + [
                    (u['end_token_index'],'ABCD'.index(u['dst'])) for u in e['update_events']
                    if u['op']=='SET' and u['end_token_index']<end]
                pos,var=sites[int(rng.integers(len(sites)))]
                logical=any(u['dst']=='ABCD'[var] and u['op']!='SET' and pos<u['end_token_index']<end
                            for u in e['update_events'])
                if ('composition' if logical else 'memory')!=condition: reject['condition']+=1; continue
                if not changed and var==q: reject['unchanged_query_must_differ']+=1; continue
                original=e['token_ids'][:end]; counter=original.copy(); counter[pos]=27-counter[pos]
                try: altered=replay(counter,partial=True)
                except AssertionError: reject['previous_answer_changed']+=1; continue
                if (altered['answer']!=target['answer'])!=changed: reject['target_effect']+=1; continue
                hashes=[digest(original),digest(counter)]
                if any(h in self.prefixes for h in hashes): reject['prefix_collision']+=1; continue
                # Register both complete origin forms, even though suffixes are not model inputs.
                full=e['token_ids'].copy(); full[pos]=27-full[pos]
                replayed_full=replay(full,check_answers=False)['serialized']
                counter_full_hash=digest(replayed_full)
                if counter_full_hash in self.hashes: reject['counter_origin_collision']+=1; continue
                select_target(e,rid)
                reason=self.accept(e,True)
                if reason: reject[reason]+=1; continue
                replay(replayed_full)
                self.hashes.add(counter_full_hash)
                self.prefixes.update(hashes); self.causal_prefixes.update(hashes)
                diffs=[i for i,(a,b) in enumerate(zip(original,counter)) if a!=b]
                assert diffs==[pos] and e['token_roles'][pos] in ('init_literal','update_literal')
                original_replay=replay(original,partial=True)
                assert original_replay['answer']==target['answer']
                assert original_replay['reads']==altered['reads'] or all(
                    a['answer']==b['answer'] for a,b in zip(original_replay['reads'],altered['reads']))
                pair=dict(schema_version='language-v1.0',split=name,rng_seed=derived,
                    pair_id=f"{e['canonical_hash']}:{pos}:{rid}",origin_hash=e['canonical_hash'],
                    counterfactual_origin_hash=counter_full_hash,origin_sequence_id=e['sequence_id'],
                    changed_target=changed,condition=condition,modified_token_index=pos,modified_var='ABCD'[var],
                    target_read_id=rid,target_query_var=target['query_var'],target_query_token_index=end-1,
                    target_answer_token_index=end,original_prefix_ids=original,counterfactual_prefix_ids=counter,
                    original_answer=target['answer'],counterfactual_answer=altered['answer'],
                    original_state=target['state_at_read'],counterfactual_state=altered['state'],
                    prefix_diff_indices=diffs,prefix_hashes=hashes)
                f.write(dump(pair)+'\n'); origins.write(dump(e)+'\n'); stats.add(e)
                misses=0; accepted+=1; self.pair_count+=1
        self.manifest[name]=dict(purpose=purpose,index=index,rng_seed=derived,rng_final_state=rng.bit_generator.state,
            attempts=attempts,accepted=accepted,rejections=dict(reject),acceptance_rate=accepted/attempts,
            changed_target=changed,condition=condition,requested_pairs=count)
        self.statistics[name]=stats.result()
        print(f'{name}: {accepted} pairs ({attempts} attempts)',flush=True)

    def run(self):
        self.out.mkdir(parents=True,exist_ok=False)
        validation=self_test()
        self.produce('val_iid','val_iid',0,self.count(512))
        self.produce('test_iid','test_iid',0,self.count(2048))
        conditions=[('other_variable',lambda e,r:not r['same_as_last_dst']),
            ('repeated_update',lambda e,r:r['query_update_count']>=2),
            ('first_read_after_set',lambda e,r:r['reads_of_query_var_since_latest_set']==0)]
        for split,n in [('val',512),('test',1024)]:
            for index,(name,predicate) in enumerate(conditions):
                self.produce(f'diagnostics/{split}_{name}',f'{split}_diagnostic',index,self.count(n),predicate)
        for index,(lo,hi) in enumerate([(0,0),(1,2),(3,10000)],start=3):
            self.produce(f'diagnostics/test_distance_{lo}_{hi}','test_diagnostic',index,self.count(1024),
                         lambda e,r,lo=lo,hi=hi:lo<=r['updates_since_last_update']<=hi)
        for index,(op,a,b) in enumerate(itertools.product(OPS[2:],range(2),range(2)),start=6):
            def truth(e,r,op=op,a=a,b=b):
                u=e['update_events'][e['blocks'][r['block_id']]['update_ids'][-1]]
                return r['same_as_last_dst'] and (u['op'],u['dst_before'],u['src_before_or_null'])==(op,a,b)
            self.produce(f'diagnostics/test_truth_{op}_{a}{b}','test_diagnostic',index,self.count(256),truth)
        for answer in (0,1):
            self.produce(f'diagnostics/test_balanced_B{answer}','test_diagnostic',18+answer,self.count(512),
                         lambda e,r,answer=answer:r['answer']==answer)
        for pattern in range(2):
            self.produce(f'composition/test_pattern_{pattern}','test_composition',pattern,self.count(1024),pattern=pattern)
        self.produce('test_length','test_length',0,self.count(1024),length=True)
        for split,quota in [('train',50000),('val',10000),('test',20000)]:
            self.produce(f'interpretation/{split}',f'interp_{split}',0,position_quota=80 if self.smoke else quota)
        for split,n in [('val',256),('test',512)]:
            for index,(changed,condition) in enumerate(itertools.product((True,False),('memory','composition'))):
                self.pairs(f"causal_pairs/{split}_{'changed' if changed else 'unchanged'}_{condition}",
                           f'causal_{split}',index,self.count(n),changed,condition)
        reserved_count=len(self.hashes)
        (self.out/'reserved_hashes.txt').write_text(''.join(h+'\n' for h in sorted(self.hashes)))
        (self.out/'causal_prefix_hashes.txt').write_text(''.join(h+'\n' for h in sorted(self.causal_prefixes)))
        total=seqs=shard=0; target=4000 if self.smoke else 3_000_000
        while total<target:
            n=64 if self.smoke else 4096
            tokens,number=self.produce(f'train_shards/{shard:05d}','lm_train',shard,n,prediction_budget=target-total)
            total+=tokens; seqs+=number; shard+=1
        # Logical train shards are exactly 4,096 sequences except the final shard.
        train=[v for k,v in self.statistics.items() if k.startswith('train_shards/')]
        coverage={key:set().union(*(set(v['histograms'][key]) for v in train))
                  for key in ('states','commands','truth_table','adjacent_operators')}
        if not self.smoke:
            assert len(coverage['states'])==16 and len(coverage['commands'])==48 and len(coverage['truth_table'])==12
            assert len(coverage['adjacent_operators'])==25
        validation.update(status='passed',sequences_independently_replayed=self.validated,
            pairs_independently_replayed=self.pair_count,global_exact_duplicate_check='passed',
            holdout_policy='passed',causal_prefix_isolation='passed',
            train_coverage={k:len(v) for k,v in coverage.items()},
            train_prediction_tokens=total,train_sequences=seqs,batch_multiple_64=seqs%64==0,
            metadata_role_index_snapshot_checks='passed',quota_checks='passed')
        write_json(self.out/'vocab.json',dict(enumerate(VOCAB)))
        write_json(self.out/'corpus_statistics.json',self.statistics)
        write_json(self.out/'cpu_validation.json',validation)
        (self.out/'all_sequence_hashes.txt').write_text(''.join(h+'\n' for h in sorted(self.hashes)))
        code={str(p):file_hash(p) for p in sorted(Path('corpus').glob('*.py'))}
        manifest=dict(schema_version='language-v1.0',generator_version='1.0.0',profile='smoke' if self.smoke else 'full',
            specification=['02_language_and_corpus.md','03_experiment_spec.md section 5 (counts/storage)'],
            master_seed=MASTER,seed_derivation='SHA256(20260909|language-v1.0|<purpose>|<index>)[:8], unsigned big endian',
            rng='numpy.random.Generator(PCG64)',numpy_version=np.__version__,python_version=platform.python_version(),
            created_at=self.started,completed_at=datetime.now(timezone.utc).isoformat(),code_sha256=code,
            vocab=dict(enumerate(VOCAB)),holdout_patterns=PATTERNS,max_attempts_per_target=MAX_ATTEMPTS,
            distribution=dict(initial_bit_p1=.5,blocks=[8,24],length_test_blocks=[33,48],
                updates_per_block={'1':.8,'2':.15,'3':.05},operators=dict(zip(OPS,[.25,.25,1/6,1/6,1/6])),
                dst='uniform',src='uniform excluding dst',set_bit_p1=.5,read_same_dst_probability=.5),
            metadata_conventions=dict(state_order=list('ABCD'),indices='zero-based',
                token_distance='query_token_index - last relevant update end (or own initialization end)',
                sensitivity='target block start, all non-LM READs; null on LM train',
                previous_value='value before latest ordinary update of query variable'),
            storage='gzip JSONL complete metadata; separate JSONL token arrays; row order matches',
            resume_policy='offline immutable build; existing output directory rejected; RNG states and accepted counters recorded',
            reserved_sequence_hash_count=reserved_count,all_sequence_hash_count=len(self.hashes),
            causal_prefix_hash_count=len(self.causal_prefixes),splits=self.manifest,
            files={str(p.relative_to(self.out)):{'sha256':file_hash(p),'bytes':p.stat().st_size}
                   for p in sorted(self.out.rglob('*')) if p.is_file()})
        write_json(self.out/'manifest.json',manifest)
        print(f'COMPLETE: {seqs:,} train sequences, {total:,} prediction tokens; all CPU checks passed.',flush=True)



def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=Path('data/language_v1'))
    parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    Builder(args.output,args.smoke).run()


if __name__=='__main__': main()
