"""Independent post-write audit, including actual stored bytes and split isolation."""
import argparse
import gzip
import json
from pathlib import Path
from .generate import file_hash, write_json
from .language import digest, PATTERNS, VOCAB
from .replay import replay, validate


def audit(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,info in manifest['files'].items():
        p=root/name
        assert p.stat().st_size==info['bytes'] and file_hash(p)==info['sha256'],name
    hashes=set(); prefixes=set(); causal=set(); count=pair_count=0
    expected=set((root/'all_sequence_hashes.txt').read_text().splitlines())
    for path in sorted(root.rglob('*.jsonl.gz')):
        if 'causal_pairs' in path.parts and not path.name.endswith('.origins.jsonl.gz'): continue
        token_path=Path(str(path).replace('.jsonl.gz','.tokens.jsonl'))
        tf=token_path.open() if token_path.exists() else None
        try:
            with gzip.open(path,'rt') as f:
                for line in f:
                    e=json.loads(line); validate(e); count+=1
                    t=e['token_ids']; h=digest(t)
                    assert h==e['canonical_hash'] and h not in hashes
                    hashes.add(h)
                    if tf: assert json.loads(next(tf))==t
                    assert (33<=len(e['blocks'])<=48 and len(t)<=590) if e['split']=='test_length' else (8<=len(e['blocks'])<=24 and len(t)<=302)
                    for b in e['blocks']:
                        us=[e['update_events'][i] for i in b['update_ids']]
                        ops=tuple(u['op'] for u in us)
                        if b['block_id']==e['holdout_block_id']:
                            assert ops==PATTERNS[e['holdout_pattern_id']]
                            assert len({u['dst'] for u in us})==1
                            assert e['read_events'][b['read_id']]['query_var']==us[-1]['dst']
                            assert e['target_read_ids']==[b['read_id']]
                        else: assert ops not in PATTERNS
                    nodes=e['dependency_nodes']; current=list(range(4))
                    for u in e['update_events']:
                        d='ABCD'.index(u['dst']); op=u['op']; node=nodes[u['dependency_node_id']]
                        parents=[] if op=='SET' else [current[d]] if op=='NOT' else [current[d],current['ABCD'.index(u['src_or_null'])]]
                        assert node['parent_ids']==parents
                        depth=0 if not parents else 1+max(nodes[i]['depth'] for i in parents)
                        assert depth==node['depth']==u['structural_depth']
                        current[d]=node['node_id']
                    for r in e['read_events']:
                        prefixes.add(digest(t[:r['answer_token_index']]))
                        assert nodes[r['dependency_node_id']]['depth']==r['structural_depth']
                        if r['local_sensitivity'] is not None:
                            b=e['blocks'][r['block_id']]; sens=[]
                            for v in range(4):
                                state=b['state_at_start'].copy(); state[v]^=1
                                probe=[1]
                                for d,bit in enumerate(state): probe.extend([3,9+d,13+bit])
                                probe.extend(t[b['start_token_index']:r['answer_token_index']])
                                sens.append(int(replay(probe,partial=True)['answer']!=r['answer']))
                            assert sens==r['local_sensitivity']
                    if e['holdout_pattern_id'] is not None:
                        r=e['read_events'][e['target_read_ids'][0]]
                        assert e['holdout_start_dst_flip_survives']==bool(r['local_sensitivity']['ABCD'.index(r['query_var'])])
            if tf: assert tf.read()==''
        finally:
            if tf: tf.close()
        print(f'audited {path.relative_to(root)}',flush=True)
    for path in sorted((root/'causal_pairs').glob('*.jsonl.gz')):
        if '.origins.' in path.name: continue
        with gzip.open(path,'rt') as f:
            for line in f:
                p=json.loads(line); a=p['original_prefix_ids']; b=p['counterfactual_prefix_ids']
                assert len(a)==len(b)==p['target_answer_token_index']
                diffs=[i for i in range(len(a)) if a[i]!=b[i]]
                assert diffs==[p['modified_token_index']]==p['prefix_diff_indices']
                i=diffs[0]; assert a[i] in (13,14) and b[i]==27-a[i] and a[i-2]==3
                ra=replay(a,partial=True); rb=replay(b,partial=True)
                assert ra['answer']==p['original_answer'] and rb['answer']==p['counterfactual_answer']
                assert (ra['answer']!=rb['answer'])==p['changed_target']
                assert ra['state']==p['original_state'] and rb['state']==p['counterfactual_state']
                assert [r['answer'] for r in ra['reads']]==[r['answer'] for r in rb['reads']]
                ah,bh=digest(a),digest(b)
                assert [ah,bh]==p['prefix_hashes'] and ah not in causal and bh not in causal
                assert bh not in prefixes and ah in prefixes
                causal.update((ah,bh)); hashes.add(p['counterfactual_origin_hash']); pair_count+=1
    assert hashes==expected
    assert causal==set((root/'causal_prefix_hashes.txt').read_text().splitlines())
    # Every selected interpretation position must be unique and refer to its own split.
    for split,quota in [('train',50000),('val',10000),('test',20000)]:
        available={'read':set(),'update':set()}
        with gzip.open(root/f'interpretation/{split}.jsonl.gz','rt') as f:
            for line in f:
                e=json.loads(line)
                for kind,key in [('read','query_token_index'),('update','end_token_index')]:
                    available[kind].update((e['sequence_id'],r[key],r[f'{kind}_id']) for r in e[f'{kind}_events'])
        for kind in ('read','update'):
            records=json.loads((root/f'interpretation/{split}.{kind}_positions.json').read_text())
            keys=[(r['sequence_id'],r['token_index'],r['event_id']) for r in records]
            assert len(keys)==(80 if manifest['profile']=='smoke' else quota)
            assert keys==sorted(keys) and len(set(keys))==len(keys) and set(keys)<=available[kind]
    result=dict(status='passed',stored_sequences=count,causal_pairs=pair_count,
        file_checksums='passed',token_metadata_alignment='passed',independent_replay='passed',
        dependency_graph_and_local_sensitivity='passed',global_sequence_hashes='passed',
        interpretation_positions='passed',holdout_policy='passed',causal_prefixes='passed')
    write_json(root/'postwrite_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    audit(parser.parse_args().root)
