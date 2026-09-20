"""Read-only audit of closed v1.4 files while the generator is still running."""
import gzip, hashlib, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from corpus.replay import validate
from corpus.language import digest
from corpus.v1_4 import seed
OUT = Path(__file__).resolve().parent
DATA = ROOT / json.loads((ROOT/'experiment_v1_4/corpus_rebuild.json').read_text())['active_data_root']
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
report = dict(status='partial_snapshot_only', started_at=datetime.now(timezone.utc).isoformat(), checks={}, model_scores_observed=False)
def record():
    (OUT/'snapshot.json').write_text(json.dumps(report, indent=2)+'\n')
manifest=json.loads((ROOT/'experiment_v1_4/design_manifest.json').read_text())
report['design_hashes']={}
for group in ('files','source_documents','proposal_input','v1_3_inputs'):
    for name, expected in manifest[group].items():
        actual=sha(ROOT/name)
        report['design_hashes'][name]=dict(expected=expected, actual=actual, matches=actual==expected)
record()
prefix={}
for path in sorted((ROOT/'data/language_v1_3/train_shards').glob('*')):
    if path.is_file() and path.name!='.DS_Store':
        actual=sha(DATA/'train_shards'/path.name)
        assert actual==sha(path), path
        prefix[path.name]=actual
report['checks']['copied_train_prefix']=dict(status='passed', files=len(prefix), hashes=prefix)
record()
seen_hashes=set(); seen_prefixes=set(); checked={}
def reserve(examples):
    hashes=set(); prefixes=set()
    for e in examples:
        validate(e)
        assert e['canonical_hash']==digest(e['token_ids'])
        assert e['canonical_hash'] not in seen_hashes and e['canonical_hash'] not in hashes
        hashes.add(e['canonical_hash'])
        prefixes.update(digest(e['token_ids'][:r['answer_token_index']]) for r in e['read_events'])
    assert not prefixes & seen_prefixes
    seen_hashes.update(hashes); seen_prefixes.update(prefixes)
for name, count, purpose, idx in [('select/general',512,'select_general',0),('gate/general',1024,'gate_general',0)]+[(f'gate/legacy_diagnostics/{n}',1024,'gate_legacy_diagnostic',i) for i,n in enumerate(('other_variable','repeated_update','first_read_after_set'))]:
    p=DATA/(name+'.jsonl.gz'); before=sha(p); n=0
    with gzip.open(p,'rt') as f, (DATA/(name+'.tokens.jsonl')).open() as tokens:
        for line, tokenline in zip(f,tokens,strict=True):
            e=json.loads(line); assert e['token_ids']==json.loads(tokenline)
            assert e['rng_seed']==seed(purpose,idx)
            reserve([e]); n+=1
            if 'legacy_diagnostics' in name:
                assert len(e['target_read_ids'])==1
                r=e['read_events'][e['target_read_ids'][0]]
                assert (not r['same_as_last_dst'] if idx==0 else r['query_update_count']>=2 if idx==1 else r['reads_of_query_var_since_latest_set']==0)
    assert n==count and before==sha(p)
    checked[name]=dict(sequences=n, sha256=before)
p=DATA/'first_repeat/select.jsonl.gz'; before=sha(p); counts=Counter()
with gzip.open(p,'rt') as f:
    for line in f:
        pair=json.loads(line); a,b=pair['origin'],pair['repeat']; reserve([a,b])
        r=a['read_events'][pair['origin_target_read_id']]; s=b['read_events'][pair['repeat_target_read_id']]
        u=a['update_events'][r['last_update_id_for_query_var_or_null']]
        pattern=str(u['dst_before']) if u['op']=='NOT' else u['input_truth_pattern_or_null']
        d=r['structural_depth']; depth='1' if d==1 else '2-3' if d in (2,3) else '4+'
        assert pair['cell_id']==f"{u['op']}:{pattern}:depth_{depth}"
        assert (pair['operator'],pair['input_pattern'],pair['depth_bin'])==(u['op'],pattern,depth)
        assert r['reads_of_query_var_since_last_update']==0 and s['reads_of_query_var_since_last_update']==1
        for key in ('query_var','state_at_read','answer','structural_depth'): assert r[key]==s[key]
        assert pair['answer']==r['answer']
        assert a['rng_seed']==b['rng_seed']==seed('select_first_repeat',pair['cell_index'])
        inserted=b['read_events'][pair['inserted_read_id']]['query_token_index']-1
        assert b['token_ids'][:inserted]+b['token_ids'][inserted+3:]==a['token_ids']
        counts[pair['cell_id']]+=1
assert len(counts)==42 and set(counts.values())=={64} and before==sha(p)
checked['first_repeat/select']=dict(pairs=sum(counts.values()),cells=dict(counts),sha256=before)
report['checks']['closed_splits']=checked
record()
history={}
for version in ('language_v1','language_v1_2','language_v1_3'):
    for filename, values in [('all_sequence_hashes.txt',seen_hashes),('reserved_prefix_hashes.txt',seen_prefixes)]:
        p=ROOT/'data'/version/filename
        if not p.exists():
            history[str(p.relative_to(ROOT))]=dict(status='absent'); continue
        overlap=set(); n=0
        with p.open() as f:
            for line in f:
                value=line.strip(); n+=1
                if value in values: overlap.add(value)
        assert not overlap, (p,overlap)
        history[str(p.relative_to(ROOT))]=dict(entries=n,overlap=0,sha256=sha(p))
report['checks']['historical_registry_intersections']=history
report['limitations']=['Not a completed corpus audit: gate pairs and later splits are still being generated.', 'Historical prefix check covers stored registries, not an independent replay of every historical row.', '64M extension, complete quotas, frozen inputs and persistent CLI resume remain unverified.']
report['completed_at']=datetime.now(timezone.utc).isoformat()
record()
print(json.dumps(dict(status=report['status'], closed_splits=list(checked), unique_sequences=len(seen_hashes), unique_prefixes=len(seen_prefixes))))
