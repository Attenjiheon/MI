"""Read-only completed-corpus audit with independently enumerated historical inputs.

Never rewrites the corpus or its generation evidence. Model scores are not read.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from corpus.language import digest
from corpus.replay import validate
from scripts.generate_v1_4_corpus import audit_v1_4, file_hash


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def history(_base):
    sequences, prefixes = set(), set()
    # Deliberately do not call the generator's recursive historical loader.
    for version in ('language_v1', 'language_v1_2', 'language_v1_3'):
        base = ROOT / 'data' / version
        for name, dest in (('all_sequence_hashes.txt', sequences), ('reserved_prefix_hashes.txt', prefixes)):
            p = base / name
            if p.exists():
                with p.open() as f:
                    dest.update(line.strip() for line in f)
        n = 0
        for p in sorted(base.rglob('*.jsonl.gz')):
            if 'provenance' in p.relative_to(base).parts:
                continue
            for row in rows(p):
                if 'token_ids' in row:
                    examples = [row]
                elif 'origin' in row and 'repeat' in row:
                    examples = [row['origin'], row['repeat']]
                else:
                    assert 'original_prefix_ids' in row and 'counterfactual_prefix_ids' in row, p
                    prefixes.update(digest(row[k]) for k in ('original_prefix_ids', 'counterfactual_prefix_ids'))
                    sequences.update((row['origin_hash'], row['counterfactual_origin_hash']))
                    examples = []
                for example in examples:
                    sequences.add(digest(example['token_ids']))
                    prefixes.update(digest(example['token_ids'][:r['answer_token_index']]) for r in example['read_events'])
                n += 1
        print('historical complete', version, n, len(sequences), len(prefixes), flush=True)
    return sequences, prefixes


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    data = ROOT / json.loads((ROOT/'experiment_v1_4/corpus_rebuild.json').read_text())['active_data_root']
    before = {n: file_hash(data/n) for n in ('manifest.json', 'postwrite_audit.json', 'cpu_validation.json', 'corpus_statistics.json')}
    manifest = json.loads((data/'manifest.json').read_text())
    print('checksum, full replay and independent historical isolation audit', flush=True)
    audit_v1_4(data, output=output/'full_replay.json', history_loader=history)
    print('independent quota, diagnostics and cursor checks', flush=True)
    counts = {}
    for name, info in manifest['splits'].items():
        if name.startswith('first_repeat/'):
            continue
        path = data/(name+'.jsonl.gz')
        count = 0
        for e in rows(path):
            count += 1
            if '/legacy_diagnostics/' in name:
                assert len(e['target_read_ids']) == 1
                r=e['read_events'][e['target_read_ids'][0]]
                condition=name.rsplit('/',1)[-1]
                assert {'other_variable': not r['same_as_last_dst'], 'repeated_update': r['query_update_count'] >= 2, 'first_read_after_set': r['reads_of_query_var_since_latest_set'] == 0}[condition]
        assert count == info['accepted'], (name,count,info['accepted'])
        if 'requested_pairs' in info:
            assert count == info['requested_pairs']
        counts[name]=count
    for split,n in (('select',512),('gate',1024),('test',2048)):
        assert counts[f'{split}/general']==n
        if split!='select':
            for condition in ('other_variable','repeated_update','first_read_after_set'):
                assert counts[f'{split}/legacy_diagnostics/{condition}']==n
    assert counts['test/length']==1024
    assert all(counts[f'test/composition/pattern_{i}']==1024 for i in range(2))
    for split,n in (('val',256),('test',512)):
        for changed in ('changed','unchanged'):
            for condition in ('memory','composition'):
                assert counts[f'causal_pairs/{split}_{changed}_{condition}']==n
    lengths=[]
    for p in sorted((data/'train_shards').glob('*.tokens.jsonl')):
        with p.open() as f: lengths.extend(len(json.loads(line))-1 for line in f)
    frozen=manifest['frozen_training']; assert len(lengths)==frozen['final_cursor']
    assert len(lengths)%64==0 and sum(lengths)==frozen['actual_prediction_tokens']
    assert sum(lengths[:-64])<64_000_000<=sum(lengths)
    milestones={}; cumulative=0
    boundaries=json.loads((ROOT/'experiment_v1_4/design_config.json').read_text())['checkpoint_selection']['candidate_milestones_prediction_tokens']
    for cursor in range(64,len(lengths)+1,64):
        cumulative+=sum(lengths[cursor-64:cursor])
        for b in boundaries:
            if cumulative>=b and str(b) not in milestones:
                milestones[str(b)]=dict(update=cursor//64,cursor=cursor,prediction_tokens=cumulative)
    assert milestones==frozen['milestones']
    assert all(file_hash(data/n)==h for n,h in before.items())
    result=dict(status='passed_data_integrity', completed_at=datetime.now(timezone.utc).isoformat(), data_hashes=before, counts=counts, frozen_training=frozen, model_scores_observed=False, attempt_cap_history='unverified: original generation stored per-cell totals, not per-target attempts', script_sha256=file_hash(Path(__file__)))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('data integrity passed; historical per-target attempt compliance remains unverified',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
