"""Supplement the immutable-run audit using the exact executed input bundle.

Does not modify execution inputs or inspect test scores. The original verifier
must already have produced local_run_audit.json in the evidence directory.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--executed-root', type=Path, required=True)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, release, evidence = args.executed_root, args.release_root, args.evidence
    sys.path.insert(0, str(root))
    import torch
    from interp_v1_2.runtime import deterministic
    from interp_v1_2.model import Transformer
    from interp_v1_2.data import metadata

    run = evidence / 'lm_seed_0'
    result = read(run / 'result.json')
    audit = read(evidence / 'local_run_audit.json')
    assert audit['status'] == 'passed' and audit['reevaluation'] is not None
    bundle = read(evidence / 'p3_bundle_manifest.json')
    assert bundle == read(root / 'p3_bundle_manifest.json')
    for name, digest in bundle.items():
        assert sha(root / name) == digest, name
    old = read(root / 'data/language_v1_2/manifest.json')
    new = read(release / 'data/language_v1_2/manifest.json')
    scientific_files = [n for n in old['files'] if not n.endswith('.DS_Store') and n != 'cpu_validation.json']
    old_validation = read(root / 'data/language_v1_2/cpu_validation.json')
    old_validation['inherited_files'].pop('.DS_Store')
    assert old_validation == read(release / 'data/language_v1_2/cpu_validation.json')
    for name in scientific_files:
        assert old['files'][name] == new['files'][name], name
        assert sha(release / 'data/language_v1_2' / name) == old['files'][name]['sha256'], name
    old_config = read(root / 'experiment_v1_2/configs/run.json')
    new_config = read(release / 'experiment_v1_2/configs/run.json')
    changed_config_keys = [k for k in old_config if old_config[k] != new_config[k]]
    assert changed_config_keys == ['data_hashes'], changed_config_keys
    for name, digest in result['hashes']['code'].items():
        assert sha(root / name) == digest == sha(release / name), name

    smoke_reports = []
    for path in sorted((evidence / 'preflight').glob('*/[c]*/smoke.json')):
        smoke = read(path)
        assert smoke['status'] == 'passed' and smoke['debug_only'] and not smoke['reuse_in_experiment']
        for key in ('configs', 'corpus_manifest'):
            assert smoke['input_hashes'][key] == result['hashes'][key]
        assert smoke['input_hashes']['code'] == {k: v for k, v in result['hashes']['code'].items() if k.startswith('interp_v1_2/')}
        env = smoke['environment']
        assert sha(path.parent / 'requirements.lock.txt') == env['lock_sha256']
        identity = {k: env[k] for k in ('python', 'platform', 'torch', 'cuda', 'cudnn', 'gpu', 'lock_sha256')}
        assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16] == env['environment_id']
        assert env['environment_id'] == result['environment']['environment_id']
        smoke_reports.append(dict(path=str(path.relative_to(evidence)), scope=smoke['scope'], checks=smoke['checks'], elapsed_seconds=smoke['elapsed_seconds']))
    assert {s['scope'] for s in smoke_reports} == {'cpu', 'cuda'}
    sessions = list((run / 'sessions').glob('*/environment.json'))
    assert len(sessions) == 1
    session = read(sessions[0])
    assert session['resume'] == 'None' and session['environment'] == result['environment']
    assert sha(sessions[0].parent / 'requirements.lock.txt') == result['environment']['lock_sha256']
    assert '8 passed' in next((evidence / 'preflight').glob('*/pytest.txt')).read_text()

    deterministic(0)
    torch.set_num_threads(2)
    init = torch.load(run / 'checkpoints/init.pt', map_location='cpu', weights_only=False)
    fresh = Transformer().state_dict()
    assert fresh.keys() == init['model'].keys()
    fresh_check = dict(local_torch=torch.__version__, executed_torch=result['environment']['torch'], bitwise_equal=all(torch.equal(fresh[k], init['model'][k]) for k in fresh), max_abs_difference=max(float((fresh[k]-init['model'][k]).abs().max()) for k in fresh), note='Cross-platform fresh initialization is descriptive; exact comparison uses preserved prior Colab initialization.')
    events = [read(p) for p in sorted((run / 'events').glob('*.json'))]
    curves = []
    best = float('inf')
    for event in events:
        assert all(math.isfinite(event[k]) for k in ('loss', 'gradient_norm', 'lr', 'training_seconds'))
        assert event['lr'] == 3e-4 * min(1, event['state']['prediction_tokens'] / 50000)
        assert event['microbatch'] == 16
        if 'validation' in event:
            metrics = event['validation']['general']
            best = min(best, metrics['answer_ce'])
            curves.append(dict(update=event['state']['update'], tokens=event['state']['prediction_tokens'], current_ce=metrics['answer_ce'], best_ce=best, general_accuracy=metrics['correct'], **{n: m['correct'] for n, m in event['validation']['diagnostics'].items()}))
    first50 = result['measurement_first_50_updates']
    assert first50['updates'] == 50
    assert first50['prediction_tokens'] == sum(e['tokens'] for e in events[:50])
    assert abs(first50['training_seconds'] - sum(e['training_seconds'] for e in events[:50])) < 1e-9

    # Compare preserved, already published v1.1 state; no new test evaluation.
    previous = release / 'experiment_v1_1/evidence/p3_colab_20260916T085123549826/lm_seed_0'
    comparisons = {}
    for label, old_name, new_name in [('initialization', 'init.pt', 'checkpoints/init.pt'), ('3M', 'best.pt', 'checkpoints/update_000407.pt')]:
        before = torch.load(previous / old_name, map_location='cpu', weights_only=False)
        after = torch.load(run / new_name, map_location='cpu', weights_only=False)
        identical = before['model'].keys() == after['model'].keys() and all(torch.equal(before['model'][k], after['model'][k]) for k in before['model'])
        comparisons[label] = dict(model_tensors_bitwise_equal=identical, previous_sha256=sha(previous / old_name), current_sha256=sha(run / new_name))
    assert comparisons['initialization']['model_tensors_bitwise_equal']
    for i in range(7):
        assert sha(root / f'data/language_v1/train_shards/{i:05d}.tokens.jsonl') == sha(root / f'data/language_v1_2/train_shards/{i:05d}.tokens.jsonl')

    # Baselines on exactly the same validation READ targets, using token prefixes.
    baselines = {}
    for name, filename, diagnostic in [('general', 'val_iid.jsonl.gz', False)] + [(n, f'diagnostics/val_{n}.jsonl.gz', True) for n in result['selected']['validation']['diagnostics']]:
        answers, set_predictions, copy_predictions = [], [], []
        for example in metadata(root / 'data/language_v1_2' / filename):
            latest_set = {}; latest_read = 0
            events_by_query = {r['query_token_index']: r for r in example['read_events']}
            tokens = example['token_ids']; i = 0
            while i < len(tokens):
                if tokens[i] == 3:  # SET var bit, including initialization
                    latest_set[tokens[i + 1]] = tokens[i + 2] - 13
                    i += 3
                    continue
                if i in events_by_query:
                    row = events_by_query[i]
                    if not diagnostic or row['read_id'] in example['target_read_ids']:
                        answers.append(row['answer'])
                        set_predictions.append(latest_set[tokens[i]])
                        copy_predictions.append(latest_read)
                    latest_read = row['answer']
                i += 1
        n = len(answers)
        assert n == (result['selected']['validation']['diagnostics'][name] if diagnostic else result['selected']['validation']['general'])['answer_count']
        baselines[name] = dict(answer_count=n, majority=max(sum(answers), n-sum(answers))/n, independent_coin_expected=.5, latest_set=sum(a == b for a, b in zip(answers, set_predictions))/n, previous_read=sum(a == b for a, b in zip(answers, copy_predictions))/n)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'p3_learning_curve.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(curves)
    report = dict(status='passed',meaning='Evidence audit passed; behavioral gate failed',decision=result['gate']['decision'], bundle_file_count=len(bundle), scientific_files_unchanged=len(scientific_files), executed_config_sha256=result['hashes']['configs'], executed_corpus_manifest_sha256=result['hashes']['corpus_manifest'], release_config_changed_keys=changed_config_keys, executed_code_identical_to_release=True, fresh_cpu_initialization_comparison=fresh_check, smoke=smoke_reports, environment=result['environment'], microbatch=16, first50_tokens_per_second=first50['prediction_tokens']/first50['training_seconds'], training_seconds=sum(e['training_seconds'] for e in events), validation_seconds=sum(e.get('validation_seconds',0) for e in events), session_seconds=result['session_seconds'], checkpoint_count=len(list((run/'checkpoints').glob('*.pt'))), v1_1_comparison=comparisons, validation_baselines=baselines, test_scores_accessed=False)
    (args.output / 'p3_supplemental_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','decision','checkpoint_count','v1_1_comparison','validation_baselines')},indent=2))


if __name__ == '__main__':
    main()
