"""Audit returned P7 raw evidence; never declare phase completion automatically."""
from pathlib import Path
import argparse
import sys
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp_v1_4.p5 import read, write, derived, domains, LABELS
from interp_v1_4.p7 import verify, CONTRACT, records
from interp_v1_4.runtime import sha
from interp_v1_4.p7_metrics import random_sets, matched_indices
from interp_v1_4.p7_patching import valid_arrays
from interp_v1_4 import p5_probe


def close(actual, expected):
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def audit(root, output):
    root, output = Path(root), Path(output)
    config = verify(root); ch = sha(root/CONTRACT)
    assert read(output/'contract.json') == config
    finished = read(output/'evaluation_complete.json')
    assert finished['config_sha256'] == ch and finished['runs'] == 24
    for path, digest in finished['artifacts'].items():
        assert sha(output/path) == digest
    labels = read(output/'labels/test.json')
    targets = read(root/config['reconstruction_targets'])
    suites = {name: records(root/config['data_root']/'causal_pairs'/f'{name}.jsonl.gz')
              for name in config['causal_quotas'] if name.startswith('test_')}
    checked = []
    for entry in config['saes']:
        run = entry['run']; folder = output/'runs'/run['name']
        selection = read(folder/'selection.json'); matching = read(folder/'matching.json')
        assert selection['config_sha256'] == ch and not selection['test_used_for_selection']
        assert selection['checkpoint_sha256'] == entry['sha256']
        assert matching['identity'] == dict(config_sha256=ch, selection_sha256=sha(folder/'selection.json'))
        assert not matching['test_used_for_bins']
        for fit in selection['fits'].values():
            assert fit['status'] in ('passed', 'NA'), 'Failed optimization is not completion'
        assert valid_arrays(folder/'semantic_predictions.npz', ch)
        semantic = read(folder/'semantic.json')
        assert semantic['selection_sha256'] == sha(folder/'selection.json')
        with np.load(folder/'semantic_predictions.npz') as predictions:
            for task, fit in selection['fits'].items():
                if fit['status'] != 'passed': continue
                _, label, domain = task.rsplit('_', 2)
                mask = domains(labels, label, domain, 'test')
                y = np.array([r[label] if r[label] is not None else -1 for r in labels])[mask]
                for size, model in fit['selected'].items():
                    p = predictions[task+'_'+size]
                    assert np.isfinite(p).all() and len(p) == len(y)
                    result = p5_probe.metrics(y, p, LABELS[label], model['threshold'])
                    saved = semantic['semantic'][task]['evaluation'][size]
                    for key in ('balanced_accuracy', 'macro_f1', 'binary_auroc', 'confusion_matrix'):
                        close(saved[key], result[key])
        replacement_count = 0
        for start in range(0, len(targets), 16):
            path = folder/'replacement'/f'{start:05d}.npz'; assert valid_arrays(path, ch)
            with np.load(path) as z:
                indices = np.arange(start, min(start+16, len(targets)))
                np.testing.assert_array_equal(z['indices'], indices)
                np.testing.assert_array_equal(z['answers'], [targets[i]['answer'] for i in indices])
                for kind in ('original', 'patched'):
                    logits = z[kind+'_logits']; assert np.isfinite(logits).all()
                    ce = logsumexp(logits.astype(float), axis=1)-logits[np.arange(len(indices)), z['answers']+13]
                    close(z[kind+'_ce'], ce)
                replacement_count += len(indices)
        pair_count = 0
        for name, pairs in suites.items():
            paths = sorted((folder/'causal'/name).glob('*.json'))
            assert len(paths) == len(pairs)
            for i, p in enumerate(pairs):
                saved = read(folder/'causal'/name/f'{i:05d}.json')
                assert saved['pair_id'] == p['pair_id']
                assert saved['identity'] == dict(config_sha256=ch, selection_sha256=sha(folder/'selection.json'), matching_sha256=sha(folder/'matching.json'))
                for size, candidate in saved['candidates'].items():
                    fit = selection['fits']['sae_current_iid']
                    features = fit['selected'][size]['columns'] if fit['status']=='passed' else None
                    assert candidate['features'] == features
                    seed = derived('random_patch', run['run_key']+'|'+p['pair_id']+'|'+size)
                    assert candidate['seed'] == seed
                    if features is not None:
                        assert candidate['candidates'] == random_sets(512, features, seed, config['random_candidates'])
                        expected = matched_indices(candidate['selected_h_norm'], candidate['selected_z_norm'], candidate['h_norms'], candidate['z_norms'], matching['rules'][size]['rule'], config['matched_limit'])
                        assert candidate['matched_indices'] == expected
                    for direction in (0, 1):
                        rows = [r for r in saved['rows'] if r['size']==size and r['direction']==direction]
                        controls = {r['control'] for r in rows}
                        assert {'identity','mean','approximation','full_donor','random_direction_pure'} <= controls
                        if features is not None:
                            assert {'selected','random_unmatched'} <= controls
                            assert ('random_matched' in controls) == bool(candidate['matched_indices'])
                            assert ('selected_matched' in controls) == bool(candidate['matched_indices'])
                        for rep, control in [('coordinate','coordinate'),('random','random_direction_selected')]:
                            f = selection['fits'][rep+'_current_iid']
                            if f['status']=='passed':
                                assert control in controls
                                assert next(r for r in rows if r['control']==control)['features']==f['selected'][size]['columns']
                        answer = p['original_answer'] if direction==0 else p['counterfactual_answer']
                        donor = p['counterfactual_answer'] if direction==0 else p['original_answer']
                        a = 13+answer; b = 13+(donor if p['changed_target'] else 1-answer)
                        for r in rows:
                            o, v = np.array(r['original_logits']), np.array(r['patched_logits'])
                            assert o.shape == v.shape == (15,) and np.isfinite(v).all() and np.isfinite(o).all()
                            assert r['origin']==p['origin_hash'] and r['condition']==p['condition'] and r['changed']==p['changed_target']
                            close(r['delta_margin'], (v[b]-v[a])-(o[b]-o[a]))
                            close(r['full_flip'], float(v.argmax()==13+donor) if p['changed_target'] else 0.)
                            close(r['binary_flip'], float(v[13:15].argmax()==donor) if p['changed_target'] else 0.)
                            close(r['error_induced'], float(o.argmax()==a and v.argmax()!=a))
                            close(r['prediction_changed'], float(o.argmax()!=v.argmax()))
                            if r['control']=='identity': np.testing.assert_allclose(o,v,atol=1e-5,rtol=1e-4)
                pair_count += 1
        checked.append(dict(run=run['name'],pairs=pair_count,replacement_targets=replacement_count))
        print('raw evidence verified', run['name'], flush=True)
    result = dict(status='raw_evidence_checks_passed_external_review_pending',p7_complete=False,config_sha256=ch,runs=checked,
                  scope='Input hashes, frozen identities, probe point metrics, candidate RNG/bin membership, quotas, raw-logit CE and causal scores. Full fitting/fidelity/CI reproduction requires source caches and separate return review.')
    write(output/'raw_evidence_audit.json',result)
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(ROOT));parser.add_argument('--output',required=True)
    args=parser.parse_args();audit(args.root,args.output)
