"""Consolidate P4 local checks; never mark production training or P4 complete."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from interp_v1_4.model import Transformer
from interp_v1_4.runtime import restore, sha, tensor_digest
from interp_v1_4.training import lm_optimizer


def verify():
    folder = ROOT / 'experiment_v1_4'
    prep = folder / 'results/p4_preparation_r1'
    suite = ET.parse(prep / 'tests.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['tests']) == 33
    assert all(int(suite.attrib[name]) == 0 for name in ('failures', 'errors', 'skipped'))
    smoke_path = folder / 'smoke/cpu_p4_r1/smoke.json'
    smoke = json.loads(smoke_path.read_text())
    assert smoke['status'] == 'passed' and smoke['scope'] == 'cpu'
    reports, initial = {}, {}
    for seed in (1, 2):
        run = folder / f'smoke/resume_p4_seed{seed}_r1'
        report = json.loads((run / 'resume_equivalence.json').read_text())
        assert report['status'] == 'passed' and report['lm_seed'] == seed and report['stage'] == 'p4'
        reports[str(seed)] = report
        initial[seed] = torch.load(run / 'full/checkpoints/init.pt', map_location='cpu', weights_only=False)
    digests = {seed: tensor_digest(payload['model']) for seed, payload in initial.items()}
    assert digests[1] != digests[2]
    model = Transformer()
    optimizer = lm_optimizer(model)
    source = folder / 'smoke/resume_p4_seed1_r1/full/checkpoints/init.pt'
    try:
        restore(source, model, optimizer, initial[2]['hashes'])
    except ValueError as error:
        assert 'hashes differ' in str(error)
    else:
        raise AssertionError('Cross-seed resume was accepted')
    delivery_path = prep / 'delivery.json'
    delivery = json.loads(delivery_path.read_text())
    assert delivery['status'] == 'passed_delivery_checks'
    assert sha(ROOT / delivery['bundle']) == delivery['bundle_sha256']
    for name, digest in delivery['notebooks'].items():
        assert sha(ROOT / name) == digest
        notebook = json.loads((ROOT / name).read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(cell['source'], name, 'exec')
    result = dict(
        status='passed_local_preparation', phase='v1.4-P4', p4_complete=False,
        tests_passed=33, cpu_smoke='passed', cpu_environment_id=smoke['environment']['environment_id'],
        persistent_resume=reports, independent_initialization_tensor_sha256=digests,
        cross_seed_resume='rejected', delivery=delivery,
        frozen_numerical_dependencies='byte_identical_to_seed0_except_cli_orchestration',
        production_seed1='not_run', production_seed2='not_run', test_scores_observed=False,
        prerequisites=dict(seed0_completion_sha256=sha(folder / 'results/p3_audit_20260921_01/completion.json')),
        evidence_sha256={name: sha(prep / name) for name in ('tests.xml', 'tests_anaconda.log', 'smoke.log',
                                                           'resume_seed1.log', 'resume_seed2.log', 'delivery.json')},
    )
    with (prep / 'verification.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'tests_passed', 'cpu_smoke', 'cross_seed_resume', 'p4_complete')}, indent=2))


if __name__ == '__main__':
    verify()
