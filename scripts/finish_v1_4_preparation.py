"""Finish local preparation after the already running corrected P1 job.

This is a finite local build pipeline, never a GPU training launcher. It may
wait for the corrected audit, then freezes configs, tests, CPU-smokes, checks
resume, and builds the notebook/bundle. It never marks P2/P3 GPU work complete
or commits/pushes without a separate review.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(wait):
    folder = ROOT / 'experiment_v1_4'
    out = folder / 'results/local_preparation'
    out.mkdir(parents=True, exist_ok=False)
    amendment = json.loads((folder / 'corpus_rebuild.json').read_text())
    audit_path = ROOT / amendment['active_data_root'] / 'postwrite_audit.json'
    state = dict(status='waiting_for_corrected_p1', started_at=datetime.now(timezone.utc).isoformat())

    def record():
        temporary = out / 'state.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'state.json')

    record()
    try:
        deadline = time.monotonic() + 12 * 60 * 60
        while not audit_path.exists():
            if not wait or time.monotonic() > deadline:
                raise RuntimeError('Corrected P1 audit not available; no downstream step was run')
            time.sleep(10)
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'passed' and audit['historical_reserved_registry_loaded']
        assert audit['rng_metadata_matches_v14_namespace']
        commands = [
            ('freeze', ['scripts/freeze_v1_4_config.py']),
            ('tests', ['-m', 'pytest', 'tests_v1_4', '-q', '--junitxml=' + str(out / 'tests.xml')]),
            ('cpu_smoke', ['-m', 'interp_v1_4.smoke', '--root', str(ROOT), '--device', 'cpu',
                           '--output', str(folder / 'smoke/cpu_01')]),
            ('resume', ['scripts/check_v1_4_resume.py', '--output', str(folder / 'smoke/resume_01')]),
            ('bundle', ['scripts/build_v1_4_colab.py', '--revision', 'r1']),
        ]
        for name, args in commands:
            state.update(status='running', step=name)
            record()
            print(name, flush=True)
            with (out / (name + '.log')).open('x') as log:
                subprocess.run([sys.executable] + args, cwd=ROOT,
                               env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(status='local_preparation_passed_pending_review', gpu_smoke='not_run', p3_training='not_run')
    except BaseException as error:
        state.update(status='stopped', error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        record()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-p1', action='store_true')
    run(parser.parse_args().wait_for_p1)
