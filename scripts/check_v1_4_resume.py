"""Exercise the real debug CLI, persistent-index recovery, and evidence verifier."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from interp_v1_4.persistence import atomic_json, recover
from interp_v1_4.runtime import sha, tensor_digest


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def run(out, stage="p3", lm_seed=0):
    out.mkdir(parents=True, exist_ok=False)
    base = [sys.executable, '-m', 'interp_v1_4.cli', '--root', str(ROOT),
            '--cell', 'deepwide12_read4', '--stage', stage, '--lm-seed', str(lm_seed),
            '--debug', '--device', 'cpu', '--microbatch', '2']
    subprocess.run(base + ['--output', str(out / 'full')], check=True, cwd=ROOT)
    subprocess.run(base + ['--output', str(out / 'paused'), '--persistent-dir', str(out / 'persistent'),
                           '--pause-after-updates', '1'], check=True, cwd=ROOT)
    checkpoint = recover(out / 'persistent', out / 'resumed')
    subprocess.run(base + ['--output', str(out / 'resumed'), '--persistent-dir', str(out / 'persistent'),
                           '--resume', str(checkpoint)], check=True, cwd=ROOT)
    first = torch.load(out / 'full/checkpoints/update_000003.pt', map_location='cpu', weights_only=False)
    second = torch.load(out / 'resumed/checkpoints/update_000003.pt', map_location='cpu', weights_only=False)
    for key in ('model', 'optimizer', 'state', 'rng_states', 'microbatch', 'next_evaluation_boundary'):
        equal(first[key], second[key])
    for name in ('full', 'resumed'):
        subprocess.run([sys.executable, 'scripts/verify_v1_4_evidence.py', str(out / name), '--debug',
                        '--output', str(out / (name + '_audit.json'))], check=True, cwd=ROOT)
    atomic_json(out / 'resume_equivalence.json', dict(
        status='passed', stage=stage, lm_seed=lm_seed, debug_only=True, reuse_in_experiment=False,
        model_optimizer_rng_cursor_microbatch_next_boundary='bitwise_equal',
        state=first['state'], model_tensor_sha256=tensor_digest(first['model']),
        audits={name: sha(out / (name + '_audit.json')) for name in ('full', 'resumed')}), immutable=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['p3', 'p4'], default='p3')
    parser.add_argument('--lm-seed', type=int, default=0)
    args = parser.parse_args()
    run(args.output.resolve(), args.stage, args.lm_seed)
