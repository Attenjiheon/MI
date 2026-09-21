import copy
import json
from pathlib import Path

import pytest

from interp_v1_4.cli import stage_contract
from interp_v1_4.replication import AUDIT, replication_authorization, validate_seed

ROOT = Path(__file__).resolve().parents[1]


def test_stage_seed_namespace():
    for stage, seed in [('p3', 0), ('p4', 1), ('p4', 2)]:
        validate_seed(stage, seed)
    for stage, seed in [('p3', 1), ('p4', 0), ('p4', 3), ('p4', -1), ('test', 1)]:
        with pytest.raises(ValueError):
            validate_seed(stage, seed)


def test_replication_requires_audited_inputs_and_preserves_budget_and_batch():
    contract = stage_contract(ROOT, 'p4', 'deepwide12_read4')
    frozen = json.loads((ROOT / AUDIT / 'frozen_seed0.json').read_text())
    hashes = frozen['hashes']
    authorization = replication_authorization(ROOT, contract, hashes, 16)
    assert authorization['selected_seed0_checkpoint_sha256'] == frozen['selected_checkpoint_sha256']
    with pytest.raises(ValueError, match='microbatch'):
        replication_authorization(ROOT, contract, hashes, 8)
    changed = copy.deepcopy(hashes)
    changed['configs'] = 'changed'
    with pytest.raises(ValueError, match='configs'):
        replication_authorization(ROOT, contract, changed, 16)
    changed_contract = copy.deepcopy(contract)
    changed_contract['expected']['cursor'] -= 64
    with pytest.raises(ValueError, match='budget'):
        replication_authorization(ROOT, changed_contract, hashes, 16)


def test_replication_rejects_changed_completion_and_numerical_code(monkeypatch):
    from interp_v1_4 import replication
    actual_sha = replication.sha
    contract = stage_contract(ROOT, 'p4', 'deepwide12_read4')
    hashes = json.loads((ROOT / AUDIT / 'frozen_seed0.json').read_text())['hashes']
    for suffix, message in [('completion.json', 'completion'), ('training.py', 'numerical')]:
        monkeypatch.setattr(replication, 'sha', lambda p: 'bad' if str(p).endswith(suffix) else actual_sha(p))
        with pytest.raises(ValueError, match=message):
            replication_authorization(ROOT, contract, hashes, 16)
