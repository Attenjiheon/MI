import copy
import json
from pathlib import Path

import pytest
import torch

from corpus.v1_3 import cells, rng, sample_pair
from interp_v1_2.model import Transformer as V12Transformer
from interp_v1_2.training import Progress as V12Progress
from interp_v1_2.training import lm_optimizer as v12_optimizer
from interp_v1_2.training import lm_update as v12_update
from interp_v1_2.runtime import deterministic
from interp_v1_3.behavior import gate, select_pilot
from interp_v1_3.model import ARCHITECTURES, Transformer, batch, parameter_count
from interp_v1_3.training import Progress, lm_optimizer, lm_update, loss_weights


@pytest.fixture(autouse=True)
def setup():
    deterministic(101)
    torch.set_num_threads(2)


def sequences():
    records = json.loads(Path("experiment_v1_2/debug/sequences.json").read_text())
    return [record["token_ids"] for record in records]


def test_architecture_parameter_counts_and_shapes():
    expected = {"base4": 797_184, "wide4": 1_640_544, "deep8": 1_590_272}
    sample = sequences()[:2]
    ids, mask, _ = batch(sample)
    for architecture, count in expected.items():
        model = Transformer(architecture)
        assert parameter_count(model) == count
        assert len(model.blocks) == ARCHITECTURES[architecture].blocks
        assert model(ids, mask).shape == (*ids.shape, 15)


def test_base4_initialization_and_uniform_update_bitwise_match_v12():
    deterministic(0)
    old = V12Transformer()
    deterministic(0)
    new = Transformer("base4")
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    old_optimizer = v12_optimizer(old)
    new_optimizer = lm_optimizer(new)
    subset = sequences()[:64]
    old_log = v12_update(old, old_optimizer, subset, V12Progress(), microbatch=16)
    new_log = lm_update(new, new_optimizer, subset, Progress(), loss_id="uniform", microbatch=16)
    assert old_log["loss"] == new_log["loss"]
    assert old_log["tokens"] == new_log["prediction_tokens"]
    assert old_log["lr"] == new_log["lr"]
    assert old_log["gradient_norm"] == new_log["gradient_norm"]
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    old_state, new_state = old_optimizer.state_dict(), new_optimizer.state_dict()
    assert old_state["param_groups"] == new_state["param_groups"]
    assert old_state["state"].keys() == new_state["state"].keys()
    for parameter, values in old_state["state"].items():
        assert values.keys() == new_state["state"][parameter].keys()
        for key, value in values.items():
            other = new_state["state"][parameter][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


def test_read4_uses_only_token_pattern_and_expected_denominator():
    ids, mask, targets = batch(sequences()[:8])
    uniform, answer = loss_weights(ids, targets, mask, "uniform")
    weighted, weighted_answer = loss_weights(ids, targets, mask, "read4")
    assert torch.equal(answer, weighted_answer)
    assert int(answer.sum()) > 0
    assert torch.equal(weighted, uniform + 3 * answer)
    assert torch.all((targets[answer] == 13) | (targets[answer] == 14))
    broken = targets.clone()
    broken[answer] = 3
    with pytest.raises(ValueError, match="B0/B1"):
        loss_weights(ids, broken, mask, "read4")


def test_first_repeat_all_42_cells_are_valid_and_semantically_paired():
    for cell in cells():
        pair, _, _ = sample_pair(rng("unit_pair", cell.index), "unit_pair", cell)
        first = pair["origin"]["read_events"][pair["origin_target_read_id"]]
        repeated = pair["repeat"]["read_events"][pair["repeat_target_read_id"]]
        assert first["answer"] == repeated["answer"]
        assert first["state_at_read"] == repeated["state_at_read"]
        assert first["structural_depth"] == repeated["structural_depth"]
        assert first["reads_of_query_var_since_last_update"] == 0
        assert repeated["reads_of_query_var_since_last_update"] == 1
        assert 8 <= len(pair["origin"]["blocks"]) <= 23
        assert 9 <= len(pair["repeat"]["blocks"]) <= 24


def metric(ce, accuracy, validated=True):
    cells42 = {str(index): {"count": 16, "answer_ce": ce, "accuracy": accuracy} for index in range(42)}
    groups = lambda names: {name: {"count": 1, "answer_ce": ce, "accuracy": accuracy} for name in names}
    pair_member = {
        "cell": {"cells": cells42, "macro_answer_ce": ce, "macro_accuracy": accuracy},
        "operator": {"cells": groups(("NOT", "AND", "OR", "XOR")), "macro_answer_ce": ce, "macro_accuracy": accuracy},
        "depth": {"cells": groups(("1", "2-3", "4+")), "macro_answer_ce": ce, "macro_accuracy": accuracy},
        "answer": {"cells": groups(("0", "1")), "macro_answer_ce": ce, "macro_accuracy": accuracy},
    }
    return {"validated": validated, "select_general": {"answer_ce": ce, "accuracy": accuracy}, "select_pairs": {"first": pair_member, "repeat": copy.deepcopy(pair_member)}}


def test_pilot_selection_and_gate_follow_frozen_rules():
    design = json.loads(Path("experiment_v1_3/design_config.json").read_text())
    candidates = {candidate: metric(0.5, 0.8) for candidate in design["pilot"]["cells"]}
    candidates["wide4_read4"] = metric(0.44, 0.8)
    result = select_pilot(candidates, design)
    assert result["decision"] == "promote" and result["winner"] == "wide4_read4"
    candidates["wide4_read4"] = metric(0.49, 0.8)
    assert select_pilot(candidates, design)["decision"] == "pilot_failure"
    value = metric(0.1, 0.96)
    general = {"accuracy": 0.99}
    diagnostics = {name: {"accuracy": 0.95} for name in ("other_variable", "repeated_update", "first_read_after_set")}
    assert gate(general, diagnostics, value["select_pairs"], design["gate"])["passed"]
    value["select_pairs"]["first"]["depth"]["cells"]["4+"]["accuracy"] = 0.94
    assert not gate(general, diagnostics, value["select_pairs"], design["gate"])["passed"]
