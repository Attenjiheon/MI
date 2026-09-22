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


def test_recorded_confirm_gate_is_bound_to_selected_checkpoint_and_quotas():
    from scripts.verify_v1_3_evidence import verify_recorded_gate

    evaluation = json.loads(Path("experiment_v1_3/configs/evaluation.json").read_text())
    pairs = metric(0.1, 0.96)["select_pairs"]
    expected_cells = {
        f"{operator}:{pattern}:depth_{depth}": {"count": 64, "answer_ce": 0.1, "accuracy": 0.96}
        for operator in ("NOT", "AND", "OR", "XOR")
        for pattern in (("0", "1") if operator == "NOT" else ("00", "01", "10", "11"))
        for depth in ("1", "2-3", "4+")
    }
    for member in ("first", "repeat"):
        pairs[member]["cell"]["cells"] = copy.deepcopy(expected_cells)
    for name, count in {"NOT": 384, "AND": 768, "OR": 768, "XOR": 768}.items():
        pairs["first"]["operator"]["cells"][name]["count"] = count
    for name in ("1", "2-3", "4+"):
        pairs["first"]["depth"]["cells"][name]["count"] = 896
    for name in ("0", "1"):
        pairs["first"]["answer"]["cells"][name]["count"] = 1344
    general = {"sequence_count": 1024, "accuracy": 0.99}
    diagnostics = {
        name: {"sequence_count": 1024, "answer_count": 1024, "accuracy": 0.95}
        for name in ("other_variable", "repeated_update", "first_read_after_set")
    }
    decision = gate(general, diagnostics, pairs, evaluation["gate"])
    result = {
        "stage": "confirm",
        "status": "passed",
        "selected": {"checkpoint": "checkpoints/update_004249.pt", "sha256": "abc"},
        "gate": {
            "checkpoint": "checkpoints/update_004249.pt",
            "checkpoint_sha256": "abc",
            "decision": decision,
            "general": general,
            "diagnostics": diagnostics,
            "pairs": {"pair_count": 2688, **pairs},
        },
    }
    checked = verify_recorded_gate(result, evaluation["gate"])
    assert checked["decision"]["passed"]
    result["gate"]["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint hash"):
        verify_recorded_gate(result, evaluation["gate"])


def milestone(ce, general_ce, update, prediction_tokens):
    value = metric(ce, 0.5)
    return {
        "current": {
            "checkpoint": f"checkpoints/update_{update:06d}.pt",
            "update": update,
            "prediction_tokens": prediction_tokens,
            "validation": {
                "general": {"answer_ce": general_ce, "accuracy": 0.5},
                "pairs": value["select_pairs"],
            },
        }
    }


def test_stage_contract_matches_the_frozen_budget_and_cursor():
    from interp_v1_3.cli import stage_contract

    pilot = stage_contract(Path("."), "pilot", "base4_uniform")
    assert pilot["budget"] == 8_000_000 and pilot["milestones"] == [1_000_000, 3_000_000, 8_000_000]
    assert pilot["expected"] == {"cursor": 69_248, "prediction_tokens": 8_001_583, "update": 1_082}
    assert (pilot["architecture"], pilot["loss"], pilot["expected_parameters"]) == ("base4", "uniform", 797_184)
    confirm = stage_contract(Path("."), "confirm", "wide4_read4")
    assert confirm["budget"] == 32_000_000 and confirm["milestones"][-1] == 32_000_000
    assert confirm["expected"] == {"cursor": 271_936, "prediction_tokens": 32_004_917, "update": 4_249}
    assert (confirm["architecture"], confirm["loss"], confirm["expected_parameters"]) == ("wide4", "read4", 1_640_544)
    with pytest.raises(ValueError, match="six frozen pilot cells"):
        stage_contract(Path("."), "pilot", "deep8_read8")


def test_checkpoint_selection_uses_first_member_macro_ce_then_general_then_earlier():
    from interp_v1_3.cli import select_checkpoint

    milestones = {
        "1000000": milestone(0.90, 0.40, 136, 1_002_099),
        "3000000": milestone(0.50, 0.50, 407, 3_004_531),
        "8000000": milestone(0.60, 0.30, 1082, 8_001_583),
    }
    assert select_checkpoint(milestones, "pilot", 8_000_000, 1e-4)["update"] == 1082
    assert select_checkpoint(milestones, "confirm", 8_000_000, 1e-4)["update"] == 407
    milestones["8000000"] = milestone(0.50 + 5e-5, 0.10, 1082, 8_001_583)
    assert select_checkpoint(milestones, "confirm", 8_000_000, 1e-4)["update"] == 1082
    milestones["8000000"] = milestone(0.50, 0.50, 1082, 8_001_583)
    assert select_checkpoint(milestones, "confirm", 8_000_000, 1e-4)["update"] == 407


def test_progress_cursor_and_token_accounting_is_validated_on_resume():
    from interp_v1_3.cli import validate_progress

    rows = sequences()[:256]
    state = Progress(update=2, prediction_tokens=sum(len(r) - 1 for r in rows[:128]), next_data_cursor=128)
    validate_progress(state, rows)
    with pytest.raises(ValueError, match="cursor/update"):
        validate_progress(Progress(update=2, prediction_tokens=0, next_data_cursor=64), rows)
    with pytest.raises(ValueError, match="token cursor"):
        validate_progress(Progress(update=2, prediction_tokens=1, next_data_cursor=128), rows)


def test_learning_rate_is_warmup_stable_decay_without_extension():
    from interp_v1_3.training import learning_rate

    assert learning_rate(25_000) == pytest.approx(1.5e-4)
    assert learning_rate(50_000) == pytest.approx(3e-4)
    assert learning_rate(8_000_000) == pytest.approx(3e-4)
    assert learning_rate(28_800_000) == pytest.approx(3e-4)
    assert learning_rate(30_400_000) == pytest.approx(1.65e-4)
    assert learning_rate(32_000_000) == pytest.approx(3e-5)
    assert learning_rate(40_000_000) == pytest.approx(3e-5)


def test_runner_and_smoke_never_reference_the_test_split():
    """No executable string in the v1.3 runner may name a test/ data path."""
    import ast

    for name in ("cli.py", "smoke.py", "behavior.py"):
        source = Path("interp_v1_3") / name
        tree = ast.parse(source.read_text())
        docstrings = {
            id(ast.get_docstring(node, clean=False))
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node.value) not in docstrings
        ]
        assert literals, name
        assert not [text for text in literals if "test" in text and "/" in text], name


def test_production_runs_refuse_cpu_and_an_absent_gpu_smoke(tmp_path):
    import argparse

    from interp_v1_3.cli import run as cli_run

    args = argparse.Namespace(
        root=Path("."), output=tmp_path / "run", cell="base4_uniform", stage="pilot", lm_seed=0,
        device="cpu", microbatch=16, persistent_dir=None, resume=None, smoke_report=None,
        debug=False, pause_after_updates=None,
    )
    with pytest.raises(ValueError, match="CUDA"):
        cli_run(args)
    assert not (tmp_path / "run").exists()


def test_pilot_tie_break_prefers_uniform_then_fewer_parameters():
    design = json.loads(Path("experiment_v1_3/design_config.json").read_text())
    candidates = {candidate: metric(0.5, 0.8) for candidate in design["pilot"]["cells"]}
    for candidate in ("wide4_read4", "deep8_read4", "deep8_uniform"):
        candidates[candidate] = metric(0.44, 0.8)
    # Equal first-member macro CE and equal general CE: uniform wins over both read4 cells.
    assert select_pilot(candidates, design)["winner"] == "deep8_uniform"
    candidates["wide4_read4"] = metric(0.44 - 2e-4, 0.8)
    assert select_pilot(candidates, design)["winner"] == "wide4_read4"
    unvalidated = dict(candidates)
    unvalidated["wide4_read4"] = metric(0.44 - 2e-4, 0.8, validated=False)
    assert select_pilot(unvalidated, design)["winner"] == "deep8_uniform"


def test_pilot_winner_report_rejects_debug_runs(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("winner", Path("scripts/select_v1_3_pilot_winner.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "result.json").write_text(
        json.dumps({"stage": "pilot", "debug_only": True, "lm_seed": 0, "cell": "base4_uniform"})
    )
    with pytest.raises(ValueError, match="production pilot"):
        module.candidate(tmp_path, None)
