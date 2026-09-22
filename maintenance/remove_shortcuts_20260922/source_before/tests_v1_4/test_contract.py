import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from corpus.v1_4 import cells, rng, sample_pair
from interp_v1_2.runtime import deterministic
from interp_v1_4.behavior import gate
from interp_v1_4.model import ARCHITECTURES, Transformer, batch, parameter_count
from interp_v1_4.training import Progress, learning_rate, loss_weights


@pytest.fixture(autouse=True)
def setup():
    deterministic(101)
    torch.set_num_threads(2)


def sequences():
    records = json.loads(Path("experiment_v1_2/debug/sequences.json").read_text())
    return [record["token_ids"] for record in records]


def pair_metric(ce=0.1, accuracy=0.96, quota=64):
    cell_rows = {
        cell.cell_id: {"count": quota, "answer_ce": ce, "accuracy": accuracy}
        for cell in cells()
    }
    groups = lambda names, counts: {
        name: {"count": counts[name], "answer_ce": ce, "accuracy": accuracy}
        for name in names
    }
    operator_counts = {"NOT": 6 * quota, "AND": 12 * quota, "OR": 12 * quota, "XOR": 12 * quota}
    depth_counts = {"1": 14 * quota, "2-3": 14 * quota, "4+": 14 * quota}
    answer_counts = {"0": 21 * quota, "1": 21 * quota}
    member = {
        "cell": {"cells": cell_rows, "macro_answer_ce": ce, "macro_accuracy": accuracy},
        "operator": {
            "cells": groups(("NOT", "AND", "OR", "XOR"), operator_counts),
            "macro_answer_ce": ce,
            "macro_accuracy": accuracy,
        },
        "depth": {
            "cells": groups(("1", "2-3", "4+"), depth_counts),
            "macro_answer_ce": ce,
            "macro_accuracy": accuracy,
        },
        "answer": {
            "cells": groups(("0", "1"), answer_counts),
            "macro_answer_ce": ce,
            "macro_accuracy": accuracy,
        },
    }
    return {"pair_count": 42 * quota, "first": member, "repeat": copy.deepcopy(member)}


def milestone(ce, general_ce, update, prediction_tokens):
    return {
        "current": {
            "checkpoint": f"checkpoints/update_{update:06d}.pt",
            "update": update,
            "prediction_tokens": prediction_tokens,
            "validation": {
                "general": {"answer_ce": general_ce, "accuracy": 0.5},
                "pairs": pair_metric(ce=ce, accuracy=0.5),
            },
        }
    }


def test_architecture_parameter_count_shape_and_initialization_contract():
    architecture = ARCHITECTURES["deepwide12"]
    assert (architecture.blocks, architecture.width, architecture.heads, architecture.head_width, architecture.mlp_width) == (
        12,
        256,
        4,
        64,
        1024,
    )
    model = Transformer("deepwide12")
    assert parameter_count(model) == 9_485_312
    assert len(model.blocks) == 12
    assert model.embedding.weight.data_ptr() != model.unembedding.weight.data_ptr()
    ids, mask, _ = batch(sequences()[:2])
    assert model(ids, mask).shape == (*ids.shape, 15)


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
        assert pair["schema_version"] == "language-v1.4-first-repeat-pair"
        assert first["answer"] == repeated["answer"]
        assert first["state_at_read"] == repeated["state_at_read"]
        assert first["structural_depth"] == repeated["structural_depth"]
        assert first["reads_of_query_var_since_last_update"] == 0
        assert repeated["reads_of_query_var_since_last_update"] == 1


def test_gate_follows_all_frozen_thresholds():
    design = json.loads(Path("experiment_v1_4/design_config.json").read_text())
    pairs = pair_metric()
    general = {"accuracy": 0.99}
    diagnostics = {
        name: {"accuracy": 0.95}
        for name in ("other_variable", "repeated_update", "first_read_after_set")
    }
    assert gate(general, diagnostics, pairs, design["gate"])["passed"]
    pairs["first"]["depth"]["cells"]["4+"]["accuracy"] = 0.94
    assert not gate(general, diagnostics, pairs, design["gate"])["passed"]


def test_stage_contract_matches_frozen_64m_budget_and_cursor():
    from interp_v1_4.cli import stage_contract

    contract = stage_contract(Path("."), "p3", "deepwide12_read4")
    assert contract["budget"] == 64_000_000
    assert contract["milestones"] == [
        1_000_000,
        3_000_000,
        8_000_000,
        16_000_000,
        24_000_000,
        32_000_000,
        48_000_000,
        57_600_000,
        60_800_000,
        64_000_000,
    ]
    assert contract["expected"] == contract["frozen"]["milestones"]["64000000"]
    assert (contract["architecture"], contract["loss"], contract["expected_parameters"]) == (
        "deepwide12",
        "read4",
        9_485_312,
    )
    with pytest.raises(ValueError, match="Only the frozen"):
        stage_contract(Path("."), "p3", "wide4_read4")


def test_checkpoint_selection_uses_global_near_tie_then_general_then_earlier():
    from interp_v1_4.cli import select_checkpoint

    values = {
        "1000000": milestone(0.90, 0.40, 136, 1_002_099),
        "3000000": milestone(0.50, 0.50, 407, 3_004_531),
        "8000000": milestone(0.60, 0.30, 1082, 8_001_583),
    }
    assert select_checkpoint(values, "p3", 8_000_000, 1e-4)["update"] == 407
    values["8000000"] = milestone(0.50 + 5e-5, 0.10, 1082, 8_001_583)
    assert select_checkpoint(values, "p3", 8_000_000, 1e-4)["update"] == 1082
    values["8000000"] = milestone(0.50, 0.50, 1082, 8_001_583)
    assert select_checkpoint(values, "p3", 8_000_000, 1e-4)["update"] == 407


def test_progress_cursor_and_token_accounting_is_validated_on_resume():
    from interp_v1_4.cli import validate_progress

    rows = sequences()[:256]
    state = Progress(update=2, prediction_tokens=sum(len(r) - 1 for r in rows[:128]), next_data_cursor=128)
    validate_progress(state, rows)
    with pytest.raises(ValueError, match="cursor/update"):
        validate_progress(Progress(update=2, prediction_tokens=0, next_data_cursor=64), rows)
    with pytest.raises(ValueError, match="token cursor"):
        validate_progress(Progress(update=2, prediction_tokens=1, next_data_cursor=128), rows)


def test_learning_rate_is_warmup_stable_decay_without_extension():
    assert learning_rate(25_000) == pytest.approx(1.5e-4)
    assert learning_rate(50_000) == pytest.approx(3e-4)
    assert learning_rate(57_600_000) == pytest.approx(3e-4)
    assert learning_rate(60_800_000) == pytest.approx(1.65e-4)
    assert learning_rate(64_000_000) == pytest.approx(3e-5)
    assert learning_rate(70_000_000) == pytest.approx(3e-5)


def test_runner_and_smoke_never_reference_the_test_split():
    for name in ("cli.py", "smoke.py", "behavior.py"):
        source = Path("interp_v1_4") / name
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
        assert not [text for text in literals if "test" in text and "/" in text], name


def test_production_run_refuses_cpu_and_absent_gpu_smoke(tmp_path):
    import argparse

    from interp_v1_4.cli import run as cli_run

    args = argparse.Namespace(
        root=Path("."),
        output=tmp_path / "run",
        cell="deepwide12_read4",
        stage="p3",
        lm_seed=0,
        device="cpu",
        microbatch=16,
        persistent_dir=None,
        resume=None,
        smoke_report=None,
        debug=False,
        pause_after_updates=None,
    )
    with pytest.raises(ValueError, match="CUDA"):
        cli_run(args)
    assert not (tmp_path / "run").exists()


def test_design_is_exactly_the_promoted_proposal_contract():
    design = json.loads(Path("experiment_v1_4/design_config.json").read_text())
    proposal = json.loads(Path("experiment_v1_3/results/next_architecture_proposal.json").read_text())
    architecture = proposal["architecture"]
    assert design["architecture"]["candidate_id"] == proposal["candidate_id"]
    assert design["architecture"]["blocks"] == architecture["blocks"]
    assert design["architecture"]["residual_width"] == architecture["residual_width"]
    assert design["architecture"]["expected_trainable_parameters"] == architecture["expected_trainable_parameters"]
    assert design["training"]["nominal_prediction_tokens"] == proposal["training"]["nominal_prediction_tokens"]
    assert design["checkpoint_selection"]["candidate_milestones_prediction_tokens"] == proposal["evaluation"]["checkpoint_milestones"]
    assert design["evaluation_splits"]["select"]["first_repeat_pairs_per_cell"] == 64


def test_frozen_corpus_preserves_v13_prefix_bytes_and_quotas():
    data = Path(json.loads(Path("experiment_v1_4/corpus_rebuild.json").read_text())["active_data_root"])
    manifest = json.loads((data / "manifest.json").read_text())
    audit = json.loads((data / "postwrite_audit.json").read_text())
    assert manifest["frozen_training"]["nominal_prediction_tokens"] == 64_000_000
    assert manifest["frozen_training"]["prefix_prediction_tokens"] == 32_004_917
    assert manifest["prefix_contract"]["shards"] == 68
    assert audit["status"] == "passed"
    assert audit["v1_3_train_prefix_byte_identical"] == "passed"
    assert audit["first_repeat"] == {
        "select": {"pairs": 2688, "members": 5376, "cells": 42},
        "gate": {"pairs": 2688, "members": 5376, "cells": 42},
        "test": {"pairs": 5376, "members": 10752, "cells": 42},
    }


def test_verify_inputs_binds_every_config_and_corpus_file():
    from interp_v1_4.runtime import verify_inputs

    checked = verify_inputs(Path("."))
    assert checked["verified_files"] > 0
    assert len(checked["configs"]) == len(checked["corpus_manifest"]) == 64


def test_history_loader_keeps_prefixes_without_surviving_rows(tmp_path):
    from scripts.generate_v1_4_corpus import load_prior_registry
    (tmp_path / "all_sequence_hashes.txt").write_text("historical-sequence\n")
    (tmp_path / "reserved_prefix_hashes.txt").write_text("historical-prefix\n")
    hashes, prefixes = load_prior_registry(tmp_path)
    assert "historical-sequence" in hashes and "historical-prefix" in prefixes


def test_v14_builder_seed_metadata_and_repeatability(tmp_path):
    import gzip
    from corpus.v1_4 import seed
    from scripts.generate_v1_4_corpus import V14Builder
    outputs = []
    for name in ("a", "b"):
        builder = V14Builder(tmp_path / name, set(), set())
        builder.produce("fixture", "unit_fixture", 3, 2)
        with gzip.open(tmp_path / name / "fixture.jsonl.gz", "rt") as handle:
            outputs.append(handle.read())
        assert builder.manifest["fixture"]["rng_seed"] == seed("unit_fixture", 3)
        assert all(json.loads(line)["rng_seed"] == seed("unit_fixture", 3) for line in outputs[-1].splitlines())
    assert outputs[0] == outputs[1]


def test_pair_coverage_rejects_empty_groups_and_wrong_ids():
    from interp_v1_4.behavior import valid_pair_coverage
    pairs = pair_metric()
    assert valid_pair_coverage(pairs, 64)
    pairs["first"]["operator"]["cells"] = {}
    assert not valid_pair_coverage(pairs, 64)
    pairs = pair_metric()
    values = pairs["first"]["cell"]["cells"]
    values["unknown"] = values.pop(next(iter(values)))
    assert not valid_pair_coverage(pairs, 64)


def test_debug_lm_resume_is_bitwise_and_keeps_cursor(tmp_path):
    from interp_v1_4.smoke import check_resume
    from interp_v1_4.training import lm_optimizer, lm_update
    model = Transformer()
    optimizer = lm_optimizer(model)
    state = Progress()
    rows = sequences()
    lm_update(model, optimizer, rows[:64], state, loss_id="read4", microbatch=2)
    result = check_resume(model, optimizer, state, rows, tmp_path,
                          "deepwide12_read4", "read4", {"debug": "unit"},
                          "cpu", "deepwide12", 2)
    assert result["status"] == "passed" and result["next_cursor"] == 128


def test_interpretation_smoke_width256():
    from interp_v1_4.integration import check_interpretation
    records = json.loads(Path("experiment_v1_2/debug/sequences.json").read_text())
    report = check_interpretation(Transformer(), records, 2)
    assert report["probe"]["status"] == "passed"
    for kind in ("sae", "transcoder"):
        for k in (4, 16):
            assert report[f"{kind}_k{k}"]["updates"] == 100
        assert report[kind + "_patches"]["identity"] == "bitwise"
