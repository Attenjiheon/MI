"""Verify one recovered immutable v1.4 run; optionally re-evaluate the selected checkpoint.

The optional re-evaluation repeats `select/` only. The P3 `gate/` result is checked
against its recorded quotas, selected checkpoint, and frozen thresholds without opening the
gate split a second time. `test/` is never opened.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from interp_v1_4.behavior import LEGACY_DIAGNOSTICS, gate as gate_decision
from interp_v1_4.cli import evaluate_select, select_checkpoint, stage_contract, validate_progress
from interp_v1_4.data import token_rows
from interp_v1_4.model import Transformer, parameter_count
from interp_v1_4.persistence import atomic_json, read_index
from interp_v1_4.runtime import deterministic, sha, tensor_digest, verify_inputs
from interp_v1_4.training import Progress

EFFECTIVE_BATCH = 64


def verify_recorded_gate(result, thresholds):
    """Validate a recorded P3 gate without reading the gate split again."""
    if result.get("debug_only"):
        if result["gate"] is not None:
            raise ValueError("Debug run must not contain a gate result")
        return None
    if result["stage"] != "p3":
        raise ValueError("Unsupported stage: " + result["stage"])
    recorded = result["gate"]
    if not recorded:
        raise ValueError("P3 run is missing its one-time gate result")
    if recorded["checkpoint"] != result["selected"]["checkpoint"]:
        raise ValueError("Gate checkpoint differs from the frozen selected checkpoint")
    if recorded["checkpoint_sha256"] != result["selected"]["sha256"]:
        raise ValueError("Gate checkpoint hash differs from the frozen selected checkpoint")
    general = recorded["general"]
    if general["sequence_count"] != 1024:
        raise ValueError("gate/general quota changed")
    diagnostics = recorded["diagnostics"]
    if set(diagnostics) != set(LEGACY_DIAGNOSTICS):
        raise ValueError("Gate diagnostic set changed")
    for name, value in diagnostics.items():
        if value["sequence_count"] != 1024 or value["answer_count"] != 1024:
            raise ValueError("Gate diagnostic quota changed: " + name)
    pairs = recorded["pairs"]
    if pairs["pair_count"] != 2688:
        raise ValueError("gate/first_repeat pair quota changed")
    expected_cells = {
        f"{operator}:{pattern}:depth_{depth}"
        for operator in ("NOT", "AND", "OR", "XOR")
        for pattern in (("0", "1") if operator == "NOT" else ("00", "01", "10", "11"))
        for depth in ("1", "2-3", "4+")
    }
    for member in ("first", "repeat"):
        cells = pairs[member]["cell"]["cells"]
        if set(cells) != expected_cells or any(value["count"] != 64 for value in cells.values()):
            raise ValueError("Gate 42-cell coverage or quota changed: " + member)
    first = pairs["first"]
    expected_groups = {
        "operator": {"NOT": 384, "AND": 768, "OR": 768, "XOR": 768},
        "depth": {"1": 896, "2-3": 896, "4+": 896},
        "answer": {"0": 1344, "1": 1344},
    }
    for group, counts in expected_groups.items():
        values = first[group]["cells"]
        if set(values) != set(counts) or any(values[name]["count"] != count for name, count in counts.items()):
            raise ValueError("Gate first-member group quota changed: " + group)
    recomputed = gate_decision(general, diagnostics, pairs, thresholds)
    if recorded["decision"] != recomputed:
        raise ValueError("Recorded gate decision does not follow the frozen thresholds")
    expected_status = "passed" if recomputed["passed"] else "failed"
    if result["status"] != expected_status:
        raise ValueError("Run status disagrees with the gate decision")
    return dict(
        checkpoint=recorded["checkpoint"],
        checkpoint_sha256=recorded["checkpoint_sha256"],
        decision=recomputed,
        general_accuracy=general["accuracy"],
        diagnostic_accuracy={name: value["accuracy"] for name, value in diagnostics.items()},
        first_macro_accuracy=pairs["first"]["cell"]["macro_accuracy"],
        repeat_macro_accuracy=pairs["repeat"]["cell"]["macro_accuracy"],
        operator_accuracy={name: value["accuracy"] for name, value in pairs["first"]["operator"]["cells"].items()},
        depth_accuracy={name: value["accuracy"] for name, value in pairs["first"]["depth"]["cells"].items()},
        answer_accuracy={name: value["accuracy"] for name, value in pairs["first"]["answer"]["cells"].items()},
    )


def verify(run, reevaluate=False, device="cpu", anchor=None, debug=False):
    """`debug` verifies a short debug run with the same contract logic, for runner self-tests."""
    run = Path(run)
    hashes = verify_inputs(ROOT)
    index = read_index(run)
    for name, digest in index["files"].items():
        if sha(run / name) != digest:
            raise ValueError("Checksum mismatch: " + name)
    result = json.loads((run / "result.json").read_text())
    assert result["version"] == "v1.4" and result["debug_only"] == debug
    assert result["hashes"]["configs"] == hashes["configs"]
    assert result["hashes"]["corpus_manifest"] == hashes["corpus_manifest"]
    for name, digest in result["hashes"]["code"].items():
        assert sha(ROOT / name) == digest, name
    contract = stage_contract(ROOT, result["stage"], result["cell"])
    expected = contract["expected"]
    assert result["nominal_budget"] == contract["budget"]
    assert result["parameter_count"] == contract["expected_parameters"]
    data_root = ROOT / contract["config"]["data_root"]
    events = [json.loads(p.read_text()) for p in sorted((run / "events").glob("*.json"))]
    if debug:
        rows = [r["token_ids"] for r in json.loads((ROOT / "experiment_v1_2/debug/sequences.json").read_text())]
        expected = dict(
            update=len(events),
            cursor=len(events) * EFFECTIVE_BATCH,
            prediction_tokens=sum(len(r) - 1 for r in rows[: len(events) * EFFECTIVE_BATCH]),
        )
        boundaries = [contract["budget"]]
    else:
        rows = list(token_rows((data_root / "train_shards").glob("*.tokens.jsonl")))
        boundaries = contract["milestones"]
    assert len(events) == expected["update"], (len(events), expected["update"])
    cumulative = 0
    milestones = {}
    evaluations = 0
    for update, event in enumerate(events, 1):
        previous = cumulative
        cumulative += sum(len(r) - 1 for r in rows[(update - 1) * EFFECTIVE_BATCH : update * EFFECTIVE_BATCH])
        state = event["state"]
        assert state["update"] == update
        assert state["prediction_tokens"] == cumulative
        assert state["next_data_cursor"] == update * EFFECTIVE_BATCH
        reached = (
            boundaries if (debug and update == expected["update"]) else [b for b in boundaries if previous < b <= cumulative]
        )
        assert ("validation" in event) == bool(reached), update
        if not reached:
            continue
        evaluations += 1
        name = f"checkpoints/update_{update:06d}.pt"
        payload = torch.load(run / name, map_location="cpu", weights_only=False)
        assert payload["debug_only"] == debug and payload["hashes"] == result["hashes"]
        assert payload["state"] == state and payload["microbatch"] == event["microbatch"]
        assert payload["cell"] == result["cell"] and payload["stage"] == result["stage"]
        assert "optimizer" in payload and "rng_states" in payload
        digest = tensor_digest(payload["model"])
        assert payload["model_tensor_sha256"] == digest
        for boundary in reached:
            stamped = json.loads((run / f"milestones/{boundary}.json").read_text())
            assert stamped["current"]["checkpoint"] == name
            assert stamped["current"]["update"] == update
            assert stamped["current"]["prediction_tokens"] == cumulative
            assert stamped["current"]["model_tensor_sha256"] == digest
            assert stamped["current"]["sha256"] == sha(run / name)
            assert stamped["current"]["validation"] == event["validation"]
            assert result["milestones"][str(boundary)]["current"]["validation"] == event["validation"]
            milestones[str(boundary)] = dict(
                update=update,
                prediction_tokens=cumulative,
                checkpoint=name,
                sha256=stamped["current"]["sha256"],
                model_tensor_sha256=digest,
            )
    assert sorted(map(int, milestones)) == sorted(boundaries)
    state = Progress(**result["final_state"])
    validate_progress(state, rows)
    assert (state.update, state.next_data_cursor, state.prediction_tokens) == (
        expected["update"],
        expected["cursor"],
        expected["prediction_tokens"],
    )
    assert cumulative == expected["prediction_tokens"]
    recomputed = select_checkpoint(
        result["milestones"], "debug" if debug else result["stage"], contract["budget"], contract["near_tie_nats"]
    )
    recomputed["sha256"] = sha(run / recomputed["checkpoint"])
    assert result["selected"] == recomputed, "Recorded selection does not follow the frozen rule"
    assert index["checkpoint"] == result["last_checkpoint"]
    assert sha(run / index["checkpoint"]) == result["last_checkpoint_sha256"]
    init = torch.load(run / "checkpoints/init.pt", map_location="cpu", weights_only=False)
    assert init["state"] == {"update": 0, "prediction_tokens": 0, "next_data_cursor": 0}
    assert not init["optimizer"]["state"] and init["hashes"] == result["hashes"] and init["debug_only"] == debug
    checked = dict(
        status="passed",
        scope="run_contract_and_retention",
        stage=result["stage"],
        cell=result["cell"],
        lm_seed=result["lm_seed"],
        run_status=result["status"],
        selected=result["selected"],
        actual_tokens=cumulative,
        overshoot=cumulative - contract["budget"],
        updates=len(events),
        evaluations=evaluations,
        milestones=milestones,
        environment_ids=sorted({event["environment_id"] for event in events}),
        anchor=None,
        reevaluation=None,
    )
    thresholds = json.loads((ROOT / "experiment_v1_4/configs/evaluation.json").read_text())["gate"]
    checked["gate"] = verify_recorded_gate(result, thresholds)
    if anchor:
        reference = torch.load(anchor, map_location="cpu", weights_only=False)
        reference_digest = tensor_digest(reference["model"] if "model" in reference else reference)
        matches = {
            boundary: value["model_tensor_sha256"] == reference_digest for boundary, value in milestones.items()
        }
        checked["anchor"] = dict(
            reference=str(anchor), reference_model_tensor_sha256=reference_digest, milestone_matches=matches
        )
    if reevaluate and not debug:
        deterministic(result["lm_seed"])
        torch.set_num_threads(2)
        model = Transformer(result["architecture"]).to(device)
        selected_payload = torch.load(
            run / result["selected"]["checkpoint"], map_location="cpu", weights_only=False
        )
        model.load_state_dict(selected_payload["model"])
        assert parameter_count(model) == contract["expected_parameters"]
        actual = evaluate_select(model, data_root, selected_payload["microbatch"])
        expected_validation = result["selected"]["validation"]
        # Cross-environment CE rounding is allowed; counts and gate-relevant rates must agree.
        assert actual["general"]["answer_count"] == expected_validation["general"]["answer_count"]
        assert abs(actual["general"]["answer_ce"] - expected_validation["general"]["answer_ce"]) < 1e-5
        assert abs(actual["general"]["accuracy"] - expected_validation["general"]["accuracy"]) < 1e-8
        assert actual["pairs"]["pair_count"] == expected_validation["pairs"]["pair_count"]
        for member in ("first", "repeat"):
            observed = actual["pairs"][member]["cell"]
            recorded = expected_validation["pairs"][member]["cell"]
            assert abs(observed["macro_answer_ce"] - recorded["macro_answer_ce"]) < 1e-5
            assert abs(observed["macro_accuracy"] - recorded["macro_accuracy"]) < 1e-8
        checked["reevaluation"] = actual
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--anchor-checkpoint", type=Path, help="Preserved v1.2 checkpoint for the base4_uniform anchor")
    parser.add_argument("--debug", action="store_true", help="Verify a debug run produced by --debug training")
    args = parser.parse_args()
    atomic_json(
        args.output,
        verify(args.run, args.reevaluate, args.device, args.anchor_checkpoint, args.debug),
        immutable=True,
    )
    print("Run evidence verified; the local smoke/environment audit is still required.")
