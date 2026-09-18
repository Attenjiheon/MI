"""Verify one recovered immutable v1.3 run; optionally re-evaluate the selected checkpoint.

The re-evaluation repeats `select/` only. `gate/` is re-read exclusively for confirmatory runs
that already recorded a gate decision, and `test/` is never opened.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from interp_v1_3.cli import evaluate_select, select_checkpoint, stage_contract, validate_progress
from interp_v1_3.data import token_rows
from interp_v1_3.model import Transformer, parameter_count
from interp_v1_3.persistence import atomic_json, read_index
from interp_v1_3.runtime import deterministic, sha, tensor_digest, verify_inputs
from interp_v1_3.training import Progress

EFFECTIVE_BATCH = 64


def verify(run, reevaluate=False, device="cpu", anchor=None, debug=False):
    """`debug` verifies a short debug run with the same contract logic, for runner self-tests."""
    run = Path(run)
    hashes = verify_inputs(ROOT)
    index = read_index(run)
    for name, digest in index["files"].items():
        if sha(run / name) != digest:
            raise ValueError("Checksum mismatch: " + name)
    result = json.loads((run / "result.json").read_text())
    assert result["version"] == "v1.3" and result["debug_only"] == debug
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
        result["milestones"], "pilot" if debug else result["stage"], contract["budget"], contract["near_tie_nats"]
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
        model.load_state_dict(
            torch.load(run / result["selected"]["checkpoint"], map_location="cpu", weights_only=False)["model"]
        )
        assert parameter_count(model) == contract["expected_parameters"]
        actual = evaluate_select(model, data_root, 16)
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
