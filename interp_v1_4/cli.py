"""v1.4 P3 LM runner for the frozen deepwide12_read4 candidate.

Each LM seed consumes the same ordered 64M stream once. Selection uses `select/` only.
`gate/` is read once after training at the single selected checkpoint. `test/` is never
opened here. Seed 0 must pass before orchestration may launch seeds 1 and 2.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .behavior import (
    LEGACY_DIAGNOSTICS,
    evaluate_first_repeat,
    evaluate_records,
    gate,
    metadata,
    without_rows,
    valid_pair_coverage,
)
from .data import token_rows
from .model import ARCHITECTURES, Transformer, parameter_count
from .persistence import atomic_json, publish, read_index
from .runtime import (
    deterministic,
    environment,
    restore,
    restore_rng,
    rng_state,
    save,
    sha,
    tensor_digest,
    verify_inputs,
)
from .training import Progress, lm_optimizer, lm_update

EFFECTIVE_BATCH = 64
CODE_FOLDERS = ("interp_v1_4", "interp_v1_2", "corpus")


def split_cell(cell):
    architecture, loss = cell.rsplit("_", 1)
    if architecture not in ARCHITECTURES or loss != "read4":
        raise ValueError("Unknown v1.4 candidate: " + cell)
    return architecture, loss


def stage_contract(root, stage, cell):
    """Budgets, milestones and the frozen final cursor come from the frozen config set only."""
    config = json.loads((root / "experiment_v1_4/configs/run.json").read_text())
    transformer = json.loads((root / "experiment_v1_4/configs/transformer.json").read_text())
    if stage != "p3" or cell != config["candidate"]:
        raise ValueError("Only the frozen v1.4 P3 candidate is executable")
    budget = config["nominal_prediction_tokens"]
    milestones = list(config["milestones"])
    if budget not in milestones or sorted(milestones) != milestones:
        raise ValueError("Milestone contract does not end at the budget")
    frozen = config["frozen_training"]
    expected = frozen["milestones"][str(budget)]
    architecture, loss = split_cell(cell)
    frozen_architecture = transformer["architecture"]
    if architecture != frozen_architecture["id"] or loss != transformer["loss"]["id"]:
        raise ValueError("Candidate does not match the frozen architecture/loss")
    return dict(
        config=config,
        transformer=transformer,
        budget=budget,
        milestones=milestones,
        expected=expected,
        frozen=frozen,
        architecture=architecture,
        loss=loss,
        expected_parameters=frozen_architecture["expected_trainable_parameters"],
        near_tie_nats=transformer["checkpoint_selection"]["near_tie_nats"],
    )


def evaluate_select(model, data_root, microbatch, *, debug_records=None):
    """Selection metrics. Quotas are re-checked at every evaluation, not only at the end."""
    if debug_records is not None:
        return dict(general=without_rows(evaluate_records(model, debug_records, microbatch=microbatch)), pairs=None)
    general = evaluate_records(model, metadata(data_root / "select/general.jsonl.gz"), microbatch=microbatch)
    if general["sequence_count"] != 512:
        raise ValueError("select/general quota changed")
    pairs = evaluate_first_repeat(model, data_root / "first_repeat/select.jsonl.gz", microbatch=microbatch)
    if not valid_pair_coverage(pairs, 64):
        raise ValueError("select/first_repeat quota changed")
    return dict(general=without_rows(general), pairs=without_rows(pairs))


def evaluate_gate_suite(model, data_root, thresholds, microbatch):
    """Read once, at the single selected checkpoint, after training and selection are frozen."""
    general = evaluate_records(model, metadata(data_root / "gate/general.jsonl.gz"), microbatch=microbatch)
    if general["sequence_count"] != 1024:
        raise ValueError("gate/general quota changed")
    diagnostics = {}
    for name in LEGACY_DIAGNOSTICS:
        value = evaluate_records(
            model,
            metadata(data_root / f"gate/legacy_diagnostics/{name}.jsonl.gz"),
            target_only=True,
            microbatch=microbatch,
        )
        if value["answer_count"] != 1024 or value["sequence_count"] != 1024:
            raise ValueError("gate diagnostic quota changed: " + name)
        diagnostics[name] = value
    pairs = evaluate_first_repeat(model, data_root / "first_repeat/gate.jsonl.gz", microbatch=microbatch)
    if not valid_pair_coverage(pairs, 64):
        raise ValueError("gate/first_repeat quota changed")
    decision = gate(general, diagnostics, pairs, thresholds)
    return dict(
        decision=decision,
        general=without_rows(general),
        diagnostics={name: without_rows(value) for name, value in diagnostics.items()},
        pairs=without_rows(pairs),
    )


def first_macro_ce(entry):
    return entry["validation"]["pairs"]["first"]["cell"]["macro_answer_ce"]


def select_checkpoint(milestones, stage, budget, near_tie):
    """Debug uses its final update; production minimizes the frozen select metric."""
    if stage == "debug":
        chosen = milestones[str(budget)]["current"]
        return dict(rule="final_milestone_only", **chosen)
    entries = [value["current"] for value in milestones.values()]
    for entry in entries:
        if not math.isfinite(first_macro_ce(entry)):
            raise ValueError("Nonfinite selection metric")
    minimum = min(first_macro_ce(entry) for entry in entries)
    contenders = [entry for entry in entries if first_macro_ce(entry) - minimum <= near_tie]
    chosen = min(contenders, key=lambda e: (e["validation"]["general"]["answer_ce"], e["update"]))
    return dict(rule="minimum_first_member_42_cell_macro_answer_ce", **chosen)


def validate_progress(state, rows):
    if state.next_data_cursor != EFFECTIVE_BATCH * state.update or not 0 <= state.next_data_cursor <= len(rows):
        raise ValueError("Invalid cursor/update")
    if sum(len(row) - 1 for row in rows[: state.next_data_cursor]) != state.prediction_tokens:
        raise ValueError("Invalid token cursor")


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def run(a):
    root = a.root.resolve()
    output = a.output.resolve()
    if a.lm_seed != 0:
        raise ValueError("This P3 runner only permits seed 0; replication requires separately audited seed-0 gate evidence")
    if a.microbatch not in (16, 8, 4, 2, 1):
        raise ValueError("Invalid frozen microbatch")
    if not a.debug and (a.device != "cuda" or not torch.cuda.is_available()):
        raise ValueError("Production v1.4 runs require a CUDA runtime")
    deterministic(a.lm_seed)
    torch.set_num_threads(2)
    contract = stage_contract(root, a.stage, a.cell)
    data_root = root / contract["config"]["data_root"]
    hashes = verify_inputs(root)
    hashes["code"] = {
        str(p.relative_to(root)): sha(p)
        for folder in CODE_FOLDERS
        for p in sorted((root / folder).glob("*.py"))
    }
    hashes["run_settings"] = dict(
        schema="lm-run-v1.4", stage=a.stage, cell=a.cell, lm_seed=a.lm_seed, debug=a.debug
    )
    if not a.debug:
        if a.device != "cuda" or not torch.cuda.is_available():
            raise ValueError("Production v1.4 runs require a CUDA runtime")
        smoke = json.loads(a.smoke_report.read_text()) if a.smoke_report else {}
        if smoke.get("status") != "passed" or smoke.get("scope") != "cuda":
            raise ValueError("A current v1.4 GPU smoke report is required")
        if a.cell not in smoke.get("cells", {}):
            raise ValueError("GPU smoke did not cover this cell")
        for key in ("configs", "corpus_manifest"):
            if smoke["input_hashes"][key] != hashes[key]:
                raise ValueError("Smoke input mismatch: " + key)
        if smoke["input_hashes"]["code"] != hashes["code"]:
            raise ValueError("Smoke code mismatch")
        if a.microbatch != smoke["cells"][a.cell]["selected_microbatch"]:
            raise ValueError("Microbatch must equal the value frozen by GPU smoke")
    if a.resume:
        index = read_index(output)
        if a.resume.resolve() != (output / index["checkpoint"]).resolve():
            raise ValueError("Resume only the last complete checkpoint")
        for name, digest in index["files"].items():
            if sha(output / name) != digest:
                raise ValueError("Resume artifact checksum mismatch: " + name)
        extras = {
            str(p.relative_to(output))
            for p in output.rglob("*")
            if p.is_file() and p.suffix != ".tmp" and "indices" not in p.relative_to(output).parts and p.name != "LATEST.json"
        } - set(index["files"])
        if extras:
            raise ValueError("Uncommitted local tail; recover the persistent index into a fresh directory")
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        if a.persistent_dir and (a.persistent_dir / "LATEST.json").exists():
            raise FileExistsError("A persistent run exists at this path; recover it instead")
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    env_dir = output / "sessions" / session
    env = environment(env_dir)
    atomic_json(
        env_dir / "environment.json",
        dict(environment=env, command=shlex.join(sys.argv), resume=str(a.resume), started_at=session),
        immutable=True,
    )
    if not a.debug and smoke["environment"]["environment_id"] != env["environment_id"]:
        raise ValueError("GPU smoke environment mismatch")
    model = Transformer(contract["architecture"]).to(a.device)
    if parameter_count(model) != contract["expected_parameters"]:
        raise ValueError("Architecture parameter count differs from the frozen contract")
    optimizer = lm_optimizer(model)
    state = Progress()
    records = json.loads((root / "experiment_v1_2/debug/sequences.json").read_text()) if a.debug else None
    rows = (
        [record["token_ids"] for record in records]
        if a.debug
        else list(token_rows((data_root / "train_shards").glob("*.tokens.jsonl")))
    )
    budget = contract["budget"]
    microbatch = a.microbatch
    milestones = {}
    measurement = dict(updates=0, prediction_tokens=0, training_seconds=0.0, peak_memory_bytes=0)
    if a.resume:
        payload = restore(a.resume, model, optimizer, hashes)
        state = Progress(**payload["state"])
        validate_progress(state, rows)
        microbatch = payload["microbatch"]
        if microbatch != a.microbatch:
            raise ValueError("Resume must preserve the checkpoint microbatch")
        measurement = payload["measurement"]
        milestones = payload["milestones"]
    elif not a.debug:
        frozen = contract["frozen"]
        if len(rows) != frozen["final_cursor"] or sum(len(r) - 1 for r in rows) != frozen["actual_prediction_tokens"]:
            raise ValueError("Train stream differs from the frozen 64M contract")

    def checkpoint(name):
        path = output / name
        if path.exists():
            raise FileExistsError("Immutable checkpoint: " + name)
        tick = time.monotonic()
        digest = save(
            path,
            model,
            optimizer,
            state.payload(),
            hashes=hashes,
            debug_only=a.debug,
            microbatch=microbatch,
            measurement=measurement,
            milestones=milestones,
            next_evaluation_boundary=next((b for b in contract["milestones"] if b > state.prediction_tokens), None),
            environment_id=env["environment_id"],
            cell=a.cell,
            stage=a.stage,
            model_tensor_sha256=tensor_digest(model.state_dict()),
        )
        atomic_json(output / "storage" / (path.stem + ".json"),
                    dict(checkpoint=name, sha256=digest, serialization_seconds=time.monotonic() - tick), immutable=True)
        return digest

    if not a.resume:
        checkpoint("checkpoints/init.pt")
        publish(output, a.persistent_dir, "checkpoints/init.pt")
    last_checkpoint = read_index(output)["checkpoint"]
    last_saved = time.monotonic()
    started = time.monotonic()
    debug_end = 3
    while state.prediction_tokens < budget and (not a.debug or state.update < debug_end):
        chunk = rows[state.next_data_cursor : state.next_data_cursor + EFFECTIVE_BATCH]
        if len(chunk) != EFFECTIVE_BATCH:
            raise ValueError("Insufficient data for a complete effective batch")
        before = state.payload()
        # OOM pauses the run. Never change the smoke-frozen numerical batch
        # partition after training starts; recover the last complete index.
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        tick = time.monotonic()
        log = lm_update(model, optimizer, chunk, state, loss_id=contract["loss"], microbatch=microbatch)
        if a.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        if not math.isfinite(log["loss"]):
            raise ValueError("Nonfinite loss")
        if state.update <= 50:
            measurement["updates"] += 1
            measurement["prediction_tokens"] += log["prediction_tokens"]
            measurement["training_seconds"] += elapsed
            if a.device == "cuda":
                measurement["peak_memory_bytes"] = max(measurement["peak_memory_bytes"], torch.cuda.max_memory_allocated())
        reached = [b for b in contract["milestones"] if before["prediction_tokens"] < b <= state.prediction_tokens]
        final = state.prediction_tokens >= budget or (a.debug and state.update == debug_end)
        if a.debug and final:
            reached = [budget]
        pause = a.pause_after_updates is not None and state.update >= a.pause_after_updates and not final
        evaluation_due = bool(reached)
        checkpoint_due = evaluation_due or pause or time.monotonic() - last_saved >= 900
        checkpoint_name = f"checkpoints/update_{state.update:06d}.pt"
        log.update(state=state.payload(), microbatch=microbatch, training_seconds=elapsed, environment_id=env["environment_id"])
        if evaluation_due:
            tick = time.monotonic()
            validation = evaluate_select(
                model, data_root, microbatch, debug_records=records[:32] if a.debug else None
            )
            log["validation_seconds"] = time.monotonic() - tick
            log["validation"] = validation
            current = dict(
                checkpoint=checkpoint_name,
                update=state.update,
                prediction_tokens=state.prediction_tokens,
                model_tensor_sha256=tensor_digest(model.state_dict()),
                validation=validation,
            )
            for boundary in reached:
                milestones[str(boundary)] = dict(current=current)
        log["state"] = state.payload()
        log["microbatch"] = microbatch
        atomic_json(output / f"events/update_{state.update:06d}.json", log, immutable=True)
        if checkpoint_due:
            checkpoint(checkpoint_name)
            last_checkpoint = checkpoint_name
            for boundary, value in milestones.items():
                path = output / f"milestones/{boundary}.json"
                if not path.exists():
                    stamped = copy.deepcopy(value)
                    stamped["current"]["sha256"] = sha(output / stamped["current"]["checkpoint"])
                    atomic_json(path, stamped, immutable=True)
            publish(output, a.persistent_dir, last_checkpoint)
            last_saved = time.monotonic()
        if evaluation_due or state.update == 50:
            summary = dict(update=state.update, prediction_tokens=state.prediction_tokens, microbatch=microbatch)
            if "validation" in log:
                summary["general"] = log["validation"]["general"]["accuracy"]
                summary["general_answer_ce"] = log["validation"]["general"]["answer_ce"]
                if log["validation"]["pairs"]:
                    summary["first_macro_ce"] = log["validation"]["pairs"]["first"]["cell"]["macro_answer_ce"]
                    summary["first_macro_accuracy"] = log["validation"]["pairs"]["first"]["cell"]["macro_accuracy"]
            print(json.dumps(summary), flush=True)
        if pause:
            print("PAUSED at complete update", state.update, flush=True)
            return
    if not a.debug:
        expected = contract["expected"]
        actual = (state.update, state.next_data_cursor, state.prediction_tokens)
        if actual != (expected["update"], expected["cursor"], expected["prediction_tokens"]):
            raise ValueError("Final cursor differs from the frozen contract")
    if str(budget) not in milestones:
        raise ValueError("Final milestone evaluation missing")
    selected = select_checkpoint(milestones, "debug" if a.debug else a.stage, budget, contract["near_tie_nats"])
    selected["sha256"] = sha(output / selected["checkpoint"])
    gate_result = None
    if a.stage == "p3" and not a.debug:
        if (output / "gate.json").exists():
            gate_result = json.loads((output / "gate.json").read_text())
            if gate_result["checkpoint_sha256"] != selected["sha256"]:
                raise ValueError("Persisted gate selection mismatch")
        else:
            if (output / "gate_started.json").exists():
                raise RuntimeError("Gate was already opened but completion is missing; recover its evidence, never evaluate again")
            atomic_json(output / "gate_started.json", dict(selected=selected, session=session), immutable=True)
            publish(output, a.persistent_dir, last_checkpoint)
            del optimizer
            model.load_state_dict(torch.load(output / selected["checkpoint"], map_location="cpu", weights_only=False)["model"])
            gate_result = evaluate_gate_suite(
                model, data_root, json.loads((root / "experiment_v1_4/configs/evaluation.json").read_text())["gate"], microbatch
            )
            gate_result["checkpoint"] = selected["checkpoint"]
            gate_result["checkpoint_sha256"] = selected["sha256"]
            atomic_json(output / "gate.json", gate_result, immutable=True)
            publish(output, a.persistent_dir, last_checkpoint)
    manifest = dict(
        phase="v1.4-" + a.stage,
        version="v1.4",
        stage=a.stage,
        cell=a.cell,
        architecture=contract["architecture"],
        loss=contract["loss"],
        lm_seed=a.lm_seed,
        status="debug_only"
        if a.debug
        else ("passed" if gate_result["decision"]["passed"] else "failed"),
        debug_only=a.debug,
        hashes=hashes,
        environment=env,
        parameter_count=parameter_count(model),
        nominal_budget=budget,
        actual_prediction_tokens=state.prediction_tokens,
        overshoot=state.prediction_tokens - budget,
        final_state=state.payload(),
        milestones=milestones,
        selected=selected,
        gate=gate_result,
        last_checkpoint=last_checkpoint,
        last_checkpoint_sha256=sha(output / last_checkpoint),
        measurement_first_50_updates=measurement,
        session_seconds=time.monotonic() - started,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(output / "result.json", manifest, immutable=True)
    publish(output, a.persistent_dir, last_checkpoint)
    print(
        json.dumps(
            dict(
                status=manifest["status"],
                cell=a.cell,
                selected_update=selected["update"],
                actual_tokens=state.prediction_tokens,
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--stage", choices=["p3"], default="p3")
    parser.add_argument("--lm-seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--persistent-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pause-after-updates", type=int)
    a = parser.parse_args()
    if (a.output / "result.json").exists():
        raise FileExistsError("Run already finalized")
    try:
        run(a)
    except BaseException as error:
        if a.output.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            resource_stop = isinstance(error, (KeyboardInterrupt, torch.cuda.OutOfMemoryError)) or (
                isinstance(error, OSError) and error.errno == 28
            )
            atomic_json(
                a.output / "interruptions" / f"{stamp}.json",
                dict(
                    status="paused" if resource_stop else "failed",
                    cell=a.cell,
                    stage=a.stage,
                    error_type=type(error).__name__,
                    reason=str(error),
                    resume_policy="Recover the last completed persistent index into a fresh local directory; no partial optimizer state is saved",
                ),
                immutable=True,
            )
        raise


if __name__ == "__main__":
    main()
