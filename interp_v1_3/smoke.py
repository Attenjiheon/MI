"""v1.3 integration checks for all six frozen pilot cells. Outputs are DEBUG ONLY.

Every architecture and both loss objectives are exercised on independent debug fixtures:
shapes and parameter counts, causal/PAD masking, RoPE at each head width, gradient
accumulation equality, token-only `read4` weighting, checkpoint/resume bitwise equality,
and the real selection-metric code path on a small slice of `select/`. `gate/` and `test/`
are never scored here.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from .behavior import evaluate_first_repeat, evaluate_records, gate, select_pilot, without_rows
from .model import ARCHITECTURES, Transformer, batch, parameter_count, rope
from .runtime import deterministic, environment, restore, save, sha, tensor_digest, verified_copy, verify_inputs
from .training import Progress, lm_optimizer, lm_update, loss_weights, uniform_ce_sum, weighted_ce_sum

CELLS = ("base4_uniform", "base4_read4", "wide4_uniform", "wide4_read4", "deep8_uniform", "deep8_read4")
SMOKE_CELLS = 42
SMOKE_SEQUENCES = 24


def close(a, b, exact=False):
    torch.testing.assert_close(a, b, atol=0 if exact else 1e-5, rtol=0 if exact else 1e-4)


def head(path, count):
    with gzip.open(path, "rt") as handle:
        for index, line in enumerate(handle):
            if index >= count:
                return
            yield json.loads(line)


def write_pairs(source, destination):
    """One real select pair per cell, so the 42-cell macro and coverage paths run on real data."""
    seen = set()
    with gzip.open(source, "rt") as handle, gzip.open(destination, "wt") as out:
        for line in handle:
            pair = json.loads(line)
            if pair["cell_id"] in seen:
                continue
            seen.add(pair["cell_id"])
            out.write(json.dumps(pair) + "\n")
    if len(seen) != SMOKE_CELLS:
        raise ValueError("select/first_repeat does not contain 42 cells")
    return destination


def check_masking_and_rope(model, sequences, device):
    """Padding, causality and RoPE are re-checked at every residual width."""
    width = model.architecture.head_width
    ids, mask, targets = batch(sequences[:4], device)
    with torch.no_grad():
        logits = model(ids, mask)
        padding_error = 0.0
        for row, sequence in enumerate(sequences[:4]):
            single_ids, single_mask, _ = batch([sequence], device)
            single = model(single_ids, single_mask)
            close(logits[row, : len(sequence) - 1], single[0])
            padding_error = max(padding_error, float((logits[row, : len(sequence) - 1] - single[0]).abs().max()))
        altered = ids.clone()
        cutoff = 15
        altered[:, cutoff:] = (altered[:, cutoff:] + 1) % 15
        close(logits[:, :cutoff], model(altered, mask)[:, :cutoff], exact=True)
        altered = ids.clone()
        altered[~mask] = 14
        close(logits[mask], model(altered, mask)[mask], exact=True)
        sample = torch.randn(1, 4, 3, width, device=device)
        rotated = rope(sample)
        for position in range(3):
            for index in range(width // 2):
                angle = position * 10000 ** (-2 * index / width)
                close(
                    rotated[..., position, 2 * index],
                    sample[..., position, 2 * index] * np.cos(angle) - sample[..., position, 2 * index + 1] * np.sin(angle),
                )
    return dict(padding_max_abs_error=padding_error, causal_and_pad_mask="passed", rope_head_width=width)


def check_loss_weights(sequences, device, loss_id):
    """`read4` positions come from token IDs alone; the denominator is the weight sum."""
    ids, mask, targets = batch(sequences[:8], device)
    uniform, answer = loss_weights(ids, targets, mask, "uniform")
    weighted, weighted_answer = loss_weights(ids, targets, mask, "read4")
    assert torch.equal(answer, weighted_answer)
    assert int(answer.sum()) > 0
    assert torch.equal(weighted, uniform + 3 * answer)
    assert torch.all((targets[answer] == 13) | (targets[answer] == 14))
    assert torch.equal(uniform, mask.to(uniform.dtype))
    broken = targets.clone()
    broken[answer] = 3
    try:
        loss_weights(ids, broken, mask, "read4")
        raise AssertionError("read4 accepted a non B0/B1 answer target")
    except ValueError:
        pass
    selected = weighted if loss_id == "read4" else uniform
    return dict(
        read_answer_targets=int(answer.sum()),
        prediction_tokens=int(mask.sum()),
        loss_weight_sum=float(selected.sum()),
        rejects_non_bit_answer_target=True,
    )


def check_gradient_accumulation(model, sequences, device, loss_id):
    """One complete effective batch must equal its microbatch accumulation before clipping."""
    subset = sequences[:8]
    ids, mask, targets = batch(subset, device)
    weights, _ = loss_weights(ids, targets, mask, loss_id)
    total = float(weights.sum())
    model.zero_grad(set_to_none=True)
    logits = model(ids, mask)
    loss = uniform_ce_sum(logits, targets) if loss_id == "uniform" else weighted_ce_sum(logits, targets, weights)
    (loss / total).backward()
    reference = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for offset in range(0, 8, 2):
        part = subset[offset : offset + 2]
        ids, mask, targets = batch(part, device)
        weights, _ = loss_weights(ids, targets, mask, loss_id)
        logits = model(ids, mask)
        loss = uniform_ce_sum(logits, targets) if loss_id == "uniform" else weighted_ce_sum(logits, targets, weights)
        (loss / total).backward()
    error = 0.0
    for expected, parameter in zip(reference, model.parameters()):
        close(expected, parameter.grad)
        error = max(error, float((expected - parameter.grad).abs().max()))
    model.zero_grad(set_to_none=True)
    return error


def check_resume(model, optimizer, state, sequences, out, cell, loss_id, inputs, device, architecture):
    """The update after a restored checkpoint must reproduce bitwise."""
    checkpoint = out / f"{cell}_debug.pt"
    digest = save(checkpoint, model, optimizer, state.payload(), hashes=inputs, debug_only=True, cell=cell)
    verified_copy(checkpoint, out / f"persistent_copy/{cell}_debug.pt")
    following = sequences[state.next_data_cursor : state.next_data_cursor + 64]
    expected_log = lm_update(model, optimizer, following, state, loss_id=loss_id, microbatch=16)
    expected = copy.deepcopy(model.state_dict())
    restored = Transformer(architecture).to(device)
    restored_optimizer = lm_optimizer(restored)
    payload = restore(checkpoint, restored, restored_optimizer, inputs)
    restored_state = Progress(**payload["state"])
    actual_log = lm_update(
        restored,
        restored_optimizer,
        sequences[restored_state.next_data_cursor : restored_state.next_data_cursor + 64],
        restored_state,
        loss_id=loss_id,
        microbatch=16,
    )
    for name, value in expected.items():
        close(value, restored.state_dict()[name], exact=True)
    assert expected_log == actual_log and restored_state == state
    return dict(
        status="passed",
        checkpoint_sha256=digest,
        next_cursor=restored_state.next_data_cursor,
        next_update=restored_state.update,
        prediction_tokens=restored_state.prediction_tokens,
        model_tensor_sha256=tensor_digest(restored.state_dict()),
    )


def check_anchor_equivalence(sequences):
    """base4 + uniform must stay the audited v1.2 kernel, bit for bit."""
    from interp_v1_2.model import Transformer as LegacyTransformer
    from interp_v1_2.training import Progress as LegacyProgress
    from interp_v1_2.training import lm_optimizer as legacy_optimizer
    from interp_v1_2.training import lm_update as legacy_update

    deterministic(0)
    legacy = LegacyTransformer()
    deterministic(0)
    current = Transformer("base4")
    for name, value in legacy.state_dict().items():
        close(value, current.state_dict()[name], exact=True)
    legacy_opt = legacy_optimizer(legacy)
    current_opt = lm_optimizer(current)
    subset = sequences[:64]
    legacy_log = legacy_update(legacy, legacy_opt, subset, LegacyProgress(), microbatch=16)
    current_log = lm_update(current, current_opt, subset, Progress(), loss_id="uniform", microbatch=16)
    for name, value in legacy.state_dict().items():
        close(value, current.state_dict()[name], exact=True)
    assert legacy_log["loss"] == current_log["loss"]
    assert legacy_log["tokens"] == current_log["prediction_tokens"]
    assert legacy_log["lr"] == current_log["lr"]
    assert legacy_log["gradient_norm"] == current_log["gradient_norm"]
    return dict(
        status="passed",
        initialization="bitwise_equal",
        first_update="bitwise_equal",
        loss=current_log["loss"],
        model_tensor_sha256=tensor_digest(current.state_dict()),
    )


def run(root, out, device):
    root = Path(root).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    deterministic(20260917)
    torch.set_num_threads(2)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    inputs = verify_inputs(root)
    debug_path = root / "experiment_v1_2/debug/sequences.json"
    records = json.loads(debug_path.read_text())
    sequences = [record["token_ids"] for record in records]
    inputs["debug"] = sha(debug_path)
    inputs["code"] = {str(p.relative_to(root)): sha(p) for p in sorted((root / "interp_v1_3").glob("*.py"))}
    data_root = root / json.loads((root / "experiment_v1_3/configs/run.json").read_text())["data_root"]
    contract = json.loads((root / "experiment_v1_3/configs/transformer.json").read_text())
    design = json.loads((root / "experiment_v1_3/design_config.json").read_text())
    expected_parameters = {entry["id"]: entry["expected_trainable_parameters"] for entry in contract["architectures"]}
    pairs_fixture = write_pairs(data_root / "first_repeat/select.jsonl.gz", out / "select_pairs_slice.jsonl.gz")
    general_slice = list(head(data_root / "select/general.jsonl.gz", SMOKE_SEQUENCES))

    cells = {}
    candidates = {}
    for cell in CELLS:
        architecture, loss_id = cell.rsplit("_", 1)
        deterministic(0)
        model = Transformer(architecture).to(device)
        assert parameter_count(model) == expected_parameters[architecture], cell
        assert len(model.blocks) == ARCHITECTURES[architecture].blocks
        assert model.embedding.weight.data_ptr() != model.unembedding.weight.data_ptr()
        checks = dict(parameters=parameter_count(model), blocks=len(model.blocks), width=model.architecture.width)
        checks.update(check_masking_and_rope(model, sequences, device))
        checks["loss_weights"] = check_loss_weights(sequences, device, loss_id)
        checks["gradient_accumulation_max_abs_error"] = check_gradient_accumulation(model, sequences, device, loss_id)
        optimizer = lm_optimizer(model)
        state = Progress()
        training = [
            lm_update(model, optimizer, sequences[offset : offset + 64], state, loss_id=loss_id, microbatch=16)
            for offset in (0, 64)
        ]
        assert state.update == 2 and state.next_data_cursor == 128
        checks["training"] = training
        checks["resume"] = check_resume(
            model, optimizer, state, sequences, out, cell, loss_id, inputs, device, architecture
        )
        general = evaluate_records(model, general_slice, microbatch=16)
        pairs = evaluate_first_repeat(model, pairs_fixture, microbatch=16)
        assert pairs["pair_count"] == SMOKE_CELLS
        assert len(pairs["first"]["cell"]["cells"]) == SMOKE_CELLS
        assert set(pairs["first"]["answer"]["cells"]) <= {"0", "1"}
        checks["select_metric_path"] = dict(
            general=without_rows(general),
            first_macro_answer_ce=pairs["first"]["cell"]["macro_answer_ce"],
            first_macro_accuracy=pairs["first"]["cell"]["macro_accuracy"],
            repeat_macro_answer_ce=pairs["repeat"]["cell"]["macro_answer_ce"],
            paired_gap=without_rows(pairs["paired_gap"]),
            pair_cells=len(pairs["first"]["cell"]["cells"]),
        )
        candidates[cell] = dict(
            validated=True,
            select_general=dict(answer_ce=general["answer_ce"], accuracy=general["accuracy"]),
            select_pairs=without_rows(pairs),
        )
        cells[cell] = checks

    # Selection and gate arithmetic run on untrained debug numbers only, to exercise the code.
    decision = select_pilot(candidates, design)
    reference = candidates["base4_uniform"]["select_pairs"]
    gate_debug = gate(
        candidates["base4_uniform"]["select_general"],
        {
            name: dict(accuracy=candidates["base4_uniform"]["select_general"]["accuracy"])
            for name in ("other_variable", "repeated_update", "first_read_after_set")
        },
        reference,
        design["gate"],
    )
    assert set(decision) == {"decision", "winner", "eligibility"}
    assert set(gate_debug) == {"passed", "checks"}
    assert gate_debug["checks"]["coverage"], "42-cell coverage must be structurally satisfied"
    assert not gate_debug["passed"], "Untrained debug metrics must not satisfy the gate"

    anchor = check_anchor_equivalence(sequences)
    info = environment(out)
    result = dict(
        phase="v1.3-smoke",
        status="passed",
        scope=device,
        debug_only=True,
        reuse_in_experiment=False,
        input_hashes=inputs,
        environment=info,
        cells=cells,
        anchor_v1_2_equivalence=anchor,
        selection_code_path=dict(decision=decision["decision"], winner=decision["winner"], debug_only=True),
        gate_code_path=dict(passed=gate_debug["passed"], checks=gate_debug["checks"], debug_only=True),
        smoke_select_pairs_one_per_cell=SMOKE_CELLS,
        smoke_select_sequences=SMOKE_SEQUENCES,
        elapsed_seconds=time.time() - start,
        peak_memory_bytes=torch.cuda.max_memory_allocated()
        if device == "cuda"
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
    )
    (out / "smoke.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            dict(status="passed", scope=device, cells=list(cells), output=str(out), elapsed_seconds=result["elapsed_seconds"]),
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()
    run(args.root, args.output, args.device)
