"""Token-only uniform/read4 objectives and exact resumable LM updates."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from .model import batch


@dataclass
class Progress:
    update: int = 0
    prediction_tokens: int = 0
    next_data_cursor: int = 0

    def payload(self):
        return asdict(self)


def lm_optimizer(model):
    return torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.01},
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=False,
    )


def loss_weights(input_ids, targets, mask, loss_id):
    if loss_id not in ("uniform", "read4"):
        raise ValueError(f"Unknown loss {loss_id}")
    weights = mask.to(torch.float32)
    answer = torch.zeros_like(mask)
    answer[:, 1:] = (input_ids[:, :-1] == 8) & (input_ids[:, 1:] >= 9) & (input_ids[:, 1:] <= 12)
    answer &= mask
    if answer.any() and not torch.all((targets[answer] == 13) | (targets[answer] == 14)):
        raise ValueError("READ <VAR> did not predict B0/B1")
    if loss_id == "read4":
        weights = weights + 3.0 * answer.to(weights.dtype)
    return weights, answer


def weighted_ce_sum(logits, targets, weights):
    ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
    return (ce * weights.flatten()).sum()


def uniform_ce_sum(logits, targets):
    # Preserve the exact v1.2 reduction kernel for the base4_uniform anchor.
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0, reduction="sum")


def learning_rate(prediction_tokens_after_update):
    if prediction_tokens_after_update <= 50_000:
        return 3e-4 * prediction_tokens_after_update / 50_000
    if prediction_tokens_after_update <= 28_800_000:
        return 3e-4
    fraction = min(1.0, (prediction_tokens_after_update - 28_800_000) / 3_200_000)
    return 3e-4 + fraction * (3e-5 - 3e-4)


def lm_update(model, optimizer, sequences, state, *, loss_id="uniform", microbatch=16):
    device = next(model.parameters()).device
    prepared = []
    prediction_tokens = 0
    weight_total = 0.0
    answer_targets = 0
    for offset in range(0, len(sequences), microbatch):
        ids, mask, targets = batch(sequences[offset : offset + microbatch], device)
        weights, answer = loss_weights(ids, targets, mask, loss_id)
        prepared.append((ids, mask, targets, weights))
        prediction_tokens += int(mask.sum())
        weight_total += float(weights.sum())
        answer_targets += int(answer.sum())
    lr = learning_rate(state.prediction_tokens + prediction_tokens)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    model.train()
    weighted_loss = 0.0
    for ids, mask, targets, weights in prepared:
        logits = model(ids, mask)
        loss = uniform_ce_sum(logits, targets) if loss_id == "uniform" else weighted_ce_sum(logits, targets, weights)
        (loss / weight_total).backward()
        weighted_loss += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    state.update += 1
    state.prediction_tokens += prediction_tokens
    state.next_data_cursor += len(sequences)
    return {
        "loss": weighted_loss / weight_total,
        "prediction_tokens": prediction_tokens,
        "loss_weight_sum": weight_total,
        "read_answer_targets": answer_targets,
        "lr": lr,
        "gradient_norm": float(norm),
    }
