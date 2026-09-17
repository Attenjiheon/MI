"""Configurable v1.3 decoder-only transformer with named intervention points."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Architecture:
    id: str
    blocks: int
    width: int
    heads: int
    mlp_width: int

    @property
    def head_width(self):
        if self.width % self.heads:
            raise ValueError("Residual width must be divisible by attention heads")
        return self.width // self.heads


ARCHITECTURES = {
    "base4": Architecture("base4", 4, 128, 4, 512),
    "wide4": Architecture("wide4", 4, 184, 4, 736),
    "deep8": Architecture("deep8", 8, 128, 4, 512),
}


def rope(x):
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE head width must be even")
    angles = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)[:, None] * (
        10000 ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    )
    c, s = angles.cos(), angles.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, architecture: Architecture):
        super().__init__()
        d = architecture.width
        self.architecture = architecture
        self.ln_attn = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d)
        self.attn_out = nn.Linear(d, d)
        self.ln_mlp = nn.LayerNorm(d, eps=1e-5)
        self.mlp_in = nn.Linear(d, architecture.mlp_width)
        self.mlp_out = nn.Linear(architecture.mlp_width, d)

    def forward(self, residual, mask, tap, layer):
        batch_size, length, width = residual.shape
        heads = self.architecture.heads
        head_width = self.architecture.head_width
        q, k, v = (
            self.qkv(self.ln_attn(residual))
            .reshape(batch_size, length, 3, heads, head_width)
            .permute(2, 0, 3, 1, 4)
        )
        scores = rope(q) @ rope(k).transpose(-1, -2) / math.sqrt(head_width)
        allowed = torch.ones(length, length, device=residual.device, dtype=torch.bool).tril()[None, None]
        allowed = allowed & mask[:, None, None, :]
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        attention = self.attn_out((weights @ v).transpose(1, 2).reshape(batch_size, length, width))
        mid = tap(f"blocks.{layer}.resid_mid", residual + attention)
        mlp_input = tap(f"blocks.{layer}.mlp_in", self.ln_mlp(mid))
        mlp_output = tap(
            f"blocks.{layer}.mlp_out",
            self.mlp_out(F.gelu(self.mlp_in(mlp_input), approximate="none")),
        )
        return tap(f"blocks.{layer}.resid_post", mid + mlp_output)


class Transformer(nn.Module):
    def __init__(self, architecture="base4"):
        super().__init__()
        self.architecture = ARCHITECTURES[architecture] if isinstance(architecture, str) else architecture
        d = self.architecture.width
        self.embedding = nn.Embedding(15, d)
        self.blocks = nn.ModuleList([Block(self.architecture) for _ in range(self.architecture.blocks)])
        self.final_ln = nn.LayerNorm(d, eps=1e-5)
        self.unembedding = nn.Linear(d, 15, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        multiplier = 1 / math.sqrt(2 * self.architecture.blocks)
        with torch.no_grad():
            for block in self.blocks:
                block.attn_out.weight.mul_(multiplier)
                block.mlp_out.weight.mul_(multiplier)
        self.interventions = {}
        self.capture = None

    def forward(self, token_ids, padding_mask):
        if token_ids.ndim != 2 or token_ids.shape != padding_mask.shape:
            raise ValueError("Expected equal [batch,time] token and mask shapes")
        if token_ids.shape[1] > 768 or not padding_mask[:, 0].all():
            raise ValueError("Context exceeds 768 or empty/left-padded sequence")

        def tap(name, value):
            if name in self.interventions:
                value = self.interventions[name](value)
            if self.capture is not None:
                self.capture[name] = value
            return value

        residual = self.embedding(token_ids)
        for index, block in enumerate(self.blocks):
            residual = block(residual, padding_mask, tap, index)
        return self.unembedding(self.final_ln(residual))


def batch(sequences, device="cpu", shift=True):
    if not sequences or any(len(sequence) < 2 for sequence in sequences):
        raise ValueError("Empty/short sequence")
    length = max(map(len, sequences)) - int(shift)
    ids = torch.zeros(len(sequences), length, dtype=torch.long, device=device)
    targets = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        count = len(sequence) - int(shift)
        ids[row, :count] = torch.as_tensor(sequence[:count], device=device)
        mask[row, :count] = True
        if shift:
            targets[row, :count] = torch.as_tensor(sequence[1:], device=device)
    return ids, mask, targets


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
