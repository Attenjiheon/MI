"""Explicit float32 attention and named intervention points; no metadata input."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def rope(x):
    angles = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)[:, None] * (
        10000 ** (-torch.arange(0, 32, 2, device=x.device, dtype=x.dtype) / 32))
    c, s = angles.cos(), angles.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*c-b*s, a*s+b*c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_attn = nn.LayerNorm(128, eps=1e-5)
        self.qkv = nn.Linear(128, 384)
        self.attn_out = nn.Linear(128, 128)
        self.ln_mlp = nn.LayerNorm(128, eps=1e-5)
        self.mlp_in = nn.Linear(128, 512)
        self.mlp_out = nn.Linear(512, 128)

    def forward(self, r, mask, tap, layer):
        b, t, _ = r.shape
        q, k, v = self.qkv(self.ln_attn(r)).reshape(b,t,3,4,32).permute(2,0,3,1,4)
        scores = rope(q) @ rope(k).transpose(-1,-2) / math.sqrt(32)
        allowed = torch.ones(t,t,device=r.device,dtype=torch.bool).tril()[None,None] & mask[:,None,None,:]
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        a = self.attn_out((weights @ v).transpose(1,2).reshape(b,t,128))
        mid = tap(f'blocks.{layer}.resid_mid', r+a)
        u = tap(f'blocks.{layer}.mlp_in', self.ln_mlp(mid))
        m = tap(f'blocks.{layer}.mlp_out', self.mlp_out(F.gelu(self.mlp_in(u), approximate='none')))
        return tap(f'blocks.{layer}.resid_post', mid+m)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(15,128)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.final_ln = nn.LayerNorm(128, eps=1e-5)
        self.unembedding = nn.Linear(128,15,bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
        with torch.no_grad():
            for b in self.blocks:
                b.attn_out.weight.mul_(.5)
                b.mlp_out.weight.mul_(.5)
        self.interventions = {}
        self.capture = None

    def forward(self, token_ids, padding_mask):
        if token_ids.ndim != 2 or token_ids.shape != padding_mask.shape:
            raise ValueError('Expected equal [batch,time] token and mask shapes')
        if token_ids.shape[1] > 768 or not padding_mask[:,0].all():
            raise ValueError('Context exceeds 768 or empty/left-padded sequence')
        def tap(name, value):
            if name in self.interventions: value = self.interventions[name](value)
            if self.capture is not None: self.capture[name] = value
            return value
        r = self.embedding(token_ids)
        for i, block in enumerate(self.blocks): r = block(r,padding_mask,tap,i)
        return self.unembedding(self.final_ln(r))


def batch(sequences, device='cpu', shift=True):
    if not sequences or any(len(s)<2 for s in sequences): raise ValueError('Empty/short sequence')
    t = max(map(len,sequences)) - int(shift)
    ids = torch.zeros(len(sequences),t,dtype=torch.long,device=device)
    target = torch.zeros_like(ids)
    mask = torch.zeros_like(ids,dtype=torch.bool)
    for i, s in enumerate(sequences):
        n = len(s)-int(shift)
        ids[i,:n] = torch.as_tensor(s[:n],device=device)
        mask[i,:n] = True
        if shift: target[i,:n] = torch.as_tensor(s[1:],device=device)
    return ids, mask, target


def ce_sum(logits, targets):
    return F.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=0,reduction='sum')
