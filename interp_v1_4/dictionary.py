"""Width-aware TopK SAE/Transcoder used after the v1.4 behavior gate."""
import torch
from torch import nn
from torch.nn import functional as F


def statistics(train):
    value = train.double()
    mean = value.mean(0)
    scale = ((value - mean).square().mean()).sqrt()
    if not torch.isfinite(scale) or scale < 1e-8:
        raise ValueError("Constant/nonfinite activation")
    return {"mu": mean.float(), "scale": scale.float()}


def normalize(value, stats):
    return (value - stats["mu"]) / stats["scale"]


def denormalize(value, stats):
    return stats["mu"] + stats["scale"] * value


class Dictionary(nn.Module):
    def __init__(self, kind, k, seed, width=128, latents=512):
        super().__init__()
        if kind not in ("sae", "transcoder") or k not in (4, 16):
            raise ValueError("Unsupported dictionary")
        self.kind, self.k, self.width, self.latents = kind, k, width, latents
        generator = torch.Generator().manual_seed(seed % (2**63 - 1))
        decoder = F.normalize(torch.randn(width, latents, generator=generator), dim=0)
        encoder = decoder.T.clone() if kind == "sae" else F.normalize(torch.randn(latents, width, generator=generator), dim=1)
        self.decoder = nn.Parameter(decoder)
        self.encoder = nn.Parameter(encoder)
        self.encoder_bias = nn.Parameter(torch.zeros(latents))
        self.decoder_bias = nn.Parameter(torch.zeros(width))

    def encode(self, value):
        activation = F.relu(F.linear(value, self.encoder, self.encoder_bias))
        indices = torch.argsort(activation, dim=-1, descending=True, stable=True)[..., : self.k]
        return torch.zeros_like(activation).scatter(-1, indices, activation.gather(-1, indices))

    def forward(self, value):
        latent = self.encode(value)
        return F.linear(latent, self.decoder, self.decoder_bias), latent


def parameter_count(width, latents=512):
    return 2 * width * latents + latents + width


def optimizer(dictionary):
    return torch.optim.Adam(dictionary.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)


def update(dictionary, optimizer_value, source, target):
    optimizer_value.zero_grad(set_to_none=True)
    prediction, latent = dictionary(source)
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite dictionary loss")
    loss.backward()
    with torch.no_grad():
        decoder, gradient = dictionary.decoder, dictionary.decoder.grad
        gradient.sub_(decoder * (decoder * gradient).sum(0, keepdim=True))
    torch.nn.utils.clip_grad_norm_(dictionary.parameters(), 1.0, error_if_nonfinite=True)
    optimizer_value.step()
    with torch.no_grad():
        norms = dictionary.decoder.norm(dim=0)
        if not torch.isfinite(norms).all() or (norms == 0).any():
            raise FloatingPointError("Invalid decoder norm")
        dictionary.decoder.div_(norms)
    return float(loss.detach())
