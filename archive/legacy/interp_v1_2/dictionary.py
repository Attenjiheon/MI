"""Unsupervised TopK SAE/TC; scales computed on training positions only."""
import torch
from torch import nn
from torch.nn import functional as F


def statistics(train):
    x=train.double(); mu=x.mean(0); scale=((x-mu).square().mean()).sqrt()
    if not torch.isfinite(scale) or scale<1e-8: raise ValueError('Constant/nonfinite activation')
    return dict(mu=mu.float(),scale=scale.float())


def normalize(x, stats): return (x-stats['mu'])/stats['scale']
def denormalize(x, stats): return stats['mu']+stats['scale']*x


class Dictionary(nn.Module):
    def __init__(self, kind, k, seed):
        super().__init__()
        if kind not in ('sae','transcoder') or k not in (4,16): raise ValueError('Unsupported dictionary')
        self.kind, self.k=kind,k
        generator=torch.Generator().manual_seed(seed % (2**63-1))
        d=F.normalize(torch.randn(128,512,generator=generator),dim=0)
        e=d.T.clone() if kind=='sae' else F.normalize(torch.randn(512,128,generator=generator),dim=1)
        self.decoder=nn.Parameter(d); self.encoder=nn.Parameter(e)
        self.encoder_bias=nn.Parameter(torch.zeros(512)); self.decoder_bias=nn.Parameter(torch.zeros(128))

    def encode(self,x):
        a=F.relu(F.linear(x,self.encoder,self.encoder_bias))
        ids=torch.argsort(a,dim=-1,descending=True,stable=True)[...,:self.k]
        return torch.zeros_like(a).scatter(-1,ids,a.gather(-1,ids))

    def forward(self,x):
        z=self.encode(x)
        return F.linear(z,self.decoder,self.decoder_bias),z


def optimizer(dictionary):
    return torch.optim.Adam(dictionary.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0)


def update(dictionary, opt, x, y):
    opt.zero_grad(set_to_none=True)
    prediction,z=dictionary(x); loss=(prediction-y).square().mean()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite dictionary loss')
    loss.backward()
    with torch.no_grad():
        d=dictionary.decoder; g=d.grad
        g.sub_(d*(d*g).sum(0,keepdim=True))
    torch.nn.utils.clip_grad_norm_(dictionary.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    with torch.no_grad():
        norms=d.norm(dim=0)
        if not torch.isfinite(norms).all() or (norms==0).any(): raise FloatingPointError('Invalid decoder norm')
        d.div_(norms)
        if not all(torch.isfinite(p).all() for p in dictionary.parameters()): raise FloatingPointError('Invalid parameters')
    return float(loss.detach())


def fidelity(target,prediction,z,train_mean):
    error=(prediction-target).double(); v=target.double()
    numerator=error.square().sum()
    nmse_d=(v-train_mean.double()).square().sum()
    r2_d=(v-v.mean(0)).square().sum()
    ev_d=v.var(0,unbiased=False).sum()
    ratio=lambda a,b: float(a/b) if b>0 else None
    nmse=ratio(numerator,nmse_d); r=ratio(numerator,r2_d)
    ev=ratio(error.var(0,unbiased=False).sum(),ev_d)
    l0=(z>0).sum(-1).float()
    return dict(mse=float(error.square().mean()),nmse=nmse,r2=None if r is None else 1-r,
                explained_variance=None if ev is None else 1-ev,l0_mean=float(l0.mean()),
                l0_quantiles=torch.quantile(l0,torch.tensor([0.,.25,.5,.75,1.],device=l0.device)).tolist(),
                activation_rate=(z>0).float().mean(0).tolist())
