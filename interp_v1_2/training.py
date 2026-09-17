"""Token weighted LM updates and cursor state."""
from dataclasses import dataclass, asdict
import torch
from .model import batch, ce_sum


@dataclass
class Progress:
    update: int = 0
    prediction_tokens: int = 0
    next_data_cursor: int = 0
    next_validation_boundary: int = 100000
    best_validation: float = float('inf')

    def payload(self): return asdict(self)


def lm_optimizer(model):
    return torch.optim.AdamW([
        {'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.01},
        {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0}],
        lr=3e-4,betas=(.9,.95),eps=1e-8,amsgrad=False)


def lm_update(model, optimizer, sequences, state, microbatch=16):
    n=sum(len(s)-1 for s in sequences)
    lr=3e-4*min(1,(state.prediction_tokens+n)/50000)
    for group in optimizer.param_groups: group['lr']=lr
    optimizer.zero_grad(set_to_none=True); model.train()
    loss_total=0.
    for i in range(0,len(sequences),microbatch):
        ids,mask,target=batch(sequences[i:i+microbatch],next(model.parameters()).device)
        loss=ce_sum(model(ids,mask),target)
        (loss/n).backward(); loss_total+=loss.detach().item()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    state.update+=1; state.prediction_tokens+=n; state.next_data_cursor+=len(sequences)
    due=state.prediction_tokens>=state.next_validation_boundary
    while state.prediction_tokens>=state.next_validation_boundary: state.next_validation_boundary+=100000
    return dict(loss=loss_total/n,tokens=n,lr=lr,gradient_norm=float(norm),validation_due=due)
