"""Update-specific hook/causal-mask checks and same-engine optimizer/RNG resume."""
from pathlib import Path
import time
import numpy as np
import torch
from .p10 import verify, CONTRACT, LAYER, join, collect
from .p5 import read,write
from .p9_smoke import resume_check
from .model import Transformer,batch
from .runtime import deterministic,environment,sha


def update_hook_check(model,records,device):
    positions=sorted([dict(sequence_id=r['sequence_id'],event_id=e['update_id'],token_index=e['end_token_index'])
                      for r in records for e in r['update_events']],key=lambda p:(p['sequence_id'],p['token_index']))
    _,rows=join(records,positions);values=collect(model,records,rows,device)
    ids,mask,_=batch([r['token_ids'] for r in records],device)
    model.capture={}
    with torch.no_grad():original=model(ids,mask)
    cached=model.capture;model.capture=None
    lookup={r['sequence_id']:i for i,r in enumerate(records)}
    for i,row in enumerate(rows):
        b=lookup[row['sequence_id']];t=row['token_index']
        for h,hook in dict(h='resid_post',u='mlp_in',m='mlp_out').items():
            np.testing.assert_array_equal(values[h][i],cached[f'blocks.{LAYER}.{hook}'][b,t].cpu().numpy())
    torch.testing.assert_close(cached[f'blocks.{LAYER}.resid_post'],cached[f'blocks.{LAYER}.resid_mid']+cached[f'blocks.{LAYER}.mlp_out'],atol=0,rtol=0)
    torch.testing.assert_close(cached[f'blocks.{LAYER}.mlp_in'],model.blocks[LAYER].ln_mlp(cached[f'blocks.{LAYER}.resid_mid']),atol=0,rtol=0)
    r=rows[0];b=lookup[r['sequence_id']];t=r['token_index']
    # Changing every future token must not change the update activation or its logits.
    changed=ids.clone();changed[b,t+1:]=(changed[b,t+1:]+1)%15
    with torch.no_grad():later=model(changed,mask)
    torch.testing.assert_close(original[b,:t+1],later[b,:t+1],atol=0,rtol=0)
    for hook in ('resid_post','mlp_out'):
        saved=cached[f'blocks.{LAYER}.{hook}'][b,t].clone()
        def identity(x):
            y=x.clone();y[b,t]=saved;return y
        model.interventions={f'blocks.{LAYER}.{hook}':identity}
        try:
            with torch.no_grad():actual=model(ids,mask)
            torch.testing.assert_close(original,actual,atol=0,rtol=0)
        finally:model.interventions={}
    return dict(positions=len(rows),hooks='h/u/m independently indexed; h=mid+m; u=LN(mid)',future_token_invariance='bitwise',identity_patches='bitwise',initialization_excluded=True)


def smoke(root,output,device):
    root,output=Path(root),Path(output);verify(root);output.mkdir(parents=True,exist_ok=False)
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    deterministic(0);torch.set_num_threads(2);start=time.monotonic()
    records=read(root/'archive/legacy/experiment_v1_2/debug/sequences.json')[:2]
    model=Transformer().eval().requires_grad_(False).to(device)
    checks=update_hook_check(model,records,device);del model
    resumes=[resume_check(output/f'{tool}_k{k}',device,LAYER,k,tool) for tool in ('sae','transcoder') for k in (4,16)]
    env=environment(output/'environment');write(output/'environment/environment.json',env)
    report=dict(status='passed',debug_only=True,p10_complete=False,config_sha256=sha(root/CONTRACT),device=device,
                environment_id=env['environment_id'],update_checks=checks,resume=resumes,elapsed_seconds=time.monotonic()-start)
    write(output/'smoke.json',report);return report
