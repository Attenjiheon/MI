"""Current-source four-layer debug checks, including the production resume engine."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from .p6 import verify, train_run, load_tensor, LAYERS, CONTRACT
from .p5 import read, write
from .runtime import deterministic, environment, sha, tensor_digest
from .model import Transformer, batch
from .smoke import check_masking_and_rope
from .p6_integration import check_interpretation


def resume_check(output, device, layer, k):
    output=Path(output);g=torch.Generator().manual_seed(712+layer)
    train=torch.randn(128,256,generator=g);val=torch.randn(37,256,generator=g)
    stats={'mu':torch.zeros(256),'scale':torch.tensor(1.)}
    run=dict(name=f'debug_l{layer}_k{k}',k=k,init_seed=123+layer,draw_seed=456+layer)
    identity={'debug_only':True};args=(run,train,val,stats)
    deterministic(0)
    full=train_run(*args,output/'full',identity,'debug',device,updates=4,interval=2)
    deterministic(0)
    train_run(*args,output/'resumed',identity,'debug',device,updates=4,interval=2,stop_at=2)
    train_run(*args,output/'resumed',identity,'debug',device,updates=4,interval=2)
    a=load_tensor(output/'full/update_00004.pt');b=load_tensor(output/'resumed/update_00004.pt')
    assert tensor_digest(a['model'])==tensor_digest(b['model'])
    assert a['curve']==b['curve'] and a['sampler_rng']==b['sampler_rng']
    for key in a['optimizer']['state']:
        for name,value in a['optimizer']['state'][key].items():
            other=b['optimizer']['state'][key][name]
            if torch.is_tensor(value):assert torch.equal(value,other)
            else:assert value==other
    assert full['best_update']==min(a['curve'][1::2],key=lambda r:(r['val_mse'],r['update']))['update']
    return dict(layer=layer,k=k,status='bitwise',draws=a['draws'],model_sha256=tensor_digest(a['model']))


def smoke(root, output, device):
    root,output=Path(root),Path(output);verify(root)
    output.mkdir(parents=True,exist_ok=False);start=time.time()
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    deterministic(0);torch.set_num_threads(2)
    records=read(root/'archive/legacy/experiment_v1_2/debug/sequences.json')
    model=Transformer('deepwide12').to(device)
    masking=check_masking_and_rope(model,[r['token_ids'] for r in records],device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for offset in (0,2):
        ids,mask,targets=batch([r['token_ids'] for r in records[offset:offset+2]],device)
        opt.zero_grad();logits=model(ids,mask)
        torch.nn.functional.cross_entropy(logits[mask],targets[mask]).backward();opt.step()
    checks={};resumes=[]
    for layer in LAYERS:
        checks[str(layer)]=check_interpretation(model,records,8,layer)
        for k in (4,16):resumes.append(resume_check(output/f'resume_l{layer}_k{k}',device,layer,k))
    env=environment(output/'environment');write(output/'environment/environment.json',env)
    result=dict(status='passed',debug_only=True,p6_complete=False,device=device,config_sha256=sha(root/CONTRACT),environment_id=env['environment_id'],masking=masking,checks=checks,resume=resumes,elapsed_seconds=time.time()-start)
    write(output/'smoke.json',result)
    return result
