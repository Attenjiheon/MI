"""Audit completed P6 checkpoints and independently recompute validation MSE.

Use on Colab with the prepared input pools. This is numerical evidence for a
later local return review, never automatic phase completion.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p6 import CONTRACT,verify,load_tensor
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha, deterministic


def audit(root,output,device='cpu'):
    root,output=Path(root),Path(output);config=verify(root)
    inputs=read(output/'input_manifest.json');ch=sha(root/CONTRACT);ih=sha(output/'input_manifest.json')
    assert inputs['config_sha256']==ch and len(inputs['statistics'])==36
    for section in ('statistics','tensors'):
        for name,digest in inputs[section].items():assert sha(output/name)==digest
    deterministic(0);torch.set_num_threads(2);reports=[];start=time.time()
    for run in config['runs']:
        folder=output/'runs'/run['name'];result=read(folder/'result.json')
        assert result['updates']==5000 and result['draws']==2560000 and result['parameter_count']==262912
        assert result['identity']==dict(config_sha256=ch,input_sha256=ih) and result['run']==run
        key=f"seed{run['lm_seed']}_l{run['layer']}_h"
        s=read(output/'statistics'/f'{key}.json')
        val=load_tensor(output/'inputs'/f'{key}_val.pt').to(device)
        val=(val-torch.tensor(s['mean'],device=device))/s['scale']
        expected=set(range(250,5001,250))|{100}
        assert set(result['checkpoints'])=={f'update_{u:05d}.pt' for u in expected}
        scores=[];generator=np.random.Generator(np.random.PCG64(run['draw_seed']));cursor=0
        for step in sorted(expected):
            p=folder/f'update_{step:05d}.pt';assert sha(p)==result['checkpoints'][p.name]
            state=load_tensor(p)
            assert state['identity']==result['identity'] and state['run']==run
            assert state['update']==step and state['draws']==step*512 and state['unique_train_positions']==50000
            assert [r['update'] for r in state['curve']]==list(range(1,step+1))
            assert all(np.isfinite(r['train_mse']) for r in state['curve'])
            while cursor<step:generator.integers(0,50000,size=512,dtype=np.int64);cursor+=1
            assert state['sampler_rng']==generator.bit_generator.state
            assert torch.equal(state['stats']['mu'],torch.tensor(s['mean'])) and float(state['stats']['scale'])==s['scale']
            m=state['model'];assert all(torch.isfinite(t).all() for t in m.values())
            assert sum(t.numel() for t in m.values())==262912
            torch.testing.assert_close(m['decoder'].norm(dim=0),torch.ones(512),atol=2e-6,rtol=2e-6)
            assert len(state['optimizer']['state'])==4
            for v in state['optimizer']['state'].values():
                assert int(v['step'])==step
                assert torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
            if step%250==0:
                m={k:t.to(device) for k,t in m.items()};total=0.
                with torch.no_grad():
                    for x in val.split(512):
                        a=torch.relu(x@m['encoder'].T+m['encoder_bias'])
                        idx=torch.argsort(a,descending=True,stable=True,dim=1)[:,:run['k']]
                        z=torch.zeros_like(a).scatter(1,idx,a.gather(1,idx))
                        y=z@m['decoder'].T+m['decoder_bias']
                        total+=float((y-x).double().square().sum())
                score=total/val.numel();saved=state['curve'][-1]['val_mse']
                assert np.isclose(score,saved,atol=1e-7,rtol=1e-6),(run['name'],step,score,saved)
                scores.append(dict(update=step,saved_mse=saved,recomputed_mse=score))
        best=min(scores,key=lambda r:(r['saved_mse'],r['update']))
        assert result['best_update']==best['update'] and result['best_val_mse']==best['saved_mse']
        assert result['best_checkpoint']==f"update_{best['update']:05d}.pt" and result['last_checkpoint']=='update_05000.pt'
        assert len(state['curve'])==5000 and read(folder/'curve.json')==state['curve']
        reports.append(dict(run=run['name'],best_update=best['update'],scores=scores,result_sha256=sha(folder/'result.json')))
        print('audited',run['name'],flush=True)
    target=output/'audits'/f'numeric_{time.time_ns()}.json'
    write(target,dict(status='passed_numeric_return_review_pending',p6_complete=False,config_sha256=ch,input_sha256=ih,runs=reports,updates=120000,draws=61440000,elapsed_seconds=time.time()-start))
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();print(audit(a.root,a.output,a.device))
