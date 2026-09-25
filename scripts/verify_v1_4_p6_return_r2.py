"""P6 audit r2: independently implemented forward with the frozen numerical recipe.

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
from interp_v1_4.runtime import sha, deterministic, environment


def normalize_values(raw, stats, device):
    # Production uses a float32 tensor divisor, not a Python scalar.
    return ((raw.to(device) - torch.tensor(stats['mean'], dtype=torch.float32, device=device))
            / torch.tensor(stats['scale'], dtype=torch.float32, device=device))


def recalculate(state_dict, raw, stats, k, device):
    """Do not import Dictionary/normalize/mse: independently assemble the recipe.

    F.linear is part of the numerical recipe: splitting GEMM and bias addition
    changes rounding before the discontinuous TopK operation. Legacy numbers
    are diagnostics only and never change acceptance or checkpoint selection.
    """
    m={name:t.to(device) for name,t in state_dict.items()}
    val=normalize_values(raw,stats,device)
    legacy=(raw.to(device)-torch.tensor(stats['mean'],dtype=torch.float32,device=device))/stats['scale']
    total=old_total=0.;changed_rows=0;max_activation_difference=0.;min_boundary_gap=None
    with torch.no_grad():
        for x,old_x in zip(val.split(512),legacy.split(512)):
            a=torch.relu(torch.nn.functional.linear(x,m['encoder'],m['encoder_bias']))
            old_a=torch.relu(old_x@m['encoder'].T+m['encoder_bias'])
            order=torch.argsort(a,descending=True,stable=True,dim=1)
            idx=order[:,:k]
            old_idx=torch.argsort(old_a,descending=True,stable=True,dim=1)[:,:k]
            changed_rows+=int((idx.sort(dim=1).values!=old_idx.sort(dim=1).values).any(dim=1).sum())
            max_activation_difference=max(max_activation_difference,float((a-old_a).abs().max()))
            gap=a.gather(1,order[:,k-1:k])-a.gather(1,order[:,k:k+1])
            g=float(gap.min());min_boundary_gap=g if min_boundary_gap is None else min(min_boundary_gap,g)
            z=torch.zeros_like(a).scatter(1,idx,a.gather(1,idx))
            old_z=torch.zeros_like(old_a).scatter(1,old_idx,old_a.gather(1,old_idx))
            y=torch.nn.functional.linear(z,m['decoder'],m['decoder_bias'])
            old_y=old_z@m['decoder'].T+m['decoder_bias']
            total+=float((y-x).double().square().sum())
            old_total+=float((old_y-old_x).double().square().sum())
    return dict(recomputed_mse=total/val.numel(),legacy_mse=old_total/val.numel(),
                topk_candidate_rows_changed=changed_rows,
                max_activation_difference=max_activation_difference,
                minimum_topk_boundary_gap=min_boundary_gap)


def audit(root,output,device='cpu'):
    root,output=Path(root),Path(output);config=verify(root)
    audit_dir=output/'audits'/f'r2_{time.time_ns()}'
    audit_dir.mkdir(parents=True,exist_ok=False)
    env=environment(audit_dir/'environment');write(audit_dir/'environment.json',env)
    audit_identity=dict(auditor_sha256=sha(Path(__file__)),production_config_sha256=sha(root/CONTRACT),environment_id=env['environment_id'],tolerance=dict(atol=1e-7,rtol=1e-6))
    write(audit_dir/'identity.json',audit_identity)
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
        raw_val=load_tensor(output/'inputs'/f'{key}_val.pt')
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
                numeric=recalculate(m,raw_val,s,run['k'],device)
                score=numeric['recomputed_mse'];saved=state['curve'][-1]['val_mse']
                passed=bool(np.isclose(score,saved,atol=1e-7,rtol=1e-6))
                detail=dict(run=run['name'],update=step,saved_mse=saved,passed=passed,
                            checkpoint_sha256=sha(p),absolute_error=abs(score-saved),**numeric)
                write(audit_dir/'checkpoints'/run['name']/f'{step:05d}.json',detail)
                if not passed:
                    raise AssertionError(dict(message='r2 MSE mismatch; unchanged tolerance',evidence=str(audit_dir),**detail))
                scores.append(detail)
        best=min(scores,key=lambda r:(r['saved_mse'],r['update']))
        assert result['best_update']==best['update'] and result['best_val_mse']==best['saved_mse']
        assert result['best_checkpoint']==f"update_{best['update']:05d}.pt" and result['last_checkpoint']=='update_05000.pt'
        assert len(state['curve'])==5000 and read(folder/'curve.json')==state['curve']
        reports.append(dict(run=run['name'],best_update=best['update'],scores=scores,result_sha256=sha(folder/'result.json')))
        write(audit_dir/'runs'/f'{run["name"]}.json',reports[-1])
        print('audited',run['name'],flush=True)
    target=audit_dir/'completion.json'
    write(target,dict(audit_revision='r2',audit_identity=audit_identity,status='passed_numeric_return_review_pending',p6_complete=False,config_sha256=ch,input_sha256=ih,runs=reports,updates=120000,draws=61440000,elapsed_seconds=time.time()-start))
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();print(audit(a.root,a.output,a.device))
