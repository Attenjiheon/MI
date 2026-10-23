"""P6: verified four-layer inputs and immutable, resumable SAE training.

No test activations or state labels enter training/selection. P6 completion is
reserved for a separate returned-evidence audit, not this GPU runner.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from . import dictionary as sparse
from .p5 import read, write, derived
from .runtime import sha, deterministic, environment, rng_state, restore_rng, tensor_digest

CONTRACT = 'experiment_v1_4/p6_r1/contract.json'
RECEIPT = 'experiment_v1_4/results/p5_final_audit_20260925_01/p6_cache_manifest.json'
LAYERS = (0, 3, 7, 11)


def verify(root):
    root = Path(root); config = read(root / CONTRACT)
    if config['schema'] != 'p6-sae-v1.4-r1': raise ValueError('Wrong P6 contract')
    for name, digest in config['files'].items():
        if sha(root/name) != digest: raise ValueError('Changed input: '+name)
    return config


def commit_tensor(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise ValueError('Refusing overwrite: '+str(path))
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    temp.replace(path)
    write(path.with_suffix('.json'), {'sha256':sha(path)})


def load_tensor(path):
    path = Path(path)
    if sha(path) != read(path.with_suffix('.json'))['sha256']: raise ValueError('Corrupt tensor')
    return torch.load(path, map_location='cpu', weights_only=False)


def arrays(source, receipt, seed, split):
    """One streaming pass per seed/split, only the four required layers."""
    n = receipt['quotas'][split]; seen = np.zeros(n, bool)
    out = {f'{l}_{h}':np.empty((n,256),np.float32) for l in LAYERS for h in 'hum'}
    prefix = f'cache/seed{seed}_trained/{split}/'
    files = [f for f in receipt['cache_files'] if f['path'].startswith(prefix)]
    for item in files:
        path = source/item['path']
        if sha(path) != item['sha256']: raise ValueError('Cache receipt mismatch: '+str(path))
        with np.load(path, allow_pickle=False) as z:
            idx=z['row_indices']
            if idx.ndim!=1 or idx.dtype.kind not in 'iu' or len(idx)!=item['positions'] or (idx<0).any() or (idx>=n).any() or len(np.unique(idx))!=len(idx) or seen[idx].any():
                raise ValueError('Invalid READ row indices')
            seen[idx]=True
            for key in out:
                x=z[key]
                if x.shape!=(len(idx),256) or x.dtype!=np.float32 or not np.isfinite(x).all(): raise ValueError('Bad activation')
                out[key][idx]=x
    if not seen.all(): raise ValueError('Missing READ rows')
    return out


def prepare(root, source, output):
    root, source, output = map(Path, (root,source,output)); config=verify(root)
    receipt=read(root/RECEIPT)
    if sha(source/'contract.json') != receipt['config_sha256']: raise ValueError('P5 source contract mismatch')
    for split,digest in receipt['labels'].items():
        if sha(source/'labels'/f'{split}.json')!=digest: raise ValueError('READ keys mismatch')
        rows=read(source/'labels'/f'{split}.json')
        keys=[(r['sequence_id'],r['token_index']) for r in rows]
        if len(keys)!=receipt['quotas'][split] or keys!=sorted(set(keys)): raise ValueError('Invalid READ keys')
    # Recheck every original byte against the independent P5 audit receipt.
    for item in receipt['cache_files']:
        if sha(source/item['path'])!=item['sha256']: raise ValueError('Source cache hash mismatch')
    output.mkdir(parents=True,exist_ok=True)
    stats_files={}; tensor_files={}
    for seed in (0,1,2):
        for split in ('train','val'):
            values=arrays(source,receipt,seed,split)
            for layer in LAYERS:
                for hook in 'hum':
                    key=f'seed{seed}_l{layer}_{hook}'; x=values[f'{layer}_{hook}']
                    if split=='train':
                        a=x.astype(np.float64); mu=a.mean(0); scale=np.sqrt(np.mean((a-mu)**2))
                        # Independent moment identity detects accumulation/formula mistakes.
                        second=np.mean(a*a)-np.mean(mu*mu)
                        if not np.isclose(scale**2,second,atol=1e-12,rtol=1e-10) or not np.isfinite(scale) or scale<1e-8: raise ValueError('Scalar statistics failed')
                        s=dict(mean=mu.astype(np.float32).tolist(),scale=float(np.float32(scale)),train_positions=len(x),accumulation='float64',stored='float32')
                        if layer==0:
                            old=receipt['preprocessing'][f'seed{seed}_{hook}.json']
                            if sha(source/'preprocessing'/f'seed{seed}_{hook}.json')!=old['sha256']: raise ValueError('P5 preprocessing hash mismatch')
                            if s['mean']!=old['values']['mean'] or s['scale']!=old['values']['scale']: raise ValueError('Block 0 scalar changed')
                            s['reused_p5_sha256']=old['sha256']
                        file=output/'statistics'/f'{key}.json';write(file,s);stats_files[str(file.relative_to(output))]=sha(file)
                    if hook=='h':
                        file=output/'inputs'/f'{key}_{split}.pt'
                        t=torch.from_numpy(x.copy())
                        if file.exists():
                            if not torch.equal(load_tensor(file),t): raise ValueError('Prepared input changed')
                        else: commit_tensor(file,t)
                        tensor_files[str(file.relative_to(output))]=sha(file)
            del values
    manifest=dict(schema='p6-four-layer-input-v1',config_sha256=sha(root/CONTRACT),receipt_sha256=sha(root/RECEIPT),source_contract_sha256=receipt['config_sha256'],source_cache_files=receipt['cache_files'],labels=receipt['labels'],statistics=stats_files,tensors=tensor_files,layers=list(LAYERS),statistics_count=len(stats_files),test_used_for_selection=False)
    write(output/'input_manifest.json',manifest)
    return manifest


def mse(model, values):
    total=0.
    with torch.no_grad():
        for part in values.split(512): total += float((model(part)[0]-part).double().square().sum())
    result=total/values.numel()
    if not np.isfinite(result): raise FloatingPointError('Nonfinite validation MSE')
    return result


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def train_run(run, train, val, stats, output, identity, environment_id, device,
              *, updates=5000, interval=250, stop_at=None):
    """The same engine serves debug resume tests and the gated production entry."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=sparse.Dictionary('sae',run['k'],run['init_seed'],width=256).to(device)
    opt=sparse.optimizer(model); gen=np.random.Generator(np.random.PCG64(run['draw_seed']))
    write(output/'initialization.json', dict(run=run,identity=identity,model_sha256=tensor_digest(model.state_dict()),sampler_rng=gen.bit_generator.state))
    start_update=0;curve=[];sessions=[];train_seconds=0.;validation_seconds=0.;benchmark=None
    checkpoints=sorted(output.glob('update_*.pt'))
    complete=[p for p in checkpoints if p.with_suffix('.json').exists()]
    # A payload without its commit receipt is uncommitted: preserve it separately.
    for p in set(checkpoints)-set(complete):
        p.rename(p.with_name(p.name+f'.uncommitted_{time.time_ns()}'))
    if complete:
        state=load_tensor(complete[-1])
        if state['identity']!=identity or state['run']!=run: raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng']);gen.bit_generator.state=state['sampler_rng']
        start_update=state['update'];curve=state['curve'];sessions=state['sessions']
        train_seconds=state['train_seconds'];validation_seconds=state['validation_seconds'];benchmark=state['benchmark']
        if state['draws']!=start_update*512 or (start_update%interval and start_update!=100): raise ValueError('Invalid resume boundary')
    sessions=copy.deepcopy(sessions)+[dict(environment_id=environment_id,resume_update=start_update,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))]
    stats={k:v.to(device) for k,v in stats.items()}
    x=sparse.normalize(train.to(device),stats);v=sparse.normalize(val.to(device),stats)
    if not torch.isfinite(x).all() or not torch.isfinite(v).all(): raise ValueError('Nonfinite normalized input')
    end=updates if stop_at is None else min(updates,stop_at)
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for step in range(start_update+1,end+1):
        sync(device);tick=time.perf_counter()
        indices=torch.from_numpy(gen.integers(0,len(x),size=512,dtype=np.int64)).to(device)
        loss=sparse.update(model,opt,x[indices],x[indices]);sync(device)
        train_seconds+=time.perf_counter()-tick
        curve.append(dict(update=step,train_mse=loss))
        if step==100:
            benchmark=dict(updates=100,seconds_per_update=train_seconds/100,peak_vram_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,included_in_budget=True)
        if step%interval==0:
            sync(device);tick=time.perf_counter();score=mse(model,v);sync(device)
            validation_seconds+=time.perf_counter()-tick;curve[-1]['val_mse']=score
        if step%interval==0 or step==100:
            payload=dict(identity=identity,run=run,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),sampler_rng=gen.bit_generator.state,update=step,draws=step*512,unique_train_positions=len(x),stats={k:t.cpu() for k,t in stats.items()},curve=curve,sessions=sessions,train_seconds=train_seconds,validation_seconds=validation_seconds,benchmark=benchmark,peak_vram_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0)
            commit_tensor(output/f'update_{step:05d}.pt',payload)
            print(run['name'],step,curve[-1],flush=True)
    if end==updates and max([start_update,end])==updates:
        state=load_tensor(output/f'update_{updates:05d}.pt')
        candidates=[r for r in state['curve'] if 'val_mse' in r]
        best=min(candidates,key=lambda r:(r['val_mse'],r['update']))
        result=dict(status='trained_pending_return_audit',p6_complete=False,identity=identity,run=run,updates=updates,draws=updates*512,best_update=best['update'],best_val_mse=best['val_mse'],best_checkpoint=f"update_{best['update']:05d}.pt",last_checkpoint=f'update_{updates:05d}.pt',checkpoints={p.name:sha(p) for p in sorted(output.glob('update_*.pt'))},parameter_count=sum(p.numel() for p in model.parameters()),train_seconds=state['train_seconds'],validation_seconds=state['validation_seconds'],benchmark=state['benchmark'])
        write(output/'result.json',result)
        write(output/'curve.json',state['curve'])
        return result
    return dict(status='paused',last_committed_update=max([0]+[int(p.stem.split('_')[1]) for p in output.glob('update_*.pt') if p.with_suffix('.json').exists()]))


def train(root, output, smoke_path, device, names=None):
    root,output=Path(root),Path(output);config=verify(root)
    if device!='cuda' or not torch.cuda.is_available(): raise ValueError('Production requires CUDA')
    manifest=read(output/'input_manifest.json')
    if manifest['config_sha256']!=sha(root/CONTRACT) or manifest['statistics_count']!=36: raise ValueError('P6 input gate missing')
    for section in ('statistics','tensors'):
        for name,digest in manifest[section].items():
            if sha(output/name)!=digest: raise ValueError('Changed prepared input')
    smoke=read(smoke_path)
    if smoke['config_sha256']!=sha(root/CONTRACT) or smoke['device']!='cuda' or smoke['status']!='passed': raise ValueError('Current GPU smoke required')
    deterministic(0);torch.set_num_threads(2)
    folder=output/'environments'/str(time.time_ns());env=environment(folder);write(folder/'environment.json',env)
    if smoke['environment_id']!=env['environment_id']: raise ValueError('Run GPU smoke in this environment first')
    identity=dict(config_sha256=sha(root/CONTRACT),input_sha256=sha(output/'input_manifest.json'))
    allowed={r['name'] for r in config['runs']}
    if names and not set(names)<=allowed: raise ValueError('Unknown run')
    for run in config['runs']:
        if names and run['name'] not in names: continue
        key=f"seed{run['lm_seed']}_l{run['layer']}_h"
        stats=read(output/'statistics'/f'{key}.json')
        try:
            train_run(run,load_tensor(output/'inputs'/f'{key}_train.pt'),load_tensor(output/'inputs'/f'{key}_val.pt'),dict(mu=torch.tensor(stats['mean']),scale=torch.tensor(stats['scale'])),output/'runs'/run['name'],identity,env['environment_id'],device)
        except Exception as exc:
            write(output/'failures'/f'{run["name"]}_{time.time_ns()}.json',dict(run=run['name'],reason=repr(exc),identity=identity,environment_id=env['environment_id'],resume='last checksum-committed complete update'))
            raise
    results=[output/'runs'/r['name']/'result.json' for r in config['runs']]
    if all(p.exists() for p in results):
        write(output/'training_complete.json',dict(status='24_runs_return_audit_pending',p6_complete=False,runs={str(p.relative_to(output)):sha(p) for p in results},identity=identity))


def export(output, archive):
    output=Path(output);archive=Path(archive)
    if archive.exists(): raise ValueError('Export already exists')
    # Large derived h pools can be reconstructed; preserve all checkpoints/statistics.
    paths=[p for p in output.rglob('*') if p.is_file() and 'inputs' not in p.relative_to(output).parts and p.suffix!='.tmp']
    files={str(p.relative_to(output)):sha(p) for p in paths}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(output)))
        z.writestr('export_manifest.json',json.dumps(dict(files=files,p6_complete=False),indent=2))
    write(archive.with_suffix('.sha256.json'),dict(sha256=sha(archive)))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train','export']);p.add_argument('--root',default='.');p.add_argument('--source');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke');p.add_argument('--runs',nargs='*');p.add_argument('--archive');a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.source,a.output)
    elif a.action=='train':train(a.root,a.output,a.smoke,a.device,a.runs)
    elif a.action=='smoke':
        from .p6_smoke import smoke
        smoke(a.root,a.output,a.device)
    else:export(a.output,a.archive)

if __name__=='__main__':main()
