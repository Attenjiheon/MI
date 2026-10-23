"""Frozen P5 READ extraction, resumable CPU probes, and evidence export.

Production never performs behavior evaluation or dictionary training.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
import torch
from .model import Transformer, batch
from .runtime import deterministic, environment, sha, verified_copy
from . import p5_probe as probe

CONTRACT = 'experiment_v1_4/p5_r2/contract.json'
HOOKS = dict(h='resid_post', u='mlp_in', m='mlp_out')
LABELS = dict(current=2, previous=2, query=4, A=2, B=2, C=2, D=2, state=16)
QUOTAS = dict(train=50000, val=10000, test=20000)


def derived(purpose, key):
    return int.from_bytes(hashlib.sha256(f'20260909|experiment-spec-v1.0|{purpose}|{key}'.encode()).digest()[:8], 'big')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != payload: raise ValueError('Refusing overwrite: ' + str(path))
        return
    temp = path.with_suffix(path.suffix + '.tmp'); temp.write_text(payload); temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def verify(root):
    root = Path(root); config = read(root / CONTRACT)
    for name, digest in config['files'].items():
        if sha(root / name) != digest: raise ValueError('Input hash mismatch: ' + name)
    if config['schema'] != 'p5-read-v1.4-r2': raise ValueError('Unsupported P5 contract')
    return config


def load_split(root, config, split):
    folder = Path(root) / config['data_root'] / 'interpretation'
    with gzip.open(folder / f'{split}.jsonl.gz', 'rt') as f:
        records = [json.loads(line) for line in f]
    positions = read(folder / f'{split}.read_positions.json')
    return join(records, positions, QUOTAS[split])


def join(records, positions, expected=None):
    by_id = {r['sequence_id']: r for r in records}
    if len(by_id) != len(records): raise ValueError('Duplicate sequence')
    keys = [(p['sequence_id'], p['token_index']) for p in positions]
    if keys != sorted(set(keys)): raise ValueError('Position keys not unique/sorted')
    if expected is not None and len(keys) != expected: raise ValueError('READ quota mismatch')
    rows = []
    for p in positions:
        r = by_id[p['sequence_id']]; e = r['read_events'][p['event_id']]; t = p['token_index']
        q = 'ABCD'.index(e['query_var']); state = e['state_at_read']
        if (e['read_id'] != p['event_id'] or e['query_token_index'] != t or
            e['answer_token_index'] != t+1 or r['token_ids'][t] != 9+q or
            r['token_ids'][t-1] != 8 or r['token_ids'][t+1] != 13+e['answer'] or
            e['answer'] != state[q]):
            raise ValueError('READ metadata/token join mismatch')
        previous = e['previous_value_or_null']
        if previous is not None and previous not in (0, 1): raise ValueError('Invalid previous label')
        rows.append(dict(sequence_id=p['sequence_id'], token_index=t, event_id=p['event_id'],
            token_id=r['token_ids'][t], current=e['answer'], previous=previous, query=q,
            **dict(zip('ABCD', state)), state=sum(int(v)*2**i for i,v in enumerate(state))))
    return by_id, rows


def support_table(rows):
    groups = np.asarray([r['sequence_id'] for r in rows])
    out = {}
    for label, classes in LABELS.items():
        mask = np.asarray([r[label] is not None for r in rows]); y = np.asarray([r[label] if r[label] is not None else -1 for r in rows])
        out[label] = dict(missing=int((~mask).sum()), classes=probe.support(y[mask], groups[mask], classes))
    return out


def task_key(seed, checkpoint, layer, hook, kind='trained'):
    return f'v1.4|lm_seed={seed}|checkpoint={checkpoint}|layer={layer}|hook={hook}|READ|tool=p5-{kind}|k=NA|sparse_seed=0'


def collect(model, records, rows, device):
    """Capture only reserved READ rows, never full batch activation tensors."""
    lookup = {r['sequence_id']: i for i,r in enumerate(records)}
    batch_rows = torch.tensor([lookup[r['sequence_id']] for r in rows], device=device)
    tokens = torch.tensor([r['token_index'] for r in rows], device=device)
    captured = {}
    def take(name):
        def hook(value):
            captured[name] = value[batch_rows, tokens].detach().cpu().numpy().copy()
            return value
        return hook
    model.interventions = {f'blocks.{l}.{hook}': take(f'{l}_{short}') for l in range(len(model.blocks)) for short,hook in HOOKS.items()}
    try:
        ids, mask, _ = batch([r['token_ids'] for r in records], device)
        with torch.no_grad(): model(ids, mask)
    finally:
        model.interventions = {}
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in captured.values()):
        raise ValueError('Nonfinite/wrong dtype activation')
    return captured


def mirror_file(path, output, persistent):
    if persistent is not None: verified_copy(path, Path(persistent) / path.relative_to(output))


def save_unit(path, arrays, metadata, output, persistent):
    temp = path.with_suffix('.npz.tmp'); path.parent.mkdir(parents=True, exist_ok=True)
    with temp.open('wb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    temp.replace(path)
    record = path.with_suffix('.json')
    write(record, dict(**metadata, sha256=sha(path)))
    # Commit marker copied last: only a checksum-complete unit is resumable.
    mirror_file(path, output, persistent); mirror_file(record, output, persistent)


def valid_unit(path, identity):
    marker = path.with_suffix('.json')
    if not marker.exists(): return False
    item = read(marker)
    if item['identity'] != identity or sha(path) != item['sha256']:
        raise ValueError('Cache identity/hash mismatch: ' + str(path))
    return True


def session(output, config_hash, device):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    folder = output / 'sessions' / stamp
    info = environment(folder)
    info.update(config_sha256=config_hash, device=device, command=sys.argv, utc=stamp)
    write(folder / 'environment.json', info)
    return info


def extract(root, output, persistent=None, seeds=None):
    root, output = Path(root), Path(output); config = verify(root); ch = sha(root / CONTRACT)
    if not torch.cuda.is_available(): raise RuntimeError('Production extraction requires CUDA; use preflight for CPU debug')
    deterministic(0); output.mkdir(parents=True, exist_ok=True)
    env = session(output, ch, 'cuda'); torch.cuda.reset_peak_memory_stats()
    preflight_path = output / 'preflight' / env['environment_id'] / 'preflight.json'
    if not preflight_path.exists(): preflight(root, preflight_path.parent, 'cuda')
    if read(preflight_path)['contract_sha256'] != ch: raise ValueError('Stale GPU preflight')
    for p in (output/'sessions').rglob('*'):
        if p.is_file(): mirror_file(p, output, persistent)
    for p in preflight_path.parent.rglob('*'):
        if p.is_file(): mirror_file(p, output, persistent)
    write(output/'contract.json', config); mirror_file(output/'contract.json', output, persistent)
    for split in QUOTAS:
        records, rows = load_split(root, config, split)
        labels_path = output/'labels'/f'{split}.json'; write(labels_path, rows); mirror_file(labels_path, output, persistent)
        support_path = output/'labels'/f'{split}.support.json'; write(support_path, support_table(rows)); mirror_file(support_path, output, persistent)
        selected_ids = sorted({r['sequence_id'] for r in rows})
        row_by_sequence = defaultdict(list)
        for i, row in enumerate(rows): row_by_sequence[row['sequence_id']].append(i)
        for spec in config['models']:
            seed = spec['lm_seed']
            if seeds is not None and seed not in seeds: continue
            for kind in ('trained', 'init'):
                checkpoint = spec['checkpoint'] if kind == 'trained' else spec['init_checkpoint']
                digest = config['files'][checkpoint]
                deterministic(seed); model = Transformer()
                payload = torch.load(root/checkpoint, map_location='cpu', weights_only=False)
                model.load_state_dict(payload['model']); del payload
                model.eval().requires_grad_(False).to('cuda')
                mb = 16
                for offset in range(0, len(selected_ids), config['chunk_sequences']):
                    ids = selected_ids[offset:offset+config['chunk_sequences']]
                    indices = [i for sid in ids for i in row_by_sequence[sid]]
                    identity = dict(config_sha256=ch, lm_seed=seed, checkpoint_sha256=digest, split=split,
                                    position_type='READ', sequence_offset=offset, labels_sha256=sha(labels_path))
                    path = output/'cache'/f'seed{seed}_{kind}'/split/f'{offset:06d}.npz'
                    if valid_unit(path, identity):
                        mirror_file(path, output, persistent); mirror_file(path.with_suffix('.json'), output, persistent); continue
                    # A data file without its marker is an incomplete transaction and may be recomputed.
                    started = time.monotonic(); chunks = defaultdict(list); cursor = 0
                    while cursor < len(ids):
                        selected = ids[cursor:cursor+mb]
                        local_rows = [rows[i] for sid in selected for i in row_by_sequence[sid]]
                        try: values = collect(model, [records[s] for s in selected], local_rows, 'cuda')
                        except torch.cuda.OutOfMemoryError:
                            if mb == 1: raise
                            mb //= 2; torch.cuda.empty_cache(); continue
                        for name, values_ in values.items(): chunks[name].append(values_)
                        cursor += len(selected)
                    arrays = {k:np.concatenate(v) for k,v in chunks.items()}; arrays['row_indices'] = np.asarray(indices)
                    save_unit(path, arrays, dict(identity=identity, positions=len(indices),
                        environment_id=env['environment_id'], microbatch=mb, elapsed_seconds=time.monotonic()-started,
                        peak_vram_bytes=torch.cuda.max_memory_allocated(), hooks=HOOKS, layers=list(range(12)),
                        rng='no stochastic extraction; eval/dropout=0; deterministic seed='+str(seed)), output, persistent)
                    print(f'{seed} {kind} {split} sequences {offset+len(ids)}/{len(selected_ids)}', flush=True)
                del model; torch.cuda.empty_cache()
    audit_cache(output, ch, expected_seeds=seeds or [0,1,2])
    for p in (output/'audits').rglob('*.json'): mirror_file(p, output, persistent)


def cache_array(output, seed, kind, split, layer, hook, expected_hash=None):
    output = Path(output); rows = read(output/'labels'/f'{split}.json'); n = len(rows)
    result = np.empty((n,256),np.float32); seen = np.zeros(n, bool)
    for path in sorted((output/'cache'/f'seed{seed}_{kind}'/split).glob('*.npz')):
        marker = path.with_suffix('.json')
        if not marker.exists(): continue
        item = read(marker)
        if sha(path) != item['sha256']: raise ValueError('Corrupt cache: '+str(path))
        if expected_hash is not None and item['identity']['config_sha256'] != expected_hash: raise ValueError('Wrong config')
        with np.load(path) as z:
            idx = z['row_indices']; x = z[f'{layer}_{hook}']
            if x.shape != (len(idx),256) or x.dtype != np.float32 or not np.isfinite(x).all(): raise ValueError('Bad tensor')
            if len(np.unique(idx))!=len(idx) or (idx<0).any() or (idx>=n).any() or seen[idx].any(): raise ValueError('Repeated/bad cache rows')
            result[idx] = x; seen[idx]=True
    if not seen.all(): raise ValueError('Cache incomplete')
    return result


def audit_cache(output, config_hash, expected_seeds=(0,1,2)):
    output = Path(output); config=read(output/'contract.json'); report=[]
    for split, quota in QUOTAS.items():
        rows=read(output/'labels'/f'{split}.json')
        if len(rows)!=quota: raise ValueError('Wrong quota')
        for seed in expected_seeds:
            spec=next(m for m in config['models'] if m['lm_seed']==seed)
            for kind in ('trained','init'):
                count=0; seen=np.zeros(quota,bool)
                folder=output/'cache'/f'seed{seed}_{kind}'/split
                for path in sorted(folder.glob('*.npz')):
                    if not path.with_suffix('.json').exists(): continue
                    item=read(path.with_suffix('.json')); identity=item['identity']
                    digest=config['files'][spec['checkpoint'] if kind=='trained' else spec['init_checkpoint']]
                    if any(identity[k]!=v for k,v in dict(config_sha256=config_hash,lm_seed=seed,checkpoint_sha256=digest,
                        split=split,position_type='READ',labels_sha256=sha(output/'labels'/f'{split}.json')).items()): raise ValueError('Cache identity mismatch')
                    if sha(path)!=item['sha256']: raise ValueError('Corrupt cache')
                    with np.load(path) as z:
                        idx=z['row_indices']
                        if len(np.unique(idx))!=len(idx) or (idx<0).any() or (idx>=quota).any() or seen[idx].any(): raise ValueError('Duplicate/out-of-bounds rows')
                        seen[idx]=True; count+=len(idx)
                        for layer in range(12):
                            for hook in HOOKS:
                                a=z[f'{layer}_{hook}']
                                if a.shape!=(len(idx),256) or a.dtype!=np.float32 or not np.isfinite(a).all(): raise ValueError('Bad tensor')
                if count!=quota or not seen.all(): raise ValueError('Incomplete cache')
                report.append(dict(split=split,seed=seed,kind=kind,positions=count))
    write(output/'audits'/('cache_'+''.join(map(str,expected_seeds))+'.json'),dict(status='passed_cache_only',config_sha256=config_hash,items=report,p5_complete=False))
    return report


def domains(rows,label,domain,split):
    mask=np.asarray([r[label] is not None for r in rows])
    if domain=='transfer': mask &= np.asarray([r['query']==3 if split=='test' else r['query']!=3 for r in rows])
    return mask


def analyze_one(arrays, labels, label, domain, prefixes, candidates, shuffle_seed, bootstrap_seed):
    masks={s:domains(labels[s],label,domain,s) for s in QUOTAS}
    ys={s:np.asarray([r[label] if r[label] is not None else -1 for r in labels[s]],int)[masks[s]] for s in QUOTAS}
    groups={s:np.asarray([r['sequence_id'] for r in labels[s]])[masks[s]] for s in QUOTAS}
    x={s:arrays[s][masks[s]] for s in QUOTAS}
    if shuffle_seed is not None: ys['train']=np.random.Generator(np.random.PCG64(shuffle_seed)).permutation(ys['train'])
    fit=probe.fit(x['train'],ys['train'],groups['train'],x['val'],ys['val'],groups['val'],LABELS[label],candidates,prefixes)
    fit['test_support']=probe.support(ys['test'],groups['test'],LABELS[label])
    if fit['status']!='passed': return fit
    fit['evaluation']={}
    for size,model in fit['selected'].items():
        prob=probe.probabilities(model,x['test'])
        score=probe.metrics(ys['test'],prob,LABELS[label],model['threshold'],groups['test'],bootstrap_seed)
        different=np.asarray([r['previous'] is not None and r['current']!=r['previous'] for r in labels['test']])[masks['test']]
        if label in ('current','previous'):
            score['current_ne_previous']=probe.metrics(ys['test'][different],prob[different],LABELS[label],model['threshold'],groups['test'][different],bootstrap_seed)
        fit['evaluation'][size]=score
    return fit


def probes(root, output, persistent=None, seeds=None):
    root,output=Path(root),Path(output);config=verify(root);ch=sha(root/CONTRACT)
    if read(output/'contract.json')!=config: raise ValueError('Different cache contract')
    seeds=seeds or [0,1,2]; audit_cache(output,ch,seeds); env=session(output,ch,'cpu')
    labels={s:read(output/'labels'/f'{s}.json') for s in QUOTAS}
    # Verify row order against the original reserved metadata, independently of cache hashes.
    for s in QUOTAS:
        _,expected=load_split(root,config,s)
        if labels[s]!=expected: raise ValueError('Changed labels/order')
    for spec in config['models']:
        seed=spec['lm_seed']
        if seed not in seeds: continue
        for layer in range(12):
            for hook in HOOKS:
                for kind in ('trained','init'):
                    arrays={s:cache_array(output,seed,kind,s,layer,hook,ch) for s in QUOTAS}
                    checkpoint=config['files'][spec['checkpoint'] if kind=='trained' else spec['init_checkpoint']]
                    key=task_key(seed,checkpoint,layer,hook,kind)
                    variants=[('full',arrays,False,None,False)]
                    if kind=='trained': variants.append(('shuffled',arrays,False,None,True))
                    if kind=='trained' and layer==0 and hook in ('h','m'):
                        a=arrays['train'].astype(np.float64);mu=a.mean(0).astype(np.float32)
                        scale=np.float32(np.sqrt(np.mean((a-a.mean(0))**2))); del a
                        if scale<1e-8: raise ValueError('Constant dictionary activation')
                        stats=output/'preprocessing'/f'seed{seed}_{hook}.json'
                        write(stats,dict(mean=mu.tolist(),scale=float(scale),accumulation='float64',stored='float32',train_positions=50000,config_sha256=ch))
                        mirror_file(stats,output,persistent)
                        random_spec=config['random'][key]
                        rng=np.random.Generator(np.random.PCG64(random_spec['projection_seed']))
                        R=rng.normal(size=(256,512));R/=np.linalg.norm(R,axis=0)
                        randoms={s:((a.astype(np.float64)-mu)/scale)@R for s,a in arrays.items()}
                        directions=output/'directions'/f'seed{seed}_{hook}.json'
                        write(directions,dict(seed=random_spec['projection_seed'],matrix=R.tolist(),subsets=random_spec['subsets'],config_sha256=ch));mirror_file(directions,output,persistent)
                        for representation,values in [('coordinate',arrays),('random',randoms)]:
                            variants.extend([(representation,values,True,None,False),
                                (representation+'_128',values,True,random_spec['subsets'][representation],False)])
                    # TC input scalar normalization is separately fitted on all READ train rows.
                    if kind=='trained' and layer==0 and hook=='u':
                        a=arrays['train'].astype(np.float64);mu=a.mean(0);scale=float(np.sqrt(np.mean((a-mu)**2)))
                        if scale<1e-8: raise ValueError('Constant TC input')
                        stats=output/'preprocessing'/f'seed{seed}_u.json'
                        write(stats,dict(mean=mu.astype(np.float32).tolist(),scale=float(np.float32(scale)),accumulation='float64',stored='float32',train_positions=50000,config_sha256=ch));mirror_file(stats,output,persistent)
                    for variant,values,prefix,candidates,shuffled in variants:
                        for label in LABELS:
                            for domain in (('iid','transfer') if label in ('current','previous') else ('iid',)):
                                name=f'seed{seed}_{kind}_l{layer}_{hook}_{variant}_{label}_{domain}'
                                path=output/'probes'/f'{name}.json'
                                if path.exists():
                                    if read(path)['config_sha256']!=ch: raise ValueError('Stale probe result')
                                    mirror_file(path,output,persistent); continue
                                task=key+'|'+variant+'|'+label+'|'+domain
                                bootstrap_seed=derived('bootstrap',task); shuffle_seed=derived('shuffle_label',task) if shuffled else None
                                started=time.monotonic()
                                result=analyze_one(values,labels,label,domain,prefix,candidates,shuffle_seed,bootstrap_seed)
                                write(path,dict(config_sha256=ch,task_key=task,bootstrap_seed=bootstrap_seed,shuffle_seed=shuffle_seed,
                                    environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-started,result=result))
                                mirror_file(path,output,persistent); print(name,result['status'],flush=True)
                    del arrays, variants
        # Shared token/position control is repeated per seed with explicit run identity.
        values={}
        for split,rows in labels.items():
            pos=np.asarray([r['token_index']/767 for r in rows]);tok=np.asarray([r['token_id'] for r in rows])
            values[split]=np.c_[np.eye(15)[tok],pos,pos**2]
        for label in LABELS:
            for domain in (('iid','transfer') if label in ('current','previous') else ('iid',)):
                task=f'v1.4|lm_seed={seed}|token_position|READ|{label}|{domain}'
                path=output/'probes'/f'seed{seed}_token_position_{label}_{domain}.json'
                if path.exists():
                    if read(path)['config_sha256']!=ch: raise ValueError('Stale control')
                else:
                    started=time.monotonic();bs=derived('bootstrap',task)
                    result=analyze_one(values,labels,label,domain,False,None,None,bs)
                    write(path,dict(config_sha256=ch,task_key=task,bootstrap_seed=bs,environment_id=env['environment_id'],
                        elapsed_seconds=time.monotonic()-started,result=result))
                mirror_file(path,output,persistent)
    for p in (output/'sessions').rglob('*'):
        if p.is_file():mirror_file(p,output,persistent)


def preflight(root, output, device='cpu'):
    root,output=Path(root),Path(output);config=verify(root);env=session(output,sha(root/CONTRACT),device)
    deterministic(501);torch.set_num_threads(2)
    records=read(root/'archive/legacy/experiment_v1_2/debug/sequences.json')[:4]
    positions=sorted([dict(sequence_id=r['sequence_id'],event_id=e['read_id'],token_index=e['query_token_index']) for r in records for e in r['read_events']],key=lambda p:(p['sequence_id'],p['token_index']))
    by_id,rows=join(records,positions); model=Transformer().to(device).eval().requires_grad_(False)
    records=[by_id[s] for s in sorted(by_id)]
    a=collect(model,records,rows,device)
    singles=defaultdict(list)
    for r in records:
        rr=[row for row in rows if row['sequence_id']==r['sequence_id']]
        for key,value in collect(model,[r],rr,device).items(): singles[key].append(value)
    for key in a: np.testing.assert_allclose(a[key],np.concatenate(singles[key]),atol=1e-5,rtol=1e-4)
    # Changing the answer and all later input must not affect the READ activation.
    r=json.loads(json.dumps(records[0]));row=next(row for row in rows if row['sequence_id']==r['sequence_id']); t=row['token_index']
    before=collect(model,[r],[row],device)
    r['token_ids'][t+1:]=[13]*len(r['token_ids'][t+1:]);after=collect(model,[r],[row],device)
    for k in before:np.testing.assert_allclose(before[k],after[k],atol=1e-5,rtol=1e-4)
    path=output/'debug_chunk.npz';identity=dict(debug=True,config_sha256=sha(root/CONTRACT))
    save_unit(path,a,dict(identity=identity),output,None)
    assert valid_unit(path,identity)
    with np.load(path) as z:
        for k,v in a.items():np.testing.assert_array_equal(z[k],v)
    write(output/'preflight.json',dict(status='passed',scope='debug_only',device=device,environment_id=env['environment_id'],
        contract_sha256=sha(root/CONTRACT),positions=len(rows),layers=12,hooks=HOOKS,
        checks=['metadata join','all-layer batched versus single','pre-answer causality','atomic cache roundtrip/resume checksum'],p5_complete=False))
    print(read(output/'preflight.json'))


def export(output, archive):
    output,archive=Path(output),Path(archive)
    files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name in files:z.write(output/name,name)
        z.writestr('return_manifest.json',json.dumps(dict(schema='p5-return-v1.4-r2',files=files)))
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['extract','probes','preflight','audit','export'])
    parser.add_argument('--root',default='.');parser.add_argument('--output',required=True);parser.add_argument('--persistent')
    parser.add_argument('--seeds',type=int,nargs='+',choices=[0,1,2]);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--archive')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='extract':extract(args.root,args.output,args.persistent,args.seeds)
    elif args.action=='probes':probes(args.root,args.output,args.persistent,args.seeds)
    elif args.action=='preflight':preflight(args.root,args.output,args.device)
    elif args.action=='audit':audit_cache(args.output,sha(Path(args.root)/CONTRACT),args.seeds or [0,1,2])
    else:export(args.output,args.archive)

if __name__=='__main__':main()
