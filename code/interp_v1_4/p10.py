"""P10 Update block-3 cache and dictionary training; never marks P10 complete."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from corpus.replay import replay
from .model import Transformer, batch
from .p5 import read, write, session, save_unit, valid_unit, HOOKS
from .p9 import commit_tensor, load_tensor, train_run as dictionary_train_run
from .runtime import sha, deterministic, environment

CONTRACT = 'experiment_v1_4/p10_update_r1/contract.json'
LAYER = 3
QUOTAS = dict(train=50000, val=10000, test=20000)
LABELS = dict(operator=5, dst=4, src=4, dst_before=2, src_before=2,
              input_disagreement=2, dst_after=2, operator_truth=12)
OPS = ('SET', 'NOT', 'AND', 'OR', 'XOR')


def verify(root):
    root = Path(root); c = read(root / CONTRACT)
    if c['schema'] != 'p10-update-v1.4-r1' or c['layer'] != LAYER:
        raise ValueError('Wrong P10 contract')
    for name, digest in c['files'].items():
        if sha(root/name) != digest: raise ValueError('Changed input: '+name)
    return c


def join(records, positions, expected=None):
    """Independent token replay validates each selected non-initialization update."""
    by_id = {r['sequence_id']: r for r in records}
    keys = [(p['sequence_id'], p['token_index']) for p in positions]
    if len(by_id) != len(records) or keys != sorted(set(keys)):
        raise ValueError('Duplicate/unsorted sequence or position keys')
    if expected is not None and len(keys) != expected: raise ValueError('Update quota mismatch')
    parsed = {sid: replay(r['token_ids']) for sid, r in by_id.items()}
    rows = []
    for p in positions:
        r = by_id[p['sequence_id']]; eid = p['event_id']; t = p['token_index']
        e = r['update_events'][eid]; a = parsed[p['sequence_id']]['updates'][eid]
        start = e['start_token_index']; op = e['op']; dst = 'ABCD'.index(e['dst'])
        binary = op in ('AND', 'OR', 'XOR'); src = 'ABCD'.index(e['src_or_null']) if binary else None
        tokens = r['token_ids']; before = a['before']; after = a['after']
        operand = 9+src if binary else (13+e['literal_or_null'] if op=='SET' else 9+dst)
        truth = f'{before[dst]}{before[src]}' if binary else None
        if not (start >= 13 and e['update_id'] == eid and t == e['end_token_index'] == a['end']
                and start == a['start'] and tokens[start] == 3+OPS.index(op)
                and tokens[start+1] == 9+dst and tokens[t] == operand
                and e['block_id'] == a['block_id'] and e['state_before'] == before
                and e['state_after'] == after and e['dst_before'] == before[dst]
                and e['dst_after'] == after[dst] and e['input_truth_pattern_or_null'] == truth
                and e['src_before_or_null'] == (before[src] if binary else None)
                and (binary or e['src_or_null'] is None)):
            raise ValueError('Update metadata/token/replay mismatch')
        rows.append(dict(sequence_id=p['sequence_id'], token_index=t, event_id=eid,
            token_id=tokens[t], operator=OPS.index(op), dst=dst, src=src,
            dst_before=before[dst] if binary else None,
            src_before=before[src] if binary else None,
            input_disagreement=before[dst]^before[src] if binary else None,
            dst_after=after[dst], operator_truth=(OPS.index(op)-2)*4+2*before[dst]+before[src] if binary else None))
    return by_id, rows


def load_split(root, config, split):
    folder = Path(root)/config['data_root']/'interpretation'
    with gzip.open(folder/f'{split}.jsonl.gz', 'rt') as f: records = [json.loads(s) for s in f]
    return join(records, read(folder/f'{split}.update_positions.json'), QUOTAS[split])


def domain_mask(rows, label, domain, split):
    if domain not in ('all', 'variable', 'operator'): raise ValueError('Unknown domain')
    if domain != 'all' and label not in ('dst_before','src_before','input_disagreement','dst_after'):
        raise ValueError('Classification label has unseen classes in transfer')
    return np.array([r[label] is not None and
        (domain != 'variable' or (r['dst']==3 if split=='test' else r['dst']!=3)) and
        (domain != 'operator' or (r['operator']==4 if split=='test' else r['operator'] in (2,3)))
        for r in rows], dtype=bool)


def supports(rows, split):
    report = {}
    for label, classes in LABELS.items():
        for domain in ('all','variable','operator'):
            if domain!='all' and label not in ('dst_before','src_before','input_disagreement','dst_after'): continue
            chosen=[r for r,m in zip(rows,domain_mask(rows,label,domain,split)) if m]
            counts=[dict(class_id=c,positions=sum(r[label]==c for r in chosen),
                         sequences=len({r['sequence_id'] for r in chosen if r[label]==c})) for c in range(classes)]
            report[f'{label}/{domain}']=dict(classes=counts,positions=len(chosen),
                fit_support_sufficient=all(x['positions']>=32 and x['sequences']>=16 for x in counts) if split!='test' else None)
    return report


def preflight(root, output):
    root,output=Path(root),Path(output); c=verify(root); seen=set(); results={}
    for split in QUOTAS:
        records,rows=load_split(root,c,split)
        hashes={r['canonical_hash'] for r in records.values()}
        if seen & hashes: raise ValueError('Sequence overlap between interpretation splits')
        seen |= hashes
        results[split]=dict(sequences=len(records),positions=len(rows),support=supports(rows,split))
    report=dict(status='passed_input_only',config_sha256=sha(root/CONTRACT),splits=results,
                p10_complete=False,test_performance_observed=False)
    write(output/'preflight.json',report);return report


def collect(model, records, rows, device, layer=LAYER):
    lookup={r['sequence_id']:i for i,r in enumerate(records)}
    br=torch.tensor([lookup[r['sequence_id']] for r in rows],device=device)
    ti=torch.tensor([r['token_index'] for r in rows],device=device); values={}
    def take(key):
        def hook(x): values[key]=x[br,ti].detach().cpu().numpy().copy(); return x
        return hook
    model.interventions={f'blocks.{layer}.{hook}':take(short) for short,hook in HOOKS.items()}
    try:
        ids,mask,_=batch([r['token_ids'] for r in records],device)
        with torch.no_grad(): model(ids,mask)
    finally: model.interventions={}
    if set(values)!=set('hum') or any(x.shape!=(len(rows),256) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in values.values()):
        raise ValueError('Invalid activation capture')
    return values


def require_smoke(root, output, smoke_path, device):
    if device!='cuda' or not torch.cuda.is_available(): raise ValueError('Production requires CUDA')
    s=read(smoke_path); env=session(output,sha(Path(root)/CONTRACT),device)
    if s['status']!='passed' or s['device']!='cuda' or s['config_sha256']!=sha(Path(root)/CONTRACT) or s['environment_id']!=env['environment_id']:
        raise ValueError('Current-source, current-environment CUDA smoke required')
    return env


def extract(root, output, smoke_path, device='cuda', max_seconds=7200):
    root,output=Path(root),Path(output); c=verify(root); deterministic(0);torch.set_num_threads(2)
    env=require_smoke(root,output,smoke_path,device);start=time.monotonic()
    write(output/'contract.json',c)
    # This delivery intentionally extracts train/val only. Test is reserved for evaluation.
    for split in ('train','val'):
        records,rows=load_split(root,c,split);write(output/'labels'/f'{split}.json',rows)
        write(output/'labels'/f'{split}.support.json',supports(rows,split))
        byseq=defaultdict(list)
        for i,r in enumerate(rows):byseq[r['sequence_id']].append(i)
        selected=sorted(byseq)
        for spec in c['models']:
            seed=spec['lm_seed']; deterministic(seed);model=Transformer()
            payload=torch.load(root/spec['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(payload['model']);del payload
            model.eval().requires_grad_(False).to(device);mb=16
            for offset in range(0,len(selected),c['chunk_sequences']):
                if time.monotonic()-start>=max_seconds:
                    print('PAUSED extraction at committed chunk; rerun this cell',flush=True);return False
                ids=selected[offset:offset+c['chunk_sequences']];indices=[i for sid in ids for i in byseq[sid]]
                identity=dict(config_sha256=sha(root/CONTRACT),checkpoint_sha256=c['files'][spec['checkpoint']],
                    lm_seed=seed,split=split,position_type='UPDATE',layer=LAYER,sequence_offset=offset,
                    labels_sha256=sha(output/'labels'/f'{split}.json'))
                path=output/'cache'/f'seed{seed}'/split/f'{offset:06d}.npz'
                if valid_unit(path,identity):continue
                if path.exists():path.rename(path.with_name(path.name+f'.uncommitted_{time.time_ns()}'))
                tick=time.monotonic();chunks=defaultdict(list);cursor=0
                torch.cuda.reset_peak_memory_stats()
                while cursor<len(ids):
                    chosen=ids[cursor:cursor+mb];rr=[rows[i] for sid in chosen for i in byseq[sid]]
                    try:v=collect(model,[records[s] for s in chosen],rr,device)
                    except torch.cuda.OutOfMemoryError:
                        if mb==1:raise
                        mb//=2;torch.cuda.empty_cache();continue
                    for key,x in v.items():chunks[key].append(x)
                    cursor+=len(chosen)
                arrays={k:np.concatenate(v) for k,v in chunks.items()};arrays['row_indices']=np.array(indices,dtype=np.int64)
                save_unit(path,arrays,dict(identity=identity,positions=len(indices),environment_id=env['environment_id'],
                    elapsed_seconds=time.monotonic()-tick,microbatch=mb,peak_vram_bytes=torch.cuda.max_memory_allocated()),output,None)
                print('UPDATE cache',seed,split,offset+len(ids),'/',len(selected),flush=True)
            del model;torch.cuda.empty_cache()
    return True


def cache_arrays(output, config, seed, split):
    output=Path(output); n=QUOTAS[split];seen=np.zeros(n,bool)
    values={h:np.empty((n,256),np.float32) for h in 'hum'};receipt={}
    spec=next(m for m in config['models'] if m['lm_seed']==seed)
    ch=sha(output/'contract.json');lh=sha(output/'labels'/f'{split}.json')
    for p in sorted((output/'cache'/f'seed{seed}'/split).glob('*.npz')):
        if not p.with_suffix('.json').exists(): continue
        meta=read(p.with_suffix('.json'));ident=meta['identity']
        expected=dict(config_sha256=ch,checkpoint_sha256=config['files'][spec['checkpoint']],lm_seed=seed,
                      split=split,position_type='UPDATE',layer=LAYER,labels_sha256=lh)
        if any(ident[k]!=v for k,v in expected.items()) or sha(p)!=meta['sha256']:raise ValueError('Cache identity/hash mismatch')
        with np.load(p,allow_pickle=False) as z:
            idx=z['row_indices']
            if idx.ndim!=1 or idx.dtype.kind not in 'iu' or len(idx)!=meta['positions'] or (idx<0).any() or (idx>=n).any() or len(np.unique(idx))!=len(idx) or seen[idx].any():raise ValueError('Bad cache row indices')
            seen[idx]=True
            for h in 'hum':
                x=z[h]
                if x.shape!=(len(idx),256) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Invalid cache tensor')
                values[h][idx]=x
        receipt[str(p.relative_to(output))]=sha(p)
        receipt[str(p.with_suffix('.json').relative_to(output))]=sha(p.with_suffix('.json'))
    if not seen.all():raise ValueError('Cache incomplete; rerun extraction')
    return values,receipt


def prepare(root, output):
    root,output=Path(root),Path(output);c=verify(root);stats_files={};tensors={};cache={};labels={}
    if sha(output/'contract.json')!=sha(root/CONTRACT):raise ValueError('Wrong cache config')
    for split in ('train','val'):
        _,expected=load_split(root,c,split);p=output/'labels'/f'{split}.json'
        if read(p)!=expected:raise ValueError('Update label/key join changed')
        labels[str(p.relative_to(output))]=sha(p)
    for model in c['models']:
        seed=model['lm_seed']
        for split in ('train','val'):
            values,rr=cache_arrays(output,c,seed,split);cache.update(rr)
            for hook,x in values.items():
                key=f'seed{seed}_l{LAYER}_{hook}'
                if split=='train':
                    a=x.astype(np.float64);mu=a.mean(0);scale=np.sqrt(np.mean((a-mu)**2))
                    if not np.isfinite(scale) or scale<1e-8 or not np.isclose(scale**2,np.mean(a*a)-np.mean(mu*mu),atol=1e-12,rtol=1e-10):raise ValueError('Scalar statistics failed')
                    p=output/'statistics'/f'{key}.json';write(p,dict(mean=mu.astype(np.float32).tolist(),scale=float(np.float32(scale)),train_positions=len(x),accumulation='float64',stored='float32'))
                    stats_files[str(p.relative_to(output))]=sha(p)
                p=output/'inputs'/f'{key}_{split}.pt';t=torch.from_numpy(x)
                if p.exists():
                    if not torch.equal(load_tensor(p),t):raise ValueError('Prepared input changed')
                else:commit_tensor(p,t)
                tensors[str(p.relative_to(output))]=sha(p)
    manifest=dict(schema='p10-update-input-v1',config_sha256=sha(root/CONTRACT),statistics=stats_files,
                  tensors=tensors,cache_files=cache,labels=labels,statistics_count=9,test_used_for_selection=False)
    write(output/'input_manifest.json',manifest);return manifest


def checked_inputs(root,output):
    output=Path(output);m=read(output/'input_manifest.json')
    if m['config_sha256']!=sha(Path(root)/CONTRACT) or m['statistics_count']!=9:raise ValueError('Missing P10 input gate')
    for section in ('statistics','tensors','labels','cache_files'):
        for n,h in m[section].items():
            if sha(output/n)!=h:raise ValueError('Prepared input changed: '+n)
    return m


def train(root,output,smoke_path,device='cuda',max_seconds=7200):
    root,output=Path(root),Path(output);c=verify(root);checked_inputs(root,output)
    deterministic(0);torch.set_num_threads(2);env=require_smoke(root,output,smoke_path,device)
    identity=dict(config_sha256=sha(root/CONTRACT),input_sha256=sha(output/'input_manifest.json'))
    start=time.monotonic()
    for run in c['runs']:
        folder=output/'runs'/run['name'];key=f"seed{run['lm_seed']}_l{LAYER}";hooks='h' if run['tool']=='sae' else 'um'
        stats={h:read(output/'statistics'/f'{key}_{h}.json') for h in hooks}
        values={s:{h:load_tensor(output/'inputs'/f'{key}_{h}_{s}.pt') for h in hooks} for s in ('train','val')}
        if any(t.shape!=(QUOTAS[s],256) for s,d in values.items() for t in d.values()):raise ValueError('Input quota mismatch')
        args=(run,values['train'],values['val'],{h:dict(mu=torch.tensor(s['mean']),scale=torch.tensor(s['scale'])) for h,s in stats.items()},folder,identity,env['environment_id'],device)
        try:
            # At most one 250-update interval beyond the wall-clock bound. Same RNG/optimizer engine as P9.
            while True:
                complete=[p for p in folder.glob('update_*.pt') if p.with_suffix('.json').exists()]
                last=max([0]+[int(p.stem.split('_')[1]) for p in complete])
                if last<5000 and time.monotonic()-start>=max_seconds:
                    print('PAUSED training at committed checkpoint; rerun this cell',flush=True);return False
                result=dictionary_train_run(*args,stop_at=min(5000,((last//250)+1)*250))
                if result['status']!='paused':break
            # Generic engine's historical p9_complete=False is retained in raw result; P10 status lives here.
            write(folder/'p10_status.json',dict(p10_complete=False,status='training_return_audit_pending',result_sha256=sha(folder/'result.json')))
        except Exception as exc:
            write(output/'failures'/f'{run["name"]}_{time.time_ns()}.json',dict(run=run['name'],reason=repr(exc),identity=identity))
            raise
    write(output/'training_complete.json',dict(status='12_runs_training_audit_and_evaluation_pending',p10_complete=False,
        identity=identity,runs={r['name']:sha(output/'runs'/r['name']/'result.json') for r in c['runs']}))
    return True


def export(output,archive):
    output,archive=Path(output),Path(archive)
    # Cache and derived tensors stay on Drive; preserve all checkpoints and compact provenance in the return ZIP.
    paths=[p for p in output.rglob('*') if p.is_file() and not {'cache','inputs','exports'} & set(p.relative_to(output).parts)
           and not ('smoke' in p.relative_to(output).parts and p.suffix=='.pt') and p.suffix!='.tmp' and '.uncommitted_' not in p.name and p.resolve()!=archive.resolve()]
    files={str(p.relative_to(output)):sha(p) for p in paths}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(output)))
        z.writestr('export_manifest.json',json.dumps(dict(files=files,p10_complete=False,retained_on_drive=['cache','inputs','smoke']),indent=2))
    write(archive.with_suffix('.sha256.json'),dict(sha256=sha(archive),bytes=archive.stat().st_size))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','smoke','extract','prepare','train','export'])
    p.add_argument('--root',default='.');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--smoke');p.add_argument('--archive');p.add_argument('--max-seconds',type=float,default=7200);a=p.parse_args()
    if a.max_seconds<=0:raise ValueError('Positive wall-clock allowance required')
    if a.action=='preflight':preflight(a.root,a.output)
    elif a.action=='smoke':
        from .p10_smoke import smoke
        smoke(a.root,a.output,a.device)
    elif a.action=='extract':extract(a.root,a.output,a.smoke,a.device,a.max_seconds)
    elif a.action=='prepare':prepare(a.root,a.output)
    elif a.action=='train':train(a.root,a.output,a.smoke,a.device,a.max_seconds)
    else:export(a.output,a.archive)

if __name__=='__main__':main()
