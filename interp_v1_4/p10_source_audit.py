"""P10 source reproduction: full GPU activation/latent replay, CPU refit and CI replay."""
from pathlib import Path
from collections import defaultdict
import argparse,json,time,zipfile
import numpy as np
import torch
from . import p10,p10_evaluation as ev,p5_probe as probe
from .model import Transformer
from .p5 import read,write,session
from .runtime import sha,deterministic
CONTRACT='experiment_v1_4/p10_audit_r1/contract.json'


def compare(a,b,atol=2e-6,rtol=2e-5):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Different fields')
        for k in a:compare(a[k],b[k],atol,rtol)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('Different lengths')
        for x,y in zip(a,b):compare(x,y,atol,rtol)
    elif a is None or isinstance(a,(bool,str,int,np.integer)):
        if a!=b:raise ValueError('Different identity/status')
    elif not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a,b,atol=atol,rtol=rtol):raise ValueError(f'Numeric mismatch {a} != {b}')


def verify(root):
    root=Path(root);a=read(root/CONTRACT)
    if a['schema']!='p10-source-audit-v1.4-r1':raise ValueError('Wrong audit contract')
    for n,h in a['files'].items():
        if sha(root/n)!=h:raise ValueError('Changed audit input: '+n)
    c=ev.verify(root)
    if sha(root/ev.CONTRACT)!=a['evaluation_contract_sha256']:raise ValueError('Wrong original config')
    return a,c


def source_gate(root,source):
    root,source=Path(root),Path(source);a,c=verify(root)
    manifest=read(root/a['return_manifest'])
    for i,(n,h) in enumerate(manifest['files'].items()):
        if sha(source/n)!=h:raise ValueError('Returned evidence changed: '+n)
        if i and i%1000==0:print('returned files verified',i,flush=True)
    ev.features_gate(root,source)
    old=read(root/p10.CONTRACT)
    for split in ev.SPLITS:
        _,rows=p10.load_split(root,old,split)
        if rows!=read(source/'labels'/f'{split}.json'):raise ValueError('Update keys changed')
    return a,c


def committed(p,identity):
    if not p.exists():return False
    r=read(p)
    if r['identity']!=identity or r['status']!='passed':raise ValueError('Stale/failed audit receipt')
    return True


def identity(root):
    a=read(Path(root)/CONTRACT)
    return dict(audit_config_sha256=sha(Path(root)/CONTRACT),evaluation_config_sha256=a['evaluation_contract_sha256'],return_manifest_sha256=sha(Path(root)/a['return_manifest']))


def independent_encode(state,raw,tool,k,device):
    ih='h' if tool=='sae' else 'u';s=state['stats'][ih];m={n:t.to(device) for n,t in state['model'].items()};ys=[];zs=[]
    with torch.no_grad():
        for i in range(0,len(raw),512):
            x=(torch.tensor(np.array(raw[i:i+512]),device=device)-s['mu'].to(device))/s['scale'].to(device)
            activation=torch.relu(torch.nn.functional.linear(x,m['encoder'],m['encoder_bias']))
            ids=torch.argsort(activation,descending=True,stable=True,dim=1)[:,:k]
            z=torch.zeros_like(activation).scatter(1,ids,activation.gather(1,ids))
            y=torch.nn.functional.linear(z,m['decoder'],m['decoder_bias']);ys.append(y.cpu().numpy());zs.append(z.cpu().numpy())
    return np.concatenate(ys),np.concatenate(zs)


def independent_fidelity(target,prediction,z,train_mean):
    x=np.asarray(target,np.float64);p=np.asarray(prediction,np.float64);err=x-p
    sse=float(np.sum(err*err));nmse_den=float(np.sum((x-train_mean)**2));r2den=float(np.sum((x-x.mean(0))**2));evden=float(np.var(x,axis=0).sum())
    live=z>0;l0=live.sum(1);rates=live.mean(0)
    return dict(positions=len(x),mse=sse/x.size,nmse=sse/nmse_den if nmse_den else None,r2=1-sse/r2den if r2den else None,
        ev=1-float(np.var(err,axis=0).sum())/evden if evden else None,
        l0=dict(mean=float(np.mean(l0)),median=float(np.median(l0)),quantiles=np.percentile(l0,[0,5,25,75,95,100]).tolist()),
        activation_rates=rates.tolist(),inactive_count=int(sum(rates==0)),inactive_fraction=float(np.mean(rates==0)))


def gpu(root,source,out,max_seconds=7200):
    root,source,out=map(Path,(root,source,out));a,c=source_gate(root,source);ident=identity(root)
    if not torch.cuda.is_available():raise ValueError('GPU audit requires CUDA')
    deterministic(0);torch.set_num_threads(2);env=session(out,ident['audit_config_sha256'],'cuda');start=time.monotonic();old=read(root/p10.CONTRACT)
    # Recompute every raw activation, not a selected test-performance subset.
    expected=[]
    for spec in old['models']:
        lm=spec['lm_seed']
        for kind in ('trained','init'):
            cp=spec['checkpoint'] if kind=='trained' else spec['init_checkpoint'];model=Transformer()
            model.load_state_dict(torch.load(root/cp,map_location='cpu',weights_only=False)['model']);model.eval().requires_grad_(False).cuda()
            for split in ev.SPLITS:
                records,rows=p10.load_split(root,old,split);byseq=defaultdict(list)
                for i,r in enumerate(rows):byseq[r['sequence_id']].append(i)
                ids=sorted(byseq);original={h:ev.load_array(ev.raw_path(source,lm,kind,split,h)) for h in 'hum'}
                for h,x in original.items():
                    if x.shape!=(p10.QUOTAS[split],256):raise ValueError('Wrong source quota')
                for offset in range(0,len(ids),128):
                    path=out/'chunks'/f'seed{lm}_{kind}_{split}_{offset:06d}.json';expected.append(path)
                    if committed(path,ident):continue
                    if time.monotonic()-start>max_seconds:print('PAUSED GPU audit; rerun same cell',flush=True);return False
                    chosen=ids[offset:offset+128];errors={h:0. for h in 'hum'};count=0;mb=16;cursor=0;tick=time.monotonic()
                    while cursor<len(chosen):
                        ss=chosen[cursor:cursor+mb];idx=[i for sid in ss for i in byseq[sid]]
                        try:values=p10.collect(model,[records[s] for s in ss],[rows[i] for i in idx],'cuda')
                        except torch.cuda.OutOfMemoryError:
                            if mb==1:raise
                            mb//=2;torch.cuda.empty_cache();continue
                        for h,x in values.items():
                            target=np.asarray(original[h][idx]);np.testing.assert_allclose(x,target,atol=a['activation_tolerance']['atol'],rtol=a['activation_tolerance']['rtol'])
                            errors[h]=max(errors[h],float(np.max(np.abs(x-target))))
                        count+=len(idx);cursor+=len(ss)
                    write(path,dict(status='passed',identity=ident,lm_seed=lm,kind=kind,split=split,sequence_offset=offset,positions=count,max_absolute_errors=errors,microbatch=mb,environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-tick))
                    print('replayed',path.stem,flush=True)
            del model;torch.cuda.empty_cache()
    for e in c['selected']:
        r=e['run'];name=r['name'];path=out/'dictionaries'/f'{name}.json'
        if committed(path,ident):continue
        if time.monotonic()-start>max_seconds:print('PAUSED GPU audit; rerun same cell',flush=True);return False
        tick=time.monotonic();state=torch.load(root/e['checkpoint'],map_location='cpu',weights_only=False);ih,th=('h','h') if r['tool']=='sae' else ('u','m')
        for hook in set((ih,th)):
            raw=np.asarray(ev.load_array(ev.raw_path(source,r['lm_seed'],'trained','train',hook)),np.float64)
            mu=raw.mean(0);scale=np.sqrt(np.mean((raw-mu)**2));s=state['stats'][hook]
            np.testing.assert_allclose(mu.astype(np.float32),s['mu'].numpy(),atol=2e-7,rtol=2e-6)
            np.testing.assert_allclose(np.float32(scale),s['scale'].numpy(),atol=1e-7,rtol=1e-6)
        mu=state['stats'][th]['mu'].numpy();scale=float(state['stats'][th]['scale'])
        train=np.asarray(ev.load_array(ev.raw_path(source,r['lm_seed'],'trained','train',th)),np.float64);train_mean=((train-mu)/scale).mean(0);del train
        saved=read(source/'fidelity'/f'{name}.json');results={}
        for split in ev.SPLITS:
            raw=ev.load_array(ev.raw_path(source,r['lm_seed'],'trained',split,ih));pred,z=independent_encode(state,raw,r['tool'],r['k'],'cuda')
            original=ev.load_array(source/'latents'/name/f'{split}.npy');np.testing.assert_allclose(z,original,**a['metric_tolerance'])
            target=(np.asarray(ev.load_array(ev.raw_path(source,r['lm_seed'],'trained',split,th)),np.float64)-mu)/scale
            f=independent_fidelity(target,pred,z,train_mean);compare(f,saved['fidelity'][split],**a['metric_tolerance']);results[split]=dict(positions=len(z),maximum_latent_error=float(np.max(np.abs(z-original))))
        write(path,dict(status='passed',identity=ident,run=name,checkpoint_sha256=e['sha256'],splits=results,environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-tick))
        print('dictionary verified',name,flush=True)
    reports=[read(p) for p in expected]
    if sum(r['positions'] for r in reports)!=480000:raise ValueError('Wrong activation replay coverage')
    paths=expected+[out/'dictionaries'/f'{e["run"]["name"]}.json' for e in c['selected']]
    write(out/'gpu_complete.json',dict(status='passed',identity=ident,activation_position_visits=480000,unique_reserved_positions=80000,activation_chunks=len(expected),dictionary_runs=12,files={str(p.relative_to(out)):sha(p) for p in paths}))
    return True


def cpu(root,source,out,max_seconds=7200):
    root,source,out=map(Path,(root,source,out));a,c=source_gate(root,source);ident=identity(root)
    g=read(out/'gpu_complete.json')
    if g['identity']!=ident or g['status']!='passed':raise ValueError('Complete GPU source audit first')
    for n,h in g['files'].items():
        if sha(out/n)!=h:raise ValueError('Changed GPU audit receipt')
    env=session(out,ident['audit_config_sha256'],'cpu');torch.set_num_threads(2);start=time.monotonic()
    labels={s:read(source/'labels'/f'{s}.json') for s in ev.SPLITS};previous=None;values=None
    for i,t in enumerate(c['tasks']):
        path=out/'tasks'/f'{t["id"]}.json'
        if committed(path,ident):continue
        if time.monotonic()-start>max_seconds:print('PAUSED CPU source audit; rerun same cell',flush=True);return False
        tick=time.monotonic();key=(t['lm_seed'],t['representation'],t['hook'],t.get('run'))
        if key!=previous:values={s:ev.representation(source,c,t,s) for s in ev.SPLITS};previous=key
        # Fitting receives only train/validation arrays. Test never controls selection.
        fresh=ev.fit_task({s:values[s] for s in ('train','val')},{s:labels[s] for s in ('train','val')},t)
        original=read(source/'fits'/f'{t["id"]}.json')['fit']
        for field in ('status','support','ranking','candidate_count','selected'):compare(fresh[field],original[field],**a['metric_tolerance'])
        if fresh['status']!='passed':raise ValueError('Source refit failed')
        mask,y,groups=ev.target(labels['test'],t['label'],t['domain'],'test');saved=read(source/'semantic'/f'{t["id"]}.json');C=p10.LABELS[t['label']]
        for size,f in original['selected'].items():
            prob=probe.probabilities(f,values['test'][mask]);stored=ev.load_array(source/'predictions'/t['id']/f'{size}.npy');np.testing.assert_allclose(prob,stored,**a['metric_tolerance'])
            metrics=probe.metrics(y,prob,C,f['threshold'],groups,t['bootstrap_seed']);gaps={}
            for refid in t['full_references']:
                fit=read(source/'fits'/f'{refid}.json')['fit'];ref=fit['selected']['full'];rt=next(x for x in c['tasks'] if x['id']==refid)
                raw=ev.representation(source,c,rt,'test');refprob=probe.probabilities(ref,raw[mask])
                gaps[refid]=ev.paired_gap(y,prob,refprob,C,f['threshold'],ref['threshold'],groups,t['bootstrap_seed'])
            metrics['paired_full_probe_gaps']=gaps;compare(metrics,saved['reports'][size],**a['metric_tolerance'])
        write(path,dict(status='passed',identity=ident,task=t['id'],source_fit_sha256=sha(source/'fits'/f'{t["id"]}.json'),source_report_sha256=sha(source/'semantic'/f'{t["id"]}.json'),reports=len(original['selected']),selected=fresh['selected'],environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-tick))
        print('source task verified',i+1,'/',len(c['tasks']),t['id'],flush=True)
    paths=[out/'tasks'/f'{t["id"]}.json' for t in c['tasks']]
    if sum(read(p)['reports'] for p in paths)!=2016:raise ValueError('Wrong semantic coverage')
    write(out/'source_audit_complete.json',dict(status='source_reproduction_passed_return_review_pending',p10_complete=False,identity=ident,
        gpu_complete_sha256=sha(out/'gpu_complete.json'),refits=1248,semantic_reports=2016,tasks={str(p.relative_to(out)):sha(p) for p in paths}))
    return True


def export(out,archive):
    out,archive=Path(out),Path(archive);paths=[p for p in out.rglob('*') if p.is_file() and 'exports' not in p.relative_to(out).parts and p.suffix!='.tmp']
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(out)))
        z.writestr('return_manifest.json',json.dumps(dict(files={str(p.relative_to(out)):sha(p) for p in paths},p10_complete=False),indent=2))
    write(archive.with_suffix('.sha256.json'),dict(sha256=sha(archive),bytes=archive.stat().st_size))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gpu','cpu','export']);p.add_argument('--root',default='.');p.add_argument('--source');p.add_argument('--output',required=True);p.add_argument('--max-seconds',type=float,default=7200);p.add_argument('--archive');a=p.parse_args()
    if a.max_seconds<=0:raise ValueError('Positive time allowance required')
    if a.action=='gpu':gpu(a.root,a.source,a.output,a.max_seconds)
    elif a.action=='cpu':cpu(a.root,a.source,a.output,a.max_seconds)
    else:export(a.output,a.archive)

if __name__=='__main__':main()
