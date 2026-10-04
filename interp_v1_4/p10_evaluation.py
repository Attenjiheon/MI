"""P10 Update evaluation: GPU cache/latent preparation, resumable CPU fits and reports."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import time,zipfile,json
import numpy as np
import torch
from . import p10, dictionary as sparse, p5_probe as probe
from .model import Transformer
from .p5 import read,write,save_unit,valid_unit,session
from .runtime import sha,seed,deterministic
from .p7_metrics import fidelity
CONTRACT='experiment_v1_4/p10_eval_r1/contract.json'
SPLITS=('train','val','test')


def verify(root):
    root=Path(root);c=read(root/CONTRACT)
    if c['schema']!='p10-update-evaluation-v1.4-r1':raise ValueError('Wrong evaluation contract')
    for n,h in c['files'].items():
        if sha(root/n)!=h:raise ValueError('Changed evaluation input: '+n)
    return c


def commit_array(p,a,identity):
    p=Path(p);marker=p.with_suffix('.json')
    if marker.exists():
        m=read(marker)
        if m['identity']!=identity or m['sha256']!=sha(p):raise ValueError('Changed array')
        return
    if p.exists():p.rename(p.with_name(p.name+f'.uncommitted_{time.time_ns()}'))
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,a,allow_pickle=False)
    tmp.replace(p);write(marker,dict(identity=identity,sha256=sha(p),shape=list(a.shape),dtype=str(a.dtype)))


def raw_path(out,lm,kind,split,hook):return Path(out)/'arrays'/f'seed{lm}_{kind}_{split}_{hook}.npy'


def load_array(p):
    if sha(p)!=read(Path(p).with_suffix('.json'))['sha256']:raise ValueError('Corrupt array')
    return np.load(p,mmap_mode='r',allow_pickle=False)


def extract(root,source,out,max_seconds=7200):
    root,source,out=map(Path,(root,source,out));c=verify(root);old=read(root/p10.CONTRACT);ch=sha(root/CONTRACT)
    deterministic(0);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise ValueError('GPU extraction requires CUDA')
    if sha(source/'input_manifest.json')!=c['training_input_sha256'] or sha(source/'contract.json')!=sha(root/p10.CONTRACT):raise ValueError('Wrong training source')
    p10.checked_inputs(root,source)
    env=session(out,ch,'cuda');tick=time.monotonic()
    from .p10_smoke import update_hook_check
    records=read(root/'archive/legacy/experiment_v1_2/debug/sequences.json')[:2]
    debug=Transformer().eval().requires_grad_(False).cuda()
    checks=update_hook_check(debug,records,'cuda');del debug;torch.cuda.empty_cache()
    write(out/'smoke'/f'{time.time_ns()}.json',dict(status='passed',environment_id=env['environment_id'],config_sha256=ch,checks=checks))
    write(out/'contract.json',c)
    for split in SPLITS:
        records,rows=p10.load_split(root,old,split);lp=out/'labels'/f'{split}.json';write(lp,rows)
        byseq=defaultdict(list)
        for i,r in enumerate(rows):byseq[r['sequence_id']].append(i)
        ids=sorted(byseq)
        for spec in old['models']:
            lm=spec['lm_seed']
            for kind in ('trained','init'):
                if kind=='trained' and split in ('train','val'):
                    values,_=p10.cache_arrays(source,old,lm,split)
                    for h,x in values.items():commit_array(raw_path(out,lm,kind,split,h),x,dict(config=ch,source_input=c['training_input_sha256'],seed=lm,kind=kind,split=split,hook=h))
                    del values;continue
                cp=spec['checkpoint'] if kind=='trained' else spec['init_checkpoint']
                deterministic(lm);model=Transformer();payload=torch.load(root/cp,map_location='cpu',weights_only=False)
                model.load_state_dict(payload['model']);del payload;model.eval().requires_grad_(False).cuda();mb=16
                folder=out/'cache'/f'seed{lm}_{kind}'/split
                for offset in range(0,len(ids),128):
                    chosen=ids[offset:offset+128];indices=[i for sid in chosen for i in byseq[sid]]
                    identity=dict(config_sha256=ch,checkpoint_sha256=c['files'][cp],lm_seed=lm,kind=kind,split=split,layer=3,position_type='UPDATE',labels_sha256=sha(lp),sequence_offset=offset)
                    p=folder/f'{offset:06d}.npz'
                    if valid_unit(p,identity):continue
                    if time.monotonic()-tick>max_seconds:print('PAUSED extraction; rerun',flush=True);return False
                    if p.exists():p.rename(p.with_name(p.name+f'.uncommitted_{time.time_ns()}'))
                    chunks=defaultdict(list);cursor=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
                    while cursor<len(chosen):
                        ss=chosen[cursor:cursor+mb];rr=[rows[i] for sid in ss for i in byseq[sid]]
                        try:v=p10.collect(model,[records[s] for s in ss],rr,'cuda')
                        except torch.cuda.OutOfMemoryError:
                            if mb==1:raise
                            mb//=2;torch.cuda.empty_cache();continue
                        for h,x in v.items():chunks[h].append(x)
                        cursor+=len(ss)
                    values={h:np.concatenate(x) for h,x in chunks.items()};values['row_indices']=np.array(indices,dtype=np.int64)
                    save_unit(p,values,dict(identity=identity,positions=len(indices),environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-started,microbatch=mb,peak_vram_bytes=torch.cuda.max_memory_allocated()),out,None)
                    print('cache',lm,kind,split,offset+len(chosen),'/',len(ids),flush=True)
                values={h:np.empty((len(rows),256),np.float32) for h in 'hum'};seen=np.zeros(len(rows),bool)
                for offset in range(0,len(ids),128):
                    p=folder/f'{offset:06d}.npz';meta=read(p.with_suffix('.json'))
                    if sha(p)!=meta['sha256']:raise ValueError('Corrupt cache')
                    with np.load(p,allow_pickle=False) as z:
                        idx=z['row_indices'];expected=[i for sid in ids[offset:offset+128] for i in byseq[sid]]
                        if not np.array_equal(idx,expected) or seen[idx].any():raise ValueError('Wrong cache join')
                        seen[idx]=True
                        for h in 'hum':
                            x=z[h]
                            if x.shape!=(len(idx),256) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Invalid activation')
                            values[h][idx]=x
                if not seen.all():raise ValueError('Cache incomplete')
                for h,x in values.items():commit_array(raw_path(out,lm,kind,split,h),x,dict(config=ch,checkpoint=c['files'][cp],seed=lm,kind=kind,split=split,hook=h))
                del model,values;torch.cuda.empty_cache()
    files={str(p.relative_to(out)):sha(p) for p in sorted((out/'arrays').glob('*'))}
    if len(files)!=108:raise ValueError('Expected 54 raw arrays and 54 receipts')
    write(out/'cache_complete.json',dict(config_sha256=ch,files=files,labels={s:sha(out/'labels'/f'{s}.json') for s in SPLITS},status='passed_cache_only'))
    return True


def encode(root,out,device='cuda'):
    root,out=Path(root),Path(out);c=verify(root);ch=sha(root/CONTRACT);deterministic(0);torch.set_num_threads(2)
    cc=read(out/'cache_complete.json')
    if cc['config_sha256']!=ch:raise ValueError('Wrong cache')
    env=session(out,ch,device)
    for n,h in cc['files'].items():
        if sha(out/n)!=h:raise ValueError('Changed cache array')
    for entry in c['selected']:
        run=entry['run'];lm=run['lm_seed'];name=run['name'];done=out/'fidelity'/f'{name}.json'
        if done.exists():
            existing=read(done)
            if existing['config_sha256']!=ch:raise ValueError('Stale fidelity')
            for split,h in existing['latent_sha256'].items():
                if sha(out/'latents'/name/f'{split}.npy')!=h:raise ValueError('Changed latent')
            continue
        st=torch.load(root/entry['checkpoint'],map_location='cpu',weights_only=False)
        if st['run']!=run or st['update']!=entry['selected_update']:raise ValueError('Changed selection')
        model=sparse.Dictionary(run['tool'],run['k'],0,width=256).to(device);model.load_state_dict(st['model']);model.eval().requires_grad_(False)
        stats={h:{k:t.to(device) for k,t in v.items()} for h,v in st['stats'].items()};ih,th=('h','h') if run['tool']=='sae' else ('u','m')
        train=np.asarray(load_array(raw_path(out,lm,'trained','train',th)),np.float64)
        mu=stats[th]['mu'].cpu().numpy();scale=float(stats[th]['scale']);train_mean=((train-mu)/scale).mean(0);del train
        fids={};latents={};started=time.monotonic()
        for split in SPLITS:
            raw=load_array(raw_path(out,lm,'trained',split,ih));target=load_array(raw_path(out,lm,'trained',split,th))
            preds=[];zs=[]
            with torch.no_grad():
                for i in range(0,len(raw),512):
                    x=sparse.normalize(torch.tensor(np.array(raw[i:i+512]),device=device),stats[ih]);p,z=model(x)
                    preds.append(p.cpu().numpy());zs.append(z.cpu().numpy())
            prediction=np.concatenate(preds);latent=np.concatenate(zs);target=(np.asarray(target,np.float64)-mu)/scale
            fids[split]=fidelity(target,prediction,latent,train_mean)
            p=out/'latents'/name/f'{split}.npy';commit_array(p,latent,dict(config=ch,checkpoint=entry['sha256'],split=split))
            latents[split]=sha(p)
        write(done,dict(config_sha256=ch,checkpoint_sha256=entry['sha256'],run=run,fidelity=fids,latent_sha256=latents,
            dead_count=fids['train']['inactive_count'],dead_definition='zero positive activations over full train pool',environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-started))
        print('encoded',name,flush=True)
    paths=[p for folder in ('arrays','latents','labels','fidelity') for p in (out/folder).rglob('*') if p.is_file()]
    write(out/'features_complete.json',dict(config_sha256=ch,files={str(p.relative_to(out)):sha(p) for p in sorted(paths)},status='features_ready_for_CPU_only'))


def features_gate(root,out):
    ch=sha(Path(root)/CONTRACT);m=read(Path(out)/'features_complete.json')
    if m['config_sha256']!=ch:raise ValueError('Wrong prepared features')
    for n,h in m['files'].items():
        if sha(Path(out)/n)!=h:raise ValueError('Changed prepared feature: '+n)
    return sha(Path(out)/'features_complete.json')


def representation(out,c,task,split):
    lm=task['lm_seed'];rep=task['representation'];hook=task['hook']
    if rep=='token':
        rows=read(Path(out)/'labels'/f'{split}.json');x=np.zeros((len(rows),17),np.float64)
        x[np.arange(len(rows)),[r['token_id'] for r in rows]]=1
        x[:,15]=np.array([r['token_index'] for r in rows])/767;x[:,16]=x[:,15]**2;return x
    if rep=='latent':return load_array(Path(out)/'latents'/task['run']/f'{split}.npy')
    x=load_array(raw_path(out,lm,'init' if rep=='init' else 'trained',split,hook))
    if rep=='random':
        rng=c['rules']['baselines'][f'seed{lm}_{hook}'];R=np.random.Generator(np.random.PCG64(rng['projection_seed'])).normal(size=(256,512));R/=np.linalg.norm(R,axis=0)
        stats=c['statistics'][f'seed{lm}_l3_{hook}.json'];return ((np.asarray(x,np.float64)-np.array(stats['mean']))/stats['scale'])@R
    return x


def target(rows,label,domain,split):
    mask=p10.domain_mask(rows,label,domain,split)
    y=np.array([r[label] if r[label] is not None else -1 for r in rows],int)[mask]
    groups=np.array([r['sequence_id'] for r in rows])[mask]
    return mask,y,groups


def fit_task(values,labels,task):
    data={s:target(labels[s],task['label'],task['domain'],s) for s in ('train','val')}
    y=data['train'][1]
    if task['representation']=='shuffled':y=np.random.Generator(np.random.PCG64(task['shuffle_seed'])).permutation(y)
    return probe.fit(values['train'][data['train'][0]],y,data['train'][2],values['val'][data['val'][0]],data['val'][1],data['val'][2],
                     p10.LABELS[task['label']],task['candidates'],task['prefixes'])


def selection(root,out,max_seconds=7200):
    root,out=Path(root),Path(out);c=verify(root);ch=sha(root/CONTRACT);fh=features_gate(root,out)
    env=session(out,ch,'cpu');torch.set_num_threads(2);start=time.monotonic();labels={s:read(out/'labels'/f'{s}.json') for s in ('train','val')}
    values=None;previous=None
    for i,t in enumerate(c['tasks']):
        path=out/'fits'/f'{t["id"]}.json'
        if path.exists():
            obj=read(path)
            if obj['config_sha256']!=ch or obj['features_sha256']!=fh or obj['task']!=t:raise ValueError('Stale fit')
            if obj['fit']['status']=='failed':raise ValueError('Prior optimizer failure requires review: '+t['id'])
            continue
        if time.monotonic()-start>max_seconds:print('PAUSED CPU fitting; rerun',flush=True);return False
        key=(t['lm_seed'],t['representation'],t['hook'],t.get('run'))
        if key!=previous:values={s:representation(out,c,t,s) for s in ('train','val')};previous=key
        tick=time.monotonic();fit=fit_task(values,labels,t)
        write(path,dict(config_sha256=ch,features_sha256=fh,task=t,fit=fit,test_used_for_selection=False,environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-tick))
        if fit['status']=='failed':raise ValueError('Probe optimizer failed; preserve evidence: '+t['id'])
        print('fit',i+1,'/',len(c['tasks']),t['id'],fit['status'],flush=True)
    files={t['id']:sha(out/'fits'/f'{t["id"]}.json') for t in c['tasks']}
    write(out/'selection_complete.json',dict(config_sha256=ch,features_sha256=fh,fits=files,test_used_for_selection=False))
    return True


def paired_gap(y,prob,ref,classes,threshold,ref_threshold,groups,bootstrap_seed,draws=1000):
    """BA and macro-F1 difference with identical sequence resamples."""
    pred=(prob>=threshold).astype(int) if classes==2 else prob.argmax(1)
    base=(ref>=ref_threshold).astype(int) if classes==2 else ref.argmax(1)
    _,g=np.unique(groups,return_inverse=True);n=int(g.max())+1
    counts=[]
    for x in (pred,base):
        v=np.zeros((n,classes*classes),np.int64);np.add.at(v,(g,y*classes+x),1);counts.append(v)
    observed=[probe.from_confusion(x.sum(0).reshape(classes,classes)) for x in counts]
    rng=np.random.Generator(np.random.PCG64(bootstrap_seed));values={k:[] for k in ('balanced_accuracy','macro_f1')}
    for _ in range(draws):
        w=np.bincount(rng.integers(n,size=n),minlength=n)
        metrics=[probe.from_confusion((w@x).reshape(classes,classes)) for x in counts]
        for k in values:
            if all(m[k] is not None for m in metrics):values[k].append(metrics[0][k]-metrics[1][k])
    return {k:dict(difference=observed[0][k]-observed[1][k] if all(m[k] is not None for m in observed) else None,
                   ci95=np.percentile(v,[2.5,97.5]).tolist() if v else None,valid=len(v),requested=draws,seed=bootstrap_seed) for k,v in values.items()}


def evaluate(root,out,max_seconds=7200):
    root,out=Path(root),Path(out);c=verify(root);ch=sha(root/CONTRACT);fh=features_gate(root,out)
    selected=read(out/'selection_complete.json')
    if selected['config_sha256']!=ch or selected['features_sha256']!=fh or selected['test_used_for_selection']:raise ValueError('Selection must be frozen first')
    if set(selected['fits'])!={t['id'] for t in c['tasks']}:raise ValueError('Incomplete selection')
    for n,h in selected['fits'].items():
        if sha(out/'fits'/f'{n}.json')!=h:raise ValueError('Fit changed after selection')
    env=session(out,ch,'cpu');start=time.monotonic();rows=read(out/'labels/test.json');previous=None;values=None
    for i,t in enumerate(c['tasks']):
        dest=out/'semantic'/f'{t["id"]}.json';fit=read(out/'fits'/f'{t["id"]}.json')['fit']
        if dest.exists():
            d=read(dest)
            if d['selection_sha256']!=sha(out/'selection_complete.json'):raise ValueError('Stale evaluation')
            for n,h in d.get('predictions',{}).items():
                if sha(out/n)!=h:raise ValueError('Changed predictions')
            continue
        if time.monotonic()-start>max_seconds:print('PAUSED CPU reporting; rerun',flush=True);return False
        key=(t['lm_seed'],t['representation'],t['hook'],t.get('run'))
        if key!=previous:values=representation(out,c,t,'test');previous=key
        mask,y,groups=target(rows,t['label'],t['domain'],'test');reports={};predictions={}
        tick=time.monotonic()
        if fit['status']=='passed':
            for size,f in fit['selected'].items():
                prob=probe.probabilities(f,values[mask]);p=out/'predictions'/t['id']/f'{size}.npy'
                commit_array(p,prob,dict(config=ch,selection=sha(out/'selection_complete.json'),task=t['id'],size=size))
                predictions[str(p.relative_to(out))]=sha(p)
                score=probe.metrics(y,prob,p10.LABELS[t['label']],f['threshold'],groups,t['bootstrap_seed'])
                gaps={}
                for refid in t['full_references']:
                    rf=read(out/'fits'/f'{refid}.json')['fit']
                    if rf['status']!='passed':gaps[refid]=dict(status=rf['status']);continue
                    ref=rf['selected']['full'];rt=next(x for x in c['tasks'] if x['id']==refid)
                    rx=representation(out,c,rt,'test');rp=probe.probabilities(ref,rx[mask])
                    gaps[refid]=paired_gap(y,prob,rp,p10.LABELS[t['label']],f['threshold'],ref['threshold'],groups,t['bootstrap_seed'])
                score['paired_full_probe_gaps']=gaps;reports[size]=score
        write(dest,dict(config_sha256=ch,selection_sha256=sha(out/'selection_complete.json'),task=t,status=fit['status'],
             test_support=probe.support(y,groups,p10.LABELS[t['label']]),reports=reports,predictions=predictions,
             environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-tick))
        print('report',i+1,'/',len(c['tasks']),t['id'],flush=True)
    write(out/'evaluation_complete.json',dict(status='evaluation_return_audit_pending',p10_complete=False,config_sha256=ch,
        selection_sha256=sha(out/'selection_complete.json'),tasks=len(c['tasks']),semantic={t['id']:sha(out/'semantic'/f'{t["id"]}.json') for t in c['tasks']},
        fidelity={e['run']['name']:sha(out/'fidelity'/f'{e["run"]["name"]}.json') for e in c['selected']}))
    return True


def export(out,archive):
    out,archive=Path(out),Path(archive)
    paths=[p for p in out.rglob('*') if p.is_file() and not {'cache','arrays','latents','exports'}&set(p.relative_to(out).parts) and p.suffix!='.tmp' and '.uncommitted_' not in p.name]
    files={str(p.relative_to(out)):sha(p) for p in paths}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(out)))
        z.writestr('return_manifest.json',json.dumps(dict(files=files,p10_complete=False,retained_on_drive=['cache','arrays','latents']),indent=2))
    write(archive.with_suffix('.sha256.json'),dict(sha256=sha(archive),bytes=archive.stat().st_size))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','encode','select','evaluate','export']);p.add_argument('--root',default='.');p.add_argument('--source');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--max-seconds',type=float,default=7200);p.add_argument('--archive');a=p.parse_args()
    if a.max_seconds<=0:raise ValueError('Positive time limit required')
    if a.action=='extract':extract(a.root,a.source,a.output,a.max_seconds)
    elif a.action=='encode':encode(a.root,a.output,a.device)
    elif a.action=='select':selection(a.root,a.output,a.max_seconds)
    elif a.action=='evaluate':evaluate(a.root,a.output,a.max_seconds)
    else:export(a.output,a.archive)

if __name__=='__main__':main()
