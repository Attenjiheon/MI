"""P9 TC sparse seed 1; inherited p8_evaluation.py procedure with isolated contract."""
from pathlib import Path
import gzip
import json
import time
import numpy as np
import torch
from .p5 import read,write,derived,task_key,domains,LABELS,QUOTAS,cache_array,load_split
from . import p5_probe as probe
from . import dictionary as sparse
from .runtime import sha,deterministic
from .p7_metrics import fidelity

CONTRACT='experiment_v1_4/p9_eval_r1/tc_contract.json'
SELECTED='experiment_v1_4/results/p9_training_audit_20260930_01/selected_dictionary_manifest.json'
RECEIPT='experiment_v1_4/results/p5_final_audit_20260925_01/p6_cache_manifest.json'


def verify(root):
    root=Path(root);config=read(root/CONTRACT)
    if config['schema']!='p9-tc-evaluation-v1.4-r1':raise ValueError('Wrong P9 TC contract')
    for name,digest in config['files'].items():
        if sha(root/name)!=digest:raise ValueError('Changed P9 TC input: '+name)
    return config


def model_entry(root,entry,device='cpu'):
    p=Path(root)/entry['checkpoint']
    if sha(p)!=entry['sha256']:raise ValueError('Changed selected SAE')
    state=torch.load(p,map_location='cpu',weights_only=False)
    if state['run']!=entry['run'] or state['update']!=entry['selected_update']:raise ValueError('Checkpoint selection mismatch')
    d=sparse.Dictionary('transcoder',entry['run']['k'],0,width=256)
    d.load_state_dict(state['model']);d.eval().requires_grad_(False).to(device)
    stats={h:{k:v.to(device) for k,v in s.items()} for h,s in state['stats'].items()}
    return d,stats


def encode(d,stats,values,device):
    predictions=[];latents=[]
    with torch.no_grad():
        for i in range(0,len(values),512):
            x=sparse.normalize(torch.tensor(values[i:i+512],device=device),stats['u'])
            y,z=d(x);predictions.append(y.cpu().numpy());latents.append(z.cpu().numpy())
    return np.concatenate(predictions),np.concatenate(latents)


def source_gate(root,source):
    root,source=Path(root),Path(source);c=verify(root);receipt=read(root/RECEIPT)
    if sha(source/'contract.json')!=receipt['config_sha256']:raise ValueError('P5 contract mismatch')
    p5config=read(root/'experiment_v1_4/p5_r2/contract.json')
    for split in QUOTAS:
        if sha(source/'labels'/f'{split}.json')!=receipt['labels'][split]:raise ValueError('P5 label mismatch')
        _,rows=load_split(root,p5config,split)
        if rows!=read(source/'labels'/f'{split}.json'):raise ValueError('READ key join mismatch')
    # Validate source bytes from independent audit, including init controls.
    for i,item in enumerate(f for f in receipt['cache_files'] if f['path'].startswith('cache/seed0_trained/')):
        if sha(source/item['path'])!=item['sha256']:raise ValueError('P5 cache mismatch')
        if (i+1)%40==0:print('source cache verified',i+1,flush=True)
    return c


def load_arrays(root,source,entry):
    run=entry['run'];receipt=read(Path(root)/RECEIPT)
    return {s:cache_array(source,run['lm_seed'],'trained',s,run['layer'],'m',receipt['config_sha256']) for s in QUOTAS}


def directions(config,run):
    r=config['baselines'][f"seed{run['lm_seed']}_l{run['layer']}"]
    rng=np.random.Generator(np.random.PCG64(r['projection_seed']))
    R=rng.normal(size=(256,512));R/=np.linalg.norm(R,axis=0)
    return R,r


def task_name(rep,label,domain):return f'{rep}_{label}_{domain}'


def selection(root,source,output,entry,device='cpu'):
    """Commit all fits before any corresponding test prediction is evaluated."""
    root,source,output=Path(root),Path(source),Path(output);config=verify(root)
    folder=output/'runs'/entry['run']['name'];path=folder/'selection.json';ch=sha(root/CONTRACT)
    if path.exists():
        selected=read(path)
        if selected['config_sha256']!=ch:raise ValueError('Stale selection')
        return selected
    run=entry['run'];d,stats=model_entry(root,entry,device);receipt=read(root/RECEIPT)
    arrays={s:cache_array(source,run['lm_seed'],'trained',s,run['layer'],'m',receipt['config_sha256']) for s in ('train','val')}
    labels={s:read(source/'labels'/f'{s}.json') for s in arrays}
    mu,scale=stats['m']['mu'].cpu().numpy(),float(stats['m']['scale']);R,rngspec=directions(config,run)
    inputs={s:cache_array(source,run['lm_seed'],'trained',s,run['layer'],'u',receipt['config_sha256']) for s in arrays}
    latents={s:encode(d,stats,a,device)[1] for s,a in inputs.items()}
    representations=dict(coordinate=arrays,random={s:((a.astype(np.float64)-mu)/scale)@R for s,a in arrays.items()},transcoder=latents)
    fits={};start=time.monotonic()
    for rep,values in representations.items():
        for limit in ('all','128'):
            candidates=None if limit=='all' else rngspec['subsets'][rep]
            for label,classes in LABELS.items():
                for domain in (('iid','transfer') if label in ('current','previous') else ('iid',)):
                    name=task_name(rep+('_128' if limit=='128' else ''),label,domain)
                    taskpath=folder/'fits'/f'{name}.json'
                    if taskpath.exists():
                        result=read(taskpath)
                        if result['config_sha256']!=ch:raise ValueError('Stale fit')
                    else:
                        masks={s:domains(labels[s],label,domain,s) for s in arrays}
                        y={s:np.asarray([r[label] if r[label] is not None else -1 for r in labels[s]],int)[masks[s]] for s in arrays}
                        groups={s:np.asarray([r['sequence_id'] for r in labels[s]])[masks[s]] for s in arrays}
                        reuse=config['reused_probes'].get(f"seed{run['lm_seed']}_l{run['layer']}_{name}")
                        if reuse:
                            saved=read(root/reuse)
                            result=saved['result']
                            # Final independent audit files store only summaries; original coefficient files are packaged.
                            result={k:v for k,v in result.items() if k not in ('evaluation','test_support')}
                        else:
                            result=probe.fit(values['train'][masks['train']],y['train'],groups['train'],values['val'][masks['val']],y['val'],groups['val'],classes,candidates,True)
                        result=dict(config_sha256=ch,result=result)
                        write(taskpath,result)
                    fits[name]=result['result'];print(run['name'],name,fits[name]['status'],flush=True)
    for rep in ('fullu','fullm'):
        for label in LABELS:
            for domain in (('iid','transfer') if label in ('current','previous') else ('iid',)):
                name=task_name(rep,label,domain)
                reused=config['reused_probes'][f"seed{run['lm_seed']}_l{run['layer']}_{name}"]
                original=read(root/reused)['result']
                fits[name]={k:v for k,v in original.items() if k not in ('evaluation','test_support')}
    result=dict(config_sha256=ch,run=run,checkpoint_sha256=entry['sha256'],test_used_for_selection=False,
                projection_seed=rngspec['projection_seed'],subsets=rngspec['subsets'],fits=fits,elapsed_seconds=time.monotonic()-start)
    write(path,result);return result


def semantic(root,source,output,entry,device='cpu'):
    root,source,output=Path(root),Path(source),Path(output);config=verify(root);run=entry['run'];folder=output/'runs'/run['name'];ch=sha(root/CONTRACT)
    selection_path=folder/'selection.json';selected=read(selection_path)
    if selected['config_sha256']!=ch or selected['test_used_for_selection']:raise ValueError('Selection not frozen')
    destination=folder/'semantic.json'
    if destination.exists():
        if read(destination)['selection_sha256']!=sha(selection_path):raise ValueError('Stale semantic evaluation')
        return
    d,stats=model_entry(root,entry,device);arrays=load_arrays(root,source,entry);mu=stats['m']['mu'].cpu().numpy();scale=float(stats['m']['scale'])
    R,_=directions(config,run);ys={};zs={};fids={}
    receipt=read(root/RECEIPT)
    inputs={s:cache_array(source,run['lm_seed'],'trained',s,run['layer'],'u',receipt['config_sha256']) for s in QUOTAS}
    mean=((arrays['train'].astype(np.float64)-mu)/scale).mean(0)
    for s,a in arrays.items():
        ys[s],zs[s]=encode(d,stats,inputs[s],device)
        fids[s]=fidelity((a.astype(np.float64)-mu)/scale,ys[s],zs[s],mean)
    fids['dead_definition']='selected checkpoint re-encodes every train position; zero positive activations'
    labels={s:read(source/'labels'/f'{s}.json') for s in QUOTAS}
    values=dict(fullu=inputs['test'],fullm=arrays['test'],coordinate=arrays['test'],random=((arrays['test'].astype(np.float64)-mu)/scale)@R,transcoder=zs['test'])
    reports={};predictions={}
    for name,fit in selected['fits'].items():
        if fit['status']!='passed':reports[name]=fit;continue
        domain=name.rsplit('_',1)[1];label=name.rsplit('_',2)[1];rep=name.split('_')[0]
        mask=domains(labels['test'],label,domain,'test');x=values[rep][mask]
        y=np.asarray([r[label] if r[label] is not None else -1 for r in labels['test']],int)[mask]
        groups=np.asarray([r['sequence_id'] for r in labels['test']])[mask]
        key=config['bootstrap_keys'][run['name']]+'|semantic|'+name;bs=derived('bootstrap',key)
        report=dict(status='passed',support=probe.support(y,groups,LABELS[label]),bootstrap_seed=bs,evaluation={})
        for size,model in fit['selected'].items():
            prob=probe.probabilities(model,x);predictions[name+'_'+size]=prob
            score=probe.metrics(y,prob,LABELS[label],model['threshold'],groups,bs)
            if label in ('current','previous'):
                diff=np.asarray([r['previous'] is not None and r['current']!=r['previous'] for r in labels['test']])[mask]
                score['current_ne_previous']=probe.metrics(y[diff],prob[diff],LABELS[label],model['threshold'],groups[diff],bs)
            report['evaluation'][size]=score
        reports[name]=report
    # Raw prediction arrays permit independent CI recomputation and paired layer differences.
    from .p7_patching import commit_arrays
    commit_arrays(folder/'semantic_predictions.npz',predictions,ch)
    write(destination,dict(config_sha256=ch,selection_sha256=sha(selection_path),run=run,fidelity=fids,semantic=reports,p9_complete=False))


def records(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
