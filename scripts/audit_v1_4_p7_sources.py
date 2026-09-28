"""Reproduce frozen P7 results from audited P5 cache; never select new test rules."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4 import p7,p5_probe as probe
from interp_v1_4.p5 import read,write,domains,LABELS,QUOTAS,session,derived,load_split
from interp_v1_4.runtime import sha,deterministic
from interp_v1_4.dictionary import normalize,denormalize
from interp_v1_4.p7_patching import load_lm,forward,causal_inputs,causal_unit,main_features
from interp_v1_4.p7_metrics import bins


def compare(a,b,path='root',atol=2e-6,rtol=2e-5):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,'keys')
        for key in a:compare(a[key],b[key],path+'/'+str(key),atol,rtol)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),atol,rtol)
    elif a is None or isinstance(a,(str,bool)):
        assert a==b,(path,a,b)
    else:
        assert np.isfinite(a) and np.isfinite(b),path
        np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=path)


def fidelity_independent(x,y,z,mean):
    x=x.astype(float);y=y.astype(float);err=y-x;ss=np.sum(err*err)
    centered=x-np.mean(x,axis=0);error_centered=err-np.mean(err,axis=0)
    den=np.sum((x-mean)**2);var=np.sum(centered**2);activity=(z>0);l0=activity.sum(1);rates=activity.mean(0)
    return dict(positions=len(x),mse=float(ss/x.size),nmse=float(ss/den) if den else None,r2=float(1-ss/var) if var else None,
        ev=float(1-np.sum(error_centered**2)/var) if var else None,
        l0=dict(mean=float(l0.mean()),median=float(np.median(l0)),quantiles=np.percentile(l0,[0,5,25,75,95,100]).tolist()),
        activation_rates=rates.tolist(),inactive_count=int(np.sum(~activity.any(0))),inactive_fraction=float(np.mean(~activity.any(0))))


def audit_run(root,source,returned,out,c,e,labels,fit_cache):
    run=e['run'];folder=returned/'runs'/run['name'];selection=read(folder/'selection.json');semantic=read(folder/'semantic.json');matching=read(folder/'matching.json')
    d,stats=p7.model_entry(root,e,'cuda');arrays=p7.load_arrays(root,source,e)
    mu=stats['mu'].cpu().numpy();scale=float(stats['scale']);R,_=p7.directions(c,run)
    zs={};pred={};xs={s:(a.astype(float)-mu)/scale for s,a in arrays.items()};mean=xs['train'].mean(0)
    for s in QUOTAS:
        pred[s],zs[s]=p7.encode(d,stats,arrays[s],'cuda')
        compare(semantic['fidelity'][s],fidelity_independent(xs[s],pred[s],zs[s],mean),'fidelity/'+s)
    reps=dict(full=arrays,coordinate=arrays,random={s:x@R for s,x in xs.items()},sae=zs)
    predictions=np.load(folder/'semantic_predictions.npz')
    fit_count=0;metric_count=0
    for task,stored in selection['fits'].items():
        _,label,domain=task.rsplit('_',2);rep=task.split('_')[0];x=reps[rep]
        masks={s:domains(labels[s],label,domain,s) for s in QUOTAS}
        yy={s:np.array([r[label] if r[label] is not None else -1 for r in labels[s]],int)[masks[s]] for s in QUOTAS}
        groups={s:np.array([r['sequence_id'] for r in labels[s]])[masks[s]] for s in QUOTAS}
        reuse=c['reused_probes'].get(f"seed{run['lm_seed']}_l{run['layer']}_{task}")
        if reuse:
            expected=read(root/reuse)['result']
        else:
            key=f"seed{run['lm_seed']}_l{run['layer']}_{task}"+('_k'+str(run['k']) if rep=='sae' else '')
            cachefile=fit_cache/(key+'.json')
            if cachefile.exists():expected=read(cachefile)
            else:
                candidates=selection['subsets'][rep] if '_128_' in task else None
                expected=probe.fit(x['train'][masks['train']],yy['train'],groups['train'],x['val'][masks['val']],yy['val'],groups['val'],LABELS[label],candidates,True)
                write(cachefile,expected)
        assert stored['status']==expected['status'],task
        if stored['status']!='passed':compare(stored,expected,task);continue
        assert stored['ranking']==expected['ranking'],task
        assert stored['candidate_count']==expected['candidate_count'],task
        for size,model in stored['selected'].items():
            other=expected['selected'][size]
            for key in ('columns','classes','lam','threshold'):assert model[key]==other[key],(task,size,key)
            compare(model,other,task+'/'+size)
            pp=probe.probabilities(model,x['test'][masks['test']])
            np.testing.assert_allclose(pp,predictions[task+'_'+size],atol=2e-6,rtol=2e-5,err_msg=task+'/'+size)
            bs=derived('bootstrap',c['bootstrap_keys'][run['name']]+'|semantic|'+task)
            metrics=probe.metrics(yy['test'],pp,LABELS[label],model['threshold'],groups['test'],bs)
            if label in ('current','previous'):
                diff=np.array([r['previous'] is not None and r['current']!=r['previous'] for r in labels['test']])[masks['test']]
                metrics['current_ne_previous']=probe.metrics(yy['test'][diff],pp[diff],LABELS[label],model['threshold'],groups['test'][diff],bs)
            compare(semantic['semantic'][task]['evaluation'][size],metrics,task+'/'+size+'/metrics')
            metric_count+=1
        fit_count+=1;print(run['name'],'probe verified',task,flush=True)
    predictions.close();del arrays,xs,zs,pred,reps
    lm=load_lm(root,c,run['lm_seed'],'cuda')
    val=causal_inputs(root,c,'val');checked_bins=0
    for size in ('single','up_to_four'):
        J=main_features(selection,'sae',size)
        if J is None:continue
        hn=[];zn=[]
        for pairs in val.values():
            for start in range(0,len(pairs),16):
                chunk=pairs[start:start+16];n=len(chunk)
                prefixes=[p['original_prefix_ids'] for p in chunk]+[p['counterfactual_prefix_ids'] for p in chunk]
                _,h=forward(lm,prefixes,run['layer']);z=d.encode(normalize(h,stats));delta=z[n:,J]-z[:n,J]
                hn.extend((stats['scale']*(delta@d.decoder[:,J].T)).norm(dim=1).cpu().tolist()*2)
                zn.extend(delta.norm(dim=1).cpu().tolist()*2)
        saved=matching['rules'][size]
        compare(saved['validation_h_norms'],hn,'validation h norms');compare(saved['validation_z_norms'],zn,'validation z norms');compare(saved['rule'],bins(hn,zn),'matching bins');checked_bins+=1
    # Deterministic audit coverage: first corpus-order pair in each frozen suite, independent of effects.
    sampled=0
    for suite,pairs in causal_inputs(root,c,'test').items():
        p=pairs[0];actual=causal_unit(lm,d,stats,R,run,p,selection,matching,c);saved=read(folder/'causal'/suite/'00000.json')
        compare(saved['rows'],actual['rows'],'causal/'+suite,atol=1e-5,rtol=1e-4)
        compare(saved['candidates'],actual['candidates'],'candidates/'+suite,atol=1e-5,rtol=1e-4);sampled+=1
    targets=read(root/c['reconstruction_targets'])[:16];general={r['sequence_id']:r for r in p7.records(root/c['data_root']/'test/general.jsonl.gz')}
    prefixes=[general[t['sequence_id']]['token_ids'][:t['token_index']+1] for t in targets]
    original,h=forward(lm,prefixes,run['layer']);replacement=denormalize(d(normalize(h,stats))[0],stats);patched,_=forward(lm,prefixes,run['layer'],replacement)
    with np.load(folder/'replacement/00000.npz') as saved:
        np.testing.assert_allclose(original.cpu().numpy(),saved['original_logits'],atol=1e-5,rtol=1e-4)
        np.testing.assert_allclose(patched.cpu().numpy(),saved['patched_logits'],atol=1e-5,rtol=1e-4)
    # Recompute every causal/replacement cluster CI from the verified raw output,
    # capture the writer so the original returned evidence remains read-only.
    from interp_v1_4 import p7_patching as patch_module
    original_write=patch_module.write
    try:
        patch_module.write=lambda path,value:compare(read(path),value,'summary-recomputation')
        patch_module.summarize(root,c,returned,e)
    finally:patch_module.write=original_write
    del lm,d;torch.cuda.empty_cache()
    return dict(run=run['name'],status='passed',probe_tasks=fit_count,semantic_reports=metric_count,fidelity_positions=dict(QUOTAS),validation_bin_sets=checked_bins,causal_replayed_pairs=sampled,replacement_replayed_targets=16)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--source',required=True);p.add_argument('--returned',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root,source,returned,out=map(Path,(a.root,a.source,a.returned,a.output));out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);deterministic(707)
    if not torch.cuda.is_available():raise RuntimeError('Source audit requires CUDA for original numeric execution path')
    audit_config=read(root/'experiment_v1_4/p7_audit_r2/contract.json')
    for name,digest in audit_config['files'].items():assert sha(root/name)==digest,name
    c=p7.source_gate(root,source);ch=sha(root/p7.CONTRACT);identity=dict(original_contract_sha256=ch,audit_contract_sha256=sha(root/'experiment_v1_4/p7_audit_r2/contract.json'))
    for name,digest in read(root/'experiment_v1_4/p7_audit_r2/return_manifest.json')['files'].items():
        assert sha(returned/name)==digest,'Changed returned evidence: '+name
    write(out/'identity.json',identity);env=session(out,ch,'cuda');labels={s:read(source/'labels'/f'{s}.json') for s in QUOTAS}
    results=[]
    for e in c['saes']:
        name=e['run']['name'];dest=out/'runs'/(name+'.json')
        if dest.exists():results.append(read(dest));continue
        tick=time.monotonic()
        try:
            result=audit_run(root,source,returned,out,c,e,labels,out/'refits')
            result.update(identity=identity,environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-tick)
            write(dest,result);results.append(result)
        except Exception as exc:
            write(out/'failures'/f'{time.time_ns()}.json',dict(run=name,error=repr(exc),identity=identity,environment_id=env['environment_id']))
            raise
    # Reproduce tables and all paired layer contrasts without changing source files.
    from interp_v1_4 import p7_aggregate as aggregate_module
    import matplotlib.pyplot as plt
    import csv,io
    saved_write,saved_csv,saved_figure=aggregate_module.write,aggregate_module.csvfile,plt.Figure.savefig
    def check_csv(path,rows):
        fields=sorted({k for row in rows for k in row});buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        assert Path(path).read_text()==buf.getvalue().replace('\r\n','\n'),str(path)
    try:
        aggregate_module.write=lambda path,value:compare(read(path),value,'aggregate-recomputation')
        aggregate_module.csvfile=check_csv
        plt.Figure.savefig=lambda *args,**kwargs:None
        aggregate_module.aggregate(root,source,returned)
    finally:
        aggregate_module.write,aggregate_module.csvfile,plt.Figure.savefig=saved_write,saved_csv,saved_figure
    write(out/'source_audit_complete.json',dict(status='source_reproduction_passed_return_review_pending',p7_complete=False,identity=identity,runs=results,
        scope='All novel probe refits, all test predictions and semantic CIs, full-pool fidelity/dead counts, full validation bins; four fixed causal pairs and 16 replacement targets replayed per run. All causal/replacement summary CIs and paired layer contrasts reproduced. Full raw-output checks and independent summary point audit are separate local evidence.'))

if __name__=='__main__':main()
