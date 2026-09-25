"""Read-only P5 cache audit. Independent NumPy/SciPy implementation; no fitting.

Does not import the production probe, cache reader, or metric implementation.
Source metadata is anchored to the returned 2026-09-24 snapshot inventory.
"""
import argparse
import gzip
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import expit, softmax, logsumexp

LABELS = dict(current=2, previous=2, query=4, A=2, B=2, C=2, D=2, state=16)
POLICY = dict(version='p5-independent-audit-r1', atol=1e-9, rtol=1e-8,
              bootstrap_draws=1000, probability_digest='little-endian float64')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def require(ok, message):
    if not ok: raise ValueError(message)

def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if p.exists():
        require(p.read_text() == content, 'Refusing different audit overwrite: '+str(p)); return
    t = p.with_suffix(p.suffix+'.tmp'); t.write_text(content); t.replace(p)

def close(a, b, name):
    require(np.shape(a) == np.shape(b) and np.allclose(a, b, atol=POLICY['atol'], rtol=POLICY['rtol']), 'Numerical mismatch: '+name)

def compare(actual, expected, name='metrics'):
    if isinstance(expected, dict):
        for k,v in expected.items(): require(k in actual, name+'/'+k); compare(actual[k],v,name+'/'+k)
    elif expected is None: require(actual is None, name+' expected None')
    elif isinstance(expected, (list,float)): close(actual,expected,name)
    else: require(actual == expected,name+' exact mismatch')

def stat_cm(cm):
    cm = np.asarray(cm); n=cm.sum(axis=1); good=n>0
    recalls=np.divide(cm.diagonal(),n,out=np.zeros(len(n)),where=good)
    denom=n+cm.sum(axis=0)
    f=np.divide(2*cm.diagonal(),denom,out=np.zeros(len(n)),where=denom>0)
    return dict(balanced_accuracy=float(recalls.mean()) if good.all() else None,
                macro_f1=float(f.mean()) if good.all() else None,
                observed_class_balanced_accuracy=float(recalls[good].mean()) if good.any() else None)

def weighted_auc(y, scores, weights=None):
    """Integrate the empirical ROC trapezoids, including tied scores."""
    y=np.asarray(y); weights=np.ones(len(y)) if weights is None else np.asarray(weights)
    levels, codes=np.unique(scores,return_inverse=True)
    pos=np.bincount(codes,weights=weights*(y==1),minlength=len(levels))[::-1]
    neg=np.bincount(codes,weights=weights*(y==0),minlength=len(levels))[::-1]
    if pos.sum()==0 or neg.sum()==0:return None
    tpr=np.r_[0,np.cumsum(pos)/pos.sum()]; fpr=np.r_[0,np.cumsum(neg)/neg.sum()]
    return float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))

def evaluate(y, prob, classes, threshold, groups, seed, draws=1000):
    y=np.asarray(y,int); prob=np.asarray(prob)
    pred=(prob>=threshold).astype(int) if classes==2 else np.argmax(prob,axis=1)
    cm=np.zeros((classes,classes),np.int64); np.add.at(cm,(y,pred),1)
    out=dict(**stat_cm(cm),confusion_matrix=cm.tolist(),positions=len(y),class_counts=cm.sum(1).tolist(),
             binary_auroc=weighted_auc(y,prob) if classes==2 else None)
    # Preaggregate by sequence. Draws use the declared PCG64 seed and sorted group IDs.
    ids, group=np.unique(groups,return_inverse=True); n=len(ids)
    cells=np.zeros((n,classes*classes),np.int64);np.add.at(cells,(group,y*classes+pred),1)
    metrics={k:[] for k in ('balanced_accuracy','macro_f1','binary_auroc')}
    if classes==2:
        levels,codes=np.unique(prob,return_inverse=True)
    rng=np.random.Generator(np.random.PCG64(seed))
    for _ in range(draws):
        sampled=rng.integers(0,n,size=n)
        multiplicity=np.bincount(sampled,minlength=n)
        b=stat_cm(np.sum(cells*multiplicity[:,None],axis=0).reshape(classes,classes))
        for key in ('balanced_accuracy','macro_f1'):
            if b[key] is not None:metrics[key].append(b[key])
        if classes==2:
            w=multiplicity[group]
            pos=np.bincount(codes,weights=w*(y==1),minlength=len(levels))[::-1]
            neg=np.bincount(codes,weights=w*(y==0),minlength=len(levels))[::-1]
            if pos.sum()>0 and neg.sum()>0:
                tp=np.r_[0,np.cumsum(pos)/pos.sum()]; fp=np.r_[0,np.cumsum(neg)/neg.sum()]
                metrics['binary_auroc'].append(float(np.sum(np.diff(fp)*(tp[1:]+tp[:-1])/2)))
    out['cluster_bootstrap']=dict(seed=seed,requested=draws,clusters=n,metrics={
        k:dict(valid=len(v),ci95=np.quantile(v,[.025,.975]).tolist() if v else None) for k,v in metrics.items()})
    return out

def predict(fit,x):
    cols=fit['columns']; z=(np.asarray(x[:,cols],np.float64)-np.asarray(fit['mean']))/np.asarray(fit['std'])
    theta=np.asarray(fit['coefficients']);c=fit['classes']
    if c==2:
        logits=np.sum(z*theta[:-1],axis=1)+theta[-1]
        return expit(logits),logits
    w=theta.reshape(c,len(cols)+1); logits=z@w[:,:-1].T+w[:,-1]
    return softmax(logits,axis=1),logits

def rank_columns(x,y,pool):
    v=np.asarray(x[:,pool],np.float64); keep=v.std(0)>=1e-8
    ids=np.asarray(pool)[keep];v=v[:,keep]; overall=v.mean(0); classes=np.unique(y)
    ssb=np.zeros(len(ids));ssw=np.zeros(len(ids))
    for c in classes:
        group=v[y==c];mean=group.mean(0)
        ssb+=len(group)*(mean-overall)**2;ssw+=((group-mean)**2).sum(0)
    f=np.divide(ssb/(len(classes)-1),ssw/(len(y)-len(classes)),out=np.full(len(ids),np.inf),where=ssw>0)
    return ids[np.lexsort((ids,-f))].tolist()

def independent_labels(inputs,c):
    result={}
    for split,quota in c['quotas'].items():
        base=inputs/c['data_root']/'interpretation'
        for p in [base/f'{split}.jsonl.gz',base/f'{split}.read_positions.json']:
            require(sha(p)==c['files'][str(p.relative_to(inputs))],'Corpus checksum '+str(p))
        with gzip.open(base/f'{split}.jsonl.gz','rt') as f: seqs=[json.loads(line) for line in f]
        by_id={r['sequence_id']:r for r in seqs};require(len(by_id)==len(seqs),'Duplicate corpus sequence')
        positions=read(base/f'{split}.read_positions.json'); rows=[]
        require(len(positions)==quota,'Quota')
        keys=[(p['sequence_id'],p['token_index']) for p in positions];require(keys==sorted(set(keys)),'Reserved keys')
        for p in positions:
            seq=by_id[p['sequence_id']];event=seq['read_events'][p['event_id']];t=p['token_index']
            q='ABCD'.index(event['query_var']);state=event['state_at_read']
            require(event['read_id']==p['event_id'] and event['query_token_index']==t and event['answer_token_index']==t+1,'READ event mismatch')
            require(seq['token_ids'][t-1]==8 and seq['token_ids'][t]==9+q and seq['token_ids'][t+1]==13+event['answer'] and event['answer']==state[q],'Answer position mismatch')
            rows.append(dict(sequence_id=p['sequence_id'],token_index=t,event_id=p['event_id'],token_id=seq['token_ids'][t],
                current=event['answer'],previous=event['previous_value_or_null'],query=q,**dict(zip('ABCD',state)),state=sum(v*2**i for i,v in enumerate(state))))
        result[split]=rows
    return result

def stage_cache(source,work,seed,kind,c,ch,labels,inventory):
    """Hash each archive once, decompress every hook once, stage one seed/kind (~3GB)."""
    target=work/f'seed{seed}_{kind}';target.mkdir(parents=True,exist_ok=True); arrays={}; receipts=[]
    model=next(v for v in c['models'] if v['lm_seed']==seed)
    checkpoint=c['files'][model['checkpoint' if kind=='trained' else 'init_checkpoint']]
    for split,rows in labels.items():
        n=len(rows);seen=np.zeros(n,bool);arrays[split]={}
        for l in range(12):
            for h in 'hum': arrays[split][l,h]=np.lib.format.open_memmap(target/f'{split}_{l}_{h}.npy',mode='w+',dtype='float32',shape=(n,256))
        rel=Path('cache')/f'seed{seed}_{kind}'/split
        markers=sorted(k for k in inventory if k.startswith(str(rel)+'/') and k.endswith('.json'))
        ids=sorted(set(r['sequence_id'] for r in rows));offsets=[]
        for name in markers:
            marker=source/name; require(sha(marker)==inventory[name],'Marker changed '+name)
            m=read(marker);ident=m['identity']; offsets.append(ident['sequence_offset'])
            for k,v in dict(config_sha256=ch,lm_seed=seed,checkpoint_sha256=checkpoint,split=split,position_type='READ',labels_sha256=inventory[f'labels/{split}.json']).items():require(ident[k]==v,'Cache identity '+k)
            selected=set(ids[ident['sequence_offset']:ident['sequence_offset']+128]);expected=np.array([i for i,r in enumerate(rows) if r['sequence_id'] in selected])
            require(m['layers']==list(range(12)) and m['hooks']==c['hooks'],'Hook/layer mismatch')
            p=marker.with_suffix('.npz'); tmp=target/'chunk.npz'
            shutil.copyfile(p,tmp); require(sha(tmp)==m['sha256'],'Cache checksum '+str(p))
            with np.load(tmp,allow_pickle=False) as z:
                require(set(z.files)=={'row_indices'}|{f'{l}_{h}' for l in range(12) for h in 'hum'},'NPZ keys')
                idx=z['row_indices']; require(idx.dtype.kind in 'iu' and np.array_equal(idx,expected),'Cache row/sequence join')
                require(len(idx)==m['positions'] and not seen[idx].any(),'Cache duplicate/count')
                seen[idx]=True
                for l in range(12):
                    for h in 'hum':
                        x=z[f'{l}_{h}'];require(x.shape==(len(idx),256) and x.dtype==np.float32 and np.isfinite(x).all(),'Cache tensor')
                        arrays[split][l,h][idx]=x
            tmp.unlink();receipts.append(dict(path=str(p.relative_to(source)),sha256=m['sha256'],positions=len(expected)))
            print('cache',name,flush=True)
        require(offsets==list(range(0,len(ids),128)) and seen.all(),'Cache coverage')
        for a in arrays[split].values():a.flush()
    return arrays,receipts,target

def verify_task(name,item,arrays,labels,c):
    label,domain=name.rsplit('_',2)[-2:];classes=LABELS[label];task=c['tasks'][name];result=item['result']
    require(result['status']=='passed','Returned task is not passed '+name)
    masks={s:np.array([r[label] is not None and (domain!='transfer' or (r['query']==3 if s=='test' else r['query']!=3)) for r in rows]) for s,rows in labels.items()}
    x={s:np.asarray(arrays[s][masks[s]],np.float64) for s in labels}
    y={s:np.array([r[label] if r[label] is not None else -1 for r in rows])[masks[s]] for s,rows in labels.items()}
    groups={s:np.array([r['sequence_id'] for r in rows])[masks[s]] for s,rows in labels.items()}
    if task.get('shuffle_seed') is not None:y['train']=np.random.Generator(np.random.PCG64(task['shuffle_seed'])).permutation(y['train'])
    for s in labels:
        support=[dict(class_id=k,positions=int((y[s]==k).sum()),sequences=len(set(groups[s][y[s]==k]))) for k in range(classes)]
        require(support==(result['test_support'] if s=='test' else result['support'][s]),'Support '+name+'/'+s)
    prefix=result['ranking'] is not None;pool=list(range(x['train'].shape[1]))
    if prefix:
        rep='random' if '_random' in name else 'coordinate'
        if '_128_' in name:pool=c['random'][task['key'].split('|'+rep)[0]]['subsets'][rep]
        require(result['candidate_count']==len(pool),'Candidate count')
        require(rank_columns(x['train'],y['train'],pool)==result['ranking'],'ANOVA ranking '+name)
    output={}
    for size,f in result['selected'].items():
        cols=f['columns']; require(f['classes']==classes,'Classes')
        require(f['lam'] in c['probe']['lambdas'],'Lambda')
        mu=x['train'][:,cols].mean(0);sd=x['train'][:,cols].std(0)
        close(mu,f['mean'],name+'/mean');close(sd,f['std'],name+'/std'); require(np.all(sd>=1e-8),'Constant selected feature')
        removed=len(pool)-int(np.sum(x['train'][:,pool].std(0)>=1e-8));require(f['removed_constant_columns']==removed,'Constant feature count')
        if prefix:require(cols==result['ranking'][:len(cols)] and 1<=len(cols)<=4,'Prefix columns')
        else:require(cols==[i for i in pool if x['train'][:,i].std()>=1e-8],'Full columns')
        vp,logits=predict(f,x['val'])
        ce=float(np.mean(np.logaddexp(0,logits)-y['val']*logits)) if classes==2 else float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(logits)),y['val']]))
        close(ce,f['validation_ce'],name+'/validation CE')
        thresholds=c['probe']['thresholds'] if classes==2 else [None];choices=[]
        for t in thresholds:
            pred=(vp>=t).astype(int) if classes==2 else vp.argmax(1)
            ba=float(np.mean([np.mean(pred[y['val']==k]==k) for k in range(classes)]))
            choices.append((-ba,abs(t-.5) if t is not None else 0,t or 0,t))
        winner=min(choices);require(f['threshold']==winner[-1],'Selected threshold '+name);close(-winner[0],f['validation_balanced_accuracy'],'Validation BA')
        traces=[t for t in result['trace'] if size!='single' or len(t['columns'])==1]
        best=min(traces,key=lambda t:(-t['validation_balanced_accuracy'],len(t['columns']) if prefix else 0,t['validation_ce'],-t['lam'],abs((t['threshold'] or .5)-.5),t['threshold'] or 0))
        for k in ['columns','lam','threshold','validation_ce','validation_balanced_accuracy']:compare(f[k],best[k],name+'/trace/'+k)
        prob,_=predict(f,x['test']);evaluation=result['evaluation'][size]
        all_metrics=evaluate(y['test'],prob,classes,f['threshold'],groups['test'],task['bootstrap_seed'])
        compare(all_metrics,{k:v for k,v in evaluation.items() if k!='current_ne_previous'},name+'/'+size)
        different=np.array([r['previous'] is not None and r['current']!=r['previous'] for r in labels['test']])[masks['test']]
        if label in ('current','previous'):
            sub=evaluate(y['test'][different],prob[different],classes,f['threshold'],groups['test'][different],task['bootstrap_seed'])
            compare(sub,evaluation['current_ne_previous'],name+'/'+size+'/different');all_metrics['current_ne_previous']=sub
        output[size]=dict(probability_sha256=hashlib.sha256(np.asarray(prob,dtype='<f8').tobytes()).hexdigest(),metrics=all_metrics,
                          mean_max_abs_error=float(np.max(np.abs(mu-f['mean']))),std_max_abs_error=float(np.max(np.abs(sd-f['std']))))
    return output

def run(args):
    source,inputs,out,work=map(Path,[args.source,args.inputs,args.output,args.work])
    require(not out.resolve().is_relative_to(source.resolve()),'Audit output must be outside source')
    require(not work.resolve().is_relative_to(source.resolve()),'Work must be outside source')
    require(not out.resolve().is_relative_to(work.resolve()),'Output must survive scratch cleanup')
    c=read(inputs/'contract.json');ch=sha(inputs/'contract.json');manifest=read(inputs/'reference_manifest.json');inventory=manifest['files']
    require(manifest['contract_sha256']==ch,'Reference contract')
    audit_id=hashlib.sha256((sha(__file__)+sha(inputs/'reference_manifest.json')+json.dumps(POLICY,sort_keys=True)).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    session=dict(audit_id=audit_id,script_sha256=sha(__file__),policy=POLICY,python=sys.version,platform=platform.platform(),numpy=np.__version__,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    import scipy
    session['scipy']=scipy.__version__;session['pip_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    save(out/'sessions'/f'{time.time_ns()}.json',session)
    require(sha(source/'contract.json')==ch,'Source contract')
    labels=independent_labels(inputs,c)
    # Verify all frozen result/label/direction/stat/marker files before any numerical audit.
    selected={k:v for k,v in inventory.items() if k=='contract.json' or k.split('/')[0] in ('probes','labels','directions','preprocessing','cache')}
    for name,digest in selected.items():require(sha(source/name)==digest,'Source changed '+name)
    actual={p.stem for p in (source/'probes').glob('*.json')};require(actual==set(c['tasks']),'Task inventory')
    for s in labels:require(read(source/'labels'/f'{s}.json')==labels[s],'Independent corpus join '+s)
    def task_done(name):
        p=out/'tasks'/f'{name}.json'
        if not p.exists():return False
        r=read(p);require(r['audit_id']==audit_id and r['source_sha256']==inventory['probes/'+name+'.json'] and r['status']=='passed','Invalid prior receipt')
        return True
    def task_run(name,values):
        if task_done(name):return
        item=read(source/'probes'/f'{name}.json');t=c['tasks'][name]
        require(item['config_sha256']==ch and item['task_key']==t['key'],'Task identity')
        for k in ['bootstrap_seed','shuffle_seed']:require(item.get(k)==t.get(k),'Task seed')
        start=time.monotonic();v=verify_task(name,item,values,labels,c)
        save(out/'tasks'/f'{name}.json',dict(audit_id=audit_id,source_sha256=inventory['probes/'+name+'.json'],status='passed',seconds=time.monotonic()-start,checks=v))
        print('PASS',name,round(time.monotonic()-start,2),'seconds',flush=True)
    for seed in [0,1,2]:
        for kind in ['trained','init']:
            names=[n for n in c['tasks'] if n.startswith(f'seed{seed}_{kind}_')]
            cache_receipt=out/'cache'/f'seed{seed}_{kind}.json'
            if cache_receipt.exists() and all(task_done(n) for n in names):
                require(read(cache_receipt)['audit_id']==audit_id,'Stale cache receipt');continue
            arrays,receipts,target=stage_cache(source,work,seed,kind,c,ch,labels,inventory)
            # Full seed/kind stage revalidates original NPZ bytes even on partial resume.
            for layer in range(12):
                for hook in 'hum':
                    values={s:arrays[s][layer,hook] for s in labels};random_values=None
                    matching=[n for n in names if n.startswith(f'seed{seed}_{kind}_l{layer}_{hook}_')]
                    if kind=='trained' and layer==0:
                        train=np.asarray(values['train'],np.float64);mu=train.mean(0);scale=np.sqrt(np.mean((train-mu)**2))
                        st=read(source/'preprocessing'/f'seed{seed}_{hook}.json')
                        require(st['config_sha256']==ch and st['train_positions']==len(train) and st['accumulation']=='float64' and st['stored']=='float32','Scalar stats identity')
                        require(np.array_equal(mu.astype(np.float32),np.asarray(st['mean'],np.float32)) and np.float32(scale)==np.float32(st['scale']),'Dictionary scalar train stats')
                        if hook in 'hm':
                            key=c['tasks'][next(n for n in matching if '_random_current_iid' in n)]['key'].split('|random|')[0];spec=c['random'][key]
                            rng=np.random.Generator(np.random.PCG64(spec['projection_seed']));R=rng.normal(size=(256,512));R/=np.linalg.norm(R,axis=0)
                            directions=read(source/'directions'/f'seed{seed}_{hook}.json');close(R,directions['matrix'],'Random directions')
                            require(directions['subsets']==spec['subsets'] and directions['seed']==spec['projection_seed'] and directions['config_sha256']==ch,'Direction identity')
                            random_values={s:((np.asarray(a,np.float64)-mu.astype(np.float32))/np.float32(scale))@R for s,a in values.items()}
                        del train
                    for name in matching:task_run(name,random_values if '_random' in name else values)
            save(cache_receipt,dict(audit_id=audit_id,status='passed',files=receipts,scalar_statistics_checked=kind=='trained'))
            del arrays,values,random_values
            shutil.rmtree(target)
        token={}
        for s,rows in labels.items():
            pos=np.array([r['token_index']/767 for r in rows]);token[s]=np.column_stack([np.eye(15)[[r['token_id'] for r in rows]],pos,pos**2])
        for name in c['tasks']:
            if name.startswith(f'seed{seed}_token_position_'):task_run(name,token)
    require(all(task_done(n) for n in c['tasks']),'Missing task receipts')
    save(out/'completion.json',dict(status='passed_independent_numeric_audit',audit_id=audit_id,contract_sha256=ch,
        tasks=len(c['tasks']),cache_groups=6,policy=POLICY,p5_complete=False,
        limitations=['No LM inference or checkpoint reselection. Activation extraction is byte/row/tensor audited, not independently re-extracted.',
        'Saved selected coefficients are evaluated; optimizer fitting is not repeated. Unselected lambda/prefix coefficients were not saved, so their validation values are checked by frozen trace only.',
        'P5 completion requires local review of these returned audit receipts and phase records.']))
    print('Independent numerical audit complete; return evidence for review.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--work',required=True)
    a=p.parse_args()
    try:run(a)
    except Exception as e:
        dest=Path(a.output)/'failures'/f'{time.time_ns()}.json'
        save(dest,dict(status='failed',type=type(e).__name__,message=str(e),script_sha256=sha(__file__)))
        raise
