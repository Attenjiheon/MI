"""P9 SAE sparse seed 1; inherited p7_aggregate.py procedure with isolated contract."""
from pathlib import Path
import csv
import itertools
import numpy as np
from .p5 import read,write,derived,domains
from .p9_sae_evaluation import verify,CONTRACT
from .p7_metrics import paired_summary
from .runtime import sha


def csvfile(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    import io
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
    value=f.getvalue()
    if path.exists():
        if path.read_text()!=value:raise ValueError('Refusing to overwrite table')
    else:path.write_text(value)


def ba_difference(y,p1,p2,t1,t2,groups,seed):
    _,g=np.unique(groups,return_inverse=True);n=g.max()+1
    pred1=p1>=t1;pred2=p2>=t2
    count=np.zeros((n,2));correct1=np.zeros((n,2));correct2=np.zeros((n,2))
    np.add.at(count,(g,y),1);np.add.at(correct1,(g,y),pred1==y);np.add.at(correct2,(g,y),pred2==y)
    rng=np.random.Generator(np.random.PCG64(seed));values=[]
    for _ in range(1000):
        weights=np.bincount(rng.integers(n,size=n),minlength=n);den=weights@count
        if (den>0).all():values.append(float(((weights@(correct2-correct1))/den).mean()))
    mean=float(((correct2-correct1).sum(0)/count.sum(0)).mean()) if (count.sum(0)>0).all() else None
    return dict(mean=mean,ci95=np.percentile(values,[2.5,97.5]).tolist() if values else None,valid=len(values),requested=1000,seed=seed)


def aggregate(root,source,output):
    root,source,output=Path(root),Path(source),Path(output);config=verify(root);ch=sha(root/CONTRACT)
    semantic=[];fidelity=[];causal=[];replacement=[];by_run={}
    for e in config['saes']:
        r=e['run'];folder=output/'runs'/r['name'];s=read(folder/'semantic.json');p=read(folder/'summary.json')
        if s['config_sha256']!=ch or p['config_sha256']!=ch:raise ValueError('Mixed contracts')
        identity=dict(lm_seed=r['lm_seed'],layer=r['layer'],k=r['k'],sparse_seed=r['sparse_seed'])
        by_run[r['lm_seed'],r['layer'],r['k']]=(e,s,p,read(folder/'selection.json'))
        for task,result in s['semantic'].items():
            for size,m in result.get('evaluation',{}).items():
                semantic.append(dict(**identity,task=task,size=size,balanced_accuracy=m['balanced_accuracy'],macro_f1=m['macro_f1'],binary_auroc=m['binary_auroc'],status=result['status']))
        for split,m in s['fidelity'].items():
            if not isinstance(m,dict):continue
            fidelity.append(dict(**identity,split=split,mse=m['mse'],nmse=m['nmse'],r2=m['r2'],ev=m['ev'],l0=m['l0']['mean'],inactive_fraction=m['inactive_fraction']))
        for task,m in p['causal'].items():
            for metric,value in m['metrics'].items():causal.append(dict(**identity,task=task,metric=metric,origins=m['origins'],mean=value['mean'],ci_low=value['ci95'][0],ci_high=value['ci95'][1]))
        for metric,value in p['replacement']['metrics'].items():replacement.append(dict(**identity,metric=metric,mean=value['mean'],ci_low=value['ci95'][0],ci_high=value['ci95'][1]))
    for _,_,_,selection in by_run.values():
        if any(f['status']=='failed' for f in selection['fits'].values()):raise ValueError('Unresolved probe fitting failure')
    gaps=[]
    for row in semantic:
        if row['task'].startswith('full_'):continue
        _,label,domain=row['task'].rsplit('_',2)
        baseline=next((b for b in semantic if all(b[k]==row[k] for k in ('lm_seed','layer','k','sparse_seed')) and b['task']==f'full_{label}_{domain}' and b['size']=='full'),None)
        if baseline is not None:
            gaps.append({**row,'full_balanced_accuracy':baseline['balanced_accuracy'],'difference_from_full':row['balanced_accuracy']-baseline['balanced_accuracy'] if row['balanced_accuracy'] is not None and baseline['balanced_accuracy'] is not None else None})
    csvfile(output/'tables/full_probe_gaps.csv',gaps)
    for name,rows in [('semantic',semantic),('fidelity',fidelity),('causal',causal),('replacement',replacement)]:csvfile(output/'tables'/f'{name}.csv',rows)
    differences={};labels=read(source/'labels/test.json');groups=np.asarray([r['sequence_id'] for r in labels])
    y=np.asarray([r['current'] for r in labels],int)
    for seed,k in itertools.product((0,),(4,16)):
        for a,b in itertools.combinations((0,3,7,11),2):
            ea,sa,pa,fa=by_run[seed,a,k];eb,sb,pb,fb=by_run[seed,b,k]
            folders=[output/'runs'/e['run']['name'] for e in (ea,eb)]
            with np.load(folders[0]/'semantic_predictions.npz') as za,np.load(folders[1]/'semantic_predictions.npz') as zb:
                for rep,limit,domain in itertools.product(('coordinate','random','sae'),('','_128'),('iid','transfer')):
                    task=f'{rep}{limit}_current_{domain}';mask=domains(labels,'current',domain,'test')
                    for size in ('single','up_to_four'):
                        key=f'seed{seed}|k{k}|layers{a}-{b}|{task}|{size}'
                        if task+'_'+size not in za or task+'_'+size not in zb:continue
                        ta=fa['fits'][task]['selected'][size]['threshold'];tb=fb['fits'][task]['selected'][size]['threshold']
                        bs=derived('bootstrap',config['bootstrap_keys'][ea['run']['name']]+'|layer_difference|'+task+'|'+size)
                        differences[key]=ba_difference(y[mask],za[task+'_'+size],zb[task+'_'+size],ta,tb,groups[mask],bs)
            # Causal differences use identical origin intersection, averaging candidates/directions first.
            raws=[]
            for folder in folders:
                rows=[]
                for path in (folder/'causal').rglob('*.json'):rows.extend(read(path)['rows'])
                mapping={}
                for r in rows:
                    control=r['control']
                    if control.startswith('random_') and control[7:].isdigit():continue
                    key=(r['changed'],r['condition'],r['size'],control,r['origin'])
                    mapping.setdefault(key,[]).append(r)
                raws.append(mapping)
            grouped={}
            for key in sorted(raws[0].keys()&raws[1].keys()):
                changed,condition,size,control,origin=key
                row=dict(origin=origin,condition=condition)
                for metric in ('delta_margin','full_flip','binary_flip','error_induced','prediction_changed','patch_norm'):
                    row[metric]=float(np.mean([r[metric] for r in raws[1][key]])-np.mean([r[metric] for r in raws[0][key]]))
                grouped.setdefault((changed,condition,size,control),[]).append(row)
            for group,rows in grouped.items():
                name=f'seed{seed}|k{k}|layers{a}-{b}|causal|{group}'
                bs=derived('bootstrap',config['bootstrap_keys'][ea['run']['name']]+'|layer_difference|'+str(group))
                differences[name]=paired_summary(rows,['delta_margin','full_flip','binary_flip','error_induced','prediction_changed','patch_norm'],bs)
    write(output/'tables/layer_differences.json',dict(direction='higher layer minus lower layer',causal_subset='shared origins; matched controls use intersection of matching success origins',results=differences))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k in (4,16):
        for rep in ('full','coordinate','random','sae'):
            values=[]
            for layer in (0,3,7,11):
                values.append(np.mean([r['balanced_accuracy'] for r in semantic if r['layer']==layer and r['k']==k and r['task']==rep+'_current_iid' and r['size']==('full' if rep=='full' else 'up_to_four') and r['balanced_accuracy'] is not None]))
            axes[0].plot([0,3,7,11],values,marker='o',label=f'{rep} k={k}')
        current=[r for r in fidelity if r['k']==k and r['split']=='test']
        axes[1].scatter([r['l0'] for r in current],[r['nmse'] for r in current],label=f'k={k}')
    axes[0].set(xlabel='Block',ylabel='Current-value balanced accuracy');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Mean positive L0',ylabel='SAE test NMSE');axes[1].legend();fig.tight_layout();fig.savefig(figures/'semantic_fidelity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k in (4,16):
        for changed,metric,ax in [('True','delta_margin',axes[0]),('False','error_induced',axes[1])]:
            for control in ('selected','full_donor','random_unmatched','random_matched','coordinate','random_direction_selected'):
                values=[]
                for layer in (0,3,7,11):
                    task=f'{changed}|all|up_to_four|all|{control}'
                    rs=[r['mean'] for r in causal if r['layer']==layer and r['k']==k and r['task']==task and r['metric']==metric]
                    values.append(np.mean(rs) if rs else np.nan)
                ax.plot([0,3,7,11],values,marker='o',label=f'{control} k={k}')
    axes[0].set(xlabel='Block',ylabel='Changed delta margin');axes[0].legend(fontsize=6)
    axes[1].set(xlabel='Block',ylabel='Unchanged error induced');fig.tight_layout();fig.savefig(figures/'causal.png',dpi=160);plt.close(fig)
    # Keep every seed visible; layers are not independent model replicates.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k in (4,16):
        for metric,ax in [('delta_ce',axes[0]),('delta_accuracy',axes[1])]:
            rs=[r for r in replacement if r['k']==k and r['metric']==metric]
            ax.scatter([r['layer'] for r in rs],[r['mean'] for r in rs],label=f'k={k}')
            ax.set(xlabel='Block',ylabel=metric);ax.legend()
    fig.suptitle('SAE h reconstruction: paired single READ replacement');fig.tight_layout();fig.savefig(figures/'replacement.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    coverage_rows=[]
    for (seed,layer,k),(e,s,p,f) in by_run.items():
        for suite,cov in p['coverage'].items():
            coverage_rows.append(dict(lm_seed=seed,layer=layer,k=k,suite=suite,**cov,matched_fraction=cov['matched']/cov['pairs'] if cov['pairs'] else None))
    csvfile(output/'tables/matching_coverage.csv',coverage_rows)
    for k in (4,16):
        for size,ax in [('single',axes[0]),('up_to_four',axes[1])]:
            rs=[r for r in coverage_rows if r['k']==k and r['suite'].endswith('|'+size)]
            ax.scatter([r['layer'] for r in rs],[r['matched_fraction'] for r in rs],label=f'k={k}',alpha=.5)
            ax.set(xlabel='Block',ylabel='Matched pair fraction',title=size,ylim=(-.02,1.02));ax.legend()
    fig.tight_layout();fig.savefig(figures/'matching_coverage.png',dpi=160);plt.close(fig)
    for limit in ('','_128'):
        fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True)
        for label,ax in zip(('current','previous','query','A','B','C','D','state'),axes.flat):
            for rep in ('coordinate','random','sae'):
                for size,style in [('single','--'),('up_to_four','-')]:
                    rs=[r for r in semantic if r['task']==rep+limit+'_'+label+'_iid' and r['size']==size and r['balanced_accuracy'] is not None]
                    values=[np.mean([r['balanced_accuracy'] for r in rs if r['layer']==l]) if any(r['layer']==l for r in rs) else np.nan for l in (0,3,7,11)]
                    ax.plot([0,3,7,11],values,style,label=rep+' '+size)
            ax.set(title=label,xlabel='Block',ylabel='Balanced accuracy')
        axes.flat[0].legend(fontsize=6);fig.suptitle('Post-hoc supervised accessibility: '+('128 candidates' if limit else 'all candidates'));fig.tight_layout();fig.savefig(figures/('all_labels'+limit+'.png'),dpi=160);plt.close(fig)
    write(output/'evaluation_complete.json',dict(status='all_evaluations_return_audit_pending',p9_complete=False,runs=8,config_sha256=ch,artifacts={str(p.relative_to(output)):sha(p) for folder in ('tables','figures') for p in sorted((output/folder).glob('*')) if p.is_file()}))
