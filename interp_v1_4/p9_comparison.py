"""Compare dictionary initializations within LM seed 0, never align latent IDs."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .p5 import read,write
from .runtime import sha
from .p9_sae_aggregate import csvfile

CONTRACT='experiment_v1_4/p9_eval_r1/comparison.json'


def compare_rows(old,new,keys,metrics):
    def indexed(rows,seed):
        out={}
        for r in rows:
            if int(r['lm_seed'])!=0 or int(r['sparse_seed'])!=seed:continue
            key=tuple(str(r[k]) for k in keys)
            if key in out:raise ValueError('Duplicate comparison key')
            out[key]=r
        return out
    a,b=indexed(old,0),indexed(new,1)
    result=[]
    for key in sorted(a.keys()|b.keys()):
        for metric in metrics:
            x,y=a.get(key,{}).get(metric),b.get(key,{}).get(metric)
            available=x not in ('',None) and y not in ('',None)
            result.append(dict(zip(keys,key),lm_seed=0,statistic=metric,seed0=x,seed1=y,
                difference_seed1_minus_seed0=float(y)-float(x) if available else None,
                status='paired_point_estimates' if available else 'NA',
                inference='One LM, two dictionary initializations; no latent ID alignment; differences have no CI'))
    return result


def aggregate(root,output):
    root,output=Path(root),Path(output);config=read(root/CONTRACT)
    for name,digest in config['files'].items():
        if sha(root/name)!=digest:raise ValueError('Changed comparison input')
    allrows=[]
    specs=dict(semantic=(['layer','k','task','size'],['balanced_accuracy','macro_f1','binary_auroc']),
        fidelity=(['layer','k','split'],['mse','nmse','r2','ev','l0','inactive_fraction']),
        replacement=(['layer','k','metric'],['mean']),causal=(['layer','k','task','metric'],['mean']))
    for tool in ('sae','tc'):
        folder=output/tool;done=read(folder/'evaluation_complete.json')
        if done['runs']!=8 or done['config_sha256']!=sha(root/config['tool_contracts'][tool]):raise ValueError('Incomplete or stale evaluation')
        for name,digest in done['artifacts'].items():
            if sha(folder/name)!=digest:raise ValueError('Changed tool results')
        for kind,(keys,metrics) in specs.items():
            with (root/config['seed0_tables'][tool][kind]).open() as f:old=list(csv.DictReader(f))
            with (folder/'tables'/f'{kind}.csv').open() as f:new=list(csv.DictReader(f))
            # Causal/replacement already use metric as a grouping key.
            rows=compare_rows(old,new,keys,metrics)
            for r in rows:
                r.update(tool=tool,target='h reconstruction' if tool=='sae' else 'm prediction',table=kind)
            allrows.extend(rows)
    csvfile(output/'tables/sparse_seed_comparison.csv',allrows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for tool,ax in zip(('sae','tc'),axes):
        rep='sae' if tool=='sae' else 'transcoder'
        for k in (4,16):
            rows=[r for r in allrows if r['tool']==tool and r['table']=='semantic' and r['task']==rep+'_current_iid' and r['size']=='up_to_four' and int(r['k'])==k and r['statistic']=='balanced_accuracy']
            for seed,style in [(0,'--'),(1,'-')]:
                ordered=sorted(rows,key=lambda r:int(r['layer']))
                ax.plot([int(r['layer']) for r in ordered],[float(r[f'seed{seed}']) if r[f'seed{seed}'] not in ('',None) else np.nan for r in ordered],style,marker='o',label=f'k={k}, sparse seed={seed}')
        ax.set(title=tool.upper(),xlabel='Block',ylabel='Current-value balanced accuracy');ax.legend(fontsize=7)
    fig.suptitle('LM seed 0: dictionary initialization sensitivity (post-hoc probes)');fig.tight_layout()
    (output/'figures').mkdir(exist_ok=True);fig.savefig(output/'figures/sparse_seed_comparison.png',dpi=160);plt.close(fig)
    write(output/'comparison_complete.json',dict(status='comparison_return_audit_pending',p9_complete=False,runs=16,config_sha256=sha(root/CONTRACT),rows=len(allrows),interpretation='Dictionary initialization variation within LM seed 0; no cross-dictionary latent ID correspondence',artifacts={str(p.relative_to(output)):sha(p) for p in [output/'tables/sparse_seed_comparison.csv',output/'figures/sparse_seed_comparison.png']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args();aggregate(a.root,a.output)
