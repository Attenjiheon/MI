"""Build readable figures and final narrative from the P11 reporting tables."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
COLORS={'full':'#242b35','full_u':'#9a6700','full_m':'#242b35','coordinate':'#b45724','random':'#6f65a2','sae':'#087e8b','tc':'#226ec2'}


def render(root):
 root=Path(root);figdir=root/'figures';figdir.mkdir(exist_ok=True)
 s=pd.read_csv(root/'semantic_metrics.csv',low_memory=False);f=pd.read_csv(root/'fidelity_metrics.csv');c=pd.read_csv(root/'causal_metrics.csv');b=pd.read_csv(root/'behavior.csv');reg=pd.read_csv(root/'dictionary_registry.csv')
 for frame in (s,f,c,b):frame['value']=pd.to_numeric(frame['value'],errors='coerce')
 def save(fig,name):
  fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 def line(ax,df,label,color=None):
  if df.empty:return
  for seed,g in df.groupby('lm_seed'):
   ax.scatter(g.layer,g.value,s=13,color=color,alpha=.45)
  g=df.groupby('layer').value.agg(['mean','min','max'])
  ax.plot(g.index,g['mean'],marker='o',markersize=3,label=label,color=color)
  ax.fill_between(g.index,g['min'],g['max'],alpha=.07,color=color)
  ax.set_xticks([0,3,7,11]);ax.set_xlabel('Block')
 # Behavior: actual stored validation learning curve and frozen test.
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for seed in (0,1,2):
  for metric,ax in [('first_accuracy',axes[0]),('first_ce',axes[1])]:
   q=b[(b.category=='selection_curve')&(b.lm_seed==seed)&(b.metric==metric)].sort_values('budget_m');ax.plot(q.budget_m,q.value,marker='.',label=f'LM {seed}')
  q=b[(b.category=='frozen_test')&(b.lm_seed==seed)&(b.metric=='accuracy')&b.stratum.isin(['overall','42-cell macro'])]
  suites=['general','first','repeat','composition_0','composition_1'];q=q.set_index('suite').reindex(suites)
  axes[2].plot(range(5),q.value,marker='.',label=f'LM {seed}')
 axes[0].set(xlabel='Training tokens (millions)',ylabel='Select first accuracy');axes[1].set(xlabel='Training tokens (millions)',ylabel='Select first answer CE',yscale='log')
 axes[2].set(xticks=range(5),xticklabels=['IID','First','Repeat','Comp 0','Comp 1'],ylabel='Frozen test accuracy');axes[2].tick_params(axis='x',rotation=25)
 for ax in axes:ax.legend()
 save(fig,'01_behavior')
 main=s[(s.phase.isin(['P7','P8']))&(s.metric=='balanced_accuracy')&(s.label=='current')]
 for subset,domain,name in [('all','iid','02_semantic_iid'),('current_ne_previous','iid','03_current_not_previous'),('all','transfer','04_query_transfer')]:
  fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True)
  for row,tool in enumerate(('sae','tc')):
   for col,(k,size) in enumerate(((4,'single'),(4,'up_to_four'),(16,'single'),(16,'up_to_four'))):
    ax=axes[row,col];q=main[(main.tool==tool)&(main.k==k)&(main.subset==subset)&(main.domain==domain)]
    for rep in (['full','coordinate','random','sae'] if tool=='sae' else ['full_u','full_m','coordinate','random','tc']):
     line(ax,q[(q.representation==rep)&(q['size']==('full' if rep.startswith('full') else size))],rep,COLORS[rep])
    ax.set(title=f'{tool.upper()} | k={k} | {size}',ylabel='Balanced accuracy',ylim=(0.3,1.025));ax.legend(fontsize=7)
  fig.suptitle(f'Current value: {domain}; {subset}. Dots: LM seeds, lines/ranges: 3-LM mean/min/max',y=1.02);save(fig,name)
 fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
 for row,tool in enumerate(('sae','tc')):
  for col,k in enumerate((4,16)):
   ax=axes[row,col];q=main[(main.tool==tool)&(main.k==k)&(main.subset=='all')&(main.domain=='iid')&(main['size']=='up_to_four')]
   for rep in ('coordinate_128','random_128',tool+'_128'):
    line(ax,q[q.representation==rep],rep,COLORS.get(rep.replace('_128','')))
   ax.set(title=f'{tool.upper()} k={k}: equal 128 candidates',ylabel='Balanced accuracy',ylim=(.45,1.025));ax.legend()
 save(fig,'05_equal_candidate_control')
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 q=s[(s.phase=='P5')&(s.metric=='balanced_accuracy')&(s.label=='current')&(s.domain=='iid')&(s.subset=='all')&(s['size']=='full')]
 for ax,hook in zip(axes,['h','u','m']):
  for state,color in [('trained','#087e8b'),('init','#b45724')]:
   line(ax,q[(q.hook==hook)&(q.model_state==state)&(q.representation=='full')],state,color)
  ax.set(title=f'All-layer full {hook} probe',ylabel='Balanced accuracy');ax.set_xticks(range(12));ax.legend()
 save(fig,'06_full_probe_diagnostic')
 fig,axes=plt.subplots(2,3,figsize=(14,7))
 for row,tool in enumerate(('sae','tc')):
  q=f[(f.position_type=='READ')&(f.tool==tool)&(f.sparse_seed==0)]
  for k,color in [(4,'#087e8b'),(16,'#b45724')]:
   nm=q[(q.k==k)&(q.split=='test')&(q.metric=='nmse')];l0=q[(q.k==k)&(q.split=='test')&(q.metric=='l0_mean')]
   a=nm.merge(l0,on='name',suffixes=('_nm','_l0'));axes[row,0].scatter(a.value_l0,a.value_nm,label=f'k={k}',color=color)
   for col,metric in [(1,'delta_ce'),(2,'delta_accuracy')]:line(axes[row,col],q[(q.k==k)&(q.metric==metric)],f'k={k}',color)
  axes[row,0].set(xlabel='Mean positive L0',ylabel='Test NMSE',yscale='log',title=f'{tool.upper()}: '+('h reconstruction' if tool=='sae' else 'm prediction from u'))
  axes[row,1].set(title='One READ replacement: answer CE',ylabel='Patched minus original CE');axes[row,2].set(title='One READ replacement: accuracy',ylabel='Patched minus original accuracy')
  for ax in axes[row]:ax.legend()
 save(fig,'07_fidelity_replacement')
 controls=['selected','selected_matched','full_donor','random_matched','random_unmatched','coordinate','random_direction_selected','random_direction_pure','identity','mean','approximation']
 fig,axes=plt.subplots(2,4,figsize=(17,8))
 for row,tool in enumerate(('sae','tc')):
  for col,(changed,metric,k) in enumerate([('True','delta_margin',4),('True','full_flip',4),('True','delta_margin',16),('False','error_induced',16)]):
   ax=axes[row,col];q=c[(c.tool==tool)&(c.sparse_seed==0)&(c.k==k)&(c.changed.astype(str)==changed)&(c.metric==metric)&(c.condition=='all')&(c.subset=='all')&(c['size']=='up_to_four')]
   for ctl in controls:line(ax,q[q.control==ctl],ctl)
   ax.set(title=f'{tool.upper()} k={k} {changed=}',ylabel=metric)
  axes[row,0].legend(fontsize=6,loc='best')
 fig.suptitle('One-layer interventions; both directions averaged within origin; all pairs',y=1.01);save(fig,'08_causal_controls')
 cov=pd.read_csv(root/'matching_coverage.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,tool in zip(axes,('sae','tc')):
  for k in (4,16):
   q=cov[(cov.tool==tool)&(cov.sparse_seed==0)&(cov.k==k)];ax.scatter(q.layer,q.matched_fraction,label=f'k={k}',alpha=.35,s=20)
  ax.set(title=tool.upper(),xlabel='Block',ylabel='Matched origin fraction',xticks=[0,3,7,11],ylim=(-.02,1.02));ax.legend()
 save(fig,'09_matching_coverage')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,tool in zip(axes,('sae','tc')):
  q=s[(s.tool==tool)&(s.phase.isin(['P7','P8','P9']))&(s.metric=='balanced_accuracy')&(s.representation==tool)&(s.label=='current')&(s.domain=='iid')&(s.subset=='all')&(s['size']=='up_to_four')&(s.lm_seed==0)]
  for (k,sp),v in q.groupby(['k','sparse_seed']):ax.plot(v.sort_values('layer').layer,v.sort_values('layer').value,marker='o',label=f'k={int(k)}, sparse {int(sp)}')
  ax.set(title=f'{tool.upper()}: LM seed 0 only',xlabel='Block',ylabel='Current balanced accuracy',xticks=[0,3,7,11]);ax.legend()
 save(fig,'10_sparse_initialization')
 ld=pd.read_csv(root/'layer_differences.csv',low_memory=False);fig,axes=plt.subplots(1,2,figsize=(12,5))
 for ax,tool in zip(axes,('sae','tc')):
  q=ld[(ld.tool==tool)&(ld.k==16)&(ld.sparse_seed==0)&(ld.metric=='balanced_accuracy')&(ld.task==f'{tool}_current_iid|up_to_four')]
  for idx,(_,r) in enumerate(q.iterrows()):
   ax.plot([r.ci_low,r.ci_high],[idx,idx],color=COLORS[tool]);ax.scatter([r.value],[idx],s=14,color=COLORS[tool])
  ax.set_yticks(range(len(q)),[f'LM {r.lm_seed}: {r.layer_high}−{r.layer_low}' for r in q.itertuples()]);ax.axvline(0,color='grey',lw=.7);ax.set(title=f'{tool.upper()} k=16, ≤4, current IID',xlabel='Paired BA difference (95% cluster CI)')
 save(fig,'11_paired_layer_intervals')
 uq=s[(s.phase=='P10')&(s.metric=='balanced_accuracy')&(s.subset=='all')]
 fig,axes=plt.subplots(2,3,figsize=(16,8),sharey=True)
 for row,hook in enumerate(('h','m')):
  for col,domain in enumerate(('all','variable','operator')):
   ax=axes[row,col];q=uq[(uq.domain==domain)&(uq.hook==hook)&(uq.candidates!=128)&uq.representation.isin(['full','latent','coordinate','random'])&uq['size'].isin(['full','up_to_four'])].copy()
   q['series']=[f'latent k={int(k)}' if rep=='latent' else rep for rep,k in zip(q.representation,q.k)]
   g=q.groupby(['label','series']).value.mean().unstack();g.plot.bar(ax=ax)
   ax.set(title=f'Update {hook}: {domain}',ylabel='Balanced accuracy',ylim=(0,1.05));ax.tick_params(axis='x',rotation=30);ax.legend(fontsize=6)
 fig.suptitle('Exploratory Update probe, block 3; separate h and m targets; LM means',y=1.02);save(fig,'12_update')
 fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True)
 for ax,label in zip(axes.flat,('current','previous','query','A','B','C','D','state')):
  for tool in ('sae','tc'):
   q=s[(s.phase.isin(['P7','P8']))&(s.metric=='balanced_accuracy')&(s.representation==tool)&(s.label==label)&(s.domain=='iid')&(s.subset=='all')&(s['size']=='up_to_four')&(s.k==16)]
   line(ax,q,tool,COLORS[tool])
  ax.set(title=label,ylabel='Balanced accuracy',ylim=(0,1.025));ax.legend()
 fig.suptitle('Auxiliary labels: supervised ≤4 features, k=16; all results retained in tables',y=1.02);save(fig,'13_all_labels')
 return dict(figures=len(list(figdir.glob('*.png'))),semantic=s,fidelity=f,causal=c,behavior=b,registry=reg)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();result=render(args.output);print('Rendered',result['figures'],'figures')
