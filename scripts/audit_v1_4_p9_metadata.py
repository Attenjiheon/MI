"""Independent trace, support, bin, provenance and report-point checks for P8."""
from pathlib import Path
import sys,json,collections,time,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write,load_split,LABELS,domains
from interp_v1_4.runtime import sha
from interp_v1_4.p9_sae_evaluation import records
from scripts.audit_v1_4_p7_sources import compare


def audit(returned, folder, names=None, tool="tc", root=ROOT):
 from interp_v1_4 import p9_sae_evaluation, p9_tc_evaluation
 ev=p9_sae_evaluation if tool=="sae" else p9_tc_evaluation
 verify,CONTRACT,ROOT=ev.verify,ev.CONTRACT,Path(root)
 full_reps=("full_",) if tool=="sae" else ("fullu_","fullm_")
 task_count=70 if tool=="sae" else 80
 returned,folder=Path(returned),Path(folder);c=verify(ROOT);ch=sha(ROOT/CONTRACT)
 assert sha(returned/'contract.json')==ch
 p5=read(ROOT/'experiment_v1_4/p5_r2/contract.json')
 for split in ('train','val','test'):
  _,rows=load_split(ROOT,p5,split);assert rows==read(returned/'labels'/f'{split}.json')
 envs=[]
 for path in returned.glob('**/environment.json'):
  info=read(path);assert info['cuda'] and info['gpu'] and info['torch']=='2.11.0+cu128'
  import hashlib
  assert hashlib.sha256(__import__('json').dumps({k:info[k] for k in ('python','platform','torch','cuda','cudnn','gpu','lock_sha256')},sort_keys=True).encode()).hexdigest()[:16]==info['environment_id']
  assert sha(path.parent/'requirements.lock.txt')==info['lock_sha256'];envs.append(info['environment_id'])
 smoke=read(next(returned.glob('smoke/*/smoke.json')))
 assert smoke['environment_id'] in envs
 assert smoke['device']=='cuda' and smoke['config_sha256']==ch and len(smoke['layers'])==8 and smoke['status']=='passed_debug_only'
 pairs={p['pair_id']:p for suite in ('test_changed_memory','test_changed_composition','test_unchanged_memory','test_unchanged_composition') for p in records(ROOT/c['data_root']/'causal_pairs'/(suite+'.jsonl.gz'))}
 reports=[]
 for entry in c['saes' if tool=='sae' else 'tcs']:
  if names and entry['run']['name'] not in names:continue
  name=entry['run']['name'];runfolder=returned/'runs'/name;s=read(runfolder/'selection.json');m=read(runfolder/'matching.json');sem=read(runfolder/'semantic.json');summary=read(runfolder/'summary.json')
  assert len(s['fits'])==task_count
  for task,fit in s['fits'].items():
   assert fit['status']=='passed'
   prefix=not task.startswith(full_reps);width=128 if '_128_' in task else 256 if task.startswith(full_reps+('coordinate_',)) else 512
   assert fit['candidate_count']==width
   trace=fit['trace'];assert len(trace)==(16 if prefix else 4)
   assert all(t['attempts'][-1]['success'] for t in trace)
   assert {t['lam'] for t in trace}=={.01,.1,1.,10.}
   def key(t):
    return (-t['validation_balanced_accuracy'],len(t['columns']) if prefix else 0,t['validation_ce'],-t['lam'],abs((t['threshold'] or .5)-.5),t['threshold'] or 0)
   for size,model in fit['selected'].items():
    eligible=[t for t in trace if size!='single' or len(t['columns'])==1];winner=min(eligible,key=key)
    for k in ('columns','lam','threshold','validation_balanced_accuracy','validation_ce'):assert model[k]==winner[k],(name,task,size,k)
    if prefix:
     assert model['columns']==fit['ranking'][:len(model['columns'])]
     if '_128_' in task:assert set(model['columns'])<=set(s['subsets'][task.split('_')[0]])
    assert np.isfinite(model['coefficients']).all() and min(model['std'])>=1e-8
   reuse=c['reused_probes'].get(f"seed{entry['run']['lm_seed']}_l{entry['run']['layer']}_{task}")
   if reuse:
    original=read(ROOT/reuse)['result'];compare(fit,{k:v for k,v in original.items() if k not in ('evaluation','test_support')},'reuse/'+task)
  for size,rule in m['rules'].items():
   h,z=np.array(rule['validation_h_norms']),np.array(rule['validation_z_norms']);mask=h>0
   assert len(h)==len(z)==2048
   expected=dict(h_edges=np.unique(np.quantile(h[mask],[.2,.4,.6,.8])).tolist() if mask.any() else [],z_edges=np.unique(np.quantile(z[mask],[.2,.4,.6,.8])).tolist() if mask.any() else [],fitted_pairs=len(h),zero_count=int((~mask).sum()))
   compare(rule['rule'],expected,'bin/'+name+'/'+size)
  for split,quota in [('train',50000),('val',10000),('test',20000)]:
   f=sem['fidelity'][split];rates=np.array(f['activation_rates']);assert f['positions']==quota and len(rates)==512
   assert f['inactive_count']==int((rates==0).sum()) and f['inactive_fraction']==float((rates==0).mean())
   np.testing.assert_allclose(sum(rates),f['l0']['mean'],atol=1e-10)
   assert f['l0']['quantiles'][-1]<=entry['run']['k']
  # Recompute all reported point estimates with origin clustering and condition macros.
  grouped={};coverage={};pair_count=0;row_count=0
  metrics=('delta_margin','full_flip','binary_flip','error_induced','prediction_changed','patch_norm')
  for suite in ('test_changed_memory','test_changed_composition','test_unchanged_memory','test_unchanged_composition'):
   paths=sorted((runfolder/'causal'/suite).glob('*.json'));assert len(paths)==512
   for path in paths:
    item=read(path);pair_count+=1
    for size,record in item['candidates'].items():
     key=suite+'|'+size;cov=coverage.setdefault(key,dict(pairs=0,matched=0,unmatched=0,NA=0));cov['pairs']+=1
     cov['matched']+=int(bool(record['matched_indices']));cov['unmatched']+=int(not record['matched_indices']);cov['NA']+=int(record['features'] is None)
    rows=item['rows'];row_count+=len(rows)
    # Both-correct must use original outputs in both directions, not patched outputs.
    baseline={r['direction']:r for r in rows if r['control']=='identity' and r['size']=='single'}
    source_pair=pairs[item['pair_id']]
    answers={0:source_pair['original_answer'],1:source_pair['counterfactual_answer']}
    both=all(np.argmax(r['original_logits'])==13+answers[direction] for direction,r in baseline.items())
    for r in rows:
     assert r['both_correct']==both
     assert r['original_correct']==bool(np.argmax(r['original_logits'])==13+answers[r['direction']])
     assert r['original_logits']==baseline[r['direction']]['original_logits']
     control=r['control']
     if control.startswith('random_') and control[7:].isdigit():continue
     values=np.array([r[k] for k in metrics])
     for condition in (r['condition'],'all'):
      for subset in ('all',)+(('both_correct',) if both else ())+ (('original_correct',) if r['original_correct'] else ()):
       key=f"{r['changed']}|{condition}|{r['size']}|{subset}|{control}"
       origin=grouped.setdefault(key,{}).setdefault(r['origin'],[np.zeros(6),0,r['condition']]);origin[0]+=values;origin[1]+=1
  assert summary['coverage']==coverage
  for key,saved in summary['causal'].items():
   origins=grouped.get(key,{});assert saved['origins']==len(origins) and saved['rows']==sum(v[1] for v in origins.values())
   if not origins:assert saved['metrics']=={};continue
   conditions=sorted({v[2] for v in origins.values()})
   point=np.mean([np.mean([v[0]/v[1] for v in origins.values() if v[2]==condition],axis=0) for condition in conditions],axis=0)
   for metric,value in zip(metrics,point):np.testing.assert_allclose(saved['metrics'][metric]['mean'],value,atol=1e-12,rtol=1e-12)
  record=dict(run=name,selection_tasks=task_count,paired_causal_units=pair_count,causal_rows=row_count,summary_point_estimates=sum(len(x['metrics']) for x in summary['causal'].values()),status='passed')
  write(folder/'metadata_runs_r2'/(name+'.json'),record);reports.append(record);print('metadata and independent report points passed',name,flush=True)
 return dict(status='passed',p9_complete=False,runs=reports,gpu_environment_id=envs[0],checks=['corpus labels and READ key equality','GPU smoke and software lock','all validation trace winners and prefix rules','reused P5 coefficients','validation percentile bins','fidelity internal identities','original-correct and both-correct subsets independently derived from full-vocabulary logits','all causal coverage and origin-macro point estimates'])


