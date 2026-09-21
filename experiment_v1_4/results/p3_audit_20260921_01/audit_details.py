"""Recorded-evidence checks only: no model inference, no gate/test re-evaluation."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from interp_v1_4.runtime import sha
from interp_v1_4.training import learning_rate
from interp_v1_4.behavior import valid_pair_coverage
OUT=ROOT/'experiment_v1_4/results/p3_audit_20260921_01'
base=Path(json.loads((OUT/'archive_verification.json').read_text())['extracted_root'])
run=base/'deepwide12_read4_seed0'
r=json.loads((run/'result.json').read_text());smoke=json.loads((base/'gpu_smoke/smoke.json').read_text())
assert r['lm_seed']==0 and r['cell']=='deepwide12_read4'
assert r['hashes']['code']==smoke['input_hashes']['code']
assert r['environment']['environment_id']==smoke['environment']['environment_id']
assert json.loads((run/'gate.json').read_text())==r['gate']
assert json.loads((run/'gate_started.json').read_text())['selected']==r['selected']
assert not list(run.glob('interruptions/*'))
sessions=sorted((run/'sessions').glob('*/environment.json')); assert len(sessions)==2
session_records=[json.loads(p.read_text()) for p in sessions]
assert session_records[0]['resume']=='None'
assert session_records[1]['resume'].endswith('/checkpoints/update_004249.pt')
for p in sessions:
 s=json.loads(p.read_text());env=s['environment']
 assert sha(p.parent/'requirements.lock.txt')==env['lock_sha256']
 identity={k:env[k] for k in ('python','platform','torch','cuda','cudnn','gpu','lock_sha256')}
 assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]==env['environment_id']
 assert env['environment_id']==r['environment']['environment_id']

def finite(v):
 if isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
 elif isinstance(v,float):assert math.isfinite(v)
finite(r)
events=[json.loads(p.read_text()) for p in sorted((run/'events').glob('*.json'))]
for e in events:
 assert e['microbatch']==16
 assert e['environment_id']==r['environment']['environment_id']
 assert math.isclose(e['lr'],learning_rate(e['state']['prediction_tokens']),abs_tol=1e-14)
 assert e['loss_weight_sum']==e['prediction_tokens']+3*e['read_answer_targets']
 assert e['training_seconds']>0
 finite(e)
first=r['measurement_first_50_updates']
assert first['updates']==50
assert first['prediction_tokens']==sum(e['prediction_tokens'] for e in events[:50])
assert math.isclose(first['training_seconds'],sum(e['training_seconds'] for e in events[:50]),abs_tol=1e-8)
metrics=[]
for b,m in sorted(r['milestones'].items(),key=lambda x:int(x[0])):
 c=m['current'];v=c['validation'];assert v['general']['sequence_count']==512
 assert valid_pair_coverage(v['pairs'],64)
 metrics.append(dict(boundary=int(b),update=c['update'],general_accuracy=v['general']['accuracy'],first_macro_ce=v['pairs']['first']['cell']['macro_answer_ce'],first_macro_accuracy=v['pairs']['first']['cell']['macro_accuracy']))
for v in [m['current']['validation'] for m in r['milestones'].values()]+[r['gate']]:
 pair=v['pairs'];u=pair['uncertainty']
 assert u['draws']==u['valid_draws']==1000
 for member in ('first','repeat'):
  for cell in pair[member]['cell']['cells'].values():
   assert cell['count']==64
   for k in ('accuracy_ci95','answer_ce_ci95'):
    lo,hi=cell[k];assert 0<=lo<=hi
    if k=='accuracy_ci95':assert hi<=1
 for g in [v['general']]+list(v.get('diagnostics',{}).values()):
  assert g['prediction_tokens']>0 and g['all_token_ce']>=0
  assert g['baselines']['count']==g['answer_count']
  assert set(g['strata'])=={'operator','query_var','answer','blocks','distance','truth_table'}
all_training=sum(e['training_seconds'] for e in events)
final_events=events[4249:]
training=sum(e['training_seconds'] for e in final_events)
validation=sum(e.get('validation_seconds',0) for e in final_events)
serialization=sum(json.loads(p.read_text())['serialization_seconds'] for p in (run/'storage').glob('update_*.json') if int(p.stem.split('_')[1])>4249)
summary=dict(status='passed',scope='reported_environment_optimizer_schedule_metrics_and_timing',gate_reexecuted=False,test_scores_observed=False,
 checks=['same GPU smoke/runtime hashes and environment lock','two matching-environment sessions; resumed from complete update 4249','8399 finite updates with frozen read4 weights LR and microbatch','first 50 measurement reconciliation','10 select suites and all paired quotas','cell bootstrap CI and extended report presence','separate gate and started checkpoint binding'],
 sessions=session_records,
 timing=dict(all_recorded_training_seconds=all_training,final_session_resume_update=4249,session_seconds=r['session_seconds'],training_seconds=training,select_validation_seconds=validation,checkpoint_serialization_seconds=serialization,
 unallocated_seconds=r['session_seconds']-training-validation-serialization,
 unallocated_scope='Drive copy/checksum, local event writes, gate evaluation and orchestration; not separately timed. Final session only; unrecorded/recomputed work from the first session is not recoverable.'),
 milestones=metrics,selected_update=r['selected']['update'],gate=r['gate']['decision'])
(OUT/'details_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
