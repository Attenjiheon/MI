import csv,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
from interp_v1_4.runtime import sha
out=Path('experiment_v1_4/results/p3_audit_20260921_01')
r=json.loads((out/'result.json').read_text());d=json.loads((out/'details_verification.json').read_text())
assert json.loads((out/'run_verification.json').read_text())['status']=='passed'
assert json.loads((out/'gpu_smoke/verification.json').read_text())['status']=='passed_return_evidence_verification'
assert d['status']=='passed' and r['status']=='passed'
archive=json.loads((out/'archive_verification.json').read_text())
frozen=dict(schema='v1.4-p3-validated-seed0-freeze',lm_seed=0,candidate=r['cell'],status='passed',
 actual_prediction_tokens=r['actual_prediction_tokens'],nominal_budget=r['nominal_budget'],overshoot=r['overshoot'],
 final_state=r['final_state'],selected_checkpoint=r['selected']['checkpoint'],selected_checkpoint_sha256=r['selected']['sha256'],
 selected_update=r['selected']['update'],selected_prediction_tokens=r['selected']['prediction_tokens'],
 hashes=r['hashes'],environment_id=r['environment']['environment_id'],microbatch=16,effective_batch=64,
 archive_sha256=archive['archive_sha256'],replication='seeds 1 and 2 eligible; not run',
 test='not evaluated',interpretation='blocked until at least two passing LM seeds')
(out/'frozen_seed0.json').write_text(json.dumps(frozen,indent=2)+'\n')
with (out/'select_curve.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(d['milestones'][0]),lineterminator='\n');w.writeheader();w.writerows(d['milestones'])
g=r['gate'];rows=[dict(metric='general_read',accuracy=g['general']['accuracy'],count=g['general']['answer_count'],threshold=.99)]
rows += [dict(metric='legacy_'+k,accuracy=v['accuracy'],count=v['answer_count'],threshold=.95) for k,v in g['diagnostics'].items()]
for m in ('first','repeat'):
 rows.append(dict(metric=m+'_42_cell_macro',accuracy=g['pairs'][m]['cell']['macro_accuracy'],count=g['pairs']['pair_count'],threshold=.95))
for group in ('operator','depth','answer'):
 rows += [dict(metric='first_'+group+'_'+k,accuracy=v['accuracy'],count=v['count'],threshold=.95) for k,v in g['pairs']['first'][group]['cells'].items()]
with (out/'gate_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
completion=dict(status='passed',phase='v1.4-P3',lm_seed=0,gate_passed=True,completed_at=datetime.now(timezone.utc).isoformat(),
 evidence_sha256={n:sha(out/n) for n in ('archive_verification.json','run_verification.json','details_verification.json','gpu_smoke/verification.json','frozen_seed0.json','select_curve.csv','gate_summary.csv')},
 gate_reexecuted=False,test_scores_observed=False,next_phase='P4 seed 1 and 2 replication; not started',
 limitation='Recorded CUDA evidence and checkpoint/state integrity verified locally; no independent production GPU replay. Missing timing decomposition cannot identify the full cause of runtime overhead.')
(out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
registry=Path('experiment_v1_4/results/run_registry.csv')
with registry.open() as f: fields=next(csv.reader(f))
row=dict(phase='P3',run_id='seed0_64m_return_20260921',status='passed',scope='seed0_training_selection_one_time_behavior_gate_verified',environment_id=r['environment']['environment_id'],seed='0',
 config_sha256=r['hashes']['configs'],code_sha256=hashlib.sha256(json.dumps(r['hashes']['code'],sort_keys=True).encode()).hexdigest(),input_sha256=r['hashes']['corpus_manifest'],actual_tokens=r['actual_prediction_tokens'],
 elapsed_seconds=r['session_seconds'],peak_memory_bytes=r['measurement_first_50_updates']['peak_memory_bytes'],evidence_paths=str(out/'completion.json'),
 failure_or_skip_reason='elapsed_seconds is final resumed session only; test and replication not run',next_action='P4 seed 1 and 2 at same frozen budget and data order',actual_command='scripts/verify_v1_4_evidence.py RUN --output run_verification.json; see audit REPORT.md')
with registry.open('a',newline='') as f:csv.DictWriter(f,fieldnames=fields,lineterminator='\n').writerow(row)
