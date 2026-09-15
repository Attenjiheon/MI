"""Read-only P3 evidence audit; writes a separate verification report."""
import json,math,time,hashlib,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from interp.runtime import sha,verify_inputs,deterministic
from interp.data import token_rows
from interp.cli import evaluate_gate_suite,metric_summary
from interp.model import Transformer

base=ROOT/'experiment_v1/evidence/p3_colab_20260915T174256751247'
run=base/'lm_seed_0'; start=time.time()
load=lambda p:json.loads(p.read_text())
manifest=load(run/'manifest.json'); decision=load(run/'gate_decision.json')
inputs=verify_inputs(ROOT)
assert inputs=={k:manifest['input_hashes'][k] for k in inputs}
for name,digest in manifest['input_hashes']['code'].items(): assert sha(ROOT/name)==digest,name
bundle=load(base/'p3_bundle_manifest.json')
for name,digest in manifest['input_hashes']['code'].items():assert bundle[name]==digest
assert sha(run/'best.pt')==decision['selected_checkpoint_sha256']==manifest['checkpoint_sha256']
assert sha(run/'requirements.lock.txt')==manifest['environment']['lock_sha256']
smoke=load(next((base/'preflight').glob('*/smoke/smoke.json')))
assert smoke['status']=='passed' and smoke['debug_only']
assert smoke['input_hashes']['code']==manifest['input_hashes']['code']
assert smoke['environment']['environment_id']==manifest['environment_id']
rows=list(token_rows((ROOT/'data/language_v1/train_shards').glob('*.tokens.jsonl')))
logs=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
tokens=0; boundary=100000; evals=[]
for update,log in enumerate(logs,1):
    n=sum(len(s)-1 for s in rows[(update-1)*64:update*64]);tokens+=n
    state=log['state'];assert state['update']==update and state['next_data_cursor']==64*update
    assert state['prediction_tokens']==tokens and log['tokens']==n
    assert math.isclose(log['lr'],3e-4*min(1,tokens/50000),rel_tol=1e-12)
    assert math.isfinite(log['loss']) and math.isfinite(log['gradient_norm'])
    due=tokens>=boundary
    while tokens>=boundary:boundary+=100000
    assert state['next_validation_boundary']==boundary
    assert ('validation' in log)==due
    if due:evals.append(log)
assert tokens==decision['actual_tokens'] and len(logs)==decision['final_update']
for budget in (1000000,3000000):
    d=load(run/f'gate_decision_{budget}.json')
    stage=[e for e in evals if e['state']['update']<=d['final_update']]
    best=min(stage,key=lambda e:(e['validation']['general']['answer_ce'],e['state']['update']))
    assert best['state']['update']==d['selected_update']
    assert best['validation']['general']['answer_ce']==d['general_answer_ce']
    recent=[e['validation']['general']['answer_ce'] for e in stage[-4:]]
    assert recent==d['recent_general_answer_ce']
    improves=sum(a-b>=1e-4 for a,b in zip(recent,recent[1:]))
    assert improves==d['improving_transitions']
    assert logs[d['final_update']-2]['state']['prediction_tokens']<budget<=d['actual_tokens']
    assert d['overshoot']==d['actual_tokens']-budget
    expected='passed' if d['general_accuracy']>=.99 and all(v>=.95 for v in d['diagnostic_accuracies'].values()) else ('extend_to_3m' if budget==1000000 and improves>=2 else 'failed')
    assert d['decision']==expected
checkpoints={}
for name in ('init.pt','best.pt','last.pt'):
    p=torch.load(run/name,map_location='cpu',weights_only=False)
    assert not p['debug_only'] and p['hashes']==manifest['input_hashes']
    assert all(k in p for k in ('model','optimizer','rng_states','state'))
    assert sum(v.numel() for v in p['model'].values())==400640
    assert all(torch.isfinite(v).all() for v in p['model'].values())
    if name!='init.pt':
        assert p['state']==logs[p['state']['update']-1]['state']
        assert all(int(v['step'])==p['state']['update'] for v in p['optimizer']['state'].values())
    checkpoints[name]=dict(sha256=sha(run/name),state=p['state'])
result=load(run/'result.json');measure=result['measurement_first_50_updates']
assert measure['updates']==50 and measure['prediction_tokens']==sum(e['tokens'] for e in logs[:50])
assert math.isclose(measure['training_seconds'],sum(e['training_seconds'] for e in logs[:50]))
deterministic(0);torch.set_num_threads(2)
model=Transformer();model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=False)['model'])
general,diagnostics=evaluate_gate_suite(model,ROOT,16)
assert general['correct']==decision['general_accuracy']
assert abs(general['answer_ce']-decision['general_answer_ce'])<1e-5
for name,d in diagnostics.items():
    assert d['correct']==decision['diagnostic_accuracies'][name]
    assert len(d['rows'])==len({r['sequence_id'] for r in d['rows']})==512
report=dict(status='passed',meaning='Evidence consistency verified; behavior gate FAILED',decision='failed',
    evidence_zip_sha256=sha(base.parent/'p3_evidence_20260915T174256751247.zip'),
    checks=['107 corpus file hashes and config hashes','executed code and GPU smoke hashes','407 ordered full updates and token counts','warmup LR and finite losses/gradients','30 validation boundaries','1M extension and 3M stopping rule','minimum CE checkpoint selection','checkpoint model/optimizer/RNG/progress','first 50 updates measurement','CPU validation reevaluation'],
    checkpoints=checkpoints,first_50=measure,environment=manifest['environment'],
    cpu_validation=dict(general=metric_summary(general),diagnostics={k:metric_summary(v) for k,v in diagnostics.items()}),
    actual_tokens=tokens,updates=len(logs),evaluations=len(evals),
    total_session_elapsed_seconds=sum(load(run/f'manifest_{b}.json')['elapsed_seconds'] for b in (1000000,3000000)),
    training_seconds=sum(e['training_seconds'] for e in logs),validation_seconds=sum(e.get('validation_seconds',0) for e in logs),
    limitations=['1M checkpoint bytes were superseded in the export; its hash and selection metrics remain in the 1M decision and logs. Drive snapshots were not included.','CPU re-evaluation verifies the final selected checkpoint; it does not establish why learning failed.'],elapsed_seconds=time.time()-start)
# Initial checkpoint best_validation is inf; serialize it explicitly as null.
report['checkpoints']['init.pt']['state']['best_validation']=None
out=ROOT/'experiment_v1/results/p3_evidence_verification.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ('status','decision','actual_tokens','updates','evaluations','elapsed_seconds')},indent=2))
