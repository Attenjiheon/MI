"""Audit both P4 archives sequentially with bounded temporary disk use; no inference."""
import csv,hashlib,json,math,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
import torch
from interp_v1_4.runtime import sha
from interp_v1_4.training import learning_rate
from interp_v1_4.behavior import valid_pair_coverage
from interp_v1_4.cli import stage_contract
OUT=ROOT/'experiment_v1_4/results/p4_audit_20260921_01'

def write(path,obj):
 if path.exists():
  assert json.loads(path.read_text())==obj;return
 with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def finite(v):
 if isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
 elif isinstance(v,float):assert math.isfinite(v)

def audit(seed):
 output=OUT/f'seed{seed}';output.mkdir(exist_ok=True)
 archive=next((ROOT/'experiment_v1_4/evidence').glob(f'v1_4_seed{seed}_evidence_*.zip'))
 digest=sha(archive);sidecar=archive.with_suffix('.zip.sha256')
 if sidecar.exists():assert sidecar.read_text().split()[0]==digest
 temp=Path('/private/tmp/mi_p4_seed1_zgjp98dw') if seed==1 else Path(tempfile.mkdtemp(prefix=f'mi_p4_seed{seed}_',dir='/private/tmp'))
 with zipfile.ZipFile(archive) as z:
  infos=z.infolist();assert len(infos)==len(set(z.namelist()))
  sums=json.loads(z.read('checksums.json'))
  assert {i.filename for i in infos if not i.is_dir()}==set(sums)|{'checksums.json'}
  for i in infos:
   path=temp/i.filename;assert path.resolve().is_relative_to(temp.resolve())
   assert (i.external_attr>>16)&0o170000!=0o120000
   if i.is_dir():continue
   path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
   if path.exists():
    assert i.filename=='checksums.json' or sha(path)==sums[i.filename]
    continue
   with z.open(i) as f,path.open('xb') as g:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block);g.write(block)
   if i.filename!='checksums.json':assert h.hexdigest()==sums[i.filename],i.filename
 write(output/'archive_verification.json',dict(status='passed',archive=str(archive.relative_to(ROOT)),sha256=digest,bytes=archive.stat().st_size,files_verified=len(sums),outer_checksum_supplied=sidecar.exists()))
 run=temp/f'deepwide12_read4_seed{seed}'
 for name in ('result.json','gate.json','gate_started.json','LATEST.json'):
  shutil.copy2(run/name,output/name)
 shutil.copy2(temp/'verification.json',output/'colab_verification.json')
 shutil.copytree(temp/'delivery_contract',output/'delivery_contract',dirs_exist_ok=True)
 bundle=json.loads((temp/'input_bundle.json').read_text())
 delivery=json.loads((ROOT/'experiment_v1_4/results/p4_preparation_r1/delivery.json').read_text())
 assert bundle['sha256']==delivery['bundle_sha256']
 with zipfile.ZipFile(ROOT/delivery['bundle']) as z:
  for p in (temp/'delivery_contract').iterdir():assert p.read_bytes()==z.read('experiment_v1_4/p4/'+p.name)
 with (output/'run_verification.log').open('w') as log:
  subprocess.run([sys.executable,'scripts/verify_v1_4_evidence.py',str(run),'--allow-cross-environment-init','--output',str(output/'run_verification.json')],check=True,stdout=log,stderr=subprocess.STDOUT)
 smokezip=temp/'smoke_return.zip'
 with zipfile.ZipFile(smokezip,'x',zipfile.ZIP_STORED) as z:
  for p in sorted((temp/'gpu_smoke').rglob('*')):z.write(p,str(p.relative_to(temp/'gpu_smoke'))+('/' if p.is_dir() else ''))
 with (output/'gpu_verification.log').open('w') as log:
  subprocess.run([sys.executable,'scripts/verify_v1_4_gpu_smoke.py',str(smokezip),str(output/'gpu_smoke')],check=True,stdout=log,stderr=subprocess.STDOUT)
 r=json.loads((run/'result.json').read_text());finite(r)
 assert r['lm_seed']==seed and r['stage']=='p4' and not r['debug_only']
 assert json.loads((run/'gate.json').read_text())==r['gate']
 assert json.loads((run/'gate_started.json').read_text())['selected']==r['selected']
 local_verification=json.loads((output/'run_verification.json').read_text())
 initialization=local_verification.pop('initialization_verification')
 assert local_verification==json.loads((output/'colab_verification.json').read_text())
 smoke=json.loads((temp/'gpu_smoke/smoke.json').read_text())
 assert r['hashes']['code']==smoke['input_hashes']['code']
 assert r['environment']['environment_id']==smoke['environment']['environment_id']
 sessions=[]
 for p in sorted((run/'sessions').glob('*/environment.json')):
  s=json.loads(p.read_text());e=s['environment'];assert sha(p.parent/'requirements.lock.txt')==e['lock_sha256']
  identity={k:e[k] for k in ('python','platform','torch','cuda','cudnn','gpu','lock_sha256')}
  assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]==e['environment_id']
  assert e['environment_id']==r['environment']['environment_id']
  assert f'--lm-seed {seed}' in s['command'] and '--stage p4' in s['command']
  if s['resume']!='None':assert (run/'checkpoints'/Path(s['resume']).name).exists()
  sessions.append(s)
 assert sessions[0]['resume']=='None'
 events=[json.loads(p.read_text()) for p in sorted((run/'events').glob('*.json'))]
 for e in events:
  finite(e);assert e['microbatch']==16 and e['environment_id']==r['environment']['environment_id']
  assert math.isclose(e['lr'],learning_rate(e['state']['prediction_tokens']),abs_tol=1e-14)
  assert e['loss_weight_sum']==e['prediction_tokens']+3*e['read_answer_targets']
  assert e['training_seconds']>0
 first=r['measurement_first_50_updates'];assert first['updates']==50
 assert first['prediction_tokens']==sum(e['prediction_tokens'] for e in events[:50])
 assert math.isclose(first['training_seconds'],sum(e['training_seconds'] for e in events[:50]),abs_tol=1e-8)
 milestones=[]
 for boundary,m in sorted(r['milestones'].items(),key=lambda x:int(x[0])):
  c=m['current'];v=c['validation'];assert v['general']['sequence_count']==512
  assert valid_pair_coverage(v['pairs'],64)
  milestones.append(dict(seed=seed,boundary=int(boundary),update=c['update'],tokens=c['prediction_tokens'],general_accuracy=v['general']['accuracy'],general_ce=v['general']['answer_ce'],first_macro_ce=v['pairs']['first']['cell']['macro_answer_ce'],first_macro_accuracy=v['pairs']['first']['cell']['macro_accuracy']))
 for v in [m['current']['validation'] for m in r['milestones'].values()]+[r['gate']]:
  u=v['pairs']['uncertainty'];assert u['draws']==u['valid_draws']==1000
  for member in ('first','repeat'):
   for cell in v['pairs'][member]['cell']['cells'].values():
    assert cell['count']==64
    for k in ('accuracy_ci95','answer_ce_ci95'):
     lo,hi=cell[k];assert 0<=lo<=hi
  for g in [v['general']]+list(v.get('diagnostics',{}).values()):
   assert g['prediction_tokens']>0 and g['all_token_ce']>=0
   assert g['baselines']['count']==g['answer_count']
   assert set(g['strata'])=={'operator','query_var','answer','blocks','distance','truth_table'}
 contract=stage_contract(ROOT,'p4','deepwide12_read4')
 checkpoint_records=[]
 for p in sorted((run/'checkpoints').glob('*.pt')):
  payload=torch.load(p,map_location='cpu',weights_only=False)
  assert payload['hashes']==r['hashes'] and not payload['debug_only'] and payload['microbatch']==16
  state=payload['state'];assert state['next_data_cursor']==state['update']*64
  assert payload['next_evaluation_boundary']==next((b for b in contract['milestones'] if b>state['prediction_tokens']),None)
  assert set(payload['rng_states'])=={'python','numpy','torch','cuda'}
  assert payload['rng_states']['cuda'] is not None
  if state['update']:
   assert payload['optimizer']['state']
   assert all(int(v['step'])==state['update'] for v in payload['optimizer']['state'].values())
  checkpoint_records.append(dict(path=p.name,sha256=sha(p),state=state,next_boundary=payload['next_evaluation_boundary']))
  del payload
 interruptions=[json.loads(p.read_text()) for p in (run/'interruptions').glob('*.json')]
 timing=dict(recorded_training_seconds=sum(e['training_seconds'] for e in events),select_validation_seconds=sum(e.get('validation_seconds',0) for e in events),session_seconds=r['session_seconds'],first_50_tokens_per_second=first['prediction_tokens']/first['training_seconds'],peak_memory_bytes=first['peak_memory_bytes'],note='Session seconds describe final session only. Uncommitted work before init-checkpoint recovery is not measurable from committed events.')
 write(output/'details_verification.json',dict(status='passed',initialization_verification=initialization,sessions=sessions,interruptions=interruptions,milestones=milestones,checkpoints=checkpoint_records,timing=timing,gate_reexecuted=False,test_scores_observed=False,limitation='Recorded CUDA evidence audited locally; no independent GPU replay.'))
 write(output/'completion.json',dict(status='passed_return_audit',seed=seed,gate_passed=r['gate']['decision']['passed'],selected_checkpoint_sha256=r['selected']['sha256'],archive_sha256=digest,evidence_sha256={name:sha(output/name) for name in ('archive_verification.json','run_verification.json','gpu_smoke/verification.json','details_verification.json','result.json')},test_scores_observed=False,p4_complete=False))
 with (output/'select_curve.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(milestones[0]),lineterminator='\n');w.writeheader();w.writerows(milestones)
 shutil.rmtree(temp) # only this invocation's verified temporary extraction; source ZIP is immutable
 print(json.dumps(dict(seed=seed,status='passed_return_audit',gate=r['gate']['decision'],selected_update=r['selected']['update'])),flush=True)

for seed in (1,2):audit(seed)
