"""Verify source-audit return against frozen contracts and prior P9 evidence."""
from pathlib import Path
import hashlib,json,zipfile,sys,math
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'experiment_v1_4'
OUT=E/'results/p9_final_audit_20261002_01'
ARCHIVE=E/'evidence/P9/v1_4_p9_source_audit_return_1790870630393490455.zip'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def compare(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:compare(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):compare(x,y)
 elif a is None or isinstance(a,(str,bool)):assert a==b
 else:assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-6+2e-5*abs(b),(a,b)
def main():
 checksum=sha(ARCHIVE);assert checksum==read(ARCHIVE.with_suffix('.sha256.json'))['sha256']
 audit=read(E/'p9_audit_r1/contract.json')
 for n,h in audit['files'].items():assert sha(ROOT/n)==h,n
 identity=dict(audit_contract_sha256=sha(E/'p9_audit_r1/contract.json'),return_manifest_sha256=sha(E/'p9_audit_r1/return_manifest.json'),original_contract_sha256=audit['original_contract_sha256'])
 with zipfile.ZipFile(ARCHIVE) as z:
  names=z.namelist();assert len(names)==len(set(names))
  manifest=json.loads(z.read('return_manifest.json'))
  assert set(names)==set(manifest['files'])|{'return_manifest.json'}
  for n,h in manifest['files'].items():
   assert not Path(n).is_absolute() and '..' not in Path(n).parts
   assert hashlib.sha256(z.read(n)).hexdigest()==h,n
  assert not any(n.startswith('failures/') for n in names)
  complete=json.loads(z.read('source_audit_complete.json'))
  assert complete['identity']==identity==json.loads(z.read('identity.json'))
  assert complete['status']=='source_reproduction_passed_return_review_pending'
  expected={f'seed0_l{l}_{t}_k{k}_s1' for l in (0,3,7,11) for t in ('sae','tc') for k in (4,16)}
  assert {r['run'] for r in complete['runs']}==expected and len(complete['runs'])==16
  assert {n for n in names if n.startswith('runs/')}=={f'runs/{n}.json' for n in expected}
  environments=set()
  sessions=[n for n in names if n.startswith('sessions/') and n.endswith('/environment.json')]
  for n in sessions:
   env=json.loads(z.read(n));environments.add(env['environment_id'])
   assert env['device']=='cuda' and env['gpu']=='Tesla T4'
   assert env['config_sha256']==identity['audit_contract_sha256']
   assert env['torch']=='2.11.0+cu128' and env['numpy']=='2.1.3'
   lock=z.read(str(Path(n).parent/'requirements.lock.txt'))
   assert hashlib.sha256(lock).hexdigest()==env['lock_sha256']
  for r in complete['runs']:
   assert r==json.loads(z.read('runs/'+r['run']+'.json'))
   assert r['status']=='passed' and r['identity']==identity and r['environment_id'] in environments
   assert r['probe_tasks']==(70 if r['tool']=='sae' else 80)
   assert r['semantic_reports']==(130 if r['tool']=='sae' else 140)
   assert r['fidelity_positions']==dict(train=50000,val=10000,test=20000)
   assert r['causal_replayed_pairs']==4 and r['replacement_replayed_targets']==16 and r['validation_bin_sets']==2
   assert r['elapsed_seconds']>0
  for t in ('sae','tc'):assert json.loads(z.read('aggregates/'+t+'.json'))==dict(status='passed',identity=identity)
  compared=0;refits=set()
  meta=E/'results/p9_return_audit_20261001_01/returned_metadata'
  for t in ('sae','tc'):
   c=read(E/f'p9_eval_r1/{t}_contract.json');assert sha(E/f'p9_eval_r1/{t}_contract.json')==identity['original_contract_sha256'][t]
   for e in c['saes' if t=='sae' else 'tcs']:
    r=e['run'];selection=read(meta/t/'runs'/r['name']/'selection.json')
    for task,stored in selection['fits'].items():
     key=f"seed0_l{r['layer']}_{task}"
     if key in c['reused_probes']:continue
     rep=task.split('_')[0]
     if rep in ('sae','transcoder'):key+='_k'+str(r['k'])
     n='refits/'+t+'/'+key+'.json';refits.add(n);fit=json.loads(z.read(n))
     assert fit['status']==stored['status']=='passed'
     assert fit['ranking']==stored['ranking'] and fit['candidate_count']==stored['candidate_count']
     compare(stored['selected'],fit['selected']);compared+=1
  assert len(refits)==560 and refits=={n for n in names if n.startswith('refits/')}
  logs=[z.read(n).decode() for n in names if n.startswith('logs/')]
  assert all(any('resume verified '+r in log for log in logs) for r in expected)
  assert sum(log.count('probe verified ') for log in logs)>=1200
 p8=read(E/'results/p8_final_audit_20260930_01/completion.json');assert p8['p8_complete']
 training=read(E/'results/p9_training_audit_20260930_01/completion.json');assert training['p9_training_complete']
 for n,h in training['evidence_sha256'].items():assert sha(E/'results/p9_training_audit_20260930_01'/n)==h
 local=read(E/'results/p9_return_audit_20261001_01/status.json');assert local['status']=='passed_local_source_reproduction_pending' and local['runs']==16
 for n,h in local['evidence_sha256'].items():assert sha(E/'results/p9_return_audit_20261001_01'/n)==h
 for n,h in local['code_sha256'].items():assert sha(ROOT/n)==h
 evidence=[ARCHIVE,ARCHIVE.with_suffix('.sha256.json'),E/'p9_audit_r1/contract.json',E/'results/p8_final_audit_20260930_01/completion.json',E/'results/p9_training_audit_20260930_01/completion.json',E/'results/p9_return_audit_20261001_01/status.json',Path(__file__)]
 result=dict(status='passed_P9_complete',p9_complete=True,p10_eligible=True,whole_experiment_complete=False,runs=16,source_return_files_verified=len(manifest['files']),novel_refits_verified=len(refits),refit_selection_comparisons=compared,probe_tasks=1200,semantic_reports=2160,fidelity_positions_per_run=80000,causal_gpu_replayed_pairs=64,replacement_gpu_replayed_targets=256,full_local_causal_pairs=32768,full_local_replacement_targets=32768,initialization_comparison_rows=44880,source_sessions=len(sessions),environment_ids=sorted(environments),source_run_elapsed_seconds=sum(r['elapsed_seconds'] for r in complete['runs']),evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence},limitations=['GPU replay uses four fixed pairs and 16 fixed replacement targets per run; full raw metrics checked separately.','Sparse seed comparison is within LM seed 0, point estimates only; latent IDs are not aligned across dictionaries.','P10 decision and P11 final synthesis remain pending.'])
 OUT.mkdir(exist_ok=True)
 p=OUT/'completion.json'
 if p.exists():assert read(p)==result
 else:p.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
