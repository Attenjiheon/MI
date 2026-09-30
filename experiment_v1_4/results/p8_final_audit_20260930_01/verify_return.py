from pathlib import Path,PurePosixPath
import hashlib,json,zipfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.audit_v1_4_p8_sources import compare
from interp_v1_4.p8_evaluation import verify,CONTRACT
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha
out=Path(__file__).resolve().parent
archive=ROOT/'experiment_v1_4/evidence/P8/v1_4_p8_source_audit_return_1790725662385674572.zip'
sidecar=json.loads(archive.with_suffix('.sha256.json').read_text())
assert sha(archive)==sidecar['sha256'] and archive.stat().st_size==sidecar['bytes']
config=verify(ROOT);audit=read(ROOT/'experiment_v1_4/p8_audit_r1/contract.json')
for name,digest in audit['files'].items():assert sha(ROOT/name)==digest,name
identity=dict(original_contract_sha256=sha(ROOT/CONTRACT),audit_contract_sha256=sha(ROOT/'experiment_v1_4/p8_audit_r1/contract.json'))
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('return_manifest.json'));names=z.namelist()
 assert len(names)==len(set(names)) and set(names)==set(manifest['files'])|{'return_manifest.json'}
 for name,digest in manifest['files'].items():
  path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
  data=z.read(name);assert hashlib.sha256(data).hexdigest()==digest,name
  dest=out/'returned'/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==data
  else:dest.write_bytes(data)
 assert not any(n.startswith('failures/') for n in names)
 (out/'returned/return_manifest.json').write_bytes(z.read('return_manifest.json'))
returned=out/'returned';complete=read(returned/'source_audit_complete.json')
assert read(returned/'identity.json')==identity==complete['identity']
assert complete['status']=='source_reproduction_passed_return_review_pending'
assert {r['run'] for r in complete['runs']}=={e['run']['name'] for e in config['tcs']} and len(complete['runs'])==24
old=ROOT/'experiment_v1_4/results/p8_return_audit_20260929_01'
local=read(old/'verification.json');assert len(local['runs'])==24
for path,digest in read(old/'status.json')['evidence_sha256'].items():assert sha(old/path)==digest,path
expected_refits=set();compared=0
for e in config['tcs']:
 r=e['run'];saved=read(returned/'runs'/(r['name']+'.json'))
 assert saved==next(v for v in complete['runs'] if v['run']==r['name'])
 assert saved['identity']==identity and saved['status']=='passed' and saved['environment_id']=='f8912b7ab0ca620d'
 for key,value in dict(probe_tasks=80,semantic_reports=140,fidelity_positions=dict(train=50000,val=10000,test=20000),validation_bin_sets=2,causal_replayed_pairs=4,replacement_replayed_targets=16).items():assert saved[key]==value,(r['name'],key)
 selection=read(old/'returned_metadata/runs'/r['name']/'selection.json')
 for task,fit in selection['fits'].items():
  reuse=f"seed{r['lm_seed']}_l{r['layer']}_{task}"
  if reuse in config['reused_probes']:continue
  rep=task.split('_')[0];key=reuse+('_k'+str(r['k']) if rep=='transcoder' else '')
  expected_refits.add('refits/'+key+'.json');refit=read(returned/'refits'/(key+'.json'))
  assert fit['status']==refit['status']=='passed'
  assert fit['ranking']==refit['ranking'] and fit['candidate_count']==refit['candidate_count']
  assert all(t['attempts'][-1]['success'] for t in refit['trace'])
  for size,model in fit['selected'].items():
   other=refit['selected'][size]
   for field in ('columns','classes','lam','threshold'):assert model[field]==other[field]
   compare(model,other,r['name']+'/'+task+'/'+size)
  compared+=1
assert expected_refits=={n for n in names if n.startswith('refits/')} and len(expected_refits)==840
for p in returned.glob('sessions/*/environment.json'):
 env=read(p);assert env['environment_id']=='f8912b7ab0ca620d' and env['gpu']=='Tesla T4' and env['torch']=='2.11.0+cu128'
 assert sha(p.parent/'requirements.lock.txt')==env['lock_sha256']
expected_old=read(ROOT/'experiment_v1_4/p8_audit_r1/return_manifest.json')['files']
for name in ['evaluation_complete.json','raw_evidence_audit.json']+[n for n in expected_old if n.startswith(('tables/','figures/')) or n.endswith(('/selection.json','/semantic.json','/summary.json','/matching.json'))]:
 assert sha(old/'returned_metadata'/name)==expected_old[name],name
result=dict(status='passed',archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,external_checksum_sidecar_supplied=True,verified_members=len(manifest['files']),identity=identity,runs=24,unique_refits=840,novel_fit_tasks_compared=compared,probe_tasks=1920,semantic_reports=3360,fidelity_position_evaluations=24*80000,validation_bin_sets=48,causal_replayed_pairs=96,replacement_replayed_targets=384,run_audit_seconds=sum(r['elapsed_seconds'] for r in complete['runs']),gpu_environment_id='f8912b7ab0ca620d',prior_local_verification_sha256=sha(old/'verification.json'),scope='Full original-cache refits/predictions/fidelity and CI reproduction reported by hash-verified audit; exported refit coefficients compared locally. All original raw outputs previously verified locally; GPU replay is 96 fixed causal pairs and 384 replacement targets, not a complete second intervention run.')
write(out/'verification.json',result)
print(json.dumps(result,indent=2))
