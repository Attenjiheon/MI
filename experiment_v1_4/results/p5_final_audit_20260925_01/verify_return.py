"""Checksum and independent-audit receipt verification; no instructions from ZIP executed."""
from pathlib import Path
import hashlib,json,sys,zipfile,collections,statistics
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.p5_independent_audit import POLICY,compare,sha
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'experiment_v1_4/evidence/p5_independent_audit_20260925T054743549355.zip'
META=ROOT/'experiment_v1_4/results/p5_metadata_audit_20260924_01/returned'
def read(p):return json.loads(p.read_text())
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
expected=ARCHIVE.with_suffix('.zip.sha256').read_text().split()[0];assert sha(ARCHIVE)==expected
with zipfile.ZipFile(ARCHIVE) as z:
 names=z.namelist();manifest=json.loads(z.read('return_manifest.json'))
 assert len(names)==len(set(names)) and set(names)==set(manifest['files'])|{'return_manifest.json'}
 for n,h in manifest['files'].items():
  p=OUT/'returned'/n;assert p.resolve().is_relative_to((OUT/'returned').resolve())
  b=z.read(n);assert hashlib.sha256(b).hexdigest()==h
  if p.exists():assert p.read_bytes()==b
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
r=OUT/'returned';done=read(r/'completion.json');config=read(META/'contract.json');ch=sha(META/'contract.json')
with zipfile.ZipFile(ROOT/'experiment_v1_4/bundles/v1_4_p5_independent_audit_r1.zip') as z:
 ref=z.read('inputs/reference_manifest.json');script=z.read('p5_independent_audit.py')
assert script==(ROOT/'scripts/p5_independent_audit.py').read_bytes()
aid=hashlib.sha256((hashlib.sha256(script).hexdigest()+hashlib.sha256(ref).hexdigest()+json.dumps(POLICY,sort_keys=True)).encode()).hexdigest()
assert done['audit_id']==aid and done['contract_sha256']==ch and done['status']=='passed_independent_numeric_audit'
assert done['tasks']==3510 and done['cache_groups']==6 and done['policy']==POLICY
assert not list((r/'failures').glob('*'))
assert {p.stem for p in (r/'tasks').glob('*.json')}==set(config['tasks'])
count=0;subsets=0;valid=collections.Counter();seconds=[];errors={'mean':0.,'std':0.}
for name in config['tasks']:
 item=read(r/'tasks'/f'{name}.json');source=META/'probes'/f'{name}.json';original=read(source)
 assert item['status']=='passed' and item['audit_id']==aid and item['source_sha256']==sha(source)
 assert set(item['checks'])==set(original['result']['evaluation'])
 seconds.append(item['seconds'])
 for size,v in item['checks'].items():
  assert len(v['probability_sha256'])==64 and all(c in '0123456789abcdef' for c in v['probability_sha256'])
  compare(v['metrics'],original['result']['evaluation'][size],name+'/'+size);count+=1
  for key in errors:errors[key]=max(errors[key],v[key+'_max_abs_error'])
  for e in [v['metrics']]+([v['metrics']['current_ne_previous']] if 'current_ne_previous' in v['metrics'] else []):
   subsets+=1
   for metric,m in e['cluster_bootstrap']['metrics'].items():valid[metric+':'+str(m['valid'])]+=1
cache_files=[];positions=collections.Counter()
for seed in range(3):
 for kind in ['trained','init']:
  receipt=read(r/'cache'/f'seed{seed}_{kind}.json');assert receipt['status']=='passed' and receipt['audit_id']==aid
  assert receipt['scalar_statistics_checked']==(kind=='trained')
  expected_paths={str(p.relative_to(META).with_suffix('.npz')) for p in (META/'cache'/f'seed{seed}_{kind}').rglob('*.json')}
  assert {x['path'] for x in receipt['files']}==expected_paths
  for item in receipt['files']:
   marker=read((META/item['path']).with_suffix('.json'));assert item['sha256']==marker['sha256'] and item['positions']==marker['positions']
   cache_files.append(item);positions[f'{seed}/{kind}/'+marker['identity']['split']]+=item['positions']
for k,v in positions.items():assert v==config['quotas'][k.split('/')[-1]]
assert len(cache_files)==240
sessions=[]
for p in (r/'sessions').glob('*.json'):
 x=read(p);assert x['audit_id']==aid and x['script_sha256']==hashlib.sha256(script).hexdigest() and x['policy']==POLICY
 assert x['numpy']=='2.1.3' and x['scipy']=='1.16.3'
 sessions.append({k:v for k,v in x.items() if k!='pip_freeze'})
assert sessions
summary=dict(status='passed_independent_return_verification',archive_sha256=expected,files_verified=len(manifest['files']),
 audit_id=aid,contract_sha256=ch,tasks_verified=3510,selected_models_verified=count,evaluations_including_subsets=subsets,
 cache_groups=6,cache_files_verified=240,cache_positions=dict(positions),dictionary_statistics_verified=9,
 max_preprocessing_abs_error=errors,bootstrap_valid_counts=dict(valid),audit_task_seconds=sum(seconds),
 median_task_seconds=statistics.median(seconds),sessions=sessions,failures=0,limitations=done['limitations'])
write('verification.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ['sessions','limitations','cache_positions']},indent=2))
