"""Validate P10 source return against frozen contracts, coverage and original fits."""
from pathlib import Path
import sys,json,zipfile,hashlib,collections,math,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p10_source_audit import compare,identity
from interp_v1_4.runtime import sha
from interp_v1_4.p5 import read,write

def main():
 start=time.monotonic();out=ROOT/'experiment_v1_4/results/p10_final_audit_20261004_01';out.mkdir(exist_ok=True)
 archive=ROOT/'experiment_v1_4/evidence/P10/v1_4_p10_source_audit_return_1791123980642991185.zip';check=archive.with_suffix('.sha256.json');cs=read(check)
 assert sha(archive)==cs['sha256'] and archive.stat().st_size==cs['bytes']
 ac=read(ROOT/'experiment_v1_4/p10_audit_r1/contract.json');ec=read(ROOT/'experiment_v1_4/p10_eval_r1/contract.json');ident=identity(ROOT)
 for n,h in ac['files'].items():assert sha(ROOT/n)==h,n
 previous=ROOT/'experiment_v1_4/results/p10_return_audit_20261004_01'
 for file in [previous/'status.json',ROOT/'experiment_v1_4/results/p10_source_audit_preparation_r1/verification.json']:
  for n,h in read(file)['evidence_sha256'].items():assert sha(ROOT/n)==h,n
 original=read(previous/'return_manifest.json')['files'];metadata=previous/'returned_metadata'
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();assert len(names)==len(set(names));manifest=json.loads(z.read('return_manifest.json'))
  assert set(names)==set(manifest['files'])|{'return_manifest.json'}
  for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
  def j(n):return json.loads(z.read(n))
  gpu=j('gpu_complete.json');cpu=j('source_audit_complete.json')
  assert gpu['status']=='passed' and cpu['status']=='source_reproduction_passed_return_review_pending'
  assert gpu['identity']==cpu['identity']==ident and cpu['gpu_complete_sha256']==manifest['files']['gpu_complete.json']
  assert (gpu['activation_position_visits'],gpu['unique_reserved_positions'],gpu['activation_chunks'],gpu['dictionary_runs'])==(480000,80000,240,12)
  assert (cpu['refits'],cpu['semantic_reports'])==(1248,2016)
  expected={};quotas={'train':50000,'val':10000,'test':20000}
  for split,quota in quotas.items():
   rows=read(metadata/'labels'/f'{split}.json');assert len(rows)==quota
   counts=collections.Counter(r['sequence_id'] for r in rows);ids=sorted(counts)
   for lm in range(3):
    for kind in ['trained','init']:
     for offset in range(0,len(ids),128):expected[f'chunks/seed{lm}_{kind}_{split}_{offset:06d}.json']=(lm,kind,split,offset,sum(counts[i] for i in ids[offset:offset+128]))
  dictionary_names={f'dictionaries/{e["run"]["name"]}.json' for e in ec['selected']}
  assert set(gpu['files'])==set(expected)|dictionary_names
  envs=set();elapsed=collections.defaultdict(float);maxerr={h:0. for h in 'hum'}
  for n,h in {**gpu['files'],**cpu['tasks']}.items():
   assert manifest['files'][n]==h;r=j(n);assert r['identity']==ident and r['status']=='passed'
   assert math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>=0;elapsed[n.split('/')[0]]+=r['elapsed_seconds'];envs.add(r['environment_id'])
  for n,wanted in expected.items():
   r=j(n);assert tuple(r[k] for k in ['lm_seed','kind','split','sequence_offset','positions'])==wanted
   assert r['microbatch'] in [1,2,4,8,16]
   for h,v in r['max_absolute_errors'].items():assert math.isfinite(v) and v>=0;maxerr[h]=max(maxerr[h],v)
  maxlatent=0.
  for e in ec['selected']:
   r=j(f'dictionaries/{e["run"]["name"]}.json');assert r['run']==e['run']['name'] and r['checkpoint_sha256']==e['sha256']
   assert set(r['splits'])==set(quotas)
   for s,q in quotas.items():
    x=r['splits'][s];assert x['positions']==q and math.isfinite(x['maximum_latent_error']) and x['maximum_latent_error']>=0;maxlatent=max(maxlatent,x['maximum_latent_error'])
  assert set(cpu['tasks'])=={f'tasks/{t["id"]}.json' for t in ec['tasks']}
  reports=0
  for t in ec['tasks']:
   r=j(f'tasks/{t["id"]}.json');assert r['task']==t['id']
   for field,folder in [('source_fit_sha256','fits'),('source_report_sha256','semantic')]:assert r[field]==original[f'{folder}/{t["id"]}.json']
   f=read(metadata/'fits'/f'{t["id"]}.json')['fit'];compare(r['selected'],f['selected'],**ac['metric_tolerance']);assert r['reports']==len(f['selected']);reports+=r['reports']
  assert reports==2016
  sessions=[j(n) for n in names if n.startswith('sessions/') and n.endswith('environment.json')]
  assert envs <= {s['environment_id'] for s in sessions}
  events=[]
  for n in names:
   if n.startswith('logs/'):
    text=z.read(n).decode();events.append(dict(path=n,paused='PAUSED' in text,traceback='Traceback' in text))
  assert not any(x['traceback'] for x in events)
  for n in names:
   p=out/'returned'/n;assert p.resolve().is_relative_to((out/'returned').resolve());p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==z.read(n)
   else:p.write_bytes(z.read(n))
 result=dict(status='passed',p10_complete=True,experiment_complete=False,runs=12,verified_return_files=len(manifest['files']),activation_chunks=len(expected),activation_position_visits=480000,latent_positions=960000,refits=1248,semantic_reports=reports,maximum_activation_errors=maxerr,maximum_latent_error=maxlatent,environment_ids=sorted(envs),elapsed_task_seconds=dict(elapsed),operational_events=events,elapsed_seconds=time.monotonic()-start,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in [archive,check,ROOT/'experiment_v1_4/p10_audit_r1/contract.json',ROOT/'experiment_v1_4/p10_eval_r1/contract.json',previous/'status.json',ROOT/'experiment_v1_4/results/p10_training_audit_20261003_01/completion.json',Path(__file__)]},limitations=['Update exploratory after READ results; supervised probe transfer only','Additional layers/length/m-to-m SAE/Update interventions skipped; no causal-use claim','P11 final aggregation pending'])
 write(out/'completion.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','operational_events']},indent=2))
if __name__=='__main__':main()
