"""Read untrusted return JSON/ZIP as data; never execute its supplied source."""
import hashlib,json,math
from pathlib import Path,PurePosixPath
import zipfile
ROOT=Path(__file__).resolve().parents[3]
ARCHIVE=ROOT/'experiment_v1_4/evidence/p6_audit_r2_evidence_1790344844084851588.zip'
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
 with zipfile.ZipFile(ARCHIVE) as z,zipfile.ZipFile(ROOT/'experiment_v1_4/bundles/P6/v1_4_p6_audit_r2.zip') as delivery:
  names=z.namelist();assert len(names)==len(set(names))
  assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
  read=lambda n:json.loads(z.read(n))
  manifest=read('p6_audit_r2_manifest.json');assert z.read('p6_audit_r2_manifest.json')==delivery.read('p6_audit_r2_manifest.json')
  for name,digest in manifest['files'].items():assert sha(delivery.read(name))==digest
  script='scripts/verify_v1_4_p6_return_r2.py'
  assert z.read(script)==delivery.read(script)==(ROOT/script).read_bytes()
  contract_path='experiment_v1_4/p6_r1/contract.json'
  assert delivery.read(contract_path)==(ROOT/contract_path).read_bytes()
  contract=json.loads(delivery.read(contract_path));ch=sha(delivery.read(contract_path))
  bases=[n[:-len('/completion.json')] for n in names if n.endswith('/completion.json')];assert len(bases)==1
  base=bases[0];c=read(base+'/completion.json');identity=read(base+'/identity.json')
  assert c['audit_identity']==identity and identity['auditor_sha256']==sha(z.read(script))
  assert identity['production_config_sha256']==ch and identity['tolerance']==dict(atol=1e-7,rtol=1e-6)
  assert c['status']=='passed_numeric_return_review_pending' and c['p6_complete'] is False
  assert c['config_sha256']==ch and c['input_sha256']==sha(z.read('input_manifest.json'))
  inputs=read('input_manifest.json');receipt_path='experiment_v1_4/results/p5_final_audit_20260925_01/p6_cache_manifest.json'
  receipt=json.loads((ROOT/receipt_path).read_text());assert inputs['receipt_sha256']==sha((ROOT/receipt_path).read_bytes())
  assert inputs['source_cache_files']==receipt['cache_files'] and inputs['labels']==receipt['labels']
  assert inputs['source_contract_sha256']==receipt['config_sha256'] and inputs['config_sha256']==ch
  assert inputs['layers']==[0,3,7,11] and inputs['statistics_count']==36 and inputs['test_used_for_selection'] is False
  assert set(inputs['statistics'])=={f'statistics/seed{s}_l{l}_{h}.json' for s in range(3) for l in (0,3,7,11) for h in 'hum'}
  assert set(inputs['tensors'])=={f'inputs/seed{s}_l{l}_h_{split}.pt' for s in range(3) for l in (0,3,7,11) for split in ('train','val')}
  env=read(base+'/environment.json');assert env['environment_id']==identity['environment_id']
  assert sha(z.read(base+'/environment/requirements.lock.txt'))==env['lock_sha256']
  assert hashlib.sha256(json.dumps({k:env[k] for k in ('python','platform','torch','cuda','cudnn','gpu','lock_sha256')},sort_keys=True).encode()).hexdigest()[:16]==env['environment_id']
  expected={r['name']:r for r in contract['runs']};actual={r['run']:r for r in c['runs']}
  assert len(c['runs'])==len(actual)==24 and actual.keys()==expected.keys()
  scores=[];summary=[]
  for name,r in expected.items():
   report=actual[name];assert read(base+'/runs/'+name+'.json')==report
   result_path=f'runs/{name}/result.json';result=read(result_path)
   assert sha(z.read(result_path))==report['result_sha256']
   assert result['run']==r and result['identity']==dict(config_sha256=ch,input_sha256=c['input_sha256'])
   assert result['updates']==5000 and result['draws']==2560000 and result['parameter_count']==262912
   assert set(result['checkpoints'])=={f'update_{u:05d}.pt' for u in [100]+list(range(250,5001,250))}
   assert len(report['scores'])==20 and [s['update'] for s in report['scores']]==list(range(250,5001,250))
   for score in report['scores']:
    assert score==read(base+f'/checkpoints/{name}/{score["update"]:05d}.json')
    assert score['run']==name and score['passed'] is True
    assert score['checkpoint_sha256']==result['checkpoints'][f'update_{score["update"]:05d}.pt']
    assert math.isfinite(score['saved_mse']) and math.isfinite(score['recomputed_mse'])
    error=abs(score['saved_mse']-score['recomputed_mse'])
    assert score['absolute_error']==error and error<=1e-7+1e-6*abs(score['saved_mse'])
    scores.append(score)
   best=min(report['scores'],key=lambda s:(s['saved_mse'],s['update']))
   assert best['update']==report['best_update']==result['best_update']
   assert best['saved_mse']==result['best_val_mse'] and result['best_checkpoint']==f'update_{best["update"]:05d}.pt'
   assert result['last_checkpoint']=='update_05000.pt' and result['benchmark']['included_in_budget'] is True
   summary.append(dict(run=name,best_update=best['update'],best_val_mse=best['saved_mse']))
  assert c['updates']==120000 and c['draws']==61440000
  target=next(s for s in scores if s['run']=='seed0_l3_sae_k16_s0' and s['update']==250)
  assert target['legacy_mse']==0.03099766511893298 and target['saved_mse']==target['recomputed_mse']==0.030997336196491705
  assert target['topk_candidate_rows_changed']==1
  result=dict(status='passed_numeric_evidence_review_full_return_pending',p6_complete=False,archive_sha256=sha(ARCHIVE.read_bytes()),members=len(names),member_sha256={n:sha(z.read(n)) for n in names},runs=24,validation_checkpoints=len(scores),checkpoint_hash_references=24*21,max_absolute_error=max(s['absolute_error'] for s in scores),legacy_topk_change_checkpoint_count=sum(s['topk_candidate_rows_changed']>0 for s in scores),original_failure=target,environment=env,elapsed_seconds=c['elapsed_seconds'],updates_reported=c['updates'],draws_reported=c['draws'],selected=summary,remaining=['Full export: checkpoint payloads/optimizer/RNG/curve/init statistics/GPU smoke/training environment/failures','Phase completion Git update after final evidence verification'])
  (Path(__file__).parent/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k not in ('member_sha256','selected','environment')},indent=2))
if __name__=='__main__':run()
