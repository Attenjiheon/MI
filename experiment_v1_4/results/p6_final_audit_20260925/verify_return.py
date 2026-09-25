"""Full P6 return audit. ZIP contents are data; checkpoints use restricted loading."""
import csv,hashlib,io,json,math,random,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from numpy.core.multiarray import _reconstruct
import torch
ROOT=Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0,str(ROOT))
from interp_v1_4.runtime import tensor_digest
from interp_v1_4.dictionary import Dictionary
OUT=Path(__file__).parent
ARCHIVE=ROOT/'experiment_v1_4/evidence/v1_4_p6_return_after_audit_r2_1790347089538567117.zip'
def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data,p
 else:p.write_bytes(data)
def load(b):
 allowed=[(_reconstruct,'numpy._core.multiarray._reconstruct'),np.ndarray,np.dtype,type(np.dtype('uint32'))]
 with torch.serialization.safe_globals(allowed):return torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
def main():
 torch.set_num_threads(2)
 digest=file_sha(ARCHIVE);assert digest==json.loads(ARCHIVE.with_suffix('.sha256.json').read_text())['sha256']
 config=json.loads((ROOT/'experiment_v1_4/p6_r1/contract.json').read_text());ch=file_sha(ROOT/'experiment_v1_4/p6_r1/contract.json')
 receipt=json.loads((ROOT/'experiment_v1_4/results/p5_final_audit_20260925_01/p6_cache_manifest.json').read_text())
 with zipfile.ZipFile(ARCHIVE) as z,zipfile.ZipFile(ROOT/'experiment_v1_4/evidence/p6_audit_r2_evidence_1790344844084851588.zip') as small:
  names=z.namelist();assert len(names)==len(set(names))
  assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
  read=lambda n:json.loads(z.read(n))
  manifest=read('export_manifest.json');assert set(names)==set(manifest['files'])|{'export_manifest.json'}
  for n,d in manifest['files'].items():
   h=hashlib.sha256()
   with z.open(n) as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   assert h.hexdigest()==d,n
  print('all',len(manifest['files']),'member checksums passed',flush=True)
  for n in small.namelist():
   if n.startswith(('audits/','runs/')) or n=='input_manifest.json':assert small.read(n)==z.read(n),n
  inp=read('input_manifest.json');ih=sha(z.read('input_manifest.json'))
  assert inp['config_sha256']==ch and inp['statistics_count']==36
  assert inp['source_cache_files']==receipt['cache_files'] and inp['labels']==receipt['labels']
  assert inp['layers']==[0,3,7,11] and inp['test_used_for_selection'] is False
  for n,d in inp['statistics'].items():
   assert manifest['files'][n]==d
   s=read(n);assert len(s['mean'])==256 and np.isfinite(s['mean']).all() and math.isfinite(s['scale']) and s['scale']>=1e-8
   assert s['train_positions']==50000 and s['accumulation']=='float64' and s['stored']=='float32'
   if '_l0_' in n:
    old=receipt['preprocessing'][Path(n).name.replace('_l0','')]
    assert s['mean']==old['values']['mean'] and s['scale']==old['values']['scale'] and s['reused_p5_sha256']==old['sha256']
  envs={}
  for n in names:
   if n.startswith('environments/') and n.endswith('/environment.json'):
    e=read(n);assert sha(z.read(str(PurePosixPath(n).parent/'requirements.lock.txt')))==e['lock_sha256']
    assert sha(json.dumps({k:e[k] for k in ('python','platform','torch','cuda','cudnn','gpu','lock_sha256')},sort_keys=True).encode())[:16]==e['environment_id']
    envs[e['environment_id']]=e
  smokes=[n for n in names if n.startswith('smoke/') and n.endswith('/smoke.json')];assert len(smokes)==1
  smoke=read(smokes[0]);assert smoke['status']=='passed' and smoke['device']=='cuda' and smoke['config_sha256']==ch and smoke['environment_id'] in envs
  assert set(smoke['checks'])=={'0','3','7','11'} and len(smoke['resume'])==8
  smoke_root=str(PurePosixPath(smokes[0]).parent)
  for r in smoke['resume']:
   assert r['status']=='bitwise'
   states=[load(z.read(f'{smoke_root}/resume_l{r["layer"]}_k{r["k"]}/{kind}/update_00004.pt')) for kind in ('full','resumed')]
   assert tensor_digest(states[0]['model'])==tensor_digest(states[1]['model'])==r['model_sha256']
   assert states[0]['curve']==states[1]['curve'] and states[0]['sampler_rng']==states[1]['sampler_rng']
   for key,v in states[0]['optimizer']['state'].items():
    for k,t in v.items():assert torch.equal(t,states[1]['optimizer']['state'][key][k])
  for layer,c in smoke['checks'].items():
   for tool in ('sae','transcoder'):
    for k in (4,16):assert c[f'{tool}_k{k}']['updates']==100 and c[f'{tool}_k{k}']['draws']==51200
    assert c[tool+'_patches']['identity']=='bitwise' and c[tool+'_patches']['full']==c[tool+'_patches']['sparse']=='passed'
   assert c['probe']['status']=='passed'
  done=read('training_complete.json');assert done['identity']==dict(config_sha256=ch,input_sha256=ih)
  assert len(done['runs'])==24
  rows=[];selected=[];initialization_replay=[];all_steps=[100]+list(range(250,5001,250));sessions=[]
  for run in config['runs']:
   name=run['name'];prefix=f'runs/{name}';result=read(prefix+'/result.json');curve=read(prefix+'/curve.json');initial=read(prefix+'/initialization.json')
   assert done['runs'][prefix+'/result.json']==sha(z.read(prefix+'/result.json'))
   assert result['run']==initial['run']==run and result['identity']==initial['identity']==done['identity']
   model=Dictionary('sae',run['k'],run['init_seed'],width=256)
   local_digest=tensor_digest(model.state_dict())
   initialization_replay.append(dict(run=name,recorded_sha256=initial['model_sha256'],local_sha256=local_digest,identical=initial['model_sha256']==local_digest,local_torch=torch.__version__,status='cross_environment_diagnostic_not_a_gate'))
   gen=np.random.Generator(np.random.PCG64(run['draw_seed']));assert gen.bit_generator.state==initial['sampler_rng']
   assert [r['update'] for r in curve]==list(range(1,5001)) and all(math.isfinite(r['train_mse']) for r in curve)
   vals=[r for r in curve if 'val_mse' in r];assert [r['update'] for r in vals]==list(range(250,5001,250))
   best=min(vals,key=lambda r:(r['val_mse'],r['update']));assert result['best_update']==best['update'] and result['best_val_mse']==best['val_mse']
   assert result['updates']==5000 and result['draws']==2560000 and result['parameter_count']==262912
   assert set(result['checkpoints'])=={f'update_{u:05d}.pt' for u in all_steps}
   stats=read(f'statistics/seed{run["lm_seed"]}_l{run["layer"]}_h.json');cursor=0
   for step in all_steps:
    member=prefix+f'/update_{step:05d}.pt';b=z.read(member)
    assert sha(b)==result['checkpoints'][Path(member).name]==read(member[:-3]+'.json')['sha256']
    state=load(b);assert state['identity']==done['identity'] and state['run']==run
    assert state['update']==step and state['draws']==step*512 and state['unique_train_positions']==50000
    assert state['curve']==curve[:step]
    while cursor<step:gen.integers(0,50000,size=512,dtype=np.int64);cursor+=1
    assert state['sampler_rng']==gen.bit_generator.state
    assert torch.equal(state['stats']['mu'],torch.tensor(stats['mean'])) and float(state['stats']['scale'])==stats['scale']
    m=state['model'];assert sum(v.numel() for v in m.values())==262912 and all(torch.isfinite(v).all() for v in m.values())
    torch.testing.assert_close(m['decoder'].norm(dim=0),torch.ones(512),atol=2e-6,rtol=2e-6)
    groups=state['optimizer']['param_groups'];assert len(groups)==1
    g=groups[0];assert g['lr']==1e-3 and tuple(g['betas'])==(0.9,0.999) and g['eps']==1e-8 and g['weight_decay']==0
    assert len(state['optimizer']['state'])==4
    for v in state['optimizer']['state'].values():
     assert int(v['step'])==step and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
    rng=state['rng'];random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy']);torch.Generator().set_state(rng['torch'])
    assert rng['cuda'] and all(t.dtype==torch.uint8 and t.numel()>0 for t in rng['cuda'])
    assert all(s['environment_id'] in envs for s in state['sessions'])
    assert state['benchmark']['updates']==100 and state['benchmark']['included_in_budget'] is True and state['benchmark']['seconds_per_update']>0
    if step==result['best_update']:
     dest=ROOT/'experiment_v1_4/p6_selected'/name/Path(member).name;save(dest,b)
     selected.append(dict(run=run,checkpoint=str(dest.relative_to(ROOT)),sha256=sha(b),source_member=member,selected_update=step,validation_mse=result['best_val_mse'],stats=f'statistics/seed{run["lm_seed"]}_l{run["layer"]}_h.json'))
   assert state['train_seconds']==result['train_seconds'] and state['validation_seconds']==result['validation_seconds']
   sessions.append(dict(run=name,sessions=state['sessions']))
   rows.append(dict(run=name,best_update=result['best_update'],val_mse=result['best_val_mse'],updates=5000,draws=2560000,train_seconds=result['train_seconds'],validation_seconds=result['validation_seconds'],seconds_per_update_100=result['benchmark']['seconds_per_update'],peak_vram_bytes=state['peak_vram_bytes']))
   print('passed',name,flush=True)
  failures=[n for n in names if n.startswith('failures/')];assert not failures
  # Preserve compact raw evidence; all payloads remain in the immutable LFS archive.
  for n in names:
   if not n.endswith('.pt'):save(OUT/'returned_metadata'/n,z.read(n))
  with (OUT/'runs.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  result=dict(status='passed_full_return',p6_complete=True,p7_eligible=True,archive_sha256=digest,archive_files=len(names),checksums_verified=len(manifest['files']),runs=24,updates=120000,position_draws=61440000,production_checkpoints=504,scalar_statistics=36,selected_checkpoints=selected,initialization_replay=initialization_replay,sessions=sessions,gpu_smoke=smokes[0],environment_ids=list(envs),training_seconds=sum(r['train_seconds'] for r in rows),validation_seconds=sum(r['validation_seconds'] for r in rows),failed_runs=0,limitations=['Initialization byte replay on local macOS/PyTorch 2.10 differs from recorded Colab Linux/PyTorch 2.11; init identity/seed/code/hash are preserved, cross-environment bitwise replay is not claimed.','Original activation pools remain on Drive; their hashes and scalar fitting were verified by frozen preparation and r2 GPU audit, not recalculated locally.','Recorded train_seconds covers update computation and sampling; checkpoint I/O and complete wall-clock runtime were not separately measured.','SAE semantic/fidelity/causal evaluation is P7 and has not run.'])
  (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
  (OUT/'selected_sae_manifest.json').write_text(json.dumps(dict(config_sha256=ch,input_sha256=ih,source_archive_sha256=digest,models=selected),indent=2)+'\n')
  print('P6 full return verified',flush=True)
if __name__=='__main__':main()
