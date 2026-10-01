"""Finalize local return checks; P9 still requires original-cache reproduction."""
from pathlib import Path
import json,csv,tempfile,zipfile,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha
from scripts.audit_v1_4_p9_return import OUT,BUNDLE
from scripts.audit_v1_4_p9_sources import compare_writes
from interp_v1_4 import p9_comparison


def main():
 reports={t:read(OUT/f'{t}_verification.json') for t in ('sae','tc')}
 archive=read(OUT/'archive_verification.json');returned=OUT/'returned_metadata'
 for t,r in reports.items():
  assert r['status']=='passed_local_source_reproduction_pending' and len(r['runs'])==8
  assert r['archive_sha256']==archive['archive_sha256']
 with tempfile.TemporaryDirectory(prefix='mi_p9_comparison_') as td:
  root=Path(td)
  with zipfile.ZipFile(BUNDLE) as z:z.extractall(root)
  with compare_writes(p9_comparison):p9_comparison.aggregate(root,returned)
  with (returned/'tables/sparse_seed_comparison.csv').open() as f:rows=list(csv.DictReader(f))
  for r in rows:
   assert int(r['lm_seed'])==0
   if r['status']=='NA':assert r['difference_seed1_minus_seed0']==''
   else:assert abs(float(r['difference_seed1_minus_seed0'])-(float(r['seed1'])-float(r['seed0'])))<1e-12
  assert len(rows)==read(returned/'comparison_complete.json')['rows']
  # Connect every table point to its immutable per-run report, without GPU inference.
  checked_tables=0
  for tool in reports:
   folder=returned/tool
   configs=read(root/f'experiment_v1_4/p9_eval_r1/{tool}_contract.json')
   entries=configs['saes' if tool=='sae' else 'tcs']
   loaded={e['run']['name']:(read(folder/'runs'/e['run']['name']/'semantic.json'),read(folder/'runs'/e['run']['name']/'summary.json')) for e in entries}
   for kind in ('semantic','fidelity','causal','replacement'):
    with (folder/'tables'/f'{kind}.csv').open() as f:table=list(csv.DictReader(f))
    for row in table:
     assert row['lm_seed']=='0' and row['sparse_seed']=='1'
     name=f"seed0_l{row['layer']}_{tool}_k{row['k']}_s1";sem,summary=loaded[name]
     if kind=='semantic':
      metric=sem['semantic'][row['task']]['evaluation'][row['size']]
      values={k:metric[k] for k in ('balanced_accuracy','macro_f1','binary_auroc')}
     elif kind=='fidelity':
      metric=sem['fidelity'][row['split']];values={k:metric[k] for k in ('mse','nmse','r2','ev','inactive_fraction')};values['l0']=metric['l0']['mean']
     else:
      metric=(summary['causal'][row['task']] if kind=='causal' else summary['replacement'])['metrics'][row['metric']]
      values=dict(mean=metric['mean'],ci_low=metric['ci95'][0],ci_high=metric['ci95'][1])
     for k,v in values.items():
      if v is None:assert row[k]==''
      else:assert float(row[k])==v,(name,kind,k)
     checked_tables+=1
 counts=dict(runs=16,causal_pairs=0,replacement_targets=0,probe_tasks=0,causal_rows=0,summary_point_estimates=0)
 envs=set()
 for report in reports.values():
  for run in report['runs']:
   raw=run['raw']['runs'][0];meta=run['metadata']['runs'][0]
   assert raw['run']==meta['run']==run['run'] and meta['status']=='passed'
   counts['causal_pairs']+=raw['pairs'];counts['replacement_targets']+=raw['replacement_targets']
   counts['probe_tasks']+=meta['selection_tasks'];counts['causal_rows']+=meta['causal_rows'];counts['summary_point_estimates']+=meta['summary_point_estimates']
   envs.add(run['metadata']['gpu_environment_id'])
 status=dict(status='passed_local_source_reproduction_pending',p9_complete=False,p10_eligible=False,whole_experiment_complete=False,**counts,
  archive_sha256=archive['archive_sha256'],hashes_verified=archive['hashes_verified'],environment_ids=sorted(envs),table_rows_verified=checked_tables,initialization_comparison_rows=len(rows),
  evidence_sha256={n:sha(OUT/n) for n in ('archive_verification.json','sae_verification.json','tc_verification.json')},
  code_sha256={n:sha(ROOT/n) for n in ('scripts/audit_v1_4_p9_return.py','scripts/audit_v1_4_p9_raw.py','scripts/audit_v1_4_p9_metadata.py','scripts/finalize_v1_4_p9_local_audit.py')},
  limitations=['Original-cache refits, complete prediction/fidelity/CI reproduction and fixed GPU replay are pending; no P9 completion.','Sparse seed difference table contains point estimates, not a new LM replication or cross-dictionary latent ID correspondence.'])
 write(OUT/'status.json',status);print(json.dumps(status,indent=2))

if __name__=='__main__':main()
