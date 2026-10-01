"""Verify all P9 ZIP members, then audit one tool/run at a time without expanding 5.9 GB."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile,tempfile,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha
from scripts.audit_v1_4_p9_raw import audit as raw_audit
from scripts.audit_v1_4_p9_metadata import audit as metadata_audit
ARCHIVE=ROOT/'experiment_v1_4/evidence/P9/v1_4_p9_evaluation_return_1790842429996719913.zip'
OUT=ROOT/'experiment_v1_4/results/p9_return_audit_20261001_01'
BUNDLE=ROOT/'experiment_v1_4/bundles/P9/v1_4_p9_evaluation_r1.zip'

def extract(z,n,root):
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 digest=sha(ARCHIVE);assert digest==read(ARCHIVE.with_suffix('.sha256.json'))['sha256']
 assert sha(BUNDLE)==read(BUNDLE.with_suffix('.sha256.json'))['sha256']
 with zipfile.ZipFile(ARCHIVE) as z:
  ns=z.namelist();assert len(ns)==len(set(ns))
  assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in ns)
  m=json.loads(z.read('return_manifest.json'));assert set(ns)==set(m['files'])|{'return_manifest.json'}
  assert not [n for n in ns if '/failures/' in n]
  for i,(n,d) in enumerate(m['files'].items()):
   h=hashlib.sha256()
   with z.open(n) as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   assert h.hexdigest()==d,n
   if '/causal/' not in n and not n.endswith(('.npz','.pt')):
    p=OUT/'returned_metadata'/n
    if p.exists():assert sha(p)==d
    else:extract(z,n,OUT/'returned_metadata')
   if i%5000==0:print('hashes',i,'/',len(m['files']),flush=True)
  write(OUT/'returned_metadata/return_manifest.json',m)
  # Return contracts must equal the delivered contracts, independent of local iCloud placeholders.
  with zipfile.ZipFile(BUNDLE) as original:
   for tool in ('sae','tc'):
    assert z.read(f'{tool}/contract.json')==original.read(f'experiment_v1_4/p9_eval_r1/{tool}_contract.json')
  write(OUT/'archive_verification.json',dict(status='passed',archive_sha256=digest,bundle_sha256=sha(BUNDLE),members=len(ns),hashes_verified=len(m['files']),expanded_bytes=sum(i.file_size for i in z.infolist())))
  print('all member hashes verified',flush=True)

def run_tool(tool):
 receipt=read(OUT/'archive_verification.json');assert receipt['archive_sha256']==sha(ARCHIVE)
 from interp_v1_4 import p9_sae_evaluation,p9_tc_evaluation
 ev=p9_sae_evaluation if tool=='sae' else p9_tc_evaluation
 start=time.monotonic();reports=[]
 with tempfile.TemporaryDirectory(prefix='mi_p9_inputs_') as rd,tempfile.TemporaryDirectory(prefix='mi_p9_rows_') as td:
  root,tmp=Path(rd),Path(td)
  with zipfile.ZipFile(BUNDLE) as z:z.extractall(root)
  c=ev.verify(root)
  with zipfile.ZipFile(ARCHIVE) as z:
   ns=z.namelist()
   for n in ns:
    if n.startswith(tool+'/') and not n.startswith(tool+'/runs/'):extract(z,n,tmp)
   for e in c['saes' if tool=='sae' else 'tcs']:
    name=e['run']['name'];done=OUT/'local_runs'/f'{name}.json'
    if done.exists():reports.append(read(done));continue
    for n in ns:
     if n.startswith(f'{tool}/runs/{name}/'):extract(z,n,tmp)
    raw=raw_audit(root,tmp/tool,[name],tool)
    meta=metadata_audit(tmp/tool,OUT,[name],tool,root)
    report=dict(run=name,tool=tool,raw=raw,metadata=meta);write(done,report);reports.append(report)
    shutil.rmtree(tmp/tool/'runs'/name)
    print('local audit passed',name,flush=True)
  write(OUT/f'{tool}_verification.json',dict(status='passed_local_source_reproduction_pending',p9_complete=False,archive_sha256=receipt['archive_sha256'],config_sha256=sha(root/ev.CONTRACT),runs=reports,elapsed_seconds=time.monotonic()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tool',choices=['sae','tc']);a=p.parse_args()
 if a.tool:run_tool(a.tool)
 else:prepare()
