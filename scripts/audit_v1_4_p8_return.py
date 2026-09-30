"""Streaming ZIP verification and one-run-at-a-time P8 local return audit."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile,tempfile,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha
from interp_v1_4.p8_evaluation import verify,CONTRACT
from scripts.audit_v1_4_p8_raw import audit as raw_audit
from scripts.audit_v1_4_p8_metadata import audit as metadata_audit
ARCHIVE=ROOT/'experiment_v1_4/evidence/P8/v1_4_p8_evaluation_return_1790665484889058353.zip'
OUT=ROOT/'experiment_v1_4/results/p8_return_audit_20260929_01'

def main():
 OUT.mkdir(parents=True,exist_ok=True);c=verify(ROOT);digest=sha(ARCHIVE)
 assert digest==read(ARCHIVE.with_suffix('.sha256.json'))['sha256']
 start=time.monotonic()
 with zipfile.ZipFile(ARCHIVE) as z:
  ns=z.namelist();assert len(ns)==len(set(ns))
  assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in ns)
  manifest=json.loads(z.read('return_manifest.json'));assert set(ns)==set(manifest['files'])|{'return_manifest.json'}
  assert json.loads(z.read('contract.json'))==c
  failures=[n for n in ns if n.startswith('failures/')];assert not failures,failures
  for i,(name,d) in enumerate(manifest['files'].items()):
   h=hashlib.sha256()
   with z.open(name) as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   assert h.hexdigest()==d,name
   # Keep compact original evidence, leaving large causal rows and arrays in ZIP.
   if '/causal/' not in name and not name.endswith(('.npz','.pt')):
    p=OUT/'returned_metadata'/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert sha(p)==d
    else:p.write_bytes(z.read(name))
   if i%5000==0:print('member hashes',i,'/',len(manifest['files']),flush=True)
  write(OUT/'returned_metadata/return_manifest.json',manifest)
  write(OUT/'archive_verification.json',dict(status='passed',archive_sha256=digest,members=len(ns),hashes_verified=len(manifest['files']),expanded_bytes=sum(i.file_size for i in z.infolist())))
  print('all member hashes verified',flush=True)
  reports=[]
  with tempfile.TemporaryDirectory(prefix='mi_p8_audit_') as td:
   tmp=Path(td)
   for n in ns:
    if not n.startswith('runs/'):
     p=tmp/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
   for e in c['tcs']:
    name=e['run']['name'];done=OUT/'local_runs'/f'{name}.json'
    if done.exists():reports.append(read(done));continue
    for n in ns:
     if n.startswith(f'runs/{name}/'):
      p=tmp/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
    raw=raw_audit(ROOT,tmp,[name]);meta=metadata_audit(tmp,OUT,[name])
    report=dict(run=name,raw=raw,metadata=meta);write(done,report);reports.append(report)
    shutil.rmtree(tmp/'runs'/name)
    print('local audit passed',name,flush=True)
  write(OUT/'verification.json',dict(status='passed_local_source_reproduction_pending',p8_complete=False,p9_eligible=False,archive_sha256=digest,config_sha256=sha(ROOT/CONTRACT),hashes_verified=len(manifest['files']),runs=reports,elapsed_seconds=time.monotonic()-start,remaining='Original cache refits, predictions/fidelity/CI reproduction and fixed GPU replay must be returned and verified.'))
if __name__=='__main__':main()
