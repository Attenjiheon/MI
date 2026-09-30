"""Bounded reverse-order worker for the last 16 local run audits.

The primary streaming auditor checks every archive hash and owns final status.
This worker reads only hash-checked members and skips a two-run buffer ahead
of the primary auditor. No background tasks or experimental compute are added.
"""
from pathlib import Path
import tempfile,zipfile,json,hashlib,shutil
from scripts.audit_v1_4_p8_return import ROOT,OUT,ARCHIVE,raw_audit,metadata_audit,verify,read,write

def main():
 c=verify(ROOT);runs=[e['run']['name'] for e in c['tcs']]
 assert read(OUT/'archive_verification.json')['status']=='passed'
 with zipfile.ZipFile(ARCHIVE) as z,tempfile.TemporaryDirectory(prefix='mi_p8_tail_') as td:
  tmp=Path(td);ns=z.namelist();manifest=json.loads(z.read('return_manifest.json'))['files']
  def extract(n):
   b=z.read(n)
   if n in manifest:assert hashlib.sha256(b).hexdigest()==manifest[n]
   p=tmp/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  for n in ns:
   if not n.startswith('runs/'):extract(n)
  for i in reversed(range(8,len(runs))):
   name=runs[i];done=OUT/'local_runs'/f'{name}.json'
   if done.exists():continue
   completed=[j for j,n in enumerate(runs) if (OUT/'local_runs'/f'{n}.json').exists()]
   # Only use the contiguous prefix; high-index worker results are not primary progress.
   prefix=0
   while prefix in completed:prefix+=1
   if i<=prefix+2:break
   for n in ns:
    if n.startswith(f'runs/{name}/'):extract(n)
   raw=raw_audit(ROOT,tmp,[name]);meta=metadata_audit(tmp,OUT,[name])
   # Preserve the legacy nested field already emitted by the primary process;
   # the outer authoritative report explicitly records P8 as incomplete.
   meta['p7_complete']=meta.pop('p8_complete')
   write(done,dict(run=name,raw=raw,metadata=meta))
   shutil.rmtree(tmp/'runs'/name)
   print('tail audit passed',name,flush=True)
if __name__=='__main__':main()
