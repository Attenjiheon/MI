import hashlib,json,shutil,tempfile,zipfile
from pathlib import Path
ROOT=Path.cwd()
OUT=ROOT/'experiment_v1_4/results/p3_audit_20260921_01'
archive=ROOT/'experiment_v1_4/evidence/v1_4_seed0_evidence_20260921T064114169595.zip'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
digest=sha(archive)
assert digest==archive.with_suffix('.zip.sha256').read_text().split()[0]
dest=Path(tempfile.mkdtemp(prefix='mi_v14_p3_audit_',dir='/private/tmp'))
with zipfile.ZipFile(archive) as z:
 names=z.namelist(); assert len(names)==len(set(names))
 files={i.filename for i in z.infolist() if not i.is_dir()}
 sums=json.loads(z.read('checksums.json'))
 assert files==set(sums)|{'checksums.json'}
 for i in z.infolist():
  p=dest/i.filename
  assert p.resolve().is_relative_to(dest.resolve())
  assert (i.external_attr>>16)&0o170000!=0o120000
  if i.is_dir():continue
  p.parent.mkdir(parents=True,exist_ok=True)
  h=hashlib.sha256()
  with z.open(i) as src,p.open('xb') as target:
   for b in iter(lambda:src.read(1024*1024),b''):
    h.update(b);target.write(b)
  if i.filename!='checksums.json':assert h.hexdigest()==sums[i.filename],i.filename
 for name in ('result.json','gate.json','gate_started.json','LATEST.json'):
  shutil.copy2(dest/'deepwide12_read4_seed0'/name,OUT/name)
 shutil.copy2(dest/'verification.json',OUT/'colab_verification.json')
 with zipfile.ZipFile(dest/'gpu_smoke_return.zip','w',zipfile.ZIP_STORED) as target:
  for p in sorted((dest/'gpu_smoke').rglob('*')):
   target.write(p,str(p.relative_to(dest/'gpu_smoke'))+('/' if p.is_dir() else ''))
report=dict(status='passed',archive_sha256=digest,archive_bytes=archive.stat().st_size,files_verified=len(sums),extracted_root=str(dest))
(OUT/'archive_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
