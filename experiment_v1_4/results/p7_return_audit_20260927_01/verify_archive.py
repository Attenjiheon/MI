"""Read-only verification of P7 delivered ZIP and every manifest member."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
root=Path(__file__).resolve().parents[3]
archive=root/'experiment_v1_4/evidence/v1_4_p7_return_1790455082926962917.zip'
expected=json.loads(archive.with_suffix('.sha256.json').read_text());h=hashlib.sha256()
with archive.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==expected['sha256'] and archive.stat().st_size==expected['bytes']
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('return_manifest.json'));names=z.namelist()
 assert len(names)==len(set(names)) and set(names)==set(manifest['files'])|{'return_manifest.json'}
 for name,digest in manifest['files'].items():
  path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
  assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
print('Passed archive checksum and',len(manifest['files']),'member checksums')
