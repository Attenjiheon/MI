"""Audit saved frozen-test measurements without model inference."""
from pathlib import Path, PurePosixPath
import json,hashlib,zipfile,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from interp_v1_4.runtime import sha
from interp_v1_4.frozen_test import verify_return, CONTRACT, FREEZE, FREEZE_SHA
OUT=Path(__file__).parent
ARCHIVE=ROOT/'experiment_v1_4/evidence/v1_4_frozen_test_evidence_20260922T033233833147.zip'
def save(name,value):
 (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
with zipfile.ZipFile(ARCHIVE) as z:
 names=z.namelist();assert len(names)==len(set(names))
 checks=json.loads(z.read('checksums.json'));assert set(names)==set(checks)|{'checksums.json'}
 for name in names:
  p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
  assert (z.getinfo(name).external_attr>>16)&0o170000!=0o120000
  data=z.read(name)
  if name in checks:assert hashlib.sha256(data).hexdigest()==checks[name],name
  dest=OUT/'returned'/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==data
  else:dest.write_bytes(data)
sidecar=ARCHIVE.with_suffix('.zip.sha256')
if sidecar.exists():assert sidecar.read_text().split()[0]==sha(ARCHIVE)
save('archive_verification.json',dict(status='passed',archive=str(ARCHIVE.relative_to(ROOT)),sha256=sha(ARCHIVE),files_verified=len(checks),outer_checksum_supplied=sidecar.exists()))
base=OUT/'returned';run=base/'run';contract=json.loads((ROOT/CONTRACT).read_text())
attempts=list((run/'notebook_attempts').iterdir());assert len(attempts)==1
attempt=attempts[0]
assert sha(attempt/'contract.json')==sha(ROOT/CONTRACT)
assert sha(attempt/'validation_freeze.json')==FREEZE_SHA
prep=json.loads((ROOT/'experiment_v1_4/results/frozen_test_preparation_r1/verification.json').read_text())
assert json.loads((attempt/'input_bundle.json').read_text())['sha256']==prep['bundle_sha256']
for name,digest in contract['files'].items():
 if name.startswith(('interp_v1_4/','interp_v1_2/','corpus/')):assert sha(base/'runtime_code'/name)==digest,name
for name in ('test_error.txt','preflight_error.txt'):assert not (attempt/name).exists()
smoke=json.loads((attempt/'gpu_preflight/preflight.json').read_text())
assert smoke['status']=='passed' and smoke['scope']=='cuda' and smoke['debug_only'] and not smoke['test_scores_observed']
assert smoke['contract_sha256']==sha(ROOT/CONTRACT) and smoke['microbatch']==16 and smoke['maximum_length_fixture']==302
assert sha(attempt/'gpu_preflight/requirements.lock.txt')==smoke['environment']['lock_sha256']
cases=ET.parse(attempt/'tests.xml').getroot().findall('.//testcase')
assert len(cases)==6 and all(not list(c) for c in cases)
print('Archive, frozen bindings, runtime source, GPU preflight, six tests verified.',flush=True)
report=verify_return(ROOT,run)
assert json.loads((attempt/'verification.json').read_text())==report
report.update(archive_sha256=sha(ARCHIVE),runtime_source_verified=True,gpu_preflight_verified=True,
              environment_id=smoke['environment']['environment_id'],gpu=smoke['environment']['gpu'],recorded_notebook_attempts=1)
save('verification.json',report)
print(json.dumps(report,indent=2),flush=True)
