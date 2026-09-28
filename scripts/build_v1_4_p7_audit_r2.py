"""Ship a read-only P7 source reproduction audit, preserving original contracts."""
from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
 if path.exists():assert path.read_text()==payload,str(path)
 else:path.write_text(payload)

def build():
 base=ROOT/'experiment_v1_4/p7_audit_r2';base.mkdir(parents=True,exist_ok=True)
 returned=ROOT/'experiment_v1_4/results/p7_return_audit_20260927_01/returned'
 manifest=(returned/'return_manifest.json').read_bytes();dest=base/'return_manifest.json'
 if dest.exists():assert dest.read_bytes()==manifest
 else:dest.write_bytes(manifest)
 paths=[ROOT/'scripts/audit_v1_4_p7_sources.py',ROOT/'scripts/build_v1_4_p7_audit_r2.py',ROOT/'tests_v1_4/test_p7_source_audit.py',dest]
 contract=dict(schema='p7-source-return-audit-v1.4-r2',date='2026-09-27',original_contract_sha256=sha(ROOT/'experiment_v1_4/p7_r1/contract.json'),return_zip_sha256='2d8d65b3bca7b6a0ec09d6f524d78c3495239ee2cf4a96fbc59a5187bf943449',
  scope='Original cache checksum and labels; all novel refits; every test prediction and semantic CI; full train/val/test fidelity; full causal val bins; fixed first pair per test suite and first 16 reconstruction targets per run; all summary CIs and layer contrasts.',
  changes_to_experiment='none: audit only; no new feature/checkpoint/threshold selection applied',numeric_tolerance=dict(metrics_atol=2e-6,metrics_rtol=2e-5,logits_atol=1e-5,logits_rtol=1e-4),replay_sampling='first frozen corpus order, never selected by effect',files={str(p.relative_to(ROOT)):sha(p) for p in paths})
 save(base/'contract.json',contract);paths.append(base/'contract.json')
 old=ROOT/'experiment_v1_4/bundles/P7/v1_4_p7_bundle_r1.zip';bundle=ROOT/'experiment_v1_4/bundles/P7/v1_4_p7_source_audit_r2.zip'
 if not bundle.exists():
  with zipfile.ZipFile(old) as src,zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as out:
   for info in src.infolist():out.writestr(info.filename,src.read(info.filename))
   for p in paths:out.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(bundle) as z:
  oldc=json.loads(z.read('experiment_v1_4/p7_r1/contract.json'))
  for name,digest in {**oldc['files'],**contract['files'],'experiment_v1_4/p7_r1/contract.json':contract['original_contract_sha256'],'experiment_v1_4/p7_audit_r2/contract.json':sha(base/'contract.json')}.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
 digest=sha(bundle);save(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
 cells=[]
 def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
 def code(s):
  compile(s,'P7 source audit','exec');cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
 md('''# P7 반환 후 원본 재현 검산 r2

본평가 24개를 다시 학습하지 않습니다. 기존 P5 cache와 P7 결과를 읽어 선택·예측·fidelity·CI를 재현합니다. 원래 feature/threshold/checkpoint를 바꾸지 않으며 불일치는 실패로 보존합니다.

**T4 GPU 런타임**에서 위부터 실행하세요. CPU probe refit와 1,000회 bootstrap 때문에 수 시간이 걸릴 수 있습니다. 결과는 Drive `boolean_interp_v1_4/P7_source_audit_r2`에 저장하며, 같은 노트북으로 개별 refit/완료 run부터 재개할 수 있습니다. 기존 P7 산출물은 읽기만 합니다. 검산 결과를 반환한 뒤 P7 완료 여부를 판정합니다.''')
 code(f'''from google.colab import files, drive
from pathlib import Path
import hashlib, zipfile, subprocess, sys, os, time, json
uploaded=files.upload()  # v1_4_p7_source_audit_r2.zip 하나
bundle=Path('v1_4_p7_source_audit_r2.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='{digest}', 'ZIP checksum 불일치'
ROOT=Path('/content/MI_P7_audit_r2');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for name in z.namelist():assert (ROOT/name).resolve().is_relative_to(ROOT.resolve())
    z.extractall(ROOT)
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/boolean_interp_v1_4')
SOURCE=BASE/'P5_r2';RETURNED=BASE/'P7_r1';OUT=BASE/'P7_source_audit_r2';OUT.mkdir(exist_ok=True)
assert (SOURCE/'contract.json').exists() and (RETURNED/'evaluation_complete.json').exists(), '기존 Drive 경로를 확인하세요'
''')
 md('## 1. 기존 실행과 같은 수치 라이브러리 및 환경 기록\n설치 후 재시작 안내가 나오면 런타임을 재시작하고 첫 셀부터 다시 실행합니다.')
 code('''os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
log=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','pytest==8.4.2','matplotlib==3.10.8','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(OUT/f'install_{time.time_ns()}.txt').write_text(log.stdout);print(log.stdout);log.check_returncode()
subprocess.run([sys.executable,'-m','pytest','tests_v1_4/test_p7.py','tests_v1_4/test_p7_source_audit.py','-q'],cwd=ROOT,check=True)
''')
 md('''## 2. 원본 재현 감사

원본 반환 파일 56,919개의 checksum과 P5 cache hash를 먼저 검사합니다. 기존 P5 감사 기준선은 재사용하고, 새로운 probe는 train/validation에서 재계산해 저장된 선택과 비교합니다. Test는 저장된 계수와 선택으로만 평가합니다.

전체 test 예측/의미 CI, 80,000개 위치의 fidelity와 train dead 비율, 전체 causal validation bin을 확인합니다. GPU 패칭 재현 표본은 효과를 보지 않고 고정 corpus 순서의 suite별 첫 pair와 general test 첫 16개입니다. 전체 causal/replacement raw 수치는 로컬 반환 검사 대상으로, 전체 summary CI와 층간 비교는 여기서 다시 계산합니다.''')
 code("subprocess.run([sys.executable,str(ROOT/'scripts/audit_v1_4_p7_sources.py'),'--root',str(ROOT),'--source',str(SOURCE),'--returned',str(RETURNED),'--output',str(OUT)],cwd=ROOT,check=True)\n")
 md('## 3. 완료·중단 결과 다운로드\n위 셀이 실패해도 이 셀을 실행할 수 있습니다. 아래 ZIP과 checksum JSON을 이 작업에 전달하세요. 완료되지 않은 검산을 성공으로 표시하지 않습니다.')
 code('''archive=Path(f'/content/v1_4_p7_source_audit_return_{time.time_ns()}.zip')
subprocess.run([sys.executable,'-m','interp_v1_4.p7_runner','export','--output',str(OUT),'--archive',str(archive)],cwd=ROOT,check=True)
files.download(str(archive));files.download(str(archive.with_suffix('.sha256.json')))
''')
 nb=ROOT/'experiment_v1_4/notebooks/P7/P7_source_return_audit_r2.ipynb'
 save(nb,dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',name='python3',language='python'),language_info=dict(name='python'),accelerator='GPU'),cells=[dict(c,id=f'p7-audit-{i}') for i,c in enumerate(cells)]))
 print(json.dumps(dict(bundle=str(bundle),notebook=str(nb),bytes=bundle.stat().st_size,sha256=digest),indent=2))

if __name__=='__main__':build()
