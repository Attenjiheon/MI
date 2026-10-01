"""Immutable overlay for P9 original-cache source reproduction audit."""
from pathlib import Path
import json,zipfile,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.runtime import sha
from interp_v1_4.p5 import read,write
BASE='experiment_v1_4/p9_audit_r1'

def build():
 base=ROOT/BASE;base.mkdir(parents=True,exist_ok=True)
 returned=ROOT/'experiment_v1_4/results/p9_return_audit_20261001_01'
 receipt=read(returned/'archive_verification.json');assert receipt['status']=='passed'
 manifest=(returned/'returned_metadata/return_manifest.json').read_bytes();dest=base/'return_manifest.json'
 if dest.exists():assert dest.read_bytes()==manifest
 else:dest.write_bytes(manifest)
 paths=[ROOT/n for n in ('scripts/audit_v1_4_p9_sources.py','scripts/audit_v1_4_p9_sae_sources.py','scripts/audit_v1_4_p9_tc_sources.py','scripts/build_v1_4_p9_audit_r1.py','tests_v1_4/test_p9_source_audit.py')]+[dest]
 contracts={t:sha(ROOT/f'experiment_v1_4/p9_eval_r1/{t}_contract.json') for t in ('sae','tc')}
 audit=dict(schema='p9-source-return-audit-v1.4-r1',date='2026-10-01',original_contract_sha256=contracts,return_zip_sha256=receipt['archive_sha256'],
  scope='All novel refits; all test probabilities, semantic metrics/CI; full train/val/test fidelity/dead rates; full causal validation bins; first frozen pair per test suite and first 16 reconstruction targets per run; all summary CI, layer contrasts and sparse seed comparison.',
  runs=16,novel_refits_expected=560,probe_tasks_expected=1200,fidelity_positions_per_run=80000,causal_replayed_pairs_per_run=4,replacement_replayed_targets_per_run=16,
  numeric_tolerance=dict(metrics_atol=2e-6,metrics_rtol=2e-5,logits_atol=1e-5,logits_rtol=1e-4),replay_sampling='fixed original corpus order, independent of effects',changes_to_experiment='none; audit does not change model/feature/threshold/candidates/bins',
  files={str(p.relative_to(ROOT)):sha(p) for p in paths})
 write(base/'contract.json',audit);paths.append(base/'contract.json')
 old=ROOT/'experiment_v1_4/bundles/P9/v1_4_p9_evaluation_r1.zip';assert sha(old)==receipt['bundle_sha256']
 bundle=ROOT/'experiment_v1_4/bundles/P9/v1_4_p9_source_audit_r1.zip'
 if not bundle.exists():
  with zipfile.ZipFile(old) as src,zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as out:
   for i in src.infolist():out.writestr(i.filename,src.read(i.filename))
   for p in paths:
    assert str(p.relative_to(ROOT)) not in src.namelist();out.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(bundle) as z:
  for tool in ('sae','tc'):
   c=json.loads(z.read(f'experiment_v1_4/p9_eval_r1/{tool}_contract.json'))
   for n,d in c['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==d,n
  for p in paths:assert hashlib.sha256(z.read(str(p.relative_to(ROOT)))).hexdigest()==sha(p)
 checksum=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=checksum,bytes=bundle.stat().st_size))
 cells=[]
 def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
 def code(s):
  compile(s,'p9 source audit','exec');cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
 md('''# P9 — 평가 반환 후 원본 cache 재현 감사

SAE 8개·TC 8개를 재학습하지 않습니다. 기존 평가의 feature·threshold·checkpoint·bin을 바꾸지 않고 원본 P5 cache에서 재현합니다.

입력 `v1_4_p9_source_audit_r1.zip`을 내 드라이브 `boolean_interp_v1_4/`에 업로드하고 **기존 평가와 같은 T4 GPU 런타임**에서 위부터 실행하세요.
기존 `P5_r2`, `P9_evaluation_r1/sae`, `P9_evaluation_r1/tc`를 읽고 새 결과를 `P9_source_audit_r1`에 저장합니다.

CPU probe refit와 bootstrap 때문에 수 시간이 걸릴 수 있습니다. 개별 refit·완료 run을 저장하며 같은 노트북으로 재개합니다. 환경 ID와 로그를 보존합니다.
완료 또는 실패 후 마지막 셀에서 ZIP·checksum을 내려받아 반환하세요. 실제 감사 반환을 확인하기 전에는 P9를 완료하지 않습니다.''')
 code(f'''from google.colab import drive,files
from pathlib import Path
import hashlib,zipfile,subprocess,sys,time,os,json,shutil
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/boolean_interp_v1_4')
bundle=Path('/content/v1_4_p9_source_audit_r1.zip')
def checksum(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
EXPECTED='{checksum}'
if not bundle.exists() or checksum(bundle)!=EXPECTED:shutil.copyfile(BASE/bundle.name,bundle)
assert checksum(bundle)==EXPECTED, '입력 checksum 불일치'
ROOT=Path('/content/MI_P9_source_audit_r1');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for n in z.namelist():assert (ROOT/n).resolve().is_relative_to(ROOT.resolve())
    z.extractall(ROOT)
SOURCE=BASE/'P5_r2';RETURNED=BASE/'P9_evaluation_r1';OUT=BASE/'P9_source_audit_r1';OUT.mkdir(exist_ok=True)
assert (SOURCE/'contract.json').exists()
assert all((RETURNED/t/'evaluation_complete.json').exists() for t in ('sae','tc')), '기존 평가 경로 확인 필요'
''')
 md('## 1. 수치 라이브러리·환경 준비\n기존 평가 버전을 설치하고 설치 로그·환경 lock·GPU 정보를 보존합니다. 재시작 안내가 나오면 런타임을 재시작하고 첫 셀부터 실행하세요.')
 code('''os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
log=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','pytest==8.4.2','matplotlib==3.10.8','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(OUT/f'install_{time.time_ns()}.txt').write_text(log.stdout);print(log.stdout);log.check_returncode()
subprocess.run([sys.executable,'-m','pytest','tests_v1_4/test_p9_source_audit.py','tests_v1_4/test_p9_sae_evaluation.py','tests_v1_4/test_p9_tc_evaluation.py','-q'],cwd=ROOT,check=True)
''')
 md('''## 2. 원본 재현 감사

반환 파일 38,080개와 P5 원본 hash를 검증합니다. 두 도구 합계 560개 신규 고유 probe를 다시 fit하며, 재사용 포함 1,200 probe 작업의 선택·계수·threshold와 모든 test 예측·의미 CI를 대조합니다.
각 모델의 train/val/test 80,000 위치 fidelity와 train dead 비율, 전체 causal validation bins를 확인합니다.

GPU 패칭은 효과와 무관하게 고정된 suite별 첫 pair 4개와 general READ 첫 16개를 모델마다 재현합니다. 모든 인과·대체 CI, 층간 차이와 seed 0/1 비교표도 다시 계산합니다.
**전체 인과 본평가를 다시 실행하는 것이 아닙니다.** 불일치는 실패로 보존하고 선택값을 변경하지 않습니다. 단계별 로그는 OUT/logs에 저장됩니다.''')
 code('''command=[sys.executable,'-u',str(ROOT/'scripts/audit_v1_4_p9_sources.py'),'--root',str(ROOT),'--source',str(SOURCE),'--returned',str(RETURNED),'--output',str(OUT)]
logs=OUT/'logs';logs.mkdir(exist_ok=True)
with (logs/f'audit_{time.time_ns()}.log').open('w',buffering=1) as log:
    proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in proc.stdout:log.write(line);print(line,end='',flush=True)
        code=proc.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        proc.stdout.close()
''')
 md('## 3. 완료·중단 결과 다운로드\n위 셀이 실패해도 실행할 수 있습니다. 아래 ZIP과 checksum JSON을 이 작업에 전달하세요. 기존 평가 파일은 수정하지 않습니다.')
 code('''archive=Path(f'/content/v1_4_p9_source_audit_return_{time.time_ns()}.zip')
subprocess.run([sys.executable,'-m','interp_v1_4.p9_sae_runner','export','--output',str(OUT),'--archive',str(archive)],cwd=ROOT,check=True)
files.download(str(archive));files.download(str(archive.with_suffix('.sha256.json')))
''')
 nb=ROOT/'experiment_v1_4/notebooks/P9/P9_source_return_audit_r1.ipynb'
 write(nb,dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',name='python3',language='python'),language_info=dict(name='python'),accelerator='GPU'),cells=[dict(c,id=f'p9-source-{i}') for i,c in enumerate(cells)]))
 print(json.dumps(dict(bundle=str(bundle),notebook=str(nb),bytes=bundle.stat().st_size,sha256=checksum),indent=2))

if __name__=='__main__':build()
