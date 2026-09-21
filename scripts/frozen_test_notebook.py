"""Colab notebook for the frozen all-seed P4 reporting contract."""


def notebook(bundle_name,digest):
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text))
    def code(text):cells.append(dict(cell_type='code',metadata={},source=text,outputs=[],execution_count=None))
    md('''# v1.4 P4 · 세 seed frozen test

새 **CUDA GPU 런타임**에서 위에서 아래로 실행하세요. 업로드는 `v1_4_frozen_test_bundle_r1.zip` 하나입니다.
이미 선택·동결된 seed 0/1의 update 7,983, seed 2의 update 8,399를 사용합니다.
각 seed의 일반 READ, legacy 3종, first/repeat 42-cell, composition 2종을 총 21회 평가합니다.
Length는 P10 범위입니다. Test는 최종 보고이며 checkpoint 선택이나 새로운 gate가 아닙니다.

학습은 실행하지 않습니다. GPU 사전 검증은 별도의 **미학습 모델과 debug fixture**만 사용합니다.
실제 평가의 raw 결과·환경·checksum은 고정 Drive 경로에 저장합니다.
**동일 노트북을 동시에 실행하거나 Drive 실행 폴더를 삭제·변경하지 마세요.**
재접속하면 같은 노트북을 위에서부터 실행하세요. 완료된 평가를 재사용하고,
원시 결과만 저장된 평가는 보고서 생성만 이어갑니다. 시작 기록만 남은 평가는 자동 반복하지 않습니다.
이 경우 오류 증빙 ZIP을 반환하세요. 런타임/패키지 환경이 달라져도 자동으로 덮어쓰지 않습니다.

마지막 다운로드 ZIP과 `.sha256`을 반환해야 로컬 감사를 거쳐 P4를 완료할 수 있습니다.
''')
    md('## 1. 입력 업로드 · ZIP 및 모든 내부 파일 checksum 검사')
    code(f'''from google.colab import files, drive
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, subprocess, sys, zipfile, traceback
from datetime import datetime, timezone
BUNDLE_NAME = {bundle_name!r}
EXPECTED_SHA256 = {digest!r}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
os.chdir('/content')
bundle=Path('/content')/BUNDLE_NAME
if not bundle.exists():
    uploaded=files.upload()
    del uploaded
assert bundle.exists() and sha(bundle)==EXPECTED_SHA256, 'Input ZIP checksum mismatch'
ROOT=Path('/content/boolean_interp_v1_4_frozen_test_r1')
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    names=z.namelist()
    assert len(names)==len(set(names))
    for name in names:
        p=PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and not name.endswith('/')
        assert (z.getinfo(name).external_attr >> 16) & 0o170000 != 0o120000
    manifest=json.loads(z.read('bundle_manifest.json'))
    assert set(names)==set(manifest['files'])|{{'bundle_manifest.json'}}
    for name,expected in manifest['files'].items():
        dst=ROOT/name
        if dst.exists():
            assert sha(dst)==expected, 'Existing extracted input changed: '+name
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        with z.open(name) as src, dst.open('xb') as out: shutil.copyfileobj(src,out,1024*1024)
        assert sha(dst)==expected, name
(ROOT/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
os.chdir(ROOT)
print('Input verified:', len(manifest['files']), 'files')
''')
    md('''## 2. 기록된 seed 0 환경으로 설치

주요 패키지는 정확한 버전, 나머지는 원본 lock의 constraints를 적용합니다.
설치 실패 시 다른 버전으로 임의 변경하지 말고 오류를 반환하세요.
''')
    code('''subprocess.run([sys.executable,'-m','pip','install',
    '-r','experiment_v1_4/frozen_test_r1/requirements-primary.lock.txt',
    '-c','experiment_v1_4/frozen_test_r1/requirements-seed0.constraints.txt',
    '--extra-index-url','https://download.pytorch.org/whl/cu128'],check=True)
subprocess.run(['nvidia-smi'],check=True)
subprocess.run([sys.executable,'-c',"import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"],check=True)
''')
    md('## 3. Drive 연결 · 고정 실행 경로 · 실패 증빙 저장')
    code('''drive.mount('/content/drive')
PERSIST=Path('/content/drive/MyDrive/MI/v1_4/frozen_test_r1')
PERSIST.mkdir(parents=True,exist_ok=True)
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
ATTEMPT=PERSIST/'notebook_attempts'/STAMP
ATTEMPT.mkdir(parents=True,exist_ok=False)
(ATTEMPT/'input_bundle.json').write_text(json.dumps(dict(name=BUNDLE_NAME,sha256=EXPECTED_SHA256)))
shutil.copyfile(ROOT/'experiment_v1_4/frozen_test_r1/contract.json',ATTEMPT/'contract.json')
shutil.copyfile(ROOT/'experiment_v1_4/results/p4_audit_20260921_01/validation_freeze.json',ATTEMPT/'validation_freeze.json')
def logged_run(command,name):
    with (ATTEMPT/name).open('w') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='');log.write(line);log.flush()
        code=proc.wait()
    if code: raise RuntimeError(f'{name} exited {code}; export evidence below')
READY=False
try:
    logged_run([sys.executable,'-m','pytest','tests_v1_4/test_frozen_test.py','-q',
                '--junitxml='+str(ATTEMPT/'tests.xml')],'tests.log')
    logged_run([sys.executable,'-m','interp_v1_4.frozen_test','preflight','--device','cuda',
                '--output',str(ATTEMPT/'gpu_preflight')],'gpu_preflight.log')
    READY=True
except Exception:
    (ATTEMPT/'preflight_error.txt').write_text(traceback.format_exc())
    print(traceback.format_exc())
''')
    md('''## 4. 동결된 세 seed 순차 평가 · 저장된 결과 검증

21개 평가 완료 여부를 출력합니다. 결과 검증은 저장된 원시 측정값의 집계만 재계산합니다.
오류가 출력되어도 다음 다운로드 셀을 실행해 부분 증빙을 반환하세요.
''')
    code('''if READY:
    try:
        logged_run([sys.executable,'-m','interp_v1_4.frozen_test','run','--output',str(PERSIST),
                    '--smoke',str(ATTEMPT/'gpu_preflight/preflight.json')],'test_run.log')
        logged_run([sys.executable,'-m','interp_v1_4.frozen_test','verify','--output',str(PERSIST)],'verification.json')
        print('21 evaluations saved and verified. Export evidence for local audit; P4 is not yet marked complete.')
    except Exception:
        (ATTEMPT/'test_error.txt').write_text(traceback.format_exc())
        print(traceback.format_exc())
else:
    print('Preflight failed. No production evaluation was started. Export evidence below.')
''')
    md('''## 5. 전체/부분 증빙 ZIP 다운로드

완료·미완료 모두 이 셀을 실행하세요. Drive에도 ZIP 사본을 저장합니다.
이 셀을 실행하는 동안 다른 런타임에서 평가하지 마세요.
''')
    code('''EXPORT=Path('/content')/('v1_4_frozen_test_evidence_'+STAMP)
EXPORT.mkdir(exist_ok=False)
shutil.copytree(PERSIST,EXPORT/'run')
shutil.copyfile(ROOT/'bundle_manifest.json',EXPORT/'input_bundle_manifest.json')
shutil.copytree(ROOT/'interp_v1_4',EXPORT/'runtime_code/interp_v1_4',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(ROOT/'interp_v1_2',EXPORT/'runtime_code/interp_v1_2',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(ROOT/'corpus',EXPORT/'runtime_code/corpus',ignore=shutil.ignore_patterns('__pycache__'))
inventory={str(p.relative_to(EXPORT)):sha(p) for p in sorted(EXPORT.rglob('*')) if p.is_file()}
(EXPORT/'checksums.json').write_text(json.dumps(inventory,indent=2))
archive=EXPORT.with_suffix('.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in sorted(EXPORT.rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(EXPORT)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,digest in inventory.items(): assert hashlib.sha256(z.read(name)).hexdigest()==digest
sidecar=archive.with_suffix('.zip.sha256')
sidecar.write_text(sha(archive)+'  '+archive.name+'\\n')
saved=PERSIST.parent/'frozen_test_exports';saved.mkdir(exist_ok=True)
for p in (archive,sidecar):
    target=saved/p.name
    assert not target.exists()
    shutil.copyfile(p,target)
    assert sha(target)==sha(p)
print('Evidence:',archive.name, 'bytes:',archive.stat().st_size)
files.download(str(archive))
files.download(str(sidecar))
''')
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)
