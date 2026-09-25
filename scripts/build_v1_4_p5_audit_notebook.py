"""Package read-only independent audit and CPU Colab notebook."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiment_v1_4/results/p5_independent_audit_preparation_r1';OUT.mkdir(parents=True,exist_ok=True)
BUNDLE=ROOT/'experiment_v1_4/bundles/v1_4_p5_independent_audit_r1.zip'
def digest(b):return hashlib.sha256(b).hexdigest()
contract=ROOT/'experiment_v1_4/p5_r2/contract.json';c=json.loads(contract.read_text())
with zipfile.ZipFile('/Users/jangjiheon/Desktop/p5_metadata_001.zip') as z:reference=z.read('transfer_manifest.json')
assert json.loads(reference)['contract_sha256']==digest(contract.read_bytes())
payload={'p5_independent_audit.py':(ROOT/'scripts/p5_independent_audit.py').read_bytes(),'inputs/contract.json':contract.read_bytes(),'inputs/reference_manifest.json':reference}
for split in c['quotas']:
 for suffix in ['jsonl.gz','read_positions.json']:
  name=f"{c['data_root']}/interpretation/{split}.{suffix}";data=(ROOT/name).read_bytes();assert digest(data)==c['files'][name];payload['inputs/'+name]=data
manifest=dict(schema='p5-independent-audit-input-r1',contract_sha256=digest(contract.read_bytes()),files={k:digest(v) for k,v in payload.items()})
with zipfile.ZipFile(BUNDLE,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,v in payload.items():z.writestr(k,v)
 z.writestr('audit_bundle_manifest.json',json.dumps(manifest,indent=2))
sha=digest(BUNDLE.read_bytes());BUNDLE.with_suffix('.zip.sha256').write_text(sha+'  '+BUNDLE.name+'\n')
cells=[]
def md(t):cells.append(dict(cell_type='markdown',metadata={},source=t))
def code(t):compile(t,'audit notebook','exec');cells.append(dict(cell_type='code',metadata={},source=t,execution_count=None,outputs=[]))
md('''# P5 원본 activation 독립 검산 r1 — CPU 전용

기존 Drive `boolean_interp_v1_4/P5_r2`의 cache와 완료 probe를 **읽기 전용**으로 감사합니다. 모델 추론·GPU 추출·probe 재학습·test 기반 재선택은 없습니다. 현재 노트북은 검산 준비물이며 실제 완료 증빙은 실행 후 검토해야 합니다.

순서: Drive 연결 → 작은 감사 ZIP 입력/checksum → 의존성/환경 → 원본 cache 및 전체 3,510 probe 검산 → 작은 결과 ZIP 다운로드.

- 기존 P5 폴더의 파일을 이동하거나 변경하지 마세요. 이 노트북과 동시에 원본 폴더에 쓰는 실행은 종료하세요.
- CPU 런타임으로 충분합니다. 로컬 임시 디스크 5GB 이상 권장. Cache를 seed/kind 단위로 약 3GB씩 처리해 전체 17.7GB 로컬 복사 공간은 필요 없습니다. 실제 NPZ는 Drive에서 읽습니다.
- 모든 선택 결과의 AUROC 및 1,000회 sequence bootstrap을 독립 재계산하므로 시간이 걸립니다. 표본 수·draw 수는 줄이지 않습니다. 첫 작업 로그로 처리량을 확인하세요.
- Drive 별도 `P5_independent_audit_r1`에 작업별 원자적 완료 증빙을 저장합니다. 중단되면 같은 노트북을 처음부터 다시 실행합니다. 완료 작업은 audit/source hash를 확인해 건너뛰고, 미완료 cache 묶음은 다시 hash 검증합니다.
- 선택되지 않은 lambda/prefix의 계수는 원본에 없어 해당 후보의 validation 값은 저장 trace로만 검사합니다. 선택 계수의 train 통계·validation 예측/threshold·test 점수와 CI는 원본 배열로 직접 검사합니다.
''')
code('''from google.colab import drive, files
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile
from datetime import datetime, timezone
drive.mount('/content/drive')
SOURCE=Path('/content/drive/MyDrive/boolean_interp_v1_4/P5_r2')
AUDIT=Path('/content/drive/MyDrive/boolean_interp_v1_4/P5_independent_audit_r1')
ROOT=Path('/content/p5_independent_audit_r1')
WORK=Path('/content/p5_audit_scratch_r1')
BUNDLE_ON_DRIVE=Path('/content/drive/MyDrive/boolean_interp_v1_4/v1_4_p5_independent_audit_r1.zip')
assert (SOURCE/'contract.json').is_file(), 'SOURCE를 기존 P5_r2 cache 폴더로 수정하세요.'
assert shutil.disk_usage('/content').free > 5*1024**3, '로컬 임시 디스크 5GB 이상 필요'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
print('읽기 전용 원본:',SOURCE)
print('새 감사 결과:',AUDIT)
''')
md('## 감사 입력 ZIP\n함께 받은 `v1_4_p5_independent_audit_r1.zip`을 위 Drive 경로에 넣으면 업로드를 건너뜁니다. 없으면 아래 셀에서 이 작은 ZIP 한 개만 업로드합니다. 기존 대용량 LM 번들은 필요 없습니다.')
code(f'''EXPECTED_SHA={sha!r}
NAME={BUNDLE.name!r}
local=Path('/content')/NAME
if BUNDLE_ON_DRIVE.exists():
    shutil.copyfile(BUNDLE_ON_DRIVE,local)
elif not local.exists() or sha(local)!=EXPECTED_SHA:
    uploaded=files.upload()
    assert NAME in uploaded, '감사 ZIP 파일명을 확인하세요.'
    local.write_bytes(uploaded[NAME])
assert sha(local)==EXPECTED_SHA, '감사 ZIP SHA256 불일치'
ROOT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(local) as z:
    inventory=json.loads(z.read('audit_bundle_manifest.json'))['files']
    assert len(z.namelist())==len(set(z.namelist()))
    assert set(z.namelist())==set(inventory)|{{'audit_bundle_manifest.json'}}
    for name,digest in inventory.items():
        dest=ROOT/name
        assert dest.resolve().is_relative_to(ROOT.resolve())
        data=z.read(name);assert hashlib.sha256(data).hexdigest()==digest
        if dest.exists():assert sha(dest)==digest, '기존 감사 코드/입력 충돌: 새 런타임을 사용하세요.'
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
print('감사 입력 checksum 검증 완료')
''')
md('## 의존성 설치\n감사기는 production probe 코드를 import하지 않습니다. NumPy/SciPy만 고정 설치하며, 실제 Python/패키지 환경은 실행별로 Drive에 기록합니다.')
code("subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3'],check=True)")
md('''## 전체 검산 실행·저장·재개

`cache ...`는 원본 NPZ checksum·row join·모든 hook의 shape/dtype/유한성 확인입니다. `PASS task ... seconds`는 해당 작업의 전처리·예측·평가·CI 검산 완료입니다. 첫 PASS 전에도 원본 metadata checksum/코퍼스 join/첫 cache 묶음 확인에 시간이 걸립니다.

숫자 불일치가 나면 허용 오차를 임의로 늘리거나 원본 결과를 수정하지 마세요. 실패 증빙은 별도로 저장됩니다. 아래 다운로드 셀은 실패/중단 후에도 실행할 수 있습니다.
''')
code('''env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='2'
subprocess.run([sys.executable,'-u',str(ROOT/'p5_independent_audit.py'),
    '--source',str(SOURCE),'--inputs',str(ROOT/'inputs'),
    '--output',str(AUDIT),'--work',str(WORK)],env=env,check=True)
print(json.loads((AUDIT/'completion.json').read_text()))
''')
md('## 증빙 다운로드\n원본 activation은 ZIP에 넣지 않습니다. 새 검산 결과·실패·실행 환경과 manifest만 다운로드합니다. **ZIP과 SHA256 파일을 함께 반환**하면 로컬에서 검산 범위와 완료 여부를 확인합니다. `completion.json`이 없으면 부분 실행이며 P5 완료로 처리하지 않습니다.')
code('''assert AUDIT.exists(), '실행 셀을 먼저 시작하세요.'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
archive=Path('/content')/f'p5_independent_audit_{stamp}.zip'
items=sorted(p for p in AUDIT.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
inventory={str(p.relative_to(AUDIT)):sha(p) for p in items}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in items:z.write(p,str(p.relative_to(AUDIT)))
    z.writestr('return_manifest.json',json.dumps(dict(schema='p5-independent-audit-return-r1',files=inventory),indent=2))
checksum=archive.with_suffix('.zip.sha256');checksum.write_text(sha(archive)+'  '+archive.name+'\\n')
print('완료 marker:',(AUDIT/'completion.json').exists(),'작업 증빙:',len(list((AUDIT/'tasks').glob('*.json'))))
print(archive,archive.stat().st_size)
files.download(str(archive))
''')
code("files.download(str(checksum))")
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),colab=dict(name='P5_v1_4_independent_audit_r1.ipynb')),cells=cells)
for i,cell in enumerate(cells):cell['id']=f'audit-{i:02d}'
path=ROOT/'experiment_v1_4/notebooks/P5_v1_4_independent_audit_r1.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
(OUT/'bundle_verification.json').write_text(json.dumps(dict(bundle=str(BUNDLE.relative_to(ROOT)),bundle_sha256=sha,bundle_bytes=BUNDLE.stat().st_size,files=len(payload),notebook_sha256=digest(path.read_bytes()),script_sha256=manifest['files']['p5_independent_audit.py'],production_executed=False,p5_complete=False),indent=2)+'\n')
print(path);print(BUNDLE,BUNDLE.stat().st_size,sha)
