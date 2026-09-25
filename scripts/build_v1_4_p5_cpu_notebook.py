"""Standalone CPU-only notebook using the unchanged r2 bundle from Drive."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
helper=(ROOT/'scripts/p5_cpu_resume.py').read_text()
transfer=(ROOT/'scripts/p5_transfer.py').read_text()
v=json.loads((ROOT/'experiment_v1_4/results/p5_preparation_r2/verification.json').read_text())
cells=[]
def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text))
def code(text):compile(text,'P5 CPU notebook','exec');cells.append(dict(cell_type='code',metadata={},source=text,execution_count=None,outputs=[]))
md('''# P5 r2 — Drive 입력 · 안전 재개 · CPU probe 전용

기존 GPU 추출 결과와 완료 probe를 이어갑니다. **GPU 추출 셀은 없습니다.** 기존 노트북에서 실행 중인 probe/복사 셀을 중단하고 이 노트북만 실행하세요. 같은 Drive 폴더에 두 실행을 동시에 쓰지 마세요.

1. 기존 `v1_4_p5_bundle_r2.zip`을 Google Drive `내 드라이브/boolean_interp_v1_4/`에 한 번 올립니다. 새 bundle은 필요 없습니다. 이미 다른 위치에 올렸으면 첫 셀의 경로만 바꿉니다.
2. Drive mount → bundle 로컬 복사·checksum → 의존성 → 안전 복구 → CPU probe 순서입니다.
3. CPU 런타임으로 충분합니다. 런타임 변경 시 로컬 파일은 사라질 수 있으나 Drive 저장 결과를 재개합니다.
4. 이번 변경은 전달·복구 절차만 바꿉니다. 동결된 r2 실험 코드/라벨/선택 규칙을 변경하지 않습니다.
''')
code('''from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, zipfile
# 이미 mount되어 있다는 문구는 오류가 아닙니다.
drive.mount('/content/drive')
BUNDLE_ON_DRIVE=Path('/content/drive/MyDrive/boolean_interp_v1_4/v1_4_p5_bundle_r2.zip')
# 내 드라이브 최상위에 올린 경우 자동으로 찾습니다.
if not BUNDLE_ON_DRIVE.exists():
    alternative=Path('/content/drive/MyDrive/v1_4_p5_bundle_r2.zip')
    if alternative.exists():BUNDLE_ON_DRIVE=alternative
assert BUNDLE_ON_DRIVE.is_file(), 'BUNDLE_ON_DRIVE를 Drive에 올린 r2 ZIP 경로로 수정하세요.'
ROOT=Path('/content/boolean_interp')
OUT=ROOT/'experiment_v1_4/runs/p5_r2'
PERSIST=Path('/content/drive/MyDrive/boolean_interp_v1_4/P5_r2')
assert (PERSIST/'contract.json').exists(), '기존 추출 결과의 P5_r2 Drive 경로를 확인하세요.'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
print('Drive bundle:',BUNDLE_ON_DRIVE)
print('기존 결과:',PERSIST)
''')
code(f'''EXPECTED_BUNDLE_SHA={v['bundle_sha256']!r}
ROOT.mkdir(parents=True,exist_ok=True)
local_bundle=Path('/content/v1_4_p5_bundle_r2.zip')
if not local_bundle.exists() or sha(local_bundle)!=EXPECTED_BUNDLE_SHA:
    temp=local_bundle.with_suffix('.zip.copy.tmp')
    shutil.copyfile(BUNDLE_ON_DRIVE,temp)
    assert sha(temp)==EXPECTED_BUNDLE_SHA, 'Drive bundle checksum 불일치: r2 원본 ZIP을 확인하세요.'
    temp.replace(local_bundle)
input_backup=PERSIST.parent/'P5_input_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
with zipfile.ZipFile(local_bundle) as z:
    inventory=json.loads(z.read('bundle_manifest.json'))['files']
    assert len(z.namelist())==len(set(z.namelist()))
    assert set(z.namelist())==set(inventory)|{{'bundle_manifest.json'}}
    for name,digest in inventory.items():
        dest=ROOT/name
        assert dest.resolve().is_relative_to(ROOT.resolve()), '잘못된 ZIP 경로'
        if dest.exists() and sha(dest)==digest:continue
        if dest.exists():
            backup=input_backup/name;backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(dest,backup);assert sha(dest)==sha(backup)
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(dest.suffix+'.input.tmp')
        with z.open(name) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
        assert sha(temp)==digest, '내부 checksum 불일치: '+name
        temp.replace(dest)
os.chdir(ROOT)
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
print('Drive → 로컬 입력 검증 완료. 이미 일치하는 파일은 재사용했습니다.')
''')
md('## 의존성 준비\n기존 r2의 실제 패키지 lock을 사용합니다. CPU 계산은 GPU를 요구하지 않습니다. 새 런타임에서는 의존성 다운로드가 한 번 필요할 수 있습니다. 패키지 변경 후 Colab이 재시작을 요구하면 재시작하고 첫 셀부터 다시 실행하세요.')
code("subprocess.run([sys.executable,'-m','pip','install','-r',str(ROOT/'experiment_v1_4/frozen_test_r1/requirements-primary.lock.txt'),'--extra-index-url','https://download.pytorch.org/whl/cu128'],check=True)")
md('''## 영속 저장·안전 재개

라벨은 동결 corpus에서 독립 재생성하고, cache marker가 요구하는 label hash와 일치하는지 먼저 확인합니다. 손상·공백 차이 등으로 다른 기존 라벨은 `P5_resume_backups/실행시각/`에 원본과 hash를 보존한 뒤 복구합니다. 해당 원인은 같은 폴더의 `repair.jsonl`에 기록됩니다.

Cache는 marker SHA256과 일치하는 복사본을 사용합니다. 완료 probe는 task/config/구조를 확인해 재사용합니다. **양쪽 모두 유효한데 서로 다른 결과이면 임의로 선택하지 않고 백업 후 중단합니다.** 이 경우 오류 메시지와 repair.jsonl을 전달하세요. Drive 파일을 매번 읽고 hash를 계산하므로 이 셀도 시간이 걸릴 수 있지만 이미 동일한 로컬 cache는 다시 복사하지 않습니다.
''')
code("helper_path=Path('/content/p5_cpu_resume.py')\nhelper_path.write_text("+repr(helper)+")\nnamespace={'__name__':'p5_cpu_resume'}\nexec(compile(helper_path.read_text(),str(helper_path),'exec'),namespace)\nbackup_path=namespace['restore'](ROOT,OUT,PERSIST)\nprint('복구 기록:',backup_path)")
md('''## CPU probe만 실행

GPU 추출은 실행하지 않습니다. 기존 완료 파일을 건너뛰고 남은 probe만 계산합니다. 시작 전 전체 cache를 검증하므로 첫 probe 로그가 나오기까지 시간이 걸릴 수 있습니다. 전체 범위는 `[0,1,2]`로 유지합니다. 이 셀이 중단되면 앞의 안전 재개 셀을 실행하고 다시 이어가세요.
''')
code('''SEEDS=[0,1,2]
# 실행 실패/중단 때에도 CPU 환경 session 증빙을 Drive로 복사합니다.
try:
    subprocess.run([sys.executable,'-m','interp_v1_4.p5','probes','--root',str(ROOT),
                    '--output',str(OUT),'--persistent',str(PERSIST),'--seeds',*map(str,SEEDS)],check=True)
finally:
    from interp_v1_4.runtime import verified_copy
    for src in (OUT/'sessions').rglob('*'):
        if src.is_file():
            dest=PERSIST/src.relative_to(OUT)
            if dest.exists() and sha(src)!=sha(dest):
                print('session 충돌 보존; 다음 안전 재개 셀에서 검증:',dest)
            else:verified_copy(src,dest)
''')
md('## 작은 metadata만 회수\nCPU probe 도중 중단했거나 완료했을 때 실행할 수 있습니다. 15GB 전체 evidence ZIP을 만들지 않습니다. 결과 ZIP/index를 전달하면 진행률과 오류를 확인할 수 있습니다. 실제 cache는 Drive에 보존합니다.')
code("transfer_namespace={'__name__':'p5_transfer'}\nexec(compile("+repr(transfer)+",'p5_transfer','exec'),transfer_namespace)\nEXPORT=PERSIST.parent/('P5_metadata_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))\nindex=transfer_namespace['pack'](PERSIST,EXPORT)\nprint(json.dumps(index,indent=2))\nfiles.download(str(EXPORT/'p5_metadata_index.json'))")
code("PART=1\nfiles.download(str(EXPORT/index['parts'][PART-1]['name']))")
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),colab=dict(name='P5_v1_4_CPU_Drive_resume_r1.ipynb')),cells=cells)
path=ROOT/'experiment_v1_4/notebooks/P5_v1_4_CPU_Drive_resume_r1.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
print(path)
