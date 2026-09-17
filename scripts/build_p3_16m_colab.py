"""Build an immutable v1.2 Colab input and ordered executable notebook."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def build():
    bundle=ROOT/'experiment_v1_2/bundles/p3_16m_bundle_v2.zip'
    notebook=ROOT/'experiment_v1_2/notebooks/01_colab_lm_16m.ipynb'
    if bundle.exists(): raise FileExistsError('Increment bundle version; never overwrite')
    files=set()
    for folder in ('corpus','interp_v1_2','tests_v1_2','data/language_v1_2','experiment_v1_2/configs','experiment_v1_2/debug'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store')
    files.update((ROOT/'data/language_v1/train_shards').glob('*.tokens.jsonl'))
    for name in ('interp_v1_1/__init__.py','interp_v1_1/model.py','requirements-interp.txt','AGENTS.md','phase.md',
                 '01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md','experiment_v1_2/DESIGN.md',
                 'experiment_v1_2/design_config.json','experiment_v1_2/P1_STATUS.md','experiment_v1_2/P2_STATUS.md','experiment_v1_2/P3_STATUS.md','experiment_v1_2/results/p1_generation.json','experiment_v1_2/results/p1_release_validation.json','experiment_v1_2/results/p1_base_reaudit.json',
                 'scripts/verify_p3_16m_evidence.py','experiment_v1_1/results/p3_stop_report.md'):
        files.add(ROOT/name)
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files): z.write(path,path.relative_to(ROOT))
        z.writestr('p3_bundle_manifest.json',json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},indent=2))
    digest=sha(bundle); bundle.with_suffix('.sha256').write_text(digest+'  '+bundle.name+'\n')
    cells=[]
    def md(text): cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
    def code(text): cells.append(dict(cell_type='code',metadata={},source=text.splitlines(True),execution_count=None,outputs=[]))
    md('''# P3 v1.2 — 4-block, fresh seed 0, 고정 16M

새 Colab GPU 런타임에서 위부터 실행합니다. 입력 파일은 `p3_16m_bundle_v2.zip`입니다.
기존 3M train prefix와 모든 평가 데이터를 그대로 보존한 새 corpus입니다. **2,160 updates / 16,001,083 prediction tokens**를 한 번씩 소비합니다.
1M·3M·8M에서는 validation만 기록하고, 조기 gate 통과/실패와 관계없이 16M까지 실행합니다.
전체 validation에서 일반 READ 답 CE가 가장 낮은 checkpoint 하나를 선택합니다. 일반 full-vocabulary 정확도 ≥99%, 세 진단 각각 ≥95%여야 통과합니다.

모든 validation checkpoint를 개별 보존하며 새 파일만 Drive에 복사합니다. 완료 index 갱신 전 복사가 끊긴 파일은 복구에서 제외됩니다.
중단 시 **새 런타임**에서 `RESUME=True`로 바꾸어 같은 입력 ZIP과 Drive 경로를 사용합니다. 기존 v1.1 checkpoint는 사용할 수 없습니다.
노트북/로컬 검증 완료와 GPU 본실험 완료는 다릅니다. 마지막 결과 ZIP을 반환하여 로컬 증빙 감사를 받아야 P3 판정을 확정합니다.
''')
    code('''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, zipfile
from google.colab import drive, files

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

drive.mount('/content/drive')
RESUME=False  # 중단된 v1.2를 재개할 때 True, 새 런타임에서 실행
DRIVE_ROOT=Path('/content/drive/MyDrive/boolean_interp_p3_16m_v1_2')
WORK=Path('/content/boolean_interp')
RUN=WORK/'experiment_v1_2/runs/lm_seed_0'
PERSISTENT=DRIVE_ROOT/'lm_seed_0'
SESSION=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
DRIVE_ROOT.mkdir(parents=True,exist_ok=True)
if WORK.exists(): raise RuntimeError('작업 경로가 있습니다. 기존 결과를 보존하고 새 런타임에서 실행하세요.')
if not RESUME and (PERSISTENT/'LATEST.json').exists(): raise RuntimeError('기존 v1.2 run이 있습니다. RESUME=True로 복구하세요.')
if RESUME and not (PERSISTENT/'LATEST.json').exists(): raise RuntimeError('완료된 저장 index가 없습니다.')
''')
    code(f'''EXPECTED_SHA256={digest!r}
BUNDLE_NAME={bundle.name!r}
saved=DRIVE_ROOT/BUNDLE_NAME
if not saved.exists():
    uploaded=files.upload()
    matches=[data for name,data in uploaded.items() if hashlib.sha256(data).hexdigest()==EXPECTED_SHA256]
    if not matches: raise RuntimeError('SHA-256이 일치하는 v1.2 입력 ZIP이 없습니다.')
    tmp=saved.with_suffix('.zip.tmp'); tmp.write_bytes(matches[0])
    assert sha(tmp)==EXPECTED_SHA256
    tmp.replace(saved)
    del uploaded,matches
assert sha(saved)==EXPECTED_SHA256, 'Drive 입력 ZIP checksum 불일치'
WORK.mkdir(parents=True)
with zipfile.ZipFile(saved) as z:
    for info in z.infolist():
        if not (WORK/info.filename).resolve().is_relative_to(WORK.resolve()): raise RuntimeError('잘못된 ZIP 경로')
    z.extractall(WORK)
for name,expected in json.loads((WORK/'p3_bundle_manifest.json').read_text()).items():
    assert sha(WORK/name)==expected, name
os.chdir(WORK)
print('모든 입력 checksum 검증 완료')
''')
    md('''## 의존성과 새 환경 기록
처음에는 직접 의존성을 설치합니다. 재개에서는 이전 세션의 직접 의존성 버전을 먼저 복원합니다.
설치가 실패하면 버전을 임의로 바꾸지 말고 로그를 보존합니다. Colab 이미지/CUDA 변화는 새로운 환경 ID와 smoke로 기록합니다.
Drive는 약 160개 checkpoint와 결과 ZIP을 저장할 수 있어야 합니다(대략 4GB 이상 여유 권장).''')
    code('''deps=['torch','numpy','scipy','pandas','matplotlib','pytest']
if RESUME:
    previous=sorted((DRIVE_ROOT/'preflight').glob('*/direct_versions.json'))
    if not previous: raise RuntimeError('이전 직접 의존성 버전 기록이 없습니다.')
    versions=json.loads(previous[-1].read_text())
    specs=[name+'=='+versions[name] for name in deps]
    subprocess.run([sys.executable,'-m','pip','install',*specs],check=True)
else:
    subprocess.run([sys.executable,'-m','pip','install','-r','requirements-interp.txt'],check=True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
import importlib.metadata
import torch
assert torch.cuda.is_available(), 'GPU 런타임이 필요합니다.'
from interp_v1_2.runtime import environment,verify_inputs,verified_copy
PREFLIGHT=WORK/'experiment_v1_2/preflight'/SESSION
PREFLIGHT.mkdir(parents=True,exist_ok=True)
current=environment(PREFLIGHT)
(PREFLIGHT/'environment.json').write_text(json.dumps(current,indent=2))
(PREFLIGHT/'direct_versions.json').write_text(json.dumps({n:importlib.metadata.version(n) for n in deps},indent=2))
print(json.dumps(current,indent=2)); print(verify_inputs(WORK))
assert shutil.disk_usage(WORK).free>4*1024**3, '로컬 디스크 여유 부족'
''')
    md('''## CPU 테스트 → CPU smoke → GPU smoke
별도 debug 데이터로만 검증합니다. CPU 학습 재개와 optimizer/RNG의 일치, 16M 경계와 shard 연결, checkpoint 보존을 검사합니다.
GPU smoke가 통과해야 본학습이 시작됩니다. 이 셀의 실패는 P3 성공이 아닙니다.''')
    code('''with (PREFLIGHT/'pytest.txt').open('w') as log:
    tested=subprocess.run([sys.executable,'-m','pytest','tests_v1_2','-q'],stdout=log,stderr=subprocess.STDOUT)
print((PREFLIGHT/'pytest.txt').read_text()); tested.check_returncode()
for device in ('cpu','cuda'):
    subprocess.run([sys.executable,'-u','-m','interp_v1_2.smoke','--device',device,'--output',str(PREFLIGHT/device)],check=True)
GPU_SMOKE=PREFLIGHT/'cuda/smoke.json'
assert json.loads(GPU_SMOKE.read_text())['status']=='passed'
for source in PREFLIGHT.rglob('*'):
    if source.is_file(): verified_copy(source,DRIVE_ROOT/'preflight'/SESSION/source.relative_to(PREFLIGHT))
PREFLIGHT_PASSED=True
''')
    md('''## 실행·영속 저장·재개
첫 50 updates의 처리량 측정은 본실험 예산에 포함됩니다. validation마다 현재 CE/정확도가 출력됩니다.
`checkpoints/init.pt`, `checkpoints/update_*.pt`, `events/`, `milestones/`, `LATEST.json`을 보존합니다.
저장 완료 index가 last 참조이고 각 checkpoint의 `best` 및 최종 `result.json`의 `selected`가 best 참조입니다.
런타임이 끊기면 새 런타임에서 복구합니다. 마지막 완전 저장 update 이후의 계산만 다시 하므로 학습 토큰을 중복 계상하지 않습니다.''')
    code('''assert PREFLIGHT_PASSED
from interp_v1_2.persistence import recover
resume=recover(PERSISTENT,RUN) if RESUME else None
if (RUN/'result.json').exists():
    print('완료된 run을 복구했습니다. 재학습 없이 결과를 내보냅니다.')
else:
    command=[sys.executable,'-u','-m','interp_v1_2.cli','--device','cuda','--output',str(RUN),
             '--persistent-dir',str(PERSISTENT),'--smoke-report',str(GPU_SMOKE)]
    if resume: command+=['--resume',str(resume)]
    executed=subprocess.run(command)
    if executed.returncode:
        raise RuntimeError('학습 중단. Drive 마지막 완료 index를 보존했습니다. 새 런타임에서 재개하거나 오류를 확인하세요.')
result=json.loads((RUN/'result.json').read_text())
print(json.dumps({'status':result['status'],'final_state':result['final_state'],'selected':result['selected']},indent=2))
''')
    md('''## 결과 계약 검사 및 current / best-so-far 곡선
모든 validation checkpoint의 바이트 hash, 160개 평가 경계, 선택 및 milestone 기록을 검사합니다.
선택 checkpoint validation을 다시 평가하며 test는 읽지 않습니다. 최종 통과/실패는 최소 CE checkpoint 기준입니다.''')
    code('''audit=WORK/'experiment_v1_2/results'/('gpu_run_audit_'+SESSION+'.json')
audit.parent.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'scripts/verify_p3_16m_evidence.py',str(RUN),'--output',str(audit),'--reevaluate','--device','cuda'],check=True)
import matplotlib.pyplot as plt
entries=[json.loads(p.read_text()) for p in sorted((RUN/'events').glob('*.json'))]
evals=[e for e in entries if 'validation' in e]
x=[e['state']['prediction_tokens'] for e in evals]
current_ce=[e['validation']['general']['answer_ce'] for e in evals]
best=[]; chosen=None
for e in evals:
    if chosen is None or e['validation']['general']['answer_ce']<chosen['validation']['general']['answer_ce']: chosen=e
    best.append(chosen)
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(x,current_ce,label='Current CE')
axes[0].plot(x,[e['validation']['general']['answer_ce'] for e in best],label='Best-so-far CE',linestyle='--')
for name in ('general',*('other_variable','repeated_update','first_read_after_set')):
    values=[e['validation']['general']['correct'] if name=='general' else e['validation']['diagnostics'][name]['correct'] for e in evals]
    axes[1].plot(x,values,label=name)
axes[1].plot(x,[e['validation']['general']['correct'] for e in best],label='Selected general',linestyle='--')
for ax in axes: ax.set_xlabel('Prediction tokens'); ax.grid(alpha=.2); ax.legend(fontsize=7)
axes[0].set_ylabel('Answer CE'); axes[1].set_ylabel('Full-vocabulary accuracy')
fig.tight_layout(); plot=audit.with_suffix('.png'); fig.savefig(plot,dpi=150); plt.show()
''')
    md('''## 전체 증빙 다운로드
ZIP에는 init과 모든 validation checkpoint, 로그·milestone·최종 선택, CPU/GPU smoke와 환경 기록이 포함됩니다.
파일이 크므로 다운로드 완료를 기다립니다. ZIP과 아래 SHA-256을 Codex에 전달합니다.
`failed`이면 16M 행동 gate 미달이며 P4/SAE/TC로 진행하지 않습니다. `passed`여도 로컬 감사를 마치기 전 P3 완료로 표시하지 않습니다.''')
    code('''export=DRIVE_ROOT/('p3_16m_evidence_'+SESSION+'.zip')
with zipfile.ZipFile(export,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for source in RUN.rglob('*'):
        if source.is_file() and source.suffix!='.tmp': z.write(source,'lm_seed_0/'+str(source.relative_to(RUN)))
    for source in (DRIVE_ROOT/'preflight').rglob('*'):
        if source.is_file(): z.write(source,'preflight/'+str(source.relative_to(DRIVE_ROOT/'preflight')))
    z.write(WORK/'p3_bundle_manifest.json','p3_bundle_manifest.json')
    z.write(audit,'verification/'+audit.name); z.write(plot,'verification/'+plot.name)
checksum=sha(export); export.with_suffix('.sha256').write_text(checksum+'  '+export.name+chr(10))
print('결과 ZIP:',export.name); print('SHA-256:',checksum)
files.download(str(export))
''')
    for i,cell in enumerate(cells):
        cell['id']=f'v12-{i:02d}'
        if cell['cell_type']=='code': compile(''.join(cell['source']),f'cell-{i}','exec')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name=notebook.name),kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),cells=cells)
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
    report=dict(bundle=str(bundle.relative_to(ROOT)),sha256=digest,bytes=bundle.stat().st_size,notebook=str(notebook.relative_to(ROOT)),notebook_sha256=sha(notebook),files=len(files),code_cells_compiled=True)
    (ROOT/'experiment_v1_2/results/colab_delivery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': build()
