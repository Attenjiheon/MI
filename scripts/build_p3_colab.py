"""Build the versioned P3 handoff without downloading or regenerating corpus data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'experiment_v1/bundles/p3_colab_bundle_v4.zip'
NOTEBOOK=ROOT/'experiment_v1/notebooks/01_colab_lm.ipynb'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    files=set()
    for folder in ('corpus','interp','tests','data/language_v1','experiment_v1/configs'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ('AGENTS.md','phase.md','01_experiment_design.md','02_language_and_corpus.md',
                 '03_experiment_spec.md','requirements-interp.txt','experiment_v1/debug/sequences.json',
                 'experiment_v1/results/run_registry.csv','experiment_v1/P2_STATUS.md',
                 'experiment_v1/environment/runtime_manifest.json','experiment_v1/environment/requirements-colab.lock.txt',
                 'experiment_v1/smoke/colab_gpu_01/smoke.json'):
        files.add(ROOT/name)
    if BUNDLE.exists():
        raise FileExistsError('Versioned bundles are immutable; increment the version for a new build')
    with zipfile.ZipFile(BUNDLE,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files): z.write(path,path.relative_to(ROOT))
        z.writestr('p3_bundle_manifest.json',json.dumps({str(p.relative_to(ROOT)):digest(p) for p in sorted(files)},indent=2))
    checksum=digest(BUNDLE)
    BUNDLE.with_suffix('.sha256').write_text(f'{checksum}  {BUNDLE.name}\n')
    cells=[]
    def md(text): cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
    def code(text): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True)))
    md('''# P3 — LM seed 0 파일럿

**새 Colab GPU 런타임**에서 위부터 순서대로 실행하세요. 입력은 함께 전달한 `p3_colab_bundle_v4.zip`입니다.
P1 CPU 데이터와 P2 증빙을 확인하고 현재 코드로 GPU smoke를 다시 통과한 다음 본실험을 시작합니다.
일반 validation 최소 답 CE checkpoint에 대해 일반 정확도 ≥99%, 진단 3개 각각 ≥95%를 판정합니다.
1M 미달 시 마지막 4평가의 3변화 중 2개 이상 CE 개선 ≥1e-4인 경우에만 동일 상태로 3M까지 한 번 연장합니다.
Test 평가와 P4는 이 노트북에서 실행하지 않습니다. 결과 ZIP을 Codex에 전달한 후 증빙 판정을 진행합니다.

Drive에 입력·checkpoint·로그를 보관합니다. 복구 시에는 `RESUME=True`로 설정하고 **새 런타임에서** 같은 Drive 경로를 사용하세요.
환경이 바뀌면 새로운 environment ID와 GPU smoke 증빙을 저장합니다. 1M/3M 전체를 완주하기 전 자원 중단은 완료가 아닙니다.
''')
    code('''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile
from datetime import datetime, timezone
from google.colab import drive, files

drive.mount('/content/drive')
RESUME = False  # 중단한 run을 이어갈 때 True; 새 런타임에서 실행
DRIVE_ROOT = Path('/content/drive/MyDrive/boolean_interp_p3_v4')
WORK = Path('/content/boolean_interp')
RUN = WORK / 'experiment_v1/runs/lm_seed_0'
PERSISTENT = DRIVE_ROOT / 'lm_seed_0'
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
if not RESUME and (PERSISTENT / 'LATEST.json').exists():
    raise RuntimeError('기존 run이 있습니다. 새 런타임에서 RESUME=True로 재개하세요.')
if WORK.exists():
    raise RuntimeError('작업 경로가 이미 있습니다. 기존 파일을 보존하고 새 런타임에서 실행하세요.')
''')
    code(f'''BUNDLE_NAME = {BUNDLE.name!r}
EXPECTED_SHA256 = {checksum!r}
saved_bundle = DRIVE_ROOT / BUNDLE_NAME
if not saved_bundle.exists():
    print(BUNDLE_NAME + ' 파일을 업로드하세요.')
    uploaded = files.upload()
    # Colab may rename repeated uploads, e.g. 'p3_colab_bundle_v4 (3).zip'.
    # Identify the immutable bundle by its content hash, not its returned filename.
    matches = [(name, data) for name, data in uploaded.items()
               if hashlib.sha256(data).hexdigest() == EXPECTED_SHA256]
    if not matches:
        raise RuntimeError('업로드한 파일 중 checksum이 일치하는 v4 번들이 없습니다. 전달받은 ZIP을 선택하세요.')
    uploaded_name, bundle_data = matches[0]
    temporary = saved_bundle.with_suffix('.zip.tmp')
    temporary.write_bytes(bundle_data)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('Drive 복사 checksum 불일치')
    temporary.replace(saved_bundle)
    print('번들 확인 완료:', uploaded_name)
if hashlib.sha256(saved_bundle.read_bytes()).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError('Drive 번들 checksum 불일치')
WORK.mkdir(parents=True)
with zipfile.ZipFile(saved_bundle) as z:
    for info in z.infolist():
        target = (WORK / info.filename).resolve()
        if not target.is_relative_to(WORK.resolve()):
            raise RuntimeError('번들 경로 오류')
    z.extractall(WORK)
for name, expected in json.loads((WORK / 'p3_bundle_manifest.json').read_text()).items():
    if hashlib.sha256((WORK / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('입력 checksum 불일치: ' + name)
os.chdir(WORK)
print('번들 모든 파일 checksum 검증 통과')
''')
    md('''## 의존성·GPU 환경 검증
`requirements-colab.lock.txt`는 P2 전체 환경의 기록입니다. Colab 기본 패키지 전체를 재설치하는 입력으로 사용하지 않습니다.
직접 의존성만 설치하고 실제 환경을 새로 기록합니다. GPU smoke 실패 시 학습 셀로 진행하지 마세요.''')
    code('''subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-interp.txt'], check=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
import torch
assert torch.cuda.is_available(), '런타임 유형을 GPU로 변경하세요.'
from interp.runtime import verify_inputs, environment, verified_copy
from interp.persistence import recover
print(verify_inputs(WORK))
SESSION = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
ENV_DIR = WORK / 'experiment_v1/environment' / SESSION
ENV_DIR.mkdir()
current = environment(ENV_DIR)
(ENV_DIR / 'runtime.json').write_text(json.dumps(current, indent=2))
print(json.dumps(current, indent=2))
p2 = json.loads((WORK / 'experiment_v1/smoke/colab_gpu_01/smoke.json').read_text())['environment']
print('P2 환경과 비교:', {k: (p2.get(k), current.get(k)) for k in ('torch','numpy','cuda','cudnn','gpu')})
subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], check=True)
SMOKE = WORK / 'experiment_v1/smoke' / ('p3_preflight_' + SESSION)
subprocess.run([sys.executable, '-m', 'interp.smoke', '--device', 'cuda', '--output', str(SMOKE)], check=True)
for source_dir, label in ((ENV_DIR, 'environment'), (SMOKE, 'smoke')):
    for source in source_dir.rglob('*'):
        if source.is_file():
            verified_copy(source, DRIVE_ROOT / 'preflight' / SESSION / label / source.relative_to(source_dir))
assert json.loads((SMOKE / 'smoke.json').read_text())['status'] == 'passed'
PREFLIGHT_PASSED = True
''')
    md('''## 본실험 실행·재개
최초 50 updates는 본실험 예산에 포함됩니다. validation마다 진행 수치가 출력됩니다.
완전한 저장 묶음의 checksum을 검증한 후 `LATEST.json`을 갱신합니다. 불완전한 복사본은 재개에 사용하지 않습니다.
Drive `lm_seed_0/snapshots/`는 복구용으로 보존하세요. 실패 시 같은 셀을 새 run으로 실행하지 말고 새 런타임에서 재개하세요.''')
    code('''assert PREFLIGHT_PASSED
resume_path = recover(PERSISTENT, RUN) if RESUME else None
base = [sys.executable, '-u', '-m', 'interp.cli', 'train_lm', '--device', 'cuda',
        '--seed', '0', '--output', str(RUN), '--persistent-dir', str(PERSISTENT)]
decision_path = RUN / 'gate_decision.json'
decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
if decision and decision['decision'] in ('passed', 'failed'):
    print('이미 최종 판정이 있어 재학습하지 않습니다.')
else:
    args = base + (['--resume', str(resume_path)] if resume_path else [])
    if decision and decision['decision'] == 'extend_to_3m':
        args += ['--prediction-token-budget', '3000000', '--extension-approved']
    subprocess.run(args, check=True)
    decision = json.loads(decision_path.read_text())
    if decision['decision'] == 'extend_to_3m':
        subprocess.run(base + ['--resume', str(RUN / 'last.pt'),
                       '--prediction-token-budget', '3000000', '--extension-approved'], check=True)
        decision = json.loads(decision_path.read_text())
print(json.dumps(decision, indent=2))
''')
    md('''## 곡선·결과 다운로드
`passed`이면 동결 예산으로 P4 진행 가능 여부를 확인합니다. `failed`이면 CPU 점검과 중단 보고 대상입니다.
아래 ZIP에는 init/best/last, 1M 및 최종 결정, 로그, manifest, 환경과 GPU smoke 증빙이 포함됩니다.
Colab 런타임에서만 생성한 결과이며, 이 노트북 자체는 P3 통과 증빙이 아닙니다.''')
    code('''import matplotlib.pyplot as plt
history = [json.loads(line) for line in (RUN / 'training.jsonl').read_text().splitlines()]
evals = [row for row in history if 'validation' in row]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = [row['state']['prediction_tokens'] for row in evals]
axes[0].plot(x, [row['validation']['general']['answer_ce'] for row in evals], marker='o')
axes[0].set_ylabel('General validation answer CE')
for name in ('general', 'other_variable', 'repeated_update', 'first_read_after_set'):
    y = [row['validation']['general']['correct'] if name == 'general' else
         row['validation']['diagnostics'][name]['correct'] for row in evals]
    axes[1].plot(x, y, label=name)
axes[1].set_ylabel('Full-vocabulary accuracy'); axes[1].legend(fontsize=8)
for ax in axes: ax.set_xlabel('Prediction tokens'); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(RUN / 'learning_curve.png', dpi=160); plt.show()
export = DRIVE_ROOT / ('p3_evidence_' + SESSION + '.zip')
with zipfile.ZipFile(export, 'x', zipfile.ZIP_DEFLATED) as z:
    for source in RUN.iterdir():
        if source.is_file(): z.write(source, 'lm_seed_0/' + source.name)
    for source in (DRIVE_ROOT / 'preflight').rglob('*'):
        if source.is_file(): z.write(source, 'preflight/' + str(source.relative_to(DRIVE_ROOT / 'preflight')))
    z.write(WORK / 'p3_bundle_manifest.json', 'p3_bundle_manifest.json')
print('결과 ZIP SHA256:', hashlib.sha256(export.read_bytes()).hexdigest())
files.download(str(export))
''')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name=NOTEBOOK.name),
            kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),cells=cells)
    for i,cell in enumerate(cells):
        cell['id']=f'p3-{i:02d}'
        if cell['cell_type']=='code': compile(''.join(cell['source']),f'cell-{i}','exec')
    NOTEBOOK.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(bundle=str(BUNDLE),sha256=checksum,bytes=BUNDLE.stat().st_size,notebook=str(NOTEBOOK),files=len(files)),indent=2))

if __name__=='__main__': build()
