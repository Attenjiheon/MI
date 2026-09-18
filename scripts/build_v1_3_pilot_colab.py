"""Build the immutable v1.3 pilot Colab input bundle and its ordered executable notebook.

The notebook runs one frozen cell per invocation so an interrupted runtime never mixes cells.
Order: input checksum -> environment lock -> CPU tests/smoke -> GPU smoke -> the selected cell's
8M run -> contract audit -> evidence download. Selection happens only after all six cells return.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "experiment_v1_3/bundles/v1_3_pilot_bundle_v1.zip"
PARTIAL = BUNDLE.with_suffix(".zip.partial")
STATE = BUNDLE.with_suffix(".state.json")
NOTEBOOK = ROOT / "experiment_v1_3/notebooks/01_colab_pilot_8m.ipynb"
CELLS = ("base4_uniform", "base4_read4", "wide4_uniform", "wide4_read4", "deep8_uniform", "deep8_read4")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def bundle_files():
    files = set()
    for folder in (
        "corpus",
        "interp_v1_2",
        "interp_v1_3",
        "tests_v1_3",
        "data/language_v1_3",
        "experiment_v1_3/configs",
        "experiment_v1_2/debug",
    ):
        files.update(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
        )
    for name in (
        "requirements-interp.txt",
        "AGENTS.md",
        "phase.md",
        "01_experiment_design.md",
        "02_language_and_corpus.md",
        "03_experiment_spec.md",
        "experiment_v1_3/DESIGN.md",
        "experiment_v1_3/design_config.json",
        "experiment_v1_3/design_manifest.json",
        "experiment_v1_3/README.md",
        "experiment_v1_3/P2_STATUS.md",
        "experiment_v1_3/results/p1_generation.json",
        "experiment_v1_3/results/p1_frozen_training.json",
        "experiment_v1_3/results/p2_pytest.txt",
        "experiment_v1_3/smoke/cpu_release/smoke.json",
        "scripts/verify_v1_3_evidence.py",
        "scripts/select_v1_3_pilot_winner.py",
        "experiment_v1_2/results/p3_stop_report.md",
        "experiment_v1_2/P3_STATUS.md",
    ):
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(name)
        files.add(path)
    return sorted(files)


def markdown(text):
    return dict(cell_type="markdown", metadata={}, source=text.splitlines(True))


def code(text):
    return dict(cell_type="code", metadata={}, source=text.splitlines(True), execution_count=None, outputs=[])


def read_file(path, attempts=5, delay=3):
    """Read a source file whole before it touches the archive.

    The repository can live on cloud-backed storage that stalls mid-read. Reading first
    means a stall costs a retry instead of a truncated archive entry.
    """
    for attempt in range(attempts):
        try:
            return path.read_bytes()
        except OSError as error:
            if attempt + 1 == attempts:
                raise IOError(f"{path} is not readable after {attempts} attempts: {error}") from error
            time.sleep(delay)


def archive_pass(files, max_seconds):
    """Append pending files to the partial archive. Resumable: slow storage needs several passes."""
    state = json.loads(STATE.read_text()) if STATE.exists() else {"written": {}}
    written = state["written"]
    if PARTIAL.exists():
        with zipfile.ZipFile(PARTIAL) as archive:
            unrecorded = [name for name in archive.namelist() if name not in written]
        if unrecorded:
            print(f"discarding an interrupted archive ({len(unrecorded)} unrecorded entries)", flush=True)
            PARTIAL.unlink()
            written.clear()
    pending = [p for p in files if str(p.relative_to(ROOT)) not in written]
    started = time.monotonic()
    if pending:
        mode = "a" if PARTIAL.exists() else "x"
        try:
            with zipfile.ZipFile(PARTIAL, mode, zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in pending:
                    name = str(path.relative_to(ROOT))
                    data = read_file(path)
                    info = zipfile.ZipInfo.from_file(path, name)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data, compresslevel=6)
                    written[name] = hashlib.sha256(data).hexdigest()
                    del data
                    if max_seconds and time.monotonic() - started >= max_seconds:
                        break
        finally:
            STATE.write_text(json.dumps(state, indent=2))
        print(f"packed {len(written)}/{len(files)} files", flush=True)
    if len(written) != len(files):
        return None
    with zipfile.ZipFile(PARTIAL, "a", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        if "v1_3_bundle_manifest.json" not in archive.namelist():
            archive.writestr("v1_3_bundle_manifest.json", json.dumps(written, indent=2))
    with zipfile.ZipFile(PARTIAL) as archive:
        bad = archive.testzip()
        if bad:
            raise IOError("Corrupt archive entry: " + bad)
        if set(archive.namelist()) != set(written) | {"v1_3_bundle_manifest.json"}:
            raise IOError("Archive contents differ from the recorded file set")
    PARTIAL.replace(BUNDLE)
    STATE.unlink()
    return written


def build(max_seconds=0):
    if BUNDLE.exists():
        raise FileExistsError("Increment the bundle version; never overwrite an input bundle")
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    files = bundle_files()
    if archive_pass(files, max_seconds) is None:
        print("INCOMPLETE: run again to continue packing", flush=True)
        return
    digest = sha(BUNDLE)
    BUNDLE.with_suffix(".sha256").write_text(digest + "  " + BUNDLE.name + "\n")

    cells = [
        markdown(
            """# v1.3 pilot — 3 architecture × 2 loss, seed 0, 고정 8M

새 Colab GPU 런타임에서 위부터 순서대로 실행합니다. 입력 파일은 `"""
            + BUNDLE.name
            + """`입니다.
**한 번의 런타임에서 `CELL` 하나만 실행합니다.** 여섯 cell을 각각 실행한 뒤에야 winner 선택 셀을 돌립니다.

- 모든 cell이 같은 stream의 첫 **1,082 updates / 8,001,583 prediction tokens**를 한 번씩 소비합니다.
- 평가는 1M·3M·8M milestone에서만 수행하며 `select/`만 읽습니다. `gate/`와 `test/`는 pilot에서 열지 않습니다.
- Pilot 비교는 8M current checkpoint 하나만 사용합니다. 중간 checkpoint로 cell을 교체하지 않습니다.
- `base4_uniform`은 v1.2 anchor입니다. 1M/3M/8M model tensor digest가 보존된 v1.2 checkpoint와 일치해야 합니다.

중단되면 **새 런타임**에서 같은 `CELL`과 같은 Drive 경로로 `RESUME=True`로 실행합니다.
노트북 준비 완료와 GPU 실행 완료는 다릅니다. 결과 ZIP을 반환해 로컬 감사를 받아야 pilot 결과를 확정합니다.
"""
        ),
        code(
            """from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, zipfile
from google.colab import drive, files

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

CELL='base4_uniform'   # 여섯 cell 중 이번 런타임에서 실행할 하나
RESUME=False           # 중단된 같은 cell을 새 런타임에서 재개할 때 True
CELLS=('base4_uniform','base4_read4','wide4_uniform','wide4_read4','deep8_uniform','deep8_read4')
assert CELL in CELLS, CELL

drive.mount('/content/drive')
DRIVE_ROOT=Path('/content/drive/MyDrive/boolean_interp_v1_3_pilot')
WORK=Path('/content/boolean_interp')
RUN=WORK/('experiment_v1_3/runs/pilot_'+CELL+'_seed0')
PERSISTENT=DRIVE_ROOT/('pilot_'+CELL+'_seed0')
SESSION=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
DRIVE_ROOT.mkdir(parents=True,exist_ok=True)
if WORK.exists(): raise RuntimeError('작업 경로가 있습니다. 기존 결과를 보존하고 새 런타임에서 실행하세요.')
if not RESUME and (PERSISTENT/'LATEST.json').exists(): raise RuntimeError('이 cell의 run이 이미 있습니다. RESUME=True로 복구하세요.')
if RESUME and not (PERSISTENT/'LATEST.json').exists(): raise RuntimeError('완료된 저장 index가 없습니다.')
print('cell:',CELL,'resume:',RESUME)
"""
        ),
        markdown(
            """## 입력 ZIP checksum 검증
ZIP 자체의 SHA-256과 내부 모든 파일의 개별 hash를 검증합니다. 하나라도 어긋나면 진행하지 않습니다.

입력 ZIP은 약 830MB입니다. 브라우저 업로드(`files.upload()`)는 느리고 끊기기 쉬우므로,
**미리 Google Drive의 `MyDrive/boolean_interp_v1_3_pilot/` 폴더에 ZIP을 복사해 두는 것을 권장합니다.**
파일이 이미 있으면 이 셀은 업로드 없이 checksum만 검증하고, 여섯 cell 실행이 같은 ZIP을 재사용합니다."""
        ),
        code(
            f"""EXPECTED_SHA256={digest!r}
BUNDLE_NAME={BUNDLE.name!r}
saved=DRIVE_ROOT/BUNDLE_NAME
if not saved.exists():
    uploaded=files.upload()
    matches=[data for name,data in uploaded.items() if hashlib.sha256(data).hexdigest()==EXPECTED_SHA256]
    if not matches: raise RuntimeError('SHA-256이 일치하는 v1.3 입력 ZIP이 없습니다.')
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
manifest=json.loads((WORK/'v1_3_bundle_manifest.json').read_text())
for name,expected in manifest.items():
    assert sha(WORK/name)==expected, name
os.chdir(WORK)
print('입력 파일',len(manifest),'개 checksum 검증 완료')
"""
        ),
        markdown(
            """## 의존성과 새 환경 기록
처음 실행에서는 직접 의존성을 설치하고 버전을 기록합니다. 재개에서는 이전 세션의 버전을 먼저 복원합니다.
설치가 실패하면 버전을 임의로 바꾸지 말고 로그를 보존합니다. Colab 이미지나 CUDA가 바뀌면 새 환경 ID와 새 GPU smoke로 기록됩니다.
Drive에는 cell당 milestone checkpoint와 결과 ZIP을 저장할 여유(대략 3GB 이상)가 필요합니다."""
        ),
        code(
            """deps=['torch','numpy','scipy','pandas','matplotlib','pytest']
if RESUME:
    previous=sorted((DRIVE_ROOT/'preflight').glob('*/direct_versions.json'))
    if not previous: raise RuntimeError('이전 직접 의존성 버전 기록이 없습니다.')
    versions=json.loads(previous[-1].read_text())
    subprocess.run([sys.executable,'-m','pip','install',*[n+'=='+versions[n] for n in deps]],check=True)
else:
    subprocess.run([sys.executable,'-m','pip','install','-r','requirements-interp.txt'],check=True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
import importlib.metadata
import torch
assert torch.cuda.is_available(), 'GPU 런타임이 필요합니다.'
from interp_v1_3.runtime import environment, verify_inputs, verified_copy
PREFLIGHT=WORK/'experiment_v1_3/preflight'/SESSION
PREFLIGHT.mkdir(parents=True,exist_ok=True)
current=environment(PREFLIGHT)
(PREFLIGHT/'environment.json').write_text(json.dumps(current,indent=2))
(PREFLIGHT/'direct_versions.json').write_text(json.dumps({n:importlib.metadata.version(n) for n in deps},indent=2))
print(json.dumps(current,indent=2)); print(verify_inputs(WORK))
assert shutil.disk_usage(WORK).free>5*1024**3, '로컬 디스크 여유 부족'
"""
        ),
        markdown(
            """## CPU 테스트 → CPU smoke → GPU smoke
검증은 별도 debug 고정 데이터와 `select/` 일부만 사용합니다. 여섯 cell 전부에 대해
파라미터 수, causal/PAD masking, 각 head width의 RoPE, gradient 누적 일치, token-only `read4` 가중,
checkpoint 재개 bitwise 일치, 그리고 42-cell 평가 경로를 검사합니다.
GPU smoke가 통과하고 그 code/config/data hash와 환경 ID가 일치해야 runner가 본학습을 시작합니다."""
        ),
        code(
            """with (PREFLIGHT/'pytest.txt').open('w') as log:
    tested=subprocess.run([sys.executable,'-m','pytest','tests_v1_3','-q'],stdout=log,stderr=subprocess.STDOUT)
print((PREFLIGHT/'pytest.txt').read_text()); tested.check_returncode()
for device in ('cpu','cuda'):
    subprocess.run([sys.executable,'-u','-m','interp_v1_3.smoke','--device',device,'--output',str(PREFLIGHT/device)],check=True)
GPU_SMOKE=PREFLIGHT/'cuda/smoke.json'
report=json.loads(GPU_SMOKE.read_text())
assert report['status']=='passed' and report['scope']=='cuda'
assert set(report['cells'])==set(CELLS), '여섯 cell 모두의 GPU smoke가 필요합니다.'
assert report['anchor_v1_2_equivalence']['status']=='passed'
for source in PREFLIGHT.rglob('*'):
    if source.is_file(): verified_copy(source,DRIVE_ROOT/'preflight'/SESSION/source.relative_to(PREFLIGHT))
PREFLIGHT_PASSED=True
print('GPU smoke 통과:',report['environment']['environment_id'])
"""
        ),
        markdown(
            """## 이번 cell의 8M 학습과 영속 저장
첫 50 updates 처리량, peak VRAM, 학습/평가/저장 시간을 분리해 기록합니다.
`checkpoints/init.pt`, milestone checkpoint, `events/`, `milestones/`, `LATEST.json`을 보존하며 새 파일만 Drive에 복사합니다.
완료 index 갱신 전에 복사가 끊긴 파일은 복구에서 제외되므로, 재개는 항상 마지막 완전 저장 update부터 시작합니다."""
        ),
        code(
            """assert PREFLIGHT_PASSED
from interp_v1_3.persistence import recover
resume=recover(PERSISTENT,RUN) if RESUME else None
if (RUN/'result.json').exists():
    print('완료된 run을 복구했습니다. 재학습 없이 결과를 내보냅니다.')
else:
    command=[sys.executable,'-u','-m','interp_v1_3.cli','--cell',CELL,'--stage','pilot','--device','cuda',
             '--output',str(RUN),'--persistent-dir',str(PERSISTENT),'--smoke-report',str(GPU_SMOKE)]
    if resume: command+=['--resume',str(resume)]
    executed=subprocess.run(command)
    if executed.returncode:
        raise RuntimeError('학습 중단. Drive의 마지막 완료 index는 보존되었습니다. 새 런타임에서 RESUME=True로 재개하세요.')
result=json.loads((RUN/'result.json').read_text())
print(json.dumps({'cell':result['cell'],'status':result['status'],'final_state':result['final_state'],
                  'selected_update':result['selected']['update'],
                  'first_macro_ce':result['selected']['validation']['pairs']['first']['cell']['macro_answer_ce'],
                  'first_macro_accuracy':result['selected']['validation']['pairs']['first']['cell']['macro_accuracy'],
                  'select_general_accuracy':result['selected']['validation']['general']['accuracy']},indent=2))
"""
        ),
        markdown(
            """## 실행 계약 감사와 milestone 곡선
모든 milestone checkpoint의 바이트 hash와 model tensor digest, 평가 경계, 고정 cursor, 선택 규칙을 다시 계산해 대조합니다.
선택 checkpoint의 `select/` 평가만 재현하며 `gate/`와 `test/`는 열지 않습니다."""
        ),
        code(
            """audit=WORK/'experiment_v1_3/results'/('pilot_'+CELL+'_audit_'+SESSION+'.json')
audit.parent.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'scripts/verify_v1_3_evidence.py',str(RUN),'--output',str(audit),
                '--reevaluate','--device','cuda'],check=True)
import matplotlib.pyplot as plt
entries=[json.loads(p.read_text()) for p in sorted((RUN/'events').glob('*.json'))]
evals=[e for e in entries if 'validation' in e]
x=[e['state']['prediction_tokens'] for e in evals]
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(x,[e['validation']['pairs']['first']['cell']['macro_answer_ce'] for e in evals],marker='o',label='first 42-cell macro CE')
axes[0].plot(x,[e['validation']['pairs']['repeat']['cell']['macro_answer_ce'] for e in evals],marker='o',linestyle='--',label='repeat 42-cell macro CE')
axes[0].plot(x,[e['validation']['general']['answer_ce'] for e in evals],marker='s',label='select/general answer CE')
axes[1].plot(x,[e['validation']['pairs']['first']['cell']['macro_accuracy'] for e in evals],marker='o',label='first 42-cell macro accuracy')
axes[1].plot(x,[e['validation']['pairs']['repeat']['cell']['macro_accuracy'] for e in evals],marker='o',linestyle='--',label='repeat 42-cell macro accuracy')
axes[1].plot(x,[e['validation']['general']['accuracy'] for e in evals],marker='s',label='select/general accuracy')
for ax in axes: ax.set_xlabel('Prediction tokens'); ax.grid(alpha=.2); ax.legend(fontsize=7)
axes[0].set_ylabel('Answer CE'); axes[1].set_ylabel('Full-vocabulary accuracy')
fig.suptitle('v1.3 pilot '+CELL+' (select split only)')
fig.tight_layout(); plot=audit.with_suffix('.png'); fig.savefig(plot,dpi=150); plt.show()
print(json.dumps({k:json.loads(audit.read_text())[k] for k in ('status','cell','actual_tokens','updates','evaluations')},indent=2))
"""
        ),
        markdown(
            """## 이번 cell의 증빙 다운로드
ZIP에는 init과 milestone checkpoint, 모든 update 로그, milestone 기록, CPU/GPU smoke와 환경 lock, 감사 결과가 들어갑니다.
ZIP과 아래 SHA-256을 전달해 로컬 감사를 받습니다. 여섯 cell을 모두 마치기 전에는 winner를 선택하지 않습니다."""
        ),
        code(
            """export=DRIVE_ROOT/('v1_3_pilot_'+CELL+'_evidence_'+SESSION+'.zip')
with zipfile.ZipFile(export,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for source in RUN.rglob('*'):
        if source.is_file() and source.suffix!='.tmp': z.write(source,'pilot_'+CELL+'_seed0/'+str(source.relative_to(RUN)))
    for source in (DRIVE_ROOT/'preflight').rglob('*'):
        if source.is_file(): z.write(source,'preflight/'+str(source.relative_to(DRIVE_ROOT/'preflight')))
    z.write(WORK/'v1_3_bundle_manifest.json','v1_3_bundle_manifest.json')
    z.write(audit,'verification/'+audit.name); z.write(plot,'verification/'+plot.name)
checksum=sha(export); export.with_suffix('.sha256').write_text(checksum+'  '+export.name+chr(10))
print('결과 ZIP:',export.name); print('SHA-256:',checksum)
files.download(str(export))
"""
        ),
        markdown(
            """## (여섯 cell을 모두 마친 뒤에만) winner 선택
여섯 run과 그 감사 결과가 모두 있을 때만 실행합니다. 사전 규칙(§4.3) 외의 기준을 추가하지 않으며,
eligible cell이 없으면 `pilot_failure`로 종료하고 32M·seed 1/2·SAE/TC를 실행하지 않습니다.
이 셀은 각 cell의 run 디렉터리를 Drive에서 복구할 수 있을 때만 사용하고, 그렇지 않으면 로컬에서 같은 스크립트를 실행합니다."""
        ),
        code(
            """runs=[]
for cell in CELLS:
    persistent=DRIVE_ROOT/('pilot_'+cell+'_seed0')
    local=WORK/('experiment_v1_3/runs/recovered_'+cell)
    if not (persistent/'LATEST.json').exists(): raise RuntimeError('아직 실행되지 않은 cell: '+cell)
    if not local.exists(): recover(persistent,local)
    verification=WORK/'experiment_v1_3/results'/('pilot_'+cell+'_selection_audit.json')
    if not verification.exists():
        subprocess.run([sys.executable,'scripts/verify_v1_3_evidence.py',str(local),'--output',str(verification)],check=True)
    runs+=['--run',str(local),str(verification)]
selection=WORK/'experiment_v1_3/results/pilot_selection.json'
subprocess.run([sys.executable,'scripts/select_v1_3_pilot_winner.py',*runs,'--output',str(selection)],check=True)
print(json.dumps(json.loads(selection.read_text()),indent=2))
files.download(str(selection))
"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"v13-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
    notebook = dict(
        nbformat=4,
        nbformat_minor=5,
        metadata=dict(
            colab=dict(name=NOTEBOOK.name),
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            accelerator="GPU",
        ),
        cells=cells,
    )
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    report = dict(
        schema="colab-delivery-v1.3",
        stage="pilot",
        cells=list(CELLS),
        bundle=str(BUNDLE.relative_to(ROOT)),
        sha256=digest,
        bytes=BUNDLE.stat().st_size,
        notebook=str(NOTEBOOK.relative_to(ROOT)),
        notebook_sha256=sha(NOTEBOOK),
        files=len(files),
        code_cells_compiled=True,
    )
    (ROOT / "experiment_v1_3/results/colab_delivery.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, default=0, help="Stop after this long and resume on the next run")
    build(parser.parse_args().max_seconds)
