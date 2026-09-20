"""Build the v1.3 wide4_read4 seed-0 32M confirmatory Colab delivery.

The large frozen corpus/code bundle is reused from the pilot delivery. A small immutable
support bundle carries the audited 8M winner checkpoint and the confirmatory verifier.
The notebook deterministically replays 0->8M, checks model/optimizer/RNG/cursor bitwise
against the winner checkpoint, then continues 8M->32M and evaluates the frozen gate once.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUNDLE = ROOT / "experiment_v1_3/bundles/v1_3_pilot_bundle_v1.zip"
BASE_CHECKSUM = BASE_BUNDLE.with_suffix(".sha256")
PILOT_EVIDENCE = ROOT / "experiment_v1_3/evidence/v1_3_pilot_wide4_read4_evidence_20260918T183532331449.zip"
PILOT_SELECTION = ROOT / "experiment_v1_3/evidence/pilot_selection.json"
SUPPORT_BUNDLE = ROOT / "experiment_v1_3/bundles/v1_3_confirm_wide4_read4_seed0_support_v2.zip"
NOTEBOOK = ROOT / "experiment_v1_3/notebooks/02_colab_confirm_wide4_read4_seed0.ipynb"
DELIVERY = ROOT / "experiment_v1_3/results/confirm_colab_delivery.json"

PILOT_CHECKPOINT = "pilot_wide4_read4_seed0/checkpoints/update_001082.pt"
PILOT_RESULT = "pilot_wide4_read4_seed0/result.json"
PILOT_AUDIT = "verification/pilot_wide4_read4_audit_20260918T183532331449.json"
PILOT_VERSIONS = "preflight/20260918T183532331449/direct_versions.json"
PILOT_ENVIRONMENT = "preflight/20260918T183532331449/environment.json"


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def markdown(value):
    return dict(cell_type="markdown", metadata={}, source=value.splitlines(True))


def code(value):
    return dict(cell_type="code", metadata={}, source=value.splitlines(True), execution_count=None, outputs=[])


def build_support():
    if SUPPORT_BUNDLE.exists():
        recorded = SUPPORT_BUNDLE.with_suffix(".sha256").read_text().split()[0]
        if sha(SUPPORT_BUNDLE) != recorded:
            raise ValueError("Existing support bundle checksum mismatch")
        with zipfile.ZipFile(SUPPORT_BUNDLE) as archive:
            manifest = json.loads(archive.read("confirm_support/manifest.json"))
        expected_base = BASE_CHECKSUM.read_text().split()[0]
        if manifest["base_bundle"] != {"name": BASE_BUNDLE.name, "sha256": expected_base}:
            raise ValueError("Existing support bundle refers to a different base bundle")
        return manifest, expected_base, recorded
    expected_base = BASE_CHECKSUM.read_text().split()[0]
    if sha(BASE_BUNDLE) != expected_base:
        raise ValueError("Existing pilot input bundle checksum mismatch")
    selection = json.loads(PILOT_SELECTION.read_text())
    if selection["decision"] != "promote" or selection["winner"] != "wide4_read4":
        raise ValueError("Frozen pilot selection does not promote wide4_read4")
    with zipfile.ZipFile(PILOT_EVIDENCE) as source:
        payloads = {
            "confirm_support/pilot_checkpoint_update_001082.pt": source.read(PILOT_CHECKPOINT),
            "confirm_support/pilot_result.json": source.read(PILOT_RESULT),
            "confirm_support/pilot_audit.json": source.read(PILOT_AUDIT),
            "confirm_support/direct_versions.json": source.read(PILOT_VERSIONS),
            "confirm_support/pilot_environment.json": source.read(PILOT_ENVIRONMENT),
            "confirm_support/pilot_selection.json": PILOT_SELECTION.read_bytes(),
            "confirm_support/verify_v1_3_evidence.py": (ROOT / "scripts/verify_v1_3_evidence.py").read_bytes(),
        }
    result = json.loads(payloads["confirm_support/pilot_result.json"])
    audit = json.loads(payloads["confirm_support/pilot_audit.json"])
    environment = json.loads(payloads["confirm_support/pilot_environment.json"])
    if result["cell"] != "wide4_read4" or result["stage"] != "pilot" or result["lm_seed"] != 0:
        raise ValueError("Unexpected pilot handoff identity")
    if result["selected"]["update"] != 1082 or result["actual_prediction_tokens"] != 8_001_583:
        raise ValueError("Pilot handoff is not the frozen 8M boundary")
    if result["selected"]["sha256"] != sha_bytes(payloads["confirm_support/pilot_checkpoint_update_001082.pt"]):
        raise ValueError("Pilot checkpoint checksum differs from result.json")
    if audit["status"] != "passed" or not audit["reevaluation"]:
        raise ValueError("Pilot run audit or select re-evaluation is missing")
    if audit["selected"]["sha256"] != result["selected"]["sha256"]:
        raise ValueError("Pilot audit selected a different checkpoint")
    if selection["sources"]["wide4_read4"]["checkpoint_sha256"] != result["selected"]["sha256"]:
        raise ValueError("Pilot selection source differs from the handoff checkpoint")
    manifest = {
        "schema": "v1.3-confirm-support-v2",
        "base_bundle": {"name": BASE_BUNDLE.name, "sha256": expected_base},
        "source_evidence": {"name": PILOT_EVIDENCE.name, "sha256": sha(PILOT_EVIDENCE)},
        "pilot": {
            "cell": "wide4_read4",
            "lm_seed": 0,
            "update": 1082,
            "prediction_tokens": 8_001_583,
            "checkpoint_sha256": result["selected"]["sha256"],
            "model_tensor_sha256": result["selected"]["model_tensor_sha256"],
            "environment_id": environment["environment_id"],
        },
        "files": {name: sha_bytes(value) for name, value in payloads.items()},
    }
    payloads["confirm_support/manifest.json"] = json.dumps(manifest, indent=2).encode() + b"\n"
    SUPPORT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SUPPORT_BUNDLE, "x", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)
    with zipfile.ZipFile(SUPPORT_BUNDLE) as archive:
        if archive.testzip():
            raise IOError("Support bundle integrity test failed")
    support_sha = sha(SUPPORT_BUNDLE)
    SUPPORT_BUNDLE.with_suffix(".sha256").write_text(support_sha + "  " + SUPPORT_BUNDLE.name + "\n")
    return manifest, expected_base, support_sha


def build_notebook(manifest, base_sha, support_sha):
    intro = """# v1.3 confirm — `wide4_read4`, LM seed 0, 고정 32M + one-time gate

새 Colab **Tesla T4 GPU** 런타임에서 위부터 순서대로 실행합니다. 입력은 두 개입니다.

1. `v1_3_pilot_bundle_v1.zip`: 동결된 코드·config·32M corpus
2. `v1_3_confirm_wide4_read4_seed0_support_v2.zip`: 승격된 8M checkpoint·감사·선택 기록

이 노트북은 confirm stage를 0→8M까지 결정적으로 재생한 뒤 멈추고, pilot checkpoint와
**model, optimizer, RNG, cursor, milestone 기록을 bitwise 대조**합니다. 모두 같을 때만 8M→32M을 계속합니다.
이는 stage가 다른 checkpoint metadata를 변조하지 않으면서 검증된 8M 상태에서 정확히 연장하기 위한 절차입니다.

- 선택은 고정 milestone의 `select/first_repeat` first-member 42-cell macro answer CE 최소만 사용합니다.
- gate는 선택된 checkpoint 하나에서 딱 한 번 평가하며 checkpoint를 다시 고르지 않습니다.
- `test/`는 이 노트북에서 열지 않습니다.
- seed 0 gate 실패도 유효한 종료 결과이며, 이 경우 seed 1·2와 표현 분석을 시작하지 않습니다.
- 런타임이 끊기면 새 런타임에서 `RESUME=True`로 다시 실행합니다.
"""
    setup = """from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.metadata, json, os, shutil, subprocess, sys, zipfile
from google.colab import drive, files

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

RESUME=False
BASE_NAME='__BASE_NAME__'
BASE_SHA256='__BASE_SHA__'
SUPPORT_NAME='__SUPPORT_NAME__'
SUPPORT_SHA256='__SUPPORT_SHA__'
EXPECTED_ENVIRONMENT_ID='__ENV_ID__'
EXPECTED_PILOT_MODEL_DIGEST='__MODEL_DIGEST__'
EXPECTED_PILOT_CHECKPOINT_SHA256='__CHECKPOINT_SHA__'

drive.mount('/content/drive')
DRIVE_ROOT=Path('/content/drive/MyDrive/boolean_interp_v1_3_confirm')
INPUT_ROOT=DRIVE_ROOT/'inputs'
# Use a confirm-specific scratch path so a pilot notebook's stale
# /content/boolean_interp file or symlink cannot collide with this run.
WORK=Path('/content/boolean_interp_v1_3_confirm')
SUPPORT_ROOT=Path('/content/v1_3_confirm_support')
RUN=WORK/'experiment_v1_3/runs/confirm_wide4_read4_seed0'
PERSISTENT=DRIVE_ROOT/'confirm_wide4_read4_seed0'
HANDOFF_RECORD=DRIVE_ROOT/'handoff_verification.json'
SESSION=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
INPUT_ROOT.mkdir(parents=True,exist_ok=True)
occupied=[str(path) for path in (WORK,SUPPORT_ROOT) if os.path.lexists(str(path))]
if occupied:
    raise RuntimeError('작업 경로가 이미 있습니다(끊어진 symlink 포함): '+', '.join(occupied)+'. 결과를 보존하고 새 런타임에서 실행하세요.')
if not RESUME and (PERSISTENT/'LATEST.json').exists():
    raise RuntimeError('기존 confirm run이 있습니다. RESUME=True로 복구하세요.')
if not RESUME and HANDOFF_RECORD.exists():
    raise RuntimeError('기존 handoff 증빙이 있습니다. 같은 run은 RESUME=True로 복구하세요.')
if RESUME and not (PERSISTENT/'LATEST.json').exists():
    raise RuntimeError('재개할 완료 index가 없습니다.')
print('resume:',RESUME,'session:',SESSION)
"""
    replacements = {
        "__BASE_NAME__": BASE_BUNDLE.name,
        "__BASE_SHA__": base_sha,
        "__SUPPORT_NAME__": SUPPORT_BUNDLE.name,
        "__SUPPORT_SHA__": support_sha,
        "__ENV_ID__": manifest["pilot"]["environment_id"],
        "__MODEL_DIGEST__": manifest["pilot"]["model_tensor_sha256"],
        "__CHECKPOINT_SHA__": manifest["pilot"]["checkpoint_sha256"],
    }
    for key, value in replacements.items():
        setup = setup.replace(key, value)

    inputs = """def obtain(name,expected,alternates=()):
    candidates=[INPUT_ROOT/name,*[Path(p) for p in alternates]]
    for candidate in candidates:
        if candidate.exists() and sha(candidate)==expected:
            if candidate.parent!=INPUT_ROOT:
                shutil.copyfile(candidate,INPUT_ROOT/name)
            return INPUT_ROOT/name
    uploaded=files.upload()
    matches=[data for data in uploaded.values() if hashlib.sha256(data).hexdigest()==expected]
    if not matches: raise RuntimeError(name+' checksum과 일치하는 업로드가 없습니다.')
    temp=(INPUT_ROOT/name).with_suffix('.zip.tmp'); temp.write_bytes(matches[0])
    assert sha(temp)==expected; temp.replace(INPUT_ROOT/name)
    return INPUT_ROOT/name

base=obtain(BASE_NAME,BASE_SHA256,('/content/drive/MyDrive/boolean_interp_v1_3_pilot/'+BASE_NAME,))
support=obtain(SUPPORT_NAME,SUPPORT_SHA256)
WORK.mkdir(parents=True); SUPPORT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(base) as archive:
    for info in archive.infolist():
        if not (WORK/info.filename).resolve().is_relative_to(WORK.resolve()): raise RuntimeError('잘못된 base ZIP 경로')
    archive.extractall(WORK)
base_manifest=json.loads((WORK/'v1_3_bundle_manifest.json').read_text())
for name,expected in base_manifest.items(): assert sha(WORK/name)==expected,name
with zipfile.ZipFile(support) as archive:
    for info in archive.infolist():
        target=Path('/content')/info.filename
        if not target.resolve().is_relative_to(Path('/content').resolve()): raise RuntimeError('잘못된 support ZIP 경로')
    archive.extractall('/content')
support_manifest=json.loads((SUPPORT_ROOT/'manifest.json').read_text())
for name,expected in support_manifest['files'].items(): assert sha(Path('/content')/name)==expected,name
assert support_manifest['base_bundle']=={'name':BASE_NAME,'sha256':BASE_SHA256}
assert support_manifest['pilot']['cell']=='wide4_read4' and support_manifest['pilot']['lm_seed']==0
assert support_manifest['pilot']['checkpoint_sha256']==EXPECTED_PILOT_CHECKPOINT_SHA256
assert support_manifest['pilot']['model_tensor_sha256']==EXPECTED_PILOT_MODEL_DIGEST
shutil.copyfile(SUPPORT_ROOT/'verify_v1_3_evidence.py',WORK/'scripts/verify_v1_3_evidence.py')
os.chdir(WORK)
print('base files:',len(base_manifest),'support files:',len(support_manifest['files']))
"""

    environment = """desired=json.loads((SUPPORT_ROOT/'direct_versions.json').read_text())
specs=[name+'=='+version for name,version in desired.items()]
subprocess.run([sys.executable,'-m','pip','install',*specs],check=True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
import torch
assert torch.cuda.is_available(),'GPU 런타임이 필요합니다.'
from interp_v1_3.runtime import environment as capture_environment, verify_inputs, verified_copy
PREFLIGHT=WORK/'experiment_v1_3/preflight'/SESSION
PREFLIGHT.mkdir(parents=True,exist_ok=True)
current=capture_environment(PREFLIGHT)
(PREFLIGHT/'environment.json').write_text(json.dumps(current,indent=2))
(PREFLIGHT/'direct_versions.json').write_text(json.dumps({n:importlib.metadata.version(n) for n in desired},indent=2))
print(json.dumps(current,indent=2))
assert current['environment_id']==EXPECTED_ENVIRONMENT_ID, 'pilot과 동일한 Colab 환경/T4가 아닙니다. 실행을 중단합니다.'
assert current['gpu']=='Tesla T4'
print(verify_inputs(WORK))
assert shutil.disk_usage(WORK).free>5*1024**3,'로컬 디스크 여유가 5GB 미만입니다.'
"""

    smoke = """with (PREFLIGHT/'pytest.txt').open('w') as log:
    tested=subprocess.run([sys.executable,'-m','pytest','tests_v1_3','-q'],stdout=log,stderr=subprocess.STDOUT)
print((PREFLIGHT/'pytest.txt').read_text()); tested.check_returncode()
for device in ('cpu','cuda'):
    subprocess.run([sys.executable,'-u','-m','interp_v1_3.smoke','--device',device,'--output',str(PREFLIGHT/device)],check=True)
GPU_SMOKE=PREFLIGHT/'cuda/smoke.json'
smoke_report=json.loads(GPU_SMOKE.read_text())
assert smoke_report['status']=='passed' and smoke_report['scope']=='cuda'
assert smoke_report['environment']['environment_id']==EXPECTED_ENVIRONMENT_ID
assert 'wide4_read4' in smoke_report['cells']
for source in PREFLIGHT.rglob('*'):
    if source.is_file(): verified_copy(source,DRIVE_ROOT/'preflight'/SESSION/source.relative_to(PREFLIGHT))
print('CPU/GPU smoke 통과')
"""

    training = """from interp_v1_3.persistence import read_index, recover
from interp_v1_3.runtime import tensor_digest
import numpy as np

def resume_checkpoint():
    return RUN/read_index(RUN)['checkpoint']

def progress():
    if not (RUN/'LATEST.json').exists(): return {'update':0,'prediction_tokens':0,'next_data_cursor':0}
    return torch.load(resume_checkpoint(),map_location='cpu',weights_only=False)['state']

def launch(resume=None,pause_at_8m=False):
    command=[sys.executable,'-u','-m','interp_v1_3.cli','--cell','wide4_read4','--stage','confirm',
             '--lm-seed','0','--device','cuda','--output',str(RUN),'--persistent-dir',str(PERSISTENT),
             '--smoke-report',str(GPU_SMOKE)]
    if resume is not None: command+=['--resume',str(resume)]
    if pause_at_8m: command+=['--pause-after-updates','1082']
    executed=subprocess.run(command)
    if executed.returncode:
        raise RuntimeError('학습이 중단되었습니다. Drive의 마지막 완료 index는 보존됐습니다. 새 런타임에서 RESUME=True로 재개하세요.')

def exact(a,b,path='root'):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()): raise AssertionError(path)
        return
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b): raise AssertionError(path)
        return
    if isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys(): raise AssertionError(path+'.keys')
        for key in a: exact(a[key],b[key],path+'.'+str(key))
        return
    if isinstance(a,(list,tuple)) and isinstance(b,type(a)):
        if len(a)!=len(b): raise AssertionError(path+'.length')
        for index,(left,right) in enumerate(zip(a,b)): exact(left,right,path+'['+str(index)+']')
        return
    if a!=b: raise AssertionError(path)

resume=recover(PERSISTENT,RUN) if RESUME else None
if (RUN/'result.json').exists():
    print('완료된 confirm run을 복구했습니다.')
else:
    state=progress()
    if state['update']<1082:
        launch(resume=resume,pause_at_8m=True)
        state=progress()
    if state['update']==1082 and not HANDOFF_RECORD.exists():
        assert state=={'update':1082,'prediction_tokens':8001583,'next_data_cursor':69248}
        replay=torch.load(resume_checkpoint(),map_location='cpu',weights_only=False)
        pilot_path=SUPPORT_ROOT/'pilot_checkpoint_update_001082.pt'
        assert sha(pilot_path)==EXPECTED_PILOT_CHECKPOINT_SHA256
        pilot=torch.load(pilot_path,map_location='cpu',weights_only=False)
        assert pilot['stage']=='pilot' and pilot['cell']=='wide4_read4'
        assert replay['stage']=='confirm' and replay['cell']=='wide4_read4'
        for field in ('model','optimizer','state','rng_states','microbatch','milestones'):
            exact(pilot[field],replay[field],field)
        model_digest=tensor_digest(replay['model'])
        assert model_digest==EXPECTED_PILOT_MODEL_DIGEST
        handoff={'schema':'v1.3-confirm-handoff-v1','status':'passed','comparison':'bitwise_exact',
                 'fields':['model','optimizer','state','rng_states','microbatch','milestones'],
                 'pilot_checkpoint_sha256':EXPECTED_PILOT_CHECKPOINT_SHA256,
                 'replay_checkpoint_sha256':sha(resume_checkpoint()),'model_tensor_sha256':model_digest,
                 'environment_id':EXPECTED_ENVIRONMENT_ID,'state':state,'verified_at':SESSION}
        temp=HANDOFF_RECORD.with_suffix('.json.tmp'); temp.write_text(json.dumps(handoff,indent=2)+'\\n'); temp.replace(HANDOFF_RECORD)
        print('8M pilot handoff bitwise 검증 통과')
    if not HANDOFF_RECORD.exists():
        raise RuntimeError('8M bitwise handoff 증빙 없이 연장할 수 없습니다.')
    handoff=json.loads(HANDOFF_RECORD.read_text())
    assert handoff['status']=='passed' and handoff['model_tensor_sha256']==EXPECTED_PILOT_MODEL_DIGEST
    state=progress()
    if state['update']>1082 and handoff['status']!='passed': raise RuntimeError('검증되지 않은 8M 이후 진행 상태')
    launch(resume=resume_checkpoint(),pause_at_8m=False)

result=json.loads((RUN/'result.json').read_text())
assert result['stage']=='confirm' and result['cell']=='wide4_read4' and result['lm_seed']==0
assert result['final_state']=={'update':4249,'prediction_tokens':32004917,'next_data_cursor':271936}
assert result['gate'] is not None and result['gate']['checkpoint']==result['selected']['checkpoint']
gate_result=result['gate']
summary={'status':result['status'],'selected_update':result['selected']['update'],
         'selected_prediction_tokens':result['selected']['prediction_tokens'],
         'select_first_macro_ce':result['selected']['validation']['pairs']['first']['cell']['macro_answer_ce'],
         'gate_passed':gate_result['decision']['passed'],'gate_checks':gate_result['decision']['checks'],
         'gate_general_accuracy':gate_result['general']['accuracy'],
         'gate_diagnostics':{k:v['accuracy'] for k,v in gate_result['diagnostics'].items()},
         'gate_first_macro_accuracy':gate_result['pairs']['first']['cell']['macro_accuracy'],
         'gate_repeat_macro_accuracy':gate_result['pairs']['repeat']['cell']['macro_accuracy'],
         'gate_operator_accuracy':{k:v['accuracy'] for k,v in gate_result['pairs']['first']['operator']['cells'].items()},
         'gate_depth_accuracy':{k:v['accuracy'] for k,v in gate_result['pairs']['first']['depth']['cells'].items()},
         'gate_answer_accuracy':{k:v['accuracy'] for k,v in gate_result['pairs']['first']['answer']['cells'].items()}}
print(json.dumps(summary,indent=2))
if not gate_result['decision']['passed']:
    print('SEED 0 GATE FAILED: seed 1/2 및 P5 이후를 시작하지 않습니다.')
else:
    print('SEED 0 GATE PASSED: 설정을 동결하고 seed 1/2 노트북을 별도로 준비할 수 있습니다.')
"""

    audit = """audit=WORK/'experiment_v1_3/results'/('confirm_wide4_read4_seed0_audit_'+SESSION+'.json')
audit.parent.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'scripts/verify_v1_3_evidence.py',str(RUN),'--output',str(audit),
                '--reevaluate','--device','cuda'],check=True)
checked=json.loads(audit.read_text())
assert checked['status']=='passed' and checked['gate']['decision']['passed']==result['gate']['decision']['passed']

import matplotlib.pyplot as plt
evaluations=[]
for event_path in sorted((RUN/'events').glob('*.json')):
    event=json.loads(event_path.read_text())
    if 'validation' in event: evaluations.append(event)
x=[e['state']['prediction_tokens'] for e in evaluations]
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(x,[e['validation']['pairs']['first']['cell']['macro_answer_ce'] for e in evaluations],marker='o',label='first macro CE')
axes[0].plot(x,[e['validation']['pairs']['repeat']['cell']['macro_answer_ce'] for e in evaluations],marker='o',linestyle='--',label='repeat macro CE')
axes[0].plot(x,[e['validation']['general']['answer_ce'] for e in evaluations],marker='s',label='general answer CE')
axes[1].plot(x,[e['validation']['pairs']['first']['cell']['macro_accuracy'] for e in evaluations],marker='o',label='first macro accuracy')
axes[1].plot(x,[e['validation']['pairs']['repeat']['cell']['macro_accuracy'] for e in evaluations],marker='o',linestyle='--',label='repeat macro accuracy')
axes[1].plot(x,[e['validation']['general']['accuracy'] for e in evaluations],marker='s',label='general accuracy')
for ax in axes: ax.set_xlabel('Prediction tokens'); ax.grid(alpha=.2); ax.legend(fontsize=7)
axes[0].set_ylabel('Answer CE'); axes[1].set_ylabel('Full-vocabulary accuracy')
fig.suptitle('v1.3 confirm wide4_read4 seed 0 — select split')
fig.tight_layout(); plot=audit.with_suffix('.png'); fig.savefig(plot,dpi=150); plt.show()
print(json.dumps(checked['gate'],indent=2))
"""

    export = """export=DRIVE_ROOT/('v1_3_confirm_wide4_read4_seed0_evidence_'+SESSION+'.zip')
with zipfile.ZipFile(export,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for source in RUN.rglob('*'):
        if source.is_file() and source.suffix!='.tmp': archive.write(source,'confirm_wide4_read4_seed0/'+str(source.relative_to(RUN)))
    for source in PREFLIGHT.rglob('*'):
        if source.is_file(): archive.write(source,'preflight/'+SESSION+'/'+str(source.relative_to(PREFLIGHT)))
    archive.write(HANDOFF_RECORD,'verification/'+HANDOFF_RECORD.name)
    archive.write(audit,'verification/'+audit.name); archive.write(plot,'verification/'+plot.name)
    archive.write(SUPPORT_ROOT/'manifest.json','confirm_support_manifest.json')
checksum=sha(export); export.with_suffix('.sha256').write_text(checksum+'  '+export.name+'\\n')
print('결과 ZIP:',export.name); print('SHA-256:',checksum); print('크기:',export.stat().st_size)
files.download(str(export))
"""

    cells = [
        markdown(intro),
        code(setup),
        markdown("## 입력 checksum·내부 manifest 검증\n\n기존 pilot base bundle은 Drive의 pilot 폴더에서도 재사용합니다. support bundle은 약 20MB입니다."),
        code(inputs),
        markdown("## pilot과 동일한 환경 복원\n\n직접 의존성 버전을 고정하고 전체 환경 ID가 `e74fb1dcf8112ca0`인지 확인합니다. 다른 GPU/환경이면 진행하지 않습니다."),
        code(environment),
        markdown("## unit test와 CPU/GPU smoke\n\n동결 config·data·code hash 및 여섯 cell의 GPU 경로를 다시 확인합니다."),
        code(smoke),
        markdown("## 8M bitwise handoff → 32M 연장 → one-time gate\n\n중단 시 마지막 완전 checkpoint만 Drive에 남습니다. gate 실패도 결과로 보존합니다."),
        code(training),
        markdown("## 계약 감사와 학습 곡선\n\n선택 split만 재평가합니다. 기록된 gate는 quota·threshold·checkpoint 결속을 재계산하되 gate split을 다시 열지 않습니다."),
        code(audit),
        markdown("## evidence ZIP 다운로드\n\n전체 confirm run, 환경/smoke, bitwise handoff, gate 감사와 그림을 묶습니다."),
        code(export),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"v13-confirm-{index:02d}"
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
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    return len(cells)


def main():
    manifest, base_sha, support_sha = build_support()
    cells = build_notebook(manifest, base_sha, support_sha)
    delivery = {
        "schema": "colab-delivery-v1.3-confirm-seed0",
        "stage": "confirm",
        "cell": "wide4_read4",
        "lm_seed": 0,
        "base_bundle": {"path": str(BASE_BUNDLE.relative_to(ROOT)), "sha256": base_sha, "bytes": BASE_BUNDLE.stat().st_size},
        "support_bundle": {"path": str(SUPPORT_BUNDLE.relative_to(ROOT)), "sha256": support_sha, "bytes": SUPPORT_BUNDLE.stat().st_size},
        "notebook": {"path": str(NOTEBOOK.relative_to(ROOT)), "sha256": sha(NOTEBOOK), "code_cells_compiled": True, "cells": cells},
        "pilot_checkpoint_sha256": manifest["pilot"]["checkpoint_sha256"],
        "pilot_model_tensor_sha256": manifest["pilot"]["model_tensor_sha256"],
        "expected_environment_id": manifest["pilot"]["environment_id"],
        "gpu_execution_completed": False,
    }
    DELIVERY.write_text(json.dumps(delivery, indent=2) + "\n")
    print(json.dumps(delivery, indent=2))


if __name__ == "__main__":
    main()
