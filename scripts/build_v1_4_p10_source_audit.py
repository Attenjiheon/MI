"""Deliver small audit overlay on top of the unchanged P10 evaluation bundle."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha
from interp_v1_4 import p10_source_audit as audit,p10_evaluation as ev
from scripts.build_v1_4_p10_evaluation import make_notebook


def main():
    folder=ROOT/'experiment_v1_4/results/p10_return_audit_20261004_01';status=read(folder/'status.json')
    assert status['status']=='passed_local_source_reproduction_pending' and status['semantic_reports']==2016
    for n,h in status['evidence_sha256'].items():assert sha(ROOT/n)==h,n
    paths={folder/'status.json',folder/'return_manifest.json',ROOT/'interp_v1_4/p10_source_audit.py',ROOT/'tests_v1_4/test_p10_source_audit.py',Path(__file__),ROOT/'scripts/audit_v1_4_p10_evaluation_return.py'}
    base=ROOT/'experiment_v1_4/bundles/P10/v1_4_p10_update_evaluation_r1.zip'
    expected=read(base.with_suffix('.sha256.json'))['sha256'];assert sha(base)==expected
    c=dict(schema='p10-source-audit-v1.4-r1',date='2026-10-04',evaluation_contract_sha256=sha(ROOT/ev.CONTRACT),
        return_manifest=str((folder/'return_manifest.json').relative_to(ROOT)),base_bundle_sha256=expected,
        activation_tolerance=dict(atol=2e-5,rtol=2e-4),metric_tolerance=dict(atol=2e-6,rtol=2e-5),
        numeric_recipe='Float32 F.linear/ReLU/stable argsort TopK/F.linear, TF32 off; independent float64 fidelity; original float64 SciPy probe recipe for refit',
        planned=dict(activation_position_visits=480000,unique_reserved_positions=80000,trained_and_init_models=6,raw_arrays=54,dictionary_runs=12,latent_positions=960000,refits=1248,semantic_reports=2016),
        rng='Original frozen task seeds and source domains; no new selection rules or performance-dependent sampling',
        resume='Completed chunk/dictionary/task audit receipt with identical audit/evaluation/return identity',
        tolerance_policy='Fixed before source reproduction; mismatch stops and preserves evidence; no tolerance adjustment based on results',
        completion='GPU replay and all CPU refit/prediction/fidelity/CI checks, then independent local returned-evidence review; p10_complete remains false here',
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
    write(ROOT/audit.CONTRACT,c);paths.add(ROOT/audit.CONTRACT)
    bundle=ROOT/'experiment_v1_4/bundles/P10/v1_4_p10_source_audit_r1.zip'
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        wanted={**c['files'],audit.CONTRACT:sha(ROOT/audit.CONTRACT)};assert set(z.namelist())==set(wanted)
        for n,h in wanted.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    checksum=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=checksum,bytes=bundle.stat().st_size))
    for gpu,name in [(True,'P10_source_audit_1_T4_r1.ipynb'),(False,'P10_source_audit_2_CPU_r1.ipynb')]:
        nb=make_notebook(expected,gpu)
        nb['cells'][0]['source']=[('# P10 원본 재현 감사 1/2 — T4' if gpu else '# P10 원본 재현 감사 2/2 — CPU')+'\n\n',
            '본평가·학습은 완료되어 다시 선택하지 않습니다. 이 노트북은 고정 원본과 결과의 **재현 감사**입니다. 반환 감사 전에는 P10 완료가 아닙니다.\n\n',
            'Drive `boolean_interp_v1_4/`에 기존 `v1_4_p10_update_evaluation_r1.zip`과 새 `v1_4_p10_source_audit_r1.zip`을 둡니다. 기존 `P10_update_evaluation_r1` 전체 cache/arrays/latents/예측/fit 결과를 유지하세요. 새 출력은 `P10_update_source_audit_r1`입니다.\n\n',
            ('T4에서 trained/init 6개 모델의 예약 Update 80k 위치 전체를 재추출하여 54개 원본 배열과 비교합니다. 이어 선택 dictionary 12개의 모든 latent·fidelity·train 통계를 독립 계산합니다. 완료 후 GPU 연결을 종료하고 2/2 CPU 노트북으로 이동하세요.' if gpu else 'GPU 없는 CPU 런타임에서 1,248개 source-domain refit과 2,016개 예측·CI 및 paired 차이를 재현합니다. 이전 fitting과 비슷한 시간이 들 수 있습니다. 원본 선택을 바꾸거나 결과를 덮어쓰지 않습니다.')+'\n\n',
            '호출당 기본 2시간이며 chunk/task 경계에서 PAUSED이면 같은 실행 셀을 반복합니다. 런타임이 끊기면 해당 노트북 위부터 실행합니다. 과제 수·budget·허용오차를 바꾸지 마세요. 완료 또는 중단 시 마지막 셀의 ZIP·checksum을 반환합니다.\n']
        nb['cells'][1]['source']=f'''
from google.colab import drive,files
from pathlib import Path
import hashlib,json,zipfile,sys,subprocess,os,time,shutil
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/boolean_interp_v1_4')
ROOT=Path('/content/MI_P10_source_audit_r1');ROOT.mkdir(exist_ok=True)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
for filename,expected in [('v1_4_p10_update_evaluation_r1.zip','{expected}'),('v1_4_p10_source_audit_r1.zip','{checksum}')]:
    bundle=BASE/filename
    assert bundle.exists(), f'필요한 입력 파일: {{bundle}}'
    assert digest(bundle)==expected, 'ZIP checksum 불일치'
    with zipfile.ZipFile(bundle) as z:
        for n in z.namelist():assert (ROOT/n).resolve().is_relative_to(ROOT.resolve())
        z.extractall(ROOT)
SOURCE=BASE/'P10_update_evaluation_r1'
OUTPUT=BASE/'P10_update_source_audit_r1';OUTPUT.mkdir(exist_ok=True)
'''.strip().splitlines(True)
        for cell in nb['cells'][2:]:
            text=''.join(cell['source'])
            if cell['cell_type']=='code':
                text=text.replace('from interp_v1_4.p10_evaluation import verify,export','from interp_v1_4.p10_source_audit import verify,export').replace('tests_v1_4/test_p10_evaluation.py','tests_v1_4/test_p10_source_audit.py')
                if "'extract'" in text:
                    text="run(['-m','interp_v1_4.p10_source_audit','gpu','--root',str(ROOT),'--source',str(SOURCE),'--output',str(OUTPUT),'--max-seconds',str(MAX_SECONDS)])"
                elif "'encode'" in text:
                    text="print('GPU 감사 완료: GPU 연결을 종료하고 CPU 감사로 이동하세요.' if (OUTPUT/'gpu_complete.json').exists() else 'PAUSED: 위 GPU 실행 셀을 반복하세요. CPU 감사는 아직 시작하지 않습니다.')"
                elif "'select'" in text:
                    text="if not (OUTPUT/'gpu_complete.json').exists():\n    print('1/2 GPU 감사부터 완료하세요. CPU 감사는 실행하지 않았습니다.')\nelse:\n    run(['-m','interp_v1_4.p10_source_audit','cpu','--root',str(ROOT),'--source',str(SOURCE),'--output',str(OUTPUT),'--max-seconds',str(MAX_SECONDS)])"
                elif "'evaluate'" in text:
                    text="print('원본 재현 감사 완료: 마지막 셀로 증빙을 반환하세요.' if (OUTPUT/'source_audit_complete.json').exists() else 'PAUSED/대기: 위 감사 실행 셀을 반복하세요. 완료 판정은 아직 없습니다.')"
                text=text.replace('v1_4_p10_evaluation_return_','v1_4_p10_source_audit_return_')
                compile(text,name,'exec')
            else:
                text='중간 정지는 실패나 완료가 아닙니다. 완료된 감사 영수증은 재사용하고 남은 과제부터 재개합니다. 마지막 셀에서 감사 ZIP과 checksum을 내려받아 반환하세요.'
            cell['source']=text.splitlines(True)
        write(ROOT/'experiment_v1_4/notebooks/P10'/name,nb)
    print(json.dumps(dict(bundle=str(bundle),bytes=bundle.stat().st_size,sha256=checksum,members=len(wanted)),indent=2))

if __name__=='__main__':main()
