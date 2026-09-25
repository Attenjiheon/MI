"""Build the P6 r1 Colab delivery. Never packages or changes original P5 cache."""
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write,derived
from interp_v1_4.runtime import sha
from interp_v1_4.p6 import CONTRACT,RECEIPT


def build():
    base=ROOT/'experiment_v1_4/results/p5_final_audit_20260925_01'
    c=read(base/'completion.json');assert c['p5_complete'] and c['p6_eligible']
    for name,digest in c['evidence_sha256'].items():assert sha(base/name)==digest
    frozen=read(ROOT/'experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json')
    paths={ROOT/RECEIPT,base/'completion.json',ROOT/'experiment_v1_4/analysis_plan.json',ROOT/'experiment_v1_4/p5_r2/contract.json',ROOT/'experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json',ROOT/'archive/legacy/experiment_v1_2/debug/sequences.json',ROOT/'experiment_v1_4/frozen_test_r1/requirements-primary.lock.txt',ROOT/'scripts/build_v1_4_p6.py',ROOT/'scripts/verify_v1_4_p6_return.py',ROOT/'tests_v1_4/test_p6.py'}
    for folder in ('interp_v1_4','archive/legacy/interp_v1_2','corpus'):
        paths.update((ROOT/folder).glob('*.py'))
    paths.update(ROOT/n for n in ('01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md'))
    runs=[]
    for m in frozen['models']:
        for layer in (0,3,7,11):
            for k in (4,16):
                shared=f"v1.4|lm_seed={m['lm_seed']}|checkpoint={m['checkpoint_sha256']}|layer={layer}|READ|k={k}|sparse_seed=0"
                # h/u/m are tool-dependent. Shared draws exclude both tool and hook.
                key=shared+'|hook=h|tool=sae'
                runs.append(dict(name=f"seed{m['lm_seed']}_l{layer}_sae_k{k}_s0",lm_seed=m['lm_seed'],checkpoint_sha256=m['checkpoint_sha256'],layer=layer,hook='h',position_type='READ',tool='sae',k=k,sparse_seed=0,run_key=key,draw_key=shared,init_seed=derived('dictionary_init',key),torch_init_seed=derived('dictionary_init',key)%(2**63-1),draw_seed=derived('position_draw',shared)))
    config=dict(schema='p6-sae-v1.4-r1',runs=runs,training=dict(width=256,latents=512,parameters=262912,updates=5000,batch=512,validation_interval=250,optimizer='Adam',lr=0.001,betas=[0.9,0.999],eps=1e-8,weight_decay=0,clip_norm=1.,schedule='constant',selection='minimum full validation normalized MSE; earliest update on tie',sampling='PCG64 uniform replacement per update',label_loss=False,early_stopping=False,dead_reinitialization=False),rng_namespace='20260909|experiment-spec-v1.0',draw_key_policy='fixed field order as recorded; excludes tool and tool-dependent hook for paired SAE/TC draws',files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},budget=dict(runs=24,updates=120000,draws=61440000),input_gate='240 source cache hashes; READ keys; 36 train scalar statistics; current-source four-layer GPU smoke',completion='returned evidence audit required')
    write(ROOT/CONTRACT,config)
    bundle=ROOT/'experiment_v1_4/bundles/v1_4_p6_bundle_r1.zip';bundle.parent.mkdir(exist_ok=True)
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(paths|{ROOT/CONTRACT}):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        import hashlib
        for n,d in {**config['files'],CONTRACT:sha(ROOT/CONTRACT)}.items():assert hashlib.sha256(z.read(n)).hexdigest()==d
    digest=sha(bundle)
    write(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,'notebook','exec')
        cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(True)))
    md('''# P6 — 네 층 READ SAE 24 runs\n\n런타임을 **GPU**로 설정하고 아래 순서대로 실행하세요. P5 Drive 원본은 그대로 사용하며 새 출력은 `P6_r1`에 저장합니다. 원본 cache bytes·통계 36개·현재 GPU smoke 검증이 끝나야 학습이 시작됩니다. 각 run은 5,000 updates이며 반환 감사 전에는 P6 완료가 아닙니다.\n\n중단 시 이 노트북을 다시 실행합니다. Drive의 마지막 checksum-committed checkpoint에서 optimizer/RNG/draw cursor를 복구합니다. GPU/패키지가 바뀌면 새 환경 ID와 GPU smoke를 기록합니다. 학습 셀의 RUNS로 일부 run만 실행할 수 있지만 완료에는 24개 전부 필요합니다.''')
    code(f'''from google.colab import files, drive
from pathlib import Path
import hashlib, zipfile, subprocess, sys, json, time
uploaded=files.upload()  # v1_4_p6_bundle_r1.zip 선택
bundle=Path('v1_4_p6_bundle_r1.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == '{digest}', '입력 ZIP checksum 불일치'
ROOT=Path('/content/MI_P6_r1'); ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for item in z.infolist():
        target=(ROOT/item.filename).resolve()
        assert target.is_relative_to(ROOT.resolve()), '잘못된 ZIP 경로'
    z.extractall(ROOT)
drive.mount('/content/drive')
SOURCE=Path('/content/drive/MyDrive/boolean_interp_v1_4/P5_r2')
OUTPUT=Path('/content/drive/MyDrive/boolean_interp_v1_4/P6_r1')
assert (SOURCE/'contract.json').exists(), '기존 P5 cache 경로를 확인하세요'
''')
    md('## 1. 의존성과 입력 검증\n설치 후 재시작 안내가 나오면 런타임을 재시작하고 업로드 셀부터 다시 실행하세요. 설치 로그와 실제 pip freeze/GPU 정보도 저장합니다.')
    code('''log=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','pytest==8.4.2','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(log.stdout); log.check_returncode()
OUTPUT.mkdir(parents=True,exist_ok=True)
(OUTPUT/f'install_{time.time_ns()}.txt').write_text(log.stdout)
sys.path.insert(0,str(ROOT))
from interp_v1_4.p6 import verify,prepare,train,export
verify(ROOT)
''')
    md('## 2. 원본 검증과 네 층 전처리\n240개 cache 파일 checksum을 대조합니다. train 50k/val 10k의 h를 준비하고, train의 h/u/m scalar 통계 36개를 검증합니다. 원본 NPZ 크기와 Drive I/O에 따라 시간이 걸립니다. test는 학습·선택에 사용하지 않습니다.')
    code('''manifest=prepare(ROOT,SOURCE,OUTPUT)
print('통계',manifest['statistics_count'],'개 검증 완료')
''')
    md('## 3. 새 코드의 GPU smoke\n각 층의 SAE/TC debug 100 updates, hook·정규화·역변환·probe·패칭과 SAE k=4/16의 정확한 저장/재개를 확인합니다. Debug 결과는 본학습에 재사용하지 않습니다.')
    code('''SMOKE=OUTPUT/'smoke'/str(time.time_ns())
subprocess.run([sys.executable,'-m','interp_v1_4.p6','smoke','--root',str(ROOT),'--output',str(SMOKE),'--device','cuda'],cwd=ROOT,check=True)
print(json.loads((SMOKE/'smoke.json').read_text())['status'])
''')
    md('## 4. 본학습 및 자동 재개\n최초 100 updates 처리량/VRAM은 5,000 updates 예산 안에 포함됩니다. 매 250 updates 전체 validation 평가와 영속 checkpoint를 저장합니다. 결과의 best_checkpoint/last_checkpoint는 보존된 update 파일을 가리킵니다. 첫 run의 benchmark로 남은 시간의 대략적인 규모를 확인하세요. 숫자 오류는 실패 기록을 남기고 중단합니다.')
    code('''RUNS=None  # 예: ['seed0_l0_sae_k4_s0']; None은 24개 전체
subprocess.run([sys.executable,'-m','interp_v1_4.p6','train','--root',str(ROOT),'--output',str(OUTPUT),'--smoke',str(SMOKE/'smoke.json'),'--device','cuda'] + (['--runs']+RUNS if RUNS else []),cwd=ROOT,check=True)
''')
    md('## 5. 완료 run 독립 수치 검산\n24개 모두 끝난 경우 실행합니다. 각 평가 checkpoint의 validation MSE를 직접 행렬 연산으로 재계산하고, RNG draw cursor·optimizer·unit norm·선택 규칙을 확인합니다. 중단 상태라면 이 셀을 건너뛰고 결과를 다운로드하세요.')
    code("""subprocess.run([sys.executable,str(ROOT/'scripts/verify_v1_4_p6_return.py'),'--root',str(ROOT),'--output',str(OUTPUT),'--device','cuda'],cwd=ROOT,check=True)
""")
    md('## 6. 결과 다운로드\n완료 또는 중단 후 실행하세요. 모든 checkpoint·optimizer/RNG·통계·환경·실패 기록을 포함합니다. 원본 P5 cache와 재생성 가능한 h 입력 pool은 Drive에 남습니다. ZIP을 이 작업에 전달하면 24개 run과 선택 checkpoint를 감사합니다.')
    code('''archive=Path(f'/content/v1_4_p6_return_{time.time_ns()}.zip')
export(OUTPUT,archive)
print('SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())
files.download(str(archive))
files.download(str(archive.with_suffix('.sha256.json')))
''')
    notebook=ROOT/'experiment_v1_4/notebooks/06_colab_read_sae_r1.ipynb'
    write(notebook,dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),accelerator='GPU'),cells=[dict(c,id=f'p6-{i}') for i,c in enumerate(cells)]))
    print(json.dumps(dict(bundle=str(bundle),sha256=digest,notebook=str(notebook),runs=len(runs)),indent=2))

if __name__=='__main__':build()
