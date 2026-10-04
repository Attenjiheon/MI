"""Freeze exploratory P10 Update scope and deliver the first T4 training stage."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha,seed
from interp_v1_4.p10 import CONTRACT,LAYER,LABELS


def freeze():
    previous=ROOT/'experiment_v1_4/results/p9_final_audit_20261002_01/completion.json'
    p9=read(previous);assert p9['p9_complete'] and p9['p10_eligible']
    for n,h in p9['evidence_sha256'].items():assert sha(ROOT/n)==h,n
    data='data/language_v1_4/rebuild_01';cm=read(ROOT/data/'manifest.json')
    p5=read(ROOT/'experiment_v1_4/p5_r2/contract.json');frozen=read(ROOT/'experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json')
    paths={previous,ROOT/data/'manifest.json',ROOT/'experiment_v1_4/p5_r2/contract.json',ROOT/'experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json'}
    models=p5['models'];runs=[];baselines={}
    for m in models:
        lm=m['lm_seed'];cp=m['checkpoint'];digest=sha(ROOT/cp)
        original=next(x for x in frozen['models'] if x['lm_seed']==lm)
        assert digest==original['checkpoint_sha256']==p5['files'][cp]
        for name in ('checkpoint','init_checkpoint'):
            path=ROOT/m[name];assert sha(path)==p5['files'][m[name]];paths.add(path)
        for hook in ('h','u','m'):
            key=f'v1.4|P10-r1|lm_seed={lm}|checkpoint={digest}|layer={LAYER}|UPDATE|hook={hook}'
            baselines[f'seed{lm}_{hook}']=dict(key=key,projection_seed=seed('p10_projection',key),
                shuffled_label_seed=seed('p10_shuffle',key),bootstrap_seed=seed('p10_bootstrap',key),subsets={})
            for rep,width in [('coordinate',256),('random',512),('sae',512),('transcoder',512)]:
                s=seed('p10_subset',key+'|'+rep)
                baselines[f'seed{lm}_{hook}']['subsets'][rep]=dict(seed=s,ids=sorted(np.random.Generator(np.random.PCG64(s)).choice(width,128,replace=False).tolist()))
        for tool in ('sae','transcoder'):
            for k in (4,16):
                draw_key=f'v1.4|P10-r1|lm_seed={lm}|checkpoint={digest}|layer={LAYER}|UPDATE|k={k}|sparse_seed=0'
                rk=draw_key+'|tool='+tool;init=seed('p10_dictionary_init',rk)
                runs.append(dict(name=f'seed{lm}_l{LAYER}_update_{tool}_k{k}_s0',lm_seed=lm,checkpoint_sha256=digest,
                    layer=LAYER,position_type='UPDATE',tool=tool,k=k,sparse_seed=0,run_key=rk,draw_key=draw_key,
                    init_seed=init,torch_init_seed=init%(2**63-1),draw_seed=seed('p10_position_draw',draw_key)))
    for split in ('train','val','test'):
        for suffix in ('jsonl.gz','update_positions.json'):
            n=f'interpretation/{split}.{suffix}';path=ROOT/data/n
            assert sha(path)==cm['files'][n]['sha256'];paths.add(path)
    rules=dict(schema='p10-update-evaluation-rules-v1.4-r1',position='UPDATE',layer=LAYER,
        labels=LABELS,missing='SET/NOT: src, dst_before, src_before, input_disagreement, operator_truth excluded, never zero-filled',
        label_encoding=dict(operator=['SET','NOT','AND','OR','XOR'],variables=['A','B','C','D'],operator_truth='AND/OR/XOR x 00/01/10/11'),
        evaluation_scope=['semantic','fidelity','supervised_probe_transfer'],
        representations=dict(sae=['full_h','coordinates_h','random_h','sae_latents'],transcoder=['full_u','full_m','coordinates_m','random_m','transcoder_latents']),
        controls=['initialization checkpoint full h/u/m probes fit independently','token one-hot15 + position/767 + squared position','train shuffled labels, original validation/test','full candidate pool and fixed 128 candidates for coordinate/random/latent'],
        selection=dict(source='03 sections 6-7; existing P5/P7/P8 numerical recipe',train='dimensionwise standardization; constant <1e-8 removed; ANOVA F-score ranking, ties smaller feature ID',
            optimizer='CPU float64 SciPy L-BFGS-B: maxiter=2000, maxls=50, ftol=1e-12, gtol=1e-7; retry once maxiter=10000',
            lambdas=[.01,.1,1.,10.],thresholds=[i/20 for i in range(1,20)],prefixes=[1,2,3,4],
            loss='mean CE + lambda/2 * sum(weight^2), no bias penalty, no class weighting',
            single='m=1: validation BA max -> CE min -> larger lambda -> distance to 0.5 min -> smaller threshold',
            upto4='validation BA max -> smaller m -> CE min -> larger lambda -> distance to 0.5 min -> smaller threshold',
            multiclass='argmax, ties smaller class ID',support='train/val every class >=32 positions and >=16 distinct sequences, otherwise no fitting'),
        transfer=dict(labels=['dst_before','src_before','input_disagreement','dst_after'],variable='dst A/B/C train+val -> D test',
            operator='binary AND/OR train+val -> XOR test',refit='standardization/ranking/coefficients/lambda/m/threshold all refit on source domain; unsupervised dictionary sees all interp train',
            excluded=['operator ID','dst ID','src ID','operator_truth']),
        metrics=['balanced accuracy','macro F1','binary AUROC','confusion matrix','class counts/coverage','full-probe gaps'],
        bootstrap=dict(draws=1000,cluster='sequence_id',interval='percentile 2.5/97.5',invalid='report requested and valid draws',
            seed_rule='runtime.seed(p10_bootstrap, baseline key + |label|domain); common draw IDs for paired comparisons'),
        fidelity=dict(splits=['train','val','test'],target_sae='normalized h',target_tc='normalized m',metrics=['MSE','NMSE(train mean denominator)','R2(eval mean denominator)','EV','L0 mean/median/quantiles','latent activation rates','dead: no positive activation over complete train'],
            warning='Different targets: do not rank h SAE vs m TC solely by NMSE'),
        interventions=dict(status='skipped_for_this_optional_experiment',reason='Update-position intervention is a separate optional experiment under 03 section 11.2; this resource-bounded extension tests Update accessibility/relations/fidelity. READ causal analyses already complete. No Update causal-use claim.'),
        baselines=baselines,test_performance_used_for_choices=False,
        provenance='Exploratory extension after completed READ analyses; not preregistered before P5/P7/P8/P9 test observation',
        evaluation_delivery='After independent training return audit, bind selected checkpoints; no test-based reselection')
    rulepath=ROOT/'experiment_v1_4/p10_update_r1/evaluation_rules.json';write(rulepath,rules);paths.add(rulepath)
    for folder in ('interp_v1_4','archive/legacy/interp_v1_2','corpus'):paths.update((ROOT/folder).glob('*.py'))
    for n in ('scripts/build_v1_4_p10.py','scripts/verify_v1_4_p10_training.py','tests_v1_4/test_p10.py',
              'archive/legacy/experiment_v1_2/debug/sequences.json','01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md'):
        paths.add(ROOT/n)
    c=dict(schema='p10-update-v1.4-r1',date='2026-10-03',layer=LAYER,position_type='UPDATE',data_root=data,models=models,runs=runs,
        layer_reason='Block 3 is an existing required READ observation layer beyond block 0; choose one early/intermediate observation point for Update comparison without ranking layer test scores.',
        scope_authority='User requested P10 and supplied T4/24.59 CU; single-layer Update is the agent-selected default following phase priority, not an explicit user layer selection.',
        optional_decisions=dict(update='selected block 3 semantic/fidelity/probe transfer; 12 runs',other_layers='skipped: preserve limited T4 budget',length='skipped: prioritize Update',m_to_m_sae='skipped: no same-target tool ranking claim'),
        training=read(ROOT/'experiment_v1_4/p6_r1/contract.json')['training'],
        budget=dict(runs=12,updates_per_run=5000,total_updates=60000,total_draws=30720000,gpu='T4',reported_remaining_compute_units=24.59,
            compute_units_per_hour=None,wall_seconds_per_invocation=7200,cu_estimate='Use live Colab rate; no fixed conversion or completion guarantee',
            accounting='separate extraction, training, validation, numeric audit, CPU probes, bootstrap; evaluation time remains unmeasured'),
        quotas=dict(train=50000,val=10000,test=20000),chunk_sequences=128,
        rng_namespace='20260920|experiment-spec-v1.4|purpose|key; first 8 SHA256 bytes big-endian; PCG64',
        evaluation_rules=str(rulepath.relative_to(ROOT)),
        delivery_scope='Train/val trained-LM Update cache, 12 dictionary runs, all-checkpoint numeric audit. Test/init cache and semantic/fidelity evaluation follow audited training return.',
        completion='All selected Update runs and semantic/fidelity/transfer evaluation plus return/source audit; skipped optional analyses explicitly recorded. Preparation or training alone is not P10 completion.',
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
    write(ROOT/CONTRACT,c);return c,paths|{ROOT/CONTRACT}


def notebook(digest):
    cells=[]
    def add(kind,text):
        item=dict(cell_type=kind,metadata={},source=text.strip().splitlines(True),id=f'p10-{len(cells)}')
        if kind=='code':item.update(execution_count=None,outputs=[]);compile(text.strip(),'p10 notebook','exec')
        cells.append(item)
    add('markdown','''# P10 — Update block 3: T4 cache·SAE/TC 학습·검산

**준비·학습 전달물이며 P10 완료가 아닙니다.** LM seed 0/1/2 × SAE/TC × k=4/16 = 12 runs, 각각 5,000 updates. train 50k / validation 10k Update 위치를 사용합니다. 초기화 SET은 제외합니다. 별도 Update 전처리·RNG를 사용하고 상태 라벨은 loss에 넣지 않습니다.

1. T4 GPU 런타임을 선택합니다. 입력 ZIP을 `내 드라이브/boolean_interp_v1_4/`에 업로드합니다.
2. 아래 순서대로 실행합니다. 새 출력은 `P10_update_l3_r1`에 저장됩니다. P5 cache는 필요 없습니다.
3. 추출/학습 셀이 시간 제한으로 PAUSED이면 같은 셀을 다시 실행합니다. 런타임이 끊겼다면 맨 위부터 다시 실행합니다. 완전 저장 chunk/checkpoint부터 재개합니다.
4. 학습·수치 검산 후 마지막 셀에서 ZIP과 checksum을 내려받아 반환합니다. 오류나 자원 중단 때도 마지막 셀로 증빙을 보존합니다. Drive 원본 cache와 inputs는 삭제하지 마세요.

잔여 예산은 24.59 CU로 기록했습니다. 차감률은 Colab UI의 현재 값을 사용하며 고정 환산하지 않습니다. 추출·학습은 호출당 기본 2시간, chunk/250-update 경계에서 정지합니다. CPU 분석·최종 평가는 별도 후속 전달물입니다. 학습 전에 `evaluation_rules.json`에 라벨·전이·선택을 동결했습니다. Update 개입 실험·추가 층·length·m→m SAE는 이번 자원 범위에서 생략합니다.''')
    add('code',f'''
from google.colab import files, drive
from pathlib import Path
import hashlib, zipfile, subprocess, sys, json, time, shutil, os
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/boolean_interp_v1_4')
BUNDLE=BASE/'v1_4_p10_update_training_r1.zip'
assert BUNDLE.exists(), '입력 ZIP을 위 Drive 경로에 업로드하세요'
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
assert digest(BUNDLE)=='{digest}', '입력 ZIP checksum 불일치'
ROOT=Path('/content/MI_P10_update_r1');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    for n in z.namelist(): assert (ROOT/n).resolve().is_relative_to(ROOT.resolve()), 'ZIP 경로 오류'
    z.extractall(ROOT)
OUTPUT=BASE/'P10_update_l3_r1';OUTPUT.mkdir(parents=True,exist_ok=True)
print('checksum 검증 완료; 출력:',OUTPUT)
''')
    add('code','''
log=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','pytest==8.4.2','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(OUTPUT/f'install_{time.time_ns()}.txt').write_text(log.stdout)
print(log.stdout);log.check_returncode()
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(ROOT))
from interp_v1_4.p10 import verify, export
verify(ROOT)
def run(args):
    path=OUTPUT/'logs'/f'{time.time_ns()}.txt';path.parent.mkdir(exist_ok=True)
    with path.open('w') as log:
        proc=subprocess.Popen([sys.executable,'-u']+args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in proc.stdout: print(line,end='');log.write(line);log.flush()
            code=proc.wait()
        except BaseException:
            proc.terminate();proc.wait();raise
    if code:raise RuntimeError(f'실패 exit={code}; 로그 {path}; 마지막 셀로 증빙 반환')
run(['-m','pytest','tests_v1_4/test_p10.py','-q'])
''')
    add('markdown','''## 현재 GPU와 비용 기록
`CU_PER_HOUR`는 Colab UI에 표시된 현재 차감률을 아는 경우만 입력하세요. 값을 입력하면 요청한 CU 예산과 호출 시간 중 작은 값으로 제한합니다. 실제 사용량은 Colab UI로 확인하고 잔여량을 갱신하세요. 아래 시간은 한 셀 호출의 상한이며 전체 분석 시간 예측이 아닙니다.''')
    add('code','''
REMAINING_CU=24.59  # 이후 세션에서는 UI의 실제 잔여량으로 갱신
CU_PER_HOUR=None  # UI의 현재 값, 예측값을 넣지 마세요
MAX_SECONDS=7200
if CU_PER_HOUR is not None:
    assert CU_PER_HOUR>0 and REMAINING_CU>0
    MAX_SECONDS=min(MAX_SECONDS,int(REMAINING_CU/CU_PER_HOUR*3600*0.8))
assert MAX_SECONDS>0
import torch
assert torch.cuda.is_available(), 'GPU 런타임을 선택하세요'
print('현재 GPU:',torch.cuda.get_device_name(),'; 호출 시간 제한(초):',MAX_SECONDS)
(OUTPUT/f'budget_{time.time_ns()}.json').write_text(json.dumps(dict(reported_remaining_cu=REMAINING_CU,live_rate=CU_PER_HOUR,max_seconds=MAX_SECONDS,gpu=torch.cuda.get_device_name()),indent=2))
SMOKE=OUTPUT/'smoke'/str(time.time_ns())
run(['-m','interp_v1_4.p10','smoke','--root',str(ROOT),'--output',str(SMOKE),'--device','cuda'])
''')
    add('markdown','''## 1. Update train/validation cache 추출
각 LM의 동결 checkpoint를 사용합니다. 한 층의 h/u/m만 float32로 저장하며, metadata는 위치 선택·검증에만 사용합니다. 128개 sequence 단위로 Drive에 checksum과 함께 저장합니다. PAUSED이면 이 셀을 다시 실행하세요.''')
    add('code',"run(['-m','interp_v1_4.p10','extract','--root',str(ROOT),'--output',str(OUTPUT),'--smoke',str(SMOKE/'smoke.json'),'--max-seconds',str(MAX_SECONDS)])")
    add('markdown','''## 2. 입력 검사와 train 전용 scalar 통계
추출이 미완료이면 오류로 멈춥니다. 1번 셀부터 재개하세요. 9개 train 통계와 18개 train/val tensor를 검증합니다.''')
    add('code',"run(['-m','interp_v1_4.p10','prepare','--root',str(ROOT),'--output',str(OUTPUT)])")
    add('markdown','''## 3. SAE/TC 12 runs 학습
총 60,000 updates / 30.72M draws. 250 updates마다 전체 validation·checkpoint 저장, 100 update benchmark도 저장합니다. validation MSE 최소·동률은 이른 update를 선택합니다. PAUSED이면 이 셀을 다시 실행하세요. 전체 완료 전에는 다음 검산 셀이 성공하지 않습니다.''')
    add('code',"run(['-m','interp_v1_4.p10','train','--root',str(ROOT),'--output',str(OUTPUT),'--smoke',str(SMOKE/'smoke.json'),'--max-seconds',str(MAX_SECONDS)])")
    add('markdown','''## 4. 독립 수치 검산
252 checkpoint, 240 validation MSE, 선택·draw cursor·optimizer·RNG 및 9개 scalar 통계를 검증합니다. 학습 코드의 forward를 호출하지 않고 동결 연산 순서로 계산합니다. 실패 시 허용오차를 바꾸지 않고 증빙을 반환하세요.''')
    add('code',"run([str(ROOT/'scripts/verify_v1_4_p10_training.py'),'--root',str(ROOT),'--output',str(OUTPUT),'--device','cuda'])")
    add('markdown','''## 5. 완료 또는 중단 증빙 다운로드
전체 checkpoint·환경·로그·감사 결과를 반환합니다. 대형 cache/inputs 및 smoke tensor는 Drive에 유지하며 ZIP에서 제외합니다. 로컬 감사 후 별도 선택 checkpoint manifest와 의미·fidelity·전이 평가를 준비합니다. 새 평가에서 test로 재선택하지 않습니다.''')
    add('code','''
archive=Path(f'/content/v1_4_p10_training_return_{time.time_ns()}.zip')
export(OUTPUT,archive)
print('ZIP:',archive,'SHA256:',digest(archive))
exports=OUTPUT/'exports';exports.mkdir(exist_ok=True)
shutil.copyfile(archive,exports/archive.name)
shutil.copyfile(archive.with_suffix('.sha256.json'),exports/archive.with_suffix('.sha256.json').name)
files.download(str(archive));files.download(str(archive.with_suffix('.sha256.json')))
''')
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)


def main():
    c,paths=freeze();bundle=ROOT/'experiment_v1_4/bundles/P10/v1_4_p10_update_training_r1.zip';bundle.parent.mkdir(parents=True,exist_ok=True)
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        expected={**c['files'],CONTRACT:sha(ROOT/CONTRACT)}
        assert set(z.namelist())==set(expected)
        for n,h in expected.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    digest=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
    path=ROOT/'experiment_v1_4/notebooks/P10/P10_colab_update_training_r1.ipynb';write(path,notebook(digest))
    print(json.dumps(dict(notebook=str(path),bundle=str(bundle),sha256=digest,bytes=bundle.stat().st_size,members=len(expected)),indent=2))

if __name__=='__main__':main()
