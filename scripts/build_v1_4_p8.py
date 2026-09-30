"""P8 first delivery: frozen TC training and downstream evaluation rules.

Training is audited before binding selected checkpoints to the evaluation
runner. This delivery cannot mark P8 complete or launch P9.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write,derived,task_key
from interp_v1_4.runtime import sha
from interp_v1_4.p8 import CONTRACT,RECEIPT,P6_METADATA


def freeze():
    paths=set()
    for phase,folder in [('p5','p5_final_audit_20260925_01'),('p6','p6_final_audit_20260925'),('p7','p7_final_audit_20260928_01')]:
        base=ROOT/'experiment_v1_4/results'/folder;c=read(base/'completion.json')
        assert c[phase+'_complete']
        for name,digest in c['evidence_sha256'].items():
            assert sha(base/name)==digest,(phase,name)
            paths.add(base/name)
        paths.add(base/'completion.json')
    p6=read(ROOT/'experiment_v1_4/p6_r1/contract.json');p7=read(ROOT/'experiment_v1_4/p7_r1/contract.json')
    runs=[]
    for old in p6['runs']:
        r=copy.deepcopy(old);r['name']=r['name'].replace('_sae_','_tc_')
        r.update(tool='transcoder',hook='u_to_m',input_hook='u',target_hook='m')
        key=r['draw_key']+'|hook=u_to_m|tool=transcoder';r['run_key']=key
        r['init_seed']=derived('dictionary_init',key);r['torch_init_seed']=r['init_seed']%(2**63-1)
        assert r['draw_seed']==derived('position_draw',r['draw_key'])
        runs.append(r)
    stats=read(ROOT/P6_METADATA/'input_manifest.json')
    for n,d in stats['statistics'].items():
        p=ROOT/P6_METADATA/n;assert sha(p)==d;paths.add(p)
    paths.add(ROOT/P6_METADATA/'input_manifest.json')
    # Freeze downstream choices without reading any TC test activations or outcomes.
    baselines={};bootstrap={}
    p5=read(ROOT/'experiment_v1_4/p5_r2/contract.json')
    for r in runs:
        key=task_key(r['lm_seed'],r['checkpoint_sha256'],r['layer'],'m','trained')
        if r['layer']==0:entry=copy.deepcopy(p5['random'][key])
        else:
            entry=dict(projection_seed=derived('random_projection',key),subset_seeds={},subsets={})
            for rep,width in [('coordinate',256),('random',512),('sae',512),('transcoder',512)]:
                seed=derived('candidate_subset',key+'|'+rep)
                entry['subset_seeds'][rep]=seed
                entry['subsets'][rep]=np.random.Generator(np.random.PCG64(seed)).choice(width,128,replace=False).tolist()
        baselines[f"seed{r['lm_seed']}_l{r['layer']}"]=dict(key=key,**entry)
        bootstrap[r['name']]=f"v1.4|lm_seed={r['lm_seed']}|checkpoint={r['checkpoint_sha256']}|hook=m|READ|tool=transcoder|k={r['k']}|sparse_seed=0"
    rules={k:copy.deepcopy(p7[k]) for k in ('causal_quotas','random_candidates','matched_limit','unmatched_rule','random_direction_rule','selection','selection_scope','operator_transfer','matching','bootstrap','reconstruction_targets','reconstruction_target_rng')}
    rules.update(schema='p8-evaluation-rules-v1.4-r1',baselines=baselines,bootstrap_keys=bootstrap,
        comparison=['full_u','full_m','m_coordinate','m_random','transcoder'],
        checkpoint_selection=p6['training']['selection'],
        input_target='u_l -> m_l; separate train mean and scalar scale',
        intervention='m_original + s_m * D[:,J] @ (z_donor[J]-z_original[J]); original residual skip retained',
        causal_feature_source='full-candidate IID current-value TC/m-coordinate/m-random probes; single and <=4; 128 controls semantic only',
        fidelity='m prediction NMSE/R2/EV; encode every train u for dead fraction; h-SAE and m-TC targets explicitly distinct',
        rng_namespace='20260909|experiment-spec-v1.0',
        patch_rng_rule='derived(random_patch, run_key + |pair_id|size); directions add |directions before pair_id; PCG64',
        test_used_to_set_rules=False,execution_status='pending selected TC checkpoint return audit; no evaluation results yet')
    rule_path=ROOT/'experiment_v1_4/p8_r1/evaluation_rules.json';write(rule_path,rules);paths.add(rule_path)
    paths.add(ROOT/rules['reconstruction_targets'])
    for name in (RECEIPT,'experiment_v1_4/p6_r1/contract.json','experiment_v1_4/p7_r1/contract.json','experiment_v1_4/p5_r2/contract.json','experiment_v1_4/analysis_plan.json','archive/legacy/experiment_v1_2/debug/sequences.json','scripts/build_v1_4_p8.py','scripts/verify_v1_4_p8_training.py','tests_v1_4/test_p8.py','experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json'):
        paths.add(ROOT/name)
    for folder in ('interp_v1_4','archive/legacy/interp_v1_2','corpus'):paths.update((ROOT/folder).glob('*.py'))
    paths.update(ROOT/n for n in ('01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md'))
    config=dict(schema='p8-transcoder-v1.4-r1',date='2026-09-29',runs=runs,training=p6['training'],budget=p6['budget'],
        rng_namespace=p6['rng_namespace'],draw_key_policy=p6['draw_key_policy'],
        initialization='Independent normal decoder columns and encoder rows, each unit norm; biases zero',
        input_gate='P7 evidence hashes; P5 240 cache hashes and identical READ keys; compare all 36 train statistics to audited P6; paired u/m train and val; current-source CUDA smoke and TC resume',
        evaluation_rules=str(rule_path.relative_to(ROOT)),
        delivery_scope='TC training, checkpoint numerical audit and frozen evaluation rules; selected checkpoint binding and full evaluation delivery follow returned training audit',
        completion='24 TC training runs plus every required semantic/fidelity/replacement/causal evaluation, u/m controls, figures and returned evidence audit; training alone never completes P8',
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
    write(ROOT/CONTRACT,config)
    return config,paths|{ROOT/CONTRACT}


def delivery():
    config,paths=freeze();bundle=ROOT/'experiment_v1_4/bundles/P8/v1_4_p8_training_r1.zip';bundle.parent.mkdir(parents=True,exist_ok=True)
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        for n,d in {**config['files'],CONTRACT:sha(ROOT/CONTRACT)}.items():assert hashlib.sha256(z.read(n)).hexdigest()==d,n
    digest=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,'notebook','exec');cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(True)))
    md('''# P8 1차 — READ Transcoder 24개 학습·반환 감사

GPU 런타임을 선택하고 위부터 실행합니다. `v1_4_p8_training_r1.zip`을 **내 드라이브/boolean_interp_v1_4/**에 업로드해 두세요. 기존 P5 원본 `P5_r2`를 입력으로 읽고 새 결과는 `P8_training_r1`에 저장합니다.

세 LM × block 0·3·7·11 × k=4/16, sparse seed 0입니다. 각 TC는 같은 위치의 MLP 입력 u에서 MLP 출력 m을 예측합니다. 대응 SAE와 같은 position draw 순서, 5,000 updates/512 positions, 250 updates마다 전체 validation MSE 선택 규칙을 유지합니다.

**이 노트북은 학습 단계입니다.** 학습 결과를 반환·감사한 뒤 동결한 규칙대로 의미·fidelity·대체·인과 평가를 이어갑니다. 학습만으로 P8을 완료하지 않습니다. Test activation이나 상태 라벨은 학습·checkpoint 선택에 쓰지 않습니다.

중단되면 같은 노트북을 다시 실행합니다. 마지막 checksum이 있는 checkpoint에서 optimizer·RNG·draw cursor를 복구합니다. 다른 환경이면 새 환경 ID와 GPU smoke가 필요합니다. 완료 run과 기존 결과는 보존합니다.''')
    code(f'''from google.colab import files, drive
from pathlib import Path
import hashlib, zipfile, subprocess, sys, json, time, shutil, os
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/boolean_interp_v1_4')
BUNDLE_IN_DRIVE=BASE/'v1_4_p8_training_r1.zip'
bundle=Path('/content/v1_4_p8_training_r1.zip')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
expected='{digest}'
if not bundle.exists() or digest(bundle)!=expected:
    assert BUNDLE_IN_DRIVE.exists(), '입력 ZIP을 Drive 경로에 업로드하세요'
    shutil.copyfile(BUNDLE_IN_DRIVE,bundle)
assert digest(bundle)==expected, '입력 ZIP checksum 불일치'
ROOT=Path('/content/MI_P8_training_r1');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for item in z.infolist():assert (ROOT/item.filename).resolve().is_relative_to(ROOT.resolve()), '잘못된 ZIP 경로'
    z.extractall(ROOT)
SOURCE=BASE/'P5_r2'
OUTPUT=BASE/'P8_training_r1';OUTPUT.mkdir(parents=True,exist_ok=True)
assert (SOURCE/'contract.json').exists(), 'P5 원본 cache 경로 확인 필요'
print('입력 ZIP checksum 검증 완료', bundle.stat().st_size, 'bytes')
''')
    md('## 1. 의존성·환경 준비\n실행 환경은 P6/P7과 같은 버전을 설치합니다. 설치 로그·pip freeze·GPU 정보·환경 ID를 저장합니다. 재시작 안내가 나오면 런타임 재시작 후 첫 셀부터 실행하세요.')
    code('''log=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','pytest==8.4.2','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(OUTPUT/f'install_{time.time_ns()}.txt').write_text(log.stdout)
print(log.stdout);log.check_returncode()
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT))
from interp_v1_4.p8 import verify,prepare,export
verify(ROOT)
subprocess.run([sys.executable,'-m','pytest','tests_v1_4/test_p8.py','-q'],cwd=ROOT,check=True)
''')
    md('## 2. 원본 checksum·READ key·입출력 전처리\n240개 cache checksum을 확인하고 train 50k/val 10k의 네 층 u/m을 준비합니다. 36개 train scalar 통계가 감사된 P6 값과 정확히 같은지 대조합니다. 대형 P5 입력은 Drive에서 읽으므로 시간이 걸립니다. test는 checksum만 확인하며 학습·선택용 tensor로 만들지 않습니다.')
    code("manifest=prepare(ROOT,SOURCE,OUTPUT)\nprint('검증 완료:',manifest['statistics_count'],'통계;',len(manifest['tensors']),'train/val tensors')\n")
    md('## 3. 현재 코드의 CUDA smoke와 정확한 재개\n네 층의 hook·identity/full/sparse patch, debug dictionary 100 updates, TC k=4/16의 model·optimizer·draw RNG가 중단 없는 실행과 bitwise 일치하는지 확인합니다. 입력 s_u와 출력 s_m은 서로 다른 값으로 검사합니다. Debug 데이터는 본학습에 재사용하지 않습니다.')
    code("SMOKE=OUTPUT/'smoke'/str(time.time_ns())\nsubprocess.run([sys.executable,'-m','interp_v1_4.p8','smoke','--root',str(ROOT),'--output',str(SMOKE),'--device','cuda'],cwd=ROOT,check=True)\n")
    md('## 4. TC 24 runs 학습·재개\n최초 100 updates의 처리량·peak VRAM 측정은 5,000 updates에 포함됩니다. 250 updates마다 저장·validation 평가합니다. 마지막 평가까지 끝난 후 최소 val MSE checkpoint를 선택하며 동률은 이른 update입니다. 재실행하면 완료 run은 확인하고 중단 run은 마지막 완전 저장 경계에서 재개합니다.')
    code('''RUNS=None  # 일부 실행 예: ['seed0_l0_tc_k4_s0']; 완료에는 24개 모두 필요
subprocess.run([sys.executable,'-m','interp_v1_4.p8','train','--root',str(ROOT),'--output',str(OUTPUT),'--smoke',str(SMOKE/'smoke.json'),'--device','cuda']+(['--runs']+RUNS if RUNS else []),cwd=ROOT,check=True)
''')
    md('## 5. 전체 학습 수치 검산\n24개가 모두 끝난 뒤 실행합니다. 504개 checkpoint, 480개 validation MSE, optimizer·RNG cursor·unit norm·입출력 통계·최소 MSE 선택을 독립 검산합니다. F.linear·float32 tensor scale·stable TopK의 수치 경로를 유지합니다. 검산은 선택 규칙을 변경하지 않습니다. 중단 상태는 이 셀을 건너뛰고 다운로드하세요.')
    code("subprocess.run([sys.executable,str(ROOT/'scripts/verify_v1_4_p8_training.py'),'--root',str(ROOT),'--output',str(OUTPUT),'--device','cuda'],cwd=ROOT,check=True)\n")
    md('## 6. 결과 다운로드\n완료 또는 중단 뒤 실행합니다. 모든 checkpoint와 optimizer/RNG, 입력 manifest·통계·환경·검산·실패 기록을 포함합니다. ZIP과 checksum JSON을 이 작업에 전달하세요. Drive 결과는 보존합니다. 재생성 가능한 u/m tensor와 원본 P5 cache는 ZIP에 중복 포함하지 않습니다. 선택 TC 감사 후 별도 평가 노트북으로 이어갑니다.')
    code("archive=Path(f'/content/v1_4_p8_training_return_{time.time_ns()}.zip')\nexport(OUTPUT,archive)\nprint('SHA256:',digest(archive))\nfiles.download(str(archive))\nfiles.download(str(archive.with_suffix('.sha256.json')))\n")
    notebook=ROOT/'experiment_v1_4/notebooks/P8/P8_colab_read_tc_training_r1.ipynb'
    write(notebook,dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),accelerator='GPU'),cells=[dict(c,id=f'p8-{i}') for i,c in enumerate(cells)]))
    print(json.dumps(dict(notebook=str(notebook),bundle=str(bundle),sha256=digest,bytes=bundle.stat().st_size),indent=2))

if __name__=='__main__':delivery()
