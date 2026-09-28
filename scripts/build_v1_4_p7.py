"""Freeze P7 r1 rules and build a checksum-verified standalone Colab notebook."""
from pathlib import Path
import json
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write,derived,task_key,LABELS
from interp_v1_4.runtime import sha
from interp_v1_4.p7 import CONTRACT,SELECTED,RECEIPT,records


def freeze():
    base=ROOT/'experiment_v1_4/results/p6_final_audit_20260925'
    completion=read(base/'completion.json');assert completion['p6_complete'] and completion['p7_eligible']
    for name,digest in completion['evidence_sha256'].items():assert sha(base/name)==digest
    selected=read(ROOT/SELECTED)['models'];assert len(selected)==24
    frozen=read(ROOT/'experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json')['models']
    paths={ROOT/SELECTED,ROOT/RECEIPT,base/'completion.json',ROOT/'experiment_v1_4/results/p5_final_audit_20260925_01/completion.json',ROOT/'experiment_v1_4/analysis_plan.json',ROOT/'experiment_v1_4/p5_r2/contract.json',ROOT/'scripts/build_v1_4_p7.py',ROOT/'scripts/verify_v1_4_p7_return.py',ROOT/'tests_v1_4/test_p7.py'}
    for folder in ('interp_v1_4','archive/legacy/interp_v1_2','corpus'):paths.update((ROOT/folder).glob('*.py'))
    paths.update(ROOT/n for n in ('01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md'))
    for model in frozen:
        p=ROOT/model['checkpoint'];assert sha(p)==model['checkpoint_sha256'];paths.add(p)
    for e in selected:
        p=ROOT/e['checkpoint'];assert sha(p)==e['sha256'];paths.add(p)
    data_root='data/language_v1_4/rebuild_01';manifest=read(ROOT/data_root/'manifest.json');quotas={}
    for split in ('train','val','test'):
        for suffix in ('jsonl.gz','read_positions.json'):
            p=ROOT/data_root/'interpretation'/f'{split}.{suffix}';paths.add(p)
            assert sha(p)==manifest['files'][str(p.relative_to(ROOT/data_root))]['sha256']
    for split,changed,condition in __import__('itertools').product(('val','test'),('changed','unchanged'),('memory','composition')):
        name=f'{split}_{changed}_{condition}';p=ROOT/data_root/'causal_pairs'/f'{name}.jsonl.gz';paths.add(p)
        assert sha(p)==manifest['files'][str(p.relative_to(ROOT/data_root))]['sha256'];quotas[name]=len(records(p))
        assert quotas[name]==(256 if split=='val' else 512)
    general=ROOT/data_root/'test/general.jsonl.gz';paths.add(general);assert sha(general)==manifest['files']['test/general.jsonl.gz']['sha256']
    # Shared sequence target RNG excludes LM/tool/layer/k/sparse seed.
    target_key='v1.4|READ|general_test|single_position';target_seed=derived('reconstruction_target',target_key)
    rng=np.random.Generator(np.random.PCG64(target_seed));targets=[]
    for r in sorted(records(general),key=lambda r:r['sequence_id']):
        events=r['read_events'];e=events[int(rng.integers(len(events)))];targets.append(dict(sequence_id=r['sequence_id'],token_index=e['query_token_index'],event_id=e['read_id'],answer=e['answer']))
    assert len(targets)==2048
    target_path=ROOT/'experiment_v1_4/p7_r1/reconstruction_targets.json';write(target_path,targets);paths.add(target_path)
    p5=read(ROOT/'experiment_v1_4/p5_r2/contract.json');baselines={};reuse={};bootstrap_keys={}
    source=ROOT/'experiment_v1_4/results/p5_metadata_audit_20260924_01/returned/probes'
    audit=ROOT/'experiment_v1_4/results/p5_final_audit_20260925_01/returned/tasks'
    for model in frozen:
        seed=model['lm_seed']
        for layer in (0,3,7,11):
            key=task_key(seed,model['checkpoint_sha256'],layer,'h','trained')
            if layer==0:entry=p5['random'][key]
            else:
                entry=dict(projection_seed=derived('random_projection',key),subset_seeds={},subsets={})
                for rep,width in [('coordinate',256),('random',512),('sae',512),('transcoder',512)]:
                    value=derived('candidate_subset',key+'|'+rep);entry['subset_seeds'][rep]=value
                    entry['subsets'][rep]=np.random.Generator(np.random.PCG64(value)).choice(width,128,replace=False).tolist()
            baselines[f'seed{seed}_l{layer}']=entry
            for rep in (['full','coordinate','coordinate_128','random','random_128'] if layer==0 else ['full']):
                for label in LABELS:
                    for domain in (('iid','transfer') if label in ('current','previous') else ('iid',)):
                        name=f'seed{seed}_trained_l{layer}_h_{rep}_{label}_{domain}.json';p=source/name
                        checked=read(audit/name);assert checked['status']=='passed' and sha(p)==checked['source_sha256']
                        paths.update([p,audit/name]);reuse[f'seed{seed}_l{layer}_{rep}_{label}_{domain}']=str(p.relative_to(ROOT))
    for e in selected:
        r=e['run'];bootstrap_keys[r['name']]=f"v1.4|lm_seed={r['lm_seed']}|checkpoint={r['checkpoint_sha256']}|hook=h|READ|tool=sae|k={r['k']}|sparse_seed=0"
    # Freeze the per-pair RNG integers before any intervention effects are computed.
    patch_rng={}
    for e in selected:
        run=e['run'];values={}
        for split,change,condition in __import__('itertools').product(('val','test'),('changed','unchanged'),('memory','composition')):
            for pair in records(ROOT/data_root/'causal_pairs'/f'{split}_{change}_{condition}.jsonl.gz'):
                for size in ('single','up_to_four'):
                    key=pair['pair_id']+'|'+size
                    values[key]=dict(latent=derived('random_patch',run['run_key']+'|'+key),direction=derived('random_patch',run['run_key']+'|directions|'+key))
        patch_rng[run['name']]=values
    rng_path=ROOT/'experiment_v1_4/p7_r1/patch_rng.json';write(rng_path,patch_rng);paths.add(rng_path)
    config=dict(schema='p7-sae-evaluation-v1.4-r1',date='2026-09-26',saes=selected,models=frozen,data_root=data_root,baselines=baselines,reused_probes=reuse,
        bootstrap_keys=bootstrap_keys,rng_namespace='20260909|experiment-spec-v1.0',reconstruction_targets=str(target_path.relative_to(ROOT)),reconstruction_target_rng=dict(key=target_key,seed=target_seed,sequence_order='sorted sequence_id'),
        patch_rng=str(rng_path.relative_to(ROOT)),causal_quotas=quotas,random_candidates=200,matched_limit=20,unmatched_rule='first RNG-generated candidate; preserved irrespective of matching; no effect-based selection',random_direction_rule='selected probe directions and separate RNG subset of same m',
        selection='train ANOVA cumulative prefixes; P5 float64 L-BFGS grid and validation tie-breaks; m=1 and <=4; all and fixed 128 candidates',
        selection_scope='current value primary; previous/query/state/A/B/C/D auxiliary or exploratory',operator_transfer=dict(status='not_applicable',reason='READ labels have no AND/OR/XOR update source domain; P10 update analysis only'),
        causal_feature_source='full-candidate IID current-value SAE/coordinate/random probe; 128 controls semantic only',matching='pooled causal validation norms both directions, changed/unchanged memory/composition; zero bin; merged percentile edges; fixed test bins',
        bootstrap='1000 sequence/origin clusters; origin averages both directions and candidates; fixed conditions stratified macro; layer excluded from shared RNG keys',
        resume='immutable atomic JSON or checksum-committed NPZ; reject changed contract/selection/matching; no overwritten results',completion='24 full evaluations, required figures and independent returned evidence review; preparation or GPU report never completes P7',
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
    write(ROOT/CONTRACT,config);return config,paths|{ROOT/CONTRACT}


def delivery():
    config,paths=freeze();bundle=ROOT/'experiment_v1_4/bundles/P7/v1_4_p7_bundle_r1.zip';bundle.parent.mkdir(parents=True,exist_ok=True)
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        import hashlib
        for name,digest in {**config['files'],CONTRACT:sha(ROOT/CONTRACT)}.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    digest=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,'notebook','exec');cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md('''# P7 — 네 층 READ SAE 평가·개입\n\nGPU 런타임에서 위부터 실행합니다. 새 결과는 Drive `boolean_interp_v1_4/P7_r1`에 저장합니다. P5 원본 cache와 선택 SAE 24개를 유지합니다. 먼저 모든 train/validation feature를 고정하고, test 의미 평가 및 한 위치 대체·인과 평가를 실행합니다. P7 완료는 반환 감사 후 판정합니다.\n\n중단 후 같은 노트북을 실행하면 저장된 fit/선택/patch 단위부터 재개합니다. Colab 실제 GPU smoke는 새 환경마다 확인합니다. 모든 k·층·seed를 끝내며 결과가 나빠도 설정을 바꾸지 않습니다. 단계별 시간이 별도 기록됩니다. Test bins/candidate/feature를 재선택하지 않습니다.''')
    code(f'''from google.colab import files, drive
from pathlib import Path
import hashlib, zipfile, subprocess, sys, time, json, os
uploaded=files.upload()  # v1_4_p7_bundle_r1.zip
bundle=Path('v1_4_p7_bundle_r1.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='{digest}', '입력 ZIP checksum 불일치'
ROOT=Path('/content/MI_P7_r1');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for i in z.infolist():assert (ROOT/i.filename).resolve().is_relative_to(ROOT.resolve())
    z.extractall(ROOT)
drive.mount('/content/drive')
SOURCE=Path('/content/drive/MyDrive/boolean_interp_v1_4/P5_r2')
OUTPUT=Path('/content/drive/MyDrive/boolean_interp_v1_4/P7_r1');OUTPUT.mkdir(parents=True,exist_ok=True)
assert (SOURCE/'contract.json').exists(), 'P5 원본 경로 확인 필요'
''')
    md('## 1. 의존성 및 checksum 검증\n설치 후 재시작이 필요하면 재시작하고 업로드 셀부터 실행합니다. 설치 로그, pip freeze, GPU·환경 ID를 보존합니다.')
    code('''log=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','pytest==8.4.2','matplotlib==3.10.8','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(log.stdout);log.check_returncode();(OUTPUT/f'install_{time.time_ns()}.txt').write_text(log.stdout)
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT))
from interp_v1_4.p7 import verify
verify(ROOT)
def stage(action, *extra):
    subprocess.run([sys.executable,'-m','interp_v1_4.p7_runner',action,'--root',str(ROOT),'--source',str(SOURCE),'--output',str(OUTPUT),*extra],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-m','pytest','tests_v1_4/test_p7.py','-q'],cwd=ROOT,check=True)
''')
    md('## 2. 현재 GPU 코드 smoke\n네 층에서 identity, mean, approximation, full donor, 선택 feature, random latent, 좌표, 두 random 방향 대조와 양방향·changed/unchanged 경로를 debug 모델로 검사합니다. Debug feature·결과는 본실험에 재사용하지 않습니다.')
    code('''SMOKE=OUTPUT/'smoke'/str(time.time_ns())
subprocess.run([sys.executable,'-m','interp_v1_4.p7_runner','smoke','--root',str(ROOT),'--output',str(SMOKE),'--device','cuda'],cwd=ROOT,check=True)
''')
    md('## 3. 모든 train/validation 선택 동결\n원본 240 cache checksum·READ key·라벨을 검사합니다. P5 full probe와 block 0 대조군은 독립 감사된 계수를 재사용합니다. 다른 층의 좌표/random 및 SAE 후보 전체·128개에서 라벨별 단일/≤4 probe를 선택합니다. CPU fitting은 시간이 걸릴 수 있습니다. 개별 fit마다 저장하며 중단 후 재개됩니다.')
    code("stage('select','--device','cuda')\n")
    md('## 4. 동결된 의미 평가 및 fidelity\nTest의 balanced accuracy/F1/AUROC/CI, 현재≠과거와 감독 변수 전이, NMSE/R2/EV·L0·train 전체 재인코딩 dead 비율을 저장합니다. READ의 연산 전이는 적용 대상이 아닙니다. Test로 feature를 바꾸지 않습니다.')
    code("stage('semantic','--device','cuda')\n")
    md('## 5. 한 위치 대체·인과 평가\n24개 순서대로 실행합니다. Causal validation의 norm bins를 먼저 저장합니다. Shared general test READ 2,048개를 층당 한 곳씩 대체하고, changed/unchanged 각 1,024 pair를 memory/composition과 양방향으로 평가합니다. Random latent 후보 최대 200개를 norm으로 분류하고 matched 최대 20개, unmatched는 사전 RNG 첫 후보를 평가합니다. Matching 실패를 NA·coverage로 보존합니다. 각 pair를 저장하므로 중단해도 완료 pair를 다시 평가하지 않습니다.')
    code("stage('patch','--device','cuda','--smoke',str(SMOKE/'smoke.json'))\n")
    md('## 6. 표·그림·층간 paired CI 및 원시 결과 검산\n모든 run이 있어야 집계합니다. 재현용 raw logits·norm·candidate·probe prediction을 함께 반환합니다. 검산은 저장된 logits에서 metric을 다시 계산하고 bin/feature/hash를 대조합니다. 전체 fitting·fidelity·CI의 독립 재현은 반환 후 cache 원본을 사용한 별도 감사 대상입니다.')
    code("stage('aggregate')\nsubprocess.run([sys.executable,str(ROOT/'scripts/verify_v1_4_p7_return.py'),'--root',str(ROOT),'--output',str(OUTPUT)],cwd=ROOT,check=True)\n")
    md('## 7. 결과 다운로드\n완료 또는 중단 시 실행합니다. 아래 ZIP과 checksum JSON을 이 작업에 전달하면 실제 반환 감사를 진행합니다. P5 대형 cache와 입력 checkpoint는 ZIP에 다시 넣지 않습니다. 미완료 run도 실패/환경/재개 기록이 보존됩니다.')
    code("archive=Path(f'/content/v1_4_p7_return_{time.time_ns()}.zip')\nstage('export','--archive',str(archive))\nfiles.download(str(archive))\nfiles.download(str(archive.with_suffix('.sha256.json')))\n")
    notebook=ROOT/'experiment_v1_4/notebooks/P7/P7_colab_read_sae_evaluation_r1.ipynb'
    write(notebook,dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),accelerator='GPU'),cells=[dict(c,id=f'p7-{i}') for i,c in enumerate(cells)]))
    print(json.dumps(dict(notebook=str(notebook),bundle=str(bundle),bytes=bundle.stat().st_size,sha256=digest,runs=24),indent=2))

if __name__=='__main__':delivery()
