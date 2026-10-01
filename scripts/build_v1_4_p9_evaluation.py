"""Bind 16 audited P9 dictionaries to the pre-training P7/P8 evaluation rules."""
from pathlib import Path
import copy,hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write,derived
from interp_v1_4.runtime import sha
from interp_v1_4.p9_sae_evaluation import records,RECEIPT
BASE='experiment_v1_4/p9_eval_r1'
AUDIT='experiment_v1_4/results/p9_training_audit_20260930_01'


def freeze():
    base=ROOT/AUDIT;complete=read(base/'completion.json')
    assert complete['p9_training_complete'] and complete['p9_evaluation_eligible'] and not complete['p9_complete']
    paths={base/'completion.json',ROOT/RECEIPT}
    for n,d in complete['evidence_sha256'].items():assert sha(base/n)==d;paths.add(base/n)
    selected=read(base/'selected_dictionary_manifest.json')['models'];assert len(selected)==16
    train=read(ROOT/'experiment_v1_4/p9_r1/contract.json')
    rp='experiment_v1_4/p9_r1/evaluation_rules.json';assert sha(ROOT/rp)==train['files'][rp];rules=read(ROOT/rp)
    parents={'sae':read(ROOT/'experiment_v1_4/p7_r1/contract.json'),'tc':read(ROOT/'experiment_v1_4/p8_eval_r1/contract.json')}
    models=[m for m in parents['sae']['models'] if m['lm_seed']==0];assert len(models)==1
    for e in selected:
        assert sha(ROOT/e['checkpoint'])==e['sha256'];paths.add(ROOT/e['checkpoint'])
    for m in models:
        assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'];paths.add(ROOT/m['checkpoint'])
    for n in [rp,'experiment_v1_4/p9_r1/contract.json','experiment_v1_4/p7_r1/contract.json','experiment_v1_4/p8_eval_r1/contract.json','experiment_v1_4/p5_r2/contract.json','experiment_v1_4/analysis_plan.json','scripts/build_v1_4_p9_evaluation.py','scripts/verify_v1_4_p9_sae_evaluation.py','scripts/verify_v1_4_p9_tc_evaluation.py','tests_v1_4/test_p9_sae_evaluation.py','tests_v1_4/test_p9_tc_evaluation.py','tests_v1_4/test_p9_evaluation_contract.py','01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md']:
        paths.add(ROOT/n)
    for folder in ('interp_v1_4','archive/legacy/interp_v1_2','corpus'):paths.update((ROOT/folder).glob('*.py'))
    data_root=parents['sae']['data_root'];dm=read(ROOT/data_root/'manifest.json')
    for split in ('train','val','test'):
        for suffix in ('jsonl.gz','read_positions.json'):
            p=ROOT/data_root/'interpretation'/f'{split}.{suffix}';assert sha(p)==dm['files'][str(p.relative_to(ROOT/data_root))]['sha256'];paths.add(p)
    pairs={}
    for name,quota in rules['shared']['causal_quotas'].items():
        p=ROOT/data_root/'causal_pairs'/f'{name}.jsonl.gz';assert sha(p)==dm['files'][str(p.relative_to(ROOT/data_root))]['sha256']
        pairs[name]=records(p);assert len(pairs[name])==quota;paths.add(p)
    p=ROOT/data_root/'test/general.jsonl.gz';assert sha(p)==dm['files']['test/general.jsonl.gz']['sha256'];paths.add(p)
    p=ROOT/rules['shared']['reconstruction_targets'];assert sha(p)==parents['sae']['files'][str(p.relative_to(ROOT))];paths.add(p)
    configs={}
    for tool in ('sae','tc'):
        parent=parents[tool];kind='sae' if tool=='sae' else 'transcoder'
        entries=[e for e in selected if e['run']['tool']==kind];assert len(entries)==8
        reuse={k:v for k,v in parent['reused_probes'].items() if k.startswith('seed0_')}
        for n in reuse.values():assert sha(ROOT/n)==parent['files'][n];paths.add(ROOT/n)
        rng={}
        for e in entries:
            run=e['run'];values={}
            for rows in pairs.values():
                for pair in rows:
                    for size in ('single','up_to_four'):
                        key=pair['pair_id']+'|'+size;values[key]=dict(latent=derived('random_patch',run['run_key']+'|'+key),direction=derived('random_patch',run['run_key']+'|directions|'+key))
            rng[run['name']]=values
        rngp=ROOT/BASE/f'{tool}_patch_rng.json';write(rngp,rng);paths.add(rngp)
        config=copy.deepcopy(rules['shared'])
        config.update(schema=f'p9-{tool}-evaluation-v1.4-r1',date='2026-09-30',models=models,data_root=data_root,baselines=rules['baselines'][kind],reused_probes=reuse,
            bootstrap_keys={e['run']['name']:rules['bootstrap_keys'][e['run']['name']] for e in entries},rng_namespace='20260909|experiment-spec-v1.0',patch_rng=str(rngp.relative_to(ROOT)),
            training_audit_sha256=sha(base/'verification.json'),frozen_rules_sha256=sha(ROOT/rp),selection_independent_per_dictionary=True,
            completion='8 tool evaluations plus other tool and initialization comparison and source return audit; P9 cannot complete automatically',execution_status='prepared; seed1 evaluation not executed')
        config['saes' if tool=='sae' else 'tcs']=entries;configs[tool]=config
    # Only audited aggregate tables from sparse seed 0; never reopen its test procedure.
    sources={'sae':'experiment_v1_4/results/p7_return_audit_20260927_01/returned','tc':'experiment_v1_4/results/p8_return_audit_20260929_01/returned_metadata'}
    tables={}
    for tool,folder in sources.items():
        receipt=read(ROOT/folder/'return_manifest.json');tables[tool]={}
        for name in ('semantic','fidelity','replacement','causal'):
            rel=f'tables/{name}.csv';p=ROOT/folder/rel;assert sha(p)==receipt['files'][rel];paths.add(p);tables[tool][name]=str(p.relative_to(ROOT))
    # Every shared source is frozen before either tool's test evaluation.
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    for tool,c in configs.items():c['files']=files;write(ROOT/BASE/f'{tool}_contract.json',c)
    cp=ROOT/BASE/'comparison.json';write(cp,dict(schema='p9-initialization-comparison-v1',seed0_tables=tables,tool_contracts={t:f'{BASE}/{t}_contract.json' for t in configs},files={n:sha(ROOT/n) for ts in tables.values() for n in ts.values()},selection_used_for_comparison=False))
    paths.update(ROOT/BASE/f'{t}_contract.json' for t in configs);paths.add(cp)
    return configs,paths


def delivery():
    configs,paths=freeze();bundle=ROOT/'experiment_v1_4/bundles/P9/v1_4_p9_evaluation_r1.zip';bundle.parent.mkdir(parents=True,exist_ok=True)
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        for p in paths:assert hashlib.sha256(z.read(str(p.relative_to(ROOT)))).hexdigest()==sha(p)
    digest=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,'p9-evaluation-notebook','exec');cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(True)))
    md('''# P9 — sparse seed 1 SAE/TC 16개 전체 평가

학습 반환 감사 통과 후의 **평가 전용** 노트북입니다. 학습을 다시 실행하지 않습니다.
ZIP `v1_4_p9_evaluation_r1.zip`을 내 드라이브 `boolean_interp_v1_4/`에 업로드하고 GPU 런타임에서 위부터 실행하세요.
기존 P5_r2를 읽고 새 결과를 P9_evaluation_r1/sae 및 /tc에 구분해 저장합니다.

LM seed 0 × block 0/3/7/11 × k=4/16, SAE 8개와 TC 8개입니다. 모든 train/val 선택을 동결한 뒤 test를 평가합니다.
SAE의 h 복원과 TC의 u→m 예측을 구분하며 TC 출력 패칭에는 s_m을 곱하고 원본 residual skip을 유지합니다.

중단 후 같은 셀을 재실행하면 완료 fit/선택/pair 단위부터 재개합니다. 다른 환경은 새 환경 ID와 GPU smoke를 기록합니다.
실행 로그와 단계별 시간을 Drive에 저장합니다. 실패·NA·matching coverage를 보존하며 test로 설정을 바꾸지 않습니다.
마지막 ZIP·checksum을 반환한 뒤 원본 cache 독립 재현 감사를 거쳐야 P9 완료를 판단합니다.''')
    code(f'''from google.colab import drive, files
from pathlib import Path
import hashlib,zipfile,subprocess,sys,time,json,os,shutil
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/boolean_interp_v1_4')
bundle=Path('/content/v1_4_p9_evaluation_r1.zip')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
EXPECTED='{digest}'
if not bundle.exists() or digest(bundle)!=EXPECTED:
    shutil.copyfile(BASE/bundle.name,bundle)
assert digest(bundle)==EXPECTED, '입력 ZIP checksum 불일치'
ROOT=Path('/content/MI_P9_evaluation_r1');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for i in z.infolist():assert (ROOT/i.filename).resolve().is_relative_to(ROOT.resolve())
    z.extractall(ROOT)
SOURCE=BASE/'P5_r2';OUTPUT=BASE/'P9_evaluation_r1';OUTPUT.mkdir(parents=True,exist_ok=True)
assert (SOURCE/'contract.json').exists(), 'P5 원본 경로 확인 필요'
''')
    md('## 1. 의존성·입력 검증\n설치 로그·환경 lock·GPU 정보를 보존합니다. 런타임 재시작이 필요하면 재시작 후 첫 셀부터 실행하세요.')
    code('''log=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','pytest==8.4.2','matplotlib==3.10.8','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(OUTPUT/f'install_{time.time_ns()}.txt').write_text(log.stdout);print(log.stdout);log.check_returncode()
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';sys.path.insert(0,str(ROOT))
from interp_v1_4 import p9_sae_evaluation,p9_tc_evaluation
for ev in (p9_sae_evaluation,p9_tc_evaluation):ev.verify(ROOT)
def logged(command):
    logs=OUTPUT/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'command_{time.time_ns()}.log';print('실행:', ' '.join(command),flush=True)
    with path.open('w',buffering=1) as f:
        f.write(json.dumps(dict(command=command,started_ns=time.time_ns()))+'\\n')
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in proc.stdout:f.write(line);print(line,end='',flush=True)
            rc=proc.wait();f.write(json.dumps(dict(returncode=rc,ended_ns=time.time_ns()))+'\\n')
            if rc:raise subprocess.CalledProcessError(rc,command)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            proc.stdout.close()
def stage(tool,action,*extra):
    logged([sys.executable,'-u','-m',f'interp_v1_4.p9_{tool}_runner',action,'--root',str(ROOT),'--source',str(SOURCE),'--output',str(OUTPUT/tool),*extra])
logged([sys.executable,'-m','pytest','tests_v1_4/test_p9_sae_evaluation.py','tests_v1_4/test_p9_tc_evaluation.py','tests_v1_4/test_p9_evaluation_contract.py','-q'])
''')
    md('## 2. 현재 환경 CUDA smoke\n두 도구의 네 층·두 k에서 identity·전체 donor·평균·근사·선택·random latent·좌표·random 방향과 양방향 changed/unchanged 경로를 확인합니다.')
    code('''SMOKES={}
for tool in ('sae','tc'):
    folder=OUTPUT/tool/'smoke'/str(time.time_ns());SMOKES[tool]=folder/'smoke.json'
    logged([sys.executable,'-u','-m',f'interp_v1_4.p9_{tool}_runner','smoke','--root',str(ROOT),'--output',str(folder),'--device','cuda'])
''')
    md('## 3. train/validation 선택 동결\nSeed 0 원본 cache checksum·READ key를 대조합니다. P5의 검증된 full probe와 block 0 대조군을 재사용하고 나머지 train/val probe를 적합합니다. 각 dictionary별 단일/≤4 feature, 전체·128후보, 현재/과거/상태·보조 label과 감독 변수 전이를 독립 선택합니다. 모든 선택을 마친 뒤 다음 셀로 진행합니다.')
    code("for tool in ('sae','tc'):stage(tool,'select','--device','cuda')\n")
    md('## 4. 의미·fidelity 평가\nBalanced accuracy/F1/AUROC·sequence bootstrap CI, 현재≠과거 subset·변수 전이, NMSE/R2/EV·L0와 train 전체 재인코딩 dead 비율을 저장합니다. Test는 선택에 사용하지 않습니다.')
    code("for tool in ('sae','tc'):stage(tool,'semantic','--device','cuda')\n")
    md('## 5. 한 READ 위치 대체·인과 평가\nRun마다 causal val norm bins를 고정하고 2,048개의 shared general READ target 및 changed/unchanged 각 1,024 pairs를 양방향으로 평가합니다. 두 feature 규모와 모든 대조군을 실행하고 pair별로 저장합니다. 동일 latent ID를 dictionary 간 동일 feature로 간주하지 않습니다.')
    code("for tool in ('sae','tc'):stage(tool,'patch','--device','cuda','--smoke',str(SMOKES[tool]))\n")
    md('## 6. 전체 표·그림·원시 결과 검산·초기화 비교\n각 도구 8개 모두 완료해야 집계합니다. 층간 paired CI와 기존 seed 0 결과에 대한 초기화 민감도 표·그림을 만듭니다. Seed 차이는 LM seed 0 안의 점 추정치 비교로 표시하며 독립 LM 반복으로 취급하지 않습니다. Full fitting/fidelity/CI 원본 재현 감사는 반환 후 별도로 수행합니다.')
    code('''for tool in ('sae','tc'):
    stage(tool,'aggregate')
    logged([sys.executable,'-u',str(ROOT/f'scripts/verify_v1_4_p9_{tool}_evaluation.py'),'--root',str(ROOT),'--output',str(OUTPUT/tool)])
logged([sys.executable,'-u','-m','interp_v1_4.p9_comparison','--root',str(ROOT),'--output',str(OUTPUT)])
''')
    md('## 7. 결과 ZIP·checksum 다운로드\n완료 또는 중단 후 실행합니다. 두 도구의 결과·환경·실패·재개 기록과 초기화 비교를 하나의 ZIP으로 반환합니다. 원본 P5 cache와 입력 모델은 중복 포함하지 않습니다. 다운로드한 두 파일을 이 작업에 전달하세요.')
    code('''archive=Path(f'/content/v1_4_p9_evaluation_return_{time.time_ns()}.zip')
from interp_v1_4.p9_sae_runner import export
export(OUTPUT,archive)
print('SHA256:',digest(archive))
files.download(str(archive));files.download(str(archive.with_suffix('.sha256.json')))
''')
    nb=ROOT/'experiment_v1_4/notebooks/P9/P9_colab_sparse_seed1_evaluation_r1.ipynb'
    write(nb,dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),accelerator='GPU'),cells=[dict(c,id=f'p9-eval-{i}') for i,c in enumerate(cells)]))
    print(json.dumps(dict(notebook=str(nb),bundle=str(bundle),sha256=digest,bytes=bundle.stat().st_size),indent=2))

if __name__=='__main__':delivery()
