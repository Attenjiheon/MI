"""Freeze P5 delivery inputs and build a resumable Colab notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import CONTRACT,HOOKS,LABELS,derived,task_key,write,read,load_split,support_table
from interp_v1_4.runtime import sha
FOLDER=ROOT/'experiment_v1_4/p5_r2'
PREP=ROOT/'experiment_v1_4/results/p5_preparation_r2'


def prepare():
    base=ROOT/'experiment_v1_4/results/frozen_test_audit_20260922_01'
    completion=read(base/'completion.json');frozen=read(base/'frozen_lms.json')
    assert completion['p4_complete'] and completion['p5_eligible'] and frozen['passing_lm_seeds']==[0,1,2]
    for name,digest in completion['evidence_sha256'].items():assert sha(base/name)==digest
    freeze=ROOT/'experiment_v1_4/results/p4_audit_20260921_01/validation_freeze.json'
    assert sha(freeze)==frozen['validation_freeze_sha256']
    paths={base/'completion.json',base/'frozen_lms.json',freeze,
        ROOT/'experiment_v1_4/corpus_rebuild.json',ROOT/'experiment_v1_4/design_config.json',
        ROOT/'archive/legacy/experiment_v1_2/debug/sequences.json',
        ROOT/'experiment_v1_4/frozen_test_r1/requirements-primary.lock.txt'}
    paths.update(ROOT/p for p in ['01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md'])
    paths.update((ROOT/'interp_v1_4').glob('*.py'))
    paths.update((ROOT/'archive/legacy/interp_v1_2').glob('*.py'))
    paths.update(ROOT/p for p in ['tests_v1_4/test_p5.py','scripts/build_v1_4_p5.py','scripts/verify_v1_4_p5.py'])
    models=[];sources=[]
    for model in frozen['models']:
        seed=model['lm_seed'];checkpoint=ROOT/model['checkpoint'];assert sha(checkpoint)==model['checkpoint_sha256'];paths.add(checkpoint)
        archive=next((ROOT/'experiment_v1_4/evidence').glob(f'v1_4_seed{seed}_evidence_*.zip'))
        audit=ROOT/('experiment_v1_4/results/p3_audit_20260921_01/archive_verification.json' if seed==0 else f'experiment_v1_4/results/p4_audit_20260921_01/seed{seed}/archive_verification.json')
        verification=read(audit);digest=verification.get('archive_sha256',verification.get('sha256'));assert sha(archive)==digest
        init=FOLDER/f'checkpoints/init_seed{seed}.pt';init.parent.mkdir(parents=True,exist_ok=True)
        member=f'deepwide12_read4_seed{seed}/checkpoints/init.pt'
        with zipfile.ZipFile(archive) as z:
            if not init.exists():
                with z.open(member) as src,init.open('xb') as dst:shutil.copyfileobj(src,dst)
            assert hashlib.sha256(z.read(member)).hexdigest()==sha(init)
        paths.add(init);paths.add(audit)
        models.append(dict(lm_seed=seed,checkpoint=model['checkpoint'],init_checkpoint=str(init.relative_to(ROOT)),selected_update=model['selected_update']))
        sources.append(dict(lm_seed=seed,archive=str(archive.relative_to(ROOT)),archive_sha256=digest,init_member=member,init_sha256=sha(init)))
    data_root='data/language_v1_4/rebuild_01';manifest=ROOT/data_root/'manifest.json';paths.add(manifest)
    assert sha(manifest)==frozen['models'][0]['training_hashes']['corpus_manifest']
    cm=read(manifest)
    for split in ('train','val','test'):
        for suffix in ('jsonl.gz','read_positions.json'):
            name=f'interpretation/{split}.{suffix}';p=ROOT/data_root/name
            assert sha(p)==cm['files'][name]['sha256'];paths.add(p)
    randoms={};tasks={}
    for model in models:
        seed=model['lm_seed']
        for layer in range(12):
            for hook in HOOKS:
                for kind in ('trained','init'):
                    cp=model['checkpoint'] if kind=='trained' else model['init_checkpoint']
                    key=task_key(seed,sha(ROOT/cp),layer,hook,kind)
                    variants=['full']+(['shuffled'] if kind=='trained' else [])
                    if kind=='trained' and layer==0 and hook in ('h','m'):
                        entry=dict(projection_seed=derived('random_projection',key),subset_seeds={},subsets={})
                        for representation,width in [('coordinate',256),('random',512),('sae',512),('transcoder',512)]:
                            value=derived('candidate_subset',key+'|'+representation)
                            entry['subset_seeds'][representation]=value
                            entry['subsets'][representation]=np.random.Generator(np.random.PCG64(value)).choice(width,128,replace=False).tolist()
                        randoms[key]=entry;variants+=['coordinate','coordinate_128','random','random_128']
                    for variant in variants:
                        for label in LABELS:
                            for domain in (('iid','transfer') if label in ('current','previous') else ('iid',)):
                                name=f'seed{seed}_{kind}_l{layer}_{hook}_{variant}_{label}_{domain}'
                                task=key+'|'+variant+'|'+label+'|'+domain
                                tasks[name]=dict(key=task,bootstrap_seed=derived('bootstrap',task),shuffle_seed=derived('shuffle_label',task) if variant=='shuffled' else None)
        for label in LABELS:
            for domain in (('iid','transfer') if label in ('current','previous') else ('iid',)):
                task=f'v1.4|lm_seed={seed}|token_position|READ|{label}|{domain}'
                tasks[f'seed{seed}_token_position_{label}_{domain}']=dict(key=task,bootstrap_seed=derived('bootstrap',task),shuffle_seed=None)
    config=dict(schema='p5-read-v1.4-r2',data_root=data_root,models=models,init_sources=sources,
        layers=list(range(12)),hooks=HOOKS,position_type='READ',quotas=dict(train=50000,val=10000,test=20000),
        chunk_sequences=128,microbatch=16,oom_microbatches=[16,8,4,2,1],dtype='float32',
        raw_cache_bytes=6*80000*12*3*256*4,
        execution_order='split train/val/test -> seed0/1/2 -> trained/init; CPU probes after validated extraction',
        probe=dict(lambdas=[.01,.1,1,10],thresholds=(np.arange(1,20)/20).tolist(),dtype='float64',
          solver='L-BFGS-B',maxiter=[2000,10000],maxls=50,ftol=1e-12,gtol=1e-7,
          selection=['max val balanced accuracy','smaller m (prefix selection only)','min val CE','larger lambda','threshold nearer .5','smaller threshold'],
          support=dict(min_positions_per_class=32,min_sequences_per_class=16),
          labels=LABELS,transfer_labels=['current','previous'],transfer='train/val query ABC -> test query D',
          difference_subset='evaluate fixed probe on current != previous; no subset refitting',
          controls='full trained/init/shuffled on all 12 layers h/u/m; token_position per LM; block0 h/m coordinate and random single/<=4 with full and 128 pools',
          bootstrap=dict(draws=1000,cluster='sequence_id',ci=[2.5,97.5]),
          constant_columns='std < 1e-8 removed; all constant -> failed analysis; no fabricated coefficients'),
        random=randoms,tasks=tasks,files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
        cache_key=['lm_seed','checkpoint_sha256','split','position_type','layer','sequence_id','token_index','hook'],
        resume='atomic 128-sequence chunks and per-probe JSON; checksum and config identity required; incomplete transactions recomputed; environments recorded per unit',
        test_policy='All procedures fixed before interpretation test scoring; test cannot change selection; no behavior gate/test inference',
        p5_complete=False)
    write(ROOT/CONTRACT,config)
    support={}
    for split in config['quotas']:
        _,rows=load_split(ROOT,config,split);support[split]=support_table(rows)
    write(PREP/'input_verification.json',dict(status='passed_input_metadata_only',config_sha256=sha(ROOT/CONTRACT),
        quotas=config['quotas'],support=support,models=models,init_sources=sources,probe_tasks=len(tasks),p5_complete=False))
    print('Frozen contract',sha(ROOT/CONTRACT),'probe tasks',len(tasks))


def notebook(bundle,digest):
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text))
    def code(text):cells.append(dict(cell_type='code',metadata={},source=text,execution_count=None,outputs=[]))
    md('# P5 v1.4 — READ cache 및 probe\nGPU 런타임에서 위에서 아래로 실행합니다. 세 동결 LM과 원본 초기 LM의 12층 h/u/m을 예약 READ 위치에서 추출합니다. cache 약 17.7GB, ZIP·재개 복사를 위해 로컬 여유 45GB와 Drive 여유 40GB 이상을 확보하세요. 완료한 행동 gate/test를 실행하지 않습니다.\n\n1. 입력 ZIP 업로드·검증 → 2. 의존성 설치 → 3. Drive 연결/재개 → 4. 새 코드 GPU smoke 및 cache → 5. CPU probe → 6. 증빙 ZIP 저장·다운로드. 중단 시 동일 ZIP으로 처음부터 실행하면 checksum 완료 단위부터 재개합니다. Probe는 CPU 계산량이 크며 진행 결과를 단위별 저장합니다. cache만 먼저 반환해 로컬 CPU로 이어갈 수도 있습니다.')
    code(f'''from google.colab import files
from pathlib import Path
import hashlib, json, zipfile, os, shutil, subprocess, sys
uploaded=files.upload()
bundle=Path({bundle!r})
assert bundle.exists(), '지정된 P5 입력 ZIP을 업로드하세요.'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
assert sha(bundle)=={digest!r}, '입력 ZIP checksum 불일치'
ROOT=Path('/content/boolean_interp');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    inventory=json.loads(z.read('bundle_manifest.json'))['files']
    assert set(z.namelist())==set(inventory)|{{'bundle_manifest.json'}}
    for name,d in inventory.items():
        dest=ROOT/name
        assert dest.resolve().is_relative_to(ROOT.resolve())
        if dest.exists(): assert sha(dest)==d, '기존 파일 충돌: '+name
        else: z.extract(name,ROOT)
        assert sha(dest)==d
os.chdir(ROOT)
del uploaded
print('입력 검증 완료')''')
    md('## 의존성 설치\n이전 실제 GPU 환경에서 저장한 주요 패키지 버전을 적용합니다. Colab 이미지가 달라지면 실행기가 새 환경 ID를 기록하고 현재 환경에서 smoke를 다시 수행합니다. 설치 후 런타임 재시작 안내가 나오면 재시작하고 셀을 다시 실행하세요.')
    code("subprocess.run([sys.executable,'-m','pip','install','-r','experiment_v1_4/frozen_test_r1/requirements-primary.lock.txt','--extra-index-url','https://download.pytorch.org/whl/cu128'],check=True)\nsubprocess.run([sys.executable,'-m','pytest','tests_v1_4/test_p5.py','-q'],check=True)")
    md('## 영속 저장·재개\nDrive의 전용 폴더를 사용합니다. 저장 파일을 수정하지 마세요. 기존 결과와 다른 계약은 자동으로 거부합니다. 새 런타임에서는 Drive의 완료 조각을 로컬로 복사하고 checksum을 검사합니다.')
    code('''from google.colab import drive
drive.mount('/content/drive')
PERSIST=Path('/content/drive/MyDrive/boolean_interp_v1_4/P5_r2')
OUT=ROOT/'experiment_v1_4/runs/p5_r2'
PERSIST.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
for src in PERSIST.rglob('*'):
    if not src.is_file() or src.name.endswith('.tmp'):continue
    dest=OUT/src.relative_to(PERSIST);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(dest)==sha(src), '로컬/Drive 충돌: '+str(dest)
    else:
        shutil.copyfile(src,dest);assert sha(src)==sha(dest)
assert shutil.disk_usage(ROOT).free>25_000_000_000, '로컬 여유 공간을 확보하세요.'
SEEDS=[0,1,2] # 일부 seed 실행 후 같은 폴더에 나머지를 이어갈 수 있습니다.
def run(action):
    subprocess.run([sys.executable,'-m','interp_v1_4.p5',action,'--root',str(ROOT),'--output',str(OUT),
                    '--persistent',str(PERSIST),'--seeds',*map(str,SEEDS)],check=True)
print('재개 입력 복사 완료')''')
    md('## GPU 추출\n실행기가 현재 코드·환경의 debug smoke를 먼저 확인합니다. seed·split·checkpoint·READ 위치 hash가 같은 완전 저장 조각만 재사용합니다. GPU 메모리 부족 시 microbatch만 16→8→4→2→1로 줄입니다. quota·층·폭은 유지합니다.')
    code("run('extract')")
    md('## CPU probe\n이 셀은 CPU에서 실행합니다. 완료 JSON은 재사용하고 train/validation만으로 전처리·ANOVA·lambda·threshold를 선택합니다. test 점수는 보고에만 씁니다. 현재 GPU 런타임에서 CPU 작업을 이어가거나, cache를 보존한 뒤 CPU 런타임에서 앞 셀들의 입력/Drive 준비 후 이 셀부터 재개할 수 있습니다. GPU 추출 셀은 이미 끝난 경우 건너뜁니다. cache만 먼저 반환하려면 이 셀을 건너뛰고 다음 셀을 실행하세요.')
    code("run('probes')")
    md('## 결과 보존·다운로드\ncache와 probe를 함께 보존합니다. ZIP 생성 시 추가 로컬 공간이 필요합니다. 반환 검증에서 누락·실패·support 부족을 구분하므로 파일 생성만으로 P5를 완료 처리하지 않습니다. 대용량 ZIP 다운로드가 실패하면 Drive에 저장된 ZIP과 SHA256 파일을 내려받아 전달하세요.')
    code('''from datetime import datetime, timezone
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
archive=Path('/content')/f'v1_4_p5_evidence_{stamp}.zip'
subprocess.run([sys.executable,'-m','interp_v1_4.p5','export','--output',str(OUT),'--archive',str(archive)],check=True)
saved=PERSIST.parent/archive.name
shutil.copyfile(archive,saved);assert sha(archive)==sha(saved)
shutil.copyfile(str(archive)+'.sha256',str(saved)+'.sha256')
print('Drive 결과:',saved)
files.download(str(archive)+'.sha256')
files.download(str(archive))''')
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),colab=dict(name='P5_v1_4_READ_r2.ipynb'),accelerator='GPU'),cells=cells)


def pack():
    config=read(ROOT/CONTRACT)
    for p,d in config['files'].items():assert sha(ROOT/p)==d,p
    smoke=read(PREP/'cpu/preflight.json');assert smoke['status']=='passed' and smoke['contract_sha256']==sha(ROOT/CONTRACT)
    import xml.etree.ElementTree as ET
    cases=ET.parse(PREP/'tests.xml').getroot().findall('.//testcase')
    assert cases and not any(c.find('failure') is not None or c.find('error') is not None or c.find('skipped') is not None for c in cases)
    paths=[ROOT/p for p in config['files']]+[ROOT/CONTRACT,PREP/'input_verification.json',PREP/'cpu/preflight.json',PREP/'tests.xml']
    inventory={str(p.relative_to(ROOT)):sha(p) for p in paths}
    bundle=ROOT/'experiment_v1_4/bundles/v1_4_p5_bundle_r2.zip'
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in paths:z.write(p,str(p.relative_to(ROOT)))
        z.writestr('bundle_manifest.json',json.dumps(dict(files=inventory),indent=2))
    with zipfile.ZipFile(bundle) as z:
        assert len(z.namelist())==len(inventory)+1
        for n,d in inventory.items():assert hashlib.sha256(z.read(n)).hexdigest()==d
    digest=sha(bundle);bundle.with_suffix('.zip.sha256').write_text(digest+'  '+bundle.name+'\n')
    nb=notebook(bundle.name,digest)
    for c in nb['cells']:
        if c['cell_type']=='code':compile(c['source'],'P5 notebook','exec')
    dest=ROOT/'experiment_v1_4/notebooks/P5_v1_4_READ_r2.ipynb';write(dest,nb)
    write(PREP/'verification.json',dict(status='passed_local_preparation',contract_sha256=sha(ROOT/CONTRACT),
        bundle=str(bundle.relative_to(ROOT)),bundle_sha256=digest,bundle_bytes=bundle.stat().st_size,
        verified_members=len(inventory),notebook=str(dest.relative_to(ROOT)),notebook_sha256=sha(dest),
        tests=len(cases),gpu_execution='not_run',production_probe='not_run',p5_complete=False))
    print(read(PREP/'verification.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pack']);a=p.parse_args()
    prepare() if a.action=='prepare' else pack()
