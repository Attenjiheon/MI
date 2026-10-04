"""Bind audited Update checkpoints; freeze tasks; deliver GPU and CPU notebooks."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha,seed
from interp_v1_4 import p10
from interp_v1_4.p10_evaluation import CONTRACT


def tasks(rules,selected):
    out=[]
    for lm in (0,1,2):
        reps=[dict(representation=rep,hook=h) for rep in ('full','init','shuffled') for h in 'hum']+[dict(representation='token',hook='h')]
        reps += [dict(representation=rep,hook=h) for rep in ('coordinate','random') for h in 'hm']
        reps += [dict(representation='latent',hook='h' if e['run']['tool']=='sae' else 'm',run=e['run']['name'],tool=e['run']['tool']) for e in selected if e['run']['lm_seed']==lm]
        for r in reps:
            base=rules['baselines'][f'seed{lm}_{r["hook"]}'];rep=r['representation'];prefix=rep in ('coordinate','random','latent')
            for limit in (('all','128') if prefix else ('all',)):
                for label in p10.LABELS:
                    for domain in (('all','variable','operator') if label in rules['transfer']['labels'] else ('all',)):
                        tag=r.get('run',f'seed{lm}_{rep}_{r["hook"]}')
                        tid=f'{tag}__{limit}__{label}__{domain}'
                        candidates=None if limit=='all' else base['subsets'][r['tool'] if rep=='latent' else rep]['ids']
                        references=[]
                        if prefix:
                            hooks='um' if rep=='latent' and r['tool']=='transcoder' else r['hook']
                            references=[f'seed{lm}_full_{h}__all__{label}__{domain}' for h in hooks]
                        out.append(dict(id=tid,lm_seed=lm,**r,label=label,domain=domain,prefixes=prefix,candidates=candidates,
                            candidate_limit=limit,shuffle_seed=base['shuffled_label_seed'],
                            bootstrap_seed=seed('p10_bootstrap',base['key']+'|'+label+'|'+domain),full_references=references))
    assert len(out)==1248 and len({t['id'] for t in out})==len(out)
    return out


def freeze():
    old=p10.verify(ROOT);folder=ROOT/'experiment_v1_4/results/p10_training_audit_20261003_01';complete=read(folder/'completion.json')
    assert complete['p10_training_complete'] and not complete['p10_complete']
    for n,h in complete['evidence_sha256'].items():assert sha(ROOT/n)==h,n
    selected=read(folder/'selected_dictionary_manifest.json');rules=read(ROOT/old['evaluation_rules'])
    stats={p.name:read(p) for p in (folder/'returned_metadata/statistics').glob('*.json')};assert len(stats)==9
    paths={ROOT/n for n in old['files']}|{ROOT/p10.CONTRACT,folder/'completion.json',folder/'selected_dictionary_manifest.json'}
    paths.update(ROOT/e['checkpoint'] for e in selected['models'])
    paths.update((folder/'returned_metadata/statistics').glob('*.json'))
    paths.update((ROOT/'interp_v1_4').glob('*.py'))
    paths.update(ROOT/n for n in ('scripts/build_v1_4_p10_evaluation.py','tests_v1_4/test_p10_evaluation.py','scripts/audit_v1_4_p10_training_return.py'))
    c=dict(schema='p10-update-evaluation-v1.4-r1',date='2026-10-03',rules=rules,selected=selected['models'],statistics=stats,
        training_input_sha256=selected['input_sha256'],tasks=tasks(rules,selected['models']),
        gpu_stage='Reuse verified trained train/val; extract trained test and init train/val/test; frozen dictionary encode and fidelity. No fitting.',
        cpu_stage='Source-domain train/val fits; freeze every fit before test reporting; cluster bootstrap 1000 and paired full-probe BA/F1 differences',
        resource_policy='T4 only for extraction/encoding; CPU runtime for 1248 fitting tasks and 2016 semantic reports; no additional dictionary training',
        fidelity_numerics='Inherited float32 dictionary forward, float64 normalized targets/metric accumulation; train mean denominator; separate h/m targets',
        completion='Returned selection/prediction/fidelity/CI audit then original-cache source reproduction; evaluation notebook is not phase completion',
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
    write(ROOT/CONTRACT,c);return c,paths|{ROOT/CONTRACT}


def make_notebook(digest,gpu):
    cells=[]
    def add(kind,s):
        cell=dict(cell_type=kind,metadata={},source=s.strip().splitlines(True),id=f'p10eval-{len(cells)}')
        if kind=='code':compile(s.strip(),'notebook','exec');cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown',('# P10 평가 1/2 — T4 cache·latent 준비' if gpu else '# P10 평가 2/2 — CPU probe·전이·통계')+'''

**학습은 다시 실행하지 않습니다. P10은 반환·원본 재현 감사 전까지 미완료입니다.**
입력 ZIP `v1_4_p10_update_evaluation_r1.zip`을 내 드라이브 `boolean_interp_v1_4/`에 올리세요. 기존 학습 출력 `P10_update_l3_r1`을 유지합니다. 평가 출력은 `P10_update_evaluation_r1`입니다.

'''+('T4에서 이 노트북을 실행해 cache·latent를 준비합니다. 완료되면 GPU 연결을 종료하고 두 번째 노트북을 **GPU 없는 CPU 런타임**에서 실행하세요. 시간 제한으로 PAUSED이면 같은 셀을 반복합니다.' if gpu else '첫 번째 노트북의 features_complete.json이 있어야 합니다. 런타임 유형을 **하드웨어 가속기 없음**으로 선택하세요. 1,248개 fitting 과제·2,016개 semantic 보고를 순차 저장합니다. 계산량이 많으므로 여러 세션으로 나누어도 됩니다. 고정 예산·support·lambda·bootstrap 수를 줄이지 않습니다.'))
    add('code',f'''
from google.colab import drive,files
from pathlib import Path
import hashlib,json,zipfile,sys,subprocess,os,time,shutil
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/boolean_interp_v1_4')
BUNDLE=BASE/'v1_4_p10_update_evaluation_r1.zip'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert digest(BUNDLE)=='{digest}', 'ZIP checksum 불일치'
ROOT=Path('/content/MI_P10_evaluation_r1');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    for n in z.namelist():assert (ROOT/n).resolve().is_relative_to(ROOT.resolve())
    z.extractall(ROOT)
SOURCE=BASE/'P10_update_l3_r1'
OUTPUT=BASE/'P10_update_evaluation_r1';OUTPUT.mkdir(exist_ok=True)
''')
    add('code','''
log=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','pytest==8.4.2','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(OUTPUT/f'install_{time.time_ns()}.txt').write_text(log.stdout);print(log.stdout);log.check_returncode()
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(ROOT))
from interp_v1_4.p10_evaluation import verify,export
verify(ROOT)
def run(args):
    path=OUTPUT/'logs'/f'{time.time_ns()}.txt';path.parent.mkdir(exist_ok=True)
    with path.open('w') as log:
        p=subprocess.Popen([sys.executable,'-u']+args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in p.stdout:print(line,end='');log.write(line);log.flush()
            rc=p.wait()
        except BaseException:p.terminate();p.wait();raise
    assert rc==0, f'실행 실패: {path}; 마지막 셀에서 증빙을 반환하세요'
run(['-m','pytest','tests_v1_4/test_p10_evaluation.py','-q'])
MAX_SECONDS=7200  # 호출별 chunk/task 경계에서 정지; 중단 후 같은 셀 재실행
''')
    if gpu:
        add('code',"run(['-m','interp_v1_4.p10_evaluation','extract','--root',str(ROOT),'--source',str(SOURCE),'--output',str(OUTPUT),'--max-seconds',str(MAX_SECONDS)])")
        add('code',"run(['-m','interp_v1_4.p10_evaluation','encode','--root',str(ROOT),'--output',str(OUTPUT),'--device','cuda'])\nassert (OUTPUT/'features_complete.json').exists()\nprint('GPU 준비 완료. GPU 연결을 종료하고 2/2 CPU 노트북으로 이동하세요.')")
    else:
        add('code',"assert (OUTPUT/'features_complete.json').exists(), '1/2 T4 노트북부터 완료하세요'\nrun(['-m','interp_v1_4.p10_evaluation','select','--root',str(ROOT),'--output',str(OUTPUT),'--max-seconds',str(MAX_SECONDS)])")
        add('markdown','''모든 train/validation 선택을 저장한 뒤에만 test 평가를 시작합니다. 선택 미완료이면 위 셀을 반복하세요. 과제별 source-domain 표준화·ranking·lambda·threshold를 다시 fit하며, Update dictionary 자체는 전체 train으로 학습했습니다. 따라서 전이는 **probe의 감독 전이**입니다.''')
        add('code',"run(['-m','interp_v1_4.p10_evaluation','evaluate','--root',str(ROOT),'--output',str(OUTPUT),'--max-seconds',str(MAX_SECONDS)])")
    add('markdown','''## 결과 반환·중단 증빙 보존
완료 또는 실패/중단 시 실행합니다. cache/arrays/latents는 Drive에 유지하고 나머지 환경·로그·fitting trace·예측·의미/fidelity 통계를 ZIP으로 반환합니다. CPU 평가 중 PAUSED이면 해당 셀을 반복하세요. 실제 GPU/CPU 환경·소요 시간과 모든 수치가 반환 감사 대상입니다. 최종적으로 ZIP과 checksum JSON을 보내주세요.''')
    add('code','''
archive=Path(f'/content/v1_4_p10_evaluation_return_{time.time_ns()}.zip')
export(OUTPUT,archive)
back=OUTPUT/'exports';back.mkdir(exist_ok=True)
shutil.copyfile(archive,back/archive.name);shutil.copyfile(archive.with_suffix('.sha256.json'),back/archive.with_suffix('.sha256.json').name)
print('checksum',digest(archive));files.download(str(archive));files.download(str(archive.with_suffix('.sha256.json')))
''')
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)


def main():
    c,paths=freeze();bundle=ROOT/'experiment_v1_4/bundles/P10/v1_4_p10_update_evaluation_r1.zip'
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        expected={**c['files'],CONTRACT:sha(ROOT/CONTRACT)};assert set(z.namelist())==set(expected)
        for n,h in expected.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    digest=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
    for gpu,name in [(True,'P10_evaluation_1_T4_features_r1.ipynb'),(False,'P10_evaluation_2_CPU_probes_r1.ipynb')]:
        write(ROOT/'experiment_v1_4/notebooks/P10'/name,make_notebook(digest,gpu))
    print(json.dumps(dict(bundle=str(bundle),bytes=bundle.stat().st_size,sha256=digest,tasks=len(c['tasks']),members=len(expected)),indent=2))

if __name__=='__main__':main()
