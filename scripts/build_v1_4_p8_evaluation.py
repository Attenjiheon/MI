"""Bind audited TC checkpoints to the pre-training evaluation rules."""
from pathlib import Path
import copy,json,hashlib,zipfile,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write,derived,LABELS
from interp_v1_4.runtime import sha
from interp_v1_4.p8_evaluation import CONTRACT,SELECTED,RECEIPT,records

def freeze():
    base=ROOT/'experiment_v1_4/results/p8_training_audit_20260929_01'
    audit=read(base/'verification.json');assert audit['p8_training_complete'] and audit['p8_evaluation_eligible'] and not audit['p8_complete']
    selected=read(ROOT/SELECTED)['models'];assert len(selected)==24
    rules_path=ROOT/'experiment_v1_4/p8_r1/evaluation_rules.json';rules=read(rules_path)
    original=read(ROOT/'experiment_v1_4/p8_r1/contract.json');assert sha(rules_path)==original['files'][str(rules_path.relative_to(ROOT))]
    p7=read(ROOT/'experiment_v1_4/p7_r1/contract.json');models=p7['models']
    paths={ROOT/SELECTED,ROOT/RECEIPT,base/'verification.json',base/'verify_return.py',base/'runs.csv',rules_path,ROOT/'experiment_v1_4/p8_r1/contract.json',ROOT/'experiment_v1_4/p5_r2/contract.json',ROOT/'experiment_v1_4/analysis_plan.json'}
    for e in selected:
        p=ROOT/e['checkpoint'];assert sha(p)==e['sha256'];paths.add(p)
    for m in models:
        p=ROOT/m['checkpoint'];assert sha(p)==m['checkpoint_sha256'];paths.add(p)
    for name in ('01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md','scripts/build_v1_4_p8_evaluation.py','scripts/verify_v1_4_p8_evaluation.py','tests_v1_4/test_p8_evaluation.py','tests_v1_4/test_p7.py'):
        paths.add(ROOT/name)
    for folder in ('interp_v1_4','archive/legacy/interp_v1_2','corpus'):paths.update((ROOT/folder).glob('*.py'))
    data_root=p7['data_root'];data_manifest=read(ROOT/data_root/'manifest.json')
    for split in ('train','val','test'):
        for suffix in ('jsonl.gz','read_positions.json'):
            p=ROOT/data_root/'interpretation'/f'{split}.{suffix}'
            assert sha(p)==data_manifest['files'][str(p.relative_to(ROOT/data_root))]['sha256'];paths.add(p)
    for name,n in rules['causal_quotas'].items():
        p=ROOT/data_root/'causal_pairs'/f'{name}.jsonl.gz';assert sha(p)==data_manifest['files'][str(p.relative_to(ROOT/data_root))]['sha256']
        assert len(records(p))==n;paths.add(p)
    general=ROOT/data_root/'test/general.jsonl.gz';assert sha(general)==data_manifest['files']['test/general.jsonl.gz']['sha256'];paths.add(general)
    paths.add(ROOT/rules['reconstruction_targets'])
    source=ROOT/'experiment_v1_4/results/p5_metadata_audit_20260924_01/returned/probes';checked=ROOT/'experiment_v1_4/results/p5_final_audit_20260925_01/returned/tasks'
    reuse={}
    for model in models:
        seed=model['lm_seed']
        for layer in (0,3,7,11):
            reps=[('u','full','fullu'),('m','full','fullm')]
            if layer==0:reps.extend(('m',r,r) for r in ('coordinate','coordinate_128','random','random_128'))
            for hook,rep,alias in reps:
                for label in LABELS:
                    for domain in (('iid','transfer') if label in ('current','previous') else ('iid',)):
                        n=f'seed{seed}_trained_l{layer}_{hook}_{rep}_{label}_{domain}.json';p=source/n;a=read(checked/n)
                        assert a['status']=='passed' and sha(p)==a['source_sha256'];paths.update([p,checked/n])
                        reuse[f'seed{seed}_l{layer}_{alias}_{label}_{domain}']=str(p.relative_to(ROOT))
    rng={}
    for e in selected:
        run=e['run'];values={}
        for name in rules['causal_quotas']:
            for pair in records(ROOT/data_root/'causal_pairs'/f'{name}.jsonl.gz'):
                for size in ('single','up_to_four'):
                    key=pair['pair_id']+'|'+size
                    values[key]=dict(latent=derived('random_patch',run['run_key']+'|'+key),direction=derived('random_patch',run['run_key']+'|directions|'+key))
        rng[run['name']]=values
    rngpath=ROOT/'experiment_v1_4/p8_eval_r1/patch_rng.json';write(rngpath,rng);paths.add(rngpath)
    comparison={}
    original_manifest=read(ROOT/'experiment_v1_4/results/p7_return_audit_20260927_01/returned/return_manifest.json')
    for name in ('semantic','fidelity','replacement','causal'):
        rel=f'tables/{name}.csv';p=ROOT/'experiment_v1_4/results/p7_return_audit_20260927_01/returned'/rel
        assert sha(p)==original_manifest['files'][rel];paths.add(p);comparison[name]=str(p.relative_to(ROOT))
    config=copy.deepcopy(rules)
    config.update(schema='p8-tc-evaluation-v1.4-r1',tcs=selected,models=models,data_root=data_root,reused_probes=reuse,patch_rng=str(rngpath.relative_to(ROOT)),sae_comparison_tables=comparison,
        training_audit_sha256=sha(base/'verification.json'),frozen_rules_sha256=sha(rules_path),execution_status='prepared; no TC test results observed',
        completion='All 24 TC evaluations and returned independent audit required; no automatic P8 completion',files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
    write(ROOT/CONTRACT,config);return config,paths|{ROOT/CONTRACT}

def delivery():
    config,paths=freeze();bundle=ROOT/'experiment_v1_4/bundles/P8/v1_4_p8_evaluation_r1.zip';bundle.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        for n,d in {**config['files'],CONTRACT:sha(ROOT/CONTRACT)}.items():assert hashlib.sha256(z.read(n)).hexdigest()==d,n
    digest=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
    # Preserve the proven stage layout, while switching every entry point and input identity.
    nb=read(ROOT/'experiment_v1_4/notebooks/P7/P7_colab_read_sae_evaluation_r1.ipynb')
    for cell in nb['cells']:
        s=''.join(cell['source'])
        s=s.replace('P7','P8').replace('p7_runner','p8_eval_runner').replace('from interp_v1_4.p7 import','from interp_v1_4.p8_evaluation import')
        s=s.replace('verify_v1_4_p7_return.py','verify_v1_4_p8_evaluation.py').replace('test_p7.py','test_p8_evaluation.py')
        s=s.replace('SAE','TC').replace('P8_r1','P8_evaluation_r1').replace('MI_P8_r1','MI_P8_evaluation_r1')
        s=s.replace('v1_4_p7_bundle_r1.zip','v1_4_p8_evaluation_r1.zip').replace('v1_4_p7_return_','v1_4_p8_evaluation_return_')
        if cell['cell_type']=='code' and 'uploaded=files.upload()' in s:
            start=s.index('uploaded=files.upload()');end=s.index("ROOT=Path",start)
            s=s[:start]+f'''drive.mount('/content/drive')
BUNDLE_IN_DRIVE=Path('/content/drive/MyDrive/boolean_interp_v1_4/v1_4_p8_evaluation_r1.zip')
bundle=Path('/content/v1_4_p8_evaluation_r1.zip')
import shutil
def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
if not bundle.exists() or checksum(bundle)!='{digest}':shutil.copyfile(BUNDLE_IN_DRIVE,bundle)
assert checksum(bundle)=='{digest}', '입력 ZIP checksum 불일치'
'''+s[end:]
            # Only mount once.
            first=s.index("drive.mount('/content/drive')")
            tail=s[first+len("drive.mount('/content/drive')"):].replace("drive.mount('/content/drive')\n",'')
            s=s[:first+len("drive.mount('/content/drive')")]+tail
        if cell['cell_type']=='markdown':
            s=s.replace('네 층 READ TC 평가·개입','네 층 READ TC 평가·개입 (학습 반환 감사 통과)')
            s=s.replace('P5 full probe와 block 0 대조군','P5 full u/m probe와 block 0 m 좌표/random 대조군')
            s=s.replace('full probe와 block 0 대조군','full u/m probe와 block 0 m 좌표/random 대조군')
            if s.startswith('# P8'):
                s+='\n\n입력 ZIP을 Drive `boolean_interp_v1_4/`에 업로드해 두세요. 24 TC는 이미 학습·반환 감사를 마쳤으며 이 노트북은 재학습하지 않습니다. 입력 u로 latent를 계산하고 출력 m을 s_m으로 대체·패칭하며 원본 residual skip을 유지합니다. Full u/full m과 m 좌표/random을 비교합니다. P7 h-SAE와 P8 m-TC의 서로 다른 target을 비교 그림에 명시합니다.'
        cell['source']=s.splitlines(True)
        if cell['cell_type']=='code':compile(s,'p8-eval','exec')
    notebook=ROOT/'experiment_v1_4/notebooks/P8/P8_colab_read_tc_evaluation_r1.ipynb';write(notebook,nb)
    print(json.dumps(dict(notebook=str(notebook),bundle=str(bundle),sha256=digest,bytes=bundle.stat().st_size),indent=2))
if __name__=='__main__':delivery()
