"""Freeze P9 seed-1 training plus inherited evaluation rules; build Colab delivery."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write,derived
from interp_v1_4.runtime import sha
from interp_v1_4.p9 import CONTRACT,RECEIPT,P6_METADATA


def freeze():
    paths=set()
    for phase,folder in [('p6','p6_final_audit_20260925'),('p7','p7_final_audit_20260928_01'),('p8','p8_final_audit_20260930_01')]:
        base=ROOT/'experiment_v1_4/results'/folder;c=read(base/'completion.json')
        assert c[phase+'_complete']
        for name,digest in c['evidence_sha256'].items():
            assert sha(base/name)==digest,(phase,name)
            paths.add(base/name)
        for name,digest in c.get('prior_evidence_sha256',{}).items():
            assert sha(ROOT/name)==digest,name
            paths.add(ROOT/name)
        paths.add(base/'completion.json')
    p6=read(ROOT/'experiment_v1_4/p6_r1/contract.json')
    p8=read(ROOT/'experiment_v1_4/p8_r1/contract.json')
    p7=read(ROOT/'experiment_v1_4/p7_r1/contract.json')
    rules8=read(ROOT/'experiment_v1_4/p8_r1/evaluation_rules.json')
    runs=[]
    for source in (p6,p8):
        for old in source['runs']:
            if old['lm_seed']!=0:continue
            r=copy.deepcopy(old);r['name']=r['name'].replace('_s0','_s1');r['sparse_seed']=1
            for key in ('run_key','draw_key'):r[key]=r[key].replace('sparse_seed=0','sparse_seed=1')
            r['init_seed']=derived('dictionary_init',r['run_key']);r['torch_init_seed']=r['init_seed']%(2**63-1)
            r['draw_seed']=derived('position_draw',r['draw_key']);runs.append(r)
    assert len(runs)==16
    shared=('causal_quotas','random_candidates','matched_limit','unmatched_rule','random_direction_rule','selection','selection_scope','operator_transfer','matching','bootstrap','reconstruction_targets','reconstruction_target_rng')
    rules=dict(schema='p9-evaluation-rules-v1.4-r1',shared={k:copy.deepcopy(p7[k]) for k in shared},
        baselines={tool:{k:copy.deepcopy(v) for k,v in obj['baselines'].items() if k.startswith('seed0_')} for tool,obj in [('sae',p7),('transcoder',rules8)]},
        bootstrap_keys={},patch_seeds={},checkpoint_selection=p6['training']['selection'],
        feature_selection='Refit separately per dictionary; never identify feature semantics by equal latent ID across seeds',
        candidate_policy='Reuse frozen layer/hook-specific 128 candidate subsets and coordinate/random directions; refit latent ranking and probes for seed 1',
        initialization_comparison='Report seed 0 vs 1 within LM seed 0 separately from LM seed variation; identical latent IDs do not imply identical features',
        evaluation_procedure='Same P7 SAE and P8 TC semantic/fidelity/replacement/causal procedures and schemas; both k, full/128 candidates, all controls and source reproduction audit',
        results_schema_sources=['experiment_v1_4/p7_r1/contract.json','experiment_v1_4/p8_eval_r1/contract.json'],
        test_used_to_set_rules=False,execution_status='pending training return audit and selected checkpoint binding')
    for r in runs:
        hook='h' if r['tool']=='sae' else 'm'
        rules['bootstrap_keys'][r['name']]=f"v1.4|lm_seed=0|checkpoint={r['checkpoint_sha256']}|hook={hook}|READ|tool={r['tool']}|k={r['k']}|sparse_seed=1"
        rules['patch_seeds'][r['name']]=dict(run_key=r['run_key'],rule='derived(random_patch, run_key + |pair_id|size); directions add |directions before pair_id; PCG64')
    rp=ROOT/'experiment_v1_4/p9_r1/evaluation_rules.json';write(rp,rules);paths.add(rp)
    paths.add(ROOT/rules['shared']['reconstruction_targets'])
    stats=read(ROOT/P6_METADATA/'input_manifest.json')
    for n,d in stats['statistics'].items():
        if 'seed0_' in n:
            p=ROOT/P6_METADATA/n;assert sha(p)==d;paths.add(p)
    for n in (RECEIPT,'experiment_v1_4/p6_r1/contract.json','experiment_v1_4/p7_r1/contract.json','experiment_v1_4/p8_r1/contract.json','experiment_v1_4/p8_r1/evaluation_rules.json','experiment_v1_4/p8_eval_r1/contract.json','experiment_v1_4/analysis_plan.json','archive/legacy/experiment_v1_2/debug/sequences.json','scripts/build_v1_4_p9.py','scripts/verify_v1_4_p9_training.py','scripts/p9_progress.py','tests_v1_4/test_p9.py','experiment_v1_4/notebooks/P8/P8_colab_read_tc_training_r2_progress.ipynb'):
        paths.add(ROOT/n)
    for folder in ('interp_v1_4','archive/legacy/interp_v1_2','corpus'):paths.update((ROOT/folder).glob('*.py'))
    paths.update(ROOT/n for n in ('01_experiment_design.md','02_language_and_corpus.md','03_experiment_spec.md'))
    config=dict(schema='p9-sparse-repeat-v1.4-r1',date='2026-09-30',runs=runs,training=p6['training'],
        budget=dict(runs=16,updates_per_run=5000,total_updates=80000,draws_per_run=2560000,total_draws=40960000),
        rng_namespace=p6['rng_namespace'],draw_key_policy=p6['draw_key_policy'],
        input_gate='P8 completed evidence hashes; seed0 trained P5 cache hashes and READ keys; 12 audited train statistics; current-source CUDA smoke and resume for both tools',
        evaluation_rules=str(rp.relative_to(ROOT)),delivery_scope='Training, numerical audit, frozen evaluation rules. Evaluation delivery follows audited selection.',
        completion='16 runs plus full evaluation, seed comparison, figures and returned original-cache reproduction audit',
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
    write(ROOT/CONTRACT,config);return config,paths|{ROOT/CONTRACT}


def delivery():
    config,paths=freeze()
    bundle=ROOT/'experiment_v1_4/bundles/P9/v1_4_p9_training_r1.zip';bundle.parent.mkdir(parents=True,exist_ok=True)
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as z:
        for n,d in {**config['files'],CONTRACT:sha(ROOT/CONTRACT)}.items():assert hashlib.sha256(z.read(n)).hexdigest()==d,n
    digest=sha(bundle);write(bundle.with_suffix('.sha256.json'),dict(sha256=digest,bytes=bundle.stat().st_size))
    nb=read(ROOT/'experiment_v1_4/notebooks/P8/P8_colab_read_tc_training_r2_progress.ipynb')
    old_digest=read(ROOT/'experiment_v1_4/bundles/P8/v1_4_p8_training_r2_progress.sha256.json')['sha256']
    for i,c in enumerate(nb['cells']):
        s=''.join(c['source']).replace(old_digest,digest).replace('P8','P9').replace('p8','p9')
        s=s.replace('v1_4_p9_training_r2_progress.zip','v1_4_p9_training_r1.zip').replace('24 runs','16 runs').replace('24개','16개').replace('36개','12개').replace('504개','336개').replace('480개','320개')
        s=s.replace('세 LM ×','LM seed 0 ×').replace('sparse seed 0','sparse seed 1').replace('TC 16','SAE/TC 16').replace('READ Transcoder 16','READ SAE/TC 16')
        s=s.replace('seed0_l0_tc_k4_s0','seed0_l0_sae_k4_s1')
        if c['cell_type']=='code' and 'overlay=json.loads' in s:
            a=s.index('overlay=json.loads');b=s.index('subprocess.run',a);s=s[:a]+s[b:]
        if c['cell_type']=='markdown':
            s=s.replace('각 TC는 같은 위치의 MLP 입력 u에서 MLP 출력 m을 예측합니다.','SAE는 h를 복원하고 TC는 같은 위치의 u에서 m을 예측합니다.')
            s=s.replace('240개 cache','seed 0 trained cache').replace('40개 cache','seed 0 trained cache')
            s=s.replace('u/m','h/u/m').replace('TC k=4/16','SAE·TC k=4/16').replace('선택 TC','선택 SAE/TC')
            if '**r2 진행 표시 추가:**' in s:s=s[:s.index('**r2 진행 표시 추가:**')]+'학습 셀에 전체/현재 run 진행률·속도·ETA를 표시하며 stdout/stderr는 Drive에 보존합니다.'
        c['source']=s.splitlines(True);c['id']=f'p9-{i}'
        if c['cell_type']=='code':compile(s,'p9-notebook','exec')
    nb['cells'][0]['source']=['# P9 — sparse seed 1 SAE/TC 16개 학습·검산\n\n',
        '입력 ZIP `v1_4_p9_training_r1.zip`을 내 드라이브/boolean_interp_v1_4/에 업로드한 뒤 GPU 런타임에서 위부터 실행하세요. 원본 cache는 P5_r2, 새 출력은 P9_training_r1입니다.\n\n',
        'LM seed 0 × block 0/3/7/11 × SAE/TC × k=4/16, sparse seed 1입니다. 각 run 5,000 updates, 총 80,000 updates/40.96M draws를 실행합니다. SAE는 h→h, TC는 u→m이며 같은 층·k의 두 도구가 같은 position draws를 씁니다.\n\n',
        '중단 후 같은 노트북을 재실행하면 마지막 checksum checkpoint의 optimizer·RNG·cursor에서 재개합니다. 환경이 달라지면 새 환경 ID와 CUDA smoke를 기록합니다. 완료 또는 중단 후 마지막 셀로 결과 ZIP과 checksum JSON을 내려받아 반환하세요.\n\n',
        '**학습 전용 전달물입니다.** 336 checkpoint·320 validation 수치 검산과 반환 감사 후 선택 checkpoint를 연결한 전체 평가를 진행합니다. 학습만으로 P9를 완료하지 않습니다. 진행률·ETA·실행 로그를 저장합니다.\n']
    out=ROOT/'experiment_v1_4/notebooks/P9/P9_colab_sparse_seed1_training_r1.ipynb';write(out,nb)
    print(json.dumps(dict(notebook=str(out),bundle=str(bundle),sha256=digest,bytes=bundle.stat().st_size),indent=2))

if __name__=='__main__':delivery()
