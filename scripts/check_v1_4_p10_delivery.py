"""Local delivery checks; never runs a production LM evaluation or GPU training."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha,seed
from interp_v1_4 import p10


def main():
    start=time.monotonic();out=ROOT/'experiment_v1_4/results/p10_preparation_r1';c=p10.verify(ROOT)
    assert len(c['runs'])==12
    assert {(r['lm_seed'],r['layer'],r['tool'],r['k'],r['sparse_seed']) for r in c['runs']}=={(s,3,t,k,0) for s in (0,1,2) for t in ('sae','transcoder') for k in (4,16)}
    for r in c['runs']:
        assert r['draw_seed']==seed('p10_position_draw',r['draw_key'])
        assert r['init_seed']==seed('p10_dictionary_init',r['run_key'])
        other=next(x for x in c['runs'] if x['lm_seed']==r['lm_seed'] and x['k']==r['k'] and x['tool']!=r['tool'])
        assert other['draw_seed']==r['draw_seed'] and other['init_seed']!=r['init_seed']
    rules=read(ROOT/c['evaluation_rules'])
    for item in rules['baselines'].values():
        for rep,v in item['subsets'].items():
            assert len(v['ids'])==128 and len(set(v['ids']))==128
            width=256 if rep=='coordinate' else 512
            assert v['ids']==sorted(np.random.Generator(np.random.PCG64(v['seed'])).choice(width,128,replace=False).tolist())
    nb=ROOT/'experiment_v1_4/notebooks/P10/P10_colab_update_training_r1.ipynb'
    nbformat.validate(nbformat.read(nb,as_version=4))
    cells=read(nb)['cells']
    for cell in cells:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'notebook','exec')
    bundle=ROOT/'experiment_v1_4/bundles/P10/v1_4_p10_update_training_r1.zip'
    assert sha(bundle)==read(bundle.with_suffix('.sha256.json'))['sha256']
    expected={**c['files'],p10.CONTRACT:sha(ROOT/p10.CONTRACT)}
    with tempfile.TemporaryDirectory(prefix='p10-standalone-') as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(bundle) as z:
            assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(expected)
            for n,h in expected.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
            z.extractall(tmp)
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        proc=subprocess.run([sys.executable,'-m','pytest','tests_v1_4/test_p10.py','-q'],cwd=tmp,env=env,text=True,capture_output=True)
        (out/'standalone_tests.txt').write_text(proc.stdout+proc.stderr)
        assert proc.returncode==0,proc.stdout+proc.stderr
        proc=subprocess.run([sys.executable,'-c','from interp_v1_4.p10 import verify; print(len(verify(".")["runs"]))'],cwd=tmp,env=env,text=True,capture_output=True)
        assert proc.returncode==0 and proc.stdout.strip()=='12',proc.stderr
    smoke=read(ROOT/'experiment_v1_4/smoke/p10_cpu_r1/smoke.json')
    assert smoke['status']=='passed' and smoke['config_sha256']==sha(ROOT/p10.CONTRACT)
    pf=read(out/'preflight.json');assert pf['config_sha256']==sha(ROOT/p10.CONTRACT)
    evidence=[ROOT/p10.CONTRACT,ROOT/c['evaluation_rules'],nb,bundle,bundle.with_suffix('.sha256.json'),out/'preflight.json',out/'standalone_tests.txt',ROOT/'experiment_v1_4/smoke/p10_cpu_r1/smoke.json',Path(__file__)]
    result=dict(status='passed_preparation_only',p10_complete=False,production_training_runs=0,production_evaluated_runs=0,
                cuda_smoke_status='pending',required_training_runs=12,bundle_members_verified=len(expected),notebook_code_cells=sum(x['cell_type']=='code' for x in cells),standalone_tests_passed=8,
                input_quotas_verified={s:x['positions'] for s,x in pf['splits'].items()},
                train_val_insufficient_support=[s+'/'+k for s in ('train','val') for k,v in pf['splits'][s]['support'].items() if not v['fit_support_sufficient']],
                environment_id=smoke['environment_id'],elapsed_seconds=time.monotonic()-start,
                evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence},
                commands=['/opt/anaconda3/bin/python scripts/build_v1_4_p10.py','/opt/anaconda3/bin/python -m interp_v1_4.p10 preflight --root . --output experiment_v1_4/results/p10_preparation_r1','/opt/anaconda3/bin/python -m interp_v1_4.p10 smoke --root . --output experiment_v1_4/smoke/p10_cpu_r1 --device cpu','/opt/anaconda3/bin/python scripts/check_v1_4_p10_delivery.py'],
                limitations=['No production GPU execution performed locally','24.59 CU is user-reported; no verified CU/hour conversion','Update intervention, additional layers, length, m-to-m SAE skipped in this scope','Semantic/fidelity/transfer evaluation delivery follows independent training return audit'])
    write(out/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
