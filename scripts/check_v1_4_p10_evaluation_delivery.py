"""Standalone bundle and actual selected-model CPU fixture checks; no real test inference."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,time,zipfile
import numpy as np
import torch
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4 import p10_evaluation as ev
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha


def main():
    out=ROOT/'experiment_v1_4/results/p10_evaluation_preparation_r1';out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();c=ev.verify(ROOT);ch=sha(ROOT/ev.CONTRACT);torch.set_num_threads(2)
    bundle=ROOT/'experiment_v1_4/bundles/P10/v1_4_p10_update_evaluation_r1.zip'
    assert sha(bundle)==read(bundle.with_suffix('.sha256.json'))['sha256']
    with tempfile.TemporaryDirectory(prefix='p10eval-standalone-') as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(bundle) as z:
            expected={**c['files'],ev.CONTRACT:ch};assert set(z.namelist())==set(expected) and len(z.namelist())==len(expected)
            for n,h in expected.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
            z.extractall(tmp)
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        proc=subprocess.run([sys.executable,'-m','pytest','tests_v1_4/test_p10_evaluation.py','-q'],cwd=tmp,env=env,text=True,capture_output=True)
        (out/'standalone_tests.txt').write_text(proc.stdout+proc.stderr);assert proc.returncode==0,proc.stdout+proc.stderr
        proc=subprocess.run([sys.executable,'-c','from interp_v1_4.p10_evaluation import verify; print(len(verify(".")["tasks"]))'],cwd=tmp,env=env,text=True,capture_output=True)
        assert proc.returncode==0 and proc.stdout.strip()=='1248',proc.stderr
    notebooks=[]
    for name in ('P10_evaluation_1_T4_features_r1.ipynb','P10_evaluation_2_CPU_probes_r1.ipynb'):
        p=ROOT/'experiment_v1_4/notebooks/P10'/name;nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':compile(cell.source,name,'exec')
        notebooks.append(p)
    # Synthetic 17-row activation fixtures exercise every actual selected checkpoint.
    # These are not extracted LM activations and never become production results.
    with tempfile.TemporaryDirectory(prefix='p10eval-encode-fixture-') as tmp:
        tmp=Path(tmp);rng=np.random.Generator(np.random.PCG64(10203))
        for lm in (0,1,2):
            for kind in ('trained','init'):
                for split in ev.SPLITS:
                    for hook in 'hum':
                        st=c['statistics'][f'seed{lm}_l3_{hook}.json']
                        x=(np.asarray(st['mean'])+st['scale']*rng.normal(size=(17,256))).astype(np.float32)
                        ev.commit_array(ev.raw_path(tmp,lm,kind,split,hook),x,dict(debug_only=True,config=ch))
        paths=list((tmp/'arrays').glob('*'));write(tmp/'cache_complete.json',dict(config_sha256=ch,files={str(p.relative_to(tmp)):sha(p) for p in paths}))
        ev.encode(ROOT,tmp,'cpu');checks=[]
        for e in c['selected']:
            r=e['run'];s=torch.load(ROOT/e['checkpoint'],map_location='cpu',weights_only=False);ih,th=('h','h') if r['tool']=='sae' else ('u','m')
            report=read(tmp/'fidelity'/f'{r["name"]}.json')
            for split in ev.SPLITS:
                raw=torch.tensor(np.load(ev.raw_path(tmp,r['lm_seed'],'trained',split,ih)))
                x=(raw-s['stats'][ih]['mu'])/s['stats'][ih]['scale'];m=s['model']
                a=torch.relu(torch.nn.functional.linear(x,m['encoder'],m['encoder_bias']))
                ids=torch.argsort(a,dim=1,descending=True,stable=True)[:,:r['k']]
                latent=torch.zeros_like(a).scatter(1,ids,a.gather(1,ids))
                pred=torch.nn.functional.linear(latent,m['decoder'],m['decoder_bias']).numpy()
                np.testing.assert_array_equal(np.load(tmp/'latents'/r['name']/f'{split}.npy'),latent.numpy())
                raw_target=np.load(ev.raw_path(tmp,r['lm_seed'],'trained',split,th)).astype(np.float64)
                target=(raw_target-s['stats'][th]['mu'].numpy())/float(s['stats'][th]['scale'])
                mse=float(np.mean((target-pred)**2));assert np.isclose(mse,report['fidelity'][split]['mse'],atol=1e-12,rtol=1e-12)
            checks.append(dict(run=r['name'],status='passed',synthetic_positions_per_split=17))
        write(out/'selected_checkpoint_cpu_fixture.json',dict(debug_only=True,checks=checks))
    paths=[ROOT/ev.CONTRACT,bundle,bundle.with_suffix('.sha256.json'),out/'standalone_tests.txt',out/'selected_checkpoint_cpu_fixture.json',Path(__file__)]+notebooks
    report=dict(status='passed_preparation_only',p10_complete=False,evaluation_runs_executed=0,cuda_evaluation_smoke='pending',
        tasks=1248,semantic_reports_planned=2016,standalone_tests_passed=5,selected_models_fixture_checked=12,bundle_members_verified=len(expected),
        elapsed_seconds=time.monotonic()-start,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        commands=['/opt/anaconda3/bin/python scripts/build_v1_4_p10_evaluation.py','/opt/anaconda3/bin/python scripts/check_v1_4_p10_evaluation_delivery.py'],
        limits=['CPU selected-model test used synthetic activations only','GPU extraction/encoding and actual semantic/fidelity/transfer evaluation pending','Original-cache source reproduction and final return audit remain after evaluation'])
    write(out/'verification.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
