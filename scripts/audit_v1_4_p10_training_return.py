"""Stream P10 training return; preserve originals and bind audited selection."""
from pathlib import Path
import hashlib,io,json,sys,time,zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4.p10 import verify,CONTRACT,load_split
from interp_v1_4.runtime import sha,tensor_digest
ARCHIVE=ROOT/'experiment_v1_4/evidence/P10/v1_4_p10_training_return_1791021261419206673.zip'
OUT=ROOT/'experiment_v1_4/results/p10_training_audit_20261003_01'


def main():
    start=time.monotonic();c=verify(ROOT);torch.set_num_threads(2)
    assert sha(ARCHIVE)==read(ARCHIVE.with_suffix('.sha256.json'))['sha256']
    with zipfile.ZipFile(ARCHIVE) as z:
        names=z.namelist();assert len(names)==len(set(names))
        manifest=json.loads(z.read('export_manifest.json'))
        assert set(names)==set(manifest['files'])|{'export_manifest.json'}
        for n,h in manifest['files'].items():
            assert not Path(n).is_absolute() and '..' not in Path(n).parts
            assert hashlib.sha256(z.read(n)).hexdigest()==h,n
        def j(n):return json.loads(z.read(n))
        assert not any(n.startswith('failures/') for n in names)
        assert hashlib.sha256(z.read('contract.json')).hexdigest()==sha(ROOT/CONTRACT)
        inp=j('input_manifest.json');ih=hashlib.sha256(z.read('input_manifest.json')).hexdigest();ch=sha(ROOT/CONTRACT)
        identity=dict(config_sha256=ch,input_sha256=ih)
        assert inp['config_sha256']==ch and inp['statistics_count']==9 and not inp['test_used_for_selection']
        assert len(inp['statistics'])==9 and len(inp['tensors'])==18
        for n,h in {**inp['statistics'],**inp['labels']}.items():assert manifest['files'][n]==h
        for split in ('train','val'):
            _,rows=load_split(ROOT,c,split);assert j(f'labels/{split}.json')==rows
        complete=j('training_complete.json');assert complete['identity']==identity
        assert set(complete['runs'])=={r['name'] for r in c['runs']}
        audits=[n for n in names if n.startswith('audits/') and n.endswith('/completion.json')];assert len(audits)==1
        audit=j(audits[0]);assert audit['status']=='passed_numeric_return_review_pending'
        assert audit['audit_identity']['auditor_sha256']==sha(ROOT/'scripts/verify_v1_4_p10_training.py')
        assert audit['config_sha256']==ch and audit['input_sha256']==ih
        assert audit['updates']==60000 and audit['draws']==30720000 and len(audit['runs'])==12
        environments={}
        for n in names:
            if n.startswith('sessions/') and n.endswith('/environment.json'):
                e=j(n);assert e['config_sha256']==ch and e['device']=='cuda' and e['gpu']=='Tesla T4'
                assert hashlib.sha256(z.read(str(Path(n).parent/'requirements.lock.txt'))).hexdigest()==e['lock_sha256']
                environments[e['environment_id']]=e
        assert audit['audit_identity']['environment_id'] in environments
        smokes=[j(n) for n in names if n.startswith('smoke/') and n.endswith('/smoke.json')]
        assert smokes and all(s['status']=='passed' and s['device']=='cuda' and s['config_sha256']==ch and s['environment_id'] in environments for s in smokes)
        assert all(len(s['resume'])==4 and all(r['status']=='bitwise' for r in s['resume']) for s in smokes)
        selected=[];total_train=total_val=0.;count=0
        for run in c['runs']:
            name=run['name'];prefix='runs/'+name+'/';result=j(prefix+'result.json')
            assert complete['runs'][name]==manifest['files'][prefix+'result.json']
            assert result['run']==run and result['identity']==identity and result['updates']==5000 and result['draws']==2560000 and result['parameter_count']==262912
            curve=j(prefix+'curve.json');assert len(curve)==5000 and [v['update'] for v in curve]==list(range(1,5001))
            scores=[x for x in curve if 'val_mse' in x];assert len(scores)==20
            expected=set(range(250,5001,250))|{100}
            assert set(result['checkpoints'])=={f'update_{u:05d}.pt' for u in expected}
            ar=next(a for a in audit['runs'] if a['run']==name)
            assert ar['result_sha256']==manifest['files'][prefix+'result.json'] and len(ar['scores'])==20
            gen=np.random.Generator(np.random.PCG64(run['draw_seed']));cursor=0
            for u in sorted(expected):
                n=prefix+f'update_{u:05d}.pt';digest=manifest['files'][n]
                assert result['checkpoints'][Path(n).name]==digest==j(n[:-3]+'.json')['sha256']
                st=torch.load(io.BytesIO(z.read(n)),map_location='cpu',weights_only=False)
                assert st['identity']==identity and st['run']==run and st['update']==u and st['draws']==512*u and st['unique_train_positions']==50000
                assert st['training_boundary']==dict(updates=5000,interval=250) and st['curve']==curve[:u]
                while cursor<u:gen.integers(0,50000,size=512,dtype=np.int64);cursor+=1
                assert st['sampler_rng']==gen.bit_generator.state
                assert all(s['environment_id'] in environments for s in st['sessions'])
                for h,s in st['stats'].items():
                    orig=j(f'statistics/seed{run["lm_seed"]}_l3_{h}.json')
                    assert torch.equal(s['mu'],torch.tensor(orig['mean'])) and float(s['scale'])==orig['scale']
                assert all(torch.isfinite(t).all() for t in st['model'].values())
                torch.testing.assert_close(st['model']['decoder'].norm(dim=0),torch.ones(512),atol=2e-6,rtol=2e-6)
                assert len(st['optimizer']['state'])==4
                for x in st['optimizer']['state'].values():
                    assert int(x['step'])==u and torch.isfinite(x['exp_avg']).all() and torch.isfinite(x['exp_avg_sq']).all()
                assert set(st['rng'])=={'python','numpy','torch','cuda'}
                if u%250==0:
                    a=next(a for a in ar['scores'] if a['update']==u)
                    assert a['passed'] and a['saved_mse']==st['curve'][-1]['val_mse'] and a['checkpoint_sha256']==digest
                    assert np.isclose(a['recomputed_mse'],a['saved_mse'],atol=1e-7,rtol=1e-6)
                    assert j(str(Path(audits[0]).parent/'checkpoints'/name/f'{u:05d}.json'))==a
                count+=1
            best=min(scores,key=lambda x:(x['val_mse'],x['update']))
            assert best['update']==result['best_update']==ar['best_update'] and best['val_mse']==result['best_val_mse']
            n=prefix+result['best_checkpoint'];dest=OUT/'selected'/name/'selected.pt';dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(z.read(n))
            assert sha(dest)==manifest['files'][n]
            selected.append(dict(run=run,checkpoint=str(dest.relative_to(ROOT)),sha256=sha(dest),selected_update=result['best_update'],validation_mse=result['best_val_mse'],source_member=n))
            total_train+=result['train_seconds'];total_val+=result['validation_seconds']
            print('verified',name,flush=True)
        # Compact metadata is preserved, never duplicate all 252 checkpoint tensors.
        for n in names:
            if n.endswith('.pt'):continue
            p=OUT/'returned_metadata'/n;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert p.read_bytes()==z.read(n)
            else:p.write_bytes(z.read(n))
    sel=dict(schema='p10-selected-dictionaries-v1',contract_sha256=ch,input_sha256=ih,models=selected)
    write(OUT/'selected_dictionary_manifest.json',sel)
    write(OUT/'completion.json',dict(status='passed_training_only',p10_training_complete=True,p10_complete=False,
        runs=12,checkpoints_verified=count,validation_mse_verified=240,updates=60000,draws=30720000,
        source_members_verified=len(manifest['files']),environment_ids=sorted(environments),
        train_seconds=total_train,validation_seconds=total_val,numeric_audit_seconds=audit['elapsed_seconds'],
        elapsed_seconds=time.monotonic()-start,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ARCHIVE,ARCHIVE.with_suffix('.sha256.json'),ROOT/CONTRACT,OUT/'selected_dictionary_manifest.json',Path(__file__)]},
        limitations=['Original GPU activation caches remain on Drive; stored numerical audit checked locally, not recomputed from absent caches','Semantic/fidelity/transfer and source reproduction return audit pending']))

if __name__=='__main__':main()
