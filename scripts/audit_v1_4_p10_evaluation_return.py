"""Verify both P10 evaluation receipts and independently recompute raw point metrics."""
from pathlib import Path
import hashlib,io,json,sys,time,zipfile,collections
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import read,write
from interp_v1_4 import p10,p10_evaluation as ev
from interp_v1_4.runtime import sha
OUT=ROOT/'experiment_v1_4/results/p10_return_audit_20261004_01'
ARCHIVES=[ROOT/'experiment_v1_4/evidence/P10'/f'v1_4_p10_evaluation_return_{stamp}.zip' for stamp in ('1791025268543489116','1791083347086606306')]


def metrics(y,p,C,threshold):
    predicted=(p>=threshold).astype(int) if C==2 else p.argmax(1)
    cm=np.zeros((C,C),int);np.add.at(cm,(y,predicted),1);n=cm.sum(1);observed=n>0
    recall=np.divide(np.diag(cm),n,out=np.zeros(C),where=observed)
    denom=n+cm.sum(0);f1=np.divide(2*np.diag(cm),denom,out=np.zeros(C),where=denom>0)
    auc=None
    if C==2 and observed.all():
        pos=y==1;r=rankdata(p);a=int(pos.sum());b=len(y)-a;auc=float((r[pos].sum()-a*(a+1)/2)/(a*b))
    return dict(balanced_accuracy=float(recall.mean()) if observed.all() else None,macro_f1=float(f1.mean()) if observed.all() else None,
                observed_class_balanced_accuracy=float(recall[observed].mean()),confusion_matrix=cm.tolist(),positions=len(y),class_counts=n.tolist(),binary_auroc=auc)


def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif a is None or isinstance(a,(str,bool)):assert a==b
    else:assert np.isfinite(a) and np.isfinite(b) and np.isclose(a,b,atol=1e-10,rtol=1e-9),(a,b)


def main():
    start=time.monotonic();c=ev.verify(ROOT);ch=sha(ROOT/ev.CONTRACT);OUT.mkdir(parents=True,exist_ok=True);manifests=[]
    for archive in ARCHIVES:
        check=read(archive.with_suffix('.sha256.json'));assert sha(archive)==check['sha256'] and archive.stat().st_size==check['bytes']
        with zipfile.ZipFile(archive) as z:
            names=z.namelist();assert len(names)==len(set(names));m=json.loads(z.read('return_manifest.json'))
            assert set(names)==set(m['files'])|{'return_manifest.json'}
            for n,h in m['files'].items():
                assert not Path(n).is_absolute() and '..' not in Path(n).parts
                assert hashlib.sha256(z.read(n)).hexdigest()==h,n
            manifests.append(m)
    for n,h in manifests[0]['files'].items():assert manifests[1]['files'][n]==h,n
    with zipfile.ZipFile(ARCHIVES[-1]) as z:
        def j(n):return json.loads(z.read(n))
        hashes=manifests[-1]['files'];assert hashes['contract.json']==ch
        feature=j('features_complete.json');fh=hashes['features_complete.json'];selection=j('selection_complete.json');sh=hashes['selection_complete.json'];done=j('evaluation_complete.json')
        assert feature['config_sha256']==ch and len(feature['files'])==195
        assert selection['config_sha256']==ch and selection['features_sha256']==fh and not selection['test_used_for_selection']
        assert done['config_sha256']==ch and done['selection_sha256']==sh and done['tasks']==1248
        assert set(selection['fits'])==set(done['semantic'])=={t['id'] for t in c['tasks']}
        assert len(done['fidelity'])==12
        old=read(ROOT/p10.CONTRACT);labels={}
        for split in ev.SPLITS:
            _,rows=p10.load_split(ROOT,old,split);assert rows==j(f'labels/{split}.json');labels[split]=rows
        envs={};session_count=0
        for n in hashes:
            if n.startswith('sessions/') and n.endswith('/environment.json'):
                e=j(n);assert e['config_sha256']==ch
                assert hashlib.sha256(z.read(str(Path(n).parent/'requirements.lock.txt'))).hexdigest()==e['lock_sha256']
                envs[e['environment_id']]=e;session_count+=1
        report_count=prediction_count=0;statuses=collections.Counter();fits={};semantic={}
        for t in c['tasks']:
            tid=t['id'];obj=j(f'fits/{tid}.json');assert selection['fits'][tid]==hashes[f'fits/{tid}.json']
            assert obj['task']==t and obj['config_sha256']==ch and obj['features_sha256']==fh and not obj['test_used_for_selection'] and obj['environment_id'] in envs
            fit=obj['fit'];fits[tid]=fit;statuses[fit['status']]+=1;assert fit['status']=='passed'
            for split in ('train','val'):
                mask,y,g=ev.target(labels[split],t['label'],t['domain'],split)
                if split=='train' and t['representation']=='shuffled':y=np.random.Generator(np.random.PCG64(t['shuffle_seed'])).permutation(y)
                support=[dict(class_id=k,positions=int(sum(y==k)),sequences=len(set(g[y==k]))) for k in range(p10.LABELS[t['label']])]
                assert fit['support'][split]==support
                assert all(x['positions']>=32 and x['sequences']>=16 for x in support)
            trace=fit['trace'];assert trace and all(x['attempts'][-1]['success'] for x in trace)
            def key(f):return (-f['validation_balanced_accuracy'],len(f['columns']) if t['prefixes'] else 0,f['validation_ce'],-f['lam'],abs((f['threshold'] or .5)-.5),f['threshold'] or 0)
            for size,f in fit['selected'].items():
                eligible=[r for r in trace if size!='single' or len(r['columns'])==1];best=min(eligible,key=key)
                for k in ('columns','lam','threshold','validation_balanced_accuracy','validation_ce'):assert f[k]==best[k]
                assert len(f['mean'])==len(f['std'])==len(f['columns']) and np.isfinite(f['coefficients']).all() and min(f['std'])>=1e-8
                if t['candidates'] is not None:assert set(f['columns'])<=set(t['candidates'])
            s=j(f'semantic/{tid}.json');semantic[tid]=s
            assert done['semantic'][tid]==hashes[f'semantic/{tid}.json'] and s['selection_sha256']==sh and s['task']==t and s['config_sha256']==ch and s['environment_id'] in envs
            mask,y,g=ev.target(labels['test'],t['label'],t['domain'],'test');C=p10.LABELS[t['label']]
            assert set(s['reports'])==set(fit['selected'])
            for size,f in fit['selected'].items():
                n=f'predictions/{tid}/{size}.npy';assert s['predictions'][n]==hashes[n]
                a=np.load(io.BytesIO(z.read(n)),allow_pickle=False);assert a.shape==((len(y),) if C==2 else (len(y),C))
                assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
                if C>2:np.testing.assert_allclose(a.sum(1),1,atol=1e-10)
                marker=j(n[:-4]+'.json');assert marker['sha256']==hashes[n] and marker['identity']==dict(config=ch,selection=sh,task=tid,size=size)
                report=s['reports'][size];point=metrics(y,a,C,f['threshold']);same(point,{k:report[k] for k in point})
                bs=report['cluster_bootstrap'];assert bs['seed']==t['bootstrap_seed'] and bs['requested']==1000 and bs['clusters']==len(set(g))
                for v in bs['metrics'].values():
                    assert 0<=v['valid']<=1000
                    if v['ci95'] is not None:assert len(v['ci95'])==2 and 0<=v['ci95'][0]<=v['ci95'][1]<=1
                report_count+=1;prediction_count+=1
        gap_count=0
        for t in c['tasks']:
            for s in semantic[t['id']]['reports'].values():
                assert set(s['paired_full_probe_gaps'])==set(t['full_references'])
                for ref in t['full_references']:
                    for metric,v in s['paired_full_probe_gaps'][ref].items():
                        same(v['difference'],s[metric]-semantic[ref]['reports']['full'][metric]);assert v['seed']==t['bootstrap_seed'] and v['requested']==1000 and 0<=v['valid']<=1000
                        gap_count+=1
        for e in c['selected']:
            name=e['run']['name'];f=j(f'fidelity/{name}.json')
            assert hashes[f'fidelity/{name}.json']==done['fidelity'][name]==feature['files'][f'fidelity/{name}.json']
            assert f['run']==e['run'] and f['checkpoint_sha256']==e['sha256'] and f['environment_id'] in envs
            assert f['dead_count']==f['fidelity']['train']['inactive_count']
            for split,n in p10.QUOTAS.items():
                m=f['fidelity'][split];assert m['positions']==n and 0<=m['l0']['mean']<=e['run']['k']
                assert len(m['activation_rates'])==512 and sum(x==0 for x in m['activation_rates'])==m['inactive_count']
                assert f['latent_sha256'][split]==feature['files'][f'latents/{name}/{split}.npy']
        log_events=[]
        for n in hashes:
            if n.startswith('logs/'):
                txt=z.read(n).decode()
                if 'PAUSED' in txt or 'Traceback' in txt:log_events.append(dict(path=n,paused='PAUSED' in txt,missing_selection_guard="FileNotFoundError" in txt and 'selection_complete.json' in txt))
        for n in z.namelist():
            if n.startswith('predictions/'):continue
            dest=OUT/'returned_metadata'/n;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.read_bytes()==z.read(n)
            else:dest.write_bytes(z.read(n))
    write(OUT/'return_manifest.json',manifests[-1]);write(OUT/'operational_events.json',dict(events=log_events,resolved='Two fitting time-limit pauses; premature evaluate calls stopped at missing selection gate. Final selection/evaluation complete; no failed fitting tasks.'))
    paths=ARCHIVES+[p.with_suffix('.sha256.json') for p in ARCHIVES]+[ROOT/ev.CONTRACT,OUT/'return_manifest.json',OUT/'operational_events.json',Path(__file__)]
    write(OUT/'status.json',dict(status='passed_local_source_reproduction_pending',p10_complete=False,runs=12,fit_tasks=1248,fit_statuses=dict(statuses),semantic_reports=report_count,prediction_arrays=prediction_count,paired_gap_metrics=gap_count,
         zip_member_hashes_verified=[len(m['files']) for m in manifests],environment_ids=sorted(envs),sessions=session_count,elapsed_seconds=time.monotonic()-start,
         evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},limitations=['Point metrics and saved selection traces checked locally; original-cache refit/prediction/fidelity/CI reproduction remains pending','No Update intervention was included; no causal-use conclusion']))
    print('PASS local audit:',report_count,'reports;',gap_count,'paired gap metrics; source audit pending')

if __name__=='__main__':main()
