import copy
import json
from pathlib import Path
import numpy as np
import pytest
from interp_v1_4 import p10_evaluation as ev
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha
from scripts.build_v1_4_p10_evaluation import tasks
ROOT=Path(__file__).resolve().parents[1]


def test_task_scope_fixed_subsets_and_references():
    rules=read(ROOT/'experiment_v1_4/p10_update_r1/evaluation_rules.json')
    selected=read(ROOT/'experiment_v1_4/results/p10_training_audit_20261003_01/selected_dictionary_manifest.json')['models']
    tt=tasks(rules,selected);assert len(tt)==1248
    assert sum(2 if t['prefixes'] else 1 for t in tt)==2016
    ids={t['id'] for t in tt}
    for t in tt:
        assert set(t['full_references'])<=ids
        if t['candidates'] is not None:assert len(t['candidates'])==128 and len(set(t['candidates']))==128
        if t['domain']!='all':assert t['label'] in rules['transfer']['labels']
    # A shared coordinate/random baseline is not duplicated across k/dictionary tools.
    assert len({(t['lm_seed'],t['representation'],t['hook'],t['label'],t['domain'],t['candidate_limit']) for t in tt if t['representation']!='latent'})==864


def test_array_receipt_detects_corruption(tmp_path):
    p=tmp_path/'x.npy';ev.commit_array(p,np.eye(3),{'id':1});np.testing.assert_array_equal(ev.load_array(p),np.eye(3))
    with pytest.raises(ValueError):ev.commit_array(p,np.eye(3),{'id':2})
    p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError):ev.load_array(p)


def test_paired_gap_identical_predictors_zero():
    y=np.tile([0,1],32);prob=.1+.8*y;groups=np.repeat(np.arange(32),2)
    r=ev.paired_gap(y,prob,prob,2,.5,.5,groups,123,draws=1000)
    for v in r.values():assert v['difference']==0 and v['ci95']==[0,0] and v['valid']==1000


def fixture_pipeline(tmp_path,monkeypatch):
    root=tmp_path/'root';out=tmp_path/'out';cp=root/ev.CONTRACT;cp.parent.mkdir(parents=True);cp.write_text('{}\n')
    ch=sha(cp);rng=np.random.Generator(np.random.PCG64(11));allrows={}
    for split in ('train','val','test'):
        n=256;y=(np.arange(n)//4)%2
        rows=[dict(sequence_id=f'{split}_{i}',token_index=20,token_id=9,dst=i%4,operator=4 if split=='test' else 2,dst_after=int(y[i])) for i in range(n)]
        x=rng.normal(0,.1,(n,3));x[:,0]+=y
        ev.commit_array(ev.raw_path(out,0,'trained',split,'h'),x,{'config':ch})
        write(out/'labels'/f'{split}.json',rows);allrows[split]=rows
    tt=[dict(id='full',lm_seed=0,representation='full',hook='h',label='dst_after',domain='variable',prefixes=False,candidates=None,bootstrap_seed=31,full_references=[]),
        dict(id='coordinate',lm_seed=0,representation='coordinate',hook='h',label='dst_after',domain='variable',prefixes=True,candidates=None,bootstrap_seed=31,full_references=['full'])]
    c=dict(tasks=tt,selected=[]);monkeypatch.setattr(ev,'verify',lambda root:c)
    monkeypatch.setattr(ev,'session',lambda *args:dict(environment_id='debug'))
    paths=[p for p in out.rglob('*') if p.is_file()]
    write(out/'features_complete.json',dict(config_sha256=ch,files={str(p.relative_to(out)):sha(p) for p in paths}))
    return root,out,c,allrows


def test_selection_then_reporting_resume_and_frozen_identity(tmp_path,monkeypatch):
    root,out,c,rows=fixture_pipeline(tmp_path,monkeypatch)
    with pytest.raises(FileNotFoundError):ev.evaluate(root,out)
    assert ev.selection(root,out)
    fit=read(out/'fits/full.json')['fit'];assert fit['status']=='passed'
    # Fit standardization uses A/B/C source-domain data, excluding D rows.
    raw=np.load(ev.raw_path(out,0,'trained','train','h'));mask=np.arange(256)%4!=3
    f=fit['selected']['full'];np.testing.assert_allclose(f['mean'],raw[mask][:,f['columns']].mean(0))
    assert ev.evaluate(root,out);before=sha(out/'evaluation_complete.json')
    assert ev.selection(root,out) and ev.evaluate(root,out);assert sha(out/'evaluation_complete.json')==before
    report=read(out/'semantic/coordinate.json')
    assert set(report['reports'])=={'single','up_to_four'}
    assert report['reports']['single']['positions']==64
    p=out/'fits/full.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='Fit changed'):ev.evaluate(root,out)


def test_failed_optimizer_cannot_be_skipped_on_resume(tmp_path,monkeypatch):
    root,out,c,_=fixture_pipeline(tmp_path,monkeypatch)
    write(out/'fits/full.json',dict(config_sha256=sha(root/ev.CONTRACT),features_sha256=sha(out/'features_complete.json'),task=c['tasks'][0],fit=dict(status='failed')))
    with pytest.raises(ValueError,match='Prior optimizer'):ev.selection(root,out)
