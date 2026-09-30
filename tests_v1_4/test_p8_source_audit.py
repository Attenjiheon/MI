import numpy as np
import pytest
from scripts.audit_v1_4_p8_sources import fidelity_independent,compare
from interp_v1_4.p7_metrics import fidelity


def test_independent_fidelity_and_mismatch_detection():
    rng=np.random.default_rng(20260927)
    x=rng.normal(size=(128,6));y=x+rng.normal(size=x.shape)*.2+1
    z=np.maximum(0,rng.normal(size=(128,8)));z[:,0]=0
    mean=np.arange(6)*.1
    a=fidelity(x,y,z,mean);b=fidelity_independent(x,y,z,mean)
    compare(a,b)
    assert b['inactive_count']==1 and b['ev']>b['r2']
    b['inactive_count']=0
    with pytest.raises(AssertionError):compare(a,b)


def test_compare_rejects_missing_or_changed_choices():
    with pytest.raises(AssertionError):compare({'x':[1,2]}, {'x':[1,3]})
    with pytest.raises(AssertionError):compare({'x':None},{'x':0})
    with pytest.raises(AssertionError):compare({'x':1},{'y':1})
    with pytest.raises(AssertionError):compare(float('nan'),float('nan'))


def test_source_audit_encodes_u_and_scores_m(tmp_path,monkeypatch):
    import torch
    from pathlib import Path
    from scripts import audit_v1_4_p8_sources as audit
    from interp_v1_4.dictionary import Dictionary
    from interp_v1_4.p5 import write
    from interp_v1_4 import p5
    d=Dictionary('transcoder',4,98,width=256).eval().requires_grad_(False)
    stats={'u':dict(mu=torch.ones(256)*.2,scale=torch.tensor(.3)), 'm':dict(mu=torch.ones(256)*-.4,scale=torch.tensor(2.))}
    rng=np.random.default_rng(84)
    inputs={s:rng.normal(size=(9,256)).astype('float32') for s in ('train','val','test')}
    outputs={s:rng.normal(size=(9,256)).astype('float32')*3 for s in inputs}
    semantic={'fidelity':{}}
    mean=((outputs['train'].astype(float)-stats['m']['mu'].numpy())/2).mean(0)
    for s in inputs:
        with torch.no_grad():pred,z=d((torch.tensor(inputs[s])-stats['u']['mu'])/stats['u']['scale'])
        x=(outputs[s].astype(float)-stats['m']['mu'].numpy())/2
        semantic['fidelity'][s]=audit.fidelity_independent(x,pred.numpy(),z.numpy(),mean)
    run=dict(name='debug',lm_seed=0,layer=3);folder=tmp_path/'returned/runs/debug'
    write(folder/'selection.json',dict(fits={}))
    write(folder/'semantic.json',semantic);write(folder/'matching.json',{})
    np.savez(folder/'semantic_predictions.npz')
    write(tmp_path/audit.ev.RECEIPT,dict(config_sha256='debug'))
    monkeypatch.setattr(audit.ev,'model_entry',lambda *a:(d,stats))
    monkeypatch.setattr(audit.ev,'load_arrays',lambda *a:outputs)
    monkeypatch.setattr(audit.ev,'directions',lambda *a:(np.zeros((256,512)),{}))
    original_encode=audit.ev.encode
    monkeypatch.setattr(audit.ev,'encode',lambda d,stats,x,device:original_encode(d,stats,x,'cpu'))
    def cache(source,seed,kind,split,layer,hook,ch):
        assert hook=='u';return inputs[split]
    monkeypatch.setattr(p5,'cache_array',cache)
    class ReachedReplay(Exception):pass
    def stop(*a):raise ReachedReplay()
    monkeypatch.setattr(audit,'load_lm',stop)
    with pytest.raises(ReachedReplay):audit.audit_run(tmp_path,tmp_path,tmp_path/'returned',tmp_path,{},dict(run=run),{},tmp_path/'fits')
