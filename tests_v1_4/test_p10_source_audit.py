from pathlib import Path
import copy
import numpy as np
import pytest
import torch
from interp_v1_4.p10_source_audit import compare,committed,independent_encode,independent_fidelity
from interp_v1_4 import dictionary as sparse
from interp_v1_4.p7_metrics import fidelity
from interp_v1_4.p5 import write


def test_compare_does_not_accept_changed_identity_or_counts():
    compare({'x':[1.0,None,2]}, {'x':[1.000001,None,2]})
    for a,b in [(100000,100001),('passed','failed'),([1],[1,2]),({'x':1},{'y':1}),(1.,float('nan'))]:
        with pytest.raises(ValueError):compare(a,b)


@pytest.mark.parametrize('tool',['sae','transcoder'])
@pytest.mark.parametrize('k',[4,16])
def test_independent_forward_and_metrics(tool,k):
    torch.set_num_threads(2);g=torch.Generator().manual_seed(411);d=sparse.Dictionary(tool,k,37,width=256)
    hooks='h' if tool=='sae' else 'um';stats={h:dict(mu=torch.randn(256,generator=g),scale=torch.tensor(.3 if h=='u' else 1.7)) for h in hooks}
    state=dict(model=d.state_dict(),stats=stats);ih,th=('h','h') if tool=='sae' else ('u','m')
    raw=torch.randn(29,256,generator=g).numpy();p,z=independent_encode(state,raw,tool,k,'cpu')
    with torch.no_grad():expected,latent=d(sparse.normalize(torch.tensor(raw),stats[ih]))
    np.testing.assert_array_equal(p,expected.numpy());np.testing.assert_array_equal(z,latent.numpy())
    target=np.random.default_rng(1).normal(size=p.shape);mean=np.full(256,.2)
    compare(independent_fidelity(target,p,z,mean),fidelity(target,p,z,mean),atol=1e-12,rtol=1e-12)


def test_resume_receipt_requires_matching_contract(tmp_path):
    p=tmp_path/'receipt.json';assert not committed(p,{'id':1})
    write(p,dict(status='passed',identity={'id':1}))
    assert committed(p,{'id':1})
    with pytest.raises(ValueError):committed(p,{'id':2})


def test_cpu_source_refit_predictions_ci_and_quota_guard(tmp_path,monkeypatch):
    from tests_v1_4.test_p10_evaluation import fixture_pipeline
    from interp_v1_4 import p10_evaluation as ev,p10_source_audit as audit
    root,source,c,_=fixture_pipeline(tmp_path,monkeypatch)
    assert ev.selection(root,source) and ev.evaluate(root,source)
    out=tmp_path/'audit';ident={'debug_only':True,'audit_config_sha256':'debug'}
    monkeypatch.setattr(audit,'source_gate',lambda *args:({'metric_tolerance':dict(atol=2e-6,rtol=2e-5)},c))
    monkeypatch.setattr(audit,'identity',lambda root:ident)
    monkeypatch.setattr(audit,'session',lambda *args:dict(environment_id='debug'))
    write(out/'gpu_complete.json',dict(status='passed',identity=ident,files={}))
    with pytest.raises(ValueError,match='semantic coverage'):audit.cpu(root,source,out)
    assert len(list((out/'tasks').glob('*.json')))==2
    # A miniature fixture must not produce a production completion certificate.
    assert not (out/'source_audit_complete.json').exists()
