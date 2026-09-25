import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from interp_v1_4 import p5, p5_probe as probe
from interp_v1_4.probe import objective

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    records=json.loads((ROOT/'archive/legacy/experiment_v1_2/debug/sequences.json').read_text())[:2]
    positions=sorted([dict(sequence_id=r['sequence_id'],event_id=e['read_id'],token_index=e['query_token_index']) for r in records for e in r['read_events']],key=lambda p:(p['sequence_id'],p['token_index']))
    return records,positions


def test_join_rejects_wrong_answer_position_and_duplicate():
    records,positions=fixture();_,rows=p5.join(records,positions)
    assert len(rows)==len(positions)
    wrong=copy.deepcopy(positions);wrong[0]['token_index']+=1
    with pytest.raises(ValueError):p5.join(records,wrong)
    with pytest.raises(ValueError):p5.join(records,positions+positions[:1])


def test_missing_is_excluded_and_transfer_has_no_D_source():
    rows=[dict(previous=None,query=0),dict(previous=0,query=3),dict(previous=1,query=1)]
    assert p5.domains(rows,'previous','transfer','train').tolist()==[False,False,True]
    assert p5.domains(rows,'previous','transfer','test').tolist()==[False,True,False]


@pytest.mark.parametrize('classes',[2,4,16])
def test_objective_analytic_gradient(classes):
    rng=np.random.default_rng(51);x=rng.normal(size=(40,3));y=np.arange(40)%classes
    theta=rng.normal(size=4*(1 if classes==2 else classes))
    numeric=approx_derivative(lambda t:objective(t,x,y,classes,.1)[0],theta).ravel()
    np.testing.assert_allclose(numeric,objective(theta,x,y,classes,.1)[1],atol=1e-8,rtol=1e-5)


def test_support_requires_independent_sequences():
    x=np.arange(128)[:,None];y=np.arange(128)%2
    fit=probe.fit(x,y,np.zeros(128),x,y,np.zeros(128))
    assert fit['status']=='NA'


def test_train_only_scaling_prefix_and_selection():
    rng=np.random.default_rng(9);y=np.arange(256)%2;g=np.arange(256)
    x=np.c_[y*2+rng.normal(0,.1,256),rng.normal(size=(256,3)),np.ones(256)]
    val=x.copy();val[:,1]+=4
    fit=probe.fit(x,y,g,val,y,g,prefixes=True)
    assert fit['status']=='passed'
    assert fit['ranking'][0]==0 and 4 not in fit['ranking']
    assert len(fit['trace'])==16
    selected=fit['selected']['up_to_four'];assert selected['columns']==[0]
    np.testing.assert_allclose(selected['mean'],x[:,[0]].mean(0))
    assert probe.metrics(y,probe.probabilities(selected,val),2,selected['threshold'])['balanced_accuracy']==1


def test_metrics_missing_classes_and_auc_ties_cluster_bootstrap():
    y=np.array([0,1,0,1]);p=np.array([.2,.2,.8,.8]);groups=np.array(['a','a','b','b'])
    result=probe.metrics(y,p,2,.5,groups,71,100)
    assert result['binary_auroc']==.5
    assert result['cluster_bootstrap']['metrics']['binary_auroc']['ci95']==[.5,.5]
    missing=probe.metrics(np.zeros(4,dtype=int),p,2,.5)
    assert missing['balanced_accuracy'] is None and missing['macro_f1'] is None


def test_checksum_resume_and_reject_modified_cache(tmp_path):
    path=tmp_path/'part.npz';identity={'config_sha256':'debug'}
    p5.save_unit(path,{'0_h':np.ones((4,256),np.float32)},dict(identity=identity),tmp_path,None)
    assert p5.valid_unit(path,identity)
    with pytest.raises(ValueError):p5.valid_unit(path,{'config_sha256':'other'})
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError):p5.valid_unit(path,identity)


def test_random_seed_namespace_is_inherited():
    import hashlib
    key='v1.4|test'
    assert p5.derived('random_projection',key)==int.from_bytes(hashlib.sha256(('20260909|experiment-spec-v1.0|random_projection|'+key).encode()).digest()[:8],'big')


def test_test_labels_cannot_change_fitting():
    rng=np.random.default_rng(52);rows=[dict(sequence_id=str(i),current=i%2,previous=None,query=i%4) for i in range(128)]
    labels={s:copy.deepcopy(rows) for s in p5.QUOTAS};arrays={s:rng.normal(size=(128,3)) for s in p5.QUOTAS}
    first=p5.analyze_one(arrays,labels,'current','iid',False,None,None,19)
    for r in labels['test']:r['current']=1-r['current']
    second=p5.analyze_one(arrays,labels,'current','iid',False,None,None,19)
    assert first['selected']==second['selected'] and first['trace']==second['trace']
