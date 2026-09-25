import copy
import numpy as np
import pytest
from scripts import p5_independent_audit as audit
from interp_v1_4 import p5_probe
from interp_v1_4.p5 import analyze_one


def test_tied_auc_and_cluster_bootstrap():
    y=np.array([0,1,0,1,1,0]);p=np.array([.1,.1,.5,.5,.9,.9]);g=np.array(['a','a','b','c','c','d'])
    expected=np.mean([(b>a)+.5*(b==a) for b in p[y==1] for a in p[y==0]])
    assert audit.weighted_auc(y,p)==pytest.approx(expected)
    actual=audit.evaluate(y,p,2,.5,g,8923)
    audit.compare(actual,p5_probe.metrics(y,p,2,.5,g,8923))
    assert actual['cluster_bootstrap']['metrics']['binary_auroc']['valid']<1000


def test_multiclass_and_missing_class_draws():
    y=np.array([0,0,1,2,2]);p=np.eye(3)[[0,1,1,0,2]];g=np.array([1,1,2,3,3])
    audit.compare(audit.evaluate(y,p,3,None,g,37),p5_probe.metrics(y,p,3,None,g,37))
    assert audit.evaluate(y,p,3,None,g,37)['binary_auroc'] is None


@pytest.fixture(scope='module')
def fitted():
    rng=np.random.default_rng(71);labels={};arrays={}
    for split,n in [('train',240),('val',160),('test',180)]:
        a=rng.normal(size=(n,5));a[:,-1]=1;arrays[split]=a
        labels[split]=[dict(current=int(a[i,0]+.3*a[i,1]>0),previous=i%2,query=i%4,sequence_id=f'q{i//2:04d}') for i in range(n)]
    name='seed0_trained_l0_h_full_current_iid';key='fixture'
    config=dict(probe=dict(lambdas=[.01,.1,1,10],thresholds=(np.arange(1,20)/20).tolist()),tasks={name:dict(key=key,bootstrap_seed=71,shuffle_seed=None)})
    result=analyze_one(arrays,labels,'current','iid',False,None,None,71)
    assert result['status']=='passed'
    return name,dict(result=result),arrays,labels,config


def test_full_prediction_statistics_and_ci(fitted):
    output=audit.verify_task(*fitted)
    assert output['full']['metrics']['positions']==180


@pytest.mark.parametrize('field',['mean','coefficients','ci95'])
def test_detect_corrupted_result(fitted,field):
    name,item,arrays,labels,c=fitted;item=copy.deepcopy(item)
    if field=='ci95':item['result']['evaluation']['full']['cluster_bootstrap']['metrics']['binary_auroc']['ci95'][0]+=.01
    else:item['result']['selected']['full'][field][0]+=.1
    with pytest.raises(ValueError):audit.verify_task(name,item,arrays,labels,c)


def test_prefix_ranking_transfer_and_shuffle(fitted):
    _,_,arrays,labels,c=fitted
    name='seed0_trained_l0_h_coordinate_current_transfer';c=copy.deepcopy(c)
    c['tasks']={name:dict(key='fixture',bootstrap_seed=71,shuffle_seed=92)}
    result=analyze_one(arrays,labels,'current','transfer',True,None,92,71)
    assert result['status']=='passed'
    audit.verify_task(name,dict(result=result),arrays,labels,c)
    result['ranking']=list(reversed(result['ranking']))
    with pytest.raises(ValueError,match='ranking'):audit.verify_task(name,dict(result=result),arrays,labels,c)


def test_cache_hash_rows_and_tensor_validation(tmp_path):
    source=tmp_path/'source';work=tmp_path/'work';labels={s:[dict(sequence_id='a'),dict(sequence_id='b')] for s in ('train','val','test')}
    c=dict(models=[dict(lm_seed=0,checkpoint='ckpt')],files={'ckpt':'frozen'},hooks={'h':'resid_post','u':'mlp_in','m':'mlp_out'})
    inventory={};markers=[]
    for split in labels:
        folder=source/'cache/seed0_trained'/split;folder.mkdir(parents=True)
        p=folder/'000000.npz';values={f'{l}_{h}':np.ones((2,256),np.float32) for l in range(12) for h in 'hum'}
        np.savez(p,row_indices=np.array([0,1]),**values)
        marker=p.with_suffix('.json');audit.save(marker,dict(identity=dict(config_sha256='cfg',lm_seed=0,checkpoint_sha256='frozen',split=split,position_type='READ',labels_sha256='labels',sequence_offset=0),positions=2,layers=list(range(12)),hooks=c['hooks'],sha256=audit.sha(p)))
        inventory[str(marker.relative_to(source))]=audit.sha(marker);inventory[f'labels/{split}.json']='labels';markers.append(marker)
    arrays,receipts,_=audit.stage_cache(source,work,0,'trained',c,'cfg',labels,inventory)
    assert len(receipts)==3 and arrays['train'][0,'h'].shape==(2,256)
    p=markers[0].with_suffix('.npz');p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):audit.stage_cache(source,work,0,'trained',c,'cfg',labels,inventory)
