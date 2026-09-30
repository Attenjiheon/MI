from pathlib import Path
import json
import numpy as np
import pytest
import torch
from interp_v1_4 import dictionary as sparse
from interp_v1_4.p8 import train_run,load_tensor,mse,arrays
from interp_v1_4.p8_smoke import resume_check
from interp_v1_4.p8_patching import predict_output,feature_patch
from interp_v1_4.runtime import sha,tensor_digest
from scripts.verify_v1_4_p8_training import recalculate


@pytest.mark.parametrize('layer',[0,3,7,11])
@pytest.mark.parametrize('k',[4,16])
def test_tc_resume(tmp_path,layer,k):
    torch.set_num_threads(2)
    assert resume_check(tmp_path,'cpu',layer,k)['status']=='bitwise'
    p=tmp_path/'full/update_00004.pt'
    p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='Corrupt'):load_tensor(p)


def test_tc_target_and_independent_numeric(tmp_path):
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(912)
    train={h:torch.randn(15,256,generator=g) for h in 'um'}
    val={h:torch.randn(9,256,generator=g) for h in 'um'}
    stats={h:dict(mu=torch.full((256,),.3 if h=='u' else -.7),scale=torch.tensor(.4 if h=='u' else 2.)) for h in 'um'}
    run=dict(name='debug',k=4,init_seed=113,draw_seed=21)
    result=train_run(run,train,val,stats,tmp_path,{'debug':True},'cpu','cpu',updates=2,interval=2)
    saved=load_tensor(tmp_path/'update_00002.pt')
    d=sparse.Dictionary('transcoder',4,113,width=256);d.load_state_dict(saved['model'])
    v=sparse.normalize(val['u'],stats['u']);t=sparse.normalize(val['m'],stats['m'])
    expected=mse(d,v,t)
    assert result['best_val_mse']==expected and not np.isclose(expected,mse(d,v,v))
    plain={h:dict(mean=s['mu'].tolist(),scale=float(s['scale'])) for h,s in stats.items()}
    assert recalculate(saved['model'],val,plain,4,'cpu')['recomputed_mse']==expected
    # First SGD draw and target are independently reconstructed with the frozen optimizer.
    initial=sparse.Dictionary('transcoder',4,113,width=256);opt=sparse.optimizer(initial)
    gen=np.random.Generator(np.random.PCG64(21))
    x=sparse.normalize(train['u'],stats['u']);y=sparse.normalize(train['m'],stats['m'])
    for _ in range(2):
        idx=gen.integers(0,len(x),size=512,dtype=np.int64)
        sparse.update(initial,opt,x[idx],y[idx])
    assert tensor_digest(initial.state_dict())==tensor_digest(saved['model'])
    # Finishing the same run again is idempotent; different boundaries are rejected.
    assert train_run(run,train,val,stats,tmp_path,{'debug':True},'cpu','cpu',updates=2,interval=2)==result
    with pytest.raises(ValueError,match='identity'):
        train_run(run,train,val,stats,tmp_path,{'debug':True},'cpu','cpu',updates=4,interval=2)


def test_encoder_initialization():
    d=sparse.Dictionary('transcoder',16,123,width=256)
    assert not torch.equal(d.encoder,d.decoder.T)
    torch.testing.assert_close(d.encoder.norm(dim=1),torch.ones(512))
    torch.testing.assert_close(d.decoder.norm(dim=0),torch.ones(512))
    assert d.encoder_bias.count_nonzero()==d.decoder_bias.count_nonzero()==0


@pytest.mark.parametrize('layer',[0,3,7,11])
def test_output_scale_and_residual_skip(layer):
    from interp_v1_4.model import Transformer,batch
    from interp_v1_4.patching import capture,patch
    torch.set_num_threads(2);torch.manual_seed(19)
    model=Transformer().eval();d=sparse.Dictionary('transcoder',4,123,width=256)
    stats={'u':dict(mu=torch.zeros(256),scale=torch.tensor(.2)), 'm':dict(mu=torch.ones(256)*.3,scale=torch.tensor(2.7))}
    # Valid short initialization followed by READ A; no answer token enters.
    prefix=[0,3,9,13,3,10,14,3,11,13,3,12,14,8,9]
    ids,mask,_=batch([prefix], 'cpu', shift=False);pos=len(prefix)-1
    with torch.no_grad(),capture(model) as cache:original=model(ids,mask)
    u=cache[f'blocks.{layer}.mlp_in'][:,pos];m=cache[f'blocks.{layer}.mlp_out'][:,pos]
    donor=u+torch.linspace(-1,1,256)[None]
    z0=d.encode(sparse.normalize(u,stats['u']));z1=d.encode(sparse.normalize(donor,stats['u']))
    J=torch.argsort((z1-z0).abs()[0],descending=True)[:4].tolist()
    delta=(z1[:,J]-z0[:,J])@d.decoder[:,J].T
    value=feature_patch(d,m,u,donor,J,stats)
    torch.testing.assert_close(value,m+2.7*delta)
    assert not torch.allclose(value,m+.2*delta)
    torch.testing.assert_close(predict_output(d,u,stats),.3+2.7*d(sparse.normalize(u,stats['u']))[0])
    observed={};hooks={f'blocks.{layer}.mlp_out':lambda x: replace(x), f'blocks.{layer}.resid_post':lambda x: observe(x)}
    def replace(x):
        x=x.clone();x[:,pos]=value;return x
    def observe(x):observed['h']=x[:,pos].clone();return x
    model.interventions=hooks
    with torch.no_grad():out=model(ids,mask)
    torch.testing.assert_close(observed['h'],cache[f'blocks.{layer}.resid_mid'][:,pos]+value)
    assert torch.isfinite(out).all()
    model.interventions={}
    with torch.no_grad(),patch(model,f'blocks.{layer}.mlp_out',[pos],m):
        torch.testing.assert_close(model(ids,mask),original,atol=0,rtol=0)


def test_contract_paired_draws():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'experiment_v1_4/p8_r1/contract.json').read_text())
    old=json.loads((root/'experiment_v1_4/p6_r1/contract.json').read_text())
    assert len(c['runs'])==24
    for tc,sae in zip(c['runs'],old['runs']):
        assert tc['draw_key']==sae['draw_key'] and tc['draw_seed']==sae['draw_seed']
        assert tc['init_seed']!=sae['init_seed'] and tc['tool']=='transcoder'
        a=np.random.Generator(np.random.PCG64(tc['draw_seed']))
        b=np.random.Generator(np.random.PCG64(sae['draw_seed']))
        assert np.array_equal(a.integers(0,50000,(100,512)),b.integers(0,50000,(100,512)))
