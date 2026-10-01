import json
from pathlib import Path
import numpy as np
import pytest
import torch
from interp_v1_4.p9 import train_run,load_tensor,mse,verify
from interp_v1_4.p9_smoke import resume_check
from interp_v1_4 import dictionary as sparse
from interp_v1_4.runtime import tensor_digest
from scripts.verify_v1_4_p9_training import recalculate

@pytest.mark.parametrize('tool',['sae','transcoder'])
@pytest.mark.parametrize('layer',[0,3,7,11])
@pytest.mark.parametrize('k',[4,16])
def test_resume(tmp_path,tool,layer,k):
    torch.set_num_threads(2)
    assert resume_check(tmp_path,'cpu',layer,k,tool)['status']=='bitwise'
    p=tmp_path/'full/update_00004.pt';p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='Corrupt'):load_tensor(p)

@pytest.mark.parametrize('tool',['sae','transcoder'])
def test_independent_update_and_numeric(tmp_path,tool):
    torch.set_num_threads(2);g=torch.Generator().manual_seed(912)
    ih,th=('h','h') if tool=='sae' else ('u','m');hooks=set((ih,th))
    train={h:torch.randn(15,256,generator=g) for h in sorted(hooks)}
    val={h:torch.randn(9,256,generator=g) for h in sorted(hooks)}
    stats={h:dict(mu=torch.full((256,),.3 if h==ih else -.7),scale=torch.tensor(.4 if h==ih else 2.)) for h in hooks}
    run=dict(name='debug',tool=tool,k=4,init_seed=113,draw_seed=21)
    result=train_run(run,train,val,stats,tmp_path,{'debug':True},'cpu','cpu',updates=2,interval=2)
    state=load_tensor(tmp_path/'update_00002.pt')
    plain={h:dict(mean=s['mu'].tolist(),scale=float(s['scale'])) for h,s in stats.items()}
    assert recalculate(state['model'],val,plain,4,'cpu',tool)['recomputed_mse']==result['best_val_mse']
    initial=sparse.Dictionary(tool,4,113,width=256);opt=sparse.optimizer(initial)
    gen=np.random.Generator(np.random.PCG64(21));x=sparse.normalize(train[ih],stats[ih]);y=sparse.normalize(train[th],stats[th])
    for _ in range(2):
        idx=gen.integers(0,len(x),512,dtype=np.int64);sparse.update(initial,opt,x[idx],y[idx])
    assert tensor_digest(initial.state_dict())==tensor_digest(state['model'])
    assert train_run(run,train,val,stats,tmp_path,{'debug':True},'cpu','cpu',updates=2,interval=2)==result
    with pytest.raises(ValueError,match='identity'):train_run(run,train,val,stats,tmp_path,{'debug':True},'cpu','cpu',updates=4,interval=2)

def test_contract():
    root=Path(__file__).resolve().parents[1];c=verify(root)
    assert len(c['runs'])==16
    assert {(r['lm_seed'],r['layer'],r['tool'],r['k'],r['sparse_seed']) for r in c['runs']}=={(0,l,t,k,1) for l in (0,3,7,11) for t in ('sae','transcoder') for k in (4,16)}
    from interp_v1_4.p5 import derived
    for r in c['runs']:
        assert r['init_seed']==derived('dictionary_init',r['run_key'])
        assert r['draw_seed']==derived('position_draw',r['draw_key'])
        other=next(x for x in c['runs'] if x['layer']==r['layer'] and x['k']==r['k'] and x['tool']!=r['tool'])
        assert other['draw_seed']==r['draw_seed'] and other['init_seed']!=r['init_seed']
        assert derived('position_draw',r['draw_key'].replace('sparse_seed=1','sparse_seed=0'))!=r['draw_seed']
