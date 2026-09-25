import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from interp_v1_4 import dictionary as sparse
from interp_v1_4.p6 import arrays, commit_tensor, load_tensor, train_run
from interp_v1_4.p6_smoke import resume_check
from interp_v1_4.runtime import sha


def test_resume_and_corruption(tmp_path):
    torch.set_num_threads(2)
    assert resume_check(tmp_path,'cpu',3,4)['status']=='bitwise'
    p=tmp_path/'full/update_00004.pt'
    p.write_bytes(p.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='Corrupt'):load_tensor(p)


def test_wrong_identity_rejected(tmp_path):
    torch.set_num_threads(2)
    x=torch.randn(10,256);stats=dict(mu=torch.zeros(256),scale=torch.tensor(1.))
    r=dict(name='debug',k=16,init_seed=2,draw_seed=5)
    train_run(r,x,x,stats,tmp_path,{'id':1},'debug','cpu',updates=4,interval=2,stop_at=2)
    with pytest.raises(ValueError,match='identity'):
        train_run(r,x,x,stats,tmp_path,{'id':2},'debug','cpu',updates=4,interval=2)


def test_cache_receipt_and_row_coverage(tmp_path):
    p=tmp_path/'cache/seed0_trained/train/a.npz';p.parent.mkdir(parents=True)
    a={f'{l}_{h}':np.ones((3,256),np.float32)*(l+1) for l in (0,3,7,11) for h in 'hum'}
    np.savez(p,row_indices=np.array([2,0,1]),**a)
    receipt={'quotas':{'train':3},'cache_files':[dict(path=str(p.relative_to(tmp_path)),sha256=sha(p),positions=3)]}
    assert arrays(tmp_path,receipt,0,'train')['7_h'][0,0]==8
    np.savez(p,row_indices=np.array([0,0,1]),**a);receipt['cache_files'][0]['sha256']=sha(p)
    with pytest.raises(ValueError,match='indices'):arrays(tmp_path,receipt,0,'train')
    receipt['cache_files'][0]['sha256']='bad'
    with pytest.raises(ValueError,match='receipt'):arrays(tmp_path,receipt,0,'train')


def test_topk_tie_and_independent_encoder():
    model=sparse.Dictionary('sae',4,1,width=256)
    assert model.encoder.data_ptr()!=model.decoder.data_ptr()
    with torch.no_grad():model.encoder.zero_();model.encoder_bias.fill_(1)
    z=model.encode(torch.zeros(1,256))
    assert torch.equal(z[0,:4],torch.ones(4)) and int(torch.count_nonzero(z))==4
    with pytest.raises(ValueError):sparse.statistics(torch.ones(3,256))
