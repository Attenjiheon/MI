import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from interp.model import Transformer,batch
from interp.data import extract
from interp.dictionary import Dictionary,statistics
from interp.patching import direction_patch,coordinate_patch,patch
from interp.training import Progress,lm_optimizer,lm_update
from interp.runtime import deterministic,save,restore
from interp.probe import fit_probe,select_prefix
from interp.behavior import may_extend


@pytest.fixture(autouse=True)
def setup():
    deterministic(101); torch.set_num_threads(2)


def test_cache_exact_join_read_and_update():
    records=json.loads(Path('experiment_v1/debug/sequences.json').read_text())[:2]
    model=Transformer()
    for kind,index in [('read','query_token_index'),('update','end_token_index')]:
        positions=[dict(sequence_id=e['sequence_id'],token_index=r[index],event_id=r[kind+'_id']) for e in records for r in e[kind+'_events']]
        cache=extract(model,records,positions,lm_seed=0,checkpoint_sha256='debug',split='debug',position_type=kind)
        assert len(cache['keys'])==len(positions)
        assert all(x.shape==(len(positions),128) and x.dtype==torch.float32 for x in cache['tensors'].values())
        with pytest.raises(ValueError,match='Duplicate position'): extract(model,records,positions+positions[:1],lm_seed=0,checkpoint_sha256='debug',split='debug',position_type=kind)
        bad=copy.deepcopy(positions); bad[0]['event_id']=999
        with pytest.raises(ValueError,match='join mismatch'): extract(model,records,bad,lm_seed=0,checkpoint_sha256='debug',split='debug',position_type=kind)


def test_probe_support_and_constants():
    x=np.zeros((128,4)); y=np.arange(128)%2; g=np.arange(128)
    assert fit_probe(x,y,g,x,y,g)['status']=='NA'
    x[:,0]=y
    assert fit_probe(x,y,np.zeros(128),x,y,g)['status']=='NA'
    selected=select_prefix(x,y,g,x,y,g)
    assert selected['up_to_four']['columns'].tolist()==[0]
    assert selected['single']['validation_balanced_accuracy']==1


def test_dictionary_reject_constant_and_zero_relu():
    with pytest.raises(ValueError): statistics(torch.ones(32,128))
    for kind in ('sae','transcoder'):
        d=Dictionary(kind,4,1)
        with torch.no_grad(): d.encoder.zero_(); d.encoder_bias.fill_(-1)
        assert d.encode(torch.zeros(2,128)).count_nonzero()==0


def test_projection_and_patch_cleanup():
    o=torch.randn(2,128); c=torch.randn(2,128); q=torch.eye(128)[:,:4]
    torch.testing.assert_close(direction_patch(o,c,q),coordinate_patch(o,c,[0,1,2,3]))
    model=Transformer()
    with pytest.raises(RuntimeError):
        with patch(model,'blocks.0.resid_post',[1],o[:1]): raise RuntimeError('debug')
    assert model.interventions=={}


def test_resume_optimizer_rng_and_eval_boundary(tmp_path):
    model=Transformer(); opt=lm_optimizer(model); state=Progress(prediction_tokens=99998)
    seq=[[1,3,9,13,8,9,13,2],[1,3,9,14,4,9,8,9,13,2]]
    log=lm_update(model,opt,seq,state,microbatch=1)
    assert log['validation_due'] and state.next_validation_boundary==200000
    save(tmp_path/'test.pt',model,opt,state.payload(),hashes={'a':'b'})
    expected_rng=(np.random.rand(3),torch.rand(3)); expected_log=lm_update(model,opt,seq,state,microbatch=1)
    expected_params=copy.deepcopy(model.state_dict()); expected_opt=copy.deepcopy(opt.state_dict())
    other=Transformer(); other_opt=lm_optimizer(other)
    with pytest.raises(ValueError,match='hashes differ'): restore(tmp_path/'test.pt',other,other_opt,{'a':'wrong'})
    payload=restore(tmp_path/'test.pt',other,other_opt,{'a':'b'})
    assert np.array_equal(np.random.rand(3),expected_rng[0]); assert torch.equal(torch.rand(3),expected_rng[1])
    other_state=Progress(**payload['state']); assert lm_update(other,other_opt,seq,other_state,microbatch=1)==expected_log
    for k,v in expected_params.items(): assert torch.equal(v,other.state_dict()[k])
    for i,v in expected_opt['state'].items():
        for k,t in v.items(): assert torch.equal(t,other_opt.state_dict()['state'][i][k])
    assert other_state==state


def test_extension_gate_uses_last_four():
    assert not may_extend([1.,.9,.8])
    assert may_extend([1.,.9,.95,.8])
    assert not may_extend([1.,1.,1.,.9])
