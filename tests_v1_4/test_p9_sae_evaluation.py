import numpy as np
import pytest
import torch
from interp_v1_4.p7_metrics import fidelity,bins,bin_id,random_sets,matched_indices,paired_summary
from interp_v1_4.p9_sae_causal import commit_arrays,valid_arrays,causal_scores,validate_pair
from interp_v1_4.patching import sparse_patch,direction_patch
from interp_v1_4.dictionary import Dictionary
from interp_v1_4.p5 import read


def test_fidelity_mean_denominators_and_bias():
    x=np.array([[1.,2.],[3.,4.]])
    prediction=x+np.array([3.,-2.])
    result=fidelity(x,prediction,np.array([[1,0],[0,1]]),np.array([0.,0.]))
    assert result['nmse']==26/30 and result['r2']==1-26/4
    assert result['ev']==1. and result['l0']['mean']==1
    assert result['inactive_count']==0


def test_bins_zero_and_repeated_edges():
    rule=bins([0,1,1,1,1],[0,2,2,2,2])
    assert rule['h_edges']==[1.] and rule['z_edges']==[2.]
    assert bin_id(0,0,rule)==(-1,-1)
    assert matched_indices(1,2,[0,1,1],[0,2,2],rule)==[1,2]
    assert matched_indices(1,2,[0],[0],rule)==[]


def test_fixed_random_disjoint_and_count():
    a=random_sets(512,[1,2,3,4],123)
    assert a==random_sets(512,[1,2,3,4],123)
    assert all(len(set(item))==4 and not set(item)&{1,2,3,4} for item in a)
    assert len(a)<=200


def test_residual_error_preserved_and_nonorthogonal_projection():
    d=Dictionary('sae',4,17,width=256);o=torch.randn(1,256)
    zo=torch.zeros(1,512);zc=torch.ones(1,512)
    value=sparse_patch(o,zo,zc,d,[1,2],torch.tensor(.7))
    torch.testing.assert_close(value-o,.7*d.decoder[:,[1,2]].sum(1)[None])
    q=torch.tensor([[1.,1.],[0.,1.],[0.,0.]])
    original=torch.zeros(1,3);donor=torch.tensor([[2.,3.,5.]])
    torch.testing.assert_close(direction_patch(original,donor,q),torch.tensor([[2.,3.,0.]]))


def test_resume_checksum_and_uncommitted(tmp_path):
    p=tmp_path/'unit.npz';commit_arrays(p,{'x':np.ones(2)},'a');assert valid_arrays(p,'a')
    with pytest.raises(ValueError):valid_arrays(p,'b')
    p.write_bytes(b'bad')
    with pytest.raises(ValueError):valid_arrays(p,'a')
    q=tmp_path/'orphan.npz';q.write_bytes(b'partial');assert not valid_arrays(q,'a') and not q.exists()


def test_cluster_average_directions_candidates():
    rows=[dict(origin='a',v=0.),dict(origin='a',v=2.),dict(origin='b',v=4.)]
    r=paired_summary(rows,['v'],123,100)
    assert r['origins']==2 and r['metrics']['v']['mean']==2.5
    assert r==paired_summary(rows,['v'],123,100)


def test_full_vocabulary_flip_and_unchanged_errors():
    o=np.zeros(15);o[13]=2
    p=o.copy();p[14]=3;p[5]=4
    r=causal_scores(o,p,0,1,True)
    assert r['full_flip']==0 and r['binary_flip']==1 and r['delta_margin']==3
    assert causal_scores(o,p,0,0,False)['error_induced']==1


def test_pair_replay_all_conditions():
    from interp_v1_4.p9_sae_evaluation import records
    from pathlib import Path
    for name in ('val_changed_memory','val_changed_composition','val_unchanged_memory','val_unchanged_composition'):
        p=records(Path('data/language_v1_4/rebuild_01/causal_pairs')/(name+'.jsonl.gz'))[0]
        validate_pair(p)
        bad={**p,'original_answer':1-p['original_answer']}
        with pytest.raises(ValueError):validate_pair(bad)


def test_orphan_array_can_be_recommitted(tmp_path):
    path=tmp_path/'unit.npz';path.write_bytes(b'interrupted')
    commit_arrays(path,{'x':np.arange(3)},'contract')
    assert valid_arrays(path,'contract')
    with np.load(path) as z:np.testing.assert_array_equal(z['x'],np.arange(3))
