"""Final reporting regressions: pairing, seed units, provenance, missing support."""
import hashlib
import numpy as np
import pytest
from interp_v1_4.p11 import paired_interval, seed_summaries, semantic_rows, Evidence


def test_paired_constant_shift_has_zero_width_despite_large_between_sequence_variance():
    a={f's{i}':float(i*i) for i in range(30)}
    b={k:v+0.125 for k,v in reversed(list(a.items()))}
    result=paired_interval(a,b,123)
    assert result['mean']==0.125
    assert result['ci95']==[0.125,0.125]
    assert result['valid']==1000


def test_pairing_rejects_equal_length_different_sequences():
    with pytest.raises(ValueError):paired_interval({'a':1.},{'b':1.},1)


def test_paired_bootstrap_matches_independent_reference():
    a={'a':1.,'b':8.,'c':-3.};b={'a':2.,'b':6.,'c':1.}
    result=paired_interval(a,b,77)
    rng=np.random.Generator(np.random.PCG64(77));d=np.array([1.,-2.,4.])
    draws=np.array([np.mean(rng.choice(d,3,replace=True)) for _ in range(1000)])
    np.testing.assert_allclose(result['ci95'],np.percentile(draws,[2.5,97.5]))


def test_layers_k_and_sparse_repeats_are_not_extra_lm_seeds():
    rows=[dict(phase='P7',lm_seed=s,layer=l,tool='sae',k=k,sparse_seed=sp,metric='nmse',value=s+l+k+100*sp)
          for s in (0,1,2) for l in (0,3) for k in (4,16) for sp in (0,1)]
    result=seed_summaries({'fidelity':rows})
    assert len(result)==4
    assert all(r['n_lm']==3 and r['maximum']-r['minimum']==2 for r in result)
    assert all(r['mean']<100 for r in result)


def test_duplicate_seed_is_rejected_instead_of_silently_averaged():
    row=dict(phase='P7',lm_seed=0,layer=0,tool='sae',k=4,sparse_seed=0,metric='nmse',value=1.)
    with pytest.raises(ValueError):seed_summaries({'fidelity':[row,row]})


def test_na_auroc_keeps_zero_valid_draws_and_subsets():
    m=dict(balanced_accuracy=.6,macro_f1=.55,binary_auroc=None,positions=4,class_counts=[2,1,1],
           cluster_bootstrap={'seed':7,'requested':1000,'clusters':3,'metrics':{'binary_auroc':{'valid':0,'ci95':None}}})
    result=semantic_rows({},'task','single',m,'passed',[],'fixture')
    auroc=next(r for r in result if r['metric']=='binary_auroc')
    assert auroc['value'] is None and auroc['bootstrap_valid']==0 and auroc['na_reason']


def test_changed_evidence_is_rejected(tmp_path):
    p=tmp_path/'source.json';p.write_bytes(b'changed')
    with pytest.raises(ValueError):Evidence().raw(p,expected=hashlib.sha256(b'original').hexdigest())
