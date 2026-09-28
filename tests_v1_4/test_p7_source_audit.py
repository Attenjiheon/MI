import numpy as np
import pytest
from scripts.audit_v1_4_p7_sources import fidelity_independent,compare
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
