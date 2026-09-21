import platform
import pytest
import torch
from scripts.verify_v1_4_evidence import verify_initialization


def test_foreign_roundoff_is_explicit_and_bounded():
    local={'w':torch.tensor([.01, -.02])}
    recorded={'w':local['w']+3e-8}
    foreign={'torch':'other', 'platform':'other'}
    with pytest.raises(AssertionError):verify_initialization(local,recorded,foreign)
    report=verify_initialization(local,recorded,foreign,True)
    assert not report['bitwise_equal'] and report['max_absolute_difference']<1e-7
    with pytest.raises(AssertionError):verify_initialization(local,{'w':torch.tensor([.02,.01])},foreign,True)
    native={'torch':torch.__version__,'platform':platform.platform()}
    with pytest.raises(AssertionError):verify_initialization(local,recorded,native,True)
    assert verify_initialization(local,local,native)['bitwise_equal']
