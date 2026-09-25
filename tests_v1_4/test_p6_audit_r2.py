import pytest
import torch
from interp_v1_4.dictionary import Dictionary, normalize
from interp_v1_4.p6 import mse
from scripts.verify_v1_4_p6_return_r2 import recalculate


@pytest.mark.parametrize('k',[4,16])
@pytest.mark.parametrize('rows',[33,513])
def test_independent_forward_matches_saved_recipe(k,rows):
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(614)
    raw=torch.randn(rows,256,generator=g)
    mu=torch.randn(256,generator=g);scale=torch.tensor(0.13723461)
    model=Dictionary('sae',k,137,width=256)
    with torch.no_grad():
        model.encoder_bias.copy_(torch.randn(512,generator=g)*0.1)
        model.decoder_bias.copy_(torch.randn(256,generator=g)*0.1)
    expected=mse(model,normalize(raw,dict(mu=mu,scale=scale)))
    result=recalculate(model.state_dict(),raw,dict(mean=mu.tolist(),scale=float(scale)),k,'cpu')
    assert result['recomputed_mse']==expected
    assert 0<=result['topk_candidate_rows_changed']<=rows
    assert result['minimum_topk_boundary_gap']>=0
