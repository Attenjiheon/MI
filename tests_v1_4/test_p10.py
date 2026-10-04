import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from corpus.language import render
from interp_v1_4.p10 import join,domain_mask,collect,LAYER
from interp_v1_4.p10_smoke import update_hook_check
from interp_v1_4.p9_smoke import resume_check
from interp_v1_4.model import Transformer
from interp_v1_4.runtime import deterministic

ROOT=Path(__file__).resolve().parents[1]

def fixture():
    records=json.loads((ROOT/'archive/legacy/experiment_v1_2/debug/sequences.json').read_text())[:2]
    positions=sorted([dict(sequence_id=r['sequence_id'],event_id=e['update_id'],token_index=e['end_token_index'])
                      for r in records for e in r['update_events']],key=lambda p:(p['sequence_id'],p['token_index']))
    return records,positions


def test_token_replay_and_missing_not_zero():
    records,positions=fixture();_,rows=join(records,positions)
    for row,p in zip(rows,positions):
        r=next(r for r in records if r['sequence_id']==p['sequence_id']);e=r['update_events'][p['event_id']]
        assert row['token_index']>=13
        assert row['dst_after']==e['state_after']['ABCD'.index(e['dst'])]
        if e['op'] in ('SET','NOT'):
            assert all(row[k] is None for k in ('src','dst_before','src_before','input_disagreement','operator_truth'))
        else:
            assert row['input_disagreement']==(e['dst_before']^e['src_before_or_null'])
            assert row['operator_truth']==(('AND','OR','XOR').index(e['op'])*4+int(e['input_truth_pattern_or_null'],2))
    bad=copy.deepcopy(records);bad[0]['update_events'][0]['dst_after']^=1
    with pytest.raises(ValueError,match='replay'):join(bad,positions)
    bad=copy.deepcopy(positions);bad[0]['token_index']=3
    with pytest.raises(ValueError):join(records,bad)
    with pytest.raises(ValueError):join(records,positions+positions[:1])


def test_transfer_masks_exclude_unseen_class_tasks():
    rows=[dict(dst=d,operator=o,dst_after=d%2,input_disagreement=(o%2 if o>=2 else None)) for d in range(4) for o in range(5)]
    assert sum(domain_mask(rows,'dst_after','variable','train'))==15
    assert sum(domain_mask(rows,'dst_after','variable','test'))==5
    assert sum(domain_mask(rows,'input_disagreement','operator','train'))==8
    assert sum(domain_mask(rows,'input_disagreement','operator','test'))==4
    with pytest.raises(ValueError):domain_mask(rows,'operator','operator','train')


def test_update_capture_identity_and_future_masking():
    deterministic(0);torch.set_num_threads(2);model=Transformer().eval().requires_grad_(False)
    records,_=fixture();report=update_hook_check(model,records,'cpu')
    assert report['future_token_invariance']=='bitwise'


@pytest.mark.parametrize('tool',['sae','transcoder'])
@pytest.mark.parametrize('k',[4,16])
def test_production_engine_resumes_bitwise(tmp_path,tool,k):
    torch.set_num_threads(2)
    assert resume_check(tmp_path,'cpu',LAYER,k,tool)['status']=='bitwise'


def test_independent_numerical_auditor_matches_both_tools():
    from interp_v1_4.dictionary import Dictionary,normalize
    from interp_v1_4.p9 import mse
    from scripts.verify_v1_4_p10_training import recalculate
    torch.set_num_threads(2);g=torch.Generator().manual_seed(710)
    for tool in ('sae','transcoder'):
        hooks='h' if tool=='sae' else 'um';raw={h:torch.randn(33,256,generator=g) for h in hooks}
        stats={h:dict(mean=[.2]*256,scale=float(np.float32(.7))) for h in hooks}
        tensors={h:dict(mu=torch.tensor(s['mean']),scale=torch.tensor(s['scale'])) for h,s in stats.items()}
        ih,th=('h','h') if tool=='sae' else ('u','m')
        model=Dictionary(tool,4,771,width=256)
        actual=mse(model,normalize(raw[ih],tensors[ih]),normalize(raw[th],tensors[th]))
        assert recalculate(model.state_dict(),raw,stats,4,'cpu',tool)['recomputed_mse']==actual
