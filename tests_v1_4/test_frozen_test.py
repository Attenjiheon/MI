import json
from pathlib import Path
import pytest
import torch
from interp_v1_4.frozen_test import (evaluate_once, target_signature, cluster_intervals,
                                    evaluate_pairs, finalize)
from interp_v1_4.behavior import evaluate_first_repeat
from interp_v1_4.model import Transformer
from interp_v1_4.runtime import deterministic


def test_once_and_binding(tmp_path):
    calls=[]
    def inference():
        calls.append(1)
        return {'value':42}
    expected=evaluate_once(tmp_path,{'seed':1},inference,lambda x:x)
    assert evaluate_once(tmp_path,{'seed':1},inference,lambda x:x)==expected
    assert calls==[1]
    with pytest.raises(ValueError,match='binding mismatch'):
        evaluate_once(tmp_path,{'seed':2},inference,lambda x:x)


def test_saved_raw_resumes_reduction_only(tmp_path):
    def failed_reduce(raw):
        raise RuntimeError('report interrupted')
    with pytest.raises(RuntimeError,match='report interrupted'):
        evaluate_once(tmp_path,{},lambda:{'value':42},failed_reduce)
    def forbidden():
        pytest.fail('Repeated inference')
    assert evaluate_once(tmp_path,{},forbidden,lambda x:x)['metrics']=={'value':42}


def test_failed_inference_cannot_be_repeated(tmp_path):
    def fail():
        raise RuntimeError('GPU interrupted')
    with pytest.raises(RuntimeError,match='GPU interrupted'):
        evaluate_once(tmp_path,{},fail,lambda x:x)
    with pytest.raises(RuntimeError,match='never silently repeat'):
        evaluate_once(tmp_path,{},lambda:pytest.fail('repeat'),lambda x:x)


def test_modified_raw_rejected(tmp_path):
    evaluate_once(tmp_path,{},lambda:{'x':1},lambda x:x)
    (tmp_path/'measurements.json').write_text('{}')
    with pytest.raises(AssertionError):
        evaluate_once(tmp_path,{},lambda:pytest.fail('repeat'),lambda x:x)


def test_target_identity_and_cluster_sampling():
    rows=[dict(sequence_id='a',read_id=i,answer=0,correct=0,answer_ce=1.,binary_correct=0,bit_mass=1.) for i in range(10)]
    rows += [dict(sequence_id='b',read_id=0,answer=1,correct=1,answer_ce=0.,binary_correct=1,bit_mass=1.)]
    with pytest.raises(ValueError,match='Duplicate'):
        target_signature(rows+rows[:1])
    spec=dict(kind='records',target_signature=target_signature(rows),answer_count=11)
    with pytest.raises(ValueError,match='Wrong or incomplete'):
        finalize({'rows':rows[:-1]},spec)
    ci=cluster_intervals(rows,'fixture')
    assert ci['accuracy_ci95']==[0.,1.]  # two sequence clusters, not 11 independent READs
    assert ci==cluster_intervals(rows,'fixture')


def test_pair_predictions_match_frozen_evaluator():
    root=Path(__file__).resolve().parents[1]
    path=root/'experiment_v1_4/frozen_test_r1/debug_pairs.jsonl.gz'
    deterministic(31415);torch.set_num_threads(2)
    model=Transformer('deepwide12')
    expected=evaluate_first_repeat(model,path,microbatch=16)
    actual=evaluate_pairs(model,path,16)
    for old,new in zip(expected['rows'],actual['rows'],strict=True):
        assert all(new[k]==v for k,v in old.items())
    assert len(actual['rows'])==84
    assert all(v['prediction_tokens']>0 and v['ce_sum']>0 for v in actual['token_totals'].values())
