import copy
from types import SimpleNamespace

import pytest

from corpus.v1_4 import cells, rng, sample_pair
from scripts import generate_v1_4_corpus as generator


def test_first_repeat_attempt_cap_includes_registry_rejections(tmp_path, monkeypatch):
    cell=cells()[0]
    monkeypatch.setattr(generator,'cells',lambda:[cell])
    monkeypatch.setattr(generator,'MAX_ATTEMPTS',5)
    builder=generator.V14Builder(tmp_path,set(),set())
    calls=[]
    def candidate(rng, split, cell, remaining):
        calls.append(remaining)
        return {'answer':0},remaining,{}
    monkeypatch.setattr(generator,'sample_pair',candidate)
    monkeypatch.setattr(builder,'reserve_pair',lambda pair:'read_prefix_collision')
    with pytest.raises(RuntimeError,match='MAX_ATTEMPTS'):
        builder.first_repeat('select',1)
    assert calls==[5]
    assert 'first_repeat/select' not in builder.manifest


def test_first_repeat_records_attempts_and_resets_only_on_acceptance(tmp_path,monkeypatch):
    monkeypatch.setattr(generator,'cells',lambda:[cells()[0]])
    monkeypatch.setattr(generator,'MAX_ATTEMPTS',5)
    builder=generator.V14Builder(tmp_path,set(),set()); limits=[]
    def candidate(rng,split,cell,remaining):
        limits.append(remaining)
        return {'answer':0},2,{}
    decisions=iter(['read_prefix_collision',None,None])
    monkeypatch.setattr(generator,'sample_pair',candidate)
    monkeypatch.setattr(builder,'reserve_pair',lambda pair:next(decisions))
    builder.first_repeat('select',2)
    report=builder.manifest['first_repeat/select']['cells'][cells()[0].cell_id]
    assert limits==[5,3,5]
    assert report['attempts_per_target']==[4,2]


@pytest.mark.parametrize('corruption',['cell','index','pattern','inserted','target','repeat_count','depth'])
def test_postwrite_pair_semantics_reject_corruption(corruption):
    cell=cells()[0]
    pair,_,_=sample_pair(rng('audit_regression',0),'select',cell)
    generator.validate_stored_pair(pair,'select')
    broken=copy.deepcopy(pair)
    if corruption=='cell': broken['cell_id']=cells()[1].cell_id
    elif corruption=='index': broken['cell_index']+=1
    elif corruption=='pattern': broken['input_pattern']='1'
    elif corruption=='inserted': broken['inserted_read_id']=broken['repeat_target_read_id']
    elif corruption=='target': broken['repeat_target_read_id']-=1
    elif corruption=='depth':
        broken['origin']['read_events'][broken['origin_target_read_id']]['structural_depth']=999
        broken['repeat']['read_events'][broken['repeat_target_read_id']]['structural_depth']=999
    else:
        broken['repeat']['read_events'][broken['repeat_target_read_id']]['reads_of_query_var_since_last_update']=2
    with pytest.raises((AssertionError,StopIteration)):
        generator.validate_stored_pair(broken,'select')
