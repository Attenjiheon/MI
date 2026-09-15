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
from interp.behavior import GATE_DIAGNOSTICS,adjudicate,may_extend
from interp import cli


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


def gate_fixture(general_accuracy=.99,diagnostic_accuracy=.95):
    general={'correct':general_accuracy}
    diagnostics={name:{'correct':diagnostic_accuracy,'rows':[{'sequence_id':f'{name}-{i}'} for i in range(512)]}
                 for name in GATE_DIAGNOSTICS}
    return general,diagnostics


def test_p3_adjudication_pass_extend_and_fail():
    general,diagnostics=gate_fixture()
    assert adjudicate(general,diagnostics,[.5,.4,.3,.2],1_000_000)['decision']=='passed'
    general['correct']=.98
    extended=adjudicate(general,diagnostics,[1.,.9,.95,.8],1_000_000)
    assert extended['decision']=='extend_to_3m' and extended['improving_transitions']==2
    assert adjudicate(general,diagnostics,[1.,.9,.95,.8],3_000_000)['decision']=='failed'
    assert adjudicate(general,diagnostics,[1.,1.,1.,.9],1_000_000)['decision']=='failed'


def test_p3_gate_requires_every_diagnostic_quota():
    general,diagnostics=gate_fixture()
    diagnostics['other_variable']['rows'].pop()
    with pytest.raises(ValueError,match='Diagnostic quota'):
        adjudicate(general,diagnostics,[.5,.4,.3,.2],1_000_000)


def test_p3_gate_suite_reads_validation_only(monkeypatch,tmp_path):
    paths=[]
    def fake_metadata(path):
        paths.append(Path(path)); return [str(path)]
    def fake_evaluate(model,records,diagnostic=False,microbatch=16):
        return {'correct':1.,'answer_ce':0.,'rows':[]}
    monkeypatch.setattr(cli,'metadata',fake_metadata)
    monkeypatch.setattr(cli,'evaluate',fake_evaluate)
    cli.evaluate_gate_suite(object(),tmp_path,16)
    relative={str(path.relative_to(tmp_path)) for path in paths}
    expected={'data/language_v1/val_iid.jsonl.gz'}
    expected.update(f'data/language_v1/diagnostics/val_{name}.jsonl.gz' for name in GATE_DIAGNOSTICS)
    assert relative==expected
    assert all('/test_' not in value for value in relative)


def test_debug_lm_writes_and_persists_p3_contract(monkeypatch,tmp_path):
    output=tmp_path/'run'; persistent=tmp_path/'persistent'
    monkeypatch.setattr(cli,'environment',lambda out:{'environment_id':'test'})
    monkeypatch.setattr('sys.argv',['interp.cli','train_lm','--debug','--output',str(output),
                                   '--persistent-dir',str(persistent)])
    cli.main()
    result=json.loads((output/'result.json').read_text())
    assert result['state']['update']==2
    assert result['measurement_first_50_updates']['updates']==2
    assert result['gate']['decision']=='not_adjudicated_debug'
    for name in ('init.pt','best.pt','last.pt','training.jsonl','gate_decision.json','result.json','manifest.json'):
        assert (output/name).read_bytes()==(persistent/name).read_bytes()


def test_p3_snapshot_recovery_ignores_partial_copy(tmp_path):
    from interp.persistence import publish, recover
    output=tmp_path/'run'; output.mkdir()
    (output/'last.pt').write_bytes(b'complete checkpoint')
    (output/'training.jsonl').write_text('complete log')
    persistent=tmp_path/'drive'
    publish(output,persistent)
    partial=persistent/'snapshots'/'incomplete'; partial.mkdir()
    (partial/'last.pt').write_bytes(b'partial')
    restored=tmp_path/'restored'
    assert recover(persistent,restored)==restored/'last.pt'
    assert (restored/'training.jsonl').read_text()=='complete log'
    pointer=json.loads((persistent/'LATEST.json').read_text())
    (persistent/'snapshots'/pointer['snapshot']/'last.pt').write_bytes(b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        recover(persistent,tmp_path/'corrupt')


def test_debug_resume_discards_unsaved_log_tail(monkeypatch,tmp_path):
    output=tmp_path/'run'
    monkeypatch.setattr(cli,'environment',lambda out: {'environment_id':'test'})
    # The runtime stub provides the files that the real environment collector saves.
    output.mkdir()
    (output/'requirements.lock.txt').write_text('test')
    (output/'nvidia-smi.txt').write_text('CPU')
    initial=Transformer(); opt=lm_optimizer(initial)
    # Run via a fresh directory, keeping the environment stub self-contained.
    def env(out):
        (out/'requirements.lock.txt').write_text('test')
        (out/'nvidia-smi.txt').write_text('CPU')
        return {'environment_id':'test'}
    monkeypatch.setattr(cli,'environment',env)
    output=tmp_path/'fresh'
    args=['interp.cli','train_lm','--debug','--output',str(output)]
    monkeypatch.setattr('sys.argv',args); cli.main()
    log=output/'training.jsonl'; entries=log.read_text().splitlines()
    tail=json.loads(entries[-1]); tail['state']['update']=3
    with log.open('a') as f: f.write(json.dumps(tail)+'\n')
    monkeypatch.setattr('sys.argv',args+['--resume',str(output/'last.pt')]); cli.main()
    assert log.read_text().splitlines()==entries
    assert list(output.glob('*-discarded-training.jsonl'))


def test_p3_production_manifest_and_decision_contract(monkeypatch,tmp_path):
    """Exercise production finalization on CPU with a synthetic single-update budget."""
    class CPUTransformer(Transformer):
        def to(self, *args, **kwargs):
            return self
    monkeypatch.setattr(cli,'Transformer',CPUTransformer)
    for name in ('reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda: None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda: 123)
    monkeypatch.setattr(cli,'verify_inputs',lambda root:{'configs':'config','corpus_manifest':'data'})
    monkeypatch.setattr(cli,'token_rows',lambda paths:[[1,3,9,13,2]]*64)
    monkeypatch.setattr(cli,'environment',lambda out:{'environment_id':'cpu-mocked'})
    def update(model,opt,rows,state,microbatch):
        state.update=1; state.prediction_tokens=1_000_001; state.next_data_cursor=64
        state.next_validation_boundary=1_100_000
        return dict(tokens=1_000_001,validation_due=True)
    monkeypatch.setattr(cli,'lm_update',update)
    general,diagnostics=gate_fixture()
    general.update(answer_ce=.01,answer_count=512,rows=[{'sequence_id':str(i)} for i in range(512)])
    monkeypatch.setattr(cli,'evaluate_gate_suite',lambda *args:(general,diagnostics))
    output=tmp_path/'production_contract'
    monkeypatch.setattr('sys.argv',['interp.cli','train_lm','--device','cuda','--output',str(output)])
    cli.main()
    manifest=json.loads((output/'manifest.json').read_text())
    decision=json.loads((output/'gate_decision_1000000.json').read_text())
    assert manifest['status']=='passed' and manifest['phase']=='P3'
    assert manifest['actual_command'] and manifest['environment_id']=='cpu-mocked'
    assert decision['overshoot']==1 and decision['general_answer_count']==512
    assert decision['selected_update']==1
