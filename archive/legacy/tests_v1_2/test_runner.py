import copy
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
import torch
from interp_v1_2 import cli,persistence
from interp_v1_2.runtime import deterministic,sha,save,restore
from interp_v1_2.model import Transformer
from interp_v1_2.training import Progress,lm_optimizer,lm_update

@pytest.fixture(autouse=True)
def setup():
    deterministic(101); torch.set_num_threads(2)


def test_fixed_budget_selection_and_gate():
    def entry(update,ce,acc):
        return dict(update=update,checkpoint=f'{update}.pt',validation=dict(general=dict(answer_ce=ce,correct=acc),diagnostics={n:dict(correct=acc,answer_count=512,sequence_count=512) for n in cli.GATE_DIAGNOSTICS}))
    early=entry(1,.2,1.); low_ce_failure=entry(2,.1,.9)
    assert cli.choose_best(early,entry(2,.2,1.)) is early
    best=cli.choose_best(early,low_ce_failure)
    assert best is low_ce_failure
    with pytest.raises(ValueError,match='incomplete'): cli.decide(best,Progress(prediction_tokens=3000000),16000000)
    assert cli.decide(best,Progress(prediction_tokens=16000001),16000000)['decision']=='failed'
    assert cli.decide(early,Progress(prediction_tokens=16000001),16000000)['decision']=='passed'
    bad=copy.deepcopy(early); bad['validation']['diagnostics']['other_variable']['sequence_count']=511
    with pytest.raises(ValueError,match='quota'): cli.decide(bad,Progress(prediction_tokens=16000001),16000000)


def test_validation_only(monkeypatch,tmp_path):
    paths=[]
    monkeypatch.setattr(cli,'metadata',lambda p:paths.append(str(p)) or [])
    monkeypatch.setattr(cli,'evaluate',lambda *args,**kwargs:dict(correct=1.,answer_ce=0.,rows=[dict(sequence_id=str(i)) for i in range(512)]))
    cli.evaluate_gate_suite(object(),tmp_path,16)
    relative=[str(Path(p).relative_to(tmp_path)) for p in paths]
    assert len(relative)==4 and all('val_' in p and '/test_' not in p for p in relative)


def test_persistence_incremental_and_partial_copy(tmp_path,monkeypatch):
    run=tmp_path/'run'; run.mkdir(); drive=tmp_path/'drive'
    (run/'init.pt').write_bytes(b'init')
    copied=[]; actual=persistence.verified_copy
    monkeypatch.setattr(persistence,'verified_copy',lambda s,d: (copied.append(str(s)),actual(s,d))[-1])
    persistence.publish(run,drive,'init.pt')
    (run/'step.pt').write_bytes(b'complete')
    persistence.publish(run,drive,'step.pt')
    assert sum(p.endswith('/init.pt') for p in copied)==1
    (drive/'orphan.pt').write_bytes(b'incomplete')
    restored=tmp_path/'restored'; path=persistence.recover(drive,restored)
    assert path.read_bytes()==b'complete' and not (restored/'orphan.pt').exists()
    (drive/'step.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): persistence.recover(drive,tmp_path/'corrupt')


def test_failed_copy_does_not_advance_index(tmp_path,monkeypatch):
    run=tmp_path/'run'; run.mkdir(); drive=tmp_path/'drive'
    (run/'init.pt').write_bytes(b'init'); persistence.publish(run,drive,'init.pt')
    old=(drive/'LATEST.json').read_bytes(); (run/'new.pt').write_bytes(b'next')
    def fail(*args): raise IOError('interrupted')
    monkeypatch.setattr(persistence,'verified_copy',fail)
    with pytest.raises(IOError): persistence.publish(run,drive,'new.pt')
    assert (drive/'LATEST.json').read_bytes()==old


def test_actual_train_short_shard_join_and_final_budget():
    from interp_v1_2.data import token_rows
    root=Path('data/language_v1_2')
    rows=list(token_rows((root/'train_shards').glob('*.tokens.jsonl')))
    frozen=json.loads((root/'manifest.json').read_text())['frozen_training']
    assert rows[:26048]==list(token_rows(Path('data/language_v1/train_shards').glob('*.tokens.jsonl')))
    assert len(rows[26048:26112])==64
    assert len(rows)==frozen['final_cursor']==64*frozen['final_update']
    assert sum(len(r)-1 for r in rows[:-64])<16000000<=sum(len(r)-1 for r in rows)


def test_runner_interrupted_resume_is_exact(tmp_path,monkeypatch):
    # Runs the real optimizer and real save/restore, not mocked training updates.
    monkeypatch.setattr(cli,'environment',lambda out:{'environment_id':'test'})
    args=dict(root=Path('.'),device='cpu',debug=True,smoke_report=None,pause_after_updates=None,resume=None)
    uninterrupted=tmp_path/'full'; cli.run(Namespace(**args,output=uninterrupted,persistent_dir=None))
    paused=tmp_path/'paused'; drive=tmp_path/'drive'
    cli.run(Namespace(**{**args,'pause_after_updates':2},output=paused,persistent_dir=drive))
    checkpoint_bytes=(paused/'checkpoints/update_000002.pt').read_bytes()
    recovered=tmp_path/'recovered'; cp=persistence.recover(drive,recovered)
    cli.run(Namespace(**{**args,'resume':cp},output=recovered,persistent_dir=drive))
    a=torch.load(uninterrupted/'checkpoints/update_000003.pt',weights_only=False)
    b=torch.load(recovered/'checkpoints/update_000003.pt',weights_only=False)
    assert a['state']==b['state'] and a['best']==b['best']
    for name,value in a['model'].items(): assert torch.equal(value,b['model'][name])
    for param,values in a['optimizer']['state'].items():
        for key,value in values.items(): assert torch.equal(value,b['optimizer']['state'][param][key])
    assert torch.equal(a['rng_states']['torch'],b['rng_states']['torch'])
    assert checkpoint_bytes==(recovered/'checkpoints/update_000002.pt').read_bytes()
    assert len(list(recovered.glob('events/*.json')))==3
    wrong=copy.deepcopy(a['hashes']); wrong['configs']='v1.1'
    model=Transformer(); opt=lm_optimizer(model)
    with pytest.raises(ValueError,match='hashes differ'): restore(cp,model,opt,wrong)


def test_initialization_matches_v11():
    from interp_v1_1.model import Transformer as Previous
    deterministic(0); a=Previous(); deterministic(0); b=Transformer()
    assert len(b.blocks)==4 and sum(p.numel() for p in b.parameters())==797184
    for name,value in a.state_dict().items(): assert torch.equal(value,b.state_dict()[name])


def test_next_validation_boundary_survives_resume(tmp_path):
    model=Transformer(); opt=lm_optimizer(model); state=Progress(prediction_tokens=99998)
    seq=[[1,3,9,13,8,9,13,2],[1,3,9,14,4,9,8,9,13,2]]
    first=lm_update(model,opt,seq,state,microbatch=1)
    assert first['validation_due'] and state.next_validation_boundary==200000
    save(tmp_path/'boundary.pt',model,opt,state.payload(),hashes={'version':'1.2'})
    expected=lm_update(model,opt,seq,state,microbatch=1)
    other=Transformer(); other_opt=lm_optimizer(other)
    payload=restore(tmp_path/'boundary.pt',other,other_opt,{'version':'1.2'})
    restored=Progress(**payload['state'])
    assert lm_update(other,other_opt,seq,restored,microbatch=1)==expected
    assert not expected['validation_due'] and restored==state
    for key,value in model.state_dict().items(): assert torch.equal(value,other.state_dict()[key])
