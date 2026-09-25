import json
from pathlib import Path
import pytest
from scripts.p5_cpu_resume import Repair,canonical,sha,restore


def setup_run(tmp_path,monkeypatch):
    from interp_v1_4 import p5
    root=tmp_path/'root';local=tmp_path/'local';drive=tmp_path/'drive'
    for p in (root,local,drive):p.mkdir()
    rows=[dict(sequence_id='s',current=1,previous=None,query=0,A=1,B=0,C=1,D=0,state=5)]
    config=dict(quotas={'train':1},models=[dict(lm_seed=0,checkpoint='cp',init_checkpoint='init')],files={'cp':'a'*64,'init':'b'*64},tasks={})
    contract=root/p5.CONTRACT;contract.parent.mkdir(parents=True);contract.write_bytes(canonical(config))
    (drive/'contract.json').write_bytes(canonical(config))
    monkeypatch.setattr(p5,'verify',lambda root:config)
    monkeypatch.setattr(p5,'load_split',lambda root,config,split:({},rows))
    (drive/'labels').mkdir();(drive/'labels/train.json').write_bytes(canonical(rows))
    return root,local,drive,rows,sha(contract)


def test_repair_truncated_local_label_and_idempotent_resume(tmp_path,monkeypatch):
    root,local,drive,rows,ch=setup_run(tmp_path,monkeypatch)
    p=local/'labels/train.json';p.parent.mkdir();p.write_text('[{"partial":')
    backup=Path(restore(root,local,drive))
    assert p.read_bytes()==canonical(rows)
    assert (backup/'local/labels/train.json').read_text()=='[{"partial":'
    restore(root,local,drive)
    assert p.read_bytes()==(drive/'labels/train.json').read_bytes()


def test_repair_corrupt_drive_labels_from_frozen_corpus(tmp_path,monkeypatch):
    root,local,drive,rows,ch=setup_run(tmp_path,monkeypatch)
    (drive/'labels/train.json').write_text('truncated')
    backup=Path(restore(root,local,drive))
    assert (backup/'drive/labels/train.json').read_text()=='truncated'
    assert (drive/'labels/train.json').read_bytes()==canonical(rows)


def test_cache_checksum_recovery_and_marker_label_guard(tmp_path,monkeypatch):
    root,local,drive,rows,ch=setup_run(tmp_path,monkeypatch)
    rel=Path('cache/seed0_trained/train/000000.npz')
    src=drive/rel;src.parent.mkdir(parents=True);src.write_bytes(b'verified cache')
    marker=dict(identity=dict(config_sha256=ch,lm_seed=0,split='train',position_type='READ',checkpoint_sha256='a'*64,labels_sha256=sha(drive/'labels/train.json')),sha256=sha(src))
    src.with_suffix('.json').write_bytes(canonical(marker))
    bad=local/rel;bad.parent.mkdir(parents=True);bad.write_bytes(b'incomplete copy')
    backup=Path(restore(root,local,drive))
    assert bad.read_bytes()==src.read_bytes()
    assert (backup/'local'/rel).read_bytes()==b'incomplete copy'
    marker['identity']['labels_sha256']='wrong';src.with_suffix('.json').write_bytes(canonical(marker))
    with pytest.raises(ValueError,match='different labels'):restore(root,local,drive)


def test_do_not_select_between_different_valid_results(tmp_path):
    local=tmp_path/'local';drive=tmp_path/'drive';local.mkdir();drive.mkdir()
    (local/'x.json').write_text('{"result":1}');(drive/'x.json').write_text('{"result":2}')
    r=Repair(local,drive,tmp_path/'backup')
    with pytest.raises(ValueError,match='Two different valid'):r.reconcile(Path('x.json'),lambda p:True)
    assert json.loads((local/'x.json').read_text())['result']==1
    assert json.loads((drive/'x.json').read_text())['result']==2
    assert (tmp_path/'backup/local/x.json').exists() and (tmp_path/'backup/drive/x.json').exists()


def test_valid_local_completed_result_repairs_partial_drive(tmp_path):
    local=tmp_path/'local';drive=tmp_path/'drive';local.mkdir();drive.mkdir()
    (local/'x.json').write_text('{"complete":true}');(drive/'x.json').write_text('{')
    r=Repair(local,drive,tmp_path/'backup')
    r.reconcile(Path('x.json'),lambda p:json.loads(p.read_text()).get('complete') is True)
    assert (drive/'x.json').read_bytes()==(local/'x.json').read_bytes()
    assert (tmp_path/'backup/drive/x.json').read_text()=='{'
