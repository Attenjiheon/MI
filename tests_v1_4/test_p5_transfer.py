import json
from pathlib import Path
import zipfile
import pytest
from scripts.p5_transfer import pack, receive, sha


def fixture(tmp_path):
    root=tmp_path/'run';root.mkdir();(root/'contract.json').write_text('{"debug":true}')
    (root/'probes').mkdir();(root/'probes'/'done.json').write_text('{"status":"passed"}')
    for seed in (0,1):
        p=root/f'cache/seed{seed}_trained/train/000000.npz';p.parent.mkdir(parents=True);p.write_bytes(bytes(range(256))*4)
        p.with_suffix('.json').write_text(json.dumps({'sha256':sha(p)}))
    return root


def test_metadata_and_seed_parts_roundtrip_resume(tmp_path):
    root=fixture(tmp_path);exports=tmp_path/'export';out=tmp_path/'out'
    meta=pack(root,exports,limit=700);cache=pack(root,exports,seed=0,limit=700)
    for index in (meta,cache):
        for part in index['parts']:
            p=exports/part['name'];assert sha(p)==part['sha256']
            receive(p,out,sha(root/'contract.json'));receive(p,out,sha(root/'contract.json'))
    assert (out/'cache/seed0_trained/train/000000.npz').read_bytes()==(root/'cache/seed0_trained/train/000000.npz').read_bytes()
    assert not (out/'cache/seed1_trained/train/000000.npz').exists()
    assert pack(root,exports,seed=0,limit=700)==cache


def test_reject_contract_conflict_and_source_corruption(tmp_path):
    root=fixture(tmp_path);index=pack(root,tmp_path/'export')
    part=tmp_path/'export'/index['parts'][0]['name']
    with pytest.raises(ValueError):receive(part,tmp_path/'out','wrong')
    p=root/'cache/seed0_trained/train/000000.npz';p.write_bytes(b'corrupt')
    with pytest.raises(ValueError):pack(root,tmp_path/'export',seed=0)


def test_reject_traversal(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('../escape','x');z.writestr('transfer_manifest.json',json.dumps(dict(contract_sha256='test',files={'../escape':'invalid'})))
    with pytest.raises(ValueError):receive(archive,tmp_path/'out','test')
    assert not (tmp_path/'escape').exists()


def test_refuse_overwrite_or_recursive_export(tmp_path):
    root=fixture(tmp_path)
    with pytest.raises(ValueError):pack(root,root/'exports')
    index=pack(root,tmp_path/'export');out=tmp_path/'out';out.mkdir();(out/'contract.json').write_text('different')
    with pytest.raises(ValueError):receive(tmp_path/'export'/index['parts'][0]['name'],out,sha(root/'contract.json'))
