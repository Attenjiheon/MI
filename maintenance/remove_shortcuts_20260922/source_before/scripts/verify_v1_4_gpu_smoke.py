"""Verify returned GPU smoke integrity and recorded assertions; never unpickle checkpoints."""
import gzip
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp_v1_4.runtime import sha, verify_inputs

archive = Path(sys.argv[1])
output = Path(sys.argv[2])
expected = {'requirements.lock.txt', 'smoke.json', 'select_pairs_slice.jsonl.gz', 'nvidia-smi.txt',
            'deepwide12_read4_debug.pt', 'persistent_copy/deepwide12_read4_debug.pt'}
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert len(names) == len(set(names))
    assert set(names) == expected | {'persistent_copy/'}
    assert z.testzip() is None
    members = {}
    for name in expected:
        h = hashlib.sha256()
        with z.open(name) as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        members[name] = h.hexdigest()
    s = json.loads(z.read('smoke.json'))
    assert all(s[k] == v for k, v in dict(phase='v1.4-smoke', status='passed', scope='cuda',
        fixture_only=False, frozen_input_verified=True, debug_only=True, reuse_in_experiment=False).items())
    inputs = verify_inputs(ROOT)
    inputs['debug'] = sha(ROOT / 'experiment_v1_2/debug/sequences.json')
    inputs['code'] = {str(p.relative_to(ROOT)): sha(p) for folder in ('interp_v1_4','interp_v1_2','corpus')
                      for p in sorted((ROOT / folder).glob('*.py'))}
    assert inputs == s['input_hashes']
    env = s['environment']
    assert members['requirements.lock.txt'] == env['lock_sha256']
    identity = {k: env[k] for k in ('python','platform','torch','cuda','cudnn','gpu','lock_sha256')}
    assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16] == env['environment_id']
    assert env['gpu'] in z.read('nvidia-smi.txt').decode()
    assert ('torch==' + env['torch']) in z.read('requirements.lock.txt').decode().splitlines()
    assert set(s['cells']) == {'deepwide12_read4'}
    c = s['cells']['deepwide12_read4']
    for k,v in dict(parameters=9485312, blocks=12, width=256, rope_head_width=64,
                    selected_microbatch=16, maximum_length_forward_backward=302, causal_and_pad_mask='passed').items():
        assert c[k] == v, k
    assert c['padding_max_abs_error'] < 1e-5 and c['gradient_accumulation_max_abs_error'] < 1e-5
    assert c['loss_weights']['rejects_non_bit_answer_target'] is True
    assert len(c['training']) == 2
    r = c['resume']
    assert (r['status'],r['next_cursor'],r['next_update'],r['prediction_tokens']) == ('passed',192,3,21755)
    assert members['deepwide12_read4_debug.pt'] == members['persistent_copy/deepwide12_read4_debug.pt'] == r['checkpoint_sha256']
    i = c['interpretation']
    for k in ('sae_k4','sae_k16','transcoder_k4','transcoder_k16'):
        assert (i[k]['updates'],i[k]['draws']) == (100,51200)
    assert i['probe']['status'] == 'passed' and i['probe']['width'] == 256
    for k in ('sae_patches','transcoder_patches'):
        assert all(i[k][a] == b for a,b in dict(identity='bitwise',full='passed',sparse='passed').items())
    assert i['hooks'] == dict(blocks=12,width=256,read_and_update='passed')
    assert c['select_metric_path']['pair_cells'] == s['smoke_select_pairs_one_per_cell'] == 42
    assert s['smoke_select_sequences'] == 24
    assert s['gate_code_path']['passed'] is False and s['gate_code_path']['debug_only'] is True
    assert s['gate_code_path']['checks']['coverage'] is True
    def finite(obj):
        if isinstance(obj,dict):
            for v in obj.values(): finite(v)
        elif isinstance(obj,list):
            for v in obj: finite(v)
        elif isinstance(obj,float): assert math.isfinite(obj)
    finite(s)
    run = json.loads((ROOT/'experiment_v1_4/configs/run.json').read_text())
    assert run['effective_batch'] == 64
    pairs = []
    seen = set()
    with gzip.open(ROOT/run['data_root']/'first_repeat/select.jsonl.gz','rt') as f:
        for line in f:
            p = json.loads(line)
            if p['cell_id'] not in seen:
                seen.add(p['cell_id']); pairs.append(p)
    returned = [json.loads(line) for line in gzip.decompress(z.read('select_pairs_slice.jsonl.gz')).decode().splitlines()]
    assert returned == pairs and len(pairs) == 42
    output.mkdir(parents=True,exist_ok=True)
    for name in ('smoke.json','requirements.lock.txt','nvidia-smi.txt'):
        dest = output/name
        if dest.exists(): assert dest.read_bytes() == z.read(name)
        else: dest.write_bytes(z.read(name))
result = dict(status='passed_return_evidence_verification',archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
    member_sha256=members, input_hashes=inputs, environment_id=env['environment_id'], effective_batch=64,
    selected_microbatch=16, checks=['ZIP CRC and inventory','checkpoint/persistent copy identity','all frozen input and runtime code hashes',
    'environment lock and ID','required smoke assertions and finite metrics','42 select pairs exact semantic identity'],
    limitation='CUDA execution assertions verified against matching source and returned records; not independently rerun on local CPU. Checkpoints were not unpickled. Debug gate false is expected, not a production behavior gate result.')
(output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','archive_sha256','environment_id')}))
