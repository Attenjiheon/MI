"""Hashes, deterministic execution and atomic resumable state."""
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def seed(purpose, key):
    return int.from_bytes(hashlib.sha256(f'20260909|experiment-spec-v1.0|{purpose}|{key}'.encode()).digest()[:8],'big')


def deterministic(value):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(value); np.random.seed(value % 2**32); torch.manual_seed(value % (2**63-1))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value % (2**63-1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_default_dtype(torch.float32)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None: torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def save(path, model, optimizer, state, **extras):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,
                        rng_states=rng_state(),**extras),f)
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)
    return sha(path)


def restore(path, model, optimizer, expected_hashes=None):
    # Only load locally produced/trusted checkpoints; NumPy RNG needs full pickle.
    p=torch.load(path,map_location='cpu',weights_only=False)
    if expected_hashes is not None and p['hashes'] != expected_hashes:
        raise ValueError('Checkpoint config/code/input hashes differ')
    model.load_state_dict(p['model']); optimizer.load_state_dict(p['optimizer']); restore_rng(p['rng_states'])
    return p


def verified_copy(source, destination):
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix(destination.suffix+'.tmp')
    shutil.copyfile(source,temp)
    if sha(source)!=sha(temp): raise IOError('Persistent copy checksum failed')
    temp.replace(destination)


def verify_inputs(root):
    root=Path(root); config=json.loads((root/'experiment_v1/configs/language.json').read_text())
    for key in ('manifest','cpu_validation','postwrite_audit','statistics'):
        info=config['corpus'][key]
        if sha(root/info['path'])!=info['sha256']: raise ValueError(f'Changed corpus {key}')
    manifest=json.loads((root/config['corpus']['manifest']['path']).read_text())
    for name,info in manifest['files'].items():
        p=root/config['corpus']['root']/name
        if p.stat().st_size!=info['bytes'] or sha(p)!=info['sha256']: raise ValueError(f'Changed data {p}')
    configs=json.loads((root/'experiment_v1/configs/config_set_manifest.json').read_text())
    for name,digest in configs['files'].items():
        if sha(root/name)!=digest: raise ValueError(f'Changed config {name}')
    return dict(corpus_manifest=sha(root/config['corpus']['manifest']['path']),
                configs=configs['combined_sha256'],verified_files=len(manifest['files']))


def environment(out):
    out=Path(out)
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (out/'requirements.lock.txt').write_text(freeze)
    try: smi=subprocess.check_output(['nvidia-smi'],text=True)
    except FileNotFoundError: smi='unavailable (CPU environment)'
    (out/'nvidia-smi.txt').write_text(smi)
    info=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
              torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
              cudnn=torch.backends.cudnn.version(),disk_free_bytes=shutil.disk_usage(out).free,
              ram_bytes=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),
              gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              vram_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
              lock_sha256=sha(out/'requirements.lock.txt'))
    info['environment_id']=hashlib.sha256(json.dumps({k:info[k] for k in ('python','platform','torch','cuda','cudnn','gpu','lock_sha256')},sort_keys=True).encode()).hexdigest()[:16]
    return info
