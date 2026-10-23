"""Immutable files plus atomic completion indices; copy each artifact only once."""
import json
import os
from pathlib import Path
from .runtime import sha, verified_copy


def atomic_json(path, value, *, immutable=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists(): raise FileExistsError(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def safe_path(root,name):
    result=(root/name).resolve()
    if not result.is_relative_to(root.resolve()) or name=='LATEST.json': raise ValueError('Invalid indexed path')
    return result


def read_index(root):
    pointer=json.loads((root/'LATEST.json').read_text())
    path=safe_path(root,pointer['index'])
    if sha(path)!=pointer['sha256']: raise ValueError('Index checksum mismatch')
    return json.loads(path.read_text())


def publish(output,persistent,checkpoint):
    """Output files are immutable; only LATEST is a mutable reference."""
    output=Path(output); persistent=Path(persistent) if persistent else None
    if persistent and output.resolve()==persistent.resolve(): raise ValueError('Use separate persistent directory')
    old=read_index(output)['files'] if (output/'LATEST.json').exists() else {}
    files=dict(old)
    for source in sorted(output.rglob('*')):
        name=str(source.relative_to(output))
        if not source.is_file() or source.suffix=='.tmp' or name=='LATEST.json' or name.startswith('indices/'): continue
        if name not in files:
            files[name]=sha(source)
            if persistent:
                dest=safe_path(persistent,name)
                if dest.exists():
                    if sha(dest)!=files[name]: raise ValueError('Conflicting immutable persistent file: '+name)
                else: verified_copy(source,dest)
    if checkpoint not in files: raise ValueError('Checkpoint is not in committed files')
    index=dict(schema='immutable-run-v1.2',checkpoint=checkpoint,files=files)
    import hashlib
    digest=hashlib.sha256(json.dumps(index,sort_keys=True).encode()).hexdigest()
    name='indices/'+digest+'.json'; index_path=output/name
    if not index_path.exists(): atomic_json(index_path,index,immutable=True)
    pointer=dict(index=name,sha256=sha(index_path))
    if persistent:
        dest=persistent/name
        if not dest.exists(): verified_copy(index_path,dest)
        elif sha(dest)!=pointer['sha256']: raise ValueError('Persistent index checksum mismatch')
        atomic_json(persistent/'LATEST.json',pointer)
    atomic_json(output/'LATEST.json',pointer)
    return index


def recover(persistent,output):
    persistent,output=Path(persistent),Path(output)
    if output.exists(): raise FileExistsError('Recover into a new directory')
    index=read_index(persistent)
    for name,digest in index['files'].items():
        if sha(safe_path(persistent,name))!=digest: raise ValueError('Artifact checksum mismatch: '+name)
    output.mkdir(parents=True)
    for name in index['files']: verified_copy(safe_path(persistent,name),safe_path(output,name))
    pointer=json.loads((persistent/'LATEST.json').read_text())
    verified_copy(persistent/pointer['index'],output/pointer['index'])
    atomic_json(output/'LATEST.json',pointer)
    return output/index['checkpoint']
