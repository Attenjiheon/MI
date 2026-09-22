"""Publish consistent run snapshots; a partial Drive copy never replaces LATEST."""
import json
import uuid
from pathlib import Path
from .runtime import sha, verified_copy


def publish(output, persistent):
    if persistent is None:
        return
    output, persistent = Path(output), Path(persistent)
    snapshot = persistent / 'snapshots' / uuid.uuid4().hex
    snapshot.mkdir(parents=True)
    files = {}
    for source in sorted(output.iterdir()):
        if source.is_file() and not source.name.endswith('.tmp'):
            verified_copy(source, snapshot / source.name)
            files[source.name] = sha(source)
    index = snapshot / 'snapshot.json'
    index.write_text(json.dumps({'files': files}, indent=2))
    pointer = snapshot / 'pointer.json'
    pointer.write_text(json.dumps({'snapshot': snapshot.name, 'index_sha256': sha(index)}))
    verified_copy(pointer, persistent / 'LATEST.json')


def recover(persistent, output):
    persistent, output = Path(persistent), Path(output)
    if output.exists():
        raise FileExistsError('Restore into a new local run directory; preserve the old directory')
    pointer = json.loads((persistent / 'LATEST.json').read_text())
    name = pointer['snapshot']
    if Path(name).name != name:
        raise ValueError('Invalid snapshot path')
    snapshot = persistent / 'snapshots' / name
    index = snapshot / 'snapshot.json'
    if sha(index) != pointer['index_sha256']:
        raise ValueError('Snapshot index checksum mismatch')
    files = json.loads(index.read_text())['files']
    for name, digest in files.items():
        if Path(name).name != name or sha(snapshot / name) != digest:
            raise ValueError('Snapshot file checksum mismatch')
    output.mkdir(parents=True)
    for name in files:
        verified_copy(snapshot / name, output / name)
    return output / ('last.pt' if 'last.pt' in files else 'init.pt')
