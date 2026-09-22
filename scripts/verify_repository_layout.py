"""Verify archive aliases and immutable inputs without running model inference.

Use --full to rehash the pre-migration inventory (including local evidence).
The default check only needs the tracked contracts and source-document snapshot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'maintenance/repository_cleanup_20260922'


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(full=False):
    migration = json.loads((RECORD / 'migration.json').read_text())
    errors = []
    checked = {'aliases': 0, 'snapshot_documents': 0, 'contract_hashes': 0,
               'local_links': 0, 'preserved_files': 0, 'preserved_bytes': 0}
    for original, archived in migration['moves'].items():
        old, new = ROOT / original, ROOT / archived
        if not old.is_symlink() or not new.is_dir() or old.resolve() != new.resolve():
            errors.append(f'Invalid compatibility alias: {original}')
        checked['aliases'] += 1
    for original, record in migration['source_documents'].items():
        if sha256(ROOT / record['snapshot']) != record['sha256']:
            errors.append(f'Changed source snapshot: {original}')
        checked['snapshot_documents'] += 1
    for name in ('design_manifest.json', 'configs/config_set_manifest.json'):
        manifest = json.loads((ROOT / 'experiment_v1_4' / name).read_text())
        groups = [('files', False), ('source_documents', True)]
        for group, use_snapshot in groups:
            for filename, expected in manifest.get(group, {}).items():
                target = (ROOT / migration['source_documents'][filename]['snapshot']
                          if use_snapshot else ROOT / filename)
                if sha256(target) != expected:
                    errors.append(f'Contract hash mismatch: {filename}')
                checked['contract_hashes'] += 1
    docs = [ROOT / name for name in migration['source_documents']]
    docs += [ROOT / 'archive/README.md', ROOT / 'experiment_v1_4/CURRENT.md', RECORD / 'REPORT.md']
    for doc in docs:
        if not doc.exists():
            errors.append(f'Missing documentation: {doc.relative_to(ROOT)}')
            continue
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            filename = unquote(target.split('#', 1)[0])
            if not (doc.parent / filename).exists():
                errors.append(f'Broken local link: {doc.relative_to(ROOT)} -> {target}')
            checked['local_links'] += 1
    if full:
        for line in (RECORD / 'preserved_files.jsonl').read_text().splitlines():
            item = json.loads(line)
            path = ROOT / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                errors.append(f'Changed or missing preserved file: {item["path"]}')
            checked['preserved_files'] += 1
            checked['preserved_bytes'] += item['bytes']
    return {'schema': 'repository-layout-verification-v1', 'passed': not errors,
            'full_inventory': full, 'checked': checked, 'errors': errors,
            'model_inference_or_gate_test_evaluation': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Rehash all preserved local files')
    parser.add_argument('--output', type=Path, help='Write a new verification JSON; never overwrite')
    args = parser.parse_args()
    result = verify(args.full)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(rendered)
    print(rendered, end='')
    raise SystemExit(0 if result['passed'] else 1)
