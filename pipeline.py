"""Portable verification of the final experiment's review package (stdlib only)."""
from pathlib import Path
from collections import defaultdict
import argparse
import ast
import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(name):
    return json.loads((ROOT / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(name):
    with gzip.open(ROOT / 'results' / (name + '.gz'), 'rt', encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f)


def core_table(claims):
    lines = ['| Tool | k | BA: block 0 | BA: block 3 | BA: block 7 | BA: block 11 | Answer flip rate: block 11 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for tool, k in itertools.product(('sae', 'tc'), (4, 16)):
        values = [claims[f'iid_{tool}_k{k}_l{layer}']['value'] for layer in (0, 3, 7, 11)]
        values.append(claims[f'flip_{tool}_k{k}_l11']['value'])
        lines.append(f'| {tool.upper()} | {k} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    return '\n'.join(lines)


def verify():
    manifest = read_json('provenance/package_manifest.json')['files']
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), 'Manifest path escapes package')
        require(path.is_file() and digest(path.read_bytes()) == expected, 'Package hash mismatch: ' + name)

    completion = read_json('provenance/completion.json')
    require(completion['experiment_complete'] and completion['status'] == 'passed', 'Original experiment incomplete')
    table_counts = {}
    for path in sorted((ROOT / 'results').glob('*.csv.gz')):
        name = path.name[:-3]
        with gzip.open(path, 'rb') as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
            checksum = h.hexdigest()
        require(checksum == completion['artifacts_sha256'][name], 'Original table hash mismatch: ' + name)
        table_counts[name] = sum(1 for _ in rows(name))
    for name, expected in [('behavior.csv', 3945), ('semantic_metrics.csv', 58086),
                           ('causal_metrics.csv', 152064), ('fidelity_metrics.csv', 2436)]:
        require(table_counts[name] == expected, 'Result coverage: ' + name)

    registry = list(rows('dictionary_registry.csv'))
    expected = {(str(lm), str(layer), tool, str(k), '0', 'READ')
                for lm, layer, tool, k in itertools.product(range(3), (0, 3, 7, 11), ('sae', 'tc'), (4, 16))}
    expected |= {('0', str(layer), tool, str(k), '1', 'READ')
                 for layer, tool, k in itertools.product((0, 3, 7, 11), ('sae', 'tc'), (4, 16))}
    expected |= {(str(lm), '3', tool, str(k), '0', 'update')
                 for lm, tool, k in itertools.product(range(3), ('sae', 'tc'), (4, 16))}
    def identity(r):
        position = r['position_type']
        return (r['lm_seed'], r['layer'], r['tool'], r['k'], r['sparse_seed'],
                'READ' if position.upper() == 'READ' else position.lower())
    require(len(registry) == 76 and {identity(r) for r in registry} == expected, 'Dictionary coverage mismatch')
    require(all(r['status'] == 'passed' and int(r['updates']) == 5000 and
                int(r['position_draws']) == 2560000 for r in registry), 'Run completion mismatch')

    evidence = read_json('provenance/manuscript_evidence.json')
    require(digest((ROOT / 'provenance/completion.json').read_bytes()) == evidence['source_completion_sha256'],
            'Evidence completion anchor mismatch')
    claims = evidence['claims']
    wanted = defaultdict(lambda: defaultdict(list))
    for key, claim in claims.items():
        if 'csv_row' in claim:
            wanted[Path(claim['file']).name][claim['csv_row']].append((key, claim))
    verified_claims = 0
    for name, indices in wanted.items():
        found = set()
        for index, row in enumerate(rows(name), 2):
            for key, claim in indices.get(index, []):
                require(all(row.get(k) == str(v) for k, v in claim['filters'].items()), 'Claim filters: ' + key)
                require(digest(json.dumps(row, sort_keys=True).encode()) == claim['row_sha256'], 'Claim row: ' + key)
                require(math.isclose(float(row[claim['column']]), claim['value'], rel_tol=1e-12, abs_tol=1e-12),
                        'Claim value: ' + key)
                found.add(key)
        require(len(found) == sum(map(len, indices.values())), 'Missing evidence rows: ' + name)
        verified_claims += len(found)
    for key, claim in claims.items():
        if 'components' in claim:
            values = [claims[k]['value'] for k in claim['components']]
            require(math.isclose(statistics.mean(values), claim['value'], abs_tol=1e-12), 'Claim mean: ' + key)
            require(min(values) == claim['minimum'] and max(values) == claim['maximum'], 'Claim range: ' + key)
            verified_claims += 1

    for path in (ROOT / 'code').rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    text = (ROOT / 'README.md').read_text()
    table = text.split('<!-- CORE_TABLE_START -->')[1].split('<!-- CORE_TABLE_END -->')[0].strip()
    require(table == core_table(claims), 'README table differs from verified claims')
    return dict(status='passed', packaged_files=len(manifest), result_tables=table_counts,
                verified_claims=verified_claims, read_dictionaries=64, update_dictionaries=12,
                scope='Package integrity and frozen-result recomputation; no training or model inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'summary'))
    args = parser.parse_args()
    report = verify()
    if args.action == 'verify':
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(core_table(read_json('provenance/manuscript_evidence.json')['claims']))


if __name__ == '__main__':
    main()
