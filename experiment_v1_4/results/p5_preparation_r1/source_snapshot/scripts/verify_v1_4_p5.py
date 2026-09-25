"""Checksum-first P5 return inspection; never re-runs model or selects on test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import CONTRACT,read,write,verify,audit_cache,load_split
from interp_v1_4.runtime import sha


def unpack(archive,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
        manifest=json.loads(z.read('return_manifest.json'))
        if set(names)!=set(manifest['files'])|{'return_manifest.json'}:raise ValueError('Inventory mismatch')
        for name,digest in manifest['files'].items():
            path=destination/name
            if not path.resolve().is_relative_to(destination.resolve()):raise ValueError('Unsafe archive path')
            path.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256()
            with z.open(name) as src,path.open('xb') as dst:
                for chunk in iter(lambda:src.read(1024*1024),b''):h.update(chunk);dst.write(chunk)
            if h.hexdigest()!=digest:raise ValueError('Return checksum mismatch: '+name)
    return manifest


def inspect(root,run,report):
    root,run=Path(root),Path(run);config=verify(root);ch=sha(root/CONTRACT)
    if read(run/'contract.json')!=config:raise ValueError('Wrong returned contract')
    for split in config['quotas']:
        _,expected=load_split(root,config,split)
        if read(run/'labels'/f'{split}.json')!=expected:raise ValueError('Returned metadata/order mismatch')
    cache=audit_cache(run,ch)
    expected=config['tasks'];actual={p.stem:p for p in (run/'probes').glob('*.json')}
    if set(actual)-set(expected):raise ValueError('Unexpected probe tasks')
    counts={'passed':0,'NA':0,'failed':0};failures=[];na=[]
    for name,path in actual.items():
        item=read(path);task=expected[name]
        if item['config_sha256']!=ch or item['task_key']!=task['key']:raise ValueError('Probe identity mismatch')
        for key in ('bootstrap_seed','shuffle_seed'):
            if item.get(key)!=task.get(key):raise ValueError('Probe RNG mismatch')
        result=item['result'];status=result['status'];counts[status]+=1
        if status=='failed':failures.append(name)
        if status=='NA':
            if not any(v['positions']<32 or v['sequences']<16 for rows in result['support'].values() for v in rows):raise ValueError('Unjustified NA')
            na.append(dict(task=name,reason=result['reason'],support=result['support']))
        if status=='passed':
            if not result.get('trace') or not result.get('evaluation'):raise ValueError('Missing fitting/evaluation evidence')
            if any(not t['attempts'][-1]['success'] for t in result['trace']):raise ValueError('Unreported optimizer failure')
            for fit in result['selected'].values():
                if fit['lam'] not in config['probe']['lambdas']:raise ValueError('Non-frozen lambda')
                if fit['classes']==2 and fit['threshold'] not in config['probe']['thresholds']:raise ValueError('Non-frozen threshold')
    missing=sorted(set(expected)-set(actual))
    write(report,dict(status='passed_structural_return_checks' if not missing and not failures else 'incomplete_P5',
        config_sha256=ch,cache=cache,expected_probe_tasks=len(expected),counts=counts,missing=missing,failures=failures,NA=na,
        p5_complete=False,next_action='Independently audit selection coefficients, train-only statistics, bootstrap and summary tables before P5 completion; do not rerun model inference.'))
    print(read(report)['status'],counts,'missing',len(missing))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--root',default=str(ROOT));p.add_argument('--report',required=True)
    p.add_argument('--archive');a=p.parse_args()
    if a.archive:unpack(a.archive,a.run)
    inspect(a.root,a.run,a.report)
