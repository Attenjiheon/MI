"""P5 CPU delivery repair. Does not modify frozen inference/probe source or choices."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


class Repair:
    def __init__(self, local, drive, backup):
        self.local=Path(local);self.drive=Path(drive);self.backup=Path(backup)
        self.log=[]
        self.backup.mkdir(parents=True,exist_ok=True)

    def record(self, **row):
        self.log.append(row)
        with (self.backup/'repair.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def preserve(self,path):
        if not path.exists():return
        side='local' if path.is_relative_to(self.local) else 'drive'
        base=self.local if side=='local' else self.drive
        dst=self.backup/side/path.relative_to(base)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha(dst)!=sha(path):raise ValueError('Backup conflict: '+str(dst))
        else:
            shutil.copyfile(path,dst)
            if sha(dst)!=sha(path):raise ValueError('Backup verification failed')
        self.record(action='preserved',side=side,path=str(path),sha256=sha(dst))

    def put(self,dest,source=None,payload=None,reason='verified copy'):
        expected=sha(source) if source is not None else hashlib.sha256(payload).hexdigest()
        if dest.exists() and sha(dest)==expected:return
        self.preserve(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix(dest.suffix+'.resume.tmp')
        if source is not None:
            shutil.copyfile(source,tmp)
        else:
            with tmp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        if sha(tmp)!=expected:raise ValueError('Incomplete copy: '+str(dest))
        tmp.replace(dest)
        self.record(action='restored',path=str(dest),sha256=expected,reason=reason)

    def reconcile(self,relative,validator):
        a=self.local/relative;b=self.drive/relative
        def valid(p):
            if not p.exists():return False
            try:return bool(validator(p))
            except (ValueError,KeyError,TypeError,OSError):return False
        va,vb=valid(a),valid(b)
        if va and vb:
            if sha(a)!=sha(b):
                # Whitespace-only JSON differences do not change saved experiment results.
                same=False
                if a.suffix=='.json':
                    try:same=json.loads(a.read_text())==json.loads(b.read_text())
                    except ValueError:pass
                if not same:
                    self.preserve(a);self.preserve(b)
                    raise ValueError('Two different valid results; preserved both, no automatic selection: '+str(relative))
                self.put(a,source=b,reason='equal JSON values; normalize to Drive bytes')
        elif vb:self.put(a,source=b)
        elif va:self.put(b,source=a,reason='valid local copy repairs missing/invalid Drive copy')
        else:
            self.preserve(a);self.preserve(b)
            raise ValueError('No verified copy: '+str(relative))


def validate_probe(path,config,ch):
    x=json.loads(path.read_text());task=config['tasks'][path.stem]
    if x['config_sha256']!=ch or x['task_key']!=task['key']:return False
    if x['bootstrap_seed']!=task['bootstrap_seed'] or x.get('shuffle_seed')!=task.get('shuffle_seed'):return False
    r=x['result']
    if r['status'] in ('NA','failed'):
        return bool(r.get('reason')) # Preserve recorded failures; never silently replace by success.
    if r['status']!='passed' or not r.get('trace') or not r.get('selected') or not r.get('evaluation'):return False
    return set(r['selected'])==set(r['evaluation']) and all('coefficients' in f for f in r['selected'].values())


def restore(root,local,drive):
    from interp_v1_4.p5 import verify,load_split,support_table,CONTRACT
    root=Path(root);local=Path(local);drive=Path(drive)
    if local.resolve()==drive.resolve():raise ValueError('Separate local and Drive folders required')
    config=verify(root);ch=sha(root/CONTRACT)
    if not (drive/'contract.json').exists():raise ValueError('Original P5_r2 Drive folder not found')
    # A different experiment root is never repaired into the requested experiment.
    if json.loads((drive/'contract.json').read_text())!=config:raise ValueError('Drive contains another contract')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    repair=Repair(local,drive,drive.parent/'P5_resume_backups'/stamp)
    repair.put(local/'contract.json',payload=canonical(config),reason='frozen input contract')
    repair.put(drive/'contract.json',payload=canonical(config),reason='frozen input contract')
    print('Backup/log:',repair.backup,flush=True)
    # Labels can be independently reconstructed from frozen corpus, without any inference.
    for split in config['quotas']:
        _,rows=load_split(root,config,split);payload=canonical(rows);expected=hashlib.sha256(payload).hexdigest()
        for base in (local,drive):
            for marker in (base/'cache').glob(f'*/{split}/*.json'):
                try:identity=json.loads(marker.read_text())['identity']
                except (ValueError,KeyError):continue
                if identity.get('config_sha256')==ch and identity.get('labels_sha256')!=expected:
                    raise ValueError('Cache expects different labels; no repair performed for '+split)
        for base in (local,drive):
            path=base/'labels'/f'{split}.json'
            if path.exists() and sha(path)!=expected:
                try:equal=json.loads(path.read_text())==rows
                except ValueError:equal=False
                repair.record(action='label_mismatch',path=str(path),actual_sha256=sha(path),expected_sha256=expected,
                              semantic_equal=equal,bytes=path.stat().st_size)
            repair.put(path,payload=payload,reason='independent frozen corpus READ label reconstruction')
            repair.put(base/'labels'/f'{split}.support.json',payload=canonical(support_table(rows)),reason='frozen label support')
        print('Labels verified:',split,len(rows),flush=True)
    def inventory(base):return {p.relative_to(base) for p in base.rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    names=inventory(local)|inventory(drive)
    markers=sorted(n for n in names if n.parts[0]=='cache' and n.suffix=='.json')
    # Process checksum markers before the arrays they authenticate.
    for rel in markers:
        def marker_ok(p):
            x=json.loads(p.read_text());i=x['identity'];seed=i['lm_seed'];kind=rel.parts[1].split('_',1)[1];split=rel.parts[2]
            m=next(m for m in config['models'] if m['lm_seed']==seed)
            cp=m['checkpoint'] if kind=='trained' else m['init_checkpoint']
            return (i['config_sha256']==ch and i['split']==split and i['position_type']=='READ'
                and i['checkpoint_sha256']==config['files'][cp]
                and i['labels_sha256']==sha(local/'labels'/f'{split}.json') and len(x['sha256'])==64)
        repair.reconcile(rel,marker_ok)
        expected=json.loads((local/rel).read_text())['sha256']
        repair.reconcile(rel.with_suffix('.npz'),lambda p:sha(p)==expected)
    print('Cache chunks synchronized:',len(markers),flush=True)
    # Never adopt an uncommitted array with no complete checksum marker.
    for rel in sorted(names):
        if rel.parts[0] in ('cache','labels') or str(rel)=='contract.json':continue
        if rel.parts[0]=='probes':
            repair.reconcile(rel,lambda p:validate_probe(p,config,ch))
        elif rel.suffix=='.json':
            def json_ok(p):
                x=json.loads(p.read_text())
                return isinstance(x,dict) and x.get('config_sha256',ch)==ch
            repair.reconcile(rel,json_ok)
        else:
            # Ancillary locks/debug blobs: copy absent files only; conflicting bytes need inspection.
            a=local/rel;b=drive/rel
            if a.exists() and b.exists() and sha(a)!=sha(b):
                repair.preserve(a);repair.preserve(b);raise ValueError('Ancillary file conflict: '+str(rel))
            repair.reconcile(rel,lambda p:p.stat().st_size>0)
    done=len(list((local/'probes').glob('*.json')))
    repair.record(action='resume_copy_complete',config_sha256=ch,cache_chunks=len(markers),saved_probe_files=done)
    print('Saved probe files:',done,'/',len(config['tasks']),flush=True)
    print('Ready for frozen CPU runner; its full cache audit runs before fitting.',flush=True)
    return str(repair.backup)
