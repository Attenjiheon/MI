"""Transport-only P5 multipart export/import; frozen experiment code stays unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pack(root,destination,seed=None,limit=512*1024**2):
    root=Path(root);destination=Path(destination)
    if destination.resolve().is_relative_to(root.resolve()):raise ValueError('Export folder must be outside run folder')
    destination.mkdir(parents=True,exist_ok=True)
    contract=root/'contract.json'
    if not contract.exists():raise ValueError('Missing run contract; select P5_r2 folder')
    files=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name.endswith('.tmp'):continue
        name=p.relative_to(root)
        is_array=name.parts[0]=='cache' and p.suffix=='.npz'
        if seed is None:
            if not is_array:files.append(p)
        elif is_array and name.parts[1] in (f'seed{seed}_trained',f'seed{seed}_init'):
            marker=p.with_suffix('.json')
            if not marker.exists():continue
            info=json.loads(marker.read_text())
            if sha(p)!=info['sha256']:raise ValueError('Corrupt source cache: '+str(p))
            files.extend([p,marker])
    if not files:raise ValueError('No completed files for requested scope')
    # Each part is an independently usable ZIP, not a byte slice of a 15GB archive.
    groups=[];current=[];size=0
    for p in files:
        n=p.stat().st_size
        if current and size+n>limit:groups.append(current);current=[];size=0
        current.append(p);size+=n
    if current:groups.append(current)
    scope='metadata' if seed is None else f'seed{seed}'
    index=dict(schema='p5-transfer-v1',scope=scope,contract_sha256=sha(contract),parts=[])
    for i,group in enumerate(groups):
        path=destination/f'p5_{scope}_{i+1:03d}.zip'
        inventory={str(p.relative_to(root)):sha(p) for p in group}
        manifest=dict(schema='p5-transfer-part-v1',contract_sha256=sha(contract),scope=scope,files=inventory)
        if path.exists():
            with zipfile.ZipFile(path) as z:
                if json.loads(z.read('transfer_manifest.json'))!=manifest:raise ValueError('Existing part differs; use new export folder')
                for name,digest in inventory.items():
                    h=hashlib.sha256()
                    with z.open(name) as f:
                        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                    if h.hexdigest()!=digest:raise ValueError('Corrupt existing part')
        else:
            tmp=path.with_suffix('.zip.tmp')
            with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for p in group:z.write(p,str(p.relative_to(root)))
                z.writestr('transfer_manifest.json',json.dumps(manifest))
            tmp.replace(path)
        index['parts'].append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path),files=len(group)))
        print(path.name,path.stat().st_size,flush=True)
    target=destination/f'p5_{scope}_index.json'
    content=json.dumps(index,indent=2)+'\n'
    if target.exists() and target.read_text()!=content:raise ValueError('Index conflict')
    target.write_text(content)
    return index


def receive(archive,output,expected_contract):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();manifest=json.loads(z.read('transfer_manifest.json'))
        if manifest['contract_sha256']!=expected_contract:raise ValueError('Different experiment contract')
        if len(names)!=len(set(names)) or set(names)!=set(manifest['files'])|{'transfer_manifest.json'}:raise ValueError('Archive inventory mismatch')
        for name,digest in manifest['files'].items():
            dest=output/name
            if not dest.resolve().is_relative_to(output.resolve()):raise ValueError('Unsafe archive path')
            if dest.exists():
                if sha(dest)!=digest:raise ValueError('Existing file conflict: '+name)
                continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix(dest.suffix+'.transfer.tmp')
            try:
                with z.open(name) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                if sha(tmp)!=digest:raise ValueError('Checksum failure: '+name)
                tmp.replace(dest)
            finally:
                if tmp.exists():tmp.unlink()
    return len(manifest['files'])


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('pack');a.add_argument('root');a.add_argument('destination');a.add_argument('--seed',type=int,choices=[0,1,2])
    a=sub.add_parser('receive');a.add_argument('archive');a.add_argument('output');a.add_argument('--contract',default='experiment_v1_4/p5_r2/contract.json')
    args=p.parse_args()
    if args.action=='pack':pack(args.root,args.destination,args.seed)
    else:print('Verified imported files:',receive(args.archive,args.output,sha(args.contract)))
