"""One-time, explicit-path cleanup authorized 2026-09-21; records precede deletion."""
from pathlib import Path
import hashlib, json, os, zipfile, gzip, shutil, collections
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def run():
 plan=json.loads((OUT/'plan.json').read_text())
 if (OUT/'deleted.jsonl').exists(): raise RuntimeError('Already started; inspect journal instead of rerunning')
 before=shutil.disk_usage(ROOT)._asdict()
 (OUT/'disk_before.json').write_text(json.dumps(before))
 with (OUT/'deleted.jsonl').open('x') as journal:
  for row in plan:
   p=ROOT/row['path']
   assert p.is_file() and not p.is_symlink() and ROOT in p.resolve().parents
   assert p.stat().st_size==row['bytes']
   stat=p.stat();row.update(sha256=sha(p),device=stat.st_dev,inode=stat.st_ino,nlink=stat.st_nlink,allocated_bytes=stat.st_blocks*512)
   if p.suffix=='.zip':
    dest=OUT/'bundle_records'/p.parent.parent.name/p.stem;dest.mkdir(parents=True,exist_ok=True)
    inventory=[]
    with zipfile.ZipFile(p) as z,zipfile.ZipFile(dest/'metadata.zip','x',compression=zipfile.ZIP_DEFLATED) as compact:
     for info in z.infolist():
      inventory.append(dict(name=info.filename,bytes=info.file_size,crc32=info.CRC))
      # Preserve source/config/manifests/notebooks; never load corpus sample contents.
      if info.file_size<=8*1024*1024 and Path(info.filename).suffix in {'.json','.md','.py','.ipynb','.toml','.yaml','.yml','.csv'}:
       compact.writestr(info.filename,z.read(info))
    with gzip.open(dest/'members.json.gz','wt') as f:json.dump(inventory,f)
    row['record_directory']=str(dest.relative_to(ROOT))
   journal.write(json.dumps(row)+'\n');journal.flush();os.fsync(journal.fileno())
   p.unlink()
   print('removed',row['path'],flush=True)
 counts=collections.Counter();sizes=collections.Counter();inodes={}
 for line in (OUT/'deleted.jsonl').read_text().splitlines():
  row=json.loads(line);counts[row['reason']]+=1;sizes[row['reason']]+=row['bytes'];inodes[(row['device'],row['inode'])]=row['allocated_bytes']
 result=dict(files_by_reason=dict(counts),logical_bytes_by_reason=dict(sizes),unique_allocated_bytes_removed=sum(inodes.values()),disk_before=before,disk_after=shutil.disk_usage(ROOT)._asdict())
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':run()
