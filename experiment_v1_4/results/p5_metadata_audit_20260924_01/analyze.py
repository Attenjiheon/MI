"""Read-only metadata audit. Never runs LM, fits probes, or reselects using test."""
import collections,csv,hashlib,json,statistics,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from interp_v1_4.p5 import load_split,domains,LABELS
from scripts.p5_transfer import receive,sha
OUT=Path(__file__).resolve().parent
ARCHIVE=Path('/Users/jangjiheon/Desktop/p5_metadata_001.zip')
INDEX=Path('/Users/jangjiheon/Desktop/p5_metadata_index.json')
index=json.loads(INDEX.read_text());contract_path=ROOT/'experiment_v1_4/p5_r2/contract.json';ch=sha(contract_path)
assert index['contract_sha256']==ch and len(index['parts'])==1
part=index['parts'][0];assert ARCHIVE.stat().st_size==part['bytes'] and sha(ARCHIVE)==part['sha256']
count=receive(ARCHIVE,OUT/'returned',ch);assert count==part['files']
run=OUT/'returned'
def read(p):return json.loads(p.read_text())
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
config=read(contract_path);assert read(run/'contract.json')==config
labels={};groups={};masks={};ys={};support_checks=0
for split in config['quotas']:
 _,expected=load_split(ROOT,config,split);labels[split]=read(run/'labels'/f'{split}.json');assert labels[split]==expected
 groups[split]=np.array([r['sequence_id'] for r in expected])
 for label in LABELS:
  for domain in ('iid','transfer'):
   mask=domains(expected,label,domain,split);masks[split,label,domain]=mask
   ys[split,label,domain]=np.array([r[label] if r[label] is not None else -1 for r in expected])[mask]
cache=collections.defaultdict(lambda:dict(chunks=0,positions=0,seconds=0.,offsets=[]))
for path in sorted((run/'cache').rglob('*.json')):
 item=read(path);i=item['identity'];seed=i['lm_seed'];split=i['split'];kind=path.relative_to(run).parts[1].split('_')[1]
 spec=next(m for m in config['models'] if m['lm_seed']==seed)
 assert i['config_sha256']==ch and i['checkpoint_sha256']==config['files'][spec['checkpoint'] if kind=='trained' else spec['init_checkpoint']]
 assert i['labels_sha256']==sha(run/'labels'/f'{split}.json') and i['position_type']=='READ'
 assert item['layers']==list(range(12)) and item['hooks']==config['hooks']
 ids=sorted(set(r['sequence_id'] for r in labels[split]));selected=set(ids[i['sequence_offset']:i['sequence_offset']+128])
 assert item['positions']==sum(r['sequence_id'] in selected for r in labels[split])
 k=f'{seed}/{kind}/{split}';cache[k]['chunks']+=1;cache[k]['positions']+=item['positions'];cache[k]['seconds']+=item['elapsed_seconds'];cache[k]['offsets'].append(i['sequence_offset'])
for k,v in cache.items():
 split=k.split('/')[-1];assert v['positions']==config['quotas'][split]
 assert v['offsets']==list(range(0,len(set(r['sequence_id'] for r in labels[split])),128))
assert len(cache)==18
for path in run.rglob('environment.json'):
 env=read(path);assert env['config_sha256']==ch and sha(path.parent/'requirements.lock.txt')==env['lock_sha256']
smokes=[]
for path in (run/'preflight').glob('*/preflight.json'):
 s=read(path);assert s['status']=='passed' and s['device']=='cuda' and s['contract_sha256']==ch;smokes.append(s)
assert smokes
probes={p.stem:read(p) for p in (run/'probes').glob('*.json')}
assert set(probes)<=set(config['tasks'])
flat=[];by_layer=collections.Counter();statuses=collections.Counter();failed=[]
for name,item in sorted(probes.items()):
 task=config['tasks'][name];assert item['config_sha256']==ch and item['task_key']==task['key']
 assert item['bootstrap_seed']==task['bootstrap_seed'] and item.get('shuffle_seed')==task.get('shuffle_seed')
 r=item['result'];statuses[r['status']]+=1;by_layer[name.split('_l')[1].split('_')[0] if '_l' in name else 'token_position']+=1
 label,domain=name.rsplit('_',2)[-2:];classes=LABELS[label]
 for split in ('train','val','test'):
  y=ys[split,label,domain].copy();g=groups[split][masks[split,label,domain]]
  if split=='train' and task['shuffle_seed'] is not None:y=np.random.Generator(np.random.PCG64(task['shuffle_seed'])).permutation(y)
  expected=[dict(class_id=c,positions=int((y==c).sum()),sequences=int(len(np.unique(g[y==c])))) for c in range(classes)]
  assert (r['test_support'] if split=='test' else r['support'][split])==expected,(name,split)
  support_checks+=1
 if r['status']!='passed':failed.append(name);continue
 trace=r['trace'];assert all(t['attempts'][-1]['success'] for t in trace)
 prefixes=r['ranking'] is not None
 for size,fit in r['selected'].items():
  options=trace if size!='single' else [t for t in trace if len(t['columns'])==1]
  def key(t):return(-t['validation_balanced_accuracy'],len(t['columns']) if prefixes else 0,t['validation_ce'],-t['lam'],abs((t['threshold'] or .5)-.5),t['threshold'] or 0)
  best=min(options,key=key)
  for field in ('columns','lam','threshold','validation_ce','validation_balanced_accuracy'):assert fit[field]==best[field],(name,field)
  p=len(fit['columns']);assert len(fit['coefficients'])==(p+1)*(1 if classes==2 else classes)
  assert len(fit['mean'])==len(fit['std'])==p and np.isfinite(fit['coefficients']).all() and min(fit['std'])>=1e-8
  score=r['evaluation'][size]
  for subset,s in [('all',score)]+([('current_ne_previous',score['current_ne_previous'])] if 'current_ne_previous' in score else []):
   cm=np.asarray(s['confusion_matrix']);assert cm.shape==(classes,classes) and cm.sum()==s['positions']
   assert cm.sum(1).tolist()==s['class_counts']
   ba=np.mean(cm.diagonal()/cm.sum(1));f1=np.mean(np.divide(2*cm.diagonal(),cm.sum(0)+cm.sum(1),out=np.zeros(classes),where=cm.sum(0)+cm.sum(1)>0))
   assert np.isclose(ba,s['balanced_accuracy']) and np.isclose(f1,s['macro_f1'])
   bs=s['cluster_bootstrap'];assert bs['seed']==task['bootstrap_seed'] and bs['requested']==1000
   for metric,v in bs['metrics'].items():
    assert 0<=v['valid']<=1000
    if v['ci95'] is not None:assert 0<=v['ci95'][0]<=v['ci95'][1]<=1
   flat.append(dict(task=name,size=size,subset=subset,positions=s['positions'],balanced_accuracy=s['balanced_accuracy'],macro_f1=s['macro_f1'],auroc=s['binary_auroc'],ci95=s['cluster_bootstrap']['metrics']['balanced_accuracy']['ci95']))
seconds=[v['elapsed_seconds'] for v in probes.values()]
summary=dict(status='passed_metadata_checks_only',archive_sha256=sha(ARCHIVE),files_verified=count,contract_sha256=ch,
 labels_verified={s:len(r) for s,r in labels.items()},cache_markers=dict(cache),gpu_smoke=smokes,
 total_tasks=len(config['tasks']),returned_tasks=len(probes),remaining_tasks=len(config['tasks'])-len(probes),
 progress_fraction=len(probes)/len(config['tasks']),statuses=dict(statuses),seed_counts=dict(collections.Counter(n.split('_')[0] for n in probes)),layer_counts=dict(by_layer),
 completed_compute_seconds=sum(seconds),median_task_seconds=statistics.median(seconds),mean_task_seconds=statistics.mean(seconds),
 simple_remaining_seconds=statistics.mean(seconds)*(len(config['tasks'])-len(probes)),support_tables_verified=support_checks,
 limitations=['No production NPZ arrays: cannot independently verify cache bytes/row_indices/shapes/finiteness or statistics and coefficients against activations.',
 'Confusion-matrix BA/F1 and recorded validation selection checked; raw predictions and bootstrap/AUROC cannot be independently recomputed.',
 'Snapshot only: absent files may still be running or not yet mirrored; CPU session manifest is copied to Drive only when probes() finishes.',
 'Test scores inspected for reporting only. No configuration/model/threshold/features changed.'],p5_complete=False)
write('analysis.json',summary);write('semantic_snapshot.json',flat);write('missing_tasks.json',sorted(set(config['tasks'])-set(probes)))
with (OUT/'semantic_snapshot.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps({k:v for k,v in summary.items() if k not in ('cache_markers','gpu_smoke','limitations')},indent=2))
