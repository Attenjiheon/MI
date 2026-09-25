"""Actual returned token/position controls, no production activation required."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts import p5_independent_audit as a
import numpy as np
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'experiment_v1_4/results/p5_metadata_audit_20260924_01/returned'
c=a.read(SOURCE/'contract.json');labels=a.independent_labels(ROOT,c)
for s in labels:assert labels[s]==a.read(SOURCE/'labels'/f'{s}.json')
values={}
for s,rows in labels.items():
 pos=np.array([r['token_index']/767 for r in rows]);values[s]=np.column_stack([np.eye(15)[[r['token_id'] for r in rows]],pos,pos**2])
start=time.monotonic();items={}
for name in c['tasks']:
 if '_token_position_' not in name:continue
 p=SOURCE/'probes'/f'{name}.json';v=a.verify_task(name,a.read(p),values,labels,c)
 a.save(OUT/'local_token_receipts'/f'{name}.json',dict(source_sha256=a.sha(p),script_sha256=a.sha(a.__file__),checks=v,status='passed'))
 items[name]=a.sha(p);print(name,'passed',flush=True)
a.save(OUT/'local_token_verification.json',dict(status='passed_token_controls_only',tasks=len(items),items=items,seconds=time.monotonic()-start,
 labels={s:len(rows) for s,rows in labels.items()},production_activation_audit_executed=False,p5_complete=False))
