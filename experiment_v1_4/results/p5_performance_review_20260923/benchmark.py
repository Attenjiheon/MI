"""Synthetic CPU microbenchmark only; no experiment train/test activations or fitting."""
import json,os,platform,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info,threadpool_limits
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from interp_v1_4.probe import objective
from interp_v1_4.p5_probe import metrics
rng=np.random.default_rng(91234);x=rng.normal(size=(50000,256));y=np.arange(50000)%16
theta=np.zeros(16*257);results=[]
for threads in (1,2,4):
 with threadpool_limits(limits=threads):
  objective(theta,x,y,16,.1)
  start=time.perf_counter()
  for _ in range(3):objective(theta,x,y,16,.1)
  results.append(dict(operation='16-class objective + gradient',threads=threads,seconds=(time.perf_counter()-start)/3))
vy=np.arange(10000)%2;p=rng.random(10000)
with threadpool_limits(limits=1):
 start=time.perf_counter()
 old=[metrics(vy,p,2,t)['balanced_accuracy'] for t in np.arange(1,20)/20]
 old_seconds=time.perf_counter()-start
 start=time.perf_counter();new=[]
 for t in np.arange(1,20)/20:
  pred=p>=t;cm=np.bincount(vy*2+pred,minlength=4).reshape(2,2)
  new.append(float(np.mean(cm.diagonal()/cm.sum(1))))
 new_seconds=time.perf_counter()-start
 assert old==new
print(json.dumps(dict(platform=platform.platform(),cpu_count=os.cpu_count(),threadpools=threadpool_info(),synthetic_shape=[50000,256],objective_results=results,threshold_comparison=dict(original_seconds=old_seconds,ba_only_seconds=new_seconds,exact_equal=True),production_data_used=False),indent=2))
