"""Independent small-pool test of the actual TC semantic fidelity stage."""
from pathlib import Path
import tempfile,numpy as np,torch
from unittest.mock import patch
from interp_v1_4 import p8_evaluation as ev
from interp_v1_4.dictionary import Dictionary
from interp_v1_4.p5 import write,read
from interp_v1_4.runtime import sha
with tempfile.TemporaryDirectory() as td:
 root=Path(td);write(root/ev.CONTRACT,{})
 write(root/ev.RECEIPT,dict(config_sha256='debug'))
 d=Dictionary('transcoder',4,173,width=256)
 stats={'u':dict(mu=torch.ones(256)*.4,scale=torch.tensor(.3)),'m':dict(mu=torch.ones(256)*-.2,scale=torch.tensor(2.))}
 rng=np.random.default_rng(21)
 m={s:rng.normal(size=(9,256)).astype('float32') for s in ('train','val','test')}
 u={s:rng.normal(size=(9,256)).astype('float32')*3 for s in m}
 run=dict(name='debug',lm_seed=0,layer=3);entry=dict(run=run)
 out=root/'out';folder=out/'runs/debug'
 write(folder/'selection.json',dict(config_sha256=sha(root/ev.CONTRACT),test_used_for_selection=False,fits={}))
 for s in m:write(root/'source/labels'/f'{s}.json',[])
 with patch.object(ev,'verify',return_value={}),patch.object(ev,'model_entry',return_value=(d,stats)),patch.object(ev,'load_arrays',return_value=m),patch.object(ev,'cache_array',side_effect=lambda source,seed,kind,s,layer,hook,ch:u[s]),patch.object(ev,'directions',return_value=(np.zeros((256,512)),{})):
  ev.semantic(root,root/'source',out,entry)
 report=read(folder/'semantic.json')['fidelity']
 for s in m:
  with torch.no_grad():pred,z=d((torch.tensor(u[s])-.4)/.3)
  target=(m[s].astype('float64')-float(stats['m']['mu'][0]))/2.
  expected=np.mean((pred.numpy()-target)**2)
  assert np.isclose(report[s]['mse'],expected,atol=1e-12,rtol=1e-12)
 print('TC semantic fidelity verified: prediction encodes u; error target is independently normalized m for all three splits')
