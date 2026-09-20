import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from interp_v1_4.runtime import verify_inputs,sha
from scripts.verify_v1_4_evidence import verify
from scripts.check_v1_4_resume import equal
import torch
out=Path(__file__).parent
checked=verify_inputs(ROOT)
s=json.loads((ROOT/'experiment_v1_4/smoke/cpu_01/smoke.json').read_text())
assert s['status']=='passed' and s['scope']=='cpu' and s['frozen_input_verified'] and not s['fixture_only']
for key in ('configs','corpus_manifest'):assert s['input_hashes'][key]==checked[key]
for name,h in s['input_hashes']['code'].items():assert sha(ROOT/name)==h,name
folder=ROOT/'experiment_v1_4/smoke/resume_01'
reports={name:verify(folder/name,debug=True) for name in ('full','resumed')}
a=torch.load(folder/'full/checkpoints/update_000003.pt',map_location='cpu',weights_only=False)
b=torch.load(folder/'resumed/checkpoints/update_000003.pt',map_location='cpu',weights_only=False)
for key in ('model','optimizer','state','rng_states','microbatch','next_evaluation_boundary'):equal(a[key],b[key])
report=dict(status='passed',frozen_inputs=checked,cpu_smoke_sha256=sha(ROOT/'experiment_v1_4/smoke/cpu_01/smoke.json'),code_hashes_match=True,resume_bitwise_fields=['model','optimizer','state','rng_states','microbatch','next_evaluation_boundary'],verified_runs=reports,gpu_run=False)
(out/'existing_cpu_verified.json').write_text(json.dumps(report,indent=2)+'\n')
print('frozen inputs, CPU smoke hashes, full/resumed evidence and optimizer/RNG bitwise equality verified')
