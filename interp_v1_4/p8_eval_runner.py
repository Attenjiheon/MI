"""Phase-separated P8 TC CLI: selection -> semantics -> GPU patching -> export."""
import argparse
import itertools
from pathlib import Path
import time
import sys
import zipfile
import json
import numpy as np
import torch
from . import p8_evaluation as p7
from .p5 import read,write,session
from .runtime import sha,deterministic
from .p8_causal import load_lm,freeze_bins,causal,replacement,summarize,causal_inputs,causal_unit,forward
from .dictionary import Dictionary


def smoke(root,output,device):
    root,output=Path(root),Path(output);c=p7.verify(root);ch=sha(root/p7.CONTRACT)
    deterministic(707);env=session(output,ch,device);tick=time.monotonic()
    from .model import Transformer
    model=Transformer().to(device).eval().requires_grad_(False)
    suites=causal_inputs(root,c,'val');report={}
    stats={h:dict(mu=torch.full((256,),.2 if h=='u' else -.3,device=device),scale=torch.tensor(.25 if h=='u' else 1.75,device=device)) for h in 'um'}
    for layer,k in itertools.product((0,3,7,11),(4,16)):
        d=Dictionary('transcoder',k,707+layer,width=256).to(device).eval().requires_grad_(False)
        selection=dict(fits={f'{r}_current_iid':dict(status='passed',selected={s:dict(columns=[0] if s=='single' else [0,1,2,3]) for s in ('single','up_to_four')}) for r in ('transcoder','coordinate','random')})
        matching=dict(rules={s:dict(rule=dict(h_edges=[.1,.5,1.,2.],z_edges=[.1,.5,1.,2.])) for s in ('single','up_to_four')})
        R,_=p7.directions(c,dict(lm_seed=0,layer=layer))
        run=dict(layer=layer,run_key=f'debug|layer={layer}')
        checks=[]
        for name,rows in suites.items():
            result=causal_unit(model,d,stats,R,run,rows[0],selection,matching,c)
            assert all(np.isfinite(r['patched_logits']).all() for r in result['rows'])
            controls={r['control'] for r in result['rows']}
            assert {'identity','mean','approximation','full_donor','selected','random_unmatched','coordinate','random_direction_selected','random_direction_pure'}<=controls
            checks.append(dict(suite=name,controls=sorted(controls),rows=len(result['rows'])))
        # Padding equivalence and final-LN hook path are checked with two distinct prefix lengths.
        prefixes=[suites['val_changed_memory'][0]['original_prefix_ids'],suites['val_changed_composition'][0]['original_prefix_ids']]
        logits,_,_=forward(model,prefixes,layer)
        for i,prefix in enumerate(prefixes):
            alone,_,_=forward(model,[prefix],layer);torch.testing.assert_close(logits[i],alone[0],atol=1e-5,rtol=1e-4)
        report[f'{layer}_k{k}']=checks
    write(output/'smoke.json',dict(status='passed_debug_only',device=device,config_sha256=ch,environment_id=env['environment_id'],layers=report,elapsed_seconds=time.monotonic()-tick,p8_complete=False))


def run(root,source,output,action,device='cpu',names=None,smoke_path=None):
    root,source,output=Path(root),Path(source),Path(output);config=p7.source_gate(root,source);ch=sha(root/p7.CONTRACT)
    deterministic(707);env=session(output,ch,device);write(output/'contract.json',config)
    for split in p7.QUOTAS:write(output/'labels'/f'{split}.json',read(source/'labels'/f'{split}.json'))
    if action=='patch':
        if device!='cuda' or not torch.cuda.is_available():raise ValueError('Production patching requires CUDA')
        evidence=read(smoke_path)
        if evidence['config_sha256']!=ch or evidence['device']!='cuda' or evidence['status']!='passed_debug_only' or evidence['environment_id']!=env['environment_id']:raise ValueError('Current-environment GPU smoke required')
    stage_started=time.monotonic()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for entry in config['tcs']:
        name=entry['run']['name']
        if names and name not in names:continue
        try:
            if action=='select':p7.selection(root,source,output,entry,device)
            elif action=='semantic':p7.semantic(root,source,output,entry,device)
            elif action=='patch':
                folder=output/'runs'/name;selection=read(folder/'selection.json')
                if selection['config_sha256']!=ch:raise ValueError('Stale frozen features')
                model=load_lm(root,config,entry['run']['lm_seed'],device);d,stats=p7.model_entry(root,entry,device);R,_=p7.directions(config,entry['run'])
                matching=freeze_bins(root,config,output,entry,model,d,stats,selection)
                replacement(root,config,output,entry,model,d,stats)
                causal(root,config,output,entry,model,d,stats,R,selection,matching)
                summarize(root,config,output,entry);del model,d;torch.cuda.empty_cache()
            print('completed stage',action,name,flush=True)
        except Exception as exc:
            write(output/'failures'/f'{time.time_ns()}.json',dict(run=name,action=action,error=repr(exc),config_sha256=ch,environment_id=env['environment_id'],command=sys.argv))
            raise

    write(output/'stage_records'/f'{action}_{time.time_ns()}.json',dict(action=action,status='passed_requested_runs',runs=names or [e['run']['name'] for e in config['tcs']],elapsed_seconds=time.monotonic()-stage_started,environment_id=env['environment_id'],config_sha256=ch,peak_vram_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,command=sys.argv))


def export(output,archive):
    output,archive=Path(output),Path(archive)
    items={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in items:z.write(output/name,name)
        z.writestr('return_manifest.json',json.dumps(dict(schema='p8-evaluation-return-v1.4-r1',files=items,p8_complete=False)))
    write(archive.with_suffix('.sha256.json'),dict(sha256=sha(archive),bytes=archive.stat().st_size))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['select','semantic','patch','smoke','export','aggregate'])
    parser.add_argument('--root',default='.');parser.add_argument('--source');parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu',choices=['cpu','cuda']);parser.add_argument('--runs',nargs='+');parser.add_argument('--smoke');parser.add_argument('--archive')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='smoke':smoke(args.root,args.output,args.device)
    elif args.action=='export':export(args.output,args.archive)
    elif args.action=='aggregate':
        from .p8_aggregate import aggregate
        aggregate(args.root,args.source,args.output)
    else:run(args.root,args.source,args.output,args.action,args.device,args.runs,args.smoke)

if __name__=='__main__':main()
