"""P2 executable interfaces. Debug runs never qualify as experiment results."""
import argparse
import copy
import hashlib
import json
import shlex
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .runtime import deterministic,verify_inputs,sha,save,restore,seed,environment,verified_copy,rng_state,restore_rng
from .model import Transformer,batch
from .persistence import publish
from .training import lm_optimizer,lm_update,Progress
from .data import token_rows,metadata,extract
from .behavior import GATE_DIAGNOSTICS,adjudicate,evaluate
from . import dictionary as sparse
from .probe import fit_probe
from .patching import capture,patch,sparse_patch


def jsonable(x):
    if isinstance(x,(np.ndarray,torch.Tensor)): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    return x


def write(path,obj): Path(path).write_text(json.dumps(jsonable(obj),indent=2,allow_nan=False))


def json_digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def metric_summary(metrics):
    return {k:v for k,v in metrics.items() if k!='rows'}


def evaluate_gate_suite(model,root,microbatch):
    general=evaluate(model,metadata(root/'data/language_v1/val_iid.jsonl.gz'),microbatch=microbatch)
    diagnostics={name:evaluate(model,metadata(root/f'data/language_v1/diagnostics/val_{name}.jsonl.gz'),
                               diagnostic=True,microbatch=microbatch) for name in GATE_DIAGNOSTICS}
    return general,diagnostics


def validation_history(path):
    if not path.exists(): return []
    values=[]
    with path.open() as f:
        for line in f:
            record=json.loads(line)
            if 'validation' in record: values.append(record['validation']['general']['answer_ce'])
    return values


def sync_run_files(output,persistent,names):
    if persistent is None: return
    if output.resolve()==persistent.resolve(): raise ValueError('Persistent directory must differ from output')
    for name in names:
        source=output/name
        if source.exists(): verified_copy(source,persistent/name)


def checkpoint_model(path,device,debug=False):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('debug_only',False) and not debug: raise ValueError('Debug checkpoint cannot enter production')
    model=Transformer().to(device); model.load_state_dict(payload['model']); model.eval()
    return model,payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train_lm','eval_behavior','cache_activations','fit_probes','train_dictionary','evaluate_dictionary','run_patching','aggregate_results'])
    p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--checkpoint',type=Path)
    p.add_argument('--input',type=Path); p.add_argument('--validation',type=Path); p.add_argument('--positions',type=Path)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--sparse-seed',type=int,default=0)
    p.add_argument('--kind',choices=['sae','transcoder'],default='sae'); p.add_argument('--k',type=int,choices=[4,16],default=4)
    p.add_argument('--split',choices=['train','val','test'],default='train'); p.add_argument('--layer',type=int,choices=[0,1],default=0)
    p.add_argument('--position-type',choices=['read','update'],default='read'); p.add_argument('--resume',type=Path)
    p.add_argument('--persistent-dir',type=Path); p.add_argument('--debug',action='store_true')
    p.add_argument('--microbatch',type=int,choices=[1,2,4,8,16],default=16)
    p.add_argument('--prediction-token-budget',type=int,choices=[1_000_000,3_000_000],default=1_000_000)
    p.add_argument('--extension-approved',action='store_true')
    a=p.parse_args(); deterministic(a.seed); torch.set_num_threads(2)
    if a.extension_approved and a.prediction_token_budget!=3_000_000:
        raise ValueError('--extension-approved is valid only for the 3M continuation')
    if a.output.exists():
        if not (a.command=='train_lm' and a.resume): raise FileExistsError(f'Output already exists: {a.output}')
    else: a.output.mkdir(parents=True)
    hashes=verify_inputs(a.root); hashes['code']={str(q.relative_to(a.root)):sha(q) for q in sorted((a.root/'interp_v1_1').glob('*.py'))}
    hashes['run_settings']=dict(command=a.command,seed=a.seed,sparse_seed=a.sparse_seed,kind=a.kind,k=a.k,
                                split=a.split,layer=a.layer,position_type=a.position_type,debug=a.debug)
    for name in ('input','validation','checkpoint','positions'):
        q=getattr(a,name)
        if q is not None: hashes[name]=sha(q)
    result={}; start=time.time(); started_at=datetime.now(timezone.utc).isoformat()
    if a.command=='train_lm':
        # P2 debug pipeline is executable locally; pilot execution requires the GPU environment.
        if not a.debug and a.device!='cuda': raise ValueError('Production LM requires Colab CUDA')
        if not a.debug and a.seed!=0: raise ValueError('P3 supports seed 0 only; P4 requires a frozen budget')
        runtime_environment=environment(a.output)
        session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        sessions=a.output/'sessions.jsonl'
        with sessions.open('a') as f:
            f.write(json.dumps(dict(started_at=started_at,environment=runtime_environment,resume=str(a.resume),command=shlex.join(sys.argv)))+'\n')
        for name in ('requirements.lock.txt','nvidia-smi.txt'):
            if (a.output/name).exists(): verified_copy(a.output/name,a.output/(session_id+'-'+name))
        model=Transformer().to(a.device); opt=lm_optimizer(model); state=Progress()
        if a.debug:
            examples=json.loads((a.root/'experiment_v1_1/debug/sequences.json').read_text()); rows=[e['token_ids'] for e in examples]
        else: rows=list(token_rows((a.root/'data/language_v1/train_shards').glob('*.tokens.jsonl')))
        measurement=dict(updates=0,prediction_tokens=0,training_seconds=0.,peak_memory_bytes=0)
        if a.resume:
            if not a.resume.exists(): raise FileNotFoundError(a.resume)
            if a.resume.name not in ('last.pt','init.pt') or (a.resume.name=='init.pt' and (a.output/'last.pt').exists()):
                raise ValueError('Resume from the last complete checkpoint, never best.pt')
            if a.resume.parent.resolve()!=a.output.resolve(): raise ValueError('Resume checkpoint must be inside the same run directory')
            if a.prediction_token_budget==3_000_000:
                decision_path=a.output/'gate_decision.json'
                if not a.extension_approved or not decision_path.exists() or json.loads(decision_path.read_text()).get('decision')!='extend_to_3m':
                    raise ValueError('3M extension requires the recorded 1M extension decision and --extension-approved')
            payload=restore(a.resume,model,opt,hashes); state=Progress(**payload['state'])
            if a.prediction_token_budget==3_000_000 and state.prediction_tokens<1_000_000:
                raise ValueError('Extension checkpoint must have completed the 1M pilot')
            measurement=payload.get('measurement',measurement)
            a.microbatch=min(a.microbatch,payload.get('microbatch',a.microbatch))
            log_path=a.output/'training.jsonl'
            if log_path.exists():
                entries=[json.loads(line) for line in log_path.read_text().splitlines()]
                if any(e['state']['update']>state.update for e in entries):
                    verified_copy(log_path,a.output/(session_id+'-discarded-training.jsonl'))
                    log_path.write_text(''.join(json.dumps(e)+'\n' for e in entries if e['state']['update']<=state.update))
            if state.prediction_tokens!=sum(len(row)-1 for row in rows[:state.next_data_cursor]) or state.next_data_cursor!=64*state.update:
                raise ValueError('Checkpoint cursor/token/update mismatch')
        else: save(a.output/'init.pt',model,opt,state.payload(),hashes=hashes,debug_only=a.debug)
        if not a.debug and a.prediction_token_budget==3_000_000 and not a.resume: raise ValueError('A 3M run must resume the approved 1M pilot')
        publish(a.output,a.persistent_dir)
        budget=1_000_000 if a.debug else a.prediction_token_budget; last_saved=time.monotonic()
        while state.prediction_tokens<budget and (not a.debug or state.update<2):
            chunk=rows[state.next_data_cursor:state.next_data_cursor+64]
            if len(chunk)!=64: raise ValueError('Insufficient corpus for a complete update')
            before=state.payload(); rng_before=rng_state()
            # CPU snapshots also recover a partially executed optimizer step after OOM.
            def cpu_copy(value):
                if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
                if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
                if isinstance(value,list): return [cpu_copy(v) for v in value]
                return copy.deepcopy(value)
            model_before=cpu_copy(model.state_dict()); optimizer_before=cpu_copy(opt.state_dict())
            while True:
                try:
                    if a.device=='cuda': torch.cuda.reset_peak_memory_stats()
                    if a.device=='cuda': torch.cuda.synchronize()
                    update_start=time.monotonic()
                    log=lm_update(model,opt,chunk,state,a.microbatch)
                    if a.device=='cuda': torch.cuda.synchronize()
                    update_seconds=time.monotonic()-update_start
                    break
                except torch.cuda.OutOfMemoryError:
                    if a.microbatch==1: raise
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    model.load_state_dict(model_before); opt.load_state_dict(optimizer_before)
                    state=Progress(**before); restore_rng(rng_before)
                    a.microbatch//=2
            if state.update<=50:
                measurement['updates']+=1; measurement['prediction_tokens']+=log['tokens']; measurement['training_seconds']+=update_seconds
                if a.device=='cuda': measurement['peak_memory_bytes']=max(measurement['peak_memory_bytes'],torch.cuda.max_memory_allocated())
            log['training_seconds']=update_seconds
            checkpoint_due=False
            final=state.prediction_tokens>=budget or (a.debug and state.update==2)
            if log['validation_due'] or final:
                validation_start=time.monotonic()
                if a.debug:
                    general=evaluate(model,examples[:32],microbatch=a.microbatch); diagnostics={}
                else: general,diagnostics=evaluate_gate_suite(model,a.root,a.microbatch)
                log['validation_seconds']=time.monotonic()-validation_start
                log['validation']=dict(general=metric_summary(general),diagnostics={k:metric_summary(v) for k,v in diagnostics.items()})
                if general['answer_ce']<state.best_validation:
                    state.best_validation=general['answer_ce']
                    save(a.output/'best.pt',model,opt,state.payload(),hashes=hashes,debug_only=a.debug,measurement=measurement,microbatch=a.microbatch)
                save(a.output/'last.pt',model,opt,state.payload(),hashes=hashes,debug_only=a.debug,measurement=measurement,microbatch=a.microbatch); last_saved=time.monotonic(); checkpoint_due=True
            elif time.monotonic()-last_saved>=900:
                save(a.output/'last.pt',model,opt,state.payload(),hashes=hashes,debug_only=a.debug,measurement=measurement,microbatch=a.microbatch); last_saved=time.monotonic(); checkpoint_due=True
            log.update(state=state.payload(),microbatch=a.microbatch)
            with (a.output/'training.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
            if checkpoint_due: publish(a.output,a.persistent_dir)
            if log.get('validation') or state.update==50:
                print(json.dumps(dict(update=state.update,tokens=state.prediction_tokens,validation=log.get('validation'),measurement=measurement)),flush=True)
        if a.debug:
            decision=dict(decision='not_adjudicated_debug')
        else:
            selected_model,selected_payload=checkpoint_model(a.output/'best.pt',a.device)
            general,diagnostics=evaluate_gate_suite(selected_model,a.root,a.microbatch)
            history=validation_history(a.output/'training.jsonl')
            decision=adjudicate(general,diagnostics,history,budget)
            decision.update(selected_checkpoint=str(a.output/'best.pt'),selected_checkpoint_sha256=sha(a.output/'best.pt'),
                            selected_update=selected_payload['state']['update'],actual_tokens=state.prediction_tokens,
                            overshoot=max(0,state.prediction_tokens-budget),final_cursor=state.next_data_cursor,final_update=state.update)
        write(a.output/'gate_decision.json',decision)
        write(a.output/f'gate_decision_{budget}.json',decision)
        throughput=measurement['prediction_tokens']/measurement['training_seconds'] if measurement['training_seconds'] else None
        result=dict(state=state.payload(),actual_tokens=state.prediction_tokens,overshoot=max(0,state.prediction_tokens-budget),
                    measurement_first_50_updates=dict(**measurement,tokens_per_second=throughput),gate=decision,
                    parameter_count=sum(p.numel() for p in model.parameters()),microbatch=a.microbatch,
                    selected_checkpoint=str(a.output/'best.pt'))
    elif a.command in ('eval_behavior','cache_activations'):
        model,_=checkpoint_model(a.checkpoint,a.device,a.debug)
        if a.command=='eval_behavior':
            result=evaluate(model,metadata(a.input),diagnostic='diagnostics' in a.input.parts,microbatch=a.microbatch)
        else:
            result=extract(model,metadata(a.input),json.loads(a.positions.read_text()),lm_seed=a.seed,
                checkpoint_sha256=sha(a.checkpoint),split=a.split,position_type=a.position_type,layer=a.layer,microbatch=a.microbatch)
            torch.save(dict(**result,debug_only=a.debug,hashes=hashes),a.output/'cache.pt')
            result=dict(actual_positions=len(result['keys']),cache=str(a.output/'cache.pt'))
    elif a.command=='fit_probes':
        train=np.load(a.input); val=np.load(a.validation)
        result=fit_probe(train['x'],train['y'],train['sequence_id'],val['x'],val['y'],val['sequence_id'],classes=int(train['classes']))
        if result['status']=='passed': np.savez(a.output/'probe.npz',**{k:v for k,v in result.items() if k!='optimizer_failures'})
    elif a.command=='train_dictionary':
        train=torch.load(a.input,weights_only=False,map_location='cpu'); val=torch.load(a.validation,weights_only=False,map_location='cpu')
        if (train.get('debug_only') or val.get('debug_only')) and not a.debug: raise ValueError('Debug cache')
        if not train['keys'] or not val['keys'] or any(k['split']!='train' for k in train['keys']) or any(k['split']!='val' for k in val['keys']): raise ValueError('Expected train and val caches')
        if any(k['lm_seed']!=a.seed or k['layer']!=a.layer or k['position_type']!=a.position_type for k in train['keys']+val['keys']): raise ValueError('Cache run identity mismatch')
        if {k['sequence_id'] for k in train['keys']} & {k['sequence_id'] for k in val['keys']}: raise ValueError('Train/validation sequence overlap')
        ck=train['keys'][0]['checkpoint_sha256']
        if any(k['checkpoint_sha256']!=ck for k in train['keys']+val['keys']): raise ValueError('LM checkpoint mismatch')
        hook=f'blocks.{a.layer}.'; source='resid_post' if a.kind=='sae' else 'mlp_in'; target='resid_post' if a.kind=='sae' else 'mlp_out'
        xraw=train['tensors'][hook+source].to(a.device); yraw=train['tensors'][hook+target].to(a.device)
        ins=sparse.statistics(xraw); outs=sparse.statistics(yraw)
        x=sparse.normalize(xraw,ins); y=sparse.normalize(yraw,outs)
        vx=sparse.normalize(val['tensors'][hook+source].to(a.device),ins); vy=sparse.normalize(val['tensors'][hook+target].to(a.device),outs)
        key=f'experiment-spec-v1.1__lm-{a.seed}__ckpt-{ck[:12]}__layer-{a.layer}__hook-h-u-m__pos-{a.position_type}__k-{a.k}__sparse-{a.sparse_seed}'
        init_seed=seed('dictionary_init',key+'__tool-'+a.kind); draw_seed=seed('position_draw',key)
        sampler=np.random.Generator(np.random.PCG64(draw_seed)); d=sparse.Dictionary(a.kind,a.k,init_seed).to(a.device); opt=sparse.optimizer(d)
        state=dict(update=0,actual_draws=0,best_validation=float('inf'))
        if a.resume:
            saved=restore(a.resume,d,opt,hashes); state=saved['state']; sampler.bit_generator.state=saved['sampler_rng_state']
            for space,st in [('input',ins),('output',outs)]:
                for name,value in st.items(): torch.testing.assert_close(value.cpu(),saved['preprocessing'][space][name].cpu(),rtol=0,atol=0)
        end=100 if a.debug else 5000
        while state['update']<end:
            draw=sampler.integers(len(x),size=512); loss=sparse.update(d,opt,x[draw],y[draw])
            state['update']+=1; state['actual_draws']+=512
            if state['update']%250==0 or state['update']==end:
                with torch.no_grad():
                    mse=sum(float((d(vx[i:i+512])[0]-vy[i:i+512]).square().sum()) for i in range(0,len(vx),512))/(len(vx)*128)
                improved=mse<state['best_validation']
                if improved: state['best_validation']=mse
                extras=dict(hashes=hashes,debug_only=a.debug,kind=a.kind,k=a.k,init_seed=init_seed,draw_seed=draw_seed,
                    sampler_rng_state=sampler.bit_generator.state,preprocessing=dict(input=ins,output=outs),decoder_norms=d.decoder.norm(dim=0))
                save(a.output/'last.pt',d,opt,state,**extras)
                if improved: save(a.output/'best.pt',d,opt,state,**extras)
                if a.persistent_dir: verified_copy(a.output/'last.pt',a.persistent_dir/'last.pt')
                with (a.output/'training.jsonl').open('a') as f: f.write(json.dumps(dict(**state,train_loss=loss,validation_mse=mse))+'\n')
        result=dict(**state,actual_positions=len(x),init_seed=init_seed,draw_seed=draw_seed,draw_key=key,selected_checkpoint=str(a.output/'best.pt'))
    elif a.command=='evaluate_dictionary':
        payload=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
        if payload.get('debug_only') and not a.debug: raise ValueError('Debug dictionary')
        d=sparse.Dictionary(payload['kind'],payload['k'],payload['init_seed']).to(a.device); d.load_state_dict(payload['model'])
        cache=torch.load(a.input,map_location=a.device,weights_only=False); h=f'blocks.{a.layer}.'
        source='resid_post' if payload['kind']=='sae' else 'mlp_in'; target='resid_post' if payload['kind']=='sae' else 'mlp_out'
        x=sparse.normalize(cache['tensors'][h+source],payload['preprocessing']['input']); y=sparse.normalize(cache['tensors'][h+target],payload['preprocessing']['output'])
        with torch.no_grad():
            chunks=[d(x[i:i+512]) for i in range(0,len(x),512)]; pred=torch.cat([v[0] for v in chunks]); z=torch.cat([v[1] for v in chunks])
        result=sparse.fidelity(y,pred,z,torch.zeros(128,device=a.device))
    elif a.command=='run_patching':
        # Explicit patch-plan NPZ is fixed upstream on train/val; this interface applies it without selection.
        model,_=checkpoint_model(a.checkpoint,a.device,a.debug); plan=torch.load(a.input,map_location=a.device,weights_only=False)
        if plan['checkpoint_sha256']!=sha(a.checkpoint): raise ValueError('Patch plan LM mismatch')
        ids,mask,_=batch(plan['prefix_token_ids'],a.device,shift=False)
        with torch.no_grad():
            original=model(ids,mask)
            with patch(model,plan['hook'],plan['positions'],plan['values']): patched=model(ids,mask)
        result=dict(original_logits=[original[i,t].tolist() for i,t in enumerate(plan['positions'])],
                    patched_logits=[patched[i,t].tolist() for i,t in enumerate(plan['positions'])],plan_sha256=sha(a.input))
    else:
        paths=sorted(a.input.glob('*/manifest.json')); result=dict(runs=[json.loads(path.read_text()) for path in paths],source_files=[str(x) for x in paths])
    write(a.output/'result.json',result)
    if a.command=='train_lm': write(a.output/f'result_{budget}.json',result)
    if a.command=='train_lm': sync_run_files(a.output,a.persistent_dir,
        ['init.pt','best.pt','last.pt','training.jsonl','gate_decision.json','result.json'])
    runtime_environment=runtime_environment if a.command=='train_lm' else environment(a.output); elapsed=time.time()-start
    manifest=dict(command=a.command,status='passed',debug_only=a.debug,input_hashes=hashes,
        environment=runtime_environment,elapsed_seconds=elapsed,result_path=str(a.output/'result.json'))
    if a.command=='train_lm' and not a.debug:
        decision=result['gate']; selected_hash=decision['selected_checkpoint_sha256']
        run_status={'passed':'passed','extend_to_3m':'paused','failed':'failed'}[decision['decision']]
        next_action={'passed':'Freeze the P3 budget and proceed to P4',
                     'extend_to_3m':'Resume last.pt once to the cumulative 3M boundary',
                     'failed':'Stop LM training; audit CPU data and implementation, then prepare the P11 stop report'}[decision['decision']]
        manifest.update(
            phase='P3',
            run_id=f'experiment-spec-v1.1__lm-{a.seed}__ckpt-{selected_hash[:12]}__layer-na__hook-na__pos-na__tool-lm__k-na__sparse-na',
            status=run_status,started_at=started_at,ended_at=datetime.now(timezone.utc).isoformat(),
            config_sha256=hashes['configs'],code_sha256=json_digest(hashes['code']),input_sha256=hashes['corpus_manifest'],
            environment_id=runtime_environment['environment_id'],seed=a.seed,derived_seed_keys_and_values={},
            actual_tokens=result['actual_tokens'],actual_positions=None,actual_draws=None,
            peak_memory_bytes=result['measurement_first_50_updates']['peak_memory_bytes'],
            throughput=result['measurement_first_50_updates']['tokens_per_second'],
            evidence_paths=[str(a.output/name) for name in ('gate_decision.json','training.jsonl','best.pt','last.pt','result.json')],
            selected_checkpoint=decision['selected_checkpoint'],checkpoint_sha256=selected_hash,best_update=decision['selected_update'],
            failure_or_skip_reason=None if run_status!='failed' else 'Selected checkpoint failed the frozen P3 behavior gate',
            resume_checkpoint_and_cursor=dict(checkpoint=str(a.output/'last.pt'),cursor=result['state']['next_data_cursor']) if run_status=='paused' else None,
            next_action=next_action,actual_command=shlex.join(sys.argv),nominal_budget=decision['nominal_budget'])
    write(a.output/'manifest.json',manifest)
    if a.command=='train_lm': write(a.output/f'manifest_{budget}.json',manifest)
    if a.command=='train_lm': sync_run_files(a.output,a.persistent_dir,
        ['init.pt','best.pt','last.pt','training.jsonl','gate_decision.json','result.json','manifest.json','requirements.lock.txt','nvidia-smi.txt'])
    if a.command=='train_lm': publish(a.output,a.persistent_dir)


if __name__=='__main__': main()
