"""P2 executable interfaces. Debug runs never qualify as experiment results."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .runtime import deterministic,verify_inputs,sha,save,restore,seed,environment,verified_copy,rng_state,restore_rng
from .model import Transformer,batch
from .training import lm_optimizer,lm_update,Progress
from .data import token_rows,metadata,extract
from .behavior import evaluate
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
    a=p.parse_args(); deterministic(a.seed); torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=False)
    hashes=verify_inputs(a.root); hashes['code']={str(q.relative_to(a.root)):sha(q) for q in sorted((a.root/'interp').glob('*.py'))}
    hashes['run_settings']=dict(command=a.command,seed=a.seed,sparse_seed=a.sparse_seed,kind=a.kind,k=a.k,
                                split=a.split,layer=a.layer,position_type=a.position_type,debug=a.debug)
    for name in ('input','validation','checkpoint','positions'):
        q=getattr(a,name)
        if q is not None: hashes[name]=sha(q)
    result={}; start=time.time()
    if a.command=='train_lm':
        # P2 debug pipeline is executable locally; pilot execution requires the GPU environment.
        if not a.debug and a.device!='cuda': raise ValueError('Production LM requires Colab CUDA')
        model=Transformer().to(a.device); opt=lm_optimizer(model); state=Progress()
        if a.debug:
            examples=json.loads((a.root/'experiment_v1/debug/sequences.json').read_text()); rows=[e['token_ids'] for e in examples]
        else: rows=list(token_rows((a.root/'data/language_v1/train_shards').glob('*.tokens.jsonl')))
        if a.resume:
            payload=restore(a.resume,model,opt,hashes); state=Progress(**payload['state'])
        else: save(a.output/'init.pt',model,opt,state.payload(),hashes=hashes,debug_only=a.debug)
        # Pilot endpoint is 1M; extension/reproduction requires a separately recorded P3 decision.
        budget=1_000_000; last_saved=time.monotonic(); logs=[]
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
                try: log=lm_update(model,opt,chunk,state,a.microbatch); break
                except torch.cuda.OutOfMemoryError:
                    if a.microbatch==1: raise
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    model.load_state_dict(model_before); opt.load_state_dict(optimizer_before)
                    state=Progress(**before); restore_rng(rng_before)
                    a.microbatch//=2
            final=state.prediction_tokens>=budget or (a.debug and state.update==2)
            if log['validation_due'] or final:
                validation=evaluate(model,examples[:32] if a.debug else metadata(a.root/'data/language_v1/val_iid.jsonl.gz'),microbatch=a.microbatch)
                log['validation']={k:v for k,v in validation.items() if k!='rows'}
                if validation['answer_ce']<state.best_validation:
                    state.best_validation=validation['answer_ce']
                    save(a.output/'best.pt',model,opt,state.payload(),hashes=hashes,debug_only=a.debug)
                save(a.output/'last.pt',model,opt,state.payload(),hashes=hashes,debug_only=a.debug); last_saved=time.monotonic()
            elif time.monotonic()-last_saved>=900:
                save(a.output/'last.pt',model,opt,state.payload(),hashes=hashes,debug_only=a.debug); last_saved=time.monotonic()
            log.update(state=state.payload(),microbatch=a.microbatch); logs.append(log)
            with (a.output/'training.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
            if a.persistent_dir and (a.output/'last.pt').exists():
                verified_copy(a.output/'last.pt',a.persistent_dir/'last.pt')
        result=dict(state=state.payload(),actual_tokens=state.prediction_tokens,overshoot=max(0,state.prediction_tokens-budget),
                    gate='not_adjudicated: P3 must evaluate selected checkpoint on three diagnostics',selected_checkpoint=str(a.output/'best.pt'))
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
        key=f'experiment-spec-v1.0__lm-{a.seed}__ckpt-{ck[:12]}__layer-{a.layer}__hook-h-u-m__pos-{a.position_type}__k-{a.k}__sparse-{a.sparse_seed}'
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
    write(a.output/'manifest.json',dict(command=a.command,status='passed',debug_only=a.debug,input_hashes=hashes,
        environment=environment(a.output),elapsed_seconds=time.time()-start,result_path=str(a.output/'result.json')))


if __name__=='__main__': main()
