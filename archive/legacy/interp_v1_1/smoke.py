"""P2 integration checks; outputs and checkpoints are DEBUG ONLY."""
import argparse
import copy
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
import torch
from . import dictionary as sparse
from .model import Transformer,batch,ce_sum,rope
from .training import Progress,lm_optimizer,lm_update
from .runtime import deterministic,seed,sha,save,restore,rng_state,environment,verify_inputs,verified_copy
from .patching import capture,patch,sparse_patch
from .probe import fit_probe,objective


def close(a,b,exact=False):
    torch.testing.assert_close(a,b,atol=0 if exact else 1e-5,rtol=0 if exact else 1e-4)


def run(root,out,device):
    root=Path(root).resolve(); out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    start=time.time(); deterministic(20260910); torch.set_num_threads(2)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    inputs=verify_inputs(root)
    # Fixed, independent debug fixtures generated on local CPU and shipped to GPU.
    records=json.loads((root/'experiment_v1_1/debug/sequences.json').read_text())
    sequences=[e['token_ids'] for e in records]
    inputs['debug']=sha(root/'experiment_v1_1/debug/sequences.json')
    inputs['code']={str(p.relative_to(root)):sha(p) for p in sorted((root/'interp_v1_1').glob('*.py'))}
    checks={}; model=Transformer().to(device)
    assert sum(p.numel() for p in model.parameters())==797184
    assert model.embedding.weight.data_ptr()!=model.unembedding.weight.data_ptr()
    checks['lm_parameters']=797184
    ids,mask,targets=batch(sequences[:4],device)
    assert int((targets!=0).sum())==sum(len(s)-1 for s in sequences[:4])
    for i,s in enumerate(sequences[:4]):
        assert ids[i,:len(s)-1].tolist()==s[:-1] and targets[i,:len(s)-1].tolist()==s[1:]
    with torch.no_grad():
        logits=model(ids,mask)
        padding_error=0.
        for i,s in enumerate(sequences[:4]):
            a,b,_=batch([s],device); single=model(a,b)
            close(logits[i,:len(s)-1],single[0]); padding_error=max(padding_error,float((logits[i,:len(s)-1]-single[0]).abs().max()))
        altered=ids.clone(); cutoff=15; altered[:,cutoff:]=(altered[:,cutoff:]+1)%15
        close(logits[:,:cutoff],model(altered,mask)[:,:cutoff],exact=True)
        # PAD key masking independent of padding token embedding.
        altered=ids.clone(); altered[~mask]=14
        close(logits[mask],model(altered,mask)[mask],exact=True)
        # Independent adjacent-pair RoPE check.
        sample=torch.randn(1,4,3,32,device=device); rotated=rope(sample)
        for pos in range(3):
            for j in range(16):
                angle=pos*10000**(-2*j/32)
                close(rotated[...,pos,2*j],sample[...,pos,2*j]*np.cos(angle)-sample[...,pos,2*j+1]*np.sin(angle))
    checks['padding_max_abs_error']=padding_error; checks['causal_and_pad_mask']='passed'; checks['rope']='passed'
    # Compare gradient accumulation against one complete effective batch before clipping.
    subset=sequences[:8]; n=sum(len(s)-1 for s in subset)
    model.zero_grad(); a,b,t=batch(subset,device); (ce_sum(model(a,b),t)/n).backward()
    grads=[p.grad.clone() for p in model.parameters()]; model.zero_grad()
    for i in range(0,8,2):
        a,b,t=batch(subset[i:i+2],device); (ce_sum(model(a,b),t)/n).backward()
    gradient_error=0.
    for g,p in zip(grads,model.parameters()):
        close(g,p.grad); gradient_error=max(gradient_error,float((g-p.grad).abs().max()))
    checks['token_weighted_gradient_max_abs_error']=gradient_error
    opt=lm_optimizer(model); state=Progress(); training=[]
    for offset in (0,64): training.append(lm_update(model,opt,sequences[offset:offset+64],state))
    assert state.next_data_cursor==128 and state.update==2
    checkpoint_hash=save(out/'lm_debug.pt',model,opt,state.payload(),hashes=inputs,debug_only=True)
    verified_copy(out/'lm_debug.pt',out/'persistent_copy/lm_debug.pt')
    next_batch=sequences[state.next_data_cursor:state.next_data_cursor+64]
    expected_log=lm_update(model,opt,next_batch,state); expected=copy.deepcopy(model.state_dict())
    restored=Transformer().to(device); ropt=lm_optimizer(restored)
    payload=restore(out/'lm_debug.pt',restored,ropt,inputs); rs=Progress(**payload['state'])
    actual_log=lm_update(restored,ropt,sequences[rs.next_data_cursor:rs.next_data_cursor+64],rs)
    for name,value in expected.items(): close(value,restored.state_dict()[name],exact=True)
    assert expected_log==actual_log and rs==state
    checks['lm_resume']=dict(status='passed',checkpoint_sha256=checkpoint_hash,next_cursor=rs.next_data_cursor,
                             next_update=rs.update,prediction_tokens=rs.prediction_tokens,next_validation_boundary=rs.next_validation_boundary)
    model=restored.eval(); hpool=[]; upool=[]; mpool=[]; labels=[]; groups=[]
    with torch.no_grad():
        for offset in range(0,64,8):
            a,b,_=batch(sequences[offset:offset+8],device)
            with capture(model) as cache: model(a,b)
            for layer in range(4):
                close(cache[f'blocks.{layer}.resid_post'],cache[f'blocks.{layer}.resid_mid']+cache[f'blocks.{layer}.mlp_out'],exact=True)
                close(cache[f'blocks.{layer}.mlp_in'],model.blocks[layer].ln_mlp(cache[f'blocks.{layer}.resid_mid']),exact=True)
            for row,e in enumerate(records[offset:offset+8]):
                for event in e['read_events']:
                    pos=event['query_token_index']; assert pos+1==event['answer_token_index']
                    assert e['token_ids'][pos-1]==8 and e['token_ids'][pos+1]==13+event['answer']
                    hpool.append(cache['blocks.0.resid_post'][row,pos]); upool.append(cache['blocks.0.mlp_in'][row,pos]); mpool.append(cache['blocks.0.mlp_out'][row,pos])
                    labels.append(event['answer']); groups.append(offset+row)
                for event in e['update_events']:
                    pos=event['end_token_index']; assert pos<len(e['token_ids'])-1
                    assert e['token_ids'][event['start_token_index']] in (3,4,5,6,7)
                    assert event['block_id']>=0
    h,u,m=map(torch.stack,(hpool,upool,mpool)); checks['hooks_read_update']='passed'
    dictionary_results={}; trained={}
    for kind in ('sae','transcoder'):
        for k in (4,16):
            key=f'debug|lm-0|layer-0|read|k-{k}|sparse-0'
            init_seed=seed('dictionary_init',key+'|'+kind); draw_seed=seed('position_draw',key)
            d=sparse.Dictionary(kind,k,init_seed).to(device); do=sparse.optimizer(d)
            assert sum(p.numel() for p in d.parameters())==131712
            if kind=='sae': close(d.encoder,d.decoder.T,exact=True); assert d.encoder.data_ptr()!=d.decoder.data_ptr()
            else: assert not torch.equal(d.encoder,d.decoder.T)
            source=h if kind=='sae' else u; target=h if kind=='sae' else m
            ins=sparse.statistics(source); outs=sparse.statistics(target)
            x=sparse.normalize(source,ins); y=sparse.normalize(target,outs)
            close(sparse.denormalize(y,outs),target)
            sampler=np.random.Generator(np.random.PCG64(draw_seed)); losses=[]
            for step in range(100):
                draw=sampler.integers(len(x),size=512)
                losses.append(sparse.update(d,do,x[draw],y[draw]))
            checkpoint=out/f'{kind}_k{k}_debug.pt'
            save(checkpoint,d,do,dict(update=100,actual_draws=51200),hashes=inputs,
                 sampler_rng_state=sampler.bit_generator.state,preprocessing=dict(input=ins,output=outs),decoder_norms=d.decoder.norm(dim=0),debug_only=True)
            draw=sampler.integers(len(x),size=512); expected_loss=sparse.update(d,do,x[draw],y[draw]); expected=copy.deepcopy(d.state_dict())
            d2=sparse.Dictionary(kind,k,init_seed).to(device); o2=sparse.optimizer(d2)
            p=restore(checkpoint,d2,o2,inputs); sampler2=np.random.Generator(np.random.PCG64()); sampler2.bit_generator.state=p['sampler_rng_state']
            draw2=sampler2.integers(len(x),size=512); assert np.array_equal(draw,draw2)
            assert expected_loss==sparse.update(d2,o2,x[draw2],y[draw2])
            for name,value in expected.items(): close(value,d2.state_dict()[name],exact=True)
            with torch.no_grad():
                pred,z=d2(x); assert (z>0).sum(-1).max()<=k
                close(d2.decoder.norm(dim=0),torch.ones(512,device=device))
                metrics=sparse.fidelity(y,pred,z,y.mean(0)); metrics['dead_train_fraction']=float(((z>0).sum(0)==0).float().mean())
            dictionary_results[f'{kind}_k{k}']=dict(parameters=131712,updates=100,resume_updates=1,
                unique_positions=len(x),draws=51200,resume_draws=512,initial_loss=losses[0],final_loss=losses[-1],
                init_seed=init_seed,draw_seed=draw_seed,draw_key=key,init_key=key+'|'+kind,
                input_scale=float(ins['scale']),output_scale=float(outs['scale']),resume='bitwise_equal',fidelity=metrics)
            trained[kind]=(d2,ins,outs)
    # Positive TopK ties choose smaller latent IDs, not an unstable topk kernel.
    with torch.no_grad():
        tie=sparse.Dictionary('sae',4,1).to(device); tie.encoder.zero_(); tie.encoder_bias.fill_(1)
        assert torch.nonzero(tie.encode(h[:1])[0]).flatten().tolist()==[0,1,2,3]
    checks['dictionaries']=dictionary_results; checks['topk_ties']='passed'
    # Connect actual cached READ activations to supervised probe fitting.
    group_array=np.asarray(groups); label_array=np.asarray(labels)
    train_mask=group_array<32; val_mask=~train_mask; hp=h.cpu().numpy()
    real_probe=fit_probe(hp[train_mask],label_array[train_mask],group_array[train_mask],
                         hp[val_mask],label_array[val_mask],group_array[val_mask])
    assert real_probe['status']=='passed',real_probe
    np.savez(out/'read_probe_debug.npz',**{k:v for k,v in real_probe.items() if k!='optimizer_failures'})
    # Also fit a controlled synthetic signal and check analytical gradients.
    rng=np.random.Generator(np.random.PCG64(20260910)); px=rng.normal(size=(256,8)); py=np.tile([0,1],128)
    px[:,0]+=py*2; vx=rng.normal(size=(128,8)); vy=np.tile([0,1],64); vx[:,0]+=vy*2
    probe=fit_probe(px,py,np.arange(256),vx,vy,np.arange(128))
    assert probe['status']=='passed'
    # Analytical gradient check with central differences, binary and multiclass.
    for classes in (2,4):
        gx=rng.normal(size=(24,3)); gy=np.arange(24)%classes; theta=rng.normal(size=4*(1 if classes==2 else classes))*.1
        _,gradient=objective(theta,gx,gy,classes,.1)
        for j in range(len(theta)):
            plus=theta.copy(); minus=theta.copy(); plus[j]+=1e-6; minus[j]-=1e-6
            numerical=(objective(plus,gx,gy,classes,.1)[0]-objective(minus,gx,gy,classes,.1)[0])/2e-6
            assert abs(gradient[j]-numerical)<1e-7
    np.savez(out/'probe_debug.npz',**{k:v for k,v in probe.items() if k!='optimizer_failures'})
    checks['probe']=dict(status=real_probe['status'],real_read_train_positions=int(train_mask.sum()),
        real_read_validation_positions=int(val_mask.sum()),validation_balanced_accuracy=real_probe['validation_balanced_accuracy'],
        lam=real_probe['lam'],threshold=float(real_probe['threshold']),synthetic_check=probe['status'])
    with torch.no_grad():
        a,b,_=batch(sequences[:2],device); pos=[records[i]['read_events'][0]['query_token_index'] for i in range(2)]
        with capture(model) as c: original_logits=model(a,b)
        for kind,(d,ins,outs) in trained.items():
            hook='blocks.0.resid_post' if kind=='sae' else 'blocks.0.mlp_out'
            input_hook='blocks.0.resid_post' if kind=='sae' else 'blocks.0.mlp_in'
            original=torch.stack([c[hook][i,p] for i,p in enumerate(pos)])
            source=torch.stack([c[input_hook][i,p] for i,p in enumerate(pos)])
            donor=original.flip(0); z=d.encode(sparse.normalize(source,ins))
            features=torch.argsort((z-z.flip(0)).abs().sum(0),descending=True,stable=True)[:4]
            sparse_value=sparse_patch(original,z,z.flip(0),d,features,outs['scale'])
            assert (sparse_value-original).norm()>0
            expected_delta=sum((z.flip(0)[:,j]-z[:,j])[:,None]*d.decoder[:,j][None,:]*outs['scale'] for j in features)
            close(sparse_value-original,expected_delta)
            for mode,value in [('identity',original),('full',donor),('sparse',sparse_value)]:
                with patch(model,hook,pos,value),capture(model) as patched:
                    result=model(a,b)
                assert torch.isfinite(result).all()
                for row,p in enumerate(pos):
                    close(patched[hook][row,p],value[row],exact=True)
                    close(result[row,:p],original_logits[row,:p],exact=True)
                if mode=='identity': close(result,original_logits,exact=True)
                if kind=='transcoder': close(patched['blocks.0.resid_mid'],c['blocks.0.resid_mid'],exact=True)
    checks['identity_full_sparse_patch_sae_tc']='passed'
    info=environment(out)
    result=dict(phase='P2',status='passed',scope=device,debug_only=True,reuse_in_experiment=False,
                input_hashes=inputs,environment=info,checks=checks,lm_training=training,
                actual_tokens=state.prediction_tokens,actual_positions=len(h),actual_draws=4*(51200+512),
                elapsed_seconds=time.time()-start,peak_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    (out/'smoke.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='passed',scope=device,output=str(out),elapsed_seconds=result['elapsed_seconds']),indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='.'); p.add_argument('--output',required=True); p.add_argument('--device',choices=['cpu','cuda'],required=True)
    args=p.parse_args(); run(args.root,args.output,args.device)
