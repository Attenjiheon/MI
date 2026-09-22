"""P3 v1.2: fresh seed 0, fixed 16M budget, immutable checkpoints and strict resume."""
import argparse
import copy
import json
import math
from pathlib import Path
import shlex
import sys
import time
from datetime import datetime,timezone
import torch
from .runtime import deterministic,verify_inputs,sha,save,restore,environment,rng_state,restore_rng
from .model import Transformer
from .training import Progress,lm_optimizer,lm_update
from .data import token_rows,metadata
from .behavior import GATE_DIAGNOSTICS,evaluate,gate
from .persistence import atomic_json,publish,read_index

MILESTONES=(1_000_000,3_000_000,8_000_000,16_000_000)


def metric_summary(value):
    return {k:v for k,v in value.items() if k!='rows'}


def evaluate_gate_suite(model,root,microbatch):
    data=Path(root)/'data/language_v1_2'
    general=evaluate(model,metadata(data/'val_iid.jsonl.gz'),microbatch=microbatch)
    diagnostics={name:evaluate(model,metadata(data/f'diagnostics/val_{name}.jsonl.gz'),diagnostic=True,microbatch=microbatch) for name in GATE_DIAGNOSTICS}
    # Validate quotas at every evaluation, even though the gate is only final.
    gate(general,diagnostics)
    return dict(general=metric_summary(general),diagnostics={name:metric_summary(value) for name,value in diagnostics.items()})


def validate_progress(state,rows):
    if state.next_data_cursor!=64*state.update or not 0<=state.next_data_cursor<=len(rows): raise ValueError('Invalid cursor/update')
    tokens=sum(len(row)-1 for row in rows[:state.next_data_cursor])
    if tokens!=state.prediction_tokens: raise ValueError('Invalid token cursor')
    if state.next_validation_boundary!=(tokens//100000+1)*100000: raise ValueError('Invalid evaluation boundary')


def choose_best(best,current):
    value=current['validation']['general']['answer_ce']
    if not math.isfinite(value): raise ValueError('Nonfinite validation')
    return current if best is None or value<best['validation']['general']['answer_ce'] else best


def decide(best,state,budget,debug=False):
    if debug: return dict(decision='not_adjudicated_debug')
    if state.prediction_tokens<budget: raise ValueError('Cannot adjudicate incomplete budget')
    val=best['validation']
    if set(val['diagnostics'])!=set(GATE_DIAGNOSTICS): raise ValueError('Missing diagnostics')
    if any(x['answer_count']!=512 or x['sequence_count']!=512 for x in val['diagnostics'].values()): raise ValueError('Diagnostic quota')
    passed=val['general']['correct']>=.99 and all(x['correct']>=.95 for x in val['diagnostics'].values())
    return dict(decision='passed' if passed else 'failed',nominal_budget=budget,selected=best,actual_tokens=state.prediction_tokens,overshoot=state.prediction_tokens-budget,final_cursor=state.next_data_cursor,final_update=state.update)


def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    return copy.deepcopy(value)


def run(a):
    root=a.root.resolve(); output=a.output.resolve()
    deterministic(0); torch.set_num_threads(2)
    hashes=verify_inputs(root)
    hashes['code']={str(p.relative_to(root)):sha(p) for folder in ('interp_v1_2','corpus') for p in sorted((root/folder).glob('*.py'))}
    hashes['run_settings']=dict(schema='p3-run-v1.2',seed=0,debug=a.debug)
    config=json.loads((root/'experiment_v1_2/configs/run.json').read_text())
    if not a.debug:
        if a.device!='cuda' or not torch.cuda.is_available(): raise ValueError('Production requires Colab CUDA')
        smoke=json.loads(a.smoke_report.read_text()) if a.smoke_report else {}
        if smoke.get('status')!='passed' or smoke.get('scope')!='cuda': raise ValueError('Current v1.2 GPU smoke required')
        if smoke['input_hashes']['configs']!=hashes['configs'] or smoke['input_hashes']['corpus_manifest']!=hashes['corpus_manifest']: raise ValueError('Smoke input mismatch')
        code={k:v for k,v in hashes['code'].items() if k.startswith('interp_v1_2/')}
        if smoke['input_hashes']['code']!=code: raise ValueError('Smoke code mismatch')
    if a.resume:
        index=read_index(output)
        if a.resume.resolve()!=(output/index['checkpoint']).resolve(): raise ValueError('Resume only last complete checkpoint')
        for name,digest in index['files'].items():
            if sha(output/name)!=digest: raise ValueError('Resume artifact checksum mismatch')
        # Recover from Drive into a fresh directory if a local interrupted tail exists.
        extras={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.suffix!='.tmp' and 'indices' not in p.relative_to(output).parts and p.name!='LATEST.json'}-set(index['files'])
        if extras: raise ValueError('Uncommitted local tail; recover persistent index into a fresh directory')
    else:
        if output.exists(): raise FileExistsError(output)
        output.mkdir(parents=True)
        if a.persistent_dir and (a.persistent_dir/'LATEST.json').exists(): raise FileExistsError('Persistent run exists; recover it')
    session=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    env_dir=output/'sessions'/session; env=environment(env_dir)
    atomic_json(env_dir/'environment.json',dict(environment=env,command=shlex.join(sys.argv),resume=str(a.resume),started_at=session),immutable=True)
    if not a.debug and smoke['environment']['environment_id']!=env['environment_id']: raise ValueError('GPU smoke environment mismatch')
    model=Transformer().to(a.device); opt=lm_optimizer(model); state=Progress()
    records=json.loads((root/'experiment_v1_2/debug/sequences.json').read_text()) if a.debug else None
    rows=[r['token_ids'] for r in records] if a.debug else list(token_rows((root/config['data_root']/'train_shards').glob('*.tokens.jsonl')))
    budget=config['budget']; best=None; microbatch=16; milestones={}
    measurement=dict(updates=0,prediction_tokens=0,training_seconds=0.,peak_memory_bytes=0)
    if a.resume:
        payload=restore(a.resume,model,opt,hashes); state=Progress(**payload['state']); validate_progress(state,rows)
        best=payload['best']; microbatch=payload['microbatch']; measurement=payload['measurement']; milestones=payload['milestones']
    elif not a.debug:
        frozen=config['frozen_training']
        if len(rows)!=frozen['final_cursor'] or sum(len(r)-1 for r in rows)!=frozen['actual_prediction_tokens']: raise ValueError('Train budget mismatch')
    # Init is immutable and may resume before the first validation.
    def checkpoint(name):
        path=output/name
        if path.exists(): raise FileExistsError('Immutable checkpoint: '+name)
        return save(path,model,opt,state.payload(),hashes=hashes,debug_only=a.debug,best=best,microbatch=microbatch,measurement=measurement,milestones=milestones,environment_id=env['environment_id'])
    if not a.resume:
        checkpoint('checkpoints/init.pt'); publish(output,a.persistent_dir,'checkpoints/init.pt')
    last_checkpoint=read_index(output)['checkpoint']
    last_saved=time.monotonic(); started=time.monotonic()
    debug_end=3
    while state.prediction_tokens<budget and (not a.debug or state.update<debug_end):
        chunk=rows[state.next_data_cursor:state.next_data_cursor+64]
        if len(chunk)!=64: raise ValueError('Insufficient data for full update')
        before=state.payload(); rng_before=rng_state()
        model_before=cpu_copy(model.state_dict()); optimizer_before=cpu_copy(opt.state_dict())
        while True:
            try:
                if a.device=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                tick=time.monotonic(); log=lm_update(model,opt,chunk,state,microbatch)
                if a.device=='cuda': torch.cuda.synchronize()
                elapsed=time.monotonic()-tick
                break
            except torch.cuda.OutOfMemoryError:
                if microbatch==1: raise
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                model.load_state_dict(model_before); opt.load_state_dict(optimizer_before); state=Progress(**before); restore_rng(rng_before)
                microbatch//=2
        if not math.isfinite(log['loss']): raise ValueError('Nonfinite loss')
        if state.update<=50:
            measurement['updates']+=1; measurement['prediction_tokens']+=log['tokens']; measurement['training_seconds']+=elapsed
            if a.device=='cuda': measurement['peak_memory_bytes']=max(measurement['peak_memory_bytes'],torch.cuda.max_memory_allocated())
        final=state.prediction_tokens>=budget or (a.debug and state.update==debug_end)
        pause=a.pause_after_updates is not None and state.update>=a.pause_after_updates and not final
        validation_due=log['validation_due'] or final or (a.debug and state.update==2)
        checkpoint_due=validation_due or pause or time.monotonic()-last_saved>=900
        checkpoint_name=f'checkpoints/update_{state.update:06d}.pt'
        log.update(state=state.payload(),microbatch=microbatch,training_seconds=elapsed,environment_id=env['environment_id'])
        if validation_due:
            tick=time.monotonic()
            while True:
                try:
                    validation=dict(general=metric_summary(evaluate(model,records[:32],microbatch=microbatch)),diagnostics={}) if a.debug else evaluate_gate_suite(model,root,microbatch)
                    break
                except torch.cuda.OutOfMemoryError:
                    if microbatch==1: raise
                    torch.cuda.empty_cache(); microbatch//=2
            log['validation_seconds']=time.monotonic()-tick; log['validation']=validation
            current=dict(checkpoint=checkpoint_name,update=state.update,prediction_tokens=state.prediction_tokens,validation=validation)
            best=choose_best(best,current); state.best_validation=best['validation']['general']['answer_ce']
            for boundary in MILESTONES:
                if before['prediction_tokens']<boundary<=state.prediction_tokens: milestones[str(boundary)]=dict(current=current,best=best)
        log['state']=state.payload(); log['microbatch']=microbatch
        if not math.isfinite(log['state']['best_validation']): log['state']['best_validation']=None
        atomic_json(output/f'events/update_{state.update:06d}.json',log,immutable=True)
        if checkpoint_due:
            checkpoint(checkpoint_name); last_checkpoint=checkpoint_name
            # Checkpoint hashes are stored outside checkpoints to avoid self-reference.
            for boundary,value in milestones.items():
                path=output/f'milestones/{boundary}.json'
                if not path.exists():
                    stamped=copy.deepcopy(value)
                    for ref in stamped.values(): ref['sha256']=sha(output/ref['checkpoint'])
                    atomic_json(path,stamped,immutable=True)
            publish(output,a.persistent_dir,last_checkpoint); last_saved=time.monotonic()
        if validation_due or state.update==50:
            print(json.dumps(dict(update=state.update,prediction_tokens=state.prediction_tokens,validation=log.get('validation'),microbatch=microbatch)),flush=True)
        if pause:
            print('PAUSED at complete update',state.update,flush=True); return
    if not a.debug:
        frozen=config['frozen_training']
        if (state.update,state.next_data_cursor,state.prediction_tokens)!=(frozen['final_update'],frozen['final_cursor'],frozen['actual_prediction_tokens']): raise ValueError('Final frozen cursor mismatch')
    best=copy.deepcopy(best); best['sha256']=sha(output/best['checkpoint'])
    decision=decide(best,state,budget,a.debug)
    manifest=dict(phase='P3',version='v1.2',status='debug_only' if a.debug else decision['decision'],debug_only=a.debug,hashes=hashes,environment=env,seed=0,parameter_count=sum(p.numel() for p in model.parameters()),nominal_budget=budget,final_state=state.payload(),gate=decision,selected=best,last_checkpoint=last_checkpoint,last_checkpoint_sha256=sha(output/last_checkpoint),measurement_first_50_updates=measurement,session_seconds=time.monotonic()-started,completed_at=datetime.now(timezone.utc).isoformat())
    atomic_json(output/'result.json',manifest,immutable=True)
    publish(output,a.persistent_dir,last_checkpoint)
    print(json.dumps(dict(status=manifest['status'],selected_update=best['update'],actual_tokens=state.prediction_tokens)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.')); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu'); parser.add_argument('--persistent-dir',type=Path)
    parser.add_argument('--resume',type=Path); parser.add_argument('--smoke-report',type=Path)
    parser.add_argument('--debug',action='store_true'); parser.add_argument('--pause-after-updates',type=int)
    a=parser.parse_args()
    if (a.output/'result.json').exists(): raise FileExistsError('Run already finalized')
    try:
        run(a)
    except BaseException as error:
        if a.output.exists():
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            resource_stop=isinstance(error,(KeyboardInterrupt,torch.cuda.OutOfMemoryError)) or (isinstance(error,OSError) and error.errno==28)
            atomic_json(a.output/'interruptions'/f'{stamp}.json',dict(status='paused' if resource_stop else 'failed',error_type=type(error).__name__,reason=str(error),resume_policy='Recover last completed persistent index into a fresh local directory; no partial optimizer state is saved'),immutable=True)
        raise

if __name__=='__main__': main()
