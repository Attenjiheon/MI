"""Verify a recovered immutable v1.2 run; optionally re-evaluate selected validation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from interp_v1_2.persistence import read_index,atomic_json
from interp_v1_2.runtime import sha,verify_inputs,deterministic
from interp_v1_2.data import token_rows
from interp_v1_2.training import Progress
from interp_v1_2.cli import choose_best,decide,evaluate_gate_suite,validate_progress
from interp_v1_2.model import Transformer


def verify(run,reevaluate=False,device='cpu'):
    hashes=verify_inputs(ROOT); index=read_index(run)
    for name,digest in index['files'].items():
        if sha(run/name)!=digest: raise ValueError('Checksum mismatch: '+name)
    result=json.loads((run/'result.json').read_text())
    assert result['version']=='v1.2' and not result['debug_only']
    assert result['hashes']['configs']==hashes['configs'] and result['hashes']['corpus_manifest']==hashes['corpus_manifest']
    for name,digest in result['hashes']['code'].items(): assert sha(ROOT/name)==digest,name
    cfg=json.loads((ROOT/'experiment_v1_2/configs/run.json').read_text()); frozen=cfg['frozen_training']
    rows=list(token_rows((ROOT/cfg['data_root']/'train_shards').glob('*.tokens.jsonl')))
    events=[json.loads(p.read_text()) for p in sorted((run/'events').glob('*.json'))]
    assert len(events)==frozen['final_update']; cumulative=0; boundary=100000; best=None; evaluations=[]
    milestones={}
    for update,event in enumerate(events,1):
        previous=cumulative; cumulative+=sum(len(r)-1 for r in rows[(update-1)*64:update*64])
        state=event['state']; assert state['update']==update and state['prediction_tokens']==cumulative and state['next_data_cursor']==update*64
        due=cumulative>=boundary or update==frozen['final_update']
        while cumulative>=boundary: boundary+=100000
        assert state['next_validation_boundary']==boundary
        assert ('validation' in event)==due
        if due:
            name=f'checkpoints/update_{update:06d}.pt'
            current=dict(checkpoint=name,update=update,prediction_tokens=cumulative,validation=event['validation'])
            best=choose_best(best,current); evaluations.append(current)
            payload=torch.load(run/name,map_location='cpu',weights_only=False)
            assert not payload['debug_only'] and payload['hashes']==result['hashes']
            assert payload['state']==state and payload['best']==best
            assert payload['microbatch']==event['microbatch']
            assert 'optimizer' in payload and 'rng_states' in payload
            for threshold in (1000000,3000000,8000000,16000000):
                if previous<threshold<=cumulative:
                    stamped=json.loads((run/f'milestones/{threshold}.json').read_text())
                    for label,reference in (('current',current),('best',best)):
                        expected=dict(reference,sha256=sha(run/reference['checkpoint']))
                        assert stamped[label]==expected
                    milestones[str(threshold)]=stamped
        assert state['best_validation']==(best['validation']['general']['answer_ce'] if best else None)
    state=Progress(**result['final_state']); validate_progress(state,rows)
    assert state.update==frozen['final_update'] and cumulative==frozen['actual_prediction_tokens']
    selected=dict(best,sha256=sha(run/best['checkpoint']))
    assert result['selected']==selected and result['gate']==decide(selected,state,16000000)
    assert index['checkpoint']==result['last_checkpoint']
    assert sha(run/index['checkpoint'])==result['last_checkpoint_sha256']
    init=torch.load(run/'checkpoints/init.pt',map_location='cpu',weights_only=False)
    assert init['state']['update']==0 and init['state']['next_data_cursor']==0 and init['state']['prediction_tokens']==0 and not init['optimizer']['state']
    assert init['hashes']==result['hashes'] and not init['debug_only']
    # All smoke/session evidence must be shipped alongside the run for the local phase audit.
    envs=sorted({e['environment_id'] for e in events})
    checked=dict(status='passed',scope='run_contract_and_retention',decision=result['gate']['decision'],selected=selected,actual_tokens=cumulative,updates=len(events),evaluations=len(evaluations),milestones=milestones,environment_ids=envs,reevaluation=None)
    if reevaluate:
        deterministic(0); torch.set_num_threads(2); model=Transformer().to(device)
        model.load_state_dict(torch.load(run/best['checkpoint'],map_location='cpu',weights_only=False)['model'])
        actual=evaluate_gate_suite(model,ROOT,16)
        # Cross-environment CE rounding is allowed; gate outcomes and target counts must agree.
        for name,expected in [('general',best['validation']['general'])]+list(best['validation']['diagnostics'].items()):
            observed=actual['general'] if name=='general' else actual['diagnostics'][name]
            assert observed['answer_count']==expected['answer_count'] and observed['sequence_count']==expected['sequence_count']
            assert abs(observed['answer_ce']-expected['answer_ce'])<1e-5
            assert abs(observed['correct']-expected['correct'])<1e-8
        checked['reevaluation']=actual
    return checked

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reevaluate',action='store_true'); p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); a=p.parse_args()
    atomic_json(a.output,verify(a.run,a.reevaluate,a.device),immutable=True)
    print('Run evidence verified; local GPU smoke/environment audit still required.')
