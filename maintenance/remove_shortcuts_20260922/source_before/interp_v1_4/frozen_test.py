"""One-time P4 final reporting. No training, checkpoint selection, or test gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from .behavior import (metadata, evaluate_records, pair_summary, _target_row, without_rows, valid_pair_coverage)
from .reporting import event_context, extended_summary, interval
from .model import Transformer, batch
from .runtime import deterministic, environment, sha, seed
from .persistence import atomic_json

FREEZE = 'experiment_v1_4/results/p4_audit_20260921_01/validation_freeze.json'
FREEZE_SHA = '9772cc4f4ae0467ce0343a1aa74cbcf70c2511cf562d5e9471649ab574f33b30'
CONTRACT = 'experiment_v1_4/frozen_test_r1/contract.json'
SUITES = [dict(id='general',path='test/general.jsonl.gz',kind='records',target_only=False,sequence_count=2048)] + [
    dict(id=name,path=f'test/legacy_diagnostics/{name}.jsonl.gz',kind='records',target_only=True,sequence_count=2048)
    for name in ('other_variable','repeated_update','first_read_after_set')] + [
    dict(id='first_repeat',path='first_repeat/test.jsonl.gz',kind='pairs',pairs_per_cell=128)] + [
    dict(id=f'composition_{i}',path=f'test/composition/pattern_{i}.jsonl.gz',kind='records',target_only=True,sequence_count=1024)
    for i in (0,1)]


def target_signature(rows, pair=False):
    keys = sorted(([r['pair_id'],r['member'],r['answer']] if pair else
                   [r['sequence_id'],r['read_id'],r['answer']]) for r in rows)
    if len({tuple(k[:2]) for k in keys}) != len(keys):
        raise ValueError('Duplicate evaluated target')
    return hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()


def expected_targets(path, spec):
    rows=[]
    for record in metadata(path):
        if spec['kind']=='pairs':
            for member in ('first','repeat'):
                example=record['origin' if member=='first' else 'repeat']
                rid=record['origin_target_read_id' if member=='first' else 'repeat_target_read_id']
                rows.append(dict(pair_id=record['pair_id'],member=member,answer=example['read_events'][rid]['answer']))
        else:
            for e in record['read_events']:
                if not spec['target_only'] or e['read_id'] in record['target_read_ids']:
                    rows.append(dict(sequence_id=record['sequence_id'],read_id=e['read_id'],answer=e['answer']))
    return dict(target_signature=target_signature(rows,spec['kind']=='pairs'),answer_count=len(rows))


def check_inputs(root):
    if sha(root/FREEZE)!=FREEZE_SHA:raise ValueError('Changed validation freeze')
    freeze=json.loads((root/FREEZE).read_text())
    assert freeze['status']=='frozen_before_test' and freeze['test_scores_observed'] is False
    assert [s['lm_seed'] for s in freeze['seeds']]==[0,1,2]
    contract=json.loads((root/CONTRACT).read_text())
    assert contract['validation_freeze_sha256']==FREEZE_SHA and contract['microbatch']==16
    assert [{k:v for k,v in s.items() if k not in ('target_signature','answer_count','input_sha256')} for s in contract['suites']]==SUITES
    for name,digest in contract['files'].items():
        if sha(root/name)!=digest:raise ValueError('Changed frozen test input: '+name)
    corpus_path=root/contract['data_root']/'manifest.json'
    assert sha(corpus_path)==freeze['seeds'][0]['hashes']['corpus_manifest']
    corpus=json.loads(corpus_path.read_text())
    for spec in contract['suites']:
        name=contract['data_root']+'/'+spec['path']
        assert contract['files'][name]==spec['input_sha256']==corpus['files'][spec['path']]['sha256']
    for name,digest in freeze['seeds'][1]['hashes']['code'].items():
        assert sha(root/name)==digest, 'Changed original numerical/runtime source: '+name
    for item in freeze['seeds']:
        assert sha(root/item['audit_completion'])==item['audit_completion_sha256']
        assert contract['files'][f"experiment_v1_4/frozen_test_r1/checkpoints/seed{item['lm_seed']}.pt"]==item['selected_checkpoint_sha256']
    return contract,freeze


@torch.no_grad()
def evaluate_pairs(model, path, microbatch):
    model.eval();device=next(model.parameters()).device
    members=[(p,m) for p in metadata(path) for m in ('first','repeat')]
    rows=[];totals={m:dict(ce_sum=0.0,prediction_tokens=0) for m in ('first','repeat')}
    for offset in range(0,len(members),microbatch):
        chunk=members[offset:offset+microbatch]
        examples=[p['origin' if m=='first' else 'repeat'] for p,m in chunk]
        ids,mask,targets=batch([e['token_ids'] for e in examples],device)
        logits=model(ids,mask)
        for i,(p,m) in enumerate(chunk):
            rid=p['origin_target_read_id' if m=='first' else 'repeat_target_read_id'];e=examples[i]['read_events'][rid]
            totals[m]['ce_sum']+=float(F.cross_entropy(logits[i],targets[i],ignore_index=0,reduction='sum'))
            totals[m]['prediction_tokens']+=int((targets[i]!=0).sum())
            context=event_context(examples[i],e)
            assert context['operator']==p['operator']
            rows.append(_target_row(logits[i,e['query_token_index']],e,**context,pair_id=p['pair_id'],
                        member=m,cell_id=p['cell_id'],depth_bin=p['depth_bin'],input_pattern=p['input_pattern'],
                        sequence_id=examples[i]['sequence_id'],read_id=rid))
    return dict(rows=rows,token_totals=totals)


def cluster_intervals(rows, key):
    sums=defaultdict(lambda:np.zeros(5,dtype=float))
    for r in rows:
        sums[r['sequence_id']]+=np.array([r['correct'],r['answer_ce'],r['binary_correct'],r['bit_mass'],1.0])
    values=np.stack([sums[k] for k in sorted(sums)])
    rng_seed=seed('frozen-test-sequence-bootstrap-r1',key)
    rng=np.random.Generator(np.random.PCG64(rng_seed));draws=[]
    for _ in range(10):
        sampled=values[rng.integers(len(values),size=(100,len(values)))].sum(axis=1)
        draws.append(sampled[:,:4]/sampled[:,4,None])
    values=np.concatenate(draws)
    return dict(method='sequence-cluster percentile bootstrap; target-weighted metrics',draws=1000,valid_draws=1000,
                seed=rng_seed,**{name+'_ci95':interval(values[:,i]) for i,name in enumerate(('accuracy','answer_ce','binary_accuracy','bit_mass'))})


def finalize(raw,spec):
    if target_signature(raw['rows'],spec['kind']=='pairs')!=spec['target_signature']:
        raise ValueError('Wrong or incomplete evaluated targets')
    assert len(raw['rows'])==spec['answer_count']
    if spec['kind']=='pairs':
        value=pair_summary(raw['rows']);assert valid_pair_coverage(value,128)
        for m in ('first','repeat'):
            value[m]['extended']=extended_summary([r for r in raw['rows'] if r['member']==m])
            totals=raw['token_totals'][m]
            value[m]['all_token_ce']=totals['ce_sum']/totals['prediction_tokens']
            value[m]['prediction_tokens']=totals['prediction_tokens']
    else:
        from .behavior import summarize
        rows=raw['rows'];value=dict(**summarize(rows),**extended_summary(rows),
            all_token_ce=raw['all_token_ce'],prediction_tokens=raw['prediction_tokens'],uncertainty=cluster_intervals(rows,spec['id']))
        assert value['sequence_count']==spec['sequence_count']
    return without_rows(value)


def claim(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(record,f,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def evaluate_once(folder,binding,evaluate,reduce):
    """Persist a claim before inference; a saved raw result resumes reporting only."""
    folder.mkdir(parents=True,exist_ok=True)
    started=folder/'started.json';raw_path=folder/'measurements.json';result=folder/'result.json'
    if started.exists():
        start=json.loads(started.read_text())
        if start['binding']!=binding:raise ValueError('Existing evaluation binding mismatch')
        if result.exists():
            value=json.loads(result.read_text())
            assert value['binding']==binding and value['measurements_sha256']==sha(raw_path)
            return value
        if not raw_path.exists():raise RuntimeError('Evaluation started without saved measurements; recover evidence, never silently repeat inference')
    else:
        if raw_path.exists() or result.exists():raise ValueError('Orphan evaluation artifacts')
        claim(started,dict(binding=binding,started_at=datetime.now(timezone.utc).isoformat()))
        tick=time.monotonic();measurements=evaluate()
        atomic_json(raw_path,dict(binding=binding,metrics=measurements,inference_seconds=time.monotonic()-tick),immutable=True)
    raw=json.loads(raw_path.read_text());assert raw['binding']==binding
    value=dict(binding=binding,measurements_sha256=sha(raw_path),inference_seconds=raw['inference_seconds'],metrics=reduce(raw['metrics']))
    atomic_json(result,value,immutable=True)
    return value


def preflight(root,out,device):
    contract,_=check_inputs(root)
    deterministic(31415);torch.set_num_threads(2)
    env=environment(out)
    model=Transformer('deepwide12').to(device)
    records=json.loads((root/'experiment_v1_2/debug/sequences.json').read_text())[:4]
    raw=evaluate_records(model,records,microbatch=16)
    spec=dict(id='debug',kind='records',sequence_count=4,answer_count=len(raw['rows']),target_signature=target_signature(raw['rows']))
    summary=finalize(raw,spec)
    pair_path=root/'experiment_v1_4/frozen_test_r1/debug_pairs.jsonl.gz'
    paired=evaluate_pairs(model,pair_path,16)
    original=pair_summary(paired['rows']);assert valid_pair_coverage(original,1)
    assert len(paired['rows'])==84 and all(t['prediction_tokens']>0 for t in paired['token_totals'].values())
    # Artificial token fixture, never a trained checkpoint or production test record.
    with torch.no_grad():
        ids=torch.ones((16,302),dtype=torch.long,device=device)
        logits=model(ids,torch.ones_like(ids,dtype=torch.bool))
        assert torch.isfinite(logits).all() and tuple(logits.shape)==(16,302,15)
    report=dict(status='passed',scope=device,debug_only=True,test_scores_observed=False,contract_sha256=sha(root/CONTRACT),
                environment=env,record_targets=summary['answer_count'],pair_targets=len(paired['rows']),microbatch=16,
                maximum_length_fixture=302)
    atomic_json(out/'preflight.json',report,immutable=True)
    return report


def run(root,persistent,smoke_path):
    if not torch.cuda.is_available():raise ValueError('Frozen test production requires CUDA')
    contract,freeze=check_inputs(root)
    smoke=json.loads(smoke_path.read_text());assert smoke['status']=='passed' and smoke['scope']=='cuda'
    assert smoke['contract_sha256']==sha(root/CONTRACT) and smoke['test_scores_observed'] is False
    session=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    env=environment(persistent/'sessions'/session)
    assert env['environment_id']==smoke['environment']['environment_id']
    atomic_json(persistent/'sessions'/session/'environment.json',env,immutable=True)
    deterministic(0);torch.set_num_threads(2)
    results={}
    for item in freeze['seeds']:
        lm_seed=item['lm_seed'];deterministic(lm_seed)
        path=root/f'experiment_v1_4/frozen_test_r1/checkpoints/seed{lm_seed}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['hashes']==item['hashes'] and not payload['debug_only']
        assert payload['state']['update']==item['selected_update'] and payload['microbatch']==16
        model=Transformer('deepwide12').to('cuda');model.load_state_dict(payload['model']);del payload
        for spec in contract['suites']:
            binding=dict(contract_sha256=sha(root/CONTRACT),validation_freeze_sha256=FREEZE_SHA,lm_seed=lm_seed,
                         checkpoint_sha256=item['selected_checkpoint_sha256'],suite=spec['id'],input_sha256=spec['input_sha256'],
                         environment_id=env['environment_id'],microbatch=16)
            data=root/contract['data_root']/spec['path']
            def evaluate():
                if spec['kind']=='pairs':return evaluate_pairs(model,data,16)
                return evaluate_records(model,metadata(data),target_only=spec['target_only'],microbatch=16)
            value=evaluate_once(persistent/f'seed{lm_seed}'/spec['id'],binding,evaluate,lambda raw:finalize(raw,spec))
            results[f"seed{lm_seed}/{spec['id']}"]=dict(result_sha256=sha(persistent/f'seed{lm_seed}'/spec['id']/'result.json'),binding=binding)
            print(json.dumps(dict(seed=lm_seed,suite=spec['id'],status='reported')),flush=True)
        del model;torch.cuda.empty_cache()
    summary=dict(status='completed_test_reporting',contract_sha256=sha(root/CONTRACT),validation_freeze_sha256=FREEZE_SHA,
                 results=results,test_is_selection_or_gate=False,p4_complete=False,local_return_audit_required=True)
    path=persistent/'completion.json'
    if path.exists():assert json.loads(path.read_text())==summary
    else:atomic_json(path,summary,immutable=True)
    return summary


def verify_return(root,folder):
    contract,freeze=check_inputs(root)
    completed=json.loads((folder/'completion.json').read_text())
    assert completed['contract_sha256']==sha(root/CONTRACT) and completed['test_is_selection_or_gate'] is False
    expected={f"seed{s['lm_seed']}/{u['id']}" for s in freeze['seeds'] for u in contract['suites']}
    assert set(completed['results'])==expected
    env_ids=set()
    for s in freeze['seeds']:
        for spec in contract['suites']:
            name=f"seed{s['lm_seed']}/{spec['id']}";path=folder/name
            record=json.loads((path/'result.json').read_text());b=record['binding'];env_ids.add(b['environment_id'])
            assert b==completed['results'][name]['binding']==json.loads((path/'started.json').read_text())['binding']
            assert b['lm_seed']==s['lm_seed'] and b['checkpoint_sha256']==s['selected_checkpoint_sha256']
            assert b['contract_sha256']==sha(root/CONTRACT) and b['input_sha256']==spec['input_sha256'] and b['suite']==spec['id'] and b['microbatch']==16
            assert b['validation_freeze_sha256']==FREEZE_SHA
            assert sha(path/'result.json')==completed['results'][name]['result_sha256']
            assert sha(path/'measurements.json')==record['measurements_sha256']
            raw=json.loads((path/'measurements.json').read_text());assert raw['binding']==b
            assert finalize(raw['metrics'],spec)==record['metrics']
    for env_id in env_ids:
        matches=[p for p in (folder/'sessions').glob('*/environment.json') if json.loads(p.read_text())['environment_id']==env_id]
        assert matches
        for p in matches:
            env=json.loads(p.read_text());assert sha(p.parent/'requirements.lock.txt')==env['lock_sha256']
            identity={k:env[k] for k in ('python','platform','torch','cuda','cudnn','gpu','lock_sha256')}
            assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]==env_id
    return dict(status='passed_recorded_test_verification',units=len(expected),model_inference_repeated=False,
                validation_freeze_sha256=FREEZE_SHA,contract_sha256=sha(root/CONTRACT),p4_complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','run','verify'])
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--smoke',type=Path)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='preflight':report=preflight(root,a.output,a.device)
    elif a.action=='run':report=run(root,a.output,a.smoke)
    else:report=verify_return(root,a.output)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
