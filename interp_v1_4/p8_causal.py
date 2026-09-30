"""P8 TC single-READ replacement, paired causal controls and committed resume units."""
from pathlib import Path
import os
import time
import numpy as np
import torch
from corpus.replay import replay
from .model import Transformer,batch
from .patching import patch,sparse_patch,coordinate_patch,direction_patch
from .p5 import read,write,derived
from .runtime import sha,deterministic
from .dictionary import normalize,denormalize
from .p7_metrics import bins,random_sets,matched_indices,paired_summary


def commit_arrays(path,arrays,identity):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and valid_arrays(path,identity):return
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    tmp.replace(path);write(path.with_suffix('.json'),dict(identity=identity,sha256=sha(path)))


def valid_arrays(path,identity):
    path=Path(path);marker=path.with_suffix('.json')
    if not marker.exists():
        if path.exists():path.rename(path.with_name(path.name+f'.uncommitted_{time.time_ns()}'))
        return False
    saved=read(marker)
    if saved['identity']!=identity or saved['sha256']!=sha(path):raise ValueError('Resume unit mismatch')
    return True


def load_lm(root,config,seed,device):
    spec=next(m for m in config['models'] if m['lm_seed']==seed)
    p=Path(root)/spec['checkpoint']
    if sha(p)!=spec['checkpoint_sha256']:raise ValueError('Changed frozen LM')
    state=torch.load(p,map_location='cpu',weights_only=False)
    model=Transformer();model.load_state_dict(state['model'])
    return model.to(device).eval().requires_grad_(False)


def forward(model,prefixes,layer,values=None):
    """Only ids and padding mask enter LM forward. Never use a KV cache."""
    device=next(model.parameters()).device;positions=[len(p)-1 for p in prefixes]
    ids,mask,_=batch(prefixes,device,shift=False)
    hook=f'blocks.{layer}.mlp_out';previous=model.interventions;captured={}
    def take(x):
        captured['h']=x[torch.arange(len(prefixes),device=device),torch.tensor(positions,device=device)].detach().clone()
        return x
    try:
        def take_u(x):
            captured['u']=x[torch.arange(len(prefixes),device=device),torch.tensor(positions,device=device)].detach().clone()
            return x
        model.interventions={hook:take,f'blocks.{layer}.mlp_in':take_u}
        with torch.no_grad():
            if values is None:logits=model(ids,mask)
            else:
                with patch(model,hook,positions,values):logits=model(ids,mask)
        logits=logits[torch.arange(len(prefixes),device=device),torch.tensor(positions,device=device)]
        return logits,captured.get('h'),captured.get('u')
    finally:model.interventions=previous


def validate_pair(p):
    o,c=p['original_prefix_ids'],p['counterfactual_prefix_ids'];index=p['modified_token_index']
    differences=[i for i,(a,b) in enumerate(zip(o,c)) if a!=b]
    if len(o)!=len(c) or differences!=[index] or {o[index],c[index]}!={13,14}:raise ValueError('Pair is not a single bit modification')
    if len(o)!=p['target_query_token_index']+1 or p['target_answer_token_index']!=len(o):raise ValueError('Answer/suffix in causal input')
    if o[-2]!=8 or o[-1] not in (9,10,11,12):raise ValueError('Wrong causal READ position')
    parsed_o,parsed_c=replay(o,partial=True),replay(c,partial=True)
    if parsed_o['answer']!=p['original_answer'] or parsed_c['answer']!=p['counterfactual_answer']:raise ValueError('Causal answer replay mismatch')
    if (parsed_o['answer']!=parsed_c['answer'])!=p['changed_target']:raise ValueError('Changed/unchanged mismatch')
    if parsed_o['reads']!=parsed_c['reads']:
        # States may differ; previous answers must all agree.
        if [r['answer'] for r in parsed_o['reads']]!=[r['answer'] for r in parsed_c['reads']]:raise ValueError('Previous answers differ')
    roles=parsed_o['roles']
    if roles[index] not in ('init_literal','update_literal') or o[index-2]!=3:raise ValueError('Modification is not SET')
    var=o[index-1]-9
    logical=[u for u in parsed_o['updates'] if u['start']>index and o[u['start']] in (4,5,6,7) and o[u['start']+1]==9+var]
    if bool(logical)!=(p['condition']=='composition'):raise ValueError('Memory/composition replay mismatch')
    if not p['changed_target'] and o[-1]==9+var:raise ValueError('Unchanged query must be another variable')


def causal_inputs(root,config,split):
    from .p8_evaluation import records
    suites={}
    for changed in ('changed','unchanged'):
        for condition in ('memory','composition'):
            name=f'{split}_{changed}_{condition}'
            rows=records(Path(root)/config['data_root']/'causal_pairs'/f'{name}.jsonl.gz')
            if len(rows)!=config['causal_quotas'][name]:raise ValueError('Causal quota mismatch')
            for p in rows:validate_pair(p)
            suites[name]=rows
    return suites


def main_features(selection,rep,size):
    fit=selection['fits'][f'{rep}_current_iid']
    if fit['status']!='passed':return None
    return fit['selected'][size]['columns']


def freeze_bins(root,config,output,entry,model,d,stats,selection):
    run=entry['run'];folder=Path(output)/'runs'/run['name'];destination=folder/'matching.json'
    identity=dict(config_sha256=sha(Path(root)/'experiment_v1_4/p8_eval_r1/contract.json'),selection_sha256=sha(folder/'selection.json'))
    if destination.exists():
        result=read(destination)
        if result['identity']!=identity:raise ValueError('Stale matching rules')
        return result
    rules={};suites=causal_inputs(root,config,'val')
    for size in ('single','up_to_four'):
        J=main_features(selection,'transcoder',size)
        if J is None:rules[size]=dict(status='NA',reason='current-value probe unavailable');continue
        hn=[];zn=[]
        for name,pairs in suites.items():
            for start in range(0,len(pairs),16):
                chunk=pairs[start:start+16]
                prefixes=[p['original_prefix_ids'] for p in chunk]+[p['counterfactual_prefix_ids'] for p in chunk]
                _,h,u=forward(model,prefixes,run['layer']);z=d.encode(normalize(u,stats['u']));n=len(chunk)
                diff=z[n:,J]-z[:n,J];delta=stats['m']['scale']*(diff@d.decoder[:,J].T)
                a=delta.norm(dim=1).cpu().numpy();b=diff.norm(dim=1).cpu().numpy()
                # Both directions have equal norms; count both in fitting record.
                hn.extend(a.tolist()*2);zn.extend(b.tolist()*2)
        rules[size]=dict(status='passed',features=J,rule=bins(hn,zn),validation_h_norms=hn,validation_z_norms=zn)
    result=dict(identity=identity,rules=rules,test_used_for_bins=False)
    write(destination,result);return result


def causal_scores(original,patched,answer,donor,changed):
    original=np.asarray(original);patched=np.asarray(patched)
    a=13+answer;other=13+(donor if changed else 1-answer)
    margin=lambda v:float(v[other]-v[a])
    return dict(delta_margin=margin(patched)-margin(original),full_flip=float(int(patched.argmax())==13+donor) if changed else 0.,
        binary_flip=float(int(patched[13:15].argmax())==donor) if changed else 0.,
        error_induced=float(original.argmax()==a and patched.argmax()!=a),prediction_changed=float(original.argmax()!=patched.argmax()))


def causal_unit(model,d,stats,R,run,p,selection,matching,config):
    layer=run['layer'];prefixes=[p['original_prefix_ids'],p['counterfactual_prefix_ids']]
    logits,h,u=forward(model,prefixes,layer);z=d.encode(normalize(u,stats['u']));logits=logits.cpu().numpy()
    identity_logits,_,_=forward(model,prefixes,layer,h)
    np.testing.assert_allclose(identity_logits.cpu().numpy(),logits,atol=1e-5,rtol=1e-4)
    answers=[p['original_answer'],p['counterfactual_answer']];both=all(logits[i].argmax()==13+answers[i] for i in (0,1))
    rows=[];candidate_records={}
    for size in ('single','up_to_four'):
        J=main_features(selection,'transcoder',size);coord=main_features(selection,'coordinate',size);random=main_features(selection,'random',size)
        m=len(J) if J is not None else 1
        seed=derived('random_patch',run['run_key']+'|'+p['pair_id']+'|'+size)
        candidates=random_sets(d.latents,J,seed,config['random_candidates']) if J is not None else []
        delta_z=z[1]-z[0];hn=[];zn=[]
        for ids in candidates:
            diff=delta_z[ids];hn.append(float((stats['m']['scale']*(d.decoder[:,ids]@diff)).norm()));zn.append(float(diff.norm()))
        selected_h=selected_z=None;matched=[]
        if J is not None:
            selected_z=float(delta_z[J].norm());selected_h=float((stats['m']['scale']*(d.decoder[:,J]@delta_z[J])).norm())
            matched=matched_indices(selected_h,selected_z,hn,zn,matching['rules'][size]['rule'],config['matched_limit'])
        pure_rng=np.random.Generator(np.random.PCG64(derived('random_patch',run['run_key']+'|directions|'+p['pair_id']+'|'+size)))
        pure=pure_rng.choice(R.shape[1],m,replace=False).tolist()
        candidate_records[size]=dict(seed=seed,features=J,candidates=candidates,h_norms=hn,z_norms=zn,matched_indices=matched,unmatched_indices=[0] if candidates else [],selected_h_norm=selected_h,selected_z_norm=selected_z,pure_direction_features=pure)
        for direction in (0,1):
            donor=1-direction;original=h[direction:direction+1];other=h[donor:donor+1]
            variants=[('identity',original,None),('mean',stats['m']['mu'][None],None),('approximation',denormalize(d(normalize(u[direction:direction+1],stats['u']))[0],stats['m']),None),('full_donor',other,None)]
            if J is not None:
                variants.append(('selected',sparse_patch(original,z[direction:direction+1],z[donor:donor+1],d,J,stats['m']['scale']),J))
                for index in sorted(set([0]+matched)):
                    variants.append((f'random_{index}',sparse_patch(original,z[direction:direction+1],z[donor:donor+1],d,candidates[index],stats['m']['scale']),candidates[index]))
            if coord is not None:variants.append(('coordinate',coordinate_patch(original,other,coord),coord))
            for name,ids in [('random_direction_selected',random),('random_direction_pure',pure)]:
                if ids is not None:variants.append((name,direction_patch(original,other,torch.tensor(R[:,ids],device=h.device,dtype=h.dtype)),ids))
            v=torch.cat([a[1] for a in variants]);out,_,_=forward(model,[prefixes[direction]]*len(variants),layer,v);out=out.cpu().numpy()
            for i,(name,value,ids) in enumerate(variants):
                group='random_matched' if name.startswith('random_') and name[7:].isdigit() and int(name[7:]) in matched else name
                common=dict(origin=p['origin_hash'],pair_id=p['pair_id'],direction=direction,condition=p['condition'],changed=p['changed_target'],both_correct=both,size=size,control=group,
                    original_correct=bool(logits[direction].argmax()==13+answers[direction]),matching_success=bool(matched),patch_norm=float((value-original).norm()),features=ids,original_logits=logits[direction].tolist(),patched_logits=out[i].tolist(),
                    **causal_scores(logits[direction],out[i],answers[direction],answers[donor],p['changed_target']))
                rows.append(common)
                if name=='random_0':rows.append(dict(common,control='random_unmatched'))
                if name=='selected' and matched:rows.append(dict(common,control='selected_matched'))
    return dict(rows=rows,candidates=candidate_records)


def causal(root,config,output,entry,model,d,stats,R,selection,matching):
    run=entry['run'];folder=Path(output)/'runs'/run['name'];ch=sha(Path(root)/'experiment_v1_4/p8_eval_r1/contract.json')
    identity=dict(config_sha256=ch,selection_sha256=sha(folder/'selection.json'),matching_sha256=sha(folder/'matching.json'))
    for name,pairs in causal_inputs(root,config,'test').items():
        for i,p in enumerate(pairs):
            path=folder/'causal'/name/f'{i:05d}.json'
            if path.exists():
                if read(path)['identity']!=identity:raise ValueError('Stale causal result')
                continue
            tick=time.monotonic();result=causal_unit(model,d,stats,R,run,p,selection,matching,config)
            write(path,dict(identity=identity,pair_id=p['pair_id'],elapsed_seconds=time.monotonic()-tick,**result))
            if i%32==0:print(run['name'],name,i+1,'/',len(pairs),flush=True)


def replacement(root,config,output,entry,model,d,stats):
    from .p8_evaluation import records
    folder=Path(output)/'runs'/entry['run']['name'];run=entry['run'];ch=sha(Path(root)/'experiment_v1_4/p8_eval_r1/contract.json')
    source=records(Path(root)/config['data_root']/'test/general.jsonl.gz')
    targets=read(Path(root)/config['reconstruction_targets']);by_id={r['sequence_id']:r for r in source}
    for start in range(0,len(targets),16):
        path=folder/'replacement'/f'{start:05d}.npz'
        if valid_arrays(path,ch):continue
        current=targets[start:start+16];prefixes=[by_id[t['sequence_id']]['token_ids'][:t['token_index']+1] for t in current]
        original,h,u=forward(model,prefixes,run['layer']);same,_,_=forward(model,prefixes,run['layer'],h)
        torch.testing.assert_close(original,same,atol=1e-5,rtol=1e-4)
        reconstructed=denormalize(d(normalize(u,stats['u']))[0],stats['m']);patched,_,_=forward(model,prefixes,run['layer'],reconstructed)
        answers=np.asarray([t['answer'] for t in current]);a=torch.tensor(answers+13,device=original.device)
        ce0=torch.nn.functional.cross_entropy(original,a,reduction='none');ce1=torch.nn.functional.cross_entropy(patched,a,reduction='none')
        commit_arrays(path,dict(original_logits=original.cpu().numpy(),patched_logits=patched.cpu().numpy(),answers=answers,
            original_ce=ce0.cpu().numpy(),patched_ce=ce1.cpu().numpy(),indices=np.arange(start,start+len(current))),ch)
        print(run['name'],'replacement',start+len(current),'/',len(targets),flush=True)


def summarize(root,config,output,entry):
    run=entry['run'];folder=Path(output)/'runs'/run['name'];ch=sha(Path(root)/'experiment_v1_4/p8_eval_r1/contract.json')
    rows=[];counts={};coverage={};missing=[]
    for name,quota in config['causal_quotas'].items():
        if not name.startswith('test_'):continue
        paths=sorted((folder/'causal'/name).glob('*.json'))
        if len(paths)!=quota:raise ValueError('Incomplete causal units')
        counts[name]=len(paths)
        for path in paths:
            saved=read(path);rows.extend(saved['rows'])
            for size,record in saved['candidates'].items():
                key=name+'|'+size;item=coverage.setdefault(key,dict(pairs=0,matched=0,unmatched=0,NA=0))
                item['pairs']+=1
                item['NA']+=int(record['features'] is None)
                item['matched']+=int(bool(record['matched_indices']));item['unmatched']+=int(not record['matched_indices'])
    summaries={};controls=sorted({r['control'] for r in rows if not (r['control'].startswith('random_') and r['control'][7:].isdigit())})
    for changed in (False,True):
        for condition in ('memory','composition','all'):
            for size in ('single','up_to_four'):
                for subset in ('all','both_correct','original_correct'):
                    for control in controls:
                        current=[r for r in rows if r['changed']==changed and (condition=='all' or r['condition']==condition) and r['size']==size and r['control']==control and (subset=='all' or (r['both_correct'] if subset=='both_correct' else r['original_correct']))]
                        key=f'{changed}|{condition}|{size}|{subset}|{control}'
                        bs=derived('bootstrap',config['bootstrap_keys'][run['name']]+'|causal|'+key)
                        summaries[key]=paired_summary(current,['delta_margin','full_flip','binary_flip','error_induced','prediction_changed','patch_norm'],bs)
    targets=read(Path(root)/config['reconstruction_targets']);chunks=[]
    for start in range(0,len(targets),16):
        path=folder/'replacement'/f'{start:05d}.npz'
        if not valid_arrays(path,ch):raise ValueError('Incomplete replacement')
        with np.load(path) as z:
            for i,index in enumerate(z['indices']):
                a=z['answers'][i]+13;original=z['original_logits'][i];patched=z['patched_logits'][i]
                chunks.append(dict(origin=targets[int(index)]['sequence_id'],original_ce=float(z['original_ce'][i]),patched_ce=float(z['patched_ce'][i]),delta_ce=float(z['patched_ce'][i]-z['original_ce'][i]),original_accuracy=float(original.argmax()==a),patched_accuracy=float(patched.argmax()==a),delta_accuracy=float(patched.argmax()==a)-float(original.argmax()==a)))
    report=dict(config_sha256=ch,run=run,causal_pair_counts=counts,coverage=coverage,causal=summaries,
        replacement=paired_summary(chunks,['original_ce','patched_ce','delta_ce','original_accuracy','patched_accuracy','delta_accuracy'],derived('bootstrap',config['bootstrap_keys'][run['name']]+'|replacement')),
        p8_complete=False,status='evaluated_pending_return_audit')
    write(folder/'summary.json',report);return report
