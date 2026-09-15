"""Ordered token-only input and separately validated activation metadata joins."""
import gzip
import json
from pathlib import Path
import torch
from .model import batch
from .patching import capture


def token_rows(paths):
    for path in sorted(map(Path,paths)):
        with path.open() as f:
            for line in f: yield json.loads(line)


def metadata(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line in f: yield json.loads(line)


@torch.no_grad()
def extract(model, records, positions, *, lm_seed, checkpoint_sha256, split, position_type='read',layer=0,microbatch=16):
    model.eval(); selected={}; keys=[]; labels=[]; tensors={h:[] for h in ('resid_post','mlp_in','mlp_out')}
    for p in positions:
        key=(p['sequence_id'],p['token_index'])
        if key in selected: raise ValueError('Duplicate position key')
        selected[key]=p
    found=set(); records=list(records)
    for offset in range(0,len(records),microbatch):
        rows=records[offset:offset+microbatch]
        ids,mask,_=batch([e['token_ids'] for e in rows],next(model.parameters()).device)
        with capture(model) as cache: model(ids,mask)
        for i,e in enumerate(rows):
            for event in e[f'{position_type}_events']:
                t=event['query_token_index' if position_type=='read' else 'end_token_index']; key=(e['sequence_id'],t)
                if key not in selected: continue
                if selected[key]['event_id']!=event[f'{position_type}_id']: raise ValueError('Label join mismatch')
                if position_type=='read' and (t+1!=event['answer_token_index'] or e['token_ids'][t+1]!=13+event['answer']): raise ValueError('READ alignment')
                if key in found: raise ValueError('Duplicate sequence or event')
                found.add(key)
                keys.append(dict(lm_seed=lm_seed,checkpoint_sha256=checkpoint_sha256,split=split,
                                 position_type=position_type,layer=layer,sequence_id=key[0],token_index=t))
                labels.append(event)
                for h in tensors: tensors[h].append(cache[f'blocks.{layer}.{h}'][i,t].cpu())
    if found!=set(selected): raise ValueError('Missing selected positions')
    order=sorted(range(len(keys)),key=lambda i:(keys[i]['sequence_id'],keys[i]['token_index']))
    return dict(keys=[keys[i] for i in order],labels=[labels[i] for i in order],
                tensors={f'blocks.{layer}.{h}':torch.stack(v)[order].float() for h,v in tensors.items()})
