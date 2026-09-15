"""Teacher-forced full vocabulary metrics; target selection is metadata-only."""
import torch
from .model import batch,ce_sum


@torch.no_grad()
def evaluate(model,records,diagnostic=False,microbatch=16):
    model.eval(); total_ce=0.; tokens=0; rows=[]
    records=list(records)
    for offset in range(0,len(records),microbatch):
        examples=records[offset:offset+microbatch]
        ids,mask,target=batch([e['token_ids'] for e in examples],next(model.parameters()).device)
        logits=model(ids,mask); total_ce+=float(ce_sum(logits,target)); tokens+=int((target!=0).sum())
        for i,e in enumerate(examples):
            for r in e['read_events']:
                if diagnostic and r['read_id'] not in e['target_read_ids']: continue
                logit=logits[i,r['query_token_index']]; probs=logit.softmax(-1); bit=r['answer']
                rows.append(dict(sequence_id=e['sequence_id'],read_id=r['read_id'],answer=bit,
                    answer_ce=float(-probs[13+bit].log()),correct=int(logit.argmax()==13+bit),
                    binary_correct=int(logit[13:15].argmax()==bit),bit_mass=float(probs[13:15].sum())))
    if not rows: raise ValueError('No answer targets')
    return dict(all_token_ce=total_ce/tokens,prediction_tokens=tokens,answer_count=len(rows),
                **{k:sum(r[k] for r in rows)/len(rows) for k in ('answer_ce','correct','binary_correct','bit_mass')},rows=rows)


def gate(general,diagnostics):
    required={'other_variable','repeated_update','first_read_after_set'}
    if set(diagnostics)!=required: raise ValueError('Missing gate diagnostics')
    if any(len(set(r['sequence_id'] for r in d['rows']))!=512 for d in diagnostics.values()): raise ValueError('Diagnostic quota')
    return general['correct']>=.99 and all(d['correct']>=.95 for d in diagnostics.values())


def may_extend(validation_history):
    return len(validation_history)>=4 and sum(a-b>=1e-4 for a,b in zip(validation_history[-4:-1],validation_history[-3:]))>=2
