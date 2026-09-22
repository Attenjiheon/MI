"""Teacher-forced full vocabulary metrics; target selection is metadata-only."""
import torch
from collections import defaultdict
from .model import batch,ce_sum


GATE_DIAGNOSTICS=('other_variable','repeated_update','first_read_after_set')


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
                u=e['update_events'][r['last_update_id_for_query_var_or_null']] if r['last_update_id_for_query_var_or_null'] is not None else None
                rows.append(dict(sequence_id=e['sequence_id'],read_id=r['read_id'],answer=bit,query_var=r['query_var'],
                    operator=u['op'] if u else 'initialization',blocks=len(e['blocks']),
                    distance='0' if r['updates_since_last_update']==0 else '1-2' if r['updates_since_last_update']<=2 else '3+',
                    truth=u['input_truth_pattern_or_null'] if u else None,
                    truth_table=(u['op']+':'+u['input_truth_pattern_or_null']) if u and u['input_truth_pattern_or_null'] is not None else None,
                    answer_ce=float(-logit.log_softmax(-1)[13+bit]),correct=int(logit.argmax()==13+bit),
                    binary_correct=int(logit[13:15].argmax()==bit),bit_mass=float(probs[13:15].sum())))
    if not rows: raise ValueError('No answer targets')
    strata={}
    domains=dict(operator=['initialization','SET','NOT','AND','OR','XOR'],query_var=list('ABCD'),answer=[0,1],blocks=list(range(8,25)),distance=['0','1-2','3+'],truth=['00','01','10','11'],truth_table=[op+':'+bits for op in ('AND','OR','XOR') for bits in ('00','01','10','11')])
    for key,domain in domains.items():
        groups=defaultdict(list)
        for row in rows:
            if row[key] is not None: groups[str(row[key])].append(row)
        cells={str(value):dict(count=len(groups[str(value)]),accuracy=sum(r['correct'] for r in groups[str(value)])/len(groups[str(value)]) if groups[str(value)] else None) for value in domain}
        nonempty=[v['accuracy'] for v in cells.values() if v['count']]
        strata[key]=dict(cells=cells,macro_accuracy=sum(nonempty)/len(nonempty) if nonempty else None,coverage=len(nonempty)/len(domain))
    return dict(all_token_ce=total_ce/tokens,prediction_tokens=tokens,answer_count=len(rows),sequence_count=len(set(r['sequence_id'] for r in rows)),strata=strata,
                **{k:sum(r[k] for r in rows)/len(rows) for k in ('answer_ce','correct','binary_correct','bit_mass')},rows=rows)


def gate(general,diagnostics):
    required=set(GATE_DIAGNOSTICS)
    if set(diagnostics)!=required: raise ValueError('Missing gate diagnostics')
    if any(len(d['rows'])!=512 or len(set(r['sequence_id'] for r in d['rows']))!=512 for d in diagnostics.values()): raise ValueError('Diagnostic quota')
    return general['correct']>=.99 and all(d['correct']>=.95 for d in diagnostics.values())
