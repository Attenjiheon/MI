"""Metadata cross-checks and descriptive analysis of saved test rows only."""
from pathlib import Path
import json,sys,statistics,collections,math
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from interp_v1_4.behavior import metadata
from interp_v1_4.reporting import event_context
OUT=Path(__file__).parent;RUN=OUT/'returned/run'
c=json.loads((ROOT/'experiment_v1_4/frozen_test_r1/contract.json').read_text())
summary={};errors=[];operator_rates={};overlap={};metadata_count=0
for spec in c['suites']:
 expected={};prediction_tokens=collections.Counter()
 for record in metadata(ROOT/c['data_root']/spec['path']):
  if spec['kind']=='pairs':
   members=[(m,record['origin' if m=='first' else 'repeat'],record['origin_target_read_id' if m=='first' else 'repeat_target_read_id']) for m in ('first','repeat')]
   for member,example,rid in members:
    event=example['read_events'][rid]
    expected[(record['pair_id'],member)]=dict(answer=event['answer'],sequence_id=example['sequence_id'],read_id=rid,**event_context(example,event),cell_id=record['cell_id'],depth_bin=record['depth_bin'],input_pattern=record['input_pattern'])
    prediction_tokens[member]+=len(example['token_ids'])-1
  else:
   prediction_tokens['all']+=len(record['token_ids'])-1
   for event in record['read_events']:
    if not spec['target_only'] or event['read_id'] in record['target_read_ids']:
     expected[(record['sequence_id'],event['read_id'])]=dict(answer=event['answer'],**event_context(record,event))
 sets=[]
 for s in range(3):
  result=json.loads((RUN/f"seed{s}/{spec['id']}/result.json").read_text());m=result['metrics']
  raw=json.loads((RUN/f"seed{s}/{spec['id']}/measurements.json").read_text())['metrics'];rows=raw['rows'];keys=set();bad=set()
  for row in rows:
   key=(row['pair_id'],row['member']) if spec['kind']=='pairs' else (row['sequence_id'],row['read_id'])
   assert key not in keys;keys.add(key)
   assert all(row[k]==v for k,v in expected[key].items()),(s,spec['id'],key)
   assert row['correct'] in (0,1) and row['binary_correct'] in (0,1)
   assert math.isfinite(row['answer_ce']) and row['answer_ce']>=0 and 0<=row['bit_mass']<=1.000001
   metadata_count+=1
   if not row['correct']:bad.add(key);errors.append(dict(lm_seed=s,suite=spec['id'],**row))
  assert keys==set(expected)
  sets.append(bad)
  parts=['first','repeat'] if spec['kind']=='pairs' else ['all']
  for part in parts:
   rr=[r for r in rows if part=='all' or r['member']==part]
   if part=='all':v=m;ci=m['uncertainty'];tokens=m['prediction_tokens'];base=m['baselines']
   else:
    v=m[part]['micro']
    ci=m['uncertainty']['macro'][part];tokens=m[part]['prediction_tokens'];base=m[part]['extended']['baselines']
   assert tokens==prediction_tokens[part]
   name=spec['id'] if part=='all' else 'first_repeat_'+part
   item=dict(lm_seed=s,accuracy=v['accuracy'],errors=sum(1-r['correct'] for r in rr),targets=len(rr),answer_ce=v['answer_ce'],binary_accuracy=v['binary_accuracy'],bit_mass=v['bit_mass'],all_token_ce=m['all_token_ce'] if part=='all' else m[part]['all_token_ce'],prediction_tokens=tokens,accuracy_ci95=ci['accuracy_ci95'],answer_ce_ci95=ci['answer_ce_ci95'],baselines=base)
   if part!='all':
    assert abs(item['accuracy']-m[part]['cell']['macro_accuracy'])<1e-12
    item['paired_gap']=m['paired_gap'];item['paired_gap_ci95']=m['uncertainty']['paired_macro_gap']
   summary.setdefault(name,[]).append(item)
   operator_rates[f'{s}/{name}']={op:dict(targets=sum(r['operator']==op for r in rr),errors=sum(r['operator']==op and not r['correct'] for r in rr)) for op in sorted({r['operator'] for r in rr})}
 overlap[spec['id']]=dict(all_three_shared_errors=[list(x) for x in sorted(set.intersection(*sets))],unique_errors=len(set.union(*sets)))
aggregate={k:{metric:dict(mean=statistics.mean(r[metric] for r in rows),minimum=min(r[metric] for r in rows),maximum=max(r[metric] for r in rows)) for metric in ('accuracy','answer_ce')} for k,rows in summary.items()}
timing={str(s):sum(json.loads(p.read_text())['inference_seconds'] for p in (RUN/f'seed{s}').glob('*/result.json')) for s in range(3)}
result=dict(status='passed_metadata_and_descriptive_analysis',metadata_targets_verified=metadata_count,model_inference_repeated=False,by_suite=summary,seed_mean_min_max=aggregate,errors_by_operator=operator_rates,error_overlap=overlap,inference_seconds_by_seed=timing)
for name,value in [('analysis.json',result),('error_rows.json',errors)]:
 (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(metadata_targets_verified=metadata_count,errors=len(errors),shared_pair_errors=overlap['first_repeat'],seed_means=aggregate,timing=timing),indent=2))
