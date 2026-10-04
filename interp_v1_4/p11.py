"""Reporting-only synthesis of frozen v1.4 evidence; never fits or runs a model."""
from pathlib import Path
import csv
import hashlib
import io
import itertools
import json
import re
import zipfile
from collections import Counter, defaultdict
import numpy as np

RESULTS = Path('experiment_v1_4/results')
CONTRACT = Path('experiment_v1_4/p11_r1/contract.json')
READ_SOURCES = [
 ('P7','sae',0,'experiment_v1_4/phase_archives/P7/v1_4_p7_return_1790455082926962917.zip',''),
 ('P8','tc',0,'experiment_v1_4/phase_archives/P8/P8_evaluation_raw.zip',''),
 ('P9','sae',1,'experiment_v1_4/evidence/P9/v1_4_p9_evaluation_return_1790842429996719913.zip','sae/'),
 ('P9','tc',1,'experiment_v1_4/evidence/P9/v1_4_p9_evaluation_return_1790842429996719913.zip','tc/'),
]
COMPLETIONS = ['audit_20260921_01','p3_audit_20260921_01','frozen_test_audit_20260922_01',
 'p5_final_audit_20260925_01','p6_final_audit_20260925','p7_final_audit_20260928_01',
 'p8_final_audit_20260930_01','p9_final_audit_20261002_01','p10_final_audit_20261004_01']


def digest(data): return hashlib.sha256(data).hexdigest()
def hashfile(path):
    with open(path,'rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def json_text(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def write_json(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}),lineterminator='\n');w.writeheader();w.writerows(rows)


class Evidence:
    def __init__(self):self.files={};self.archives={};self.handles={};self.references=[];self.environments=[]
    def raw(self,path,member=None,expected=None):
        path=str(path)
        if member is None:data=Path(path).read_bytes();key=path
        else:
            if path not in self.handles:
                self.handles[path]=zipfile.ZipFile(path)
                self.archives[path]=dict(sha256=hashfile(path),bytes=Path(path).stat().st_size)
            data=self.handles[path].read(member);key=path+'::'+member
        actual=digest(data)
        if expected and actual!=expected:raise ValueError('Evidence hash mismatch: '+key)
        self.files[key]=dict(sha256=actual,bytes=len(data))
        return data
    def js(self,path,member=None,expected=None):return json.loads(self.raw(path,member,expected))


class Source:
    def __init__(self,e,path,prefix=''):
        self.e=e;self.path=Path(path);self.prefix=prefix;self.zip=self.path.suffix=='.zip'
        if self.zip:
            self.manifest=e.js(path,'return_manifest.json')['files']
            self.names=set(zipfile.ZipFile(path).namelist())
        else:
            self.manifest=e.js(self.path/'return_manifest.json')['files']
            self.names=set(self.manifest)
    def raw(self,name):
        member=self.prefix+name
        if member not in self.manifest:raise ValueError('Not covered by return manifest: '+member)
        return self.e.raw(self.path,member,self.manifest[member]) if self.zip else self.e.raw(self.path/member,expected=self.manifest[member])
    def js(self,name):return json.loads(self.raw(name))
    def ref(self,name):return str(self.path)+('::' if self.zip else '/')+self.prefix+name


def identity(run,phase):
    return {k:run.get(k,'') for k in ('name','lm_seed','layer','k','sparse_seed','position_type','hook','tool','checkpoint_sha256')}|{'phase':phase}


def ci_fields(value):
    ci=value.get('ci95')
    return dict(ci_low=ci[0] if ci else None,ci_high=ci[1] if ci else None,
                bootstrap_requested=value.get('requested',0),bootstrap_valid=value.get('valid',0),bootstrap_seed=value.get('seed',''))


def semantic_rows(base,task,size,m,status,support,source):
    rows=[]
    for subset,values in [('all',m),('current_ne_previous',m.get('current_ne_previous'))]:
        if values is None:continue
        bs=values.get('cluster_bootstrap',{})
        for metric in ('balanced_accuracy','macro_f1','binary_auroc'):
            interval=bs.get('metrics',{}).get(metric,{})
            val=values.get(metric)
            rows.append(base|dict(task=task,size=size,subset=subset,metric=metric,value=val,status=status,
                na_reason='' if val is not None else ('binary AUROC not defined for multiclass or missing class; see support' if metric=='binary_auroc' else 'support insufficient'),
                positions=values.get('positions',0),clusters=bs.get('clusters',''),support=json_text(support),
                class_counts=json_text(values.get('class_counts',[])),confusion_matrix=json_text(values.get('confusion_matrix',[])),source=source,
                **ci_fields(interval|{'requested':bs.get('requested',0),'seed':bs.get('seed','')})))
    return rows


def paired_interval(a,b,seed,draws=1000):
    """One common draw across paired observations; return b-a, never subtract CIs."""
    if set(a)!=set(b) or not a:raise ValueError('Paired cluster keys differ or are empty')
    keys=sorted(a);delta=np.asarray([b[k]-a[k] for k in keys],float)
    rng=np.random.Generator(np.random.PCG64(seed))
    values=delta[rng.integers(len(keys),size=(draws,len(keys)))].mean(axis=1)
    return dict(mean=float(delta.mean()),ci95=np.percentile(values,[2.5,97.5]).tolist(),requested=draws,valid=draws,seed=seed,clusters=len(keys))


def read_evaluations(e):
    semantic=[];fidelity=[];causal=[];coverage=[];layers=[];runs=[];replacement_raw={};target_hashes=set()
    for phase,tool,sparse,path,prefix in READ_SOURCES:
        src=Source(e,path,prefix);config=src.js('contract.json') if not prefix else e.js('experiment_v1_4/p9_eval_r1/'+tool+'_contract.json')
        config_hash=digest(src.raw('contract.json')) if not prefix else hashfile('experiment_v1_4/p9_eval_r1/'+tool+'_contract.json')
        target_path=config['reconstruction_targets'];target_hashes.add(config['files'][target_path])
        targets=e.js(target_path,expected=config['files'][target_path])
        if len(targets)!=2048 or len({t['sequence_id'] for t in targets})!=2048:raise ValueError('Replacement sequence grid')
        e.references.extend(dict(phase=phase,path=p,expected_sha256=h,role='frozen evaluation input',contract_sha256=config_hash) for p,h in config['files'].items())
        for member in sorted(src.names):
            if member.endswith(('environment.json','requirements.lock.txt')):
                data=e.raw(src.path,member,src.manifest[member]) if src.zip else e.raw(src.path/member,expected=src.manifest[member])
                e.environments.append(dict(phase=phase,source=str(src.path)+('::' if src.zip else '/')+member,sha256=digest(data)))
        entries=config.get('saes',config.get('tcs',config.get('dictionaries',[])))
        if not entries:raise ValueError('Missing dictionary manifest: '+path)
        print('Reading',phase,tool,len(entries),flush=True)
        for entry in entries:
            run=entry['run'];name=run['name'];base=identity(run,phase);base['tool']=tool
            semname=f'runs/{name}/semantic.json';s=src.js(semname);summary=src.js(f'runs/{name}/summary.json')
            if s['config_sha256']!=config_hash or summary['config_sha256']!=config_hash:raise ValueError('Mixed evaluation contracts')
            src.raw(f'runs/{name}/selection.json')
            if s['run']!=run or summary['run']!=run:raise ValueError('Run identity mismatch')
            target='h reconstruction' if tool=='sae' else 'm prediction from u'
            runs.append(base|dict(status='passed',selected_update=entry['selected_update'],selected_dictionary_sha256=entry['sha256'],
                selected_checkpoint=entry['checkpoint'],updates=5000,position_draws=2560000,config_sha256=s['config_sha256'],
                selection_sha256=s['selection_sha256'],source=src.ref(semname),target=target))
            for task,res in s['semantic'].items():
                rep,label,domain=task.rsplit('_',2)
                rep={'transcoder':'tc','transcoder_128':'tc_128','fullu':'full_u','fullm':'full_m'}.get(rep,rep)
                candidates=128 if rep.endswith('_128') else (512 if rep in ('sae','tc') else 256)
                info=base|dict(representation=rep,label=label,domain=domain,candidates=candidates,target=target)
                for size,m in res.get('evaluation',{}).items():
                    semantic+=semantic_rows(info,task,size,m,res['status'],res.get('support',[]),src.ref(semname))
                if not res.get('evaluation'):semantic.append(info|dict(task=task,status=res['status'],na_reason=json_text(res),source=src.ref(semname)))
            for split,m in s['fidelity'].items():
                if not isinstance(m,dict):continue
                vals={k:m[k] for k in ('mse','nmse','r2','ev','inactive_fraction','inactive_count')}
                vals.update(l0_mean=m['l0']['mean'],l0_median=m['l0']['median'],dead_fraction=s['fidelity']['train']['inactive_fraction'])
                for metric,value in vals.items():fidelity.append(base|dict(category='fidelity',target=target,split=split,metric=metric,value=value,
                    positions=m['positions'],l0_quantiles=json_text(m['l0']['quantiles']),ci_method='point estimate; empirical LM/sparse ranges reported separately',source=src.ref(semname)))
            for task,m in summary['causal'].items():
                changed,condition,size,subset,control=task.split('|')
                for metric,value in m['metrics'].items():causal.append(base|dict(task=task,changed=changed,condition=condition,size=size,subset=subset,
                    control=control,metric=metric,value=value['mean'],origins=m['origins'],rows=m['rows'],target=target,
                    source=src.ref(f'runs/{name}/summary.json'),**ci_fields(value)))
            for suite,m in summary['coverage'].items():coverage.append(base|dict(suite=suite,**m,matched_fraction=m['matched']/m['pairs'] if m['pairs'] else None))
            for metric,value in summary['replacement']['metrics'].items():fidelity.append(base|dict(category='replacement',target=target,split='behavior_test',
                metric=metric,value=value['mean'],positions=summary['replacement']['origins'],source=src.ref(f'runs/{name}/summary.json'),**ci_fields(value)))
            # Saved model outputs only, no inference. Exactly one target per sequence.
            raw={m:{} for m in ('delta_ce','delta_accuracy')}
            for start in range(0,2048,16):
                with np.load(io.BytesIO(src.raw(f'runs/{name}/replacement/{start:05d}.npz'))) as z:
                    for i,index in enumerate(z['indices']):
                        raw['delta_ce'][targets[int(index)]['sequence_id']]=float(z['patched_ce'][i]-z['original_ce'][i])
                        answer=z['answers'][i]+13
                        raw['delta_accuracy'][targets[int(index)]['sequence_id']]=float(z['patched_logits'][i].argmax()==answer)-float(z['original_logits'][i].argmax()==answer)
            for metric,values in raw.items():
                if not np.isclose(np.mean(list(values.values())),summary['replacement']['metrics'][metric]['mean'],atol=1e-12,rtol=1e-10):raise ValueError('Replacement raw/summary mismatch')
            if any(len(v)!=2048 for v in raw.values()):raise ValueError('Replacement target coverage')
            replacement_raw[run['lm_seed'],tool,run['k'],sparse,run['layer']]=raw
        diff=src.js('tables/layer_differences.json')
        for key,obj in diff['results'].items():
            match=re.match(r'seed(\d+)\|k(\d+)\|layers(\d+)-(\d+)\|(.*)',key)
            if not match:raise ValueError(key)
            seed,k,low,high,task=match.groups()
            for metric,value in obj.get('metrics',{'balanced_accuracy':obj}).items():
                layers.append(dict(phase=phase,tool=tool,sparse_seed=sparse,lm_seed=int(seed),k=int(k),layer_low=int(low),layer_high=int(high),
                    task=task.replace('transcoder','tc'),metric=metric,value=value['mean'],origins=obj.get('origins',''),direction=diff['direction'],
                    source=src.ref('tables/layer_differences.json'),**ci_fields(value)))
    if len(target_hashes)!=1:raise ValueError('Layer replacement targets differ')
    for seed,tool,k,sparse in sorted({x[:4] for x in replacement_raw}):
        seedkey=f'20260909|experiment-spec-v1.0|bootstrap|v1.4|P11|replacement|lm={seed}|tool={tool}|k={k}|sparse={sparse}'
        bs=int.from_bytes(hashlib.sha256(seedkey.encode()).digest()[:8],'big')
        for low,high in itertools.combinations((0,3,7,11),2):
            for metric in ('delta_ce','delta_accuracy'):
                v=paired_interval(replacement_raw[seed,tool,k,sparse,low][metric],replacement_raw[seed,tool,k,sparse,high][metric],bs)
                layers.append(dict(phase='P11',tool=tool,sparse_seed=sparse,lm_seed=seed,k=k,layer_low=low,layer_high=high,task='replacement',metric=metric,
                    value=v['mean'],origins=v['clusters'],direction='higher layer minus lower layer',bootstrap_key=seedkey,source='saved replacement arrays',**ci_fields(v)))
    return semantic,fidelity,causal,coverage,layers,runs


def p5_rows(e):
    path='experiment_v1_4/phase_archives/P5/p5_independent_audit_20260925T054743549355.zip'
    src=Source(e,path);rows=[];statuses=Counter()
    for n in sorted(n for n in src.names if n.startswith('tasks/') and n.endswith('.json')):
        j=src.js(n);source=src.ref(n)
        task=Path(n).stem;statuses[j['status']]+=1
        match=re.match(r'seed(\d+)_(init|trained)_l(\d+)_(h|u|m)_(.*)_([^_]+)_(iid|transfer)$',task)
        if match:
            seed,state,layer,hook,rep,label,domain=match.groups()
        else:
            # token/position and shuffled tasks retain their exact original identity.
            seed=re.search(r'seed(\d+)',task).group(1);state='control';layer='';hook='';rep='token_position';label=task.split('_')[-2];domain=task.split('_')[-1]
        base=dict(phase='P5',name=task,lm_seed=int(seed),layer=int(layer) if layer else '',hook=hook,tool='probe',position_type='READ',
                  representation=rep,model_state=state,label=label,domain=domain,k='',sparse_seed='',candidates=17 if rep=='token_position' else (128 if '128' in rep else 256))
        for size,v in j.get('checks',{}).items():
            if not isinstance(v,dict) or 'metrics' not in v:continue
            rows+=semantic_rows(base,task,size,v['metrics'],j['status'],{},source)
        if j['status']!='passed':rows.append(base|dict(task=task,status=j['status'],na_reason=json_text(j),source=source))
    if sum(statuses.values())!=3510:raise ValueError(('P5 task coverage',statuses))
    return rows,dict(statuses)


def update_rows(e):
    root='experiment_v1_4/results/p10_return_audit_20261004_01/returned_metadata';src=Source(e,root)
    config=src.js('contract.json');e.references.extend(dict(phase='P10',path=p,expected_sha256=h,role='frozen evaluation input') for p,h in config['files'].items());runs=[];semantic=[];fidelity=[];byrun={x['run']['name']:x for x in config['selected']}
    for entry in config['selected']:
        run=entry['run'];base=identity(run,'P10');base['tool']='tc' if run['tool'] in ('tc','transcoder') else 'sae'
        target='h reconstruction' if base['tool']=='sae' else 'm prediction from u';name=f"fidelity/{run['name']}.json";s=src.js(name)
        runs.append(base|dict(status='passed',selected_update=entry['selected_update'],selected_dictionary_sha256=entry['sha256'],selected_checkpoint=entry['checkpoint'],
            updates=5000,position_draws=2560000,config_sha256=s['config_sha256'],source=src.ref(name),target=target))
        for split,m in s['fidelity'].items():
            if not isinstance(m,dict):continue
            for metric,value in {**{k:m[k] for k in ('mse','nmse','r2','ev','inactive_fraction','inactive_count')},'l0_mean':m['l0']['mean'],
                'l0_median':m['l0']['median'],'dead_fraction':s['fidelity']['train']['inactive_fraction']}.items():
                fidelity.append(base|dict(category='fidelity',target=target,split=split,metric=metric,value=value,positions=m['positions'],source=src.ref(name)))
    for member in sorted(src.names):
        if member.endswith(('environment.json','requirements.lock.txt')):
            data=src.raw(member);e.environments.append(dict(phase='P10',source=src.ref(member),sha256=digest(data)))
    for task in config['tasks']:
        name=f"semantic/{task['id']}.json";j=src.js(name);entry=byrun.get(task.get('run'));base=identity(entry['run'],'P10') if entry else dict(phase='P10',lm_seed=task['lm_seed'],layer=3,position_type='UPDATE',tool='probe')
        base['tool']='tc' if base.get('tool')=='transcoder' else base.get('tool','probe')
        base.update(representation=task['representation'],label=task['label'],domain=task['domain'],hook=task['hook'],
            candidates=17 if task['representation']=='token' else (128 if task['candidate_limit']=='128' else (512 if entry else 256)))
        for size,m in j.get('reports',{}).items():
            semantic+=semantic_rows(base,task['id'],size,m,j['status'],j.get('test_support',[]),src.ref(name))
        if not j.get('reports'):semantic.append(base|dict(task=task['id'],status=j['status'],na_reason=json_text(j),source=src.ref(name)))
    return semantic,fidelity,runs


def behavior_rows(e):
    rows=[]
    for file,category in [('p4_audit_20260921_01/select_curves_all_seeds.csv','selection_curve'),('p4_audit_20260921_01/behavior_validation.csv','validation_gate')]:
        path=RESULTS/file
        for r in csv.DictReader(io.StringIO(e.raw(path).decode())):
            for metric,value in r.items():
                if metric not in ('seed','update','budget_m'):
                    rows.append(dict(lm_seed=int(r['seed']),category=category,suite='validation',update=r.get('update',r.get('selected_update','')),
                        budget_m=r.get('budget_m',''),metric=metric,value=value,source=str(path)))
    path='experiment_v1_4/phase_archives/P4/v1_4_frozen_test_evidence_20260922T033233833147.zip'
    e.raw(path,'checksums.json');names=e.handles[path].namelist()
    for n in sorted(n for n in names if n.endswith('/result.json') and n.startswith('run/seed')):
        j=e.js(path,n);m=j['metrics'];base=dict(lm_seed=j['binding']['lm_seed'],category='frozen_test',suite=j['binding']['suite'],source=path+'::'+n)
        def emit(vals,info,unc=None):
            unc=unc or {}
            for metric,value in vals.items():
                if value is None or (isinstance(value,(int,float)) and not isinstance(value,bool)):
                    ci=unc.get(metric+'_ci95',vals.get(metric+'_ci95'))
                    rows.append(base|info|dict(metric=metric,value=value,na_reason='empty stratum or undefined metric; see count/coverage' if value is None else '',ci_low=ci[0] if ci else None,ci_high=ci[1] if ci else None,
                        bootstrap_requested=unc.get('draws',1000 if ci else 0),bootstrap_valid=unc.get('valid_draws',1000 if ci else 0),bootstrap_seed=unc.get('seed','')))
        if base['suite']=='first_repeat':
            emit({'accuracy':m['paired_gap']['accuracy_first_minus_repeat'],'answer_ce':m['paired_gap']['answer_ce_first_minus_repeat']},
                 dict(suite='first_minus_repeat',stratum='paired 42-cell macro gap'),m['uncertainty']['paired_macro_gap']|{'draws':1000,'valid_draws':1000})
            for member in ('first','repeat'):
                emit(m[member]['micro'],dict(suite=member,stratum='micro'))
                emit({k:m[member][k] for k in ('all_token_ce','prediction_tokens')},dict(suite=member,stratum='all-token'))
                c=m[member]['cell'];emit({'accuracy':c['macro_accuracy'],'answer_ce':c['macro_answer_ce']},dict(suite=member,stratum='42-cell macro'),c|{'draws':1000,'valid_draws':1000})
                for cell,v in c['cells'].items():emit(v,dict(suite=member,stratum='cell',cell=cell),v|{'seed':m['uncertainty']['cell_seeds'][cell]})
                emit(m[member]['extended']['baselines'],dict(suite=member,stratum='baselines'))
        else:
            emit(m,dict(stratum='overall'),m.get('uncertainty',{}));emit(m['baselines'],dict(stratum='baselines'))
            for stratum,v in m['strata'].items():
                emit({k:x for k,x in v.items() if k!='cells'},dict(stratum=stratum,cell='macro'))
                for cell,values in v['cells'].items():emit(values,dict(stratum=stratum,cell=cell))
    return rows


def seed_summaries(tables):
    rows=[]
    dimensions=['position_type','layer','tool','k','category','target','representation','label','domain','candidates','task','size','subset','split','metric','changed','condition','control']
    for table,values in tables.items():
        grouped=defaultdict(dict)
        for r in values:
            if r.get('phase') not in ('P7','P8','P9','P10') or r.get('sparse_seed')!=0:continue
            if not isinstance(r.get('value'),(float,int)):continue
            # Task/run IDs in P10 contain the seed; normalize only that identity.
            rr=r.copy();rr['task']=re.sub(r'seed\d+_', 'seedX_',str(rr.get('task','')))
            key=tuple(rr.get(d,'') for d in dimensions)
            seed=r['lm_seed']
            if seed in grouped[key]:raise ValueError('Duplicate seed within summary group')
            grouped[key][seed]=r['value']
        for key,v in grouped.items():
            rows.append(dict(zip(dimensions,key))|dict(table=table,lm_seeds=json_text(sorted(v)),n_lm=len(v),
                mean=float(np.mean(list(v.values()))),minimum=min(v.values()),maximum=max(v.values()),
                seed0=v.get(0),seed1=v.get(1),seed2=v.get(2),ci_method='descriptive range; not an independent-layer or population CI'))
    return rows


def aggregate(output):
    output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; existing evidence is immutable')
    output.mkdir(parents=True)
    e=Evidence();contract=e.js(CONTRACT);prereq={}
    for name in COMPLETIONS:
        p=RESULTS/name/'completion.json';j=e.js(p)
        if not str(j.get('status','')).startswith('passed'):raise ValueError('Prerequisite failed: '+str(p))
        prereq[name]=dict(status=j['status'],sha256=e.files[str(p)]['sha256'])
    for p in (Path('experiment_v1_4/evidence/gpu_smoke_20260921_01/verification.json'),RESULTS/'p3_audit_20260921_01/gpu_smoke/verification.json'):
        j=e.js(p)
        if not str(j.get('status','')).startswith('passed'):raise ValueError('P2 GPU smoke not passed')
        prereq[str(p)]=dict(status=j['status'],sha256=e.files[str(p)]['sha256'])
    for p in sorted(Path('experiment_v1_4').glob('P*_STATUS.md')):
        if p.name!='P11_STATUS.md':e.raw(p)
    for p in ('03_experiment_spec.md','01_experiment_design.md','experiment_v1_4/analysis_plan.json','experiment_v1_4/configs/config_set_manifest.json','experiment_v1_4/design_config.json','experiment_v1_4/corpus_rebuild.json'):e.raw(p)
    semantic,fidelity,causal,coverage,layers,runs=read_evaluations(e)
    read_runs=runs.copy();p5,statuses=p5_rows(e);semantic+=p5
    s,f,r=update_rows(e);semantic+=s;fidelity+=f;runs+=r
    behavior=behavior_rows(e)
    expected={(lm,l,t,k,s) for lm in (0,1,2) for l in (0,3,7,11) for t in ('sae','tc') for k in (4,16) for s in (0,)}|{(0,l,t,k,1) for l in (0,3,7,11) for t in ('sae','tc') for k in (4,16)}
    observed={(r['lm_seed'],r['layer'],r['tool'],r['k'],r['sparse_seed']) for r in read_runs}
    if observed!=expected or len(read_runs)!=64 or len(runs)!=76:raise ValueError('Incomplete/duplicate run grid')
    for r in causal+layers:
        if r['bootstrap_requested']!=1000 or r['bootstrap_valid']>1000:raise ValueError('Bootstrap coverage')
    tables=dict(behavior=behavior,semantic_metrics=semantic,fidelity_metrics=fidelity,causal_metrics=causal,
        matching_coverage=coverage,layer_differences=layers,dictionary_registry=runs)
    tables['lm_seed_summary']=seed_summaries({k:tables[k] for k in ('semantic_metrics','fidelity_metrics','causal_metrics')})
    # Preserve original sparse sensitivity table: dictionary IDs are not aligned.
    p=RESULTS/'p9_return_audit_20261001_01/returned_metadata/tables/sparse_seed_comparison.csv'
    tables['sparse_seed_comparison']=list(csv.DictReader(io.StringIO(e.raw(p).decode())))
    registry=list(csv.DictReader(io.StringIO(e.raw(Path('experiment_v1_4/p11_r1/prior_run_registry.csv')).decode())))
    tables['run_registry']=registry
    tables['reproduction_inputs']=e.references
    tables['environment_index']=e.environments
    for name,rows in tables.items():write_csv(output/(name+'.csv'),rows)
    write_json(output/'prerequisites.json',prereq)
    anchors={
      READ_SOURCES[0][3]:('p7_return_audit_20260927_01/archive_verification.json','archive_sha256'),
      READ_SOURCES[1][3]:('p8_return_audit_20260929_01/archive_verification.json','archive_sha256'),
      READ_SOURCES[2][3]:('p9_return_audit_20261001_01/archive_verification.json','archive_sha256'),
      'experiment_v1_4/phase_archives/P4/v1_4_frozen_test_evidence_20260922T033233833147.zip':('frozen_test_audit_20260922_01/archive_verification.json','sha256')}
    for path,(ref,key) in anchors.items():
        prior=e.js(RESULTS/ref)
        if e.archives[path]['sha256']!=prior[key]:raise ValueError('Archive changed since audit: '+path)
        e.archives[path]['prior_verification']=str(RESULTS/ref)
    path='experiment_v1_4/phase_archives/P5/p5_independent_audit_20260925T054743549355.zip'
    prior=e.js(RESULTS/'p5_final_audit_20260925_01/completion.json')
    if e.archives[path]['sha256']!=prior['archive_sha256']:raise ValueError('P5 audit archive mismatch')
    write_json(output/'input_manifest.json',dict(files=e.files,archives=e.archives,contract=contract))
    write_json(output/'aggregation.json',dict(status='aggregated_validation_pending',counts={k:len(v) for k,v in tables.items()},p5_statuses=statuses,
        read_runs=64,update_runs=12,bootstrap='existing audited intervals plus new saved-array paired replacement contrasts',
        source_files=len(e.files),source_archives=len(e.archives)))
    return tables


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();aggregate(args.output)
