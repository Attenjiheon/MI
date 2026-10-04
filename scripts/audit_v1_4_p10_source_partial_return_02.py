"""Verify three P10 source-audit returns through the second CPU pause, retaining incomplete CPU status."""
from pathlib import Path
import collections,hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4.p10_source_audit import identity,compare,verify

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    a,c=verify(ROOT);ident=identity(ROOT);e=ROOT/'experiment_v1_4/evidence/P10';old=None;evidence={};counts=[]
    for stamp in ('1791099132974215497','1791112944172002724','1791121054721333170'):
        p=e/f'v1_4_p10_source_audit_return_{stamp}.zip';s=(e/'v1_4_p10_source_audit_return_1791121003271150559.sha256.json') if stamp=='1791121054721333170' else p.with_suffix('.sha256.json');check=json.loads(s.read_text())
        assert sha(p)==check['sha256'] and p.stat().st_size==check['bytes']
        evidence[str(p.relative_to(ROOT))]=sha(p);evidence[str(s.relative_to(ROOT))]=sha(s)
        with zipfile.ZipFile(p) as z:
            m=json.loads(z.read('return_manifest.json'))['files'];assert len(z.namelist())==len(set(z.namelist()));assert set(z.namelist())==set(m)|{'return_manifest.json'}
            for n,h in m.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
            if old:
                for n,h in old.items():assert m[n]==h,n
            old=m;counts.append(len(m));data={n:json.loads(z.read(n)) for n in m if n.endswith('.json')}
    g=data['gpu_complete.json'];assert g['identity']==ident and g['status']=='passed'
    assert g['activation_chunks']==240 and g['activation_position_visits']==480000 and g['dictionary_runs']==12
    for n,h in g['files'].items():assert m[n]==h
    chunks={n:d for n,d in data.items() if n.startswith('chunks/')};ds={n:d for n,d in data.items() if n.startswith('dictionaries/')}
    assert set(g['files'])==set(chunks)|set(ds) and len(chunks)==240 and len(ds)==12
    quota=collections.Counter();offsets=collections.defaultdict(list);maxerr=0.
    for d in chunks.values():
        assert d['identity']==ident and d['status']=='passed';key=(d['lm_seed'],d['kind'],d['split']);quota[key]+=d['positions'];offsets[key].append(d['sequence_offset']);maxerr=max(maxerr,*d['max_absolute_errors'].values())
    assert len(quota)==18
    for lm in range(3):
        for kind in ('trained','init'):
            for split,n in [('train',50000),('val',10000),('test',20000)]:
                key=(lm,kind,split);assert quota[key]==n;assert sorted(offsets[key])==list(range(0,128*len(offsets[key]),128))
    for selected in c['selected']:
        name=selected['run']['name'];d=ds[f'dictionaries/{name}.json'];assert d['identity']==ident and d['status']=='passed' and d['checkpoint_sha256']==selected['sha256']
        for split,n in [('train',50000),('val',10000),('test',20000)]:assert d['splits'][split]['positions']==n
    tasks={n:d for n,d in data.items() if n.startswith('tasks/')};assert set(tasks)=={f'tasks/{t["id"]}.json' for t in c['tasks'][:1033]}
    source=ROOT/'experiment_v1_4/results/p10_return_audit_20261004_01/returned_metadata';reports=0
    for d in tasks.values():
        assert d['status']=='passed' and d['identity']==ident
        fit=source/'fits'/f'{d["task"]}.json';report=source/'semantic'/f'{d["task"]}.json'
        assert sha(fit)==d['source_fit_sha256'] and sha(report)==d['source_report_sha256']
        f=json.loads(fit.read_text())['fit'];compare(d['selected'],f['selected'],**a['metric_tolerance']);assert d['reports']==len(f['selected']);reports+=d['reports']
    assert 'source_audit_complete.json' not in data
    evidence[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    out=ROOT/'experiment_v1_4/results/p10_source_partial_audit_20261004_02';out.mkdir(exist_ok=True)
    result=dict(status='gpu_passed_cpu_paused',p10_complete=False,identity=ident,zip_member_hashes_verified=counts,gpu_chunks=240,activation_position_visits=480000,dictionary_runs=12,max_activation_error=maxerr,max_latent_error=max(v['maximum_latent_error'] for d in ds.values() for v in d['splits'].values()),cpu_tasks_verified=len(tasks),cpu_tasks_total=1248,cpu_tasks_remaining=215,semantic_reports_verified=reports,cpu_elapsed_seconds=sum(d['elapsed_seconds'] for d in tasks.values()),evidence_sha256=evidence,next_action='Resume unchanged CPU notebook; preserve Drive output; no GPU rerun')
    (out/'status.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
