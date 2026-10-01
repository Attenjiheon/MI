"""Read-only P9 original-cache reproduction of both tools and initialization comparison."""
from pathlib import Path
import argparse,csv,io,time,sys,contextlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from interp_v1_4.p5 import read,write,QUOTAS,session
from interp_v1_4.runtime import sha,deterministic
from scripts.audit_v1_4_p9_tc_sources import compare
AUDIT='experiment_v1_4/p9_audit_r1/contract.json'


def check_csv(path,rows):
    fields=sorted({k for row in rows for k in row});buf=io.StringIO()
    writer=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    assert Path(path).read_text()==buf.getvalue(),str(path)


@contextlib.contextmanager
def compare_writes(module):
    """Intercept serialization/figures; audit never replaces returned evidence."""
    import matplotlib.pyplot as plt
    original=module.write,module.csvfile,plt.Figure.savefig
    try:
        module.write=lambda path,value:compare(read(path),value,'aggregate-reproduction')
        module.csvfile=check_csv
        plt.Figure.savefig=lambda *a,**k:None
        yield
    finally:module.write,module.csvfile,plt.Figure.savefig=original


def check_resume(saved,identity,name):
    assert saved['identity']==identity and saved['run']==name and saved['status']=='passed','Stale source audit'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--source',required=True);p.add_argument('--returned',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root,source,returned,out=map(Path,(a.root,a.source,a.returned,a.output));out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);deterministic(707)
    if not torch.cuda.is_available():raise RuntimeError('Source reproduction requires original CUDA numerical path')
    audit=read(root/AUDIT)
    for n,d in audit['files'].items():assert sha(root/n)==d,n
    manifest=read(root/'experiment_v1_4/p9_audit_r1/return_manifest.json')
    for i,(n,d) in enumerate(manifest['files'].items()):
        assert sha(returned/n)==d,'Changed returned evidence: '+n
        if (i+1)%5000==0:print('returned checksum',i+1,'/',len(manifest['files']),flush=True)
    identity=dict(audit_contract_sha256=sha(root/AUDIT),return_manifest_sha256=sha(root/'experiment_v1_4/p9_audit_r1/return_manifest.json'),original_contract_sha256=audit['original_contract_sha256'])
    write(out/'identity.json',identity);env=session(out,sha(root/AUDIT),'cuda')
    labels={s:read(source/'labels'/f'{s}.json') for s in QUOTAS};results=[]
    from interp_v1_4 import p9_sae_evaluation,p9_tc_evaluation,p9_sae_aggregate,p9_tc_aggregate
    from scripts import audit_v1_4_p9_sae_sources,audit_v1_4_p9_tc_sources
    for tool,ev,engine,agg in [('sae',p9_sae_evaluation,audit_v1_4_p9_sae_sources,p9_sae_aggregate),('tc',p9_tc_evaluation,audit_v1_4_p9_tc_sources,p9_tc_aggregate)]:
        c=ev.source_gate(root,source);assert sha(root/ev.CONTRACT)==identity['original_contract_sha256'][tool]
        for e in c['saes' if tool=='sae' else 'tcs']:
            name=e['run']['name'];dest=out/'runs'/(name+'.json')
            if dest.exists():
                saved=read(dest);check_resume(saved,identity,name);results.append(saved);print('resume verified',name,flush=True);continue
            tick=time.monotonic()
            try:
                result=engine.audit_run(root,source,returned/tool,out,c,e,labels,out/'refits'/tool)
                result.update(identity=identity,environment_id=env['environment_id'],elapsed_seconds=time.monotonic()-tick,tool=tool)
                write(dest,result);results.append(result)
            except Exception as exc:
                write(out/'failures'/f'{time.time_ns()}.json',dict(run=name,error=repr(exc),identity=identity,environment_id=env['environment_id']));raise
        checkpoint=out/'aggregates'/f'{tool}.json'
        if checkpoint.exists():assert read(checkpoint)==dict(status='passed',identity=identity)
        else:
            with compare_writes(agg):agg.aggregate(root,source,returned/tool)
            write(checkpoint,dict(status='passed',identity=identity))
    from interp_v1_4 import p9_comparison
    with compare_writes(p9_comparison):p9_comparison.aggregate(root,returned)
    write(out/'source_audit_complete.json',dict(status='source_reproduction_passed_return_review_pending',p9_complete=False,whole_experiment_complete=False,identity=identity,runs=results,
        scope='All novel refits, all test predictions and semantic CI, full train/val/test fidelity and dead counts, full validation bins; four fixed causal pairs and first 16 replacement targets per run. All summary CI, layer contrasts and seed0/1 comparison reproduced. Local full raw checks are separate evidence.'))

if __name__=='__main__':main()
