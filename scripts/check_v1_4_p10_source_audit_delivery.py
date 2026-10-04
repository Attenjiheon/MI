"""Check base+overlay standalone audit delivery; production source audit remains pending."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,time,zipfile
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interp_v1_4 import p10_source_audit as audit
from interp_v1_4.p5 import read,write
from interp_v1_4.runtime import sha


def main():
    start=time.monotonic();out=ROOT/'experiment_v1_4/results/p10_source_audit_preparation_r1';out.mkdir(parents=True,exist_ok=True)
    a,c=audit.verify(ROOT);base=ROOT/'experiment_v1_4/bundles/P10/v1_4_p10_update_evaluation_r1.zip';overlay=ROOT/'experiment_v1_4/bundles/P10/v1_4_p10_source_audit_r1.zip'
    assert sha(base)==a['base_bundle_sha256']
    assert sha(overlay)==read(overlay.with_suffix('.sha256.json'))['sha256']
    with tempfile.TemporaryDirectory(prefix='p10audit-delivery-') as tmp:
        tmp=Path(tmp)
        for p in (base,overlay):
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():assert (tmp/n).resolve().is_relative_to(tmp.resolve())
                z.extractall(tmp)
        for n,h in {**c['files'],**a['files'],audit.CONTRACT:sha(ROOT/audit.CONTRACT)}.items():assert sha(tmp/n)==h,n
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        proc=subprocess.run([sys.executable,'-m','pytest','tests_v1_4/test_p10_source_audit.py','-q'],cwd=tmp,env=env,text=True,capture_output=True)
        (out/'standalone_tests.txt').write_text(proc.stdout+proc.stderr);assert proc.returncode==0,proc.stdout+proc.stderr
        proc=subprocess.run([sys.executable,'-c','from interp_v1_4.p10_source_audit import verify; print(len(verify(".")[1]["tasks"]))'],cwd=tmp,env=env,text=True,capture_output=True)
        assert proc.returncode==0 and proc.stdout.strip()=='1248',proc.stderr
    paths=[ROOT/audit.CONTRACT,overlay,overlay.with_suffix('.sha256.json'),out/'standalone_tests.txt',Path(__file__)]
    for name in ('P10_source_audit_1_T4_r1.ipynb','P10_source_audit_2_CPU_r1.ipynb'):
        p=ROOT/'experiment_v1_4/notebooks/P10'/name;nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':compile(cell.source,name,'exec')
        paths.append(p)
    write(out/'verification.json',dict(status='passed_preparation_only',p10_complete=False,source_audit_runs=0,standalone_tests_passed=7,
         base_bundle_sha256=sha(base),planned=a['planned'],elapsed_seconds=time.monotonic()-start,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
         commands=['/opt/anaconda3/bin/python scripts/build_v1_4_p10_source_audit.py','/opt/anaconda3/bin/python scripts/check_v1_4_p10_source_audit_delivery.py'],
         limits=['GPU replay and full CPU original-cache refit/prediction/CI audit not run locally','No P10 completion until returned source evidence is checked']))
    print('PASS: combined base+overlay verification, 7 standalone tests, 2 notebooks. Actual source audit pending.')

if __name__=='__main__':main()
