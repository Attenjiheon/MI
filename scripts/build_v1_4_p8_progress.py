"""Versioned UI overlay; preserve every frozen r1 training byte and identity."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def digest(b):return hashlib.sha256(b).hexdigest()
def build():
    old=ROOT/'experiment_v1_4/bundles/P8/v1_4_p8_training_r1.zip'
    new=old.with_name('v1_4_p8_training_r2_progress.zip')
    expected=json.loads(old.with_suffix('.sha256.json').read_text())['sha256']
    assert digest(old.read_bytes())==expected
    extra=['scripts/p8_progress.py','scripts/build_v1_4_p8_progress.py','tests_v1_4/test_p8_progress.py']
    manifest=dict(schema='p8-progress-overlay-r2',base_bundle_sha256=expected,training_contract='experiment_v1_4/p8_r1/contract.json',training_unchanged=True,output_directory='P8_training_r1',files={n:digest((ROOT/n).read_bytes()) for n in extra})
    with zipfile.ZipFile(old) as source:
        with zipfile.ZipFile(new,'x',zipfile.ZIP_DEFLATED) as target:
            for n in source.namelist():target.writestr(n,source.read(n))
            for n in extra:target.write(ROOT/n,n)
            target.writestr('p8_progress_overlay.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(old) as a,zipfile.ZipFile(new) as b:
        for n in a.namelist():assert a.read(n)==b.read(n)
    checksum=digest(new.read_bytes());new.with_suffix('.sha256.json').write_text(json.dumps(dict(sha256=checksum,bytes=new.stat().st_size),indent=2)+'\n')
    nb=json.loads((ROOT/'experiment_v1_4/notebooks/P8/P8_colab_read_tc_training_r1.ipynb').read_text())
    for cell in nb['cells']:
        s=''.join(cell['source']).replace('v1_4_p8_training_r1.zip','v1_4_p8_training_r2_progress.zip').replace(expected,checksum)
        if cell['cell_type']=='code' and 'verify(ROOT)' in s:
            s=s.replace('verify(ROOT)',"verify(ROOT)\noverlay=json.loads((ROOT/'p8_progress_overlay.json').read_text())\nfor name,checksum in overlay['files'].items():\n    assert digest(ROOT/name)==checksum, '진행 표시 코드 checksum 불일치'")
        if cell['cell_type']=='code' and 'RUNS=None' in s:
            start=s.index('subprocess.run(')
            cmd=s[start+len('subprocess.run('):].split(',cwd=ROOT,check=True)')[0]
            s=s[:start]+"from scripts.p8_progress import run_with_progress\ncommand="+cmd+"\nprogress=run_with_progress(command,ROOT,OUTPUT,RUNS)\n"
        if cell['cell_type']=='markdown' and s.startswith('# P8'):
            s+='\n\n**r2 진행 표시 추가:** 학습 셀에서 전체/현재 run 진행률, 경과시간, 속도와 예상 남은 시간을 표시합니다. r1 학습 코드·계약·출력 경로는 동일하여 기존 checkpoint를 재개할 수 있습니다. ETA는 학습 단계의 추정이며 후속 검산·평가 시간은 제외됩니다.'
        cell['source']=s.splitlines(True)
        if cell['cell_type']=='code':compile(s,'p8-r2','exec')
    out=ROOT/'experiment_v1_4/notebooks/P8/P8_colab_read_tc_training_r2_progress.ipynb'
    out.write_text(json.dumps(nb,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(notebook=str(out),bundle=str(new),sha256=checksum,bytes=new.stat().st_size),indent=2))
if __name__=='__main__':build()
