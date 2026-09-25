"""Package an additive P6 numerical-auditor correction; preserve r1 training bytes."""
import hashlib
import json
from pathlib import Path
import zipfile
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def build():
    original=ROOT/'experiment_v1_4/bundles/P6/v1_4_p6_bundle_r1.zip'
    expected=json.loads(original.with_suffix('.sha256.json').read_text())['sha256']
    assert sha_bytes(original.read_bytes())==expected
    with zipfile.ZipFile(original) as z:items={n:z.read(n) for n in z.namelist()}
    contract=json.loads(items['experiment_v1_4/p6_r1/contract.json'])
    for n,d in contract['files'].items():assert sha_bytes(items[n])==d
    for n in ['scripts/verify_v1_4_p6_return_r2.py','tests_v1_4/test_p6_audit_r2.py']:
        assert n not in items;items[n]=(ROOT/n).read_bytes()
    manifest=dict(revision='p6-auditor-r2',original_bundle_sha256=expected,
                  production_contract_unchanged=True,tolerance_unchanged=dict(atol=1e-7,rtol=1e-6),
                  reason='Match float32 tensor normalization and fused F.linear before stable TopK; preserve legacy diagnostic values',
                  files={n:sha_bytes(v) for n,v in items.items()})
    items['p6_audit_r2_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    bundle=ROOT/'experiment_v1_4/bundles/P6/v1_4_p6_audit_r2.zip'
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for n,v in items.items():z.writestr(n,v)
    digest=sha_bytes(bundle.read_bytes())
    bundle.with_suffix('.sha256.json').write_text(json.dumps(dict(sha256=digest),indent=2)+'\n')
    n=nbformat.v4.new_notebook()
    n.metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU')
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    n.cells=[md('''# P6 완료 run 검산 r2

**GPU 런타임에서 이 노트북만 실행하세요. 재학습은 하지 않습니다.** 기존 Drive `boolean_interp_v1_4/P6_r1`의 저장 결과를 사용합니다. 새로운 런타임에서도 사용할 수 있습니다.

r1 검산기의 별도 행렬곱+덧셈을 production과 같은 `F.linear`로, 정규화 분모를 float32 tensor로 맞춥니다. TopK 직전의 연산 차이를 없애며 기존 허용 오차 `atol=1e-7, rtol=1e-6`는 유지합니다. 기존 검산식의 MSE와 TopK 후보 차이도 함께 기록합니다. 성공 여부는 GPU에서 다시 확인해야 합니다.

원본 학습 계약·코드·checkpoint·선택값은 보존하고 별도 `audits/r2_*`에 검산 기록을 추가합니다. 번들은 원본 r1 파일과 새 검산기만 포함합니다.'''),code(f'''from google.colab import files, drive
from pathlib import Path
import sys, subprocess, hashlib, zipfile, json, time
files.upload()  # v1_4_p6_audit_r2.zip 선택
bundle=Path('v1_4_p6_audit_r2.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='{digest}', 'ZIP checksum 불일치'
ROOT=Path('/content/MI_P6_audit_r2');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for name in z.namelist():
        assert (ROOT/name).resolve().is_relative_to(ROOT.resolve())
        target=ROOT/name
        if target.exists():assert target.read_bytes()==z.read(name), '기존 파일이 다릅니다: '+name
    z.extractall(ROOT)
manifest=json.loads((ROOT/'p6_audit_r2_manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
drive.mount('/content/drive')
OUTPUT=Path('/content/drive/MyDrive/boolean_interp_v1_4/P6_r1')
assert (OUTPUT/'input_manifest.json').is_file(), '기존 P6_r1 결과 경로를 확인하세요.'
'''),md('## 환경 준비\n동일한 패키지 버전을 사용합니다. 실제 환경은 검산 기록에 남습니다. 이 셀은 학습을 실행하지 않습니다.'),code('''install=subprocess.run([sys.executable,'-m','pip','install','numpy==2.1.3','scipy==1.16.3','torch==2.11.0','--extra-index-url','https://download.pytorch.org/whl/cu128'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(install.stdout);install.check_returncode()
LOGDIR=OUTPUT/'diagnostics'/f'audit_r2_{time.time_ns()}'
LOGDIR.mkdir(parents=True,exist_ok=False)
(LOGDIR/'install.log').write_text(install.stdout)
(LOGDIR/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
'''),md('## 24개 run 재검산\n기존 원본 검증을 유지하고 480개 validation checkpoint를 재검산합니다. 실패하면 실제 traceback과 해당 checkpoint의 수치를 기록합니다. 통과 기준을 완화하거나 checkpoint를 다시 선택하지 않습니다.'),code('''command=[sys.executable,'-u',str(ROOT/'scripts/verify_v1_4_p6_return_r2.py'),'--root',str(ROOT),'--output',str(OUTPUT),'--device','cuda']
with (LOGDIR/'audit.log').open('x') as log:
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        AUDIT_EXIT=process.wait()
    except KeyboardInterrupt:
        process.terminate();process.wait();log.write('Interrupted by user\\n');raise
    log.write(f'\\nRETURN CODE: {AUDIT_EXIT}\\n')
print('검산 종료 코드:',AUDIT_EXIT)
print('통과: 결과를 다운로드해 반환 검토를 받으세요.' if AUDIT_EXIT==0 else '불일치가 남았습니다. 다음 셀에서 진단 ZIP을 다운로드해 전달하세요.')
'''),md('## 검산 증빙 다운로드\n실패했어도 실행하세요. 로그·checkpoint별 비교·환경·수정 검산기·배송 manifest를 포함하며, 대용량 학습 checkpoint는 원래 Drive에 남습니다.'),code('''import shutil
stamp=str(time.time_ns())
small=Path(f'/content/p6_audit_r2_evidence_{stamp}.zip')
with zipfile.ZipFile(small,'x',zipfile.ZIP_DEFLATED) as z:
    for folder in [LOGDIR]+sorted((OUTPUT/'audits').glob('r2_*')):
        for path in folder.rglob('*'):
            if path.is_file():z.write(path,str(path.relative_to(OUTPUT)))
    for name in ['scripts/verify_v1_4_p6_return_r2.py','p6_audit_r2_manifest.json']:
        z.write(ROOT/name,name)
    z.write(OUTPUT/'input_manifest.json','input_manifest.json')
    for path in sorted((OUTPUT/'runs').glob('*/result.json')):
        z.write(path,str(path.relative_to(OUTPUT)))
print('SHA256:',hashlib.sha256(small.read_bytes()).hexdigest())
files.download(str(small))
'''),md('## 전체 P6 반환물 다운로드 — 검산 통과 후\n기존 전체 반환 ZIP을 아직 전달하지 않았다면 실행하세요. optimizer/RNG를 포함한 모든 checkpoint 때문에 파일이 클 수 있습니다. 위의 작은 검산 증빙과 함께 반환 감사에 사용합니다.'),code('''assert AUDIT_EXIT==0, '검산 불일치를 먼저 확인해야 합니다.'
archive=Path(f'/content/v1_4_p6_return_after_audit_r2_{time.time_ns()}.zip')
subprocess.run([sys.executable,'-m','interp_v1_4.p6','export','--output',str(OUTPUT),'--archive',str(archive)],cwd=ROOT,check=True)
files.download(str(archive))
files.download(str(archive.with_suffix('.sha256.json')))
''')]
    for cell in n.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook','exec')
    nbformat.validate(n)
    notebook=ROOT/'experiment_v1_4/notebooks/P6/P6_completed_run_audit_r2.ipynb'
    assert not notebook.exists();nbformat.write(n,notebook)
    print(json.dumps(dict(bundle=str(bundle),sha256=digest,notebook=str(notebook)),indent=2))
if __name__=='__main__':build()
