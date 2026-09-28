"""Create a Drive-loading notebook without changing the frozen r2 input bundle."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

def build():
    original=ROOT/'experiment_v1_4/notebooks/P7/P7_source_return_audit_r2.ipynb'
    notebook=json.loads(original.read_text())
    receipt=json.loads((ROOT/'experiment_v1_4/bundles/P7/v1_4_p7_source_audit_r2.sha256.json').read_text())
    intro=''.join(notebook['cells'][0]['source'])
    intro+='''\n\n## Drive에서 입력 번들 연결

기존 `v1_4_p7_source_audit_r2.zip`을 Google Drive의 **내 드라이브 → boolean_interp_v1_4** 폴더에 미리 올려 두세요. 번들을 새로 받을 필요는 없습니다.

첫 셀은 Drive를 마운트한 뒤 ZIP을 Colab 로컬 디스크로 복사하면서 SHA-256을 검사합니다. 브라우저 업로드 창은 사용하지 않습니다. 다른 폴더에 두었다면 `BUNDLE_IN_DRIVE`만 수정하세요. 복사 진행률을 표시하고, 같은 런타임에서 재실행하면 checksum이 맞는 로컬 ZIP을 재사용합니다.
'''
    notebook['cells'][0]['source']=intro.splitlines(True)
    code='''from google.colab import files, drive
from pathlib import Path
import hashlib, zipfile, subprocess, sys, os, time, json

drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/boolean_interp_v1_4')
# Drive에서 다른 폴더에 보관했다면 이 경로만 수정하세요.
BUNDLE_IN_DRIVE=BASE/'v1_4_p7_source_audit_r2.zip'
EXPECTED_SHA256=__DIGEST__
EXPECTED_BYTES=__BYTES__
bundle=Path('/content/v1_4_p7_source_audit_r2.zip')

def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

cached=(bundle.is_file() and bundle.stat().st_size==EXPECTED_BYTES
        and file_sha256(bundle)==EXPECTED_SHA256)
if cached:
    print('checksum 확인 완료: 기존 Colab 로컬 번들을 재사용합니다.')
else:
    if not BUNDLE_IN_DRIVE.is_file():
        raise FileNotFoundError(f'Drive에 ZIP을 올린 뒤 다시 실행하세요: {BUNDLE_IN_DRIVE}')
    if BUNDLE_IN_DRIVE.stat().st_size!=EXPECTED_BYTES:
        raise ValueError('Drive ZIP 크기가 다릅니다. Drive 업로드가 완료됐는지 확인하세요.')
    partial=bundle.with_suffix('.zip.partial')
    digest=hashlib.sha256();copied=0;last_report=0
    with BUNDLE_IN_DRIVE.open('rb') as source,partial.open('wb') as target:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):
            target.write(chunk);digest.update(chunk);copied+=len(chunk)
            if copied-last_report>=64*1024*1024 or copied==EXPECTED_BYTES:
                print(f'Drive → Colab: {copied/1e6:.1f} / {EXPECTED_BYTES/1e6:.1f} MB',flush=True)
                last_report=copied
    if copied!=EXPECTED_BYTES or digest.hexdigest()!=EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise ValueError('Drive ZIP checksum 불일치. 원본 번들을 다시 확인하세요.')
    partial.replace(bundle)
    print('복사 및 checksum 검증 완료')

ROOT=Path('/content/MI_P7_audit_r2');ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    for name in z.namelist():
        assert (ROOT/name).resolve().is_relative_to(ROOT.resolve()), 'ZIP 경로 오류'
    z.extractall(ROOT)
SOURCE=BASE/'P5_r2';RETURNED=BASE/'P7_r1';OUT=BASE/'P7_source_audit_r2';OUT.mkdir(exist_ok=True)
assert (SOURCE/'contract.json').exists() and (RETURNED/'evaluation_complete.json').exists(), '기존 Drive 경로를 확인하세요'
print('준비 완료: 다음 셀부터 실행하세요.')
'''.replace('__DIGEST__',repr(receipt['sha256'])).replace('__BYTES__',str(receipt['bytes']))
    compile(code,'Drive bundle cell','exec')
    cell=next(c for c in notebook['cells'] if c['cell_type']=='code')
    cell.update(source=code.splitlines(True),outputs=[],execution_count=None)
    destination=ROOT/'experiment_v1_4/notebooks/P7/P7_source_return_audit_Drive_r2.ipynb'
    payload=json.dumps(notebook,ensure_ascii=False,indent=2)+'\n'
    if destination.exists():assert destination.read_text()==payload
    else:destination.write_text(payload)
    print(destination)

if __name__=='__main__':build()
