"""Package only the corrected, audited corpus and passing CPU implementation.

--draft writes a visibly blocked notebook for syntax review, without a bundle.
Normal packaging refuses missing P1/CPU smoke evidence. Existing deliverables
are immutable: choose a new --revision to build another package.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp_v1_4.runtime import sha, verify_inputs


def notebook(bundle_name, bundle_sha, draft=False):
    cells = []

    def md(source):
        cells.append(dict(cell_type="markdown", metadata={}, source=textwrap.dedent(source).strip() + "\n"))

    def code(source):
        source = textwrap.dedent(source).strip() + "\n"
        compile(source, "v1.4 notebook cell", "exec")
        cells.append(dict(cell_type="code", metadata={}, source=source, execution_count=None, outputs=[]))

    md("""
    # v1.4 P3 · deepwide12_read4 · seed 0 · 64M

    위에서 아래 순서로 실행합니다. 새 CUDA 런타임이 필요합니다. 모델은 12 blocks × 256,
    FP32, effective batch 64이며 GPU smoke가 microbatch를 정합니다. 코퍼스를 GPU에서 생성하지
    않습니다. 64M 최초 완전 update까지 학습하고 select로 선택한 checkpoint의 gate를 한 번만
    평가합니다. 실패하면 seed 1·2/test/해석으로 넘어가지 않습니다. 이 노트북은 seed 0만 실행합니다.

    **재개:** Drive의 완전 저장 index를 새 로컬 디렉터리로 복구합니다. 예산과 microbatch를
    변경하지 않습니다. gate_started만 있고 완료 결과가 없으면 재평가하지 말고 증빙을 회수합니다.
    GPU smoke/본학습은 로컬에서 실행됐다고 가정하지 않습니다. 반환 ZIP의 로컬 감사가 필요합니다.
    """)
    code(f"""
    READY = {not draft!r}
    assert READY, 'DRAFT: corrected P1 audit and CPU smoke are not yet packaged'
    BUNDLE_NAME = {bundle_name!r}
    EXPECTED_SHA256 = {bundle_sha!r}
    RESUME = False  # 기존 Drive run을 이어갈 때만 True
    """)
    md("""## 1. 입력 업로드 및 checksum 검증

    제공된 ZIP을 업로드합니다. Drive에 미리 올렸다면 BUNDLE_PATH에 해당 절대 경로를 지정할 수
    있습니다. 전체 ZIP hash와 내부 모든 파일의 hash를 검증한 뒤 새 로컬 폴더에만 풉니다.
    """)
    code("""
    import hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile
    from pathlib import Path
    from datetime import datetime, timezone
    from google.colab import drive, files
    drive.mount('/content/drive')
    BUNDLE_PATH = None  # 예: '/content/drive/MyDrive/MI/inputs/' + BUNDLE_NAME
    if BUNDLE_PATH is None:
        uploaded = files.upload()
        assert BUNDLE_NAME in uploaded, 'Upload the named bundle'
        BUNDLE_PATH = '/content/' + BUNDLE_NAME
        del uploaded
    bundle = Path(BUNDLE_PATH)
    def file_sha(path):
        h = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    assert file_sha(bundle) == EXPECTED_SHA256, 'Wrong or incomplete input ZIP'
    base = Path('/content/boolean_interp')
    base.mkdir(exist_ok=True)
    ROOT = Path(tempfile.mkdtemp(prefix='v1_4_', dir=base))
    with zipfile.ZipFile(bundle) as archive:
        for entry in archive.infolist():
            path = ROOT / entry.filename
            assert path.resolve().is_relative_to(ROOT.resolve())
            assert (entry.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink not allowed'
        inventory = json.loads(archive.read('bundle_manifest.json'))
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(inventory['files']) | {'bundle_manifest.json'}
        archive.extractall(ROOT)
    for name, expected in inventory['files'].items():
        assert file_sha(ROOT / name) == expected, name
    os.chdir(ROOT)
    print('Verified input:', EXPECTED_SHA256, ROOT)
    """)
    md("""## 2. 의존성·환경·입력 계약 확인

    설치된 CUDA PyTorch를 우선 유지합니다. 정확한 버전은 GPU smoke의 lock/environment에
    기록합니다. 재개 시 이전 lock과 비교하고 환경이 바뀌면 새 environment ID로 기록됩니다.
    """)
    code("""
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-interp.txt'], check=True)
    sys.path.insert(0, str(ROOT))
    import torch
    assert torch.cuda.is_available(), 'Select a GPU runtime before continuing'
    subprocess.run(['nvidia-smi'], check=True)
    print('Disk:', shutil.disk_usage(ROOT))
    print('RAM bytes:', os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    from interp_v1_4.runtime import verify_inputs
    print(verify_inputs(ROOT))
    child_env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    subprocess.run([sys.executable, '-m', 'pytest', 'tests_v1_4', '-q'], check=True, env=child_env)
    """)
    md("""## 3. 필수 GPU smoke

    최대 길이 302, masking, read4 accumulation, bitwise resume, 모든 층 hook,
    SAE/TC k=4/16 각각 100 updates, probe, identity/full/sparse patch를 검사합니다.
    Debug checkpoint는 본학습 초기값으로 사용하지 않습니다.
    """)
    code("""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    SMOKE = ROOT / 'gpu_smoke' / stamp
    subprocess.run([sys.executable, '-m', 'interp_v1_4.smoke', '--root', str(ROOT),
                    '--device', 'cuda', '--output', str(SMOKE)], check=True)
    smoke = json.loads((SMOKE / 'smoke.json').read_text())
    assert smoke['status'] == 'passed' and smoke['scope'] == 'cuda'
    MICRO = smoke['cells']['deepwide12_read4']['selected_microbatch']
    print('Frozen microbatch:', MICRO, 'effective batch: 64')
    """)
    md("""## 4. 영속 저장 및 fresh/resume 준비

    같은 run 이름으로 새 학습을 덮어쓰지 않습니다. Drive에 run이 이미 있으면 RESUME=True로
    처음부터 새 런타임에서 진행합니다. 이전 GPU와 microbatch가 맞지 않으면 임의 변경 없이 중단합니다.
    """)
    code("""
    from interp_v1_4.persistence import recover, read_index
    PERSIST = Path('/content/drive/MyDrive/MI/v1_4/deepwide12_read4_seed0')
    OUTPUT = ROOT / 'runs' / ('seed0_' + stamp)
    resume_args = []
    if RESUME:
        checkpoint = recover(PERSIST, OUTPUT)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['microbatch'] == MICRO, 'GPU smoke/checkpoint microbatch mismatch'
        assert not (OUTPUT / 'result.json').exists(), 'Run finalized; export evidence, do not train again'
        resume_args = ['--resume', str(checkpoint)]
        del payload
    else:
        assert not PERSIST.exists(), 'Persistent run exists: use RESUME, never overwrite'
    """)
    md("""## 5. seed 0 학습

    최초 50 updates 처리량과 VRAM을 기록합니다. init, milestone 및 15분 경과 update의 완전
    checkpoint를 Drive에 저장합니다. 중단/OOM 시 batch·예산을 줄이지 말고 index에서 재개합니다.
    학습 완료 뒤 gate 결과가 failed여도 증빙 ZIP을 회수합니다. test는 채점하지 않습니다.
    """)
    code("""
    command = [sys.executable, '-u', '-m', 'interp_v1_4.cli', '--root', str(ROOT),
               '--stage', 'p3', '--cell', 'deepwide12_read4', '--lm-seed', '0', '--device', 'cuda',
               '--microbatch', str(MICRO), '--smoke-report', str(SMOKE / 'smoke.json'),
               '--output', str(OUTPUT), '--persistent-dir', str(PERSIST)] + resume_args
    subprocess.run(command, check=True)
    result = json.loads((OUTPUT / 'result.json').read_text())
    print({key: result[key] for key in ('status', 'actual_prediction_tokens', 'overshoot')})
    """)
    md("""## 6. 완전 저장 증빙 검증·다운로드 (중단 시에도 실행)

    Drive의 마지막 완전 index만 export합니다. 완료 run은 gate를 재채점하지 않고 recorded gate,
    checkpoint, cursor, hash를 검증합니다. 중단 run은 paused로 회수하며 완료로 표시하지 않습니다.
    ZIP과 checksum은 Drive에도 보관됩니다. 반환 파일을 로컬 감사에 전달해 주세요.
    """)
    code("""
    export_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    EXPORT = ROOT / ('evidence_' + export_stamp)
    EXPORTED_RUN = EXPORT / 'deepwide12_read4_seed0'
    recover(PERSIST, EXPORTED_RUN)
    shutil.copytree(SMOKE, EXPORT / 'gpu_smoke')
    if (EXPORTED_RUN / 'result.json').exists():
        subprocess.run([sys.executable, 'scripts/verify_v1_4_evidence.py', str(EXPORTED_RUN),
                        '--output', str(EXPORT / 'verification.json')], check=True)
    else:
        (EXPORT / 'resume_required.json').write_text(json.dumps({'status': 'paused', 'index': read_index(EXPORTED_RUN)}))
    checksums = {str(p.relative_to(EXPORT)): file_sha(p) for p in sorted(EXPORT.rglob('*')) if p.is_file()}
    (EXPORT / 'checksums.json').write_text(json.dumps(checksums, indent=2))
    zip_path = Path(shutil.make_archive(str(ROOT / ('v1_4_seed0_evidence_' + export_stamp)), 'zip', EXPORT))
    checksum = file_sha(zip_path)
    checksum_path = zip_path.with_suffix('.zip.sha256')
    checksum_path.write_text(checksum + '  ' + zip_path.name + '\\n')
    DELIVERY = PERSIST.parent / 'exports'
    DELIVERY.mkdir(exist_ok=True)
    shutil.copy2(zip_path, DELIVERY / zip_path.name)
    assert file_sha(DELIVERY / zip_path.name) == checksum
    shutil.copy2(checksum_path, DELIVERY / checksum_path.name)
    files.download(str(checksum_path))
    files.download(str(zip_path))
    print('Drive evidence:', DELIVERY / zip_path.name)
    """)
    return dict(nbformat=4, nbformat_minor=4, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python"), accelerator="GPU"), cells=cells)


def build(revision, draft=False):
    folder = ROOT / 'experiment_v1_4'
    bundle_name = f'v1_4_p3_bundle_{revision}.zip'
    destination = folder / 'bundles' / bundle_name
    nb_path = folder / 'notebooks' / (f'P3_v1_4_{revision}' + ('_DRAFT' if draft else '') + '.ipynb')
    if nb_path.exists() or (destination.exists() and not draft):
        raise FileExistsError('Immutable delivery revision exists')
    if draft:
        digest = 'UNAVAILABLE_PENDING_P1_AND_CPU_SMOKE'
    else:
        checked = verify_inputs(ROOT)
        report_path = folder / 'smoke/cpu_01/smoke.json'
        report = json.loads(report_path.read_text())
        assert report['status'] == 'passed' and report['scope'] == 'cpu'
        for key in ('configs', 'corpus_manifest'):
            assert report['input_hashes'][key] == checked[key]
        for name, expected in report['input_hashes']['code'].items():
            assert sha(ROOT / name) == expected, name
        data = ROOT / json.loads((folder / 'configs/run.json').read_text())['data_root']
        selected = list(data.rglob('*'))
        for directory in ('corpus', 'interp_v1_2', 'interp_v1_4', 'tests_v1_4'):
            selected += list((ROOT / directory).glob('*.py'))
        selected += list((folder / 'configs').glob('*.json'))
        selected += list((folder / 'smoke/cpu_01').glob('*.json'))
        selected += list((folder / 'smoke/cpu_01').glob('*.txt'))
        selected += [ROOT / name for name in (
            'requirements-interp.txt', 'experiment_v1_2/debug/sequences.json',
            'experiment_v1_3/results/next_architecture_proposal.json',
            'experiment_v1_4/DESIGN.md', 'experiment_v1_4/design_config.json',
            'experiment_v1_4/design_manifest.json', 'experiment_v1_4/corpus_rebuild.json',
            'scripts/verify_v1_4_evidence.py',
            'scripts/generate_v1_4_corpus.py',
        )]
        # Preserve the generator actually used for rebuild_01, separately from
        # subsequent audit/attempt-limit fixes, with final independent evidence.
        audit = folder / 'results/audit_20260921_01'
        assert json.loads((audit / 'corpus/summary.json').read_text())['status'] == 'passed_data_integrity'
        assert json.loads((audit / 'attempts_02/summary.json').read_text())['status'] == 'passed'
        assert json.loads((audit / 'policy.json').read_text())['status'] == 'passed'
        audit_files = [audit / name for name in (
                'REPORT.md', 'generator_at_creation.py', 'provenance.json',
                'existing_cpu_verified.json', 'corpus/full_replay.json',
                'corpus/summary.json', 'attempts_02/summary.json',
                'policy.json', 'language_self_test.json',
        )]
        assert all(path.is_file() for path in audit_files)
        selected += audit_files
        files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(selected)) if p.is_file() and p.name != '.DS_Store'}
        destination.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
            for name in files:
                archive.write(ROOT / name, name)
            archive.writestr('bundle_manifest.json', json.dumps(dict(schema='v1.4-input-bundle', files=files), indent=2))
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None
            for name, expected in files.items():
                h = hashlib.sha256()
                with archive.open(name) as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        h.update(block)
                assert h.hexdigest() == expected, name
        digest = sha(destination)
        destination.with_suffix('.zip.sha256').write_text(digest + '  ' + bundle_name + '\n')
    nb_path.parent.mkdir(exist_ok=True)
    nb_path.write_text(json.dumps(notebook(bundle_name, digest, draft), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(notebook=str(nb_path), draft=draft, bundle_sha256=digest), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='r1')
    parser.add_argument('--draft', action='store_true')
    args = parser.parse_args()
    build(args.revision, args.draft)
