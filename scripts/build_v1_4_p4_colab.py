"""Build P4 replication delivery from the immutable, checksum-verified r3 input bundle."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_v1_4_colab import notebook
from interp_v1_4.cli import stage_contract
from interp_v1_4.replication import AUDIT, replication_authorization
from interp_v1_4.runtime import sha, verify_inputs

BASE_SHA = '2b0613c412458a1c4a636b6421efebde9cfbf6a771b7294002a537ed83d56547'


def replication_notebook(bundle_name, digest, seed):
    nb = notebook(bundle_name, digest)
    for cell in nb['cells']:
        source = cell['source']
        source = source.replace('seed0', f'seed{seed}').replace('seed 0', f'seed {seed}')
        source = source.replace('v1.4 P3', 'v1.4 P4').replace("'--stage', 'p3'", "'--stage', 'p4'")
        source = source.replace("'--lm-seed', '0'", f"'--lm-seed', '{seed}'")
        source = source.replace("/MI/v1_4/deepwide12_read4_", "/MI/v1_4/p4_r1/deepwide12_read4_")
        if cell['cell_type'] == 'code' and 'MICRO = smoke' in source:
            source += "assert MICRO == 16, 'P4 must preserve seed 0 microbatch; use a suitable GPU'\n"
        if cell['cell_type'] == 'code' and "'pip', 'install'" in source:
            source = source.replace("'-r', 'requirements-interp.txt'", "'-r', 'experiment_v1_4/p4/requirements-primary.lock.txt', '-c', 'experiment_v1_4/p4/requirements-seed0.constraints.txt', '--extra-index-url', 'https://download.pytorch.org/whl/cu128'")
        if cell['cell_type'] == 'code' and 'resume_args = []' in source:
            source = source.replace("assert not (OUTPUT / 'result.json').exists(), 'Run finalized; export evidence, do not train again'", "# Finalized runs can be recovered for export without another training/gate call.")
        if cell['cell_type'] == 'code' and 'subprocess.run(command, check=True)' in source:
            source = source.replace('subprocess.run(command, check=True)', "if not (OUTPUT / 'result.json').exists():\n    subprocess.run(command, check=True)")
        if cell['cell_type'] == 'code' and 'shutil.copytree(SMOKE' in source:
            source = source.replace("shutil.copytree(SMOKE, EXPORT / 'gpu_smoke')", "shutil.copytree(SMOKE, EXPORT / 'gpu_smoke')\nshutil.copytree(ROOT / 'experiment_v1_4/p4', EXPORT / 'delivery_contract')\n(EXPORT / 'input_bundle.json').write_text(json.dumps({'name': BUNDLE_NAME, 'sha256': EXPECTED_SHA256}))")
        cell['source'] = source
    nb['cells'][0]['source'] = f'''# v1.4 P4 · seed {seed} 재현 · 64M

이 노트북은 seed {seed} 하나를 fresh initialization에서 실행합니다. 같은 입력 번들로
seed 1과 seed 2 노트북을 각각 실행하세요. seed 1의 gate 실패도 seed 2를 생략할 이유가 아닙니다.
각 seed는 64,005,751 prediction tokens / 8,399 updates / cursor 537,536을 소비합니다.
모델·read4·학습률·데이터 순서·effective batch 64·microbatch 16은 seed 0과 동일합니다.

감사된 seed 0 동결 증빙을 확인한 뒤 현재 코드로 GPU smoke를 수행합니다. 지정 milestone의
select first macro CE로 checkpoint 하나를 선택하고 validation gate를 한 번 평가합니다.
실패 seed도 대체하지 않고 반환합니다. 두 seed가 끝난 뒤 반환 증빙을 로컬에서 감사하고
세 seed 전체의 결정을 동결해야 최종 test 노트북을 실행할 수 있습니다.
**이 노트북은 test 평가를 실행하지 않습니다. P4 완료가 아닌 재현 실행 전달물입니다.**

새 Colab CUDA 런타임에서 위→아래 실행합니다. 각 seed의 Drive 경로는 별개입니다.
중단 시 `RESUME=True`로 새 런타임에서 마지막 완전 index를 복구합니다.
완료 run도 `RESUME=True`로 회수할 수 있으며 학습·gate를 다시 실행하지 않습니다.
`gate_started.json`만 있고 gate 완료물이 없으면 반복 평가 없이 증빙을 반환하세요.
'''
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown' and '## 2.' in cell['source']:
            cell['source'] = '''## 2. seed 0의 기록된 환경을 기준으로 설치·검증

seed 0의 전체 lock에서 추출한 constraints와 주요 패키지의 정확한 버전을 사용합니다.
전체 원본 lock도 번들에 보존합니다. 같은 버전 설치가 불가능하면 오류를 그대로 반환해 주세요.
GPU/Python/시스템 차이는 smoke와 각 학습 세션의 environment ID에 기록됩니다.
'''
        if cell['cell_type'] == 'code':
            compile(cell['source'], f'P4 seed {seed} notebook', 'exec')
    return nb


def build():
    folder = ROOT / 'experiment_v1_4'
    prep = folder / 'results/p4_preparation_r1'
    base = folder / 'bundles/v1_4_p3_bundle_r3.zip'
    destination = folder / 'bundles/v1_4_p4_bundle_r1.zip'
    if destination.exists():
        raise FileExistsError(destination)
    checked = verify_inputs(ROOT)
    replication_authorization(ROOT, stage_contract(ROOT, 'p4', 'deepwide12_read4'), checked, 16)
    smoke_path = folder / 'smoke/cpu_p4_r1/smoke.json'
    smoke = json.loads(smoke_path.read_text())
    assert smoke['status'] == 'passed' and smoke['scope'] == 'cpu'
    for key in ('configs', 'corpus_manifest'):
        assert smoke['input_hashes'][key] == checked[key]
    for name, digest in smoke['input_hashes']['code'].items():
        assert sha(ROOT / name) == digest
    for seed in (1, 2):
        report = json.loads((folder / f'smoke/resume_p4_seed{seed}_r1/resume_equivalence.json').read_text())
        assert report['status'] == 'passed' and report['lm_seed'] == seed and report['stage'] == 'p4'
    assert sha(base) == BASE_SHA
    additions = []
    for directory in ('interp_v1_4', 'tests_v1_4'):
        additions += list((ROOT / directory).glob('*.py'))
    additions += [ROOT / name for name in ('scripts/verify_v1_4_evidence.py', 'scripts/check_v1_4_resume.py',
                                          'scripts/build_v1_4_p4_colab.py', 'scripts/build_v1_4_colab.py')]
    audit = ROOT / AUDIT
    completion = json.loads((audit / 'completion.json').read_text())
    additions += [audit / 'completion.json'] + [audit / name for name in completion['evidence_sha256']]
    additions += [audit / 'gpu_smoke/requirements.lock.txt']
    additions += list(smoke_path.parent.glob('*.json')) + list(smoke_path.parent.glob('*.txt'))
    additions += [folder / f'smoke/resume_p4_seed{seed}_r1/resume_equivalence.json' for seed in (1, 2)]
    additions += [prep / 'tests.xml', prep / 'tests_anaconda.log']
    lock = (audit / 'gpu_smoke/requirements.lock.txt').read_text()
    constraints = '\n'.join(line for line in lock.splitlines() if '==' in line and not line.startswith('#')) + '\n'
    packages = {'torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'pytest'}
    primary = '\n'.join(line for line in constraints.splitlines() if line.split('==')[0] in packages) + '\n'
    assert len(primary.splitlines()) == 6
    payloads = {str(path.relative_to(ROOT)): path.read_bytes() for path in additions}
    payloads['experiment_v1_4/p4/requirements-primary.lock.txt'] = primary.encode()
    payloads['experiment_v1_4/p4/requirements-seed0.constraints.txt'] = constraints.encode()
    payloads['experiment_v1_4/p4/contract.json'] = json.dumps(dict(
        schema='v1.4-p4-replication-r1', seeds=[1, 2], seed0_audit_completion_sha256=sha(audit / 'completion.json'),
        frozen_config_sha256=checked['configs'], corpus_manifest_sha256=checked['corpus_manifest'],
        changes=['P4 seed authorization', 'CLI stage routing', 'return verification', 'notebook packaging'],
        numerical_code_unchanged=True, microbatch=16, effective_batch=64,
        actual_tokens_per_seed=64005751, updates_per_seed=8399, cursor_per_seed=537536,
        test_evaluation='Separate delivery after both replication returns are audited and all decisions frozen',
        seed0_requirements_sha256=sha(audit / 'gpu_smoke/requirements.lock.txt'),
    ), indent=2).encode()
    inventory = {}
    with zipfile.ZipFile(base) as source, zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as target:
        old = json.loads(source.read('bundle_manifest.json'))['files']
        assert set(source.namelist()) == set(old) | {'bundle_manifest.json'}
        for name, expected in old.items():
            if name in payloads:
                assert hashlib.sha256(source.read(name)).hexdigest() == expected, name
                data = payloads.pop(name)
                target.writestr(name, data)
                inventory[name] = hashlib.sha256(data).hexdigest()
            else:
                value = hashlib.sha256()
                with source.open(name) as reader, target.open(name, 'w', force_zip64=True) as writer:
                    for block in iter(lambda: reader.read(1024 * 1024), b''):
                        value.update(block)
                        writer.write(block)
                assert value.hexdigest() == expected, name
                inventory[name] = expected
        for name, data in payloads.items():
            target.writestr(name, data)
            inventory[name] = hashlib.sha256(data).hexdigest()
        target.writestr('bundle_manifest.json', json.dumps(dict(schema='v1.4-p4-input-bundle', files=inventory), indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name, expected in inventory.items():
            value = hashlib.sha256()
            with archive.open(name) as reader:
                for block in iter(lambda: reader.read(1024 * 1024), b''):
                    value.update(block)
            assert value.hexdigest() == expected, name
    digest = sha(destination)
    destination.with_suffix('.zip.sha256').write_text(digest + '  ' + destination.name + '\n')
    notebooks = {}
    for seed in (1, 2):
        path = folder / f'notebooks/P4_v1_4_seed{seed}_r1.ipynb'
        with path.open('x') as handle:
            json.dump(replication_notebook(destination.name, digest, seed), handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        notebooks[str(path.relative_to(ROOT))] = sha(path)
    report = dict(status='passed_delivery_checks', bundle=str(destination.relative_to(ROOT)), bundle_sha256=digest,
                  bytes=destination.stat().st_size, members=len(inventory), all_member_checksums_verified=True,
                  notebooks=notebooks, notebook_code_cells_compiled=14, frozen_inputs=checked,
                  gpu_replication='not_run', test_scores_observed=False)
    (prep / 'delivery.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    build()
