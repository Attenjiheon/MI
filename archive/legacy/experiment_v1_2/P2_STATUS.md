# v1.2 P2 — passed (CPU + GPU)

2026-09-17 반환 ZIP 감사 완료: 실제 Colab CPU/GPU smoke와 8개 테스트를 확인해 P2를 passed로 기록한다. 실행에는 보존된 v1 입력 번들을 사용했으며 최종 배포 v2와 학습·평가 데이터 및 실행 코드가 동일함을 검증했다. [P3 중단 보고](results/p3_stop_report.md)의 번들 출처 구분을 따른다.

GPU: Tesla T4, 환경 `e74fb1dcf8112ca0`, 9.754초. GPU smoke의 config/data/code/환경 ID가 본학습과 일치한다. 증빙: `evidence/p3_16m_20260917/preflight/20260917T073251285323/cuda/smoke.json`, 환경 lock 및 `results/p3_supplemental_audit.json`. 아래 준비 기록은 당시 이력이다.

- `results/p2_pytest_release.txt`: 테스트 **8 passed**, 14.75초. 고정 16M 선택/gate, validation 전용 입력, short shard 연결과 마지막 완전 batch, 증분 영속 저장/불완전 복사, 실제 optimizer/RNG 재개, v1.1 초기화 동일성, 평가 경계 복원 검증.
- `smoke/cpu_release/smoke.json`: **passed**, 9.82초, 환경 `d7c07004dd4c1bd4`. 797,184 parameters, causal/PAD/RoPE, token-weighted gradient, 네 block hooks, dictionary/probe/patching, checkpoint 다음 update bitwise 재현.
- 최종 smoke의 모든 `interp_v1_2/*.py` hash를 작업 공간 최종 코드와 다시 대조했다.
- `smoke/cpu_release/requirements.lock.txt`, `nvidia-smi.txt`: 실제 CPU 환경. GPU 환경은 노트북 실행 시 새로 기록한다.

## 실행 명령과 운영 기록

프로젝트와 `.venv-p2` 파일이 iCloud `dataless`로 전환되어 파일 읽기에 수분 이상 대기했다. 같은 config/code/data 바이트를 `/private/tmp/mi_v12_validation`에 복사한 뒤 시스템 Python 3.13.4 / torch 2.13.0 / numpy 2.2.6으로 검증했다. pytest 9.1.1 및 직접 의존성은 `/private/tmp/mi_v12_pytest`에 설치했다. 본실험 데이터나 모델 규격을 바꾸지 않았다.

```sh
cd /private/tmp/mi_v12_validation
PYTHONPATH=/private/tmp/mi_v12_pytest PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest tests_v1_2 -q
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m interp_v1_2.smoke --device cpu --output experiment_v1_2/smoke/cpu_release
```

초기 `cpu_01`은 config manifest 생성 전 실행되어 실패했다. `cpu_02`는 CPU smoke에 통과했지만 이후 집계/중단 기록 코드 변경이 있어 최종 증빙으로 대체하지 않았다. 초기 pytest의 1 failure는 임시 디렉터리 이름의 `test_`를 평가 split 이름으로 오인한 테스트 오류였으며, root-relative 데이터 경로 검사로 수정한 뒤 전체 8개를 통과했다. 초기 로그도 보존한다.

## 준비 당시 GPU 검사 절차 (실행·감사 완료)

`notebooks/01_colab_lm_16m.ipynb`는 전체 입력 checksum → 의존성/환경 → CPU tests/smoke → 새 CUDA smoke → 16M 학습을 순서대로 실행한다. GPU smoke의 code/config/data hash 및 환경 ID가 학습과 일치해야 runner가 본학습을 허용한다. Debug checkpoint는 본실험에 재사용하지 않는다.

Finder sidecar를 제외한 최종 배포 manifest/config로 8개 테스트와 CPU smoke를 다시 통과했다. 이전 config/manifest와 배포 checksum은 `evidence/prepackaging_v1`에 보존한다.
