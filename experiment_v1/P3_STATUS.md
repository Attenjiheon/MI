# P3 LM seed 0 파일럿 상태

갱신일: 2026-09-16 (KST)  
적용 규격: `experiment-spec-v1.0`  
상태: `failed` — 3M 행동 gate 미달, 실행·판정 종료; P4 진입 금지

## 선행 조건과 완료 조건

- P1: immutable `data/language_v1/`의 CPU validation/postwrite audit는 passed다. 배포 번들을 실제로 풀어 manifest의 107개 데이터 파일과 P0 config hash를 다시 검증했다.
- P2: CPU와 Tesla T4 GPU smoke의 통과 증빙은 `P2_STATUS.md`와 `smoke/colab_gpu_01/smoke.json`에 있다. P3 코드 변경을 P2 원본 코드와 동일하다고 주장하지 않는다. 노트북이 현재 코드로 GPU smoke를 다시 실행한다.
- 목표: seed 0을 고정 train shard 순서로 1M 경계까지 실행하고 최소 일반 validation 답 CE checkpoint를 선택한다.
- 완료 조건: 일반 full-vocabulary 정확도 ≥99%, 3개 진단 각각 ≥95% 판정과 체크포인트/hash, 실제 token/cursor/update, overshoot, 첫 50-update 처리량·VRAM, 연장·동결·중단 근거가 필요하다.

## 전달 파일

- 노트북: `experiment_v1/notebooks/01_colab_lm.ipynb`
- 입력 번들: `experiment_v1/bundles/p3_colab_bundle_v4.zip` (120,853,698 bytes)
- SHA-256: `1068fce825c2409acc606264644e792902818a9886c57fc583fe94ecb94a5a1a`
- checksum 파일: `experiment_v1/bundles/p3_colab_bundle_v4.sha256`
- 재생성 스크립트: `scripts/build_p3_colab.py` (기존 버전 덮어쓰기 거부)

새 Colab GPU 런타임에서 노트북을 열고 위부터 실행한다. 번들 업로드 후 전체 파일 checksum 검증, 직접 의존성 설치, 실제 환경 기록, pytest·GPU smoke, 본실험, 결과 ZIP 다운로드가 이어진다. Drive 기본 경로는 `MyDrive/boolean_interp_p3_v4`다. 중단 시 새 런타임에서 같은 경로와 `RESUME=True`를 사용한다.

## 구현과 재개 계약

- 규정된 1M gate와 마지막 4평가의 개선 조건에 따라 같은 run의 last checkpoint에서만 누적 3M까지 한 번 연장한다. Seed 1/2는 P4 동결 계약 전 실행을 거부한다.
- 일반 validation의 답 CE 최소값으로 선택하며 동률이면 이전 checkpoint를 유지한다. Test는 읽어 평가하지 않는다.
- 각 validation과 15분 저장 경계에서 init/best/last·로그·환경을 일관된 Drive snapshot으로 복사하고 checksum 확인 뒤 `LATEST.json`을 갱신한다. 불완전한 복사본은 복구 대상이 아니다.
- 복구는 새 로컬 디렉터리로 수행한다. cursor/token/update를 대조하고, 미저장 update 로그는 별도로 보존한 뒤 마지막 저장 update까지만 재사용한다.
- OOM은 model/optimizer/RNG/update를 복구한 뒤 microbatch를 절반으로 줄인다. Effective batch 64는 유지한다.
- 첫 50 updates 학습 시간·prediction tokens·peak allocated VRAM, update별 학습 시간, validation 시간, 세션별 환경·명령, 1M 및 3M 단계별 결정/result/manifest를 기록한다.
- GPU smoke의 debug 산출물을 본실험에 사용하지 않는다.

## 수정·이전 기록 보존

기존 P3 상태 문서는 `01_colab_lm.ipynb`와 `p3_colab_bundle_v3.zip` 준비 완료를 적고 있었지만 이번 세션 시작 시 실제 파일은 없었다. 따라서 v3의 재사용·검증을 주장하지 않고 v4를 새로 만들었다. 기존 문서의 Colab 전체 freeze 설치 실패 기록에 따라 전체 `requirements-colab.lock.txt`를 설치 입력으로 쓰지 않는다. 직접 의존성을 설치하고 실제 환경 ID를 기록하며 GPU smoke 통과를 요구한다.

기존 P3 구현에 누락된 `sys` import(본실험 manifest 작성 경로), 중단 후 로그 처리, 영속 복사 중단 시 복구 일관성을 보완했다. LM 구조·데이터·학습 예산·gate·P0 config는 변경하지 않았다. Colab 파일은 `.ipynb`로 전달한다는 지침을 `AGENTS.md §7`에 추가했다.

## 검증 증빙

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-p2/bin/python -m pytest tests -q`: **23 passed** (`smoke/p3_pytest_cpu.txt`).
- 실제 ZIP을 임시 디렉터리에 풀어 번들 152개 파일 checksum 검증, 같은 테스트 **23 passed**, corpus/config 등록 파일 107개 검증: `smoke/p3_bundle_validation.json`.
- 노트북 코드 셀은 Python compile 검사 통과. Colab 셀은 로컬에서 실행했다고 주장하지 않는다.
- 추가 CPU smoke `smoke/p3_preflight_cpu_01/`는 수치 검사 뒤 환경 기록의 `pip freeze`가 3분 이상 지연되어 중단했다. `attempt_status.json`에 보존하며 통과로 집계하지 않는다. Colab의 필수 GPU smoke는 별도로 실행한다.
- 생산 manifest 테스트는 CPU에서 모의 update/metrics로 실행한 계약 테스트다. 그 값은 본실험 측정치가 아니다.

## 최종 증빙 판정 (2026-09-16)

Colab 결과를 수령해 CPU 재평가와 전체 코퍼스 재감사를 완료했다. 1M 연장 조건 충족 후 3M을 실행했으나 일반 정확도 61.44%, 세 진단 61.52% / 64.26% / 61.33%로 행동 gate에 미달했다. 407 updates, 3,004,531 tokens, overshoot 4,531, 최종 cursor 26,048이며 최소 일반 답 CE checkpoint는 update 407이다.

- 중단 보고: `results/p3_stop_report.md`
- 증빙 감사: `results/p3_evidence_verification.json`
- CPU 전체 코퍼스 재검산: `results/p3_cpu_corpus_reaudit.json` (passed)
- 최종 구현 테스트: `results/p3_postrun_pytest.txt` (23 passed)
- 원본·추출본: `evidence/p3_evidence_20260915T174256751247.zip`, `evidence/p3_colab_20260915T174256751247/`
- 곡선: `results/p3_behavior_validation.csv`, 추출본 `lm_seed_0/learning_curve.png`

실험을 성공 처리하지 않는다. 1M checkpoint 바이트와 Drive snapshot은 제출 ZIP에 없으며 해당 시점의 hash·수치·로그만 검증했다. 최종 best/last/init는 직접 검증했다. 새 실험/추가 연장 없이 행동 학습 단계 중단 보고로 종료한다.

## Colab 업로드 파일명 오류 수정 (2026-09-16)

사용자 스크린샷에서 Colab이 파일을 `p3_colab_bundle_v4 (3).zip`으로 저장한 뒤 원래 이름을 찾는 검사에서 실패했다. 노트북 업로드 셀을 파일명 대신 업로드 바이트의 SHA-256으로 선택하도록 수정했다. Drive 복사도 임시 파일에 쓰고 checksum을 확인한 뒤 확정한다. 수정본은 `notebooks/01_colab_lm_upload_fix.ipynb`이며 기본 `01_colab_lm.ipynb`와 생성 스크립트에도 반영했다. 기존 v4 ZIP과 실험 코드는 변경하지 않았다.

수정된 실제 셀의 업로드 구간을 임시 디렉터리에서 실행해 원래 이름·`(3)`이 붙은 이름의 정상 처리와 잘못된 내용의 거부를 확인했다. 전체 코드 셀 compile도 통과했다. GPU 학습 실행 여부나 gate 통과를 이 검사로 판정하지 않는다.
