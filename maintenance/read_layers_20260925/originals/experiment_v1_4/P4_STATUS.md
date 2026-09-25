# v1.4 P4 — 세 seed frozen test 반환 감사 완료

2026-09-22. **P4 완료. 세 seed의 학습·validation 동결과 21개 frozen test 반환 감사를 마쳤다. P5 진입 조건을 충족하며 P5는 아직 미실행이다.**
P3 seed 0의 감사된 gate 통과와 64M 예산 동결이 선행 증빙이다.

현재 증빙은 [최종 분석](results/frozen_test_audit_20260922_01/REPORT.md),
[완료 manifest](results/frozen_test_audit_20260922_01/completion.json),
[동결 모델 목록](results/frozen_test_audit_20260922_01/frozen_lms.json)이다.
일반 READ는 seed 0/1/2 각각 99.9410% / 99.9472% / 99.9659%, first macro는
99.9442% / 99.9442% / 99.9628%, repeat는 모두 100%다.
Validation gate로 확정한 세 모델을 모두 유지하며 test로 재선택하지 않았다.
아래 2026-09-21 준비·반환 절은 단계별 이력이다.

## 실행 계약

- seed 1·2를 각각 fresh initialization에서 실행하며 실패 seed도 보존한다.
- 같은 12×256 / read4 / optimizer / LR / FP32 / train 순서 / effective batch 64.
- seed 0과 같은 microbatch 16, 64,005,751 prediction tokens / 8,399 updates / cursor 537,536.
- 같은 10개 milestone에서 select first-member 42-cell macro answer CE 전역 최소,
  최소와 1e-4 nats 이내는 select/general answer CE → 이른 update 순으로 선택한다.
- 선택 checkpoint에서 seed별 validation gate 1회. gate/test로 재선택하지 않는다.

## 전달·검증 순서

1. [seed 1 노트북](notebooks/P4_v1_4_seed1_r1.ipynb)과
   [seed 2 노트북](notebooks/P4_v1_4_seed2_r1.ipynb)을 각각 새 CUDA 런타임에서 실행한다.
   두 노트북은 [공통 번들](bundles/v1_4_p4_bundle_r1.zip)을 사용한다.
2. 입력 전체·내부 checksum → seed 0 lock 기반 설치 → 현재 코드 GPU smoke →
   학습·선택·one-time gate → Drive 완전 index 검증·ZIP 반환 순서다.
3. seed 1이 gate에 실패해도 seed 2를 실행하고 둘 다 반환한다. 중단 시 같은 seed의
   노트북에서 `RESUME=True`로 마지막 완전 Drive index를 복구한다.
4. 두 반환물의 checksum·환경·초기화·checkpoint·학습량·select/gate를 로컬 감사한다.
5. 세 seed 전체의 학습·validation 결정과 checkpoint hash를 동결한다.
6. 동결 증빙에 결합한 별도 최종 test 노트북을 준비하여 **학습한 모든 seed**를 한 번 평가한다.
   실패 seed도 포함한다. 이번 재현 노트북은 test 채점 코드를 호출하지 않는다.
7. test 반환까지 검증하고 통과 모델 목록·실패 사유·registry를 확정해야 P4 완료다.

## 상태와 증빙

로컬 준비 검증의 상세 상태는 [준비 보고서](results/p4_preparation_r1/REPORT.md),
[검증 manifest](results/p4_preparation_r1/verification.json)를 따른다.
GPU 실행 결과를 로컬 준비 통과로 대신하지 않는다.
P5는 P4 완료와 validation gate 최소 두 seed 통과가 모두 필요하다.

33 tests, 현재 frozen-input CPU smoke, 두 seed의 실제 CLI persistent-index resume 및 반환 감사,
서로 다른 초기화·cross-seed resume 거부, ZIP 421개 내부 파일 hash와 14개 notebook 코드 셀 검증 통과.
이는 GPU 본실험 결과가 아니다.

디스크 정리 후 전달 ZIP·노트북·코드·준비 증빙 해시 재검증도 통과했다
([재검증](results/p4_preparation_r1/post_cleanup_verification.json)). 완료된 로컬 debug checkpoint는
사용자 승인 정리로 삭제됐으며 성공 기록과 해시는 보존했다. 실제 본학습 입력에는 영향이 없다.

## seed 1·2 반환 감사 완료

[감사·분석 보고서](results/p4_audit_20260921_01/REPORT.md),
[검증 manifest](results/p4_audit_20260921_01/completion.json),
[세 seed validation 동결](results/p4_audit_20260921_01/validation_freeze.json)을 따른다.
두 seed 모두 64,005,751 tokens / 8,399 updates를 소비했고 모든 gate·quota가 통과했다.
일반 READ는 seed 1 99.9217%, seed 2 99.9699%; first macro는 99.8884% / 100%다.
Seed 2는 init checkpoint에서 재개했으며 최종 token/cursor 중복은 없다.
로컬 Mac과 Colab 초기화에는 최대 약 5e-8의 FP32 차이가 있어 교차환경 검사로 별도 기록했다.
이 반환 감사 당시 다음 작업은 세 seed 전체의 one-time frozen test였으며,
2026-09-22 최종 반환 감사로 위 현재 상태에 갱신됐다.

## Frozen test 전용 전달물 (2026-09-21)

[최종 test 노트북](notebooks/P4_v1_4_frozen_test_r1.ipynb)과
[전용 입력 번들](bundles/v1_4_frozen_test_bundle_r1.zip)을 새 CUDA 런타임에서 실행한다.
준비 증빙은 [verification.json](results/frozen_test_preparation_r1/verification.json),
평가 계약은 [contract.json](frozen_test_r1/contract.json)을 따른다.

- 세 seed의 선택된 원본 checkpoint와 7개 suite의 metadata만 포함한다. Train shard는 필요 없다.
- seed당 일반 READ, legacy 3종, first/repeat 42-cell×128쌍, composition 2종, 총 21개 평가다.
- 기존 모델·행동·보고 수치 코드는 보존하고 새 orchestration 및 test 전용 집계만 추가했다.
- 6개 테스트와 미학습 모델 CPU preflight를 통과했다. Microbatch 16 / 길이 302 fixture도 확인했다.
- 결과별 시작 기록과 원시 측정값을 Drive에 저장한다. 완료 결과는 재사용하며 저장된 raw는
  집계만 재개한다. 시작 기록만 남은 평가는 추론을 자동 반복하지 않고 부분 증빙을 반환한다.
- 이 전달 당시 실제 GPU frozen test 및 반환 감사는 미실행이었고, 2026-09-22 완료됐다.
