# v1.4 P4 — seed 1·2 재현 로컬 준비 완료

2026-09-21. **P4 진행 중. 실제 seed 1·2 GPU 학습·gate 및 frozen test는 미실행이다.**
P3 seed 0의 감사된 gate 통과와 64M 예산 동결이 선행 증빙이다.

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
