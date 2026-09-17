# v1.2 P3 — GPU 실행 대기

사용자의 2026-09-17 P3 요청에 따라 설계 전용 v1.2의 생성기·실행기·Colab 전달물을 구현했다.
P1은 통과했다. CPU smoke/테스트와 Colab 전달물 검증 결과는 `P2_STATUS.md`에 기록한다. 새 GPU smoke와 seed 0 실제 실행 증빙이 아직 없으므로 P2 전체 및 P3은 완료가 아니다.

## 실행 계약

`interp_v1_2`만 실행한다. 4 blocks / 797,184 parameters, fresh seed 0, 기존 AdamW·warmup·FP32·effective batch 64를 유지한다. 16M 고정 예산, 실제 2,160 updates / 16,001,083 tokens로 종료한다. 기존 1M 조건부 3M CLI를 재사용하지 않는다.

모든 100k validation과 마지막 update를 중복 없이 평가하고, 전체 일반 validation 답 CE 최소(동률 이른 update) checkpoint를 선택한다. 해당 checkpoint의 일반 정확도 ≥99%, 세 진단 각각 ≥95%만 행동 gate로 사용한다. Test로 선택하지 않는다.

매 validation의 immutable checkpoint, init, optimizer/RNG/cursor/다음 평가 경계를 보존한다. `LATEST.json`이 last 참조, checkpoint의 `best`와 최종 `result.json`의 `selected`가 best 참조다. 1M/3M/8M/16M milestone에서 current/best와 각 파일 SHA-256을 별도 기록한다. 새 파일만 Drive에 복사하고 checksum 검증 후 완료 index를 갱신한다.

## Colab 전달

- `notebooks/01_colab_lm_16m.ipynb`
- `bundles/p3_16m_bundle_v2.zip` 및 `.sha256`
- `results/colab_delivery.json`: 전체 파일 수·ZIP/노트북 checksum과 빌드 검증.

새 Colab GPU 런타임에서 노트북을 업로드하고 위부터 실행한다. 입력 ZIP은 파일명 대신 SHA-256으로 판별한다. Drive 경로는 `MyDrive/boolean_interp_p3_16m_v1_2`다. 중단하면 새 런타임에서 `RESUME=True`로 마지막 완료 index를 복구한다. 오류/중단은 성공으로 집계하지 않는다.

마지막 결과 ZIP을 반환하면 `scripts/verify_p3_16m_evidence.py`로 모든 checkpoint·경계·선택·gate·milestone를 검증하고, GPU smoke/환경을 감사하며 선택 checkpoint를 CPU에서 재평가한다. 그 전까지 gate, 선택 checkpoint 및 P4 진입 여부는 미정이다. 실패하면 16M에서 중단 보고하고 자동 예산 연장이나 표현 분석을 하지 않는다.

## 비교 해석

기존 3M 입력 prefix·update 407을 보존했고 v1.1/v1.2 seed 0 초기화의 동일성을 테스트한다. 환경/microbatch가 달라질 수 있으므로 과거 GPU 결과와 bitwise 동일성을 주장하지 않는다. 30회 대 약160회 선택 후보 수 차이를 고려하도록 current 및 best-so-far 곡선과 milestone 지표를 모두 저장한다. 같은 seed의 예산 비교를 독립 seed 재현으로 세지 않는다.
