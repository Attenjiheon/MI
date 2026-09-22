# v1.2 P3 — failed (16M 완주, 행동 gate 미달)

2026-09-17 반환 증빙 감사 완료. P1 및 실제 P2 CPU/GPU smoke는 통과했다. Seed 0은 **16,001,083 prediction tokens / 2,160 updates / 138,240 sequences**를 소비했다. 명목16M overshoot는 1,083이다.

160회 validation 중 일반 READ 답 CE 최소 checkpoint는 마지막 update 2,160이다. 일반 정확도 **80.72%**(기준99%), 세 진단 **81.25% / 76.37% / 86.13%**(각95%)로 모두 미달했다. 선택 checkpoint SHA-256은 `3254c12495f996da79f47978fefb01456701b316a0e765704de3739944fa1317`이다.

- [x] 실제 Colab CPU/GPU smoke·환경·입력 및 실행 code hashes 대조.
- [x] 2,160 updates의 토큰·cursor·LR·validation 경계 검증.
- [x] Init 및 160개 validation checkpoint, optimizer/RNG, current/best milestone와 완료 index 보존·검증.
- [x] 일반 CE 최소 선택 및 해당 checkpoint에서만 gate 판정.
- [x] CPU 재평가, 전체 corpus 독립 replay 재검산, 구현 tests 8 passed.
- [x] v1.1 초기 모델 및 3M 모델과 bitwise 동일성 확인.
- [x] 실패·실제 자원·학습 곡선·strata·baseline 및 실행 번들 차이 기록.
- [ ] 행동 gate 통과. **실패이므로 체크하지 않는다.**

체크 항목은 감사 완료를 뜻하며 행동 성공이 아니다. **P4 및 SAE/TC·인과 분석으로 진행하지 않는다.** 고정16M 정책에 따라 자동 예산 연장 없이 이 버전을 종료한다. Test 점수는 보지 않았다.

반환 실행은 `p3_16m_bundle_v1.zip` 기반이다. v2와 데이터/실행 코드는 같지만 packaging manifest/config hash가 다르다. 실행 hash `8503418e…` / corpus `5decfd2a…`를 그대로 보존하고 v2의 `84f36f4c…` / `87ec2e77…`로 치환하지 않았다.

상세 결과·비교·한계: [중단 보고](results/p3_stop_report.md). 증빙: [실행 감사](results/p3_run_audit.json), [보충 감사](results/p3_supplemental_audit.json), [학습 곡선](results/figures/p3_learning_curve.png), `evidence/p3_16m_verified_contents_20260917.zip`.
