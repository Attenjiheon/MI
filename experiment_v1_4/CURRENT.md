# v1.4 현재 안내

2026-09-22. **본 실험 v1.4, P1–P4 완료. P5 진입 가능·미실행.**

현재 규격은 루트 [연구 설계](../01_experiment_design.md),
[언어·코퍼스](../02_language_and_corpus.md), [상세 명세](../03_experiment_spec.md)에 통합했다.
[실행 계획](../phase.md)은 단계 순서와 현재 증빙을 관리한다.

| 역할 | 원문·증빙 |
|---|---|
| 동결 당시 설계 | [DESIGN](DESIGN.md), [design_config](design_config.json), [design_manifest](design_manifest.json) |
| 실제 동결 입력 | [config set](configs/config_set_manifest.json), [corpus 수정 계약](corpus_rebuild.json) |
| P1 CPU 감사 | [P1_STATUS](P1_STATUS.md), [완료 증빙](results/audit_20260921_01/completion.json) |
| P2 CPU/GPU smoke | [P2_STATUS](P2_STATUS.md), [GPU 검증](evidence/gpu_smoke_20260921_01/verification.json) |
| P3 seed 0 | [P3_STATUS](P3_STATUS.md), [반환 감사](results/p3_audit_20260921_01/REPORT.md) |
| P4 seed 1·2 및 frozen test | [P4_STATUS](P4_STATUS.md), [최종 감사](results/frozen_test_audit_20260922_01/REPORT.md) |
| P5 입력 | [동결 LM 3개](results/frozen_test_audit_20260922_01/frozen_lms.json), [validation 동결](results/p4_audit_20260921_01/validation_freeze.json) |
| 실행 목록 | [run registry](results/run_registry.csv) |

[README](README.md)·DESIGN·design_config의 readiness는 설계 당시 불변 기록이다.
완료 보고서의 준비 단계 설명도 당시 이력이며 현재 상태를 덮어쓰지 않는다.
과거 규격 source hash는 [통합 전 원문](../archive/specifications/pre_v1_4_integration/03_experiment_spec.md)에서 확인한다.

다음은 seed 0·1·2의 block 0 READ `h/u/m` cache·probe 및 전체 층 full probe 진단이다.
P5 전용 config·노트북·반환 검증을 준비하고 실제 결과를 확인한 뒤 완료 처리한다.
기존 행동 gate/test를 다시 채점하거나 checkpoint를 재선택하지 않는다.
