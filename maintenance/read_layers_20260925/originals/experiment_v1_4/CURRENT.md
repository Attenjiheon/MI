# v1.4 현재 안내

2026-09-25. **본 실험 v1.4, P1–P5 완료. 다음은 P6 READ SAE(미실행).**

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

다음은 세 동결 LM의 block 0 READ SAE, k=4/16이다. P6는 아직 실행하지 않았다.
P5의 검증된 위치·scalar 전처리와 [cache manifest](results/p5_final_audit_20260925_01/p6_cache_manifest.json)를 재사용한다.
기존 행동 gate/test와 probe 선택 규칙을 다시 열지 않는다.

## P5 완료 (2026-09-25)

[P5 상태](P5_STATUS.md) · [최종 감사](results/p5_final_audit_20260925_01/REPORT.md) · [완료 증빙](results/p5_final_audit_20260925_01/completion.json).
3,510 task·240 cache·9개 scalar 통계 검증, 선택 모델 3,750개와 subset 포함 5,250개 평가의 독립 수치 재현 완료.
SAE·TC·인과 평가·sparse seed 반복은 남아 있어 전체 실험 완료는 아니다.
