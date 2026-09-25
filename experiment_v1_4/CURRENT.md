# v1.4 현재 안내

2026-09-25. **P1–P5 완료. 다음은 block 0·3·7·11 READ SAE의 P6 준비·학습이다. P6–P9는 미실행이다.**

현재 규격은 [연구 설계](../01_experiment_design.md), [언어·코퍼스](../02_language_and_corpus.md),
[상세 명세](../03_experiment_spec.md), [실행 계획](../phase.md)을 따른다.
필수 해석 범위와 예산은 [READ 분석 계약](analysis_plan.json)에 고정한다.

| 역할 | 문서·증빙 |
|---|---|
| 현재 설계·안내 | [DESIGN](DESIGN.md), [README](README.md), [P6 상태](P6_STATUS.md) |
| LM 동결 입력 | [design_config](design_config.json), [config set](configs/config_set_manifest.json), [corpus 계약](corpus_rebuild.json) |
| P1 CPU 감사 | [P1 상태](P1_STATUS.md), [완료 증빙](results/audit_20260921_01/completion.json) |
| P2 CPU/GPU smoke | [P2 상태](P2_STATUS.md), [GPU 검증](evidence/gpu_smoke_20260921_01/verification.json) |
| P3 seed 0 | [P3 상태](P3_STATUS.md), [반환 감사](results/p3_audit_20260921_01/REPORT.md) |
| P4 frozen test | [P4 상태](P4_STATUS.md), [최종 감사](results/frozen_test_audit_20260922_01/REPORT.md) |
| P5 전 층 cache·full probe | [P5 상태](P5_STATUS.md), [최종 감사](results/p5_final_audit_20260925_01/REPORT.md), [완료 증빙](results/p5_final_audit_20260925_01/completion.json) |
| 해석 입력 | [동결 LM 3개](results/frozen_test_audit_20260922_01/frozen_lms.json), [전 층 추출 계약](p5_r2/contract.json), [P5 cache receipt·block 0 전처리](results/p5_final_audit_20260925_01/p6_cache_manifest.json) |
| 실행 목록 | [run registry](results/run_registry.csv) |

## 필수 READ 분석

- LM seed 0·1·2, block 0·3·7·11, k=4/16을 모두 분석한다.
- P6 SAE 24 runs → P7 의미·fidelity·인과 평가 → P8 TC 24 runs 및 같은 평가 → P9 초기화 반복 16 runs 순서다.
- 총 64 dictionary runs × 5,000 updates = 320,000 updates, 163.84M position draws다.
- 네 층의 같은 READ 위치·라벨·causal origin을 사용하고 층별 전처리·dictionary·feature 선택을 별도 저장한다.
- 한 번에 한 층·한 READ 위치를 개입하고 층별 결과와 층간 차이를 보고한다.

## 현재 증빙과 P6 준비

P5는 3,510 probe·240 cache 파일·9개 dictionary scalar 통계와 전 12층 full probe 진단의 독립 검산을 완료했다.
원본 cache는 Drive `boolean_interp_v1_4/P5_r2`에 보존한다.
P5의 `p6_cache_manifest.json`은 전 층을 담은 240개 cache 파일의 receipt와 block 0 전처리 9개를 함께 기록한다. 최상위 `layer: 0`을 네 층 전처리 완료로 해석하지 않는다.
P6에서는 block 0·3·7·11의 h/u/m train scalar 통계 총 36개를 확보·검증한다.
검증된 block 0 통계 9개는 재사용하고 block 3·7·11 통계 27개는 train cache에서 계산한다.
좌표/random·128후보 대조군도 네 층 각각에 필요하며 해당 SAE/TC 평가 전에 준비·검증한다.
P6 실행 config·source hash·Colab 노트북·입력 번들·GPU smoke·실행 증빙은 아직 준비·검증이 필요하다.

네 층의 필수 SAE·TC·인과 평가·초기화 반복이 모두 완료되어야 전체 실험을 완료로 판정한다.

## 변경 기록

2026-09-25: 필수 READ 분석을 block 0·3·7·11로 확정하고 관련 범위·예산·완료 조건을 정리했다. P5 층별 결과를 확인한 뒤, 12층 모델의 깊이에 따른 표현 차이를 평가하기 위한 변경이다. 이 확장을 P5 test 관측 전 사전등록으로 취급하지 않는다.
수정 전 원문: [보존본](../maintenance/read_layers_20260925/originals/experiment_v1_4/CURRENT.md).
동결 DESIGN/README의 기존 manifest hash는 [원본 경로·hash 목록](../maintenance/read_layers_20260925/originals.json)의 snapshot에서 검증한다.
