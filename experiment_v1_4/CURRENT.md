# v1.4 현재 안내

2026-09-30. **P1–P8 완료. 다음은 P9 sparse seed 1 반복 16 runs와 전체 평가다.**

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

## 현재 증빙과 다음 단계

P6의 24 SAE runs/120,000 updates/61.44M draws를 완료하고 전체 checkpoint 반환 감사를 통과했다.
[완료 판정](results/p6_final_audit_20260925/completion.json), [감사 보고](results/p6_final_audit_20260925/REPORT.md),
[선택 checkpoint manifest](results/p6_final_audit_20260925/selected_sae_manifest.json)를 따른다.
36개 scalar 통계·GPU smoke·504개 checkpoint·480개 validation MSE/선택 검증이 확인됐다.
원본 cache는 Drive `boolean_interp_v1_4/P5_r2`, 전체 학습 결과는 `P6_r1`과 로컬 원본 ZIP에 보존한다.
다음 P7에서 네 층의 좌표/random·128후보 대조를 준비하고 의미·fidelity·인과 평가를 수행한다.
P6 학습 완료를 전체 해석 실험 완료로 간주하지 않는다.

네 층의 필수 SAE·TC·인과 평가·초기화 반복이 모두 완료되어야 전체 실험을 완료로 판정한다.

## 변경 기록

2026-09-25: 필수 READ 분석을 block 0·3·7·11로 확정하고 관련 범위·예산·완료 조건을 정리했다. P5 층별 결과를 확인한 뒤, 12층 모델의 깊이에 따른 표현 차이를 평가하기 위한 변경이다. 이 확장을 P5 test 관측 전 사전등록으로 취급하지 않는다.
수정 전 원문: [보존본](../maintenance/read_layers_20260925/originals/experiment_v1_4/CURRENT.md).
동결 DESIGN/README의 기존 manifest hash는 [원본 경로·hash 목록](../maintenance/read_layers_20260925/originals.json)의 snapshot에서 검증한다.

## 2026-09-26 P7 실행 준비

P7 평가 계약·노트북·입력 번들의 로컬 검증을 마쳤다. GPU 본평가 0/24와 반환 감사는 대기다.
[상태 및 실행 파일](P7_STATUS.md)을 따른다. P8은 아직 시작하지 않는다.

## 2026-09-27 P7 GPU 결과 반환

P7 24/24 평가 결과를 반환받았다. 원본 P5 cache를 사용하는 재현 감사가 남아 P7 완료 판정은 보류한다.
[P7 상태와 추가 감사 노트북](P7_STATUS.md)을 따른다. P8은 아직 시작하지 않는다.

## 2026-09-28 P7 완료

[최종 감사 및 결과](results/p7_final_audit_20260928_01/REPORT.md), [완료 판정](results/p7_final_audit_20260928_01/completion.json)을 확인했다.
24개 SAE의 모든 필수 평가와 원본 재현 감사를 마쳤다. 다음은 P8 TC 24 runs 및 동일 평가이며 P9 반복 16 runs도 남아 있다. 위 준비·대기 절은 당시 기록이다.

## 2026-09-29 P8 첫 학습 전달물 준비

P8 TC 24 runs의 학습 계약·평가 규칙과 Colab 노트북·입력 번들을 준비하고 로컬 검증을 마쳤다.
실제 CUDA 학습 0/24, 평가 0/24로 P8은 미완료다. [P8 상태와 실행 파일](P8_STATUS.md)을 따른다.
학습 반환 감사 후 선택 checkpoint를 연결한 의미·대체·인과 평가를 이어간다.

## 2026-09-29 P8 학습 감사 완료·TC 평가 준비

TC 24/24 학습과 504 checkpoint·480 validation 검산 반환 감사를 통과했다.
의미·fidelity·대체·인과 평가 노트북·입력 번들의 로컬 검증을 마쳤다. 실제 평가 0/24이며 P8 전체는 미완료다.
[P8 상태와 다음 실행 파일](P8_STATUS.md), [학습 감사](results/p8_training_audit_20260929_01/REPORT.md)를 따른다.

## 2026-09-29 P8 평가 반환

24 TC 평가 결과의 전체 파일 hash·선택·원시 지표·대조군·집계 로컬 감사가 통과했다.
P5 원본 cache 재현 감사는 아직 대기로 P8 미완료다. [P8 상태와 추가 감사 파일](P8_STATUS.md)을 따른다.

## 2026-09-30 P8 완료

[최종 보고서](results/p8_final_audit_20260930_01/REPORT.md), [완료 판정](results/p8_final_audit_20260930_01/completion.json)을 확인했다.
24 TC의 전체 필수 평가와 원본 재현 반환 감사가 통과했다. 840개 고유 refit, 1,920 probe 작업, 3,360 의미 보고와 fidelity/CI 재현 기록을 검증했다.
필수 dictionary 64개 중 48개 학습·평가 완료, P9 sparse seed 1 반복 16개는 미실행이다. 전체 실험은 미완료다. 앞선 대기·실행 안내는 당시 이력이다.
