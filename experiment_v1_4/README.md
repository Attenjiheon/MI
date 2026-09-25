# experiment_v1_4

**P1–P5 완료. 다음 단계는 block 0·3·7·11 READ SAE 학습(P6)이다.**

12-block × 256-width Transformer 세 seed를 동결해 READ 표현을 분석한다.
활성 corpus는 `data/language_v1_4/rebuild_01/`이며 세 LM 모두 validation gate를 통과했다.

- [현재 안내](CURRENT.md) · [현재 설계](DESIGN.md) · [READ 분석 계약](analysis_plan.json)
- [실행 계획](../phase.md) · [상세 명세](../03_experiment_spec.md) · [P6 상태](P6_STATUS.md)
- [LM 설계 계약](design_config.json) · [동결 config](configs/config_set_manifest.json)

## 단계 상태

| 단계 | 상태 | 범위·증빙 |
|---|---|---|
| P1 CPU corpus | 완료 | [P1 상태](P1_STATUS.md) |
| P2 구현·smoke | 완료 | [P2 상태](P2_STATUS.md) |
| P3 seed 0 LM | 완료 | [P3 상태](P3_STATUS.md) |
| P4 seed 1·2 및 frozen test | 완료 | [P4 상태](P4_STATUS.md), [동결 모델](results/frozen_test_audit_20260922_01/frozen_lms.json) |
| P5 cache·probe | 완료 | 전 12층 cache·full probe 및 [독립 감사](results/p5_final_audit_20260925_01/REPORT.md) |
| P6 SAE | 미실행 | 3 LM × 4층 × 2 k = 24 runs |
| P7 SAE 평가·개입 | 미실행 | 네 층 모두 의미·후보 수 대조·fidelity·인과 평가 |
| P8 TC 학습·평가·개입 | 미실행 | 24 runs 및 네 층의 동일 평가 |
| P9 초기화 반복 | 미실행 | LM seed 0, 네 층 × 2도구 × 2 k = 16 runs |
| P10 선택 분석 | 미실행 | Update, 필수 집합 이외의 층, 길이 평가, m→m SAE |
| P11 최종 집계 | 미완료 | 네 층·LM seed·sparse seed별 결과 및 실패·미완료 기록 |

## 실행 준비

P5의 동일 READ 위치·cache를 사용해 네 층의 scalar 통계와 입력 hash를 검증한다.
P6 실행 config와 네 층의 debug/GPU smoke·재개 검증을 준비하고 Colab `.ipynb` 및 입력 번들로 전달한다.
SAE·TC는 run마다 5,000 updates를 완료한 뒤 validation MSE로 checkpoint를 선택한다.
총 필수 dictionary 예산은 64 runs, 320,000 updates, 163.84M position draws다.
각 평가 전에 층별 좌표/random·128후보 대조군을 준비한다. 실제 Colab 반환 증빙을 검증한 뒤에만 단계를 완료 처리한다.

## 변경 기록

2026-09-25: 필수 READ 분석을 block 0·3·7·11로 확정하고 관련 범위·예산·완료 조건을 정리했다. P5 층별 결과를 확인한 뒤, 12층 모델의 깊이에 따른 표현 차이를 평가하기 위한 변경이다. 이 확장을 P5 test 관측 전 사전등록으로 취급하지 않는다.
수정 전 원문: [보존본](../maintenance/read_layers_20260925/originals/experiment_v1_4/README.md).
동결 DESIGN/README의 기존 manifest hash는 [원본 경로·hash 목록](../maintenance/read_layers_20260925/originals.json)의 snapshot에서 검증한다.
