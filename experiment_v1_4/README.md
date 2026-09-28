# experiment_v1_4

**P1–P7 완료. 다음 단계는 block 0·3·7·11 READ Transcoder(P8)다.**

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
| P6 SAE | 완료 | 24 runs·120k updates·[전체 반환 감사](results/p6_final_audit_20260925/REPORT.md) |
| P7 SAE 평가·개입 | 완료 | [24개 평가와 원본 재현 반환 감사](results/p7_final_audit_20260928_01/REPORT.md) |
| P8 TC 학습·평가·개입 | 미실행 | 24 runs 및 네 층의 동일 평가 |
| P9 초기화 반복 | 미실행 | LM seed 0, 네 층 × 2도구 × 2 k = 16 runs |
| P10 선택 분석 | 미실행 | Update, 필수 집합 이외의 층, 길이 평가, m→m SAE |
| P11 최종 집계 | 미완료 | 네 층·LM seed·sparse seed별 결과 및 실패·미완료 기록 |

## 다음 단계

P6의 [선택 checkpoint 24개](results/p6_final_audit_20260925/selected_sae_manifest.json)와
검증된 P5 cache/36개 scalar 통계를 사용한다. P7에서 층별 좌표/random·128후보 기준선,
사후 감독 의미 평가, fidelity·근사 대체 및 모든 필수 인과 대조를 수행한다.
P8 TC 24 runs와 P9 초기화 반복 16 runs가 남아 있으며 전체 실험은 미완료다.

## 변경 기록

2026-09-25: 필수 READ 분석을 block 0·3·7·11로 확정하고 관련 범위·예산·완료 조건을 정리했다. P5 층별 결과를 확인한 뒤, 12층 모델의 깊이에 따른 표현 차이를 평가하기 위한 변경이다. 이 확장을 P5 test 관측 전 사전등록으로 취급하지 않는다.
수정 전 원문: [보존본](../maintenance/read_layers_20260925/originals/experiment_v1_4/README.md).
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
