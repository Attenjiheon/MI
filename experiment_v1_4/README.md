# experiment_v1_4

**P1–P10 완료. P11 최종 집계가 남았다.**

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
| P8 TC 학습·평가·개입 | 완료 | [24 runs 최종 감사](results/p8_final_audit_20260930_01/completion.json) |
| P9 초기화 반복 | 완료: 16/16 학습·평가·원본 재현 감사 통과 | [P9 실행 파일과 상태](P9_STATUS.md) |
| P10 선택 분석 | 완료: Update 12/12 학습·평가·원본 재현 감사 통과 | [범위 결정·T4 실행 파일](P10_STATUS.md) |
| P11 최종 집계 | 미완료 | 네 층·LM seed·sparse seed별 결과 및 실패·미완료 기록 |

## 다음 단계

P6의 [선택 checkpoint 24개](results/p6_final_audit_20260925/selected_sae_manifest.json)와
검증된 P5 cache/36개 scalar 통계를 사용한다. P7에서 층별 좌표/random·128후보 기준선,
사후 감독 의미 평가, fidelity·근사 대체 및 모든 필수 인과 대조를 수행한다.
P9 초기화 반복까지 필수 64개 dictionary 분석을 완료했다. P11 최종 집계가 남아 전체 실험은 미완료다.

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

## 2026-10-01 P9 평가 전달물

16개 초기화 반복의 학습 반환 감사를 통과했다. [P9 평가 노트북·입력 ZIP](P9_STATUS.md)을 준비하고 로컬 검증을 마쳤다. 실제 전체 평가 0/16으로 P9와 전체 실험은 미완료다.


## 2026-10-01 P9 평가 반환·원본 재현 감사 준비

P9 SAE 8개·TC 8개 전체 평가 반환의 로컬 감사가 통과했다. 38,080 파일 hash, causal pairs 32,768개, READ 대체 targets 32,768개, probe tasks 1,200개와 초기화 비교 44,880행을 확인했다. 원본 cache refit·예측/fidelity/CI 재현 및 고정 GPU replay 감사가 남아 P9와 전체 실험은 미완료이며 P10은 대기다.

[P9 상태·감사 노트북·입력 ZIP](P9_STATUS.md)을 따른다.


## 2026-10-02 P9 완료

16개 sparse seed 1 SAE/TC의 학습·전체 평가·원본 cache 재현 반환 감사를 통과했다. 필수 dictionary 64개 모두 학습·평가·감사 완료다. P10 선택 분석 결정과 P11 최종 집계는 남아 전체 실험은 미완료다. 앞선 P9 대기 안내는 당시 이력이다.

[최종 감사](results/p9_final_audit_20261002_01/REPORT.md), [완료 판정](results/p9_final_audit_20261002_01/completion.json)을 따른다.
