# v1.4 — Boolean 프로그램의 READ 표현 분석 설계

상태: 2026-09-25. **P1–P5 완료, block 0·3·7·11의 P6–P9 필수 해석 미실행.**

## 1. 연구 질문과 범위

Boolean 프로그램을 수행하는 동결 Transformer에서 READ 상태 정보의 선형 접근성,
SAE feature의 분리도, TC의 MLP 계산 근사 및 feature 개입 효과를 비교한다.
필수 sparse 분석 층은 **block 0·3·7·11**이다. 모든 통과 LM에 같은 네 층을 적용해
모델 깊이에 따른 표현 차이를 평가하며, 전 12층 full probe 진단을 함께 보고한다.

## 2. 모델·데이터 계약

| 항목 | 고정값 |
|---|---|
| 모델 | `deepwide12_read4`, 12 blocks, residual 256, 4 heads × 64, MLP 1024 |
| 파라미터 | 9,485,312 |
| LM 목적함수 | token-only read4, READ 답 weight 4, 기타 non-PAD weight 1 |
| LM seeds | 0·1·2, 세 seed 모두 validation gate 통과 |
| 학습량 | seed당 64,005,751 prediction tokens / 8,399 updates |
| 선택 checkpoint update | seed 0·1: 7,983; seed 2: 8,399 |
| 활성 corpus | `data/language_v1_4/rebuild_01/` |
| READ train/val/test | 동일 예약 위치 50k / 10k / 20k, 모든 층에 공유 |

LM 구조·학습·선택·gate는 [LM 계약](design_config.json)과 [동결 config](configs/config_set_manifest.json),
데이터는 [corpus 계약](corpus_rebuild.json)을 따른다. 실제 모델 hash는
[frozen_lms](results/frozen_test_audit_20260922_01/frozen_lms.json)에 있다.
LM forward 입력은 token IDs와 padding mask이며 상태·정답 metadata는 해석에만 사용한다.

## 3. 필수 READ 분석

[READ 분석 계약](analysis_plan.json)과 [상세 명세](../03_experiment_spec.md)가 실행 기준이다.

| 단계 | 범위 | 실행 수 |
|---|---|---:|
| P6 SAE | 3 LM × block 0·3·7·11 × k=4/16, sparse seed 0 | 24 |
| P7 SAE 평가 | 네 층별 의미·fidelity·대체·인과 대조 | P6의 모든 모델 |
| P8 TC | 같은 네 층의 u_l→m_l, 같은 k와 평가 | 24 |
| P9 초기화 반복 | LM seed 0, 네 층 × SAE/TC × k=4/16, sparse seed 1 | 16 |

Dictionary는 층별 독립 256→512→256 구조이며 run마다 batch 512, 5,000 updates를 수행한다.
매 250 updates 평가·저장하고 validation MSE 최소 checkpoint를 선택한다.
필수 학습 합계는 64 runs, 320,000 updates, 163.84M position draws다.
Label loss, early stopping, dead-latent 재초기화는 사용하지 않는다.

## 4. 비교와 개입

같은 LM·층·READ 위치에서 full probe, 좌표, random 방향, SAE/TC latent를 비교한다.
단일·최대 4 feature와 128후보 일치 대조를 네 층 모두 수행한다.
전처리·ranking·선택은 train/validation만 사용하고 최종 test로 재선택하지 않는다.
Dictionary scalar 통계는 각 LM·layer·hook의 train에서 계산하고 validation/test에 고정 적용한다.

근사 대체와 반사실적 패칭은 해당 한 층의 READ 한 위치에 적용한다.
다른 층은 원래 연산을 수행하며 downstream을 다시 계산한다. Block 11 residual은 final LayerNorm 이전을 사용한다.
같은 READ target·causal origin을 네 층에 공유하고 층별 결과와 paired 차이를 보고한다.
Feature ID는 dictionary 간 같은 의미를 보장하지 않으며 층을 독립 LM 반복으로 세지 않는다.

## 5. 실행 준비와 완료 조건

P5는 전 12층 cache·full probe와 block 0의 scalar 통계·좌표/random 대조군을 검증했다.
P6 학습 전에 네 층 h/u/m scalar 통계 총 36개를 검증하고 config·hash·smoke·재개 절차를 고정한다.
네 층의 좌표/random·128후보 대조군은 해당 SAE/TC 평가 전에 확보한다.
실행 상태와 누락 증빙은 [P6 상태](P6_STATUS.md), 전체 순서는 [phase](../phase.md)를 따른다.

64개 필수 dictionary run과 각 run의 평가·개입·재현 증빙이 모두 있어야 필수 해석을 완료한다.
오류·자원 중단·support 부족·미매칭은 사유와 함께 보고한다.
Update, 필수 네 층 이외의 층, 길이 평가, m→m SAE는 P9 이후 선택 분석이다.

## 변경 기록

2026-09-25: 필수 READ 분석을 block 0·3·7·11로 확정하고 관련 범위·예산·완료 조건을 정리했다. P5 층별 결과를 확인한 뒤, 12층 모델의 깊이에 따른 표현 차이를 평가하기 위한 변경이다. 이 확장을 P5 test 관측 전 사전등록으로 취급하지 않는다.
수정 전 원문: [보존본](../maintenance/read_layers_20260925/originals/experiment_v1_4/DESIGN.md).
동결 DESIGN/README의 기존 manifest hash는 [원본 경로·hash 목록](../maintenance/read_layers_20260925/originals.json)의 snapshot에서 검증한다.
