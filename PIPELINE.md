# 실험 파이프라인

## 1. 정확한 상태를 아는 데이터

변수 A/B/C/D를 SET·NOT·AND·OR·XOR로 갱신하고 READ로 조회하는 프로그램을 생성한다. READ 답은 실행 결과인 B0/B1이다. LM은 token ID와 padding mask만 입력받는다. 현재·이전 상태, 연산 관계 등의 metadata는 평가에만 사용한다.

CPU에서 shard를 미리 만들고 독립 parser/replay로 정답을 검산한다. LM seed 3개가 동일한 shard를 동일한 순서로 한 번씩 소비한다. Sequence와 READ prefix의 split 간 중복을 검사하며, 같은 origin의 파생 pair는 같은 split에 둔다.

분석 위치는 `READ A B1`에서 **A를 처리한 직후**다. 이 위치의 logits가 다음 토큰 B1을 예측하므로 정답 토큰을 본 activation을 사용하지 않는다. Interpretation은 sequence 단위로 분리한 뒤 READ 위치를 train 50,000 / validation 10,000 / test 20,000개 추출한다.

**코드:** [언어 실행](code/corpus/language.py), [독립 replay](code/corpus/replay.py), [pair 생성](code/corpus/v1_4.py), [코퍼스 구성](code/scripts/generate_v1_4_corpus.py).

## 2. 행동을 검증한 뒤 LM 동결

12-block, residual width 256, MLP width 1,024, 4-head Transformer를 학습한다. Vocabulary 15, context 768, RoPE, dropout 0, untied embedding/unembedding이며 총 9,485,312 parameters다.

각 seed의 예산은 약 64M prediction tokens다. Effective batch 64, AdamW learning rate 3e-4, FP32를 사용한다. READ 답 token의 CE weight는 4, 나머지 non-padding token은 1이며 가중치 합으로 정규화한다. 답 위치는 token pattern으로 판별한다.

10개 milestone 중 select split의 first-member 42-cell macro answer CE로 checkpoint를 고른다. 전역 최소와 1e-4 nats 이내이면 general answer CE, 이른 update 순으로 선택한다. 별도 gate split은 재선택에 쓰지 않는다. 일반 READ 정확도 99% 이상, 지정 진단·first/repeat 및 연산·깊이·답 그룹 기준 95% 이상과 전체 coverage를 요구한다. Seed 0 통과 후 seed 1·2를 실행했으며 세 seed 모두 통과했다.

모든 학습·선택 결정을 동결한 뒤 test를 평가했다. 이후 SAE/TC 성능을 이유로 LM을 바꾸지 않았다.

**코드:** [모델과 hook](code/interp_v1_4/model.py), [loss·optimizer](code/interp_v1_4/training.py), [선택·학습](code/interp_v1_4/cli.py), [행동 gate](code/interp_v1_4/behavior.py), [frozen test](code/interp_v1_4/frozen_test.py).

## 3. 같은 위치에서 표현을 비교

동결 LM에서 READ 위치의 세 표현을 수집한다.

| 표현 | 정의 | 쓰임 |
|---|---|---|
| `h_l` | block의 residual 출력, 최종 LayerNorm 이전 | SAE 입력·복원 및 residual 패칭 |
| `u_l` | 해당 block MLP의 정규화된 입력 | TC 입력 |
| `m_l` | 해당 block MLP 출력 | TC target 및 MLP-output 패칭 |

전 12층에서 full linear probe를 진단하고, block 0·3·7·11에서 sparse 분석을 수행한다. 층마다 dictionary와 전처리를 독립 관리하되 동일 sequence·READ 위치를 사용한다.

**코드:** [추출·join](code/interp_v1_4/p5.py), [probe fitting·선택](code/interp_v1_4/p5_probe.py).

## 4. SAE와 TC 학습

SAE는 `h→h`, TC는 `u→m`을 학습한다. Train activation의 평균 벡터와 scalar scale을 고정하고, ReLU 후 TopK를 적용하는 width 512 dictionary를 쓴다. k=4와 16을 모두 평가한다. 상태 라벨은 두 학습 loss에 들어가지 않으며, TC target은 원래 MLP 출력이다.

각 run은 512 positions/batch로 5,000 updates를 수행한다. 250 updates마다 validation MSE를 계산하고 최소값의 checkpoint를 선택한다. 동률은 이른 update다. Early stopping이나 dead feature 재초기화는 사용하지 않는다.

- 기본 실험: 3 LM × 4 layers × 2 tools × 2 k = **48 runs**.
- 초기화 반복: LM seed 0에서 sparse seed 1로 4 × 2 × 2 = **16 runs**.
- 필수 READ 전체: **64 runs**, 320,000 updates, 163.84M position draws.

**코드:** [dictionary 연산](code/interp_v1_4/dictionary.py), [SAE 학습](code/interp_v1_4/p6.py), [TC 학습](code/interp_v1_4/p8.py), [초기화 반복](code/interp_v1_4/p9.py).

## 5. 의미·fidelity·행동 대체

주 라벨은 READ 현재 값이다. Train에서 feature를 ranking하고, train fitting과 validation 선택을 거쳐 단일 또는 최대 4개 feature의 linear probe를 평가한다. Lambda·feature 수·binary threshold는 test 전에 고정한다.

Full probe, 원래 좌표 256개, random 방향 512개, SAE/TC latent 512개를 비교한다. 후보 수 효과를 분리하기 위해 각 표현에서 고정 RNG로 128개를 뽑는 대조도 수행한다. 현재≠이전 값 subset 및 query A/B/C→D 전이를 별도로 보고한다.

Fidelity는 NMSE, R², explained variance, L0, train dead 비율이다. SAE의 h 복원과 TC의 m 예측은 target이 다르므로 동일 과제의 점수처럼 우열을 판정하지 않는다. 각 sequence에서 고정된 READ 하나·층 하나만 근사 activation으로 바꾸어 CE와 정확도 변화를 paired 비교한다.

**코드:** [SAE 평가](code/interp_v1_4/p7.py), [TC 평가](code/interp_v1_4/p8_evaluation.py), [지표·matching·bootstrap](code/interp_v1_4/p7_metrics.py).

## 6. 반사실적 개입과 대조군

SET bit 하나만 바꾼 origin/donor 프로그램을 독립 replay한다. Target 이전 READ 답은 동일하게 유지하며, target 답이 달라지는 changed pair와 유지되는 unchanged pair를 모두 평가한다. 기억·합성 조건을 구분하고 양방향으로 개입한다.

선택 feature의 decoder 차이를 원래 activation에 더해 근사 오차를 보존한다. SAE는 h에, TC는 m에 적용하며 TC의 residual skip은 유지한다. Identity/null, full donor, 근사 대체, 좌표/random 및 matched random 등 동결된 대조를 함께 보고한다. Full donor를 보장된 상한으로 가정하지 않는다.

주 지표는 changed-pair logit margin 변화와 15-token argmax 답 전환율, unchanged-pair 오류 유발률이다. Matched random과 selected는 **같은 matching 성공 origin subset**에서 비교하고 미매칭 coverage를 보존한다.

**코드:** [SAE 패칭](code/interp_v1_4/p7_patching.py), [TC 패칭](code/interp_v1_4/p8_patching.py), [TC 인과 평가](code/interp_v1_4/p8_causal.py).

## 7. 반복·불확실성·집계

의미 지표는 sequence ID, 인과 지표는 origin ID를 재표집 단위로 삼는 1,000-draw percentile 95% cluster bootstrap을 사용한다. Pair 방향과 feature 후보를 독립 표본으로 세지 않는다. 층간 차이는 같은 재표집을 공유한 paired 비교다. 클래스 소실 등 무효 draw와 NA를 보존한다.

Sparse seed 반복은 LM seed 0에 한정하며 같은 latent ID를 같은 feature로 대응시키지 않는다. LM 3개 평균·범위와 sparse 초기화 차이는 구분한다. Update block 3의 12개 dictionary와 의미·전이 분석은 탐색 결과로 별도 보존했다. 추가 dictionary 층·length GPU 평가·m→m SAE·Update 인과 개입은 수행하지 않았다.

**코드:** [초기화 비교](code/interp_v1_4/p9_comparison.py), [Update](code/interp_v1_4/p10.py), [원본 최종 집계](code/interp_v1_4/p11.py). 실제 run·선택 hash는 [dictionary registry](results/dictionary_registry.csv.gz), 원래 완료 판정은 [completion.json](provenance/completion.json)에 있다.

**표기 정정:** 동결 P11 `semantic_metrics.csv`의 READ random 행에는 후보 수가 256으로 표기되어 있지만 실제 fit은 512개다. [정정 증빙](provenance/manuscript_evidence.json)의 64개 원본 fit 검산 기록을 따른다. 원본 CSV의 bytes·점수·선택을 보존했으며 이 안내의 후보 수에는 정정값을 사용했다.
