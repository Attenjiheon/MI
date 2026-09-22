# v1.3 — 상태 전이 shortcut 진단과 확인 실험 설계

작성: 2026-09-17. **설계만 완료된 상태이며 구현·데이터 생성·GPU 실행은 아직 하지 않았다.** 기계 판독 계약은 [design_config.json](./design_config.json)을 따른다. v1.0–v1.2의 데이터·checkpoint·결과를 수정하거나 성공으로 재분류하지 않는다.

## 1. 출발점과 연구 질문

v1.2 seed 0은 16,001,083 prediction tokens에서 일반 READ 80.72%, 세 기존 진단 81.25% / 76.37% / 86.13%로 행동 gate에 실패했다. 선택 checkpoint는 마지막 update이며 답 CE도 끝까지 감소했으므로 수렴·용량·최적화 중 하나로 원인을 단정할 수 없다.

보존 checkpoint를 test 없이 validation에서 추가 층화한 결과는 다음과 같다.

| 조건 | targets | 정확도 |
|---|---:|---:|
| 전체 READ | 8,171 | 80.72% |
| queried variable의 마지막 갱신 뒤 첫 READ | 6,175 | 76.94% |
| 같은 상태를 이미 READ한 뒤 재질문 | 1,996 | 92.43% |
| structural depth 0 | 2,585 | 93.46% |
| structural depth 1 | 1,524 | 78.94% |
| structural depth >=2 | 4,062 | 73.29% |
| structural depth >=2이면서 첫 READ | 3,124 | 67.38% |

첫 READ만 보면 SET 94.96%, NOT 68.93%, AND 73.24%, OR 77.16%, XOR 58.30%였다. 이는 인과적 원인 규명이 아니라 다음 두 가설을 분리할 근거다.

- **깊이·용량 가설:** 네 block의 계산 깊이 또는 용량이 합성된 상태 전이에 부족하다.
- **목적함수 가설:** 모든 next-token 위치를 같은 가중치로 학습하여 READ 답 신호가 부족하고, 이전 답 복사 같은 쉬운 규칙이 우선 학습된다.

v1.3의 질문은 “토큰을 더 주면 99%가 되는가”가 아니다.

> 동일 데이터와 평가에서 깊이·폭 배분과 READ-answer loss 가중치가 첫 논리 갱신 READ의 연속 CE와 정확도를 개선하는가? 탐색에서 고정한 한 설정이 새 validation과 독립 LM seed에서 행동 gate를 재현하는가?

## 2. 설계 원칙과 기존 규격의 변경

1. v1.2는 `failed`로 동결한다. checkpoint를 이어 학습해 v1.2 결과를 바꾸지 않는다.
2. v1.0–v1.2 validation은 이후 `legacy_dev`로만 취급한다. v1.3 선택·gate·중단에는 사용하지 않는다.
3. 기존 test 점수는 계속 보지 않는다. v1.3용 select validation, gate validation, test를 GPU 학습 전에 새 seed namespace로 예약한다.
4. 탐색은 사전 정의한 3 architecture × 2 loss의 여섯 cell, seed 0, 8M으로 제한한다. 임의 후보 추가나 실패 cell 교체를 금지한다.
5. 탐색 winner 하나만 32M 확인 단계로 승격한다. Seed 0이 새 gate를 통과한 뒤에만 seed 1·2를 실행한다.
6. 정확도 power-law 외삽으로 예산을 바꾸지 않는다. 8M과 32M은 고정 예산이며 32M 이후 자동 연장은 없다.
7. 모델 forward 입력은 token ID와 padding mask뿐이다. 상태·정답 metadata는 loss나 forward에 넣지 않는다.
8. `read4` loss는 정답 값이나 metadata가 아니라 입력 token pattern `READ <VAR>`만으로 답 예측 위치를 찾는다. 이는 01의 모든 위치 동일 CE를 명시적으로 바꾸는 v1.3 실험 조건이다. 결과를 순수 uniform-LM 결과와 구분한다.

## 3. 데이터와 split 동결

### 3.1 Train stream

- 새 root는 `data/language_v1_3`이다.
- v1.2의 첫 16,001,083 prediction tokens / 138,240 sequences / 35 shards를 byte-identical prefix로 보존한다.
- 총 명목 32M prediction tokens에 처음 도달·초과하는 완전한 64-sequence update까지 새 IID train shards를 추가한다.
- 반복·shuffle·정답 균형화는 하지 않는다. 모든 후보와 LM seed가 같은 순서를 한 번씩 소비한다.
- 새 예약 split 생성 시 v1.0–v1.2의 모든 sequence hash와 causal prefix hash를 먼저 적재한다. 새 split은 과거 train을 포함한 어떤 기존 split과도 exact sequence/prefix가 겹치지 않아야 한다.
- 새 validation/test/interpretation/causal split을 예약한 뒤 train extension을 생성한다. Extension은 모든 과거·신규 예약 hash를 거부한다.
- seed namespace는 `20260917|language-v1.3|<purpose>|<index>`의 SHA-256 앞 8 bytes, unsigned big-endian과 PCG64를 사용한다.

### 3.2 선택·gate·test 분리

세 평가 계층은 서로 exact sequence가 겹치지 않는다.

| 계층 | 사용 | 규모 |
|---|---|---:|
| `select/general` | checkpoint와 pilot winner 선택 | 512 sequences, 모든 READ |
| `select/first_repeat_pairs` | 핵심 연속 지표 선택 | 아래 42 cell × 16 pairs = 672 pairs |
| `gate/general` | 선택 후 seed별 행동 gate | 1,024 sequences, 모든 READ |
| `gate/legacy_diagnostics` | fresh other-variable / repeated-update / first-after-set | 조건별 1,024 targets |
| `gate/first_repeat_pairs` | 핵심 shortcut·합성 gate | 42 cell × 64 pairs = 2,688 pairs |
| `test/general` | 모든 설정·seed 동결 뒤 최종 보고 | 2,048 sequences, 모든 READ |
| `test/legacy_diagnostics` | 최종 기존 진단 | 조건별 2,048 targets |
| `test/first_repeat_pairs` | 최종 shortcut·합성 평가 | 42 cell × 128 pairs = 5,376 pairs |

`first_repeat_pairs`의 origin target은 다음을 모두 만족한다.

- queried variable의 마지막 갱신은 NOT/AND/OR/XOR이며 query는 그 dst다.
- `reads_of_query_var_since_last_update == 0`인 첫 READ다.
- structural depth bin은 `1`, `2–3`, `4+`다.
- NOT은 갱신 전 dst bit `0/1` × depth 3개 = 6 cell이다.
- AND/OR/XOR은 input truth `00/01/10/11` × depth 3개 = 36 cell이다.
- 총 42 cell을 같은 quota로 생성하며 target당 독립 origin sequence를 사용한다.

각 repeat member는 같은 origin의 의미와 target 답을 유지하면서, 마지막 갱신 뒤 같은 변수를 한 번 READ한 올바른 답 토큰을 target 전에 삽입한다. Pair는 같은 split에 묶는다. 첫 member와 repeat member의 정확도·CE 차이를 paired 보고하지만, gate의 핵심 성능은 첫 member다.

Gate validation은 checkpoint 선택에 쓰지 않는다. 선택 checkpoint 하나를 정한 뒤 seed당 한 번 평가한다. Test는 pilot, winner 선택, checkpoint 선택, threshold 수정에 사용하지 않는다.

### 3.3 CPU gate

기존 03 §5.3 검사를 모두 유지하고 다음을 추가한다.

- 과거 v1.0–v1.2 전체 hash와 신규 split/train extension의 교집합이 없다.
- 42개 cell 각각 정확한 quota, 독립 sequence 수, answer 0/1 support, operator/truth/depth metadata가 있다.
- first/repeat pair는 target 이전의 의도한 READ 삽입 외 실행 의미가 같고 target 답이 동일하다.
- first member는 실제로 이전 same-state answer가 없고 repeat member에는 정확히 하나 이상 있다.
- v1.2 train prefix의 token/metadata/file hash와 milestone cursor가 byte-identical하다.
- 새 train extension을 포함한 전체 32M stream의 replay, coverage, RNG state, 실제 final cursor/update/overshoot를 기록한다.

## 4. 제한된 3×2 탐색

### 4.1 Architecture 세 조건

공통으로 causal decoder-only, Pre-LN, GELU exact, RoPE, dropout 0, vocab 15, context 768, MLP width `4*d_model`, 4 attention heads를 사용한다.

| ID | blocks | d_model | head width | MLP | 예상 파라미터 | 목적 |
|---|---:|---:|---:|---:|---:|---|
| `base4` | 4 | 128 | 32 | 512 | 797,184 | v1.2 exact anchor |
| `wide4` | 4 | 184 | 46 | 736 | 1,640,544 | 같은 깊이의 용량 증가 |
| `deep8` | 8 | 128 | 32 | 512 | 1,590,272 | wide4와 비슷한 총량의 깊이 증가 |

`wide4`와 `deep8`의 파라미터 차이는 약 3.2%다. 이는 완전한 compute-match가 아니므로 width와 depth의 인과 효과를 단정하지 않고, parameter-matched에 가까운 depth–width tradeoff로 보고한다. Attention/MLP output weight multiplier는 각 모델에서 `1/sqrt(2L)`이다.

### 4.2 Loss 두 조건

| ID | 정의 |
|---|---|
| `uniform` | PAD를 제외한 모든 next-token CE에 weight 1 |
| `read4` | `READ <VAR>` 직후 B0/B1 target 위치 weight 4, 나머지 weight 1; weighted sum / weight sum |

`read4` 위치는 token ID만으로 계산하고 target이 B0/B1인지 assert한다. metadata, 현재 상태, 정답 값, structural depth는 사용하지 않는다. 두 loss 모두 full-vocabulary 15-token CE다.

필수 pilot cell은 `base4_uniform`, `base4_read4`, `wide4_uniform`, `wide4_read4`, `deep8_uniform`, `deep8_read4` 여섯 개다. 모두 LM seed 0, 같은 첫 8M stream, 같은 effective batch 64, FP32로 실행한다.

### 4.3 Pilot 학습과 winner 선택

- AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.01`, global clip `1.0`을 유지한다.
- 첫 50k prediction tokens에서 LR을 `3e-4`까지 linear warmup한 뒤 8M까지 상수로 둔다.
- milestone은 1M, 3M, 8M이다. 8M current checkpoint만 pilot cell 비교에 사용하며 중간 checkpoint로 cell을 교체하지 않는다.
- `base4_uniform`은 v1.2와 같은 Colab 환경 ID `e74fb1dcf8112ca0`, FP32, microbatch 16을 복원한다. 1M/3M/8M model tensors가 동일 prefix의 v1.2 보존 checkpoint와 bitwise 동일해야 한다. 불일치하면 탐색을 중단하고 구현·환경을 감사한다.
- 모든 cell의 실패·NaN·OOM·실제 tokens/update/time/VRAM을 보존하며 대체 cell을 추가하지 않는다.

Pilot primary metric은 `select/first_repeat_pairs` first member의 **42-cell macro answer CE**다. 보조 metric은 같은 macro full-vocabulary accuracy, `select/general` answer CE/accuracy, first–repeat paired gap이다.

승격 eligibility는 다음을 모두 만족해야 한다.

1. 수치·데이터·checkpoint 검증 통과.
2. `select/general` 정확도가 같은 8M `base4_uniform`보다 2 percentage points 넘게 낮지 않음.
3. first-member 42-cell macro CE가 anchor보다 최소 0.05 nats 낮거나 macro 정확도가 최소 5 percentage points 높음.

Eligible cell 중 first-member macro CE 최소를 winner로 한다. 차이가 `1e-4` nats 이내면 `select/general` 답 CE가 낮은 것, 그다음 `uniform`, 더 적은 파라미터, 위 표의 ID 순서를 사용한다. Eligible cell이 없으면 v1.3을 pilot failure로 종료하며 32M·seed 1/2·SAE/TC를 실행하지 않는다.

Pilot은 원인 진단용 탐색이므로 winner의 효과를 confirmatory 결과로 보고하지 않는다. Architecture main effect와 loss main effect는 여섯 cell 전체를 숨김없이 표로 보고한다.

## 5. Winner 32M 확인 단계

### 5.1 학습과 checkpoint 선택

Winner의 seed 0은 검증된 8M 상태에서 같은 stream을 계속 소비한다. Seed 1·2는 seed 0 gate 통과 뒤 fresh initialization으로 0→32M을 실행한다.

LR은 재예산 가능한 warmup–stable–decay로 고정한다.

- 0–50k: linear warmup to `3e-4`.
- 50k–28.8M: constant `3e-4`.
- 28.8M–32M: linear decay to `3e-5`.
- 32M을 처음 넘는 완전 update에서 종료하며 자동 연장하지 않는다.

선택 checkpoint 후보는 1M, 3M, 8M, 16M, 24M, 28.8M, 30.4M, 32M 경계뿐이다. `select/first_repeat_pairs` first-member 42-cell macro answer CE 최소를 고르고, 동률 `1e-4` 이내에서는 `select/general` 답 CE, 이른 update 순이다. Gate 결과로 checkpoint를 바꾸지 않는다.

### 5.2 새 validation gate

선택한 한 checkpoint에서 다음을 모두 만족해야 한다. 모두 15-token full-vocabulary argmax 기준이다.

1. `gate/general` 전체 READ 정확도 >=99%.
2. fresh other-variable, repeated-update, first-after-set 진단 각각 >=95%.
3. first-member 42-cell macro 정확도 >=95%.
4. first-member의 NOT/AND/OR/XOR 각 operator macro >=95%.
5. first-member의 depth `1`, `2–3`, `4+` 각 macro >=95%.
6. first-member answer 0과 answer 1 각각 >=95%.
7. repeat-member 42-cell macro 정확도 >=95%와 모든 quota/coverage 100%.

Gate는 cell macro와 함께 micro accuracy, answer CE, B0/B1 제한 정확도, bit mass, first–repeat paired CE/정확도 gap을 저장한다. 개별 42 cell은 표본 수와 CI를 보고하지만 작은 cell의 단일 관측을 별도 통과 조건으로 추가하지 않는다.

- Seed 0 실패: v1.3 중단 보고, seed 1·2 및 표현 분석 미실행.
- Seed 0 통과: architecture/loss/data/budget/선택/gate를 동결하고 seed 1·2 실행.
- 세 seed 중 최소 2개가 gate 통과해야 표현 분석으로 진행한다. 실패 seed를 교체하지 않는다.
- 모든 seed 학습과 validation 판정이 끝난 뒤 config 변경 없이 test를 한 번 평가한다. Test 실패로 winner·checkpoint·threshold를 바꾸지 않는다.

## 6. 행동 통과 후 해석 범위

Gate 통과 LM이 최소 2개일 때만 P5 이후를 시작한다.

- 원 규격의 READ block 0 full probe, SAE와 Transcoder `k={4,16}`, fidelity·근사 대체·인과 대조를 필수로 유지한다.
- 모든 block의 READ residual full linear probe는 v1.3의 저비용 진단으로 추가한다. Sparse dictionary layer는 test를 보기 전에 block 0으로 유지하며 layer sweep 결과로 바꾸지 않는다.
- LM seed 0의 sparse seed 1 반복을 유지한다.
- Weighted-loss winner이면 모든 결과 제목·run ID·표에 `read4`를 표시하고, uniform next-token LM에 대한 결론으로 일반화하지 않는다.
- 통과 모델이 1개뿐이면 행동 seed 불안정 결과로 종료하고 SAE/TC를 시작하지 않는다.

## 7. 저장·실행·증빙

- 신규 구현은 `interp_v1_3/`, tests는 `tests_v1_3/`, 실행 기록은 `experiment_v1_3/`만 사용한다.
- CPU corpus 전체 감사와 여섯 cell 모두의 CPU/GPU smoke 전에는 GPU pilot을 시작하지 않는다.
- Colab 실행은 입력 checksum, 환경 lock, CPU/GPU smoke, 여섯 pilot cell, 중단·재개, 결과 다운로드가 포함된 `.ipynb`로 전달한다.
- 후보별 init, milestone current checkpoint, optimizer/RNG/cursor, config/code/data/environment hash를 immutable하게 보존한다.
- Confirmatory winner는 모든 선택 후보 checkpoint와 gate 결과를 보존한다. Pilot 실패나 비승격 cell을 삭제하지 않는다.
- 처리량은 첫 50 updates, peak VRAM, 학습/validation/저장 시간을 분리해 기록한다.

## 8. 계산 예산과 단계 종료

| 단계 | 최대 LM 예산 |
|---|---:|
| Pilot | 6 cells × seed 0 × 8M = 48M + overshoot |
| Confirm seed 0 | winner 8M→32M 추가 24M + overshoot |
| Replication | winner seeds 1·2 × 32M = 64M + overshoot |
| 전체 최악값 | 136M prediction tokens + overshoot |

평가 빈도는 v1.2의 100k마다가 아니라 고정 milestone만 사용해 validation·checkpoint 비용과 반복 선택 편향을 줄인다. 실제 GPU 시간과 저장량은 smoke 처리량으로 다시 계산하고 실행 전 기록한다.

v1.3 설계 완료는 실험 완료가 아니다. 다음 순서를 지킨다.

`설계 동결 → 구현·unit test → 신규 split/train 생성·CPU 감사 → CPU/GPU smoke → 6-cell pilot → winner seed 0 confirm → seed 1·2 → frozen test → 통과 시 P5–P9`

어느 단계든 quota, hash, 재현, 수치 검사가 실패하면 뒤 단계를 시작하지 않는다. 행동 gate를 통과한 LM이 2개 미만이면 SAE/TC 가설을 검증했다고 쓰지 않는다.
