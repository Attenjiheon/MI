# wide4_read4 confirm 실패 이후 아키텍처 제안

작성: 2026-09-20. 상태: **설계 제안, 미학습·미검증**.
이 문서는 v1.3 결과에 대한 후속 설계 산출물이며 기존 동결 계약을 변경하지 않는다.
실행 버전으로 채택할 때 `experiment_v1_4/`, `interp_v1_4/`, `tests_v1_4/`,
`data/language_v1_4/`에 별도 계약·구현·코퍼스를 구성한다.
기계 판독 제안: [next_architecture_proposal.json](./next_architecture_proposal.json).

## 1. 결정

후속 단일 후보는 **deepwide12_read4: 12-block, residual width 256, 4 heads,
MLP width 1024, 9,485,312 parameters**로 한다. Fresh initialization으로 명목 64M
prediction tokens를 학습하는 구성을 제안한다. 목표는 기존과 같은 행동 gate의
독립 split 통과와 LM seed 재현이며, architecture만의 인과 효과를 추정하는 실험은 아니다.

12층·256폭·64M은 관측값으로 추정한 최적점이나 통과 확률이 아니다. 폭을 희생하지 않는
계산 깊이 증가와 더 긴 학습을 함께 확보하는 공학적 선택이다. 성공 확률의 숫자는
현재 결과만으로 산정할 수 없다.

## 2. 근거와 추론의 경계

원본: `../evidence/v1_3_confirm_wide4_read4_seed0_evidence_20260920T095655014484.zip`.
ZIP SHA-256: `c7cca1856c51cddcf778c50c946fad57ca68677a6dde660fdc4676cd204e57d9`.

| 관측 | 설계에 주는 근거 | 입증되지 않은 주장 |
|---|---|---|
| Gate first depth 1 99.22%, depth 2–3 94.20%, depth 4+ 90.63% | 합성 의존성을 처리할 attention/MLP 단계 확대 | 구조 depth 4+에는 특정 층 수가 반드시 필요하다는 주장 |
| Gate XOR 90.23%, AND/NOT/OR 각각 95% 이상 | 두 입력의 상태를 함께 보존·결합할 용량 확보 | 실패 원인이 XOR MLP 하나 또는 attention 하나라는 진단 |
| 8M deep8_read4(8×128)는 wide4_read4(4×184)보다 first CE가 나쁨: 0.5083 vs 0.4932 | 깊이 증가 시 폭을 줄이지 않음 | 깊이가 항상 유리하거나 기존 deep8이 32M에서도 열세라는 주장 |
| Select first CE 28.8M→30.4M→32M: 0.1577→0.1474→0.1169 | 다음 버전 예산을 사전에 64M으로 고정 | 64M이면 통과한다는 외삽; 개선이 LR decay 때문이라는 단정 |
| Gate repeat 100%, first 94.68% | 이전 답이 없는 first metric과 모든 하위 gate 유지 | 모델이 오직 복사만 한다는 결론 |

XOR은 두 입력 중 하나만 알아서는 일반적으로 답을 확정할 수 없다. AND/OR는 특정
입력값에서 한쪽 정보만으로도 답을 정할 수 있다. 따라서 XOR 취약성은 두 상태의
정확한 검색·결합이 어려운 가설과 맞지만, 그것만으로 내부 병목을 특정할 수 없다.

상태 추적 Transformer가 계층적 계산을 학습할 수 있다는 근거는
[Liu et al., Transformers Learn Shortcuts to Automata](https://arxiv.org/abs/2210.10749)에 있다.
그 논문의 계산적 shortcut은 이번 실험의 이전 READ 답 복사 shortcut과 같은 뜻이 아니다.
층 수와 프로그램 갱신 수를 1:1로 대응시키지 않는다. 별도 순열 합성 과제에서 서로 다른
상태 추적 전략이 학습될 수 있다는 관측은
[Li et al., (How) Do Language Models Track State?](https://arxiv.org/abs/2503.02854)를 참고했다.
어느 논문도 이 코퍼스에서 12×256의 gate 통과를 검증하지 않았다.

## 3. 아키텍처 계약

| 항목 | 제안 값 |
|---|---|
| ID | `deepwide12_read4` |
| 형태 | causal decoder-only, Pre-LN, attention → MLP |
| blocks / residual width | 12 / 256 |
| heads / head width | 4 / 64 |
| QKV / attention output | Linear(256,768,bias=True) / Linear(256,256,bias=True) |
| MLP | Linear(256,1024,bias=True) → exact GELU → Linear(1024,256,bias=True) |
| LayerNorm | attention 전·MLP 전 및 final, affine=True, eps=1e-5 |
| Vocabulary / context | 15 / 768 |
| 위치 | RoPE base 10,000, Q/K의 head 64차원 전체 적용 |
| Embedding / unembedding | 15×256 / 256×15, untied, unembedding bias 없음 |
| Dropout | 0 |
| 초기화 | Linear/embedding Normal(0,0.02²), Linear bias 0, LN weight 1/bias 0 |
| Attention/MLP output 초기 weight 배율 | 1/sqrt(24) = 0.2041241452 |
| 입력 | token IDs, padding mask |
| READ 예측 | `READ <VAR>`의 VAR 위치 logits로 다음 B0/B1 예측 |
| 계산 정밀도 | FP32; AMP/TF32/compile 비활성 |

한 block은 `12d² + 13d` parameters, embedding+final LN+unembedding은 `32d`다.
따라서 총수는 `12 × (12 × 256² + 13 × 256) + 32 × 256 = 9,485,312`다.
현행 model의 구성 클래스로 메모리에서 인스턴스화하여 실제 parameter 수와 대조했다.
이는 크기 검산이며 신규 버전의 forward/backward, CPU/GPU smoke 통과를 뜻하지 않는다.

4 heads를 유지하면서 head width를 46→64로 늘린다. 변수 하나당 head 하나를 강제로
지정하지 않는다. 12층의 역할이나 연산별 전용 층도 미리 할당하지 않는다.
GELU·LayerNorm·RoPE를 유지하여 새 gating, recurrence, scratchpad, 상태 라벨 등의
동시 변경 없이 기존 SAE/TC 분석 지점을 이어받는다.

## 4. 학습 계약 제안

- Loss: 기존 token-only `read4`, READ 답 위치 weight 4, 나머지 유효 target weight 1.
  Full-vocabulary CE의 weighted sum / weight sum이며 metadata/state auxiliary loss는 없다.
- Fresh initialization, LM seeds 0/1/2. v1.3 checkpoint를 확장하거나 이어 학습하지 않는다.
- AdamW: lr 3e-4, betas (0.9,0.95), eps 1e-8, weight decay 0.01(matrix/embedding만),
  global gradient clip 1.0. Effective batch 64 sequences.
- Microbatch는 GPU smoke에서 16→8→4→2→1 순으로 맞추고 accumulation=64/microbatch.
  본학습 시작 전 고정·기록한다. 동일 환경·microbatch의 저장/재개 동등성을 검증한다.
- LR: 0–50k linear warmup to 3e-4; 50k–57.6M constant 3e-4;
  57.6M–64M linear decay to 3e-5.
- 명목 예산은 처음부터 64M. 최초 도달·초과한 완전 update에서 종료한다.
  중간 성능으로 32M에서 gate를 열거나 64M 이후 자동 연장하지 않는다.
- Select checkpoint 후보는 1M, 3M, 8M, 16M, 24M, 32M, 48M, 57.6M, 60.8M, 64M.
  기존처럼 first-member 42-cell macro CE 최소; 최소값과 1e-4 이내 후보에서는
  select/general CE, 이른 update 순으로 선택한다.
- 32M 지점은 학습 곡선용이다. v1.3은 28.8M부터 decay했지만 이 모델은 57.6M부터
  decay하므로, 두 버전의 32M 차이를 architecture만의 효과로 주장할 수 없다.

Architecture와 예산을 함께 바꾸므로 최종 성공은 이 결합 설정에 대한 증거다.
독립적인 architecture 효과가 연구 질문이라면 같은 64M 스케줄의 wide4_read4 대조군을
별도로 사전 등록해야 한다. 이 제안의 기본 실행 후보는 하나이며 대조군 비용은 포함하지 않는다.

## 5. 새 평가와 데이터

이미 확인한 v1.3 select/gate는 후속 설계의 개발 근거가 되었다. 다음 버전의 gate로
재사용하지 않는다. v1.0–v1.3 test 점수는 열지 않고 다음 버전용 select/gate/test를
새 namespace `20260920|language-v1.4|<purpose>|<index>`에서 생성한다.
PCG64 seed는 namespace SHA-256의 앞 8 bytes unsigned big-endian이다.

원 언어, IID 생성 분포, 42-cell 정의를 유지한다. Select/general은 512 sequences,
select pairs는 cell당 64 pairs(총 2,688)로 늘려 깊이·연산별 선택 수치의 표본을 확보한다.
이는 v1.3의 cell당 16보다 4배 많으며 결과 비교 시 다른 split임을 표시한다.
Gate는 general 1,024, 기존 진단 각 1,024, pair cell당 64;
test는 general 2,048, 기존 진단 각 2,048, pair cell당 128을 유지한다.

새 평가·interpretation·causal split을 예약할 때 과거 모든 split과 train의 exact sequence/
causal-prefix hash를 거부한다. v1.3의 train 32,004,917-token prefix는 byte-identical하게
보존하고 새 IID shard로 최초 64M 도달 update까지 확장한다. 확장 train은 과거 및 신규
예약 hash 전체를 거부한다. 반복·shuffle·상태 라벨 기반 정답 균형화는 없다.
모든 seed가 같은 stream을 같은 순서로 한 번 소비한다.

신규 corpus root의 manifest에는 재사용 prefix hash, 생성 RNG state, 실제 token/update/
cursor/overshoot 및 신규 split hash를 기록한다. 신규 파일 경로와 version metadata 때문에
prefix byte identity가 깨지지 않도록 재사용 provenance는 외부 manifest에 둔다.

## 6. 통과 규칙과 결과 해석

선택한 checkpoint 하나에서 seed당 gate를 한 번 평가한다. 기존 수치 기준을 유지한다:

1. General READ full-vocabulary accuracy ≥99%.
2. 세 legacy diagnostic 각각 ≥95%.
3. First-member 42-cell macro ≥95%.
4. First의 NOT/AND/OR/XOR operator macro 각각 ≥95%.
5. First의 depth 1/2–3/4+ macro 각각 ≥95%.
6. First의 answer 0/1 accuracy 각각 ≥95%.
7. Repeat 42-cell macro ≥95%, 모든 quota/coverage 100%.

개별 42 cell에는 count와 CI를 보고하되 추가 gate를 붙이지 않는다.
First/repeat paired gap, full-vocabulary와 binary accuracy, bit mass, CE를 계속 보고한다.
정확한 structural depth와 READ까지의 거리 층화는 진단용이며 사후 선택 기준으로 쓰지 않는다.

Seed 0 실패 시 중단; 통과 시 동일 설정으로 seed 1·2를 학습한다. 최소 2개 seed가
통과해야 P5 이후에 진입한다. 학습한 winner seed들의 validation 판정·checkpoint를 모두
동결한 뒤 test를 한 번 평가한다. Test로 재선택하지 않는다.

필수 sparse 위치는 block 0으로 유지하며 residual/MLP 차원만 256에 맞춘다.
기존 dictionary width 및 sparse 학습 예산을 바꾸려면 이후 별도 계약이 필요하다.
모든 block의 full probe는 진단용이고 sparse 위치를 사후 선택하지 않는다.
행동 성공만으로 block 0에 완전한 상태 표현이 있다는 주장을 하지 않는다.

## 7. 비용·완료 상태

Parameters는 wide4 대비 약 5.78배다. 같은 길이·batch에서 주요 dense matrix 연산량은
`L*d²` 기준 약 5.81배, attention QK/AV는 `L*d` 기준 약 4.17배다.
64M은 기존 32M의 토큰 수 2배이므로 주요 연산량을 단순 합산한 전체 학습 비용은
대략 8.35–11.61배 범위다. Padding, GPU 활용도, kernel, checkpoint와 평가 비용이 달라
이는 wall-clock 예측이 아니다. 실제 GPU smoke로 처리량과 peak VRAM을 측정한다.

Seed 0은 64M+overshoot, 복제까지 최대 192M+seed별 overshoot다.
기존 6-cell pilot tournament는 반복하지 않는다. Smoke 실패는 수정·재검증하고,
본학습의 수치 실패나 행동 실패는 숨기거나 seed를 교체하지 않는다.

현재 완료: 관측 기반 설계, 기계 판독 제안, parameter 수 검산.
현재 미완료: v1.4 정식 계약·구현·새 corpus·감사·CPU/GPU smoke·Colab notebook/input bundle·학습.
이 문서는 Colab 실행 지시나 phase 완료 선언이 아니다.
