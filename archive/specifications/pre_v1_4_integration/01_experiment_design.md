# 작은 프로그램 언어에서 SAE의 상태·연산 표현 평가

작성일: 2026-09-09  
상태: 구현 전 실험 계획 v1.0  
기술 부록: [인공어 규칙 및 코퍼스 생성 알고리즘](./02_language_and_corpus.md)

## 초록

본 실험은 Boolean 프로그램 언어를 학습한 작은 transformer에서 sparse autoencoder(SAE)의 표현 분리 능력과 인과적 유효성을 평가한다. 네 변수에 대한 대입·반전·이항 논리 연산과 읽기로 코퍼스를 생성하고, 모든 유효 토큰에 동일한 next-token cross-entropy를 적용해 모델을 학습한다. 독립적인 프로그램 실행기가 제공하는 정답으로 행동을 검증한 뒤, linear probe와 SAE를 통해 현재 값·과거 값·연산 입력 관계의 접근성과 분리도를 측정한다. 최소 대조쌍의 feature 패칭으로 출력 변화를 검증하며, 원래 좌표와 무작위 방향을 기준선으로 사용한다. 모델 학습, 표현 분석, 인과 평가를 구분하여 해석 도구의 성공과 실패를 모두 재현 가능한 결과로 보고한다.

## 1. 실험 목적

작은 decoder-only transformer에 Boolean 상태를 갱신하고 읽는 프로그램 언어를 순수 next-token prediction으로 학습시킨다. 행동 수준에서 과제를 안정적으로 해결하는 모델을 확보한 뒤, linear probe와 sparse autoencoder(SAE)를 이용하여 상태·연산·입력 관계가 어떤 형태로 표현되는지 조사한다. SAE feature의 개입이 예측에 미치는 영향도 측정한다.

주 연구 질문:

> 모델에서 선형적으로 읽을 수 있는 과제 정보는 SAE의 소수 feature로 분리되는가? 선택한 feature의 개입은 반사실적 정답 방향으로 모델 출력을 바꾸는가?

세부 질문:

1. 현재 값과 덮어쓰기 전 값이 구별되는가?
2. 같은 값 또는 입력 관계를 서로 다른 연산·변수·문맥에 걸쳐 읽을 수 있는가?
3. SAE는 변수별 상태, 전체 상태, 연산별 결합 특징 중 무엇을 포착하는가?
4. 높은 reconstruction 성능과 높은 의미적 대응 및 인과 효과가 일치하는가?

## 2. 연구 배경과 설계 원칙

SAE의 평가에는 재구성 오차뿐 아니라 발견된 feature가 어떤 과제 정보를 나타내며 예측에 어떤 영향을 미치는지에 대한 검증이 필요하다. 실행 의미가 명확한 프로그램 언어는 각 시점의 상태와 연산 결과를 정확히 계산할 수 있어, 이러한 검증을 위한 통제된 환경을 제공한다. 본 실험은 다음 원칙에 따라 데이터와 평가를 구성한다.

| 요구 | 구현 원칙 |
|---|---|
| 규칙의 의미를 일관되게 정의한다 | 모든 변수 조합에 동일한 SET·NOT·AND·OR·XOR 의미를 적용한다 |
| 정답 예측의 빈도를 확보한다 | 짧은 갱신 블록마다 READ와 정답을 배치한다 |
| 연산 합성을 평가한다 | 이항 상태 갱신 및 일부 무질의 연산 연쇄를 포함한다 |
| 순수 LM 목적함수를 사용한다 | 모든 유효 next-token 위치에 동일한 CE를 적용한다 |
| 평가와 학습을 분리한다 | 상태 라벨은 분석에만 사용하고 LM/SAE 학습에는 사용하지 않는다 |
| 계산 예산을 통제한다 | 단일 seed 파일럿, 학습 토큰 상한, 명시적 통과 조건을 둔다 |

정답 위치 비율은 loss 항의 비율이다. 이를 gradient 기여율이나 signal-to-noise ratio로 해석하지 않는다.

## 3. 과제 개요

Boolean 변수 A, B, C, D를 사용한다. SET은 값을 지정하고, NOT은 목적변수를 반전한다. OP dst src는 갱신 직전 두 값을 연산하여 dst에 저장한다. src는 변하지 않으며 이항 연산의 dst와 src는 다르다.

각 시퀀스는 무작위 초기화와 여러 블록으로 이루어진다. 한 블록은 갱신 명령 1~3개 뒤에 READ 변수와 정답 토큰 하나를 붙인다. 초기화 이후 학습되는 정답은 읽기 결과뿐이며, 중간 상태 전체나 계산 추적은 입력에 제공하지 않는다.

```text
BOS SET A B1 SET B B0 SET C B1 SET D B0
XOR A C READ A B0
NOT B READ B B1
OR A B READ A B1
AND A C READ C B1 EOS
```

위 코드 블록의 공백으로 나뉜 항목은 각각 토큰 하나다. 줄바꿈은 설명용이며 토큰이 아니다. 자세한 문법과 의미는 별도 명세를 따른다.

전체 논리 상태는 16개다. 연구 범위는 이 유한 상태 공간에서의 상태 갱신, 새로운 프로그램 문맥에서의 합성, 내부 표현 분석이다. 모델이 반드시 인간이 정의한 상태 전이 알고리즘을 그대로 구현한다고 가정하지 않는다.

## 4. 사전 가설과 해석 범위

| ID | 검증 가능한 예상 | 반대 결과의 의미 |
|---|---|---|
| H1 | READ 변수 위치에서 현재 답이 선형적으로 예측된다 | 정보의 위치·형태가 다르거나 행동 학습이 불충분할 수 있다 |
| H2 | 일부 과제 라벨을 SAE의 1~4개 latent로 예측할 수 있다 | SAE에서 분산되어 있거나 해당 설정이 분리하지 못했을 수 있다 |
| H3 | SAE의 대응 feature는 무작위 방향보다 문맥 전이가 좋다 | 단순 방향 선택만으로 얻는 효과를 넘지 못했을 수 있다 |
| H4 | 선택 feature 패칭이 반사실적 답의 상대 logit을 높인다 | 해당 위치의 개입 효과가 약하거나 정보가 다른 경로로 전달될 수 있다 |

가설은 성공 판정에 필요한 전제가 아니다. 행동 검증을 통과한 모델에서 H2/H3/H4가 실패해도 유효한 결과다. SAE 실패를 감추기 위해 LM이나 데이터 설정을 사후 변경하지 않는다.

프로브 정확도는 선형적 접근 가능성의 증거다. 규칙의 완전한 학습, 정보의 실제 사용, 유일한 내부 알고리즘을 입증하지 않는다. 데이터 생성기의 상태는 과제 수준의 정답이며 모델 내부 feature 방향의 정답은 아니다.

## 5. 단계별 범위 및 실행 순서

### 단계 A: CPU 생성기 검증

1. 문법과 실행기를 구현하고 수작업 예제 및 진리표와 대조한다.
2. 생성된 모든 정답을 독립적인 replay 실행기로 재검산한다.
3. 분할 중복, 합성 holdout 누출, 정답 위치 및 상태 라벨 시점을 검증한다.
4. 소규모 코퍼스 통계를 저장한다. 이상이 없을 때만 GPU 학습을 시작한다.

### 단계 B: 단일 LM 파일럿

처음부터 여러 모델 크기나 seed를 병렬 탐색하지 않는다. 아래 출발 설정 하나를 사용한다.

| 항목 | 출발 설정 |
|---|---|
| 모델 | 2-layer causal decoder-only transformer |
| residual width / attention heads | 128 / 4 |
| MLP | hidden width 512, GELU |
| 정규화 | Pre-LayerNorm |
| 위치 표현 | RoPE; 설정을 학습·평가에서 고정 |
| dropout | 0 |
| vocabulary | 별도 명세의 15개 토큰 |
| context capacity | 768; 학습 길이는 최대 302토큰 |
| optimizer | AdamW, lr 3e-4, betas (0.9, 0.95), eps 1e-8 |
| weight decay | 0.01; bias와 정규화 파라미터 제외 |
| gradient clipping | global norm 1.0 |
| batch | effective 64 sequences; microbatch 및 accumulation은 메모리에 맞춤 |
| 파일럿 | LM seed 0, 비패딩 예측 토큰 최대 1M |
| LR schedule | 최초 50k 토큰 linear warmup, 이후 상수 lr |
| validation 간격 | 약 100k 비패딩 예측 토큰마다 |

이 수치는 검증된 최적값이 아닌 사전 출발값이다. 실제 라이브러리 버전, precision, 파라미터 수, device, microbatch를 결과 manifest에 기록한다. Batch 변경 시 effective batch를 유지한다. 한 optimizer update가 끝날 때만 평가하며 토큰 예산의 마지막 batch 초과량을 기록한다.

데이터는 온라인으로 새로 생성한다. 예산은 epoch 대신 처리한 비패딩 예측 토큰 수로 센다. BOS 자체는 예측 대상이 아니며, EOS는 예측 대상이다. PAD와 존재하지 않는 target만 loss에서 제외한다. Attention 및 loss mask의 역할을 구분한다.

### 단계 C: 진행/중단 기준

| 관측 | 다음 행동 |
|---|---|
| 일반 validation 답 정확도 ≥99%, 핵심 진단 조건별 ≥95% | SAE 단계로 진행 |
| 1M에서 미달이나 최근 3회 중 2회 이상 답 CE가 개선 | 동일 설정으로 누적 3M까지 한 번 연장 |
| 기준 미달이고 개선도 뚜렷하지 않음 | 중단하고 데이터/구현을 CPU에서 확인 |
| 3M에서도 기준 미달 | 대규모 sweep 금지; 짧은 시퀀스로 단순화한 v1.1을 별도 계획으로 기록 |

핵심 진단은 (a) 직전 목적변수와 다른 변수 읽기, (b) 관련 변수에 복수 갱신 존재, (c) 유효한 SET 이후 해당 변수를 처음 읽기다. 각 진단 validation은 최소 512개 독립 시퀀스의 지정 target을 사용한다. 생성기의 feasibility와 표본 수를 CPU 단계에서 확인한다.

길이 외삽과 합성 holdout은 연구 결과이며 필수 통과 gate가 아니다. 이 평가에서 실패했다는 이유로 설정을 사후 변경하지 않는다.

Checkpoint는 사전 지정한 일반 validation 정답 CE가 최소인 것을 선택한다. 동률이면 이른 checkpoint를 선택한다. 해당 checkpoint가 gate도 만족해야 한다. 전체 LM loss 또는 test 성능으로 선택하지 않는다.

### 단계 D: 재현 및 해석

파일럿이 통과하면 파일럿에서 사용한 총 토큰 예산과 설정을 고정하고 LM seed 1, 2를 학습한다. 모든 seed의 행동 결과를 보고하며 실패 seed를 다른 seed로 대체하지 않는다. 해석은 행동 gate를 통과한 모델에 대해 수행하고 적용 모델 수를 명시한다. seed 0에서만의 결과와 seed 간 재현 결과를 분리한다.

## 6. 데이터 및 평가 집합

정확한 생성법, seed 역할, holdout 패턴은 별도 명세를 따른다.

| 데이터 | 사용 | 초기 규모 |
|---|---|---|
| LM train stream | 순수 LM 학습 | 토큰 예산으로 제한 |
| 일반 validation | checkpoint 선택 | 고정 512 sequences |
| 진단 validation | 행동 gate | 조건당 512 target sequences |
| 일반 test | 최종 행동 보고 | 고정 2,048 sequences |
| 진단 test | 실패 유형 평가 | 조건당 1,024 target sequences |
| 합성 holdout test | 미노출 3연산 패턴 | 2개 패턴 × 1,024 target sequences |
| 길이 test | 33~48블록 시퀀스 | 1,024 sequences, 선택 확장 |
| 해석 train pool | SAE 학습, probe 학습 | READ 및 update 각 50k positions까지 |
| 해석 validation pool | feature·threshold·probe 선택 | 각 10k positions까지 |
| 해석 test pool | 최종 해석 지표 | 각 20k positions까지 |

해석 pool은 별도의 시퀀스 집합이며 LM train/validation/test와 정확 중복되지 않는다. 동일 시퀀스의 모든 위치는 반드시 같은 split에 속한다. READ와 update 위치를 같은 시퀀스에서 추출하는 것은 허용한다. 합성 holdout 패턴은 해석 train/validation에도 넣지 않는다.

초기 규모는 상한 목표다. 중단된 실험에서는 실제 표본 수를 보고한다. 대량 activation 수집 전에 작은 pool로 전체 파이프라인을 한 번 실행한다.

## 7. 행동 평가

다음 지표를 함께 저장한다.

- 모든 유효 위치의 LM CE.
- 정답 위치의 full-vocabulary CE와 full-vocabulary argmax 정확도.
- B0/B1만 비교한 이진 정확도와 두 토큰의 합산 확률질량. 이진 정확도만으로 gate를 통과시키지 않는다.
- 명령 종류, 블록 길이, 입력 진리표, 변수, 답 값, 진단 조건별 정확도 및 표본 수.
- macro 정확도: 사전 정의한 strata를 동일 가중 평균. 빈 strata는 0으로 채우지 않고 NA 및 coverage로 보고.
- 정상 문맥의 teacher-forced 성능. 자유 생성/연쇄 오류 평가는 선택 부록이며 기본 평가와 섞지 않는다.

학습 분포의 정답 균형을 강제하지 않는다. AND/OR 누적으로 편향이 생길 수 있으므로 실제 majority baseline을 split별로 계산한다. 평가에는 답 B0/B1을 균형화한 별도 target subset과 연산별 입력 4칸 macro 지표를 제공한다. 순수 train 분포 점수도 보존한다.

독립 추측, 가장 최근 SET 값, 직전 답 복사 같은 간단한 기준선의 구현과 실제 정확도를 각 평가 집합에서 측정한다. 기준선의 성능은 해당 집합의 상태·연산·정답 분포에 따라 달라질 수 있으므로 표본 수와 함께 보고한다.

## 8. Activation 위치와 probe

### 8.1 READ 위치

`READ A B1`에서 A 토큰을 처리한 직후, B1을 입력하기 전 activation을 사용한다. 주 분석 대상은 첫 transformer block의 residual output, 보조 대상은 둘째 block의 residual output이다. 후자는 final LayerNorm 이전으로 통일한다.

라벨:

- 읽는 변수의 현재 값.
- 읽는 변수가 직전 갱신되기 전 값. 과거 갱신이 없으면 missing.
- 읽는 변수의 종류.
- 현재 전체 4-bit 상태 및 변수별 값: 탐색적 라벨.

현재/직전 값이 다른 subset을 별도로 평가한다. 모든 변수 상태가 현재 READ 위치에서 읽혀야 한다고 요구하지 않는다.

### 8.2 갱신 위치

`XOR A C`에서 C 토큰을 처리한 직후의 activation을 사용한다. SET은 값 토큰, NOT은 변수 토큰 직후다. 시뮬레이터의 갱신 전·후 라벨을 분리해 저장한다.

라벨:

- 연산자, 목적변수, 원본변수.
- 갱신 전 목적변수 값과 원본변수 값.
- 두 입력이 서로 다른지 여부.
- 갱신 후 목적변수 값.
- 연산자 × 입력 진리패턴 결합 라벨.

연산자가 읽힌다는 사실만으로 논리 계산이 표현되었다고 결론내리지 않는다. 입력 관계와 출력 라벨은 여러 연산에 걸쳐 비교한다. 특히 XOR 안에서만 입력 불일치와 결과 1을 구분하려 하지 않는다.

READ 분석을 먼저 완결한 뒤 update 분석을 실행한다. Update 표현이 weak하게 읽히는 경우, 질의 시점의 검색 방식으로 푸는 모델일 가능성도 남긴다.

### 8.3 Probe 프로토콜

- Frozen activation 위에 L2-regularized logistic regression. 이진/다중 클래스 형식을 구분한다.
- 정규화 강도는 고정 grid {0.01, 0.1, 1, 10} 중 validation에서 선택한다. 라이브러리의 C 또는 lambda 정의를 manifest에 기록한다.
- 학습 전 동일 모델, 현재 토큰 ID 및 위치만 사용하는 기준선, shuffled-label 대조군을 포함한다.
- IID 평가 외에 목적변수 D에 해당하는 사례를 probe 학습에서 제외하고 D에서 평가하는 전이 분석을 한다. 연산 전이는 AND/OR에서 학습하고 XOR에서 평가하는 별도 분석으로 고정한다.
- 전이 분석은 라벨별 class support가 있는지 먼저 확인한다. Probe의 학습 분할과 SAE의 비지도 학습 분할을 구분해 보고한다.

## 9. SAE와 기준선

### 9.1 최초 실행 범위

첫 층 READ residual에서 TopK SAE dictionary width 512, k ∈ {4, 16} 두 설정을 비교한다. 처음에는 LM seed 0, SAE seed 0으로 실행한다. READ 결과를 완료한 뒤 같은 방식으로 update 위치를 별도로 분석한다. 서로 다른 위치를 하나의 학습 pool에 섞지 않는다.

SAE는 개념 라벨 없이 activation MSE로 학습한다. Encoder의 ReLU 출력 중 상위 k개를 유지하고 나머지는 0으로 한다. Decoder는 column norm을 1로 정규화한다. 기본 설정에서는 auxiliary loss, label loss, dead-latent 재초기화를 추가하지 않는다. dead-latent 비율을 보고한다.

입력 전처리: 해석 train의 평균 μ와 scalar RMS scale s를 사용해 x=(h−μ)/s로 변환한다. 차원별 whitening은 하지 않는다. μ와 s를 validation/test에 고정 적용한다. Raw residual로 개입할 때 s를 다시 곱한다.

초기 optimizer는 Adam lr 1e-3, batch 512, 최대 5k updates, 250 updates 간격 validation MSE 평가로 둔다. k별 최소 validation MSE checkpoint를 선택한다. k를 semantic test 성능으로 선택하지 않고 두 결과를 모두 보고한다. 수치는 파일럿 출발 설정이며 실제 연산량과 수렴 여부를 함께 기록한다.

### 9.2 비교 대상

1. 전체 residual linear probe: 정보 접근 가능성 기준.
2. 원래 residual 좌표 중 1개 또는 최대 4개를 선택한 probe.
3. 고정 seed의 unit-norm random projection 512개 중 1개 또는 최대 4개를 선택한 probe.
4. SAE latent 중 1개 또는 최대 4개를 선택한 probe.

Raw 좌표 수 128과 dictionary 512의 후보 수 차이를 명시한다. 추가로 SAE/random 후보 128개를 사전 seed로 추출한 후보 수 일치 분석을 수행한다. 큰 후보군에서 feature를 찾기 쉬워지는 효과를 숨기지 않는다.

Feature 선택은 해석 train에서 단변량 점수로 후보를 정렬하고 순차적으로 최대 4개를 선택한다. Logistic probe fitting은 train, 개수와 threshold 선택은 validation, 최종 수치는 test에서 산출한다. 순차 선택의 세부 점수와 tie-break는 코드에 고정한다. 라벨별 다른 feature 집합을 허용하되 이를 비지도 발견 자체와 구분해 '사후 감독 평가'로 기술한다.

### 9.3 지표

- Balanced accuracy, macro F1; 이진 라벨은 AUROC도 보고.
- 단일/최대 4개 feature 성능과 full probe 성능 차이.
- 현재 값/오래된 값 구분 및 변수·연산 전이.
- Normalized reconstruction MSE, explained variance, 실제 L0, dead-latent 비율.
- Reconstruction으로 해당 위치를 대체했을 때 정답 CE 및 정확도 변화.

Reconstruction MSE는 해당 activation 평균 예측기의 MSE로 정규화한다. 평균과 분모를 어느 split에서 계산했는지 기록한다. Reconstruction 행동 평가는 지정 target 한 곳의 activation만 교체하는 것이 기본이다. 전체 위치 동시 대체는 별도 실험이다.

## 10. 인과 개입

### 10.1 반사실적 데이터

동일한 token 길이·명령 구조·질의를 갖는 원본/대조 프로그램을 만든다. 초기화 또는 과거 SET의 B0/B1 한 토큰을 뒤집고 실행기를 다시 돌린다. target READ 정답이 달라지고 target 이전의 모든 답 토큰은 같은 사례만 기본 pair로 사용한다. 따라서 target 직전 prefix는 의도한 입력 한 토큰만 다르다.

단순히 오래된 SET을 바꾼 것만으로 target이 달라진다고 가정하지 않는다. 뒤의 논리 연산에 의해 영향이 사라질 수 있으므로 실행기로 검증한다. 갱신이 전혀 없는 기억 조건과 이후 연산을 거치는 합성 조건을 분리한다.

선택적 효과 대조군은 같은 과거 수정에 대해 정답이 변하지 않는 다른 변수 READ를 사용한다. 실제 시뮬레이션에서 그 변수로 효과가 전파되지 않았는지 검증한다.

### 10.2 개입 방식

정규화 activation x와 latent z에 대해, 사전 선택한 집합 J를 교체한다.

```text
x_patched = x_original + D[:, J] @ (z_counterfactual[J] - z_original[J])
h_patched = μ + s * x_patched
```

원래 activation에 decoder 차이만 더하므로 재구성 잔차는 보존된다. 모델의 나머지 계산은 다시 실행한다. 주 개입은 첫 층 READ 위치이고 다른 층은 보조 분석이다.

### 10.3 대조군과 해석

- 무개입, reconstruction-only, 선택 feature 개입.
- 동일 개수의 random latent 집합. Δh norm과 latent 변화량이 유사한 집합을 validation에서 정의한 bin 규칙으로 매칭한다. 매칭 실패 표본 수를 보고한다.
- 전체 activation을 대조 예제의 activation으로 교체하는 full patch 대조군. 이 또한 효과를 보장하지 않는다.
- 반사실적 답 대비 원래 답의 logit margin 변화, answer flip rate, unchanged-target의 불필요한 flip rate.

정답 변화는 모델 내부의 유일한 계산 경로를 입증하지 않는다. 개입 무효도 비사용의 증명은 아니다. Attention의 다른 위치/KV 경로가 정보를 전달할 수 있다. 결과를 해당 layer·position·feature 집합에 대한 증거로 한정한다.

최종 causal test는 changed-target pair 1,024개, unchanged-target pair 1,024개를 목표로 한다. Pair 선택과 feature 선택은 test 결과를 보기 전에 고정한다. 모델이 원본·대조 모두 맞힌 subset의 효과와 전체 pair 효과를 함께 보고한다.

## 11. 반복, 불확실성 및 예산

- 주요 결과는 LM seed별로 개별 표시하고 3개 seed의 평균/범위를 함께 제공한다. 3개로 정밀한 모집단 추정을 주장하지 않는다.
- LM seed 0의 주 READ SAE는 SAE seed 1을 추가해 학습 초기화 민감도를 확인한다.
- 토큰들이 같은 시퀀스에 묶여 있으므로 bootstrap은 시퀀스 또는 counterfactual pair 단위로 수행한다. 1,000회 bootstrap을 기본으로 한다.
- LM 학습 최대 예산은 기본 3 seeds × 3M = 9M 비패딩 예측 토큰이다. 이는 실제 GPU 시간 보장이 아니다.
- SAE 학습, activation 추출 및 반복 validation 비용은 LM 학습 예산과 별도로 기록한다. 첫 파일럿 처리량으로 벽시계 시간을 추산한 뒤 다음 단계로 간다.
- READ의 핵심 그림이 나오기 전 update SAE, transcoder, crosscoder, 변수 8개 확장은 시작하지 않는다.

## 12. 최종 산출물

1. 결정적 simulator, corpus generator, split manifest 및 CPU 검증 코드.
2. LM 학습 코드, checkpoint, 환경·seed·실제 토큰 수·처리 시간.
3. Activation cache, probe/SAE 설정 및 결과.
4. 그림: 행동 학습 곡선과 조건별 성능; feature 접근성/분리도; reconstruction과 인과 개입 효과.
5. 성공·실패를 모두 포함한 짧은 보고서 및 재현 명령.

완료 조건은 행동 검증을 통과한 모델에 대해 READ 위치의 기준선·SAE 두 설정·reconstruction·인과 대조 실험을 재현 가능하게 보고하는 것이다. Update 분석은 계획된 2차 분석이며 자원 제한으로 생략하면 명시한다. Transcoder와 crosscoder는 필수 완료 조건이 아니다.

## 13. 관련 연구와 본 실험의 위치

- [Hewitt & Liang, Designing and Interpreting Probes with Control Tasks (2019)](https://aclanthology.org/D19-1275/): probe 성능과 표현에 대한 해석을 구분하는 배경.
- [Transformers Learn Shortcuts to Automata](https://arxiv.org/abs/2210.10749): 상태 과제의 행동 성공이 특정 순차 알고리즘의 구현을 보장하지 않는다는 배경.
- [Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models](https://arxiv.org/abs/2408.00113): 알려진 과제 상태를 이용한 SAE 평가의 선행연구.
- [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944): 후속 MLP 계산 분석을 위한 참고.
- [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/): 후속 층간 표현 비교를 위한 참고.

본 문서는 독창성에 대한 전수 문헌 검토나 성공 결과를 주장하지 않는다. 위 논문을 배경으로, 작은 상태 갱신 과제에서 현재성·문맥 전이·인과 효과를 함께 평가하는 재현 가능한 학부 연구를 설계한다.
