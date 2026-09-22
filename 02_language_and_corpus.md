# 인공어 규칙 및 코퍼스 생성 알고리즘

작성일: 2026-09-09
상태: v1.4 본 실험 코퍼스 통합 규격 (2026-09-22)
관련 문서: [전체 실험 설계서](./01_experiment_design.md)

## 1. 목적과 적용 범위

Boolean 변수 네 개를 갱신하고 읽는 autoregressive 프로그램 언어를 정의한다. 프로그램, 정답, 분석 라벨은 CPU에서 정확하게 생성한다. 상태·진리표·의존성 라벨은 별도 metadata로 저장하며 모델 입력이나 학습 loss에 추가하지 않는다.

문법과 실행 의미는 필수 규격이다. IID 샘플링 확률은 초기 언어와 같고, 표본 수·split·seed는 v1.4 동결 계약을 따른다. 활성 데이터는 `data/language_v1_4/rebuild_01/`이다. 최초 root의 거부된 생성물은 학습 입력이 아니다. 구현은 `corpus/`, 감사 증빙은 `experiment_v1_4/P1_STATUS.md`에 있다.

## 2. 어휘와 토큰 ID

자연어 tokenizer나 BPE 없이 다음 15개 토큰을 정수 ID로 매핑한다.

| ID | 토큰 | 역할 |
|---:|---|---|
| 0 | PAD | batching 전용 |
| 1 | BOS | 시퀀스 시작 |
| 2 | EOS | 시퀀스 끝 |
| 3 | SET | 값 대입 |
| 4 | NOT | 반전 |
| 5 | AND | 논리곱 갱신 |
| 6 | OR | 논리합 갱신 |
| 7 | XOR | 배타적 논리합 갱신 |
| 8 | READ | 읽기 |
| 9 | A | 변수 |
| 10 | B | 변수 |
| 11 | C | 변수 |
| 12 | D | 변수 |
| 13 | B0 | Boolean 0 |
| 14 | B1 | Boolean 1 |

B0/B1은 SET 상수와 READ 정답에서 같은 토큰이다. 평가용 정답 위치는 parser의 role metadata로 구분한다. LM read4 loss는 metadata 대신 `READ → 변수 → B0/B1` 토큰 패턴으로 정답 prediction 위치를 판별한다. 공백·줄바꿈·화살표·등호는 토큰이 아니다.

## 3. 문법

```ebnf
sequence = BOS, initialization, block{8..24}, EOS ;
initialization = SET A bit, SET B bit, SET C bit, SET D bit ;
block = update{1..3}, READ var, bit ;
update = SET var bit | NOT var | binary_op var var ;
binary_op = AND | OR | XOR ;
var = A | B | C | D ;
bit = B0 | B1 ;
```

이는 설명용 EBNF다. READ 뒤 bit는 실제 실행 결과와 일치해야 한다.

- 이항 갱신의 첫 변수는 목적변수 dst, 둘째는 원본변수 src이며 dst ≠ src다.
- 초기화 순서는 A, B, C, D로 고정하고 초기값은 독립 Bernoulli(0.5)로 뽑는다.
- 명령을 왼쪽부터 실행한다. READ는 상태를 바꾸지 않는다.
- 초기화는 일반 갱신 수, 블록 길이, 연산 패턴 holdout에 포함하지 않는다.
- 시퀀스를 중간에 잘라 사용하지 않는다. 길이 초과 시 전체를 다시 생성한다.

## 4. 실행 의미

상태 s는 각 변수에 0 또는 1을 대응시킨다. 양쪽 입력은 갱신 전 상태에서 읽고, 목적변수 이외의 값은 유지한다.

```text
SET d b:   s'[d] = b
NOT d:     s'[d] = 1 - s[d]
AND d r:   s'[d] = s[d] & s[r]
OR d r:    s'[d] = s[d] | s[r]
XOR d r:   s'[d] = s[d] ^ s[r]
READ q:    answer = s[q], state unchanged
```

| dst_before | src_before | AND | OR | XOR |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

검산 예제:

```text
BOS SET A B1 SET B B0 SET C B1 SET D B0
XOR A C READ A B0
NOT B READ B B1
OR A B READ A B1
AND A C READ C B1 EOS
```

갱신 후 상태는 차례로 (0,0,1,0), (0,1,1,0), (1,1,1,0), (1,1,1,0)이다. 마지막 READ C는 직전 갱신 대상 A와 다르다. 이 짧은 예제는 설명용이며 실제 학습은 최소 8블록이다.

## 5. 기본 생성 분포

| 항목 | v1.4 IID 분포 |
|---|---|
| 초기 bit | 변수마다 독립 Bernoulli(0.5) |
| 블록 수 T | 8..24 정수 균등 |
| 블록 내 갱신 수 g | P(1)=0.80, P(2)=0.15, P(3)=0.05 |
| 갱신 종류 | SET:0.25, NOT:0.25, AND/OR/XOR:각 1/6 |
| dst | 네 변수에서 균등 |
| src | dst 이외 세 변수에서 균등 |
| SET 상수 | Bernoulli(0.5) |
| READ 변수 | 확률 0.5로 마지막 dst, 나머지 0.5로 다른 세 변수에서 균등 |

48개 개별 명령은 균등 분포가 아니다. 명령 종류별 노출을 확보하는 위의 계층적 분포를 사용한다. 명령과 READ 변수는 현재 값이나 정답을 보고 선택하지 않는다. 학습 데이터에서 정답 균형을 강제하는 rejection은 하지 않는다.

평균 갱신 토큰 수는 2.75, 블록당 평균 갱신 수는 1.25다. 블록 기대 길이는 2.75×1.25+3=6.4375토큰이며, 초기화를 제외한 정답 위치율은 약 15.5%다. 초기화와 holdout rejection을 포함한 실제 비율은 별도 집계한다. 이를 gradient 기여율로 해석하지 않는다.

최장 학습 길이는 BOS(1)+초기화(12)+24×최대 블록 길이(12)+EOS(1)=302토큰이다.

선택적 길이 test는 블록 수만 33..48로 변경한다. 최장 590토큰으로 context capacity 768 이내다. 길이 외삽은 필수 통과 조건이 아니다.

## 6. 코퍼스 크기와 의미적 다양성

갱신 명령은 SET 8개, NOT 4개, 이항 갱신 36개로 총 48개다. 갱신 하나와 READ 하나만 반복하는 제한된 경우에도 T블록 프로그램의 선택 수는 16×192^T다.

전체 상태는 16개뿐이다. 많은 문자열을 많은 독립 개념과 동일시하지 않는다. 상태, 연산 입력, 변수, 의존 깊이, 문맥 다양성을 별도 기록한다. 같은 상태 전이의 반복 경험은 허용하며, 정확히 같은 전체 시퀀스의 중복과 구분한다.

## 7. 분할, seed 및 중복 방지

v1.4 fresh split과 train 확장은 namespace 20260920을 사용한다. 다음 문자열의 SHA-256 앞 8bytes를 unsigned big-endian 정수로 읽어 목적별 seed를 만든다.

```text
20260920|language-v1.4|<purpose>|<index>
```

Purpose/index의 실제 목록은 활성 manifest와 `corpus/v1_4.py`, `scripts/generate_v1_4_corpus.py`의 기록을 따른다. Select/gate/test 및 first/repeat를 구분하며 Python 내장 hash는 쓰지 않는다. 계승 train prefix는 새 seed로 다시 생성하지 않고 아래 원본 bytes와 provenance를 보존한다.

RNG는 numpy.random.Generator(PCG64)로 고정하고 버전을 기록한다. 모든 LM seed에 같은 학습 데이터 순서를 사용해 모델 초기화 효과를 우선 비교한다.

1. Fresh select/gate/test → 해석 pool → 인과 pair를 먼저 예약하고 train을 확장한다.
2. PAD 없는 token ID를 쉼표로 연결한 ASCII를 canonical 형식으로 삼아 SHA-256을 저장한다.
3. 새 sequence/READ·causal prefix는 v1.0–v1.3의 inherited registry와 v1.4 예약 집합에 대해 충돌을 거부한다. LM train 내부 중복도 거부한다. 기존 train prefix 복사는 의도된 계승이며 fresh split 재사용과 구분한다.
4. Split을 나중에 추가하면 기존 train을 포함한 전체 hash와 비교한다.

시퀀스의 모든 위치는 같은 split에 둔다. 인과 pair와 같은 origin의 파생 예제도 같은 split에 둔다. 중복 확인을 위한 hash 사용은 허용하지만 test 점수를 checkpoint 선택에 사용하지 않는다.

v1.3 train 68 shards / 271,936 sequences / 32,004,917 tokens를 byte-identical하게 계승한다.
IID 분포에서 확장해 최초 완전 64-sequence update로 64M을 넘길 때 종료한다.
활성 corpus는 133 shards / 537,536 sequences / 64,005,751 prediction tokens다.
Shuffle·epoch 반복은 없다. Historical READ-prefix 누락을 수정한 rebuild_01만 사용하며
원본 manifest/config/hash는 [동결 데이터 계약](experiment_v1_4/configs/data.json)을 따른다.

## 8. 합성 holdout

사전 고정 제외 패턴:

```text
(XOR, AND, OR)
(OR, XOR, AND)
```

같은 블록 안의 연속 3갱신 operator 열이다. READ를 가로지르는 패턴과 초기화는 대상이 아니다.

- LM train, 일반 validation/test, 해석 train/validation/test에서 두 패턴을 포함하는 시퀀스를 거부한다.
- 합성 test는 임의 블록 하나를 지정 패턴의 3갱신 블록으로 대체한다.
- 그 블록의 dst를 세 번 같게 하고 src는 각각 dst 이외에서 선택한다.
- 바로 이어 같은 dst를 READ하여 그 답만 지정 target으로 평가한다.
- 다른 블록에는 holdout을 허용하지 않는다.
- 각 단일 연산과 인접 2연산 패턴이 train에 실제 존재하는지 확인한다.

AND/OR이 앞선 정보를 지울 수 있으므로 구조적 연쇄가 곧 모든 중간 계산의 인과적 중요성을 뜻하지 않는다. 블록 시작 dst의 bit-flip 영향이 남는 subset을 전체 holdout 점수와 별도로 보고한다.

한정된 연산열 holdout이며 임의 프로그램 합성이나 미지 진리함수에 대한 일반화로 주장하지 않는다.

## 9. 생성 알고리즘

아래는 의사코드다. Append 함수는 tokens와 role metadata를 함께 기록한다. 모든 난수 함수는 명시적 rng를 사용한다.

```python
def generate_candidate(rng, config):
    state = {v: sample_bit(rng) for v in VARIABLES}
    initial_state = state.copy()
    tokens, events = [BOS], []
    dependencies = initialize_dependency_nodes(initial_state)
    for v in VARIABLES:
        append_initialization(tokens, SET, v, state[v])

    n_blocks = randint_inclusive(rng, config.min_blocks, config.max_blocks)
    for block_id in range(n_blocks):
        n_updates = categorical(rng, {1: .80, 2: .15, 3: .05})
        for _ in range(n_updates):
            op = sample_operator(rng)
            dst = choice(rng, VARIABLES)
            src = choice(rng, variables_except(dst)) if is_binary(op) else None
            literal = sample_bit(rng) if op == SET else None
            before = state.copy()
            end_index = append_update(tokens, op, dst, src, literal)
            state = execute_update(before, op, dst, src, literal)
            dep = update_dependencies(dependencies, op, dst, src)
            events.append(update_metadata(before, state, end_index, dep))
        q = choose_read_variable(rng, last_dst=dst, p_same=.5)
        query_index = append_read(tokens, q)  # variable token index
        answer_index = append_answer(tokens, state[q])
        events.append(read_metadata(state, q, query_index, answer_index))
    tokens.append(EOS)
    return tokens, events, initial_state

def generate_accepted(rng, split, reserved_hashes):
    for attempt in range(MAX_ATTEMPTS):
        example = generate_candidate_for_split(rng, split)
        if violates_pattern_policy(example, split):
            continue
        if not meets_split_constraints(example, split):
            continue
        if canonical_hash(example.tokens) in reserved_hashes:
            continue
        validate_by_independent_replay(example)
        reserved_hashes.add(canonical_hash(example.tokens))
        return example
    raise GenerationError("Constraints infeasible or rejection rate too high")
```

일반 split은 기본 생성, 합성 split은 8절의 패턴 주입, 진단 split은 12절의 target 선택, 인과 split은 13절의 변형을 적용한다. 진단 제약을 학습 분포에 섞지 않는다.

MAX_ATTEMPTS는 target당 100,000을 상한으로 두고 수락률을 기록한다. 상한 도달 시 중단하며 제약을 몰래 완화하지 않는다. 낮은 수락률은 CPU 단계에서 조건을 직접 만족하는 생성기로 수정할 수 있으나 분포와 버전 변경을 기록한다.

## 10. Metadata 규격

시퀀스별:

```text
schema_version, sequence_id, split, rng_seed, token_ids, token_roles,
initial_state, blocks, update_events, read_events, canonical_hash,
target_read_ids, holdout_pattern_id
```

갱신별:

```text
update_id, block_id, op, dst, src_or_null, literal_or_null,
start_token_index, end_token_index, state_before, state_after,
dst_before, src_before_or_null, dst_after,
input_truth_pattern_or_null, dependency_node_id, structural_depth
```

READ별:

```text
read_id, block_id, query_var, query_token_index, answer_token_index,
answer, state_at_read, previous_value_or_null,
last_update_id_for_query_var_or_null,
updates_since_last_update, tokens_since_last_update,
reads_of_query_var_since_last_update,
reads_of_query_var_since_latest_set,
same_as_last_dst, structural_depth, local_sensitivity, answer_is_target
```

인덱스는 0-based다. query_token_index는 READ 다음 변수 토큰이며 end_token_index는 갱신의 마지막 operand다. READ 횟수는 현재 READ를 제외한 과거 횟수다.

previous_value는 직전 일반 갱신 직전 값이고, 초기화만 있으면 null이다. 초기화는 latest SET으로는 취급하되 일반 갱신 횟수에서는 제외한다. 일반 갱신이 없으면 거리 측정은 초기화 끝을 기준으로 하고 last_update_id는 null로 둔다. 상태 snapshot은 복사해 저장하여 이후 갱신으로 과거 라벨이 바뀌지 않게 한다.

## 11. 구조 의존성과 국소 감도

각 변수의 현재 값을 만든 node를 추적한다.

- 초기화/SET: 부모 없음, depth=0.
- NOT: 갱신 전 dst node가 부모, depth=1+parent.depth.
- 이항 갱신: 갱신 전 dst/src node가 부모, depth=1+max(parent.depth).
- READ: 해당 변수의 현재 node 참조.

SET은 이전 구조 의존을 끊는다. READ/답은 실행 의미상의 node를 추가하지 않는다. 모델은 이전 답을 사용할 수 있으므로 READ 횟수는 별도 기록한다.

국소 bit-flip sensitivity는 target 블록 시작 상태의 A/B/C/D를 하나씩 반전하고 고정된 후속 명령을 재실행해 계산한다. 답이 바뀌면 해당 bit 감도는 1이다. 이는 입력별 국소 감도이며 최소 회로 깊이나 완전한 함수 의존성이 아니다. AND의 00처럼 단독 반전은 무효여도 공동 변경이 유효할 수 있다.

Train에서는 저렴한 구조 metadata만 저장해도 된다. 국소 감도는 진단/해석 target에서 계산하고 미계산 null과 실제 0을 구분한다.

## 12. 진단 데이터

한 시퀀스의 조건 만족 READ 중 하나를 균등 선택해 지정 target으로 삼고, 없으면 시퀀스를 거부한다. 지정 target만 채점하여 긴 시퀀스가 더 큰 가중치를 받지 않게 한다.

1. **다른 변수 읽기:** query_var가 직전 갱신 dst와 다름.
2. **반복 갱신:** 해당 변수에 초기화 외 갱신이 두 번 이상. 현재/직전 값이 다른 subset도 저장.
3. **SET 이후 첫 읽기:** 최신 SET 또는 초기화 이후 해당 변수를 처음 READ. SET 이후 논리 갱신이 있는 합성 subset도 저장.
4. **방해 거리:** 최신 관련 갱신 이후 다른 변수 갱신 수를 0, 1~2, 3 이상으로 구분.
5. **진리표:** 직전 갱신이 이항이고 query_var=dst. operator × dst_before × src_before의 12 strata.

조건 중복을 허용하고 명시한다. 균형 평가에서는 target 답이나 진리표 strata에 quota를 적용한다. 일반 test와 균형 진단을 합쳐 단일 점수로 만들지 않는다. 일반 split의 실제 다수 클래스 기준선을 계산하며, 균형 이진 진단의 추측 기준선만 50%다.

### 12.1 v1.4 first/repeat 42-cell 진단

First는 query 변수의 마지막 논리 갱신 뒤 첫 READ다. NOT 입력 0/1 및
AND/OR/XOR 입력 00/01/10/11에 depth bin 1, 2–3, 4+를 곱해 42 cells를 만든다.
Repeat는 같은 origin 의미를 유지하면서 해당 갱신과 target 사이에 올바른 동일 변수 READ를
하나 삽입한 쌍이다. 문법·길이·정답·origin 독립성을 검증한다.
이 정의는 legacy의 “최신 SET 이후 첫 읽기”와 다르다.

Select와 gate는 각각 cell당 64 independent origin pairs, test는 128 pairs다.
General은 select/gate/test 각 512/1,024/2,048 sequences이며 legacy 세 진단은
gate/test 각각 조건당 1,024/2,048 targets다. Select만 checkpoint 선택에 사용한다.
First/repeat 한 쌍을 독립 표본 두 개로 세지 않으며 quota와 coverage는 100%여야 한다.
First/repeat origin·member·cell·target 식별자는 활성 metadata와 manifest에 보존한다.

## 13. 반사실적 pair

1. 일반 예제의 target READ를 선택한다.
2. target 이전 초기화/SET bit 하나를 반전한다.
3. 같은 operator·operand·READ 열을 재실행하여 모든 답을 계산한다.
4. target 이전 답이 하나라도 달라지면 기본 pair에서 제외한다.
5. target 답이 달라지면 changed-target pair로 채택한다.
6. target 뒤 suffix는 사용하지 않고 답 직전 prefix와 정답을 분리한다.
7. 동일한 과거 수정이 다른 변수의 target 답을 바꾸지 않는 예제를 실행기로 검증해 unchanged-target 대조군을 만든다.

변경 SET 이후 해당 변수의 논리 갱신이 없으면 기억 조건, 하나 이상이면 합성 조건으로 기록한다. 효과는 반드시 replay로 확인한다.

변경/target token index, 원본/대조 상태, prefix 차이를 저장한다. 모델 입력 prefix는 의도한 SET bit 한 토큰만 달라야 한다. Target 정답 차이는 prefix에 포함하지 않는다. 같은 origin 파생 pair는 같은 split에 두며 ID에 origin hash와 변경 위치를 포함한다.

## 14. LM 입출력과 누출 방지

```python
input_ids = token_ids[:-1]
labels = token_ids[1:]
weight[t] = 4 if token_only_read_answer(input_ids, labels, t) else 1
weight[t] = 0 if labels[t] == PAD else weight[t]
loss = sum(weight[t] * cross_entropy(logits[t], labels[t])) / sum(weight)
```

원본 위치 j의 정답을 예측하는 logit/activation은 j−1, 즉 READ 직후 변수 위치다.

- 정답 입력 이후 activation을 정답의 사전 표현으로 분석하지 않는다.
- 상태 라벨은 실행 도중 snapshot으로 만들고 미래 상태를 참조하지 않는다.
- 상태·holdout 라벨·answer mask를 embedding 입력에 넣지 않는다.
- Metadata answer mask는 평가 집계에만 사용한다. Read4 loss의 위치 mask는 token IDs에서 독립적으로 계산한다.
- READ 정답 prediction weight는 4, 기타 non-PAD 명령·상수·READ·EOS는 1이다. BOS 자체는 예측 target이 아니다. 상태 라벨이나 정답 metadata를 loss에 추가하지 않는다.
- Packing 시 예제 사이 attention을 차단한다. v1.4는 packing 없이 독립 시퀀스의 우측 padding batch를 사용한다.

## 15. 필수 CPU 검증

1. 전체 16상태×48갱신에서 목적변수 및 비대상 변수의 의미를 확인한다.
2. 이항 3연산×4입력, NOT 2입력, SET 2값을 독립 진리표와 대조한다.
3. 생성기와 별도 replay parser의 모든 READ 정답이 일치해야 한다.
4. NOT 두 번, src 불변 조건의 XOR d r 두 번이 상태를 복원해야 한다.
5. Parse→serialize roundtrip과 role·index·answer mask 정합성을 확인한다.
6. 고정 seed hash 재현성과 split 간 exact 중복 부재를 확인한다.
7. 제외 split의 holdout 부재 및 합성 지정 블록의 패턴 존재를 확인한다.
8. 인과 pair prefix 차이 1토큰, target 답 변화, unchanged 대조를 확인한다.
9. 16상태, 48명령, 진리표 12 strata coverage와 누락을 보고한다.

유한 표본에서 샘플링 비율의 완전한 일치를 요구하지 않는다. 정답이 50%가 아니라는 이유만으로 생성기 오류로 판정하지 않는다.

## 16. 저장 형식과 manifest

```text
data/
  language_v1_4/rebuild_01/
    manifest.json
    vocab.json
    select/
    gate/
    test/
    diagnostics/
    composition/
    interpretation/
    causal_pairs/
    corpus_statistics.json
    reserved_hashes.txt
```

Manifest에는 schema/version, 토큰 표, 분포, holdout 정의, master/파생 seed, RNG 구현과 버전, 코드 commit/hash, 생성일시, 수락/거부 수와 사유, split 예제 수, 실제 토큰·답 수, 파일별 SHA-256을 저장한다.

CPU 사전 생성은 seed·shard·생성 예제 수·RNG state와 hash를 기록한다. LM은 동결 shard cursor와 완전 update 경계로 재개하여 중복·누락을 검출한다. 위 tree는 역할별 개념도이며 실제 파일 경로는 활성 manifest를 따른다.

Corpus 통계는 길이, 명령, 변수별 0/1, READ 정답 비율, 진리표 coverage, 구조 깊이, 첫 READ 비율, 정답 위치율을 포함한다. 이 통계와 CPU 검증을 확인한 뒤 LM 학습을 시작한다.
