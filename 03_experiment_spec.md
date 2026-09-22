# Boolean 프로그램 transformer 해석 실험 상세 명세

작성일: 2026-09-09
규격 버전: experiment-spec-v1.4-integrated-r1
상태: 2026-09-22 본 실험 통합 명세. P1–P4 완료; P5–P11은 실행·검증이 남은 규격이다.
상위 설계: [01_experiment_design.md](./01_experiment_design.md)
언어·생성 규칙: [02_language_and_corpus.md](./02_language_and_corpus.md)

## 1. 범위와 문서 간 우선순위

목표는 Boolean 프로그램을 next-token prediction으로 학습한 작은 transformer에서 상태 정보의 선형 접근성, SAE feature의 분리도, Transcoder feature의 MLP 계산 설명력과 인과 효과를 비교하는 것이다. LM 행동 검증을 먼저 통과시킨 뒤 동결된 모델을 해석한다. 해석 결과가 나쁘다는 이유로 LM을 다시 선택하지 않는다.

현재 루트 01–03은 v1.4 본 실험의 통합 규격이다. 우선순위는 최신 사용자 지시 →
v1.4 동결 실행 계약·명시적 변경 → 이 문서 → 02 → 01이다. 실행 순서와 gate 증빙은
[phase.md](phase.md)를 따른다. 불일치는 원문·항목을 기록하고 해결하며 사후 결과로 절충하지 않는다.

[설계 계약](experiment_v1_4/design_config.json), [동결 config set](experiment_v1_4/configs/config_set_manifest.json),
[corpus 수정 계약](experiment_v1_4/corpus_rebuild.json), [보고 계약](experiment_v1_4/results/p3_preparation_r3/REPORTING_CONTRACT.md)을 통합했다.
설계 당시 DESIGN/README의 “미실행”은 역사적 상태이며 현재 상태가 아니다.
원문과 hash는 보존하고 [통합 전 문서](archive/specifications/pre_v1_4_integration/README.md)에서 확인한다.

Transcoder는 필수 READ 분석이며 SAE 다음에 수행한다. LM은 CPU 사전 생성 shard를 같은
순서로 한 번씩 소비한다. GPU에서 코퍼스를 새로 생성하지 않는다.

기본 필수 범위는 행동 gate를 통과한 각 LM seed의 첫 block READ residual SAE, 첫 block READ MLP Transcoder, 각각의 probe·재구성/대체·인과 평가다. update 분석과 둘째 block 분석은 명세는 제공하되 2차 분석으로 구분한다. 실패한 행동 gate, 생성 제약 미충족, 자원 중단은 실패·미완료로 보고하며 성공으로 간주하지 않는다.

## 2. 공통 표기와 분석 지점

수식은 열벡터 기준이다. 실제 batch tensor의 마지막 축은 feature 차원이다. Layer index는 **0..11**로 표기한다. 문장의 ‘첫 층’은 block 0이다.

| 기호 | 의미 | 단일 위치 shape |
|---|---|---|
| `r_l` | block l 입력 residual | 256 |
| `a_l` | attention 출력, output projection 이후 | 256 |
| `r_mid_l` | `r_l + a_l` | 256 |
| `u_l` | `LN_mlp(r_mid_l)`, 실제 MLP 입력 | 256 |
| `g_l` | MLP GELU hidden activation | 1024 |
| `m_l` | MLP output projection 및 bias 이후, residual 덧셈 이전 | 256 |
| `h_l` | block 출력 `r_mid_l + m_l` | 256 |
| `z` | SAE 또는 Transcoder sparse latent | 512 |

Hook 이름을 `blocks.{l}.resid_post`, `blocks.{l}.mlp_in`, `blocks.{l}.mlp_out`으로 고정하여 각각 `h_l`, `u_l`, `m_l`을 반환한다. 구현이 모듈 hook 대신 명시적 반환을 사용해도 이 의미와 저장 이름은 유지한다.

- READ: `READ A B1` 중 **A 토큰 처리 직후**의 위치 t. 정답 B1은 t+1에 있으며 `logits[t]`로 예측한다.
- Update: `XOR A C`의 C, `SET A B1`의 B1, `NOT A`의 A 처리 직후. 초기화 SET은 update pool에서 제외한다.
- 모든 기본 분석은 정상 정답 prefix를 제공한 teacher forcing이다. 정답 토큰 이후 activation을 해당 정답의 사전 표현으로 사용하지 않는다.
- READ/update마다 별도 SAE·Transcoder를 학습한다. 서로 다른 layer도 별도 모델·전처리를 사용한다.

## 3. 실행 환경과 재현성

### 3.1 작업 분리

| 작업 | 실행 위치 | 계산 장치 | 주요 저장물 |
|---|---|---|---|
| 실행기·생성기·독립 replay·분할 검증 | 로컬 CPU 또는 Colab CPU 런타임 | CPU만 사용 | JSONL, token shard, hash, manifest |
| Transformer 학습 및 행동 평가 | Colab Pro, Python GPU 런타임 | 단일 NVIDIA GPU | checkpoint, 학습·평가 로그 |
| Activation 추출, SAE·Transcoder 학습, 개입 | 같은 Colab Pro 환경 | GPU | float32 cache, dictionary checkpoint |
| Linear probe, bootstrap, 표·그림 | Colab CPU 또는 로컬 CPU | CPU | coefficients, CSV/JSON, 그림 |

Colab Pro의 GPU 종류·메모리·연속 사용 시간은 고정 보장이 아니다. 따라서 특정 A100 제공이나 완료 시간을 전제하지 않는다. 시작 시 실제 GPU 모델·VRAM·RAM·가용 저장 공간을 기록하고, 아래 microbatch 조정 규칙으로 단일 GPU에 맞춘다. 서비스 제약의 근거는 [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)다.

### 3.2 소프트웨어 고정 절차

구현은 Python, PyTorch, NumPy, SciPy, pandas, matplotlib, pytest를 사용한다. Hugging Face 사전학습 모델·자연어 tokenizer는 사용하지 않고 15-token transformer를 직접 구현한다. Probe는 SciPy의 최적화기로 목적함수를 직접 최소화한다.

1. 첫 CPU/GPU smoke test를 통과한 환경에서 `python --version`, `pip freeze`, PyTorch/CUDA/cuDNN 버전, `nvidia-smi` 출력을 저장한다.
2. 실제 설치된 정확한 버전으로 `requirements-cpu.lock.txt`, `requirements-colab.lock.txt`를 생성한다. 본 명세에 확인하지 않은 버전 번호를 임의로 쓰지 않는다.
3. 이후 seed와 재개 실행은 같은 lock을 사용한다. Colab 기본 이미지 변경으로 동일 환경 재현이 불가능하면 환경 ID를 새로 부여하고 결과를 구분한다.
4. 모델 및 dictionary 연산은 **float32**, AMP·TF32·`torch.compile`은 끈다. 작은 모델에서 정밀도 변경이라는 변수를 줄이기 위한 선택이다.
5. Python/NumPy/PyTorch/CUDA RNG를 명시적으로 설정한다. PyTorch deterministic algorithms를 켜고 cuDNN benchmark는 끈다. 비결정적 연산 오류가 나면 해당 연산을 결정적 구현으로 수정하고 smoke test를 다시 한다. 다른 GPU/버전 사이 bitwise 동일성을 보장한다고 쓰지 않는다.

### 3.3 저장과 재개

Colab에서는 `/content/boolean_interp/`를 작업 디렉터리로 사용한다. 데이터는 실행 전 로컬 디스크로 복사하고 체크섬을 검증한다. Google Drive 등의 영속 저장소에는 checkpoint와 결과를 주기적으로 복사한다. 매 batch를 Drive에서 읽고 쓰지 않는다.

Checkpoint에는 모델, optimizer, RNG state, update 번호, 누적 비패딩 예측 토큰, 다음 데이터 cursor, 다음 평가 경계, best validation 값, config/hash를 포함한다. SAE/Transcoder는 sampler RNG·전처리·decoder norm도 포함한다. LM은 init, 모든 지정 milestone, 15분 경과 후 다음 완전 update 경계, last를 영속 저장한다. Dictionary는 매 250 updates에 저장한다. 임시 파일 기록 후 rename하고 복사 완료 checksum을 확인한다.

복구 시 마지막으로 완전히 저장된 update부터 재개한다. 미저장 update의 재연산은 결과 토큰 예산에 중복 가산하지 않되, 실제 소요 시간에는 포함한다. 서로 다른 환경에서 재개한 사실을 manifest에 남긴다.

## 4. Transformer 상세 구조

### 4.1 고정 아키텍처

| 항목 | 값 |
|---|---|
| 형태 | 12-block causal decoder-only, attention → MLP 순차 residual |
| Vocabulary / embedding | 15 / 256, embedding scaling 없음 |
| Residual dimension | 256 |
| Attention | 4 heads, head dimension 64, Q/K/V 각각 총 256차원 |
| QKV projection | `Linear(256, 768, bias=True)` |
| Attention output | `Linear(256, 256, bias=True)` |
| Attention score | `QKᵀ / sqrt(64)` 후 causal masked softmax |
| MLP | `Linear(256,1024,bias=True)` → exact GELU → `Linear(1024,256,bias=True)` |
| LayerNorm | block당 attention 전·MLP 전 각각 1개, affine=True, eps=1e-5 |
| Final LayerNorm | 256차원, affine=True, eps=1e-5 |
| Unembedding | `Linear(256,15,bias=False)`, token embedding과 **가중치 공유 안 함** |
| 위치 표현 | RoPE, base=10,000, Q/K의 64차원 전체에 적용, V에는 미적용 |
| Context | 768, 학습 시 최대 실제 sequence 길이 302 |
| Dropout | embedding/attention/residual/MLP 모두 0 |
| 추가 구조 | learned positional embedding, GQA, gated MLP, KV cache, packing 사용 안 함 |

RoPE는 head 벡터의 `(0,1), (2,3), …, (62,63)`을 쌍으로 회전한다. 쌍 i=0..31의 각도는 `position × 10000^(-2i/64)`다. 각 시퀀스 BOS의 위치는 0이며 padding 위치의 출력은 평가하지 않는다. 학습·길이 평가에서 base나 위치 스케일을 변경하지 않는다. 위치 회전의 원리는 [RoFormer](https://arxiv.org/abs/2104.09864)를 따른다.

```text
tokens → Embedding(15,256) → r_0
각 l=0..11:
  v_l     = LN_attn_l(r_l)
  Q,K,V   = split(W_qkv_l v_l + b_qkv_l)
  a_l     = W_o_l Concat(CausalAttention(RoPE(Q),RoPE(K),V)) + b_o_l
  r_mid_l = r_l + a_l
  u_l     = LN_mlp_l(r_mid_l)
  g_l     = GELU(W_in_l u_l + b_in_l)
  m_l     = W_out_l g_l + b_out_l
  h_l     = r_mid_l + m_l
  r_(l+1) = h_l
logits = W_U LN_final(h_11)  # [batch, input_length, 15]
```

PAD도 15-class softmax의 한 출력이다. BOS/PAD logit을 인위적으로 제거하지 않는다. PAD 입력 embedding도 일반 학습 파라미터로 세되 유효 토큰에서 PAD key를 보지 못하게 한다. PAD **target**만 loss에서 제외한다.

### 4.2 파라미터 수

| 구성 | 산식 | 개수 |
|---|---|---:|
| Embedding | 15×256 | 3,840 |
| QKV / block | 256×768 + 768 | 197,376 |
| Attention output / block | 256×256 + 256 | 65,792 |
| MLP input / block | 256×1024 + 1024 | 263,168 |
| MLP output / block | 1024×256 + 256 | 262,400 |
| 두 LayerNorm / block | 2×(256+256) | 1,024 |
| Block 소계 | 위 block 항목 합계 | 789,760 |
| 12 blocks | 12×789,760 | 9,477,120 |
| Final LayerNorm | 256+256 | 512 |
| Unembedding | 256×15 | 3,840 |
| **총 학습 파라미터** | 3,840+9,477,120+512+3,840 | **9,485,312** |

RoPE와 mask는 학습 파라미터가 없다.
`sum(p.numel() for p in model.parameters()) == 9485312`를 검증한다.

### 4.3 초기화·학습 하이퍼파라미터

Embedding과 모든 Linear weight는 독립 `Normal(0,0.02²)`로 초기화한다. 각 block의 attention output 및 MLP output weight에는 추가로 `1/sqrt(2×12)=1/sqrt(24)`를 곱한다. 모든 Linear bias는 0, LayerNorm weight는 1, bias는 0이다. LM seed는 0, 1, 2다. 학습 전 probe 기준선용으로 초기 상태도 저장한다.

| 항목 | 명세 |
|---|---|
| Loss | token-only read4: READ 답 weight 4, 기타 non-PAD weight 1, label smoothing=0 |
| Optimizer | AdamW, lr=3e-4, betas=(0.9,0.95), eps=1e-8, amsgrad=False |
| Weight decay | 0.01, embedding과 matrix weight에만 적용; 모든 bias/LN 제외 |
| Batch | effective 64 sequences; GPU smoke 확정 microbatch=16, accumulation=4 |
| Smoke 메모리 조정 | 최초 GPU smoke에서 16→8→4→2→1 허용; 본학습 재현은 확정값 16 유지 |
| Clipping | accumulation 완료 후 global gradient norm=1.0 |
| LR | 첫 50k linear warmup, 57.6M까지 3e-4, 64M까지 3e-5로 linear decay |
| 학습 예산 | seed당 명목 64M 비패딩 prediction tokens, 최초 완전 update overshoot 포함 |
| 자동 연장 | 허용하지 않음 |
| Select 평가·후보 | 1M, 3M, 8M, 16M, 24M, 32M, 48M, 57.6M, 60.8M, 64M을 처음 넘는 완전 update |
| Checkpoint 선택 | select first-member 42-cell macro answer CE 전역 최소 및 §4.4 near-tie 규칙 |

길이가 다른 microbatch의 평균 loss를 단순 평균하지 않는다. Effective batch 전체의
가중치 합 W를 분모로 각 microbatch의 `sum(weight * CE)/W`를 backward한다.
READ 답 위치는 `READ → 변수 → B0/B1` 토큰 패턴만으로 검출한다. Metadata는 사용하지 않는다.

예산의 prediction token 수 N은 가중치 합과 다르며 각 sequence의 `len(tokens)-1` 합이다.
LR은 update 이후 누적 prediction tokens T로 계산한다.

```text
T <= 50,000:      lr = 3e-4 * T / 50,000
T <= 57,600,000:  lr = 3e-4
그 이후:          lr = 3e-4 + min(1, (T-57,600,000)/6,400,000) * (3e-5-3e-4)
```

Update를 쪼개지 않는다. 실제 seed당 64,005,751 tokens, overshoot 5,751,
8,399 updates, cursor 537,536을 소비했다.

Batch 내 우측 padding, causal mask와 key padding mask를 함께 사용한다. 유효 query t가 t보다 큰 key를 보지 못하고 PAD key를 보지 못함을 직접 검사한다. 구현 초기에는 명시적 matmul·mask·softmax를 사용한다. 빠른 attention kernel을 도입하면 별도 버전에서 동등성을 검증한다.

### 4.4 Checkpoint 선택과 one-time 행동 gate

10개 milestone 후보에서 select first-member 42-cell macro answer CE의 **전역 최소**를 찾는다.
그 최소와 1e-4 nats 이내 후보를 select/general answer CE, 이른 update 순으로 정렬한다.
Gate는 선택된 checkpoint에서 seed당 한 번만 열고 다른 checkpoint로 재선택하지 않는다.

| Gate 항목 | 통과 조건 |
|---|---|
| General READ | 15-token full-vocabulary accuracy ≥99% |
| Legacy diagnostic 세 종류 | 조건당 1,024 independent targets, 각각 accuracy ≥95% |
| First | 42-cell macro accuracy ≥95% |
| First group | 연산별·depth-bin별 macro 및 answer별 accuracy 각각 ≥95% |
| Repeat | 42-cell macro accuracy ≥95% |
| Quota·coverage | 100% |

개별 42 cell은 count와 CI를 보고하며 추가 gate로 쓰지 않는다.
Seed 0 실패 시 중단한다. 통과 후 동일 설정·데이터 순서·64M 예산으로 seed 1·2를
fresh initialization에서 실행하고 실패 seed도 보존한다. 최소 두 seed 통과가 해석 진입 조건이다.
전체 학습·validation 결정 동결 후 모든 학습 seed를 frozen test에서 한 번 평가한다.

현재 seed 0/1은 update 7,983, seed 2는 update 8,399가 선택됐고 세 seed 모두 통과했다.
[동결 모델 목록](experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json)과
[validation 동결](experiment_v1_4/results/p4_audit_20260921_01/validation_freeze.json)을 사용한다.
기존 gate/test 추론을 문서 정리나 표현 분석 준비를 위해 반복하지 않는다.

## 5. CPU 코퍼스 제작 규격

### 5.1 기본 분포와 크기

15개 토큰 ID, 4개 변수, SET/NOT/AND/OR/XOR/READ 실행 의미는 02를 그대로 사용한다. 기본 블록 수는 8..24 균등, 블록당 갱신 수 확률은 `(1:.80, 2:.15, 3:.05)`, operator 확률은 `(SET:.25, NOT:.25, AND:1/6, OR:1/6, XOR:1/6)`다. 값 라벨로 LM 데이터를 균형화하지 않는다.

| Split | 생성 목표 | 선택·평가 단위 |
|---|---:|---|
| Select/general / gate/general / test/general | 512 / 1,024 / 2,048 sequences | 모든 READ |
| Legacy diagnostic gate / test | 조건당 1,024 / 2,048 sequences | 시퀀스당 지정 READ 1개 |
| First/repeat select / gate / test | 42 cells × 64 / 64 / 128 pairs | independent origin별 first·repeat target |
| 거리 diagnostic test | 0, 1~2, ≥3별 1,024 sequences | 지정 READ 1개 |
| 진리표 diagnostic test | 3 operators×4 inputs별 256 sequences | 지정 READ 1개 |
| 답 균형 diagnostic test | B0/B1별 512 sequences | 지정 READ 1개 |
| Composition test | 두 holdout 패턴별 1,024 sequences | 주입 블록 직후 READ |
| Length test, 2차 분석 | 1,024 sequences, 33..48 blocks | 모든 READ |
| Interpretation train / val / test | READ 각 50k / 10k / 20k positions, update도 각각 동일 | 시퀀스 split 후 위치 추출 |
| Causal val | changed 512 + unchanged 512 pairs | origin/pair |
| Causal test | changed 1,024 + unchanged 1,024 pairs | origin/pair |
| LM train | 명목 64M, 실제 64,005,751 prediction tokens / 537,536 sequences | 동일 순서, 반복 없이 소비 |

길이 test는 초기 CPU 단계에서 예약·생성하되 GPU 평가는 2차 분석으로 둔다. 핵심 진단끼리 조건은 중첩될 수 있지만 파일 간 정확히 같은 sequence는 넣지 않는다. 진리표/답 균형 평가를 자연 분포 점수에 섞지 않는다.

### 5.2 생성·중복·샘플링 절차

1. `20260920|language-v1.4|<purpose>|<index>`와 02의 SHA-256 파생식을 사용하고 NumPy `Generator(PCG64)`를 고정한다. 목적 이름·diagnostic 조건·pair 유형·shard 번호는 활성 manifest와 생성 코드에 저장된 매핑을 따른다.
2. Fresh select/gate/test → interpretation → causal → train 확장 순서다. v1.0–v1.3 inherited sequence/READ-prefix registry와 v1.4 예약 집합을 모두 대조한다. v1.3 train prefix의 byte-identical 계승은 명시적 예외다. 해석용 시퀀스의 모든 READ/update는 같은 split에 속한다.
3. LM train, IID, diagnostic, interpretation, 기본 causal 데이터에서 두 holdout `(XOR,AND,OR)`, `(OR,XOR,AND)`가 같은 블록에 나타나는 시퀀스를 거부한다. Composition test만 지정 패턴을 허용한다.
4. Interpretation은 시퀀스를 순서대로 생성하여 READ/update 후보 수가 각각 quota 이상이 될 때 멈춘다. 별도 고정 RNG로 각 종류에서 quota만큼 균등 비복원 추출하고 `(sequence_id, token_index)` 순으로 저장한다. 라벨 값이나 모델 activation으로 위치를 선별하지 않는다.
5. Pair는 origin 전체 sequence hash와 실제 분석 prefix hash를 모두 등록한다. 같은 origin의 파생 데이터는 같은 causal split에 둔다. Full sequence가 달라도 다른 split과 prefix가 같으면 pair를 거부한다. 다른 데이터의 READ prefix hash도 예약하여 causal prefix의 정확 중복을 막는다. Pair 내부의 동일 토큰 문맥 공유는 의도된 예외다.
6. Train stream은 4,096 sequences 단위 shard로 저장한다. 계승한 과거 예산 경계의 짧은 shard는 유지하고 shard 경계를 넘어 effective batch를 이어 읽는다. Train 후보의 각 READ prefix도 예약 causal prefix와 비교하여 정확 중복이면 거부한다. 마지막 shard는 짧아도 되고, 소비할 마지막 batch는 64 sequences를 확보한다. 토큰 budget은 저장량이 아니라 실제 소비량으로 계산한다. 모든 LM seed가 같은 train shard를 읽는다.
7. 02의 target당 MAX_ATTEMPTS=100,000을 적용한다. 수락률·거부 사유·quota 부족을 저장하고, 상한에 도달하면 중단한다. 제약을 바꾼 데이터는 새 버전이다.

Canonical sequence hash는 PAD 없는 token ID의 쉼표 연결 ASCII에 대한 SHA-256이다. Prefix hash도 같은 직렬화 규칙을 사용한다. Hash 비교는 누출 방지 용도이며 test 점수는 보지 않는다.

### 5.3 CPU 검증과 납품물

GPU 작업 전에 다음 항목이 모두 통과해야 한다.

1. 16개 상태×48개 update=768개 전이를 독립 truth-table 구현과 대조한다.
2. Parse/serialize roundtrip, operand 제약, source 불변, NOT 두 번 및 source 불변 시 XOR 두 번의 복원을 확인한다.
3. **생성한 모든 sequence의 READ 정답**을 별도 parser/replay 실행기로 검산한다. 생성기 실행 함수를 검산기에 재사용하지 않는다.
4. 정답 index와 query index 관계, 이전 값의 null 처리, 초기화 제외, 상태 snapshot 복사를 확인한다.
5. Split hash 교집합, holdout 누출, train 개별 명령·인접 2연산 coverage를 검사한다.
6. Pair의 prefix 차이가 정확히 SET/초기화 bit 하나인지, 이전 답이 같은지, target 변화/불변 조건이 맞는지 검산한다.
7. 길이·정답 빈도·16상태·48명령·진리표 12칸·거리·깊이·변수별 값·quota 및 거부율 통계를 저장한다.
8. 작은 고정 seed 샘플을 두 번 생성하여 token 및 metadata hash가 같은지 확인한다.

활성 root의 `manifest.json`, `cpu_validation.json`, `corpus_statistics.json`, 파일별 SHA-256, RNG state, split 목록을 저장한다. 모델 입력용 token array와 정답·상태 metadata는 별도 파일로 관리하고 LM forward에는 token ID와 padding mask만 전달한다.

## 6. Activation cache와 라벨

### 6.1 추출

선택된 LM checkpoint를 `eval()` 및 gradient 비활성 상태로 실행한다. Interpretation split 전체 시퀀스는 teacher forcing으로 처리하되 5절에서 고정한 위치만 저장한다. 초기 microbatch는 16이며 메모리 부족 시 절반으로 줄인다. float32로 256차원 `h_0,u_0,m_0`를 수집한다. 동결 설계의 all-layer full probe 진단은 block 0..11에 포함한다. 전체 층 진단의 hook·저장 범위·실행 순서는 P5 config에 사전 고정하며, block 1 sparse dictionary·개입은 P10 선택 분석이다.

Cache key는 `(lm_seed, checkpoint_sha256, split, position_type, layer, sequence_id, token_index, hook)`다. 라벨 파일과 key join을 검증한다. Layer당 READ/update 합계 160k positions에서 세 256차원 tensor는 float32 원시 배열 기준 약 491.52 MB다. Metadata와 latent 저장 용량은 별도다. 512차원 latent는 필요한 실험만 순차 계산하며 모든 모델 결과를 GPU에 동시에 적재하지 않는다.

### 6.2 라벨과 적용 범위

| 위치 | 라벨 | 클래스/제외 규칙 |
|---|---|---|
| READ | 현재 읽는 값 | binary, 주 라벨 |
| READ | 직전 일반 갱신 전 값 | binary, 초기화만 있으면 null 제외 |
| READ | query 변수 | 4 classes |
| READ | A/B/C/D 현재 값 | 각각 binary, 탐색적 |
| READ | 전체 상태 | 16 classes, `A+2B+4C+8D`로 인코딩, 탐색적 |
| Update | operator / dst | 5 / 4 classes |
| Binary update만 | src / dst_before / src_before | 4 classes / binary / binary |
| Binary update만 | 입력 불일치 | `dst_before XOR src_before`, binary |
| 모든 update | dst_after | binary |
| Binary update만 | operator×입력 진리패턴 | 12 classes, 순서 AND/OR/XOR × 00/01/10/11 |

과거 값은 현재 값과 다른 READ subset에서도 보고한다. 입력 불일치와 결과 값은 AND/OR/XOR 간 전이로 구분한다. SET의 literal이나 NOT의 operand를 binary src로 취급하지 않는다. 결측은 0으로 채우지 않는다.

Label별 각 split에 기대 class가 존재하는지 확인한다. Train/val에서 클래스당 최소 32 positions와 16 distinct sequences가 없으면 해당 분석을 support 부족으로 표시하고 fitting하지 않는다. Test는 존재하는 클래스와 수를 모두 보고하며 기대 클래스가 없으면 완전한 macro 점수를 NA로 두고 관측 클래스 점수를 별도 표시한다.

## 7. Linear probe 상세 명세

### 7.1 구조·파라미터·학습

이진 probe는 `p(y=1|f)=sigmoid(wᵀf+b)`이며 입력 p차원에서 **p+1 parameters**다. 다중 클래스 probe는 `softmax(Wf+b)`이며 C개 클래스에서 **C(p+1) parameters**를 구현한다. Hidden layer, dropout, feature interaction 항은 없다.

| 입력 차원 | Binary | 4-class | 12-class | 16-class |
|---:|---:|---:|---:|---:|
| 256, full activation | 257 | 1,028 | 3,084 | 4,112 |
| 1 feature | 2 | 8 | 24 | 32 |
| 4 features | 5 | 20 | 60 | 80 |

각 feature의 평균·표준편차를 해당 **probe train**에서 계산해 표준화한다. 표준편차 <1e-8인 열은 제거하고 개수를 기록한다. 이는 probe fitting 전처리이며 SAE/Transcoder 학습의 scalar 정규화와 다르다. Binary class weighting은 사용하지 않는다.

```text
L_probe = (1/N) Σ_i CE(y_i, predict(f_i)) + (lambda/2) ||W||_F²
lambda ∈ {0.01, 0.1, 1, 10}
```

Bias에는 penalty를 적용하지 않는다. 이 문서의 grid는 **lambda**이며 scikit-learn의 inverse strength C로 해석하지 않는다. CPU float64, SciPy L-BFGS-B, bounds 없음, 해석적 gradient, 0 초기화, `maxiter=2000`, `maxls=50`, `ftol=1e-12`, `gtol=1e-7`을 사용한다. 수렴 실패 시 같은 목적함수와 초기 조건에서 `maxiter=10000`까지 한 번 재실행하고 실패 여부를 보고한다. 최적화 옵션은 [SciPy 공식 문서](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)를 따른다.

Binary threshold 후보는 `{0.05,0.10,…,0.95}`다. Validation balanced accuracy 최대 → validation CE 최소 → 더 큰 lambda → threshold가 0.5에 가까운 순 → 작은 threshold 순으로 선택한다. 다중 클래스는 argmax를 사용하고 동률은 작은 class ID다. AUROC는 threshold 적용 전 확률로 계산한다. Test로 threshold/lambda를 조정하지 않는다.

### 7.2 입력 표현과 feature 선택

SAE 평가는 full `h_l`, 원래 `h_l` 좌표, `h_l`의 random projection, SAE latent를 비교한다. Transcoder 평가는 full `u_l`, full `m_l`, `m_l` 좌표/random projection, Transcoder latent를 비교한다. 서로 다른 activation의 probe 점수를 동일 정보량의 보장으로 해석하지 않는다.

Random dictionary R은 task seed로부터 `Normal(0,1)` 512개 방향을 뽑아 각 column을 unit norm으로 만든다. SAE 비교에서는 8절의 x, TC 출력 비교에서는 9절의 y_m에 `Rᵀ`를 곱해 feature로 사용하고 train/val/test와 모든 label에서 같은 R을 사용한다.

1. 각 feature에 대해 train에서 class 간/내 분산비인 단변량 ANOVA F-score를 계산한다. Binary에도 같은 기준을 쓴다. 상수 feature는 제외하고 score 동률은 작은 feature ID가 우선한다.
2. 이 순위의 상위 1, 2, 3, 4개로 **누적 prefix**를 만든다. 매 단계에서 새 조합 전체를 탐색하는 forward selection은 하지 않는다. 이 규칙으로 01의 순차 선택을 고정한다.
3. 각 prefix와 lambda로 train probe를 fit한다. 단일 feature 결과는 m=1의 validation 선택값을 사용한다. ≤4 feature 결과는 m∈{1,2,3,4}에서 validation balanced accuracy 최대를 선택한다. 동률은 작은 m, 이후 7.1 규칙을 따른다.
4. 원래 좌표 256개와 SAE/Transcoder/random 512개의 후보 수 차이를 보고한다. 기존 128후보 대조 규모는 유지한다. v1.4에서는 **좌표를 포함한 네 표현 모두** 사전 RNG로 128개를 비복원 선택한 뒤 **다시** train ranking을 한다. 선택 seed·subset ID는 P5 config에 동결한다. 이는 residual 폭 변경에 따른 후보 수 일치의 명시적 정합화이며 실행 완료를 뜻하지 않는다. 성능을 보고 후보를 뽑지 않는다.
5. Label별 feature 집합과 fitted coefficient를 저장한다. 이는 라벨을 이용한 사후 감독 평가이며 dictionary 비지도 학습과 구분한다.

### 7.3 대조군·전이

- 학습 전 LM의 같은 hook에서 full probe를 다시 fit한다. 학습된 LM의 전처리/계수를 재사용하지 않는다.
- 현재 token one-hot 15차원, position/767, 그 제곱의 17차원 입력 probe를 fit한다. 파라미터 수는 binary 18, C-class 18C다. 연산 문맥·답·상태 metadata는 넣지 않는다.
- Shuffled-label 대조군은 train label을 고정 RNG로 permutation한 뒤 같은 선택·fitting을 수행하고 원래 val/test label로 평가한다. 클래스 비율은 유지한다. IID memorization 가능성을 보는 대조군이며 의미적 정답을 보존한 과제가 아니다. Probe 해석의 한계는 [Hewitt & Liang (2019)](https://aclanthology.org/D19-1275/)를 참고한다.
- 변수 전이: READ는 query=A/B/C에서 train/validation 선택, query=D test에서 최종 평가한다. Update는 dst=A/B/C에서 train/validation 선택, dst=D test에서 평가한다. 현재/과거 값·입력 관계·결과 값처럼 공통 label에만 적용한다. 학습에서 빠진 D 자체의 분류 정확도를 전이 지표로 쓰지 않는다.
- 연산 전이: binary update의 AND/OR에서 train/validation 선택, XOR test에서 입력값·입력 불일치·dst_after를 평가한다. Operator ID 및 operator×진리패턴 분류는 미지 class 문제가 되므로 제외한다.
- 전이에서는 표준화, ranking, lambda, m, threshold까지 모두 source-domain train/validation으로 다시 결정한다. SAE/Transcoder의 비지도 학습은 기본 해석 train 전체를 보므로 이를 ‘**probe의 감독 전이**’라고 보고한다. Dictionary도 D/XOR를 보지 않은 완전한 domain holdout 실험은 본 기본 범위에 포함하지 않는다.

## 8. SAE 상세 명세

### 8.1 구조와 전처리

주 SAE는 block 0의 READ `h_0∈R^256`을 입력·복원한다. Train 위치 전체에서 다음 통계를 float64 누적으로 구하고 float32로 저장한다.

```text
mu = mean_train(h)
s  = sqrt(mean_train(||h-mu||²) / 256)
x  = (h-mu)/s
z  = TopK_k(ReLU(W_enc x + b_enc))
x_hat = D z + b_dec
h_hat = mu + s*x_hat
```

`s<1e-8`이면 constant activation 오류로 중단한다. 차원별 whitening 및 sample별 norm 정규화는 하지 않는다. Encoder 입력에서 trainable b_dec를 다시 빼지 않는다. mu/s는 고정 통계이며 학습 파라미터가 아니다.

| 구성 | shape | 파라미터 |
|---|---|---:|
| Encoder weight | 512×256 | 131,072 |
| Encoder bias | 512 | 512 |
| Decoder D | 256×512 | 131,072 |
| Decoder bias | 256 | 256 |
| **합계 / SAE** | 256→512→256 | **262,912** |

Dictionary expansion factor는 2, k는 **4와 16 두 설정 모두** 실행한다. ReLU 후 상위 k개를 유지하므로 실제 positive L0는 k 이하일 수 있다. 동률은 작은 latent ID 우선으로 정렬하고 test에서도 같은 연산을 쓴다. TopK의 기본 동기는 [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)를 따르되, 본 실험은 아래의 단순 변형이며 논문 전체 학습 recipe 재현을 주장하지 않는다.

### 8.2 초기화와 학습

D의 각 column을 독립 표준정규로 생성 후 L2 norm=1로 만든다. Encoder는 `W_enc=Dᵀ` 복사로 초기화하지만 이후 **별도 파라미터**로 학습한다. 두 bias는 0이다.

| 항목 | 고정값 |
|---|---|
| Loss | `mean((x_hat-x)²)`, batch와 256차원 모두 평균 |
| Optimizer | Adam, lr=1e-3, betas=(0.9,0.999), eps=1e-8 |
| Weight decay / LR schedule | 0 / 상수 |
| Batch / 최대 update | 512 positions / 5,000 |
| Sampling | train positions에서 update마다 균등 복원 추출 |
| Gradient clipping | global norm 1.0 |
| Validation | 매 250 updates, 전체 interp_val 해당 위치, no grad |
| Checkpoint | validation MSE 최소, 동률은 이른 update |
| Sparse seed | 기본 0, LM seed 0의 READ에 seed 1 추가 |
| 보조 기법 | L1 penalty, auxiliary loss, dead-latent 재초기화 없음 |

Decoder gradient는 update 전에 각 column의 방사 성분을 제거한다: `g_j ← g_j - d_j(d_jᵀg_j)`. 이후 전체 gradient clipping → Adam step → D column unit normalization 순서로 실행한다. Zero norm 또는 NaN/Inf가 나오면 중단하고 checkpoint를 보존한다. Norm 보정 시 encoder나 latent를 임의 rescale하지 않는다.

5,000 updates는 2.56M position draws, 50k train pool에서 평균 51.2회 노출에 해당한다. 유일 activation 수와 반복 draw 수를 구분한다. Early stopping이나 semantic metric 기반 k 선택을 하지 않는다.

## 9. Transcoder 상세 명세

### 9.1 대상 함수와 구조

Transcoder는 `h→h` 복원이 아니라 **동일 block·동일 위치의 MLP 입력 u에서 MLP 출력 m을 예측**한다. 주 대상은 block 0 READ의 `u_0→m_0`다. Transformer 전체는 동결하고 원래 MLP 출력은 supervision target으로만 쓴다. 프로그램 상태 라벨은 loss에 넣지 않는다.

```text
mu_u = mean_train(u); s_u = sqrt(mean_train(||u-mu_u||²)/256)
mu_m = mean_train(m); s_m = sqrt(mean_train(||m-mu_m||²)/256)
x_u = (u-mu_u)/s_u
y_m = (m-mu_m)/s_m
z_tc = TopK_k(ReLU(E_tc x_u + b_tc_enc))
y_hat = D_tc z_tc + b_tc_dec
m_hat = mu_m + s_m*y_hat
L_tc = mean((y_hat-y_m)²)
```

어느 scale이든 <1e-8이면 중단한다. Encoder 512×256, encoder bias 512, decoder 256×512, decoder bias 256로 **262,912 parameters**다. 입력과 출력의 평균·scale은 서로 다르며 반드시 별도로 저장한다. Residual skip, attention output, LayerNorm을 예측 target에 포함하지 않는다. 추가 선형 skip도 없다.

본 실험은 SAE와 width/k를 맞춘 **TopK Transcoder 변형**이다. [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944)의 기본 목적은 MLP 입출력 근사이며, 해당 논문의 ReLU+L1 학습을 그대로 복제한 것은 아니다. 원 MLP hidden은 1024, TC latent는 512이며 residual dimension 대비 2배 dictionary를 갖는 sparse 대체 모델이다.

### 9.2 초기화·학습·평가

D_tc는 정규 난수 column을 unit norm으로 초기화한다. **E_tc는 별도의 독립 정규 난수 row를 unit norm으로 초기화**한다. 입출력 공간이 다르므로 SAE처럼 decoder transpose를 복사하지 않는다. Bias는 0이다.

Optimizer, batch, 5,000 updates, k∈{4,16}, gradient projection/clipping, decoder norm 제약, sampling, validation 간격, checkpoint 규칙은 SAE와 같다. 기본 seed=0, LM seed 0 READ에는 seed=1도 실행한다. 동일 LM·layer·position·k·sparse seed의 SAE/Transcoder는 같은 position draw 순서를 사용하되 초기화 RNG namespace는 다르게 한다.

Transcoder latent의 의미 평가는 7절 probe 절차를 그대로 사용한다. Prediction NMSE, 실제 L0, dead 비율, **원래 m을 m_hat으로 바꾼 뒤의 행동 변화**를 보고한다. `h` 복원 오차와 `m` 예측 오차는 서로 다른 대상이므로 두 NMSE만으로 SAE/Transcoder의 우열을 결론내리지 않는다.

공정한 동일 target 보조 비교가 필요하면 `m→m` SAE를 8절 그대로 별도 학습하고 같은 위치·sample·width·k에서 Transcoder와 비교한다. 이는 선택 실험이며 기본 READ residual SAE를 대체하지 않는다. 전 위치 MLP를 일괄 대체하거나 전체 회로를 자동 복원하는 분석도 기본 범위 밖이다.

## 10. 재구성·행동·의미 평가 지표

### 10.1 행동

모든 유효 target의 CE, READ 정답 full-vocabulary CE 및 argmax 정확도, B0/B1 제한 이진 정확도, `P(B0)+P(B1)`를 함께 저장한다. Operator·변수·답·블록 길이·거리·진리표별 sample 수와 macro 점수를 보고한다. 빈 strata는 NA 및 coverage로 표시한다.

기준선은 split majority, 독립 Bernoulli(0.5) 추측의 기대 정확도, query 변수의 최근 SET/초기화 값, 가장 최근 READ 답 복사다. 이전 READ가 없으면 복사 기준선은 B0로 예측한다. 각 규칙을 같은 target에 실제 적용하여 정확도를 산출한다.

First/repeat 각 cell의 accuracy·answer CE와 42-cell macro에 95% percentile CI를 보고한다.
Cell 내 independent origin을 1,000회 복원 추출하며 first/repeat가 같은 draw를 공유한다.
Macro gap CI, 요청/유효 draw 수 및 seed를 저장한다. Behavior bootstrap은
`20260920|experiment-spec-v1.4|behavior-bootstrap-r1|<cell_id>`를 SHA-256 파생하여
PCG64를 사용한다. 보고용 unweighted CE와 학습용 read4 loss는 구분하며 CI로 선택·gate를 바꾸지 않는다.

### 10.2 해석 도구 fidelity

SAE는 target v=x, prediction v_hat=x_hat, Transcoder는 v=y_m, v_hat=y_hat으로 둔다.

```text
MSE   = mean_i,dim((v_hat-v)²)
NMSE  = Σ||v_hat-v||² / Σ||v-v_mean_train||²
R2    = 1 - Σ||v_hat-v||² / Σ||v-v_mean_eval||²
EV    = 1 - Σ_dim Var_eval(v-v_hat) / Σ_dim Var_eval(v)
L0_i  = count(z_i > 0)
```

NMSE 분모는 **train 평균 predictor**를 해당 평가 split에 적용한 오차다. 정규화 공간에서 train 평균은 거의 0이다. R2와 EV를 혼동하지 않으며 각각 분모가 0이면 NA로 둔다. L0 평균·중앙값·분위수, latent별 활성 비율을 저장한다. Dead latent는 **선택 checkpoint로 전체 train pool을 다시 encode했을 때 positive activation이 0회인 latent**로 정의한다. Validation/test에서만 비활성인 latent 비율도 별도 기록한다.

Reconstruction 행동 평가는 동결 general behavior test 각 sequence에서 고정 seed로 READ 1개를 선택하고 **그 한 위치만** 대체한다. SAE는 h를 h_hat으로, Transcoder는 m을 m_hat으로 바꾸고 나머지 모델을 실행한다. 해당 답 CE/정확도 변화와 원래 모델 점수를 paired 보고한다. Null patch, 원본 activation을 그대로 반환하는 identity hook이 원래 logits을 재현하는지 먼저 검증한다.

### 10.3 의미 지표

모든 fitting 가능한 label에서 balanced accuracy, macro F1, binary AUROC, confusion matrix를 보고한다. Full probe 대비 단일/≤4 feature 차이, 현재≠과거 subset, 변수·연산 전이를 포함한다. Main label은 READ 현재 값이며 나머지 분석은 보조 또는 탐색적이라고 표시한다. 높은 probe 성능을 모델이 해당 feature를 실제 사용한다는 증거로 대체하지 않는다.

## 11. 반사실적 인과 평가

### 11.1 데이터와 feature 고정

02의 single SET-bit 수정·독립 replay를 사용한다. Target 이전 답 토큰이 모두 동일하고 target READ 직전 prefix가 정확히 한 bit token만 다르게 한다. Changed와 unchanged pair를 모두 생성한다. Changed val/test는 기억·합성 조건을 각각 절반 quota로 구성한다. Unchanged도 같은 분류로 절반씩 만들며, 원래 변경의 영향이 전파되지 않은 다른 query 변수에서 불변임을 replay로 확인한다. 조건 충족 실패는 5절의 생성 중단 규칙을 따른다.

기억/합성 분류는 02에 따라 변경 SET 이후 변경 변수에 논리 갱신이 없는지/있는지로 정한다. 이를 전체 회로가 무연산/합성이라는 강한 의미로 확장하지 않는다. Pair의 전체 origin을 한 split에 묶고 target 뒤 suffix와 정답을 모델 입력에서 제외한다.

SAE/Transcoder 각각 interp train/val의 **READ 현재 값** probe가 선택한 m=1 및 ≤4 feature ID 집합 J를 causal test 전에 고정한다. Causal val은 random matching 규칙 및 구현 점검용이며 test 효과를 보고 J를 재선택하지 않는다. 두 k 결과를 모두 보고한다.

### 11.2 개입 수식과 실행

원본을 o, 대조를 c로 둔다. 양쪽에서 같은 target 위치의 activation/latent를 추출한다.

```text
SAE residual patch:
  h_patch = h_o + s * D[:,J] @ (z_c[J]-z_o[J])

Transcoder MLP-output patch:
  m_patch = m_o + s_m * D_tc[:,J] @ (z_tc_c[J]-z_tc_o[J])
  h_patch = r_mid_o + m_patch
```

SAE는 원래 residual의 재구성 오차를, Transcoder는 원래 MLP output의 예측 오차를 보존한다. TC patch에서 input scale s_u를 output에 곱하지 않는다. TC input u 자체를 패칭하지 않으며 residual skip은 원본을 유지한다.

원본 prefix 전체를 다시 forward하면서 지정 hook 한 위치를 위 값으로 교체한다. Downstream 계산은 다시 실행하고 이전 실행의 downstream/KV cache는 재사용하지 않는다. 첫 block READ patch를 기본으로 하고, block 1 및 update patch는 별도 실험 ID로 구분한다.

### 11.3 필수 대조군

| 대조군 | SAE hook | TC hook |
|---|---|---|
| 무개입/identity | h_o | m_o |
| 평균 대체 | mu | mu_m |
| 전체 근사 대체 | h_hat_o | m_hat_o |
| 선택 feature | 11.2의 h_patch | 11.2의 m_patch |
| Full donor patch | h_c 전체 | m_c 전체, r_mid_o 유지 |
| Random latent | 같은 dictionary의 무작위 m개 donor 차이 | 동일 |
| 원래 좌표 | train/val에서 고른 m개 h 좌표의 donor 교체 | m 좌표의 donor 교체 |
| Random 방향 | 고정 R에서 선택한 m개 방향의 donor 차이 | m 공간에서 동일 |

좌표 patch는 `v_o + Σ_j e_j e_jᵀ(v_c-v_o)`다. 비직교 random 방향 Q의 patch는 `v_o + Q(QᵀQ)^+Qᵀ(v_c-v_o)`로 정의한다. SAE decoder 합과 투영 patch는 연산이 다르므로 norm과 효과를 함께 보고한다. Random 방향은 의미 평가에서 선택한 집합과 순수 무작위 집합을 구분해 저장한다.

Random latent 대조는 각 pair와 m에 대해 J와 겹치지 않는 m개 집합을 고정 RNG로 최대 200개 생성한다. Matched 비교는 causal val에서 선택 patch의 `||delta_h||₂`와 `||delta_z||₂`의 20/40/60/80 percentile로 5×5 bin을 고정한 뒤 같은 bin 후보 최대 20개를 사용한다. TC의 delta_h는 `s_m D_tc delta_z`다. Bin 경계 중복은 병합하고 선택 patch가 0인 경우 zero bin을 분리한다. Test에서 bin을 다시 추정하거나 후보를 효과로 고르지 않는다.

후보가 없으면 matched 효과는 NA, 미매칭 수·비율을 보고한다. Unmatched random도 별도로 보존한다. matched 비교는 선택 patch와 random 양쪽을 동일한 matching 성공 pair subset에서 계산한다.

### 11.4 결과 계산

Changed pair에서 `M = logit(answer_c)-logit(answer_o)`로 정의하고 `delta_M=M_patch-M_original`을 보고한다. Flip 성공은 패치 후 **15-token argmax가 counterfactual 답과 같음**이다. Binary B0/B1 flip도 별도 저장한다.

Unchanged pair에서는 정답과 반대 bit 간 margin 변화, 원래 맞은 답을 틀리게 만든 비율, 전체 예측이 바뀐 비율을 보고한다. 단순한 출력 교란을 선택적 인과 효과로 해석하지 않는다.

전체 pair 결과와 원본·대조 모두 원래 모델이 맞힌 subset을 함께 보고한다. 성공 pair만 남겨 전체 효과를 계산하지 않는다. Pair 양방향 o→c와 c→o를 평가하고 origin당 평균 후 집계한다. 방향 두 개를 독립 표본으로 세지 않는다. Full donor patch도 성공이 보장되는 상한으로 취급하지 않는다.

## 12. 실제 실행 순서와 단계별 완료 기준

아래 파일명은 **구현할 entry point의 명세**이며 현재 존재하는 실행 코드를 의미하지 않는다. 각 실행은 config와 input hash를 받아 결과 manifest를 반환해야 한다.

| 단계 | 실행 내용 | 입력 → 출력 | 완료 기준 |
|---|---|---|---|
| 0 | `generate_corpus.py`, `validate_corpus.py`를 CPU에서 실행 | 02/config → corpus/통계/검증 | 5.3 전항목 통과 |
| 1 | Colab 연결, 데이터 복사·환경 lock, `smoke_test.py` | corpus/환경 → smoke 로그 | 파라미터·mask·hook·resume 검증 |
| 2 | seed 0 `train_lm.py` 파일럿 | train/val → best/last checkpoint | 4.4에 따른 통과 또는 중단 |
| 3 | 예산 동결, seed 1·2 학습, `eval_behavior.py` | frozen 설정 → seed별 행동표 | 실패 seed 포함 결과 보존 |
| 4 | `cache_activations.py`, `fit_probes.py` | 통과 LM/interp → READ cache/full probe | key·라벨·support 검증 |
| 5 | `train_dictionary.py --kind sae` | h_0 READ → k별 SAE | 5k updates, val MSE 선택 |
| 6 | `evaluate_dictionary.py`, `run_patching.py` | SAE/probe/pairs → READ SAE 표·그림 | 후보 수 대조·fidelity·인과 대조 완료 |
| 7 | `train_dictionary.py --kind transcoder` 및 같은 평가 | u_0,m_0 READ → TC 표·그림 | SAE와 동일 절차 완료 |
| 8 | seed 0 READ sparse seed 1 재실행 | 고정 cache → 초기화 반복 | SAE/TC 두 k 모두 보고 |
| 9 | 자원 범위에서 update, block 1, length 실행 | 별도 config → 2차 결과 | 실행/생략 및 사유 표시 |
| 10 | `aggregate_results.py` | 모든 결과 → CSV/그림/보고서 | 재현 정보·실패·CI 포함 |

단계 1 smoke test는 별도 debug 데이터로 2 batch forward/backward, 작은 pool 100 dictionary updates, probe 1개 fitting, identity/full/sparse patch 각 1회, checkpoint 저장·복구 후 다음 update 일치를 검사한다. Debug checkpoint나 feature를 본 실험에 재사용하지 않는다. Padding batch 결과와 개별 sequence 결과가 유효 위치에서 허용 오차 `atol=1e-5, rtol=1e-4` 안에 드는지 확인한다. 미래 suffix를 바꾸어도 이전 logits가 바뀌지 않는지도 검사한다.

첫 본실험 실행에서 LM 50 updates 및 dictionary 100 updates의 처리량과 peak VRAM을 측정한다. LM 시간은 `남은 비패딩 토큰/측정 tokens_per_second + 평가·저장 시간`으로, dictionary는 `남은 updates × 측정 seconds_per_update + 평가 시간`으로 추산한다. 측정 구간은 본실험 예산 안에 포함한다. 시간이나 GPU 자원이 부족하면 checkpoint에서 멈추고 재개하며, width/k/token budget을 몰래 줄이지 않는다.

## 13. 반복 수와 계산 예산

| 범위 | 실행 수/최대량 |
|---|---|
| LM | 3 seeds × 64M = 192M 예측 토큰 + seed별 마지막 batch 초과량 |
| READ SAE, 주 layer | 통과 LM 수 G × 2 k × 1 sparse seed |
| READ TC, 주 layer | G × 2 k × 1 sparse seed |
| seed 0 READ 초기화 민감도 | SAE 2 k + TC 2 k = 4 runs 추가, seed 0 gate 통과 시 |
| G=3 기본 dictionary 총량 | 16 runs × 5k = 80k updates, 40.96M position draws |
| Update, 2차 분석 | G×2 k×2 도구, 최대 12 runs 추가 |
| Block 1, 2차 분석 | 선택한 위치 종류마다 G×2 k×2 도구, 별도 계상 |

Probe와 baseline은 같은 cache를 재사용하고 GPU 학습 예산에 포함하지 않는다. CPU 시간, GPU 시간, activation 추출, validation, patching 비용을 각각 기록한다. 자원이 부족하면 기본 READ 분석을 우선하고 2차 분석을 생략한다. 필수 READ TC가 남았으면 전체 실험을 완료로 표시하지 않는다.

각 LM seed 결과와 평균·최솟값·최댓값을 보고한다. Dictionary seed 반복은 LM 반복과 구분한다. Bootstrap은 1,000회, 2.5/97.5 percentile 95% CI다. Interpretation은 sequence ID, causal은 origin ID를 cluster로 복원 추출하고 cluster 내부 모든 위치·양방향을 함께 가져온다. 조건별 고정 quota 평가는 조건 내 cluster bootstrap 후 macro를 다시 계산한다. 통계 seed를 manifest에 저장한다.

3개 LM seed의 평균을 수천 개 독립 모델의 정밀한 추정처럼 다루지 않는다. 클래스가 사라져 지표 계산 불가능한 bootstrap draw는 제외하고 유효 draw 수를 보고한다. 핵심 결과는 효과 크기와 CI로 제시하며 탐색적 다중 비교의 유의성만으로 feature 의미를 확정하지 않는다.

## 14. 저장 구조와 최종 산출물

아래는 산출물 역할별 목표 구조이며 미실행 P5 이후 파일의 존재를 주장하지 않는다.
실제 동결 corpus는 `data/language_v1_4/rebuild_01/`, LM checkpoint는
`experiment_v1_4/frozen_test_r1/checkpoints/seed{0,1,2}.pt`, 환경·원본은 evidence/results에 있다.
신규 산출물 경로는 P5 config에서 별도 확정하고 기존 파일을 덮어쓰지 않는다.

```text
experiment_v1_4/
  configs/
    language.json
    transformer.json
    sae.json
    transcoder.json
    probe.json
    evaluation.json
  environment/
    requirements-cpu.lock.txt
    requirements-colab.lock.txt
    runtime_manifest.json
  data/
    corpus_manifest.json
    cpu_validation.json
    corpus_statistics.json
    train_shards/
    fixed_splits/
    interpretation/
    causal_pairs/
  runs/lm_seed_{0,1,2}/
    init.pt
    last.pt
    best.pt
    training.jsonl
    behavior.json
    activations/
    probes/
    dictionaries/{sae,transcoder}/
    interventions/
  results/
    run_registry.csv
    behavior.csv
    semantic_metrics.csv
    fidelity_metrics.csv
    causal_metrics.csv
    figures/
    report.md
  notebooks/
    00_cpu_corpus.ipynb
    01_colab_lm.ipynb
    02_colab_interpretation.ipynb
    03_analysis.ipynb
```

각 run ID는 spec version, LM seed/checkpoint hash, layer, hook, READ/update, 도구, k, sparse seed를 포함한다. Manifest에는 정확한 파라미터 수, 모든 config, 코드 hash, 환경 ID, 파일 hash, 실제 token/position/draw 수, best update, 학습·평가 시간, peak memory, 실패/생략 사유를 기록한다. Feature ID는 dictionary 간 공통 의미를 가진다고 가정하지 않는다.

P5 이후 분석 난수는 계승 규칙인 `20260909|experiment-spec-v1.0|<purpose>|<run_key>`의 SHA-256 앞 8 bytes를 unsigned big-endian 정수로 변환해 사용한다. Purpose는 `dictionary_init`, `position_draw`, `random_projection`, `candidate_subset`, `shuffle_label`, `reconstruction_target`, `random_patch`, `bootstrap`으로 고정한다. NumPy는 PCG64를 쓰고 PyTorch seed는 해당 정수를 `2^63-1`로 나눈 나머지로 설정한다. Run key는 위 run ID의 필드를 고정 순서로 연결하되, 공유해야 하는 position_draw에서는 도구 종류를 제외하고 reconstruction_target에서는 LM/도구 종류를 제외하여 같은 표본을 사용한다. Sparse seed 0/1은 이 run key에 들어가는 반복 ID이며 LM 직접 초기화 seed 0/1/2와 역할이 다르다. 최종 파생 정수와 key도 manifest에 저장한다. Namespace의 v1.0 문자열은 계승한 RNG 식별자이며 적용 실험 버전은 run key의 v1.4로 구분한다. 기존 LM·corpus·behavior bootstrap의 RNG 기록을 바꾸지 않는다. P5 config에 실제 파생값을 사전 고정한다.

필수 그림은 다음과 같다.

1. LM 답 CE·정확도 학습 곡선 및 진단·holdout 조건별 성능.
2. Full probe / 좌표 / random / SAE / TC의 단일·≤4 feature label 접근성. TC의 u/m 기준선을 함께 표시.
3. NMSE–L0 및 선택 위치 대체 전후 답 CE·정확도. SAE/TC의 서로 다른 target을 명시.
4. Changed delta margin·flip과 unchanged 오류 유발률, full/random/좌표 대조, matching coverage.
5. LM seed와 sparse seed별 분포, 현재≠과거 및 전이 분석.

**완료 판정:** CPU 검증, LM의 사전 gate 결정, 통과한 LM들의 필수 READ probe·SAE·Transcoder 두 k·후보 수 대조·근사 대체·인과 평가·재현 정보가 모두 보고되어야 한다. 해석 진입 조건인 최소 두 통과 seed를 충족하지 못하면 ‘행동 학습 단계에서 중단’한 보고서로 끝내며 SAE/TC 가설을 검증했다고 쓰지 않는다. 본 `03_experiment_spec.md`는 그 실행을 위한 명세서이고 실험 수행 자체는 별도 작업이다.
