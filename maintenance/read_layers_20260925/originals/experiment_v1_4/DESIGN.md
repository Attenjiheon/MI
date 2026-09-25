# v1.4 P3 — deepwide12_read4 64M 사전 동결 설계

작성: 2026-09-20. 상태: **정식 설계 동결, 실행 전**.

이 버전은 `experiment_v1_3/results/next_architecture_proposal.json`을 실행 계약으로 승격한
별도 실험이다. v1.3의 코드·데이터·checkpoint·평가 split을 덮어쓰지 않는다. 원 언어와
IID 생성 분포는 유지하지만 아키텍처, 학습 예산, LR decay 경계, select pair quota를 함께
바꾸므로 v1.3 대비 차이를 아키텍처만의 인과 효과로 해석하지 않는다.

## 1. 진입 근거와 선행 조건

v1.3 `wide4_read4`, LM seed 0의 32M confirm 반환 ZIP은 SHA-256
`c7cca1856c51cddcf778c50c946fad57ca68677a6dde660fdc4676cd204e57d9`이며 내부 감사가
`passed`다. 선택 checkpoint는 update 4,249 / 32,004,917 prediction tokens,
checkpoint SHA-256 `00303521…bb3`이다. Gate는 general 98.1534%, first 42-cell macro
94.6801%, repeat 100%로 실패했다. 세 legacy 진단은 통과했고 frozen test는 열지 않았다.

따라서 v1.3 seed 1·2 및 P5 이후를 실행하지 않고, 실패 결과를 보존한 채 v1.4를 새로 만든다.

## 2. 이번 버전의 고정 변경

| 항목 | v1.4 값 |
|---|---|
| 후보 | `deepwide12_read4` 하나 |
| Transformer | 12 blocks, residual 256, 4 heads × 64, MLP 1024 |
| 파라미터 | 9,485,312 |
| 목적함수 | token-only `read4`; READ 답 weight 4, 기타 non-PAD weight 1 |
| LM seed | 0; 통과 뒤에만 1·2 |
| 명목 예산 | 64M prediction tokens, 최초 완전 update overshoot 포함 |
| LR | 50k warmup, 57.6M까지 3e-4, 64M에서 3e-5로 선형 decay |
| select pair | 42 cell × 64 independent origins |
| gate pair | 42 cell × 64 independent origins |
| test pair | 42 cell × 128 independent origins |
| 필수 sparse 위치 | block 0 READ |

Embedding/unembedding은 untied이고, exact GELU, Pre-LN, RoPE base 10,000, dropout 0을
유지한다. Linear/embedding은 `Normal(0, 0.02²)`, attention/MLP output weight는 추가로
`1/sqrt(24)`를 곱한다. LM forward 입력은 token IDs와 padding mask뿐이다.

## 3. 데이터와 누출 방지

새 root는 `data/language_v1_4/`, RNG namespace는
`20260920|language-v1.4|<purpose>|<index>`다. 먼저 fresh select/gate/test,
interpretation, causal split을 예약하고, 이후에 train을 확장한다. 새 sequence와 READ/causal
prefix는 v1.0–v1.3 및 v1.4 예약 집합 전체와 충돌하면 거절한다.

v1.3 train 68 shards / 271,936 sequences / 32,004,917 prediction tokens는 파일별로
byte-identical하게 복사한다. 그 뒤 같은 IID 분포에서 64M을 처음 넘는 완전 64-sequence
update까지 확장한다. shuffle과 sequence 반복은 없고 모든 LM seed가 같은 순서를 한 번씩
소비한다. 상태·정답 metadata는 평가와 해석에만 쓰며 LM 입력이나 auxiliary loss로 넣지 않는다.

## 4. 선택과 행동 gate

Checkpoint 후보 경계는 1M, 3M, 8M, 16M, 24M, 32M, 48M, 57.6M, 60.8M, 64M이다.
Select first-member 42-cell macro answer CE의 전역 최소를 기준으로 하고, 최소와 `1e-4` nats
이내 후보는 select/general answer CE, 이른 update 순으로 고른다. Gate는 이 checkpoint에서
seed당 한 번만 열며 재선택하지 않는다.

통과에는 다음 전부가 필요하다.

1. General READ full-vocabulary accuracy ≥99%.
2. 세 legacy diagnostic 각각 ≥95%.
3. First 42-cell macro ≥95%.
4. First의 연산별, depth-bin별, answer별 accuracy 각각 ≥95%.
5. Repeat 42-cell macro ≥95%, 모든 quota와 coverage 100%.

개별 42 cell은 count와 CI를 보고하지만 추가 gate로 사용하지 않는다. Seed 0이 실패하면
v1.4를 중단하고 seed 1·2, frozen test, P5 이후를 실행하지 않는다. Seed 0이 통과하면 같은
설정으로 seed 1·2를 실행하고 최소 두 seed가 통과해야 표현 분석에 진입한다. 모든 학습·validation
결정이 동결된 뒤에만 test를 한 번 연다.

## 5. 실행·재개·증빙

본학습 전 새 코드와 새 데이터로 CPU 및 Colab GPU smoke를 모두 통과한다. GPU smoke에서
microbatch를 16→8→4→2→1 순으로 줄일 수 있으며 선택된 값을 checkpoint에 저장한다.
Effective batch 64는 바꾸지 않는다. FP32, deterministic algorithms, AMP/TF32/compile 비활성이다.

Checkpoint는 init, 모든 milestone, 15분 경과 뒤 완전 update 경계, last를 영속 저장한다.
모델·optimizer·RNG·cursor·다음 milestone·config/data/code hash·환경 ID를 포함한다. 복구는
마지막 완전 index에서만 허용하며 실제 토큰 수, overshoot, 첫 50 update 처리량, peak VRAM,
평가·저장 시간을 기록한다.

## 6. 현재 완료 판정

설계와 파라미터 산식만 동결됐다. 새 corpus 생성·독립 감사, v1.4 실행기와 test, CPU/GPU smoke,
Colab notebook/input bundle, seed 0 64M 학습은 별도 증빙이 있어야 완료다. 로컬 준비가 끝나도
GPU smoke와 학습 반환물을 검증하기 전에는 P3 또는 행동 gate를 완료로 표시하지 않는다.
