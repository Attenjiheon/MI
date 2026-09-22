# v1.2 — 4-block / 16M 학습 예산 실험 설계

작성: 2026-09-17. **설계 확정, 구현·데이터 생성·학습 미실행.** 기계 판독 계약은 [design_config.json](./design_config.json). 이는 기존 CLI에 전달할 실행 config가 아니다.

## 1. 질문과 근거

같은 4-block 모델에 같은 분포의 새로운 학습 예제를 더 제공하면, 3M에서 실패한 상태 추적 행동이 16M에서 gate를 통과하는가?

v1.1은 797,184 parameters, 407 updates / 3,004,531 prediction tokens에서 일반 validation 정확도 58.90%, 답 CE 0.670175로 실패했다. 마지막 네 CE는 0.678755 → 0.676295 → 0.671137 → 0.670175로 계속 감소했다. 데이터 부족은 검증할 가설이지 확정된 실패 원인은 아니다. [기존 중단 보고](../experiment_v1_1/results/p3_stop_report.md)를 보존한다.

Chinchilla의 약 20 tokens/parameter를 참고하면 797,184 × 20 = 15,943,680이다. 이를 반올림한 명목 16,000,000을 사용한다. [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)는 고정 계산량에서의 모델/데이터 배분을 다루며, 이보다 훨씬 작은 인공어 모델의 최적 토큰 수나 99% 정확도를 보장하지 않는다. 본 설계는 기존 validation 관측에 따른 탐색적 변경이다.

## 2. 고정 조건과 명시적 변경

| 항목 | v1.2 계약 |
|---|---|
| 모델 | v1.1과 동일: 4 blocks, width 128, 4 heads, MLP 512, vocab 15, RoPE, 797,184 parameters |
| 초기화 | v1.1과 동일, attention/MLP output weight multiplier 1/sqrt(8) |
| Optimizer/loss | 기존 AdamW 전체 설정 및 모든 유효 next-token CE 유지 |
| LR | 3e-4, 첫 50k prediction tokens warmup 후 상수; cosine 변경 없음 |
| Batch/정밀도 | effective 64, microbatch 16부터 OOM 규칙 적용, FP32/결정성 유지 |
| 예산 | seed별 명목 16M; 처음 도달/초과한 완전한 update 종료, overshoot 기록 |
| 시작 | seed 0 fresh initialization, 기존 checkpoint를 이어받지 않음 |
| 평가 | 100k 토큰 경계 및 최종 update, 같은 update의 중복 평가 금지 |
| 중간 기록 | 1M / 3M / 8M / 16M 경계에서 actual tokens, current와 best-so-far 모두 기록 |
| 종료 | 조기 gate 통과/미달로 예산 변경 없음. 수치 오류·데이터 오류는 실패, 자원 중단은 paused |

03 §4.3–4.4의 1M 조건부 3M 정책은 이 버전에서 고정 16M으로 대체한다. §5.1의 train 양과 §5.2.6의 shard 규칙은 아래 계약으로 대체한다. §13의 LM 최대량은 seed 0 통과 후 3 seeds × 16M = 48M + overshoot이다. 그 외 상세 모델·분포·누출 방지·평가·해석 규격은 v1.1 및 원문을 계승한다. 2-block 16M 대조군은 이번 필수 범위에 추가하지 않는다.

## 3. 데이터: 기존 prefix 보존 + 새 표본 추가

1. `data/language_v1`은 불변이다. 새 `data/language_v1_2`에 새 manifest를 만든다. 원래 validation/test/interpretation/causal 파일, token 파일과 metadata는 byte-identical하게 복사하고 hash를 검증한다. 평가 점수는 생성에 사용하지 않는다.
2. 기존 train 7개 shards, 26,048 sequences / 3,004,531 prediction tokens를 원래 순서·바이트 그대로 첫 구간에 둔다. 기존 마지막 짧은 shard `00006`도 보존한다. **이 prefix 경계의 짧은 shard를 허용하는 것이 기존 “최종 shard만 짧음” 규칙의 명시적 예외다.**
3. 새 train은 `00007`부터 shard당 4,096 sequences로 생성한다. 마지막만 짧게 허용하고 전체 64-sequence update를 완성한다. 기존 미완료 shard RNG를 이어받는다고 주장하지 않는다. shard index별 기존 seed 공식 `SHA256(20260909|language-v1.0|lm_train|index)[:8]` unsigned big-endian, PCG64를 사용한다. 실제 seed 정수, RNG 상태, 수락/거부 수를 manifest에 남긴다.
4. 생성 전 기존 모든 split·pair의 canonical hash 및 causal prefix 예약 집합을 검산하여 적재한다. 신규 후보는 기존/신규 전체 시퀀스 중복 및 예약 causal READ prefix 충돌을 거부한다. 동일 holdout 제외, 기본 분포, target당 MAX_ATTEMPTS=100,000을 적용한다. 정답 균형화·재표집·반복 학습·셔플을 하지 않는다.
5. 누적 소비 prediction tokens가 16M에 도달하는 첫 64-sequence update까지 확보한다. 토큰은 각 sequence의 `len(tokens)-1` 합이며 padding은 제외한다. 원시 파일 크기나 READ 개수와 혼동하지 않는다. 정확한 final cursor/update/overshoot는 생성 완료 후 계산하여 모든 LM seeds에 고정한다.
6. 기존+추가 데이터 전체에 03 §5.3 독립 replay, metadata 정합성, split/holdout/prefix 누출, coverage/quota, 두 번 생성 결정성 검사를 수행한다. 새 manifest·통계·감사 보고·코드/파일 hashes가 모두 있어야 P1 통과다. 아직 생성하지 않은 hash는 null로 둔다.

## 4. 선택, gate 및 해석

전체 16M validation 이력에서 일반 READ 답 CE 최소 checkpoint 하나를 선택한다. 동률은 이른 update다. 그 checkpoint의 full-vocabulary 정확도가 일반 ≥99%, 세 진단 각각 ≥95% (각 512 독립 sequences)여야 통과다. 다른 checkpoint로 gate를 우회하지 않는다.

- Seed 0 통과: 동일 데이터 순서·최종 update·16M 예산을 동결하고 seed 1·2 재현으로 진행한다. 각 seed 독립 CE 선택, 실패 seed 보존, 통과 LM만 표현 분석에 사용한다.
- Seed 0 실패: 16M에서 종료 보고. 자동 추가 예산·구조 변경·SAE/TC 진행 없음.
- Test는 고정된 선택 후 최종 보고에만 사용한다. 중간 milestone에서는 validation만 평가한다.

v1.2 자체의 3M→16M current 및 best-so-far CE/정확도, 진단별 지표, B0/B1 확률질량, 전체 LM CE와 strata를 보고한다. 선택 후보가 30회에서 약160회로 늘어나는 효과를 구분하도록 current와 selected를 모두 제시한다. v1.1의 3M와 새 3M의 입력 prefix·초기화·update를 대조한다. 환경이나 microbatch가 다르면 결과 차이를 기록하고 bitwise 재현을 가정하지 않는다. 같은 seed와 prefix의 연장 비교를 독립 seed 재현으로 세지 않는다. 성공해도 “16M이 최적” 또는 “데이터 부족만이 원인”으로 결론짓지 않는다.

## 5. 저장·재개·Colab 계약

매 validation의 checkpoint를 update별 immutable 파일로 보관하고 init, best/last 참조를 유지한다. 모델·optimizer·RNG·cursor·다음 평가 경계·config/code/data hashes를 모두 저장한다. milestone에서 그 시점의 current와 best checkpoint hash를 인덱스에 기록한다. 과거 1M checkpoint 바이트가 덮어써진 문제를 반복하지 않는다.

로컬 atomic write 후 영속 저장소에 **새 파일만** 복사하고 checksum을 확인한 다음 완료 인덱스를 atomic 갱신한다. 매번 누적 checkpoint 디렉터리를 통째로 중복 snapshot하지 않는다. 재개는 같은 v1.2 config/data hash의 마지막 완전 update에서만 허용한다. 기존 v1.1 checkpoint는 감사·비교용이다.

Colab 실행은 구현 완료 후 `.ipynb`와 입력 ZIP으로 납품한다. 업로드 이름 대신 SHA-256으로 번들을 식별하고, 환경 디렉터리는 `mkdir(parents=True, exist_ok=True)`로 생성한다. 노트북에 의존성/환경 기록, 전체 입력 checksum, CPU/GPU smoke, 실행·영속 저장·재개·결과 다운로드를 포함한다. 현재 기존 1M/3M CLI·노트북을 16M 지원 도구로 간주하지 않는다.

## 6. 자원과 다음 단계

기존 T4 첫50 updates 처리량 약60,099 tokens/s를 적용하면 16M 학습 자체는 약266초다. 기존 validation 30회 약145초에서 단순 외삽한160회 평가는 약773초다. 합계 약17분은 거친 참고치이며 CPU 생성/감사, smoke, 업로드, 저장, 런타임 변동은 별도다. 시작 후 실측으로 갱신한다. 약10MB checkpoint ×160회 ≈1.6GB/seed 수준의 저장 공간에 init·로그·번들·다운로드 여유를 추가로 확보한다.

| 순서 | 상태 / 완료 증빙 |
|---|---|
| 설계 | 본 문서·설계 JSON·hash manifest 작성 완료 |
| v1.2 구현 | pending: 생성 확장, 16M 예산/선택/재개, immutable checkpoint, config 검증 |
| P1 | pending: 새 corpus 및 전체 CPU 감사, exact cursor/update 동결 |
| P2 | pending: 경계·shard 연결·중단 재개 검증 및 CPU/GPU smoke |
| Colab 납품 | pending: `.ipynb`와 hash 검증 가능한 입력 번들 |
| P3 | pending: seed 0 실제 실행 증빙·선택·gate 감사 |

설계 파일의 hash는 `design_manifest.json`에 기록한다. 실행 시에는 별도로 실행 config/code/data hashes를 생성해야 한다. 설계 완료를 P1/P2/P3 통과로 표시하지 않는다.
