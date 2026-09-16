# 실험 실행 계획서

작성일: 2026-09-10  
기준: [01_experiment_design.md](./01_experiment_design.md), [03_experiment_spec.md](./03_experiment_spec.md)  
실행 규칙: [AGENTS.md](./AGENTS.md)

이 문서는 어떤 작업을 어떤 순서로 수행하고, 어떤 증빙이 있어야 다음 단계로 넘어갈지를 정한다. 체크박스는 실행 결과를 확인한 후에만 표시한다. 문서 작성 자체로 실험을 수행하거나 기존 코드·데이터의 검증 완료를 선언하지 않는다.

## 1. 전체 순서와 우선순위

`P0 규격·현황 정리 → P1 CPU 검증 → P2 구현·환경 smoke test → P3 LM 파일럿 → P4 LM 재현·행동 평가 → P5 READ cache·probe → P6 READ SAE → P7 READ SAE 평가·개입 → P8 READ TC 학습·평가·개입 → P9 sparse seed 반복 → P10 선택 분석 → P11 최종 집계`

P3에서 행동 gate에 실패하면 규정된 연장 또는 중단 절차로 분기한다. 필수 READ가 미완료이면 P10을 시작하지 않는다. P11의 기록·중간 집계는 각 단계에서 누적하며, 중단 시에도 실패 보고서를 작성한다.

| 범위 | 우선순위 |
|---|---|
| 통과 LM별 block 0 READ probe·SAE·TC, k=4/16, fidelity·의미·인과 대조 | 필수 |
| LM seed 0 READ의 두 도구·두 k, sparse seed 1 | 필수 초기화 반복 |
| Update, block 1, 길이 외삽 GPU 평가, m→m SAE | 자원 확인 후 별도 실험 |
| Crosscoder, 변수 8개, 대규모 모델/seed sweep | 기본 범위 제외 |

## 2. P0 — 규격과 현재 산출물 정리

**선행 조건:** 없음. **장소:** 로컬 CPU.

- [x] 03 §1의 우선순위를 적용한다. TC 필수화와 CPU 사전 생성 shard 사용을 config에 반영한다.
- [x] 기존 `corpus/`, `tests/test_corpus.py`, `data/language_v1/`의 코드·설정·manifest·검증 보고서를 확인한다. 파일 존재와 규격 충족을 구분한다.
- [x] 데이터가 현 규격과 맞으면 재사용하고, 다르면 차이와 버전을 남겨 별도 생성한다. 기존 결과를 덮어쓰지 않도록 run 경로를 정한다.
- [x] Language/transformer/SAE/TC/probe/evaluation config, run ID, hash, seed namespace, 상태 기록 형식을 마련한다.
- [x] Train/val/test 역할, feature·threshold 선택 규칙, 선택 분석 범위를 문서에 고정한다.

**산출물:** config 세트, 기존 산출물 재사용/수정 목록, run registry 초기본. 03 §14의 `experiment_v1/` 구조를 기준으로 실제 경로 매핑을 남긴다.

**완료 조건:** 구현자가 사용할 규격과 실제 입력 경로가 명확하고, 미검증 항목이 구분되어 있다.

## 3. P1 — CPU 생성기·코퍼스 검증

**선행 조건:** P0. **장소:** CPU만 사용.

1. 기존 언어·생성·replay·audit 구현을 명세와 대조하여 보완한다.
2. 768전이, parser roundtrip, operand/source 규칙, 복원 성질, index/snapshot/null 처리를 검증한다.
3. 작은 고정 seed 샘플을 두 번 생성하여 token·metadata hash 일치를 확인한다.
4. Validation/test → interpretation → causal → LM train 순서로 데이터를 예약·생성하거나 기존 데이터의 동일 증빙을 확인한다. 길이 test도 여기서 예약한다.
5. 독립 replay로 전체 READ를 검산하고 split/READ prefix 중복, holdout 누출, 명령/2연산 coverage, pair 조건을 검사한다.
6. Quota·수락률·거부 사유·분포 통계·파일 checksum·RNG state를 저장한다.

- [x] [03_experiment_spec.md §5.3](./03_experiment_spec.md#53-cpu-검증과-납품물)의 모든 CPU 검증 항목을 통과했다.
- [x] Train은 3M 토큰 및 마지막 64-sequence batch를 확보하며, 해석/인과/진단 quota가 충족된다.
- [x] 생성 시도 상한 도달이나 quota 부족을 정상 완료로 표시하지 않았다.

**산출물:** corpus, `corpus_manifest.json`, `cpu_validation.json`, `corpus_statistics.json`, hash·split·seed 정보.

**진행 조건:** 03 §5.3의 전 항목 통과. 오류·누출·부족이 있으면 CPU 단계에서 해결하고 관련 검증을 다시 수행한다. GPU 학습을 먼저 시작하지 않는다.

## 4. P2 — 학습·해석 구현과 Colab smoke test

**선행 조건:** P1 통과. **장소:** 로컬 CPU + Colab GPU.

1. Transformer, 학습/평가, hook/cache, probe, SAE/TC, patching, 저장/재개 기능을 구현한다. 문서 진입 규칙은 AGENTS.md, 상세 값은 03을 따른다.
2. 데이터를 `/content/boolean_interp/`로 복사해 hash를 확인하고 실제 GPU/VRAM/RAM/저장 공간을 기록한다.
3. Float32·결정성·RNG 설정을 적용하고 별도 debug 데이터로 아래 smoke test를 실행한다.
4. 통과 환경의 정확한 패키지·CUDA/cuDNN 버전과 lock 파일을 저장한다.

- [x] LM 파라미터 400,640, SAE/TC 각각 131,712가 맞다.
- [x] 2 batch forward/backward가 동작하고 token-weighted accumulation·mask·target shift가 맞다.
- [x] Padding batch와 개별 sequence의 유효 위치 결과가 `atol=1e-5, rtol=1e-4` 안에서 같다.
- [x] 미래 suffix를 바꿔도 이전 logits가 바뀌지 않는다.
- [x] READ/update hook 위치와 `h=r_mid+m`, `u=LN_mlp(r_mid)`를 확인했다.
- [x] 작은 pool에서 dictionary 100 updates와 probe 1개 fitting이 동작한다. SAE와 TC 경로의 scale·shape를 점검했다.
- [x] Identity/full/sparse patch 각 1회가 동작하고 identity는 원래 logits를 재현한다.
- [x] Checkpoint 저장·복구 후 다음 update 결과와 데이터/sampler 진행이 일치한다.

**증빙:** [experiment_v1/P2_STATUS.md](./experiment_v1/P2_STATUS.md), `experiment_v1/smoke/cpu_attempt_02/smoke.json` (CPU, env `96a2bbce68883fdf`), `experiment_v1/smoke/colab_gpu_01/smoke.json` (Tesla T4, env `e74fb1dcf8112ca0`), `experiment_v1/environment/runtime_manifest.json`, run registry의 P2 4개 행.

**산출물:** 실행 entry point, smoke 로그, CPU/Colab lock, runtime manifest, 저장·복구 증빙.

**완료 조건:** 학습부터 패칭까지 작은 데이터로 연결되고 필수 검사가 통과한다. Debug checkpoint·feature·전처리는 본실험에 재사용하지 않는다.

> 03의 `train_lm.py`, `cache_activations.py`, `fit_probes.py`, `train_dictionary.py`, `evaluate_dictionary.py`, `run_patching.py`, `aggregate_results.py` 등은 구현할 인터페이스 명세다. 존재 여부와 CLI를 확인하기 전 실행 가능한 명령으로 간주하지 않는다. 기존 모듈을 사용하면 실제 명령과 대응 관계를 기록한다.

## 5. P3 — LM seed 0 파일럿과 진행 결정

**선행 조건:** P2 통과. **장소:** Colab 단일 GPU.

1. Seed 0 초기 checkpoint를 저장하고 고정 train shard 순서로 학습한다.
2. 최초 본실험 50 updates에서 tokens/sec와 peak VRAM을 측정한다. 이 구간은 본실험 예산에 포함한다.
3. 약 100k 토큰 경계 및 최종 update에서 일반/핵심 진단 validation을 수행한다. 학습 로그, best/last checkpoint를 저장한다.
4. 1M을 처음 넘은 완전한 update에서 최소 일반 val 답 CE checkpoint를 선택하고 gate를 판정한다.

| 결과 | 다음 작업 |
|---|---|
| 일반 val ≥99%, 세 진단 각각 ≥95% | 명목 1M 및 실제 최종 cursor/update 동결 → P4 |
| 미달, 마지막 4평가에서 얻은 3변화 중 ≥2개 CE 개선 ≥1e-4 nats | 동일 상태로 누적 3M까지 한 번 연장 후 같은 선택·gate 적용 |
| 1M 개선 조건 미충족 또는 3M에서도 미달 | 학습 중단 → CPU 데이터/구현 확인 → P11 중단 보고 |

- [x] 판정에 사용한 checkpoint hash, val 수치, 표본 수, 마지막 4평가, 연장 여부·근거를 저장했다.
- [x] 다른 gate 통과 checkpoint로 교체하거나 test로 선택하지 않았다.
- [x] Actual token 수, overshoot, 최종 cursor/update, 시간·VRAM을 저장했다.

**산출물:** seed 0 init/best/last, 학습 곡선, gate 결정 기록, 동결 예산 또는 중단 사유.

**주의:** 데이터/구현 수정이나 짧은 시퀀스 v1.1은 별도 변경 기록과 재검증이 필요하다. Gate 실패 상태로 해석 단계에 진입하지 않는다.

**판정 (2026-09-16): `failed` — 3M까지 실행·증빙 검증 후 행동 gate 미달로 중단. 체크박스는 기록 검증 완료이며 gate 통과가 아니다. [중단 보고](./experiment_v1/results/p3_stop_report.md), [증빙 감사](./experiment_v1/results/p3_evidence_verification.json). P4로 진행하지 않는다.**

## 6. P4 — LM seed 1·2 재현과 최종 행동 평가

**선행 조건:** P3 통과 및 예산 동결. **장소:** GPU, 결과 집계 CPU.

1. Seed 1과 2를 같은 train 순서·effective batch·최종 update까지 실행한다. 각 seed 초기 상태도 저장한다.
2. 각 seed에서 최소 val 답 CE checkpoint를 독립 선택하고 동일 gate를 적용한다. 실패 seed를 교체하지 않는다.
3. 모델 선택을 마친 뒤 고정 IID·진단·composition test를 최종 평가한다. 길이 GPU 평가는 P10에서 선택 실행한다.
4. Full-vocabulary/이진 지표, 확률질량, strata/macro/coverage 및 간단한 행동 기준선을 같은 target에서 집계한다.

- [ ] 모든 실행 seed의 행동 결과와 실패 이유가 있다.
- [ ] 해석 대상인 통과 LM 목록 G와 checkpoint hash가 확정되었다.

**산출물:** seed별 behavior 결과, 학습 곡선, 통과 모델 목록, 고정 checkpoint 목록.

**완료 조건:** 3개 LM의 결과를 숨김없이 보고하고 해석 대상만 gate 통과 모델로 한정한다.

## 7. P5 — READ activation cache와 full probe·기준선

**선행 조건:** P4. **장소:** 추출 GPU, probe CPU.

1. 통과 LM을 동결해 interp train/val/test의 고정 READ 위치에서 block 0 `h,u,m`을 수집한다.
2. Cache key와 라벨 join, answer 이전 위치, shape/dtype, missing/class support를 검증한다.
3. 해당 train만으로 전처리 통계를 구한다. Dictionary scalar 통계와 probe 차원별 통계를 구분해 저장한다.
4. Full h/u/m probe 및 좌표/random 기준선을 fit하고 학습 전 LM·token/position·shuffled-label 대조군을 실행한다.
5. IID, 현재≠과거, READ 변수 전이를 평가할 선택 절차를 고정한다. Test를 통한 수정은 하지 않는다.

- [ ] READ train/val/test 각 50k/10k/20k 위치 또는 실제 부족 사유가 기록되었다.
- [ ] Class support 미달 분석은 fitting하지 않고 NA/사유로 표시했다.
- [ ] Random 방향과 128후보 subset seed, probe lambda/threshold/tie-break가 저장되었다.

**산출물:** cache·라벨·전처리, full probe/기준선 계수와 선택 기록, support 표.

**완료 조건:** 의미 평가와 dictionary가 동일한 검증된 위치를 사용한다. Update 분석은 아직 시작하지 않는다.

## 8. P6 — READ SAE 학습

**선행 조건:** P5. **장소:** GPU.

1. 각 통과 LM의 block 0 READ h에 sparse seed 0, k=4/16을 각각 학습한다.
2. 최초 본실험 dictionary 100 updates에서 seconds/update·peak VRAM을 측정한다. 측정 구간은 5,000 updates에 포함한다.
3. Decoder gradient 투영 → clipping → optimizer → unit norm 순서를 적용하고 매 250 updates 저장·평가한다.
4. 5,000 updates를 완료한 뒤 k별 최소 val MSE checkpoint를 선택한다.

- [ ] 두 k를 모두 완료했고 label loss·early stopping·dead 재초기화를 추가하지 않았다.
- [ ] Mu/s, 초기화·sampler RNG, checkpoint, 실제 positions/draws/시간이 저장되었다.

**산출물:** G×2 SAE runs, 학습 곡선, best/last checkpoint·manifest.

**완료 조건:** 수치 오류 없이 명세 학습을 마쳤거나 오류·자원 중단이 명시되어 있다. 중단 run은 완료로 집계하지 않는다.

## 9. P7 — READ SAE 의미·fidelity·인과 평가

**선행 조건:** P6 완료. **장소:** probe/집계 CPU, 대체·개입 GPU.

1. Train ANOVA ranking과 val 선택으로 latent 단일/≤4 probe를 fit한다. Full/좌표/random과 비교하고 128후보 일치 분석도 수행한다.
2. NMSE/R2/EV, L0 분포, train 전체 재인코딩 dead 비율과 의미 지표를 계산한다.
3. Test_iid의 고정 READ 한 곳을 h_hat으로 바꾸어 원래 모델 대비 답 CE/정확도 변화를 평가한다.
4. READ 현재 값의 단일/≤4 feature J를 동결한다. Causal val로 구현과 random matching bin을 확정한다.
5. Changed/unchanged causal test에서 선택·identity·평균·전체 근사·full donor·random latent·좌표·random 방향 대조를 양방향으로 실행한다.
6. 전체 pair와 양쪽 원래 정답 subset, matched/unmatched 결과·coverage, 기억/합성 조건을 분리해 집계한다.

- [ ] 두 k, 두 feature 규모, 후보 수 대조와 모든 필수 인과 대조가 있다.
- [ ] Feature·bin·후보를 test 효과로 다시 선택하지 않았다.
- [ ] READ SAE의 핵심 표·그림과 실패/NA 사유가 확보되었다.

**산출물:** SAE semantic/fidelity/causal 결과, 선택 feature/계수, bin·patch norm·coverage, READ SAE 그림.

**완료 조건:** SAE 학습만이 아니라 의미·대체·인과 평가까지 끝난 후 P8로 진행한다.

## 10. P8 — READ Transcoder 학습과 동일 평가

**선행 조건:** P7. **장소:** 학습·패칭 GPU, probe·집계 CPU.

1. 같은 READ 위치의 u_0→m_0에 sparse seed 0, k=4/16을 각각 학습한다. 입력/출력 통계와 encoder/decoder 초기화를 분리한다.
2. 대응 SAE와 같은 position draw 순서를 사용하며 5,000 updates·val MSE checkpoint 규칙을 적용한다.
3. Full u/full m, m 좌표/random, TC latent를 비교하고 단일/≤4·128후보 분석을 수행한다.
4. TC fidelity와 m_hat 한 위치 대체를 평가한다. SAE의 h 복원과 TC의 m 예측 target을 그림·표에 표시한다.
5. P7과 같은 feature 동결·causal val matching·test 대조를 수행한다. 패칭은 `m_o+s_m*D_tc*delta_z`, residual skip은 원본 유지다.

- [ ] 입력 scale s_u를 출력 patch에 곱하지 않았다.
- [ ] 두 k의 의미·대체·인과 결과와 u/m 기준선이 모두 있다.

**산출물:** G×2 TC runs와 semantic/fidelity/causal 결과, READ SAE/TC 비교 그림.

**완료 조건:** 필수 TC 평가까지 완료했다. TC가 남은 상태에서 전체 실험 완료를 선언하지 않는다.

## 11. P9 — Sparse 초기화 민감도 반복

**선행 조건:** P8. **장소:** GPU + CPU.

1. LM seed 0의 같은 READ cache에서 sparse seed 1로 SAE/TC 각각 k=4/16, 총 4 runs를 추가한다.
2. 각 run에 학습·probe 선택·후보 수 대조·fidelity·근사 대체·인과 평가 절차를 동일하게 적용한다.
3. Sparse 초기화 반복과 LM seed 반복을 구분해 결과를 비교한다.

- [ ] 4개 추가 run의 성공/실패와 전체 평가가 기록되었다.
- [ ] 다른 dictionary의 동일 latent ID를 동일 feature로 간주하지 않았다.

**산출물:** 초기화 민감도 표·그림, 추가 run manifest.

**완료 조건:** 기본 READ 분석과 초기화 반복의 필수 결과가 모두 갖추어졌다.

## 12. P10 — 자원에 따른 2차 분석

**선행 조건:** P9 및 남은 자원 확인. 기본 READ보다 우선하지 않는다.

권장 순서는 update → block 1 → 길이 평가다. 각 분석의 실행/생략 결정을 기록한다.

- [ ] Update: 초기화 제외 update 위치의 별도 cache/전처리/SAE/TC와 입력·관계·결과 라벨 분석. AND/OR→XOR probe 전이 포함.
- [ ] Block 1: 선택 위치별 별도 dictionary·hook·실험 ID로 실행하고 final LN 이전 residual을 사용한다.
- [ ] Length: CPU에서 예약한 33~48블록 test를 기존 RoPE 설정으로 평가한다. 결과를 gate나 모델 재선택에 사용하지 않는다.
- [ ] 필요 시 m→m SAE를 별도 실험으로 추가하고 동일 target 조건의 TC 비교임을 표시한다.

**산출물:** 별도 config·결과, 실행/생략 사유와 추가 자원 사용량.

**완료 조건:** 선택 실험의 상태가 명확하다. 생략 자체는 기본 READ 완료를 막지 않는다.

## 13. P11 — 통계 집계와 최종 보고

**선행 조건:** 정상 경로는 P9 및 P10 결정 완료. 조기 중단 경로도 여기서 보고서를 남긴다.

1. Run registry를 대조해 실패 seed, 누락 필수 run, support 부족, 미매칭, 자원 중단을 확인한다.
2. LM seed별 값과 평균/최소/최대, sparse seed별 변동을 집계한다.
3. Sequence/origin cluster bootstrap 1,000회로 percentile 95% CI를 계산한다. Quota는 조건 내 재표집, 불가능 draw는 유효 수를 보고한다.
4. 아래 필수 표·그림과 재현 명령을 작성한다.

| 산출물 | 담을 내용 |
|---|---|
| `behavior.csv` | 답 CE/정확도 학습 곡선, 진단·holdout 성능, 기준선, gate |
| `semantic_metrics.csv` | full/좌표/random/SAE/TC, 단일/≤4, 후보 128/512, support·전이 |
| `fidelity_metrics.csv` | target 명시, NMSE/R2/EV–L0, dead 비율, 한 위치 대체 전후 행동 |
| `causal_metrics.csv` | changed margin/flip, unchanged 오류, 대조군, norm·matching coverage |
| `figures/` | 위 4종 비교와 LM/sparse seed·현재≠과거·전이 분포 |
| `run_registry.csv`, `report.md` | 모든 run 상태, 실패/생략, 해석 한계, CI, 환경/hash/명령 |

- [ ] 원본 config·환경 lock·코드/data/checkpoint hash와 실제 재현 명령이 연결된다.
- [ ] Test 선택 누출, 잘못된 표본 단위, 서로 다른 target의 단순 우열 해석이 없다.
- [ ] 필수 분석 전체 완료 / 행동 학습 단계 중단 / 자원으로 미완료를 정확히 구분했다.

**완료 조건:** [AGENTS.md §5](./AGENTS.md#5-단계-종료-판정)의 해당 종료 기준을 만족하는 보고서와 증빙이 있다. 부정적 결과도 결과로 보존한다.

## 14. 계산 예산과 운영 기록

| 작업 | 기본 예산 |
|---|---|
| LM | 3 seeds × 파일럿 동결 예산 1M 또는 3M; 최대 9M + seed별 overshoot |
| READ SAE+TC, sparse seed 0 | 통과 LM 수 G × 2도구 × 2k = 4G runs |
| LM seed 0 READ, sparse seed 1 | 2도구 × 2k = 4 runs 추가 |
| G=3 전체 필수 dictionary | 16 runs × 5k = 80k updates, 40.96M position draws |
| Update 선택 분석 | 최대 12 runs 추가; block 1은 위치별 별도 계상 |

예상 시간은 고정된 일수가 아니라 측정 처리량으로 갱신한다. LM은 `남은 토큰/tokens_per_second + 평가·저장 시간`, dictionary는 `남은 updates*seconds_per_update + 평가 시간`으로 산정한다. CPU, GPU 학습, activation 추출, validation, probe, bootstrap, patching 비용은 각각 기록한다.

각 단계의 실행 기록에는 아래 항목을 남긴다.

```text
phase / run_id / status(pending|running|passed|failed|paused|skipped)
시작·종료 시각 / config·code·input hash / 환경 ID / seed
실제 token·position·draw 수 / peak memory / 처리량·소요 시간
검증 증빙 경로 / 선택 checkpoint / 실패·생략 사유
재개 checkpoint·cursor / 다음 작업 / 실제 실행 명령
```

자원이 부족하면 영속 checkpoint에서 재개한다. Width/k/token budget을 줄여 완료 처리하지 않는다. 필수 READ 결과가 끝난 뒤에만 선택 분석에 예산을 배정한다.

## 15. v1.1 4-block 실행 분기 (2026-09-16)

사용자 지시로 깊이와 초기화를 변경한 별도 실험이며 [변경 규격](./experiment_v1_1/CHANGELOG.md)을 따른다. P1은 동일 immutable 데이터 재사용·재검산, P2는 새 4-block CPU/GPU smoke를 통과했다. P3은 1M 개선 조건으로 누적 3M까지 연장한 뒤 모든 행동 gate에 미달하여 `failed`로 종료했다. 일반 validation 58.90%, 진단 58.98% / 59.96% / 55.47%; 407 updates / 3,004,531 tokens. P4 및 표현 분석은 진행하지 않는다. [중단 보고](./experiment_v1_1/results/p3_stop_report.md), [증빙 감사](./experiment_v1_1/results/p3_evidence_verification.json). v1.0 결과는 위 원래 기록에 그대로 보존한다.

## 16. v1.2 4-block / 16M 계획 (2026-09-17)

[새 설계](./experiment_v1_2/DESIGN.md)는 기존 4-block 구조를 유지하고 fresh seed 0을 고정16M까지 학습한다. 기존3M train prefix와 평가 split을 보존하고 새 train을 추가한다. 설계만 완료되었으며 새 P1/P2/P3는 모두 pending이다. 본 분기의 예산·gate 시점·shard 예외·저장 규칙은 새 설계를 따르며, 실제 CPU 검증과 GPU smoke 증빙 후에만 본학습을 시작한다.
