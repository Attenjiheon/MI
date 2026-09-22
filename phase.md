# 실험 실행 계획서

작성일: 2026-09-10 · 최종 갱신: 2026-09-22
기준: [01_experiment_design.md](./01_experiment_design.md), [03_experiment_spec.md](./03_experiment_spec.md)  
실행 규칙: [AGENTS.md](./AGENTS.md)

이 문서는 어떤 작업을 어떤 순서로 수행하고, 어떤 증빙이 있어야 다음 단계로 넘어갈지를 정한다. 체크박스는 실행 결과를 확인한 후에만 표시한다. 문서 작성 자체로 실험을 수행하거나 기존 코드·데이터의 검증 완료를 선언하지 않는다.

## 현재 활성 버전 — v1.4 (2026-09-22)

**P4 완료. 다음 단계는 P5 — 세 동결 LM의 block 0 READ activation cache·probe다.**
Seed 0·1·2의 학습·validation gate·checkpoint 선택 동결과 전체 frozen test 반환 감사를 마쳤다.
세 seed 모두 validation gate를 통과해 해석 대상은 **G=3**이다. P5 진입 조건은 충족했지만,
P5 실행 준비·cache 추출·probe fitting의 완료 증빙은 아직 없다. 전체 실험은 미완료다.

| 단계 | v1.4 현재 상태 | 근거 또는 다음 조건 |
|---|---|---|
| P1 | 완료 | [corpus 감사](experiment_v1_4/results/audit_20260921_01/completion.json) |
| P2 | 완료 | [CPU/GPU smoke 상태](experiment_v1_4/P2_STATUS.md), [r3 GPU 반환 검증](experiment_v1_4/results/p3_audit_20260921_01/gpu_smoke/verification.json) |
| P3 | **완료·passed** | [seed 0 반환 감사](experiment_v1_4/results/p3_audit_20260921_01/completion.json) |
| P4 | **완료·passed** | [최종 반환 감사](experiment_v1_4/results/frozen_test_audit_20260922_01/completion.json), [P4 상태](experiment_v1_4/P4_STATUS.md) |
| P5 | **다음 단계·진입 가능·미실행** | [동결 모델 3개](experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json)의 block 0 READ cache·full probe·기준선. 구체적 착수 순서는 §7.1 |
| P6–P7 | 미실행·P5 대기 | 세 LM × k=4/16의 READ SAE 학습 6 runs 및 의미·fidelity·인과 평가 |
| P8 | 미실행·P7 대기 | 같은 세 LM × k=4/16의 필수 READ TC 학습 6 runs 및 동일 평가 |
| P9 | 미실행·P8 대기 | LM seed 0에서 SAE/TC × k=4/16, sparse seed 1의 4 runs 및 전체 평가 |
| P10 | 미실행·선택 분석 | 필수 READ 분석 및 sparse 초기화 반복 완료 후 결정 |
| P11 | 최종 집계 미완료 | 단계별 기록은 누적하며 전체 필수 분석은 아직 남아 있음 |

## 1. 전체 순서와 우선순위

`P0 규격·현황 정리 → P1 CPU 검증 → P2 구현·환경 smoke test → P3 LM 파일럿 → P4 LM 재현·행동 평가 → P5 READ cache·probe → P6 READ SAE → P7 READ SAE 평가·개입 → P8 READ TC 학습·평가·개입 → P9 sparse seed 반복 → P10 선택 분석 → P11 최종 집계`

P3에서 행동 gate에 실패하면 동결된 중단 절차로 분기한다. 필수 READ가 미완료이면 P10을 시작하지 않는다. P11의 기록·중간 집계는 각 단계에서 누적하며, 중단 시에도 실패 보고서를 작성한다.

| 범위 | 우선순위 |
|---|---|
| 통과 LM별 block 0 READ probe·SAE·TC, k=4/16, fidelity·의미·인과 대조 | 필수 |
| LM seed 0 READ의 두 도구·두 k, sparse seed 1 | 필수 초기화 반복 |
| Update, block 1, 길이 외삽 GPU 평가, m→m SAE | 자원 확인 후 별도 실험 |
| Crosscoder, 변수 8개, 대규모 모델/seed sweep | 기본 범위 제외 |

## 2. P0 — 규격과 현재 산출물 정리

현재 본 실험은 v1.4다. 루트 01–03은 v1.4 계약을 통합했으며 과거 실행 규격과
체크박스는 [원문 snapshot](archive/specifications/pre_v1_4_integration/phase.md)에 보존했다.
동결 설계·config·corpus·완료 증빙이 기준이며 미실행 P5 이후 config는 착수 시 별도로 고정한다.
문서 통합 자체를 새로운 실험 단계 통과로 세지 않는다.

## 3. P1 — CPU 생성기·코퍼스 검증 (완료)

**선행 조건:** v1.4 설계 및 생성 계약 동결. **장소:** CPU.

1. 02 §15와 03 §5.3의 parser·768전이·replay·index·누출·quota 검증을 수행한다.
2. Historical sequence/READ-prefix registry를 포함한 fresh split 중복 거부를 확인한다.
3. v1.3 train prefix bytes를 보존하고 명목 64M의 최초 완전 update까지 확장한다.
4. Manifest·실제 token/sequence 수·RNG·거부율·파일별 hash를 남긴다.

- [x] 활성 `rebuild_01`의 전체 독립 감사와 시도 상한 재현 검증 통과.
- [x] 133 shards / 537,536 sequences / 64,005,751 tokens / 8,399 updates 확인.
- [x] 최초 거부 corpus와 수정본을 분리하고 활성 입력을 동결.

**완료 증빙:** [P1 상태](experiment_v1_4/P1_STATUS.md),
[독립 감사](experiment_v1_4/results/audit_20260921_01/completion.json).
최초 거부 root는 학습하지 않는다. 삭제된 과거 payload는 디스크 정리 기록을 따른다.

## 4. P2 — 구현·환경·CPU/GPU smoke (완료)

**선행 조건:** P1 통과. **장소:** 로컬 CPU 및 Colab GPU.

- [x] 12×256 LM 9,485,312 parameters, read4 accumulation·mask·RoPE·전 층 hook 검증.
- [x] SAE/TC k=4/16 debug 학습·probe·patch 경로 및 완전 update resume 검증.
- [x] 실제 Tesla T4 GPU 반환물·입력/code hash·환경 lock 확인.
- [x] Microbatch 16 / effective batch 64 확정. 본학습 전 보고 코드 r3 smoke 확인.

**완료 증빙:** [P2 상태](experiment_v1_4/P2_STATUS.md),
[GPU 검증](experiment_v1_4/evidence/gpu_smoke_20260921_01/verification.json),
[r3 검증](experiment_v1_4/results/p3_audit_20260921_01/gpu_smoke/verification.json).
Debug checkpoint는 본실험에 재사용하지 않는다. P5 이후 본 cache·probe·dictionary 완료와 구분한다.

## 5. P3 — 완료·seed 0 행동 gate 통과

v1.4는 별도 동결 설계의 12×256/read4/64M 계약을 따른다. 자동 예산 연장은 허용하지 않는다.
P1·P2 및 r3 보고 보완 뒤 64,005,751 tokens / 8,399 updates / cursor 537,536을 완료했다.
Select 규칙에 따라 update 7,983 (60,801,363 tokens)을 선택했고, 해당 checkpoint에서
일반/legacy/first/repeat/group/coverage gate 전부 통과했다.

- [x] 원본 ZIP·8,449개 내부 checksum·runtime/config/data/환경 lock을 검증했다.
- [x] Init/10 milestones/last, optimizer/RNG/cursor 및 32M 재개 증빙을 검증했다.
- [x] Select의 전역 최소 first macro CE 및 near-tie 선택 규칙과 선택 checkpoint가 일치한다.
- [x] 일반 READ 99.9277%, legacy 99.9023%/99.9023%/100%, first macro 99.8140%, repeat 100%.
- [x] 모든 group 기준·quota·coverage와 CI/확장 보고를 확인했다. Gate/test 재채점은 하지 않았다.

[감사 보고서](experiment_v1_4/results/p3_audit_20260921_01/REPORT.md),
[완료 manifest](experiment_v1_4/results/p3_audit_20260921_01/completion.json),
[동결 seed 0](experiment_v1_4/results/p3_audit_20260921_01/frozen_seed0.json)를 따른다.
P3 완료 커밋은 `d6b91ae`이며 원격 main 반영을 확인했다.

## 6. P4 — seed 1·2 재현 및 최종 test (2026-09-22 반환 감사 완료)

**선행 조건:** 검증된 seed 0 gate 통과와 동결 예산. 현재 충족했다.

**상태: 완료.** 아래 1–7은 완료한 절차의 기록이며 재실행 지시가 아니다.

1. Seed 1·2 실행기·Colab 노트북·입력 번들 및 검증 절차를 준비한다.
   `--stage p4`는 감사된 seed 0 증빙과 동결 입력을 확인한 후 seed 1·2만 허용한다.
   [P4 전용 전달물](experiment_v1_4/P4_STATUS.md)을 사용하며 기존 r3 노트북에서 seed 숫자만 바꾸지 않는다.
2. 같은 아키텍처·read4·optimizer/LR·effective batch·동결 train 순서로 seed 1과 2를 각각
   fresh initialization에서 학습한다. 각 seed는 실제 64,005,751 tokens / 8,399 updates를 소비한다.
3. 각 seed의 지정 milestone 후보 중 **select first-member 42-cell macro answer CE 전역 최소**를
   선택한다. 최소와 1e-4 nats 이내는 select/general answer CE, 이른 update 순으로 고른다.
4. 각 seed의 선택 checkpoint에서 동일 validation gate를 한 번 평가한다. Gate로 checkpoint를
   다시 고르지 않고, 실패 seed도 보존하며 다른 seed로 대체하지 않는다.
5. 세 seed 전체의 학습·checkpoint 선택·validation gate 결정을 동결한다.
6. **학습한 seed 전체(실패 seed 포함)**에 frozen test를 한 번 평가하고 최종 행동 결과를 보고한다.
   Test는 추가 합격 gate가 아니다. Test 결과로 checkpoint·설정·threshold를 바꾸지 않는다.
7. 반환 증빙의 checksum, 학습량, 선택/gate/test 절차, 환경과 checkpoint를 검증하고
   seed별 결과·실패 사유·통과 모델 목록과 hash를 registry에 기록한다.

- [x] Seed 1·2 실행 전달물과 로컬 사전 검증이 준비됐다. [준비 당시 증빙](experiment_v1_4/results/p4_preparation_r1/verification.json).
- [x] Seed 1의 학습·select 선택·one-time validation gate 반환 증빙을 검증했다.
- [x] Seed 2의 학습·select 선택·one-time validation gate 반환 증빙을 검증했다.
- [x] 전체 학습·validation 결정 및 checkpoint hash를 동결했다. [동결 기록](experiment_v1_4/results/p4_audit_20260921_01/validation_freeze.json).
- [x] 동결된 세 seed의 최종 test 전용 노트북·번들과 로컬 사전 검증을 준비했다. [준비 당시 증빙](experiment_v1_4/results/frozen_test_preparation_r1/verification.json).
- [x] 학습한 모든 seed의 frozen test 최종 보고와 반환 증빙 검증을 마쳤다. [완료 증빙](experiment_v1_4/results/frozen_test_audit_20260922_01/completion.json).
- [x] 실패 seed 없음과 해석 대상 seed 0·1·2를 확정하고 Git에 반영했다. [모델 목록](experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json).

**P4 완료 조건:** 위 학습·평가·동결·반환 검증·기록이 모두 끝났다.
Test 점수가 높다는 이유만으로 P4를 완료 처리하지 않는다.

**P5 진입 조건:** P4 완료에 더해 **LM seed 0·1·2 중 최소 2개가 validation gate를 통과**해야 한다.
현재 seed 0·1·2가 모두 통과했으며, [최종 test 반환 감사](experiment_v1_4/results/frozen_test_audit_20260922_01/REPORT.md)까지 완료해 P5 진입 조건을 충족했다. P5 자체는 아직 미실행이다. P4를 완료해도 전체 실험 완료는 아니며 필수 READ probe·SAE·TC·인과 평가와
LM seed 0의 sparse seed 1 반복이 남는다.

**최종 행동 결과:** 정확도 %, first/repeat는 42-cell macro다. Test로 모델을 재선택하지 않는다.

| LM seed | 선택 update | 일반 READ test | First test | Repeat test | P5 대상 |
|---|---:|---:|---:|---:|---|
| 0 | 7,983 | 99.9410 | 99.9442 | 100 | 포함 |
| 1 | 7,983 | 99.9472 | 99.9442 | 100 | 포함 |
| 2 | 8,399 | 99.9659 | 99.9628 | 100 | 포함 |

원본 ZIP 외부 hash·115개 내부 checksum, 21개 평가의 동결 입력·환경·runtime 코드,
저장된 raw 집계·CI 재계산 및 153,510 targets의 metadata 대조가 통과했다.
모델 추론과 gate/test 재채점은 하지 않았다. Legacy·composition·기준선·CI·오류 분석은
[최종 보고서](experiment_v1_4/results/frozen_test_audit_20260922_01/REPORT.md)를 따른다.
P4 완료 커밋은 `6f7d2e9`이며 원격 main 반영을 확인했다.

## 7. P5 — READ activation cache와 full probe·기준선

**선행 조건:** P4. v1.4는 LM seed 최소 2개의 validation gate 통과도 필요하다. **장소:** 추출 GPU, probe CPU.

**v1.4 현재 상태:** 선행 조건 충족, 대상 seed 0·1·2, G=3. P5 자체는 미실행이며
아래 완료 체크박스는 유지한다. 동결 입력과 준비 순서는 §18.3을 따른다.

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

### 7.1 P5 — 다음 착수 단계: block 0 READ cache·probe (미실행)

**선행 조건:** P4 완료 및 최소 두 LM seed의 validation gate 통과. 세 seed 모두 충족했다.
상세 실행 규격은 03 §2·§6–7·§10·§12–14, 01 §8, 02 §10·§14를 읽고 적용한다.
아래 항목은 준비·실행 계획이며 완료 증빙을 대신하지 않는다.

1. [동결 모델 목록](experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json)의
   seed 0·1·2 checkpoint와 hash를 확인한다. 입력 파일은
   `experiment_v1_4/frozen_test_r1/checkpoints/seed{0,1,2}.pt`이며,
   기존 [validation 동결](experiment_v1_4/results/p4_audit_20260921_01/validation_freeze.json)을 유지한다.
2. `corpus_rebuild.json`의 활성 `data/language_v1_4/rebuild_01`에서 interpretation split과
   예약 READ 위치·라벨·실제 quota를 확인한다. 완료한 behavior test와 interpretation test의 역할을 구분한다.
3. `interp_v1_4/`, `tests_v1_4/`의 cache·probe 구현을 원문과 대조하고 미구현·미검증 항목을 확인한다.
   v1.4 폭·hook·전처리·선택 규칙에 맞는 P5 config와 실행/반환 schema를 실행 전에 고정한다.
4. 별도 debug 입력으로 key/label join, 정답 이전 위치, shape/dtype, 누출 방지,
   support·전처리 및 저장/재개를 검증한다. Colab 작업은 전용 `.ipynb`와 입력 번들로 전달한다.
5. 검증된 실행기로 세 LM의 block 0 READ `h,u,m` cache를 추출하고,
   이 절의 full probe·좌표/random 및 필수 대조군을 수행한다. 전처리 fitting은 해당 train만 사용한다.
6. 실제 반환 cache·probe 결과·선택 기록을 감사한 뒤 P5 상태와 registry를 갱신한다.
   P5 완료 전 P6 SAE 본학습을 시작하지 않는다.

- [ ] P5 상태 문서·동결 config·실행/반환 계약과 debug 검증을 준비했다.
- [ ] Colab 노트북·입력 번들과 실제 GPU 실행 증빙을 확보했다.
- [ ] 세 LM의 READ cache·metadata join·quota·class support·train 전처리를 검증했다.
- [ ] Full probe·기준선·대조군·validation 선택 및 이 절의 완료 조건을 검증했다.
- [ ] 전체 12개 층 full probe 진단과 층별 support·결과·실패 사유를 보고했다.

**현재 없는 증빙:** 위 P5 전용 준비 기록, 본 cache, 본 probe 결과와 반환 감사.
P2 debug smoke와 P4 행동 평가 통과를 P5 완료로 대신하지 않는다.

전체 12개 층 full probe 진단은 동결 설계의 필수 진단이다. 필요한 hook·cache 범위를
P5 config에 고정한다. Block 1 sparse 학습·개입의 P10 선택 범위와 구분한다.

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
3. 동결 general behavior test의 고정 READ 한 곳을 h_hat으로 바꾸어 원래 모델 대비 답 CE/정확도 변화를 평가한다.
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
| `semantic_metrics.csv` | full/좌표/random/SAE/TC, 단일/≤4, 후보 128/256/512, support·전이 |
| `fidelity_metrics.csv` | target 명시, NMSE/R2/EV–L0, dead 비율, 한 위치 대체 전후 행동 |
| `causal_metrics.csv` | changed margin/flip, unchanged 오류, 대조군, norm·matching coverage |
| `figures/` | 위 4종 비교와 LM/sparse seed·현재≠과거·전이 분포 |
| `run_registry.csv`, `report.md` | 모든 run 상태, 실패/생략, 해석 한계, CI, 환경/hash/명령 |

- [ ] 원본 config·환경 lock·코드/data/checkpoint hash와 실제 재현 명령이 연결된다.
- [ ] Test 선택 누출, 잘못된 표본 단위, 서로 다른 target의 단순 우열 해석이 없다.
- [ ] 필수 분석 전체 완료 / 행동 학습 단계 중단 / 자원으로 미완료를 정확히 구분했다.

**완료 조건:** [AGENTS.md §5](./AGENTS.md#5-단계-종료-판정)의 해당 종료 기준을 만족하는 보고서와 증빙이 있다. 부정적 결과도 결과로 보존한다.

## 14. 계산 예산과 운영 기록

v1.4는 seed당 명목 64M,
실제 64,005,751 tokens / 8,399 updates이며 **세 seed 합계 192,017,253 training tokens를
이미 소비했다.** 각 seed가 동일한 동결 stream을 한 번씩 소비했고 추가 LM 학습은 계획하지 않는다.
Sparse 예산은 별도 변경 없이 해당 동결 규격을 따른다.

| 작업 | v1.4 확정 범위·예산 및 상태 |
|---|---|
| LM | 완료: 3 seeds × 64,005,751 tokens = 192,017,253 tokens; 각 8,399 updates |
| Frozen behavior test | 완료: 3 seeds × 7 suites = 21개 평가; 총 153,510 READ targets. 기록된 평가 시간 합계 215.32초이며 학습 예산과 별도 |
| READ SAE+TC, sparse seed 0 | 미실행: G=3 × 2도구 × 2k = 12 runs |
| LM seed 0 READ, sparse seed 1 | 미실행: 2도구 × 2k = 4 runs 추가 |
| G=3 전체 필수 dictionary | 미실행: 16 runs × 5k = 80k updates, 40.96M position draws |
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
