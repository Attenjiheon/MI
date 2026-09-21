# 실험 세션 진입 규칙

이 파일은 프로젝트 세션을 시작할 때 가장 먼저 읽는 **라우터**다. 세부 실험값을 요약해 두는 문서가 아니며, 실제 규격은 아래 원문에서 확인한다.

## 1. `Pn단계를 수행하자` 요청을 받았을 때

1. 이 파일을 끝까지 읽는다.
2. [phase.md](./phase.md)의 `전체 순서와 우선순위`, 요청받은 `Pn` 절, 바로 앞 단계의 완료 조건을 읽는다. 계산 자원이나 실행 기록이 관련되면 `계산 예산과 운영 기록`도 읽는다.
3. 아래 표에 따라 필요한 원문 절을 읽는다. 구현값·표본 수·수식·gate를 기억에 의존해 적용하지 않는다.
4. `experiment_v1/` 아래의 해당 단계 상태 문서, config, manifest, run registry, README와 실제 산출물을 확인한다. 체크박스나 파일 존재만으로 통과를 가정하지 않는다.
5. 작업 전에 현재 단계의 선행 조건, 이번 목표, 완료 조건, 아직 없는 증빙을 짧게 정리한다.
6. 아직 충족되지 않은 첫 항목부터 실행하고 검증한다. 단계가 끝난 뒤에만 상태·체크박스·registry를 증빙 경로와 함께 갱신한다.

단계 번호가 모호하거나 현재 상태와 맞지 않으면, 안전한 읽기·검증은 먼저 수행하되 규격 변경이나 비용이 큰 실행 전에 불일치를 알린다.

## 2. 문서 역할과 우선순위

| 문서 | 역할 |
|---|---|
| [README.md](./README.md) | 저장소 전체 구조, 버전 3종 세트 규칙, 버전 계보, 경로 이동 금지 원칙 |
| [phase.md](./phase.md) | 실행 순서, 선행 조건, 단계별 산출물과 완료 gate |
| [01_experiment_design.md](./01_experiment_design.md) | 연구 목적, 가설, 해석 범위와 보고 원칙 |
| [02_language_and_corpus.md](./02_language_and_corpus.md) | 언어 문법, 토큰 ID, 실행 의미, 생성·split·metadata 규격 |
| [03_experiment_spec.md](./03_experiment_spec.md) | 모델·학습·probe·SAE·TC·인과 평가의 상세 실행 규격 |
| `experiment_v1*/` | 각 버전의 config, 상태, 실행 기록과 산출물. 현재 활성 버전은 아래 §8–10과 [README.md](./README.md) §3을 따른다 |

규격 충돌의 적용 우선순위는 **사용자의 최신 명시 지시 → 03의 명시적 변경·상세 규격 → 02의 언어·코퍼스 규격 → 01의 연구 설계**다. 실행 순서와 단계 gate는 `phase.md`를 따르되, phase가 상세 규격을 바꾸는 근거로 쓰이지는 않는다. 충돌을 발견하면 조용히 절충하지 말고 해당 문서와 항목을 기록한다. 이 파일은 원문을 대체하지 않는다.

03이 01을 명시적으로 바꾼 핵심 사항은 다음 두 가지다.

- Transcoder는 선택 확장이 아니라 필수 READ 분석이며 READ SAE 뒤에 수행한다.
- LM 데이터는 CPU에서 미리 생성한 shard를 모든 LM seed가 같은 순서로 한 번씩 소비한다. GPU에서 새로 생성하지 않는다.

## 3. 단계별 필수 읽기

아래 범위는 최소치다. 대상 코드나 결과가 다른 절을 참조하면 그 절까지 이어서 읽는다.

| 단계 | 먼저 읽을 원문 |
|---|---|
| P0 규격·현황 | 03 §1, §12–14; 01 §4–5, §11–12; `experiment_v1/P0_STATUS.md`와 기존 config/README |
| P1 CPU 코퍼스 | 02 §2–16; 03 §5; 기존 corpus 코드·test·manifest·audit 결과 |
| P2 구현·smoke | 03 §2–4, §6–12, §14; 02 §10, §14–16; `experiment_v1/configs/`와 환경/데이터 README |
| P3 LM 파일럿 | 03 §3–5, §10.1, §12–14; 01 §5–7; 02 §14; P1/P2 증빙 |
| P4 LM 재현·행동 | 03 §3–4, §10.1, §12–14; 01 §5–7; P3 gate 기록 |
| P5 READ cache·probe | 03 §2, §6–7, §10, §12–14; 01 §8; 02 §10, §14; 통과 LM 목록 |
| P6 READ SAE 학습 | 03 §8, §12–14; 01 §9; P5 cache·전처리 증빙 |
| P7 SAE 평가·개입 | 03 §7–8, §10–14; 01 §8–10; P6 선택 checkpoint |
| P8 READ TC | 03 §7, §9–14; 01 §8–10; P5 cache와 P7 평가 절차 |
| P9 sparse seed 반복 | 03 §7–11, §13–14; P6–P8의 확정 config·선택 규칙·결과 schema |
| P10 선택 분석 | phase P10에서 실제 선택한 분석을 먼저 확정한 뒤, update는 02 §4·§10 및 03 §2·§6–11, block 1은 03 §2·§6–11, length는 02 §5·§12 및 03 §4–5·§10, `m→m` SAE는 03 §8–10을 읽는다 |
| P11 최종 집계 | 03 §10–14; 01 §11–13; 모든 status/manifest/registry와 결과 schema |

## 4. 모든 단계에서 지킬 핵심 규칙

1. **순서와 gate:** CPU 데이터 검증 → 구현 smoke → LM 행동 gate → LM 동결 → 표현 분석 → 인과 평가 순서를 지킨다. 필수 READ 분석이 끝나기 전에 P10을 시작하지 않는다.
2. **누출 방지:** 상태·정답 metadata는 분석에만 사용한다. LM forward에는 token ID와 padding mask만 넣고, LM/SAE/TC loss에 상태 라벨을 넣지 않는다. TC supervision은 원래 MLP 출력이다.
3. **선택 분리:** train은 fitting, validation은 사전 정의된 선택, test는 동결된 절차의 최종 보고에만 사용한다. Test 결과로 checkpoint, feature, threshold, `k`, matching 규칙을 다시 고르지 않는다.
4. **동결과 실패 보존:** 행동 gate 뒤 LM·데이터·예산을 동결한다. SAE/TC 결과가 나쁘다는 이유로 이를 바꾸지 않으며 실패 seed를 대체하거나 숨기지 않는다.
5. **범위:** 필수 범위는 gate 통과 LM의 block 0 READ probe·SAE·TC, `k={4,16}`, 후보 수 대조, 근사 대체와 인과 평가이며 LM seed 0의 sparse seed 1 반복을 포함한다. Update, block 1, length GPU 평가, `m→m` SAE는 P10 선택 분석이다. Crosscoder와 변수 8개 확장은 기본 범위 밖이다.
6. **규격 변경:** 미정 선택은 관련 test를 보기 전에 config에 고정한다. 예산·분포·quota·평가법을 바꾸면 이유, 영향, 새 버전, config/hash를 남기고 기존 실험과 구분한다.
7. **증빙 우선:** 검증 명령, 실제 표본/token/draw 수, seed와 RNG state, 환경 ID, config/code/data/checkpoint hash, 선택 checkpoint, 실패·생략 사유를 남긴다. 체크박스는 이 증빙을 확인한 뒤에만 갱신한다.
8. **재개와 보존:** 기존 데이터와 결과를 덮어쓰지 않는다. checkpoint의 cursor·optimizer·RNG·평가 경계를 확인해 마지막 완전 저장 update부터 재개하고, 환경이 달라지면 별도 환경 ID로 기록한다.

## 5. 단계 종료 판정

- 단계의 완료는 [phase.md](./phase.md)의 해당 완료 조건과 [03_experiment_spec.md](./03_experiment_spec.md)의 gate를 모두 만족해야 한다.
- 행동 gate 실패, quota 미달, 수치 오류, 자원 중단은 성공이 아니다. 원문에 정한 분기대로 중단·재개 대기·별도 버전으로 표시한다.
- 행동 gate 통과 모델이 없으면 행동 학습 단계에서 종료 보고만 작성하고 SAE/TC 가설을 검증했다고 쓰지 않는다.
- 필수 TC 또는 sparse 초기화 반복이 남아 있으면 전체 실험 완료로 표시하지 않는다.
- 최종 해석은 probe의 선형 접근성, sparse feature의 사후 감독 성능, 지정 위치 패칭의 개입 증거 범위를 넘어서 주장하지 않는다.

## 6. 단계 완료 후 Git 업데이트

각 phase의 완료 조건과 gate를 모두 확인하고 상태 문서·manifest·run registry 등 관련 기록을 갱신한 뒤에는 반드시 Git을 업데이트한다.

1. `git status`와 diff를 검토해 해당 phase의 코드, 설정, 문서, 증빙 산출물만 포함하고 비밀정보·로컬 환경 파일·무관한 변경은 제외한다.
2. 단계와 변경 내용을 식별할 수 있는 메시지(예: `Complete P3 LM pilot`)로 커밋한다.
3. 커밋을 원격 저장소의 `main` 브랜치에 push한다.
4. 로컬 `HEAD`와 원격 `main`의 commit SHA가 일치하는지 확인한다.
5. push가 실패하면 단계를 완료한 것으로 숨기지 말고 실패 원인을 기록한 뒤 해결하여 다시 push한다.

## 7. Colab 실행 파일 전달

- Colab에서 사용자가 실행할 작업은 실행 순서와 설명을 포함한 주피터 노트북(`.ipynb`) 파일로 전달한다. 셸 명령이나 Python 스크립트만 전달하는 것으로 대체하지 않는다.
- 노트북에는 입력 업로드·checksum 검증, 의존성 설치·환경 기록, 실행·저장·재개, 결과 다운로드 절차를 포함한다. 필요한 입력 번들도 함께 제공한다.
- 노트북 준비·로컬 검증과 Colab 실제 실행을 구분하고, 내려받은 실행 증빙을 검증하기 전에는 해당 phase를 완료 처리하지 않는다.

## 8. v1.1 4-block 보존 분기 (2026-09-16)

- 2026-09-16 사용자 지시로 시작한 **4-block v1.1** 보존 분기다. 이 버전을 재현·감사할 때는 `experiment_v1_1/CHANGELOG.md`를 먼저 읽고 `experiment_v1_1/`, `interp_v1_1/`, `tests_v1_1/`를 사용한다.
- 표의 v1.0 상세 규격 중 block 수·파라미터 수·깊이 기반 초기화는 위 변경 규격이 대체한다. 데이터·예산·선택/gate는 유지한다. 신규 CPU/GPU smoke 통과 전 GPU 본학습을 시작하지 않는다.
- `experiment_v1/`와 `interp/`는 2-block 실패 실험 재현용이다. 해당 상태·설정·checkpoint를 신규 실험으로 덮어쓰거나 재사용하지 않는다. 과거 결과 감사는 그 버전의 규격·코드를 따른다.

## 9. v1.2 16M 설계 진입 (2026-09-17)

- 신규 16M 작업은 `experiment_v1_2/DESIGN.md`와 `design_config.json`을 먼저 읽는다. 설계 문서는 당시 계획의 불변 기록이다. 현재 실행 상태는 `experiment_v1_2/P1_STATUS.md`, `P2_STATUS.md`, `P3_STATUS.md`를 확인한다. 신규 실행은 `interp_v1_2`와 `tests_v1_2`를 사용하며 기존 v1.1 CLI를 16M 실행기로 사용하지 않는다.
- 위 §8은 v1.1 재현·감사 경로다. v1.2의 예산·데이터 확장·선택·저장 변경은 새 설계를 따르며, 기존 데이터·실패 결과를 덮어쓰지 않는다.

## 10. v1.3 상태 전이 진단 설계 (2026-09-17)

- 신규 v1.3 작업은 `experiment_v1_3/DESIGN.md`와 `design_config.json`을 먼저 읽는다. v1.2는 감사된 행동 gate 실패로 보존하고, v1.0–v1.2 validation은 v1.3 선택·gate에 재사용하지 않는다.
- v1.3은 P1·P2와 6-cell 8M pilot을 완료했고 `wide4_read4`를 winner로 승격한 상태다. 현재 진행 기준은 `experiment_v1_3/P3_STATUS.md`다. seed 0 32M confirm·gate는 미실행이며, 검증된 gate 통과 전에 seed 1·2, frozen test, P5 이후를 시작하지 않는다.
- 여섯 pilot cell을 임의로 줄이거나 추가하지 않고, 새 select/gate/test 분리와 first/repeat 42-cell quota를 지킨다. Pilot winner 하나만 사전 규칙으로 승격하며 gate나 test를 보고 checkpoint·설정·threshold를 다시 고르지 않는다.

## 11. v1.4 P3 생성·검증 진입 (2026-09-20)

- 사용자 요청으로 `next_architecture_proposal.json`의 `deepwide12_read4`를 v1.4로 준비 중이다. `experiment_v1_4/DESIGN.md`, `design_config.json`, `corpus_rebuild.json`, `P1_STATUS.md`, `P2_STATUS.md`, `P3_STATUS.md`를 먼저 읽는다. 구현과 검증은 `interp_v1_4/`, `tests_v1_4/`를 사용한다.
- 최초 `data/language_v1_4/`의 64M 생성물은 historical READ-prefix 누락으로 거부됐다. 삭제·이동·학습하지 않는다. 수정본의 활성 경로는 `corpus_rebuild.json`의 `active_data_root`이며, 수정본 audit 통과 전에는 config 동결이나 GPU 학습을 허용하지 않는다.
- 진행 중인 로컬 준비는 `experiment_v1_4/results/local_preparation/state.json`과 단계별 로그를 확인한다. DRAFT notebook은 실행용이 아니다. Local preparation, 실제 GPU smoke, P3 학습·gate 반환 증빙의 검증을 구분한다.
- v1.3 confirm은 v1.4 설계에 인용된 반환 증빙상 행동 gate 실패다. v1.3 test를 열거나 seed 1·2를 추가하지 않는다. v1.4도 seed 0 gate 검증 통과 전에는 replication/test/표현 분석을 시작하지 않는다.

## 12. v1.4 CPU 감사 완료 (2026-09-21)

- §11의 생성 중 상태는 과거 기록이다. `rebuild_01`의 전체 독립 감사, 27 tests, frozen-input CPU smoke와 실제 CLI persistent-index resume가 통과했다. 현재 증빙은 `experiment_v1_4/results/audit_20260921_01/REPORT.md`와 `completion.json`이다.
- 원본 생성 코드 snapshot/hash와 감사 보완 코드를 구분한다. 시도 상한 결함은 수정했고 기존 10,752 targets의 stream 재현으로 최대 58,433회임을 확인했다. Corpus와 동결 manifest/config는 수정하지 않았다.
- P2 GPU smoke 및 P3는 미실행이다. CPU 통과를 전체 P2/P3 완료로 표시하지 않는다. P3의 cell CI 및 확장 행동 보고 항목도 최종 승인 전 보완한다.

## 13. v1.4 GPU smoke 반환 검증 완료 (2026-09-21)

- §12의 GPU 미실행 상태는 과거 기록이다. 반환 원본·환경 lock·동결 입력과 runtime code hash·checkpoint 복사본 및 smoke 결과 검증이 통과했다. 증빙은 `experiment_v1_4/evidence/gpu_smoke_20260921_01/REPORT.md`와 `verification.json`이다.
- P2 완료: Tesla T4, microbatch 16, effective batch 64. Debug checkpoint는 본실험에 재사용하지 않는다.
- P3는 미실행이다. Cell CI·확장 행동 보고 구현 보완과 검증을 마친 뒤 학습 준비를 이어간다. GPU smoke 통과를 행동 gate나 P3 완료로 해석하지 않는다.


## 14. v1.4 P3 seed 0 gate·반환 검증 완료 (2026-09-21)

- §13의 P3 미실행은 과거 상태다. 64,005,751 tokens / 8,399 updates로 seed 0 학습을 완료했고,
  select 규칙으로 update 7,983을 선택해 one-time gate를 통과했다.
- 증빙: `experiment_v1_4/results/p3_audit_20260921_01/REPORT.md`, `completion.json`,
  `frozen_seed0.json`. 원본 ZIP·8,449개 checksum·GPU smoke·config/data/code·checkpoint·재개 검증 통과.
- 일반 READ 99.9277%, first macro 99.8140%, repeat 100%; legacy와 모든 group/coverage gate 통과.
- 다음은 같은 동결 설정의 P4 seed 1·2 재현이며 아직 미실행이다. 최소 두 seed 통과 전 표현 분석,
  전체 학습·validation 결정 동결 전 frozen test는 시작하지 않는다. Seed 0 결과만으로 전체 실험
  완료를 주장하지 않는다. Gate/test를 재평가하지 않았다.

## 15. v1.4 P4 로컬 실행 준비 완료 (2026-09-21)

- P4 현재 상태는 `experiment_v1_4/P4_STATUS.md`, 준비 증빙은
  `experiment_v1_4/results/p4_preparation_r1/verification.json`을 먼저 확인한다.
- `--stage p4`에서 seed 1·2만 허용하며 감사된 seed 0 증빙과 같은 config/data/예산,
  microbatch 16을 검사한다. 모델·학습·평가 수치 코드는 seed 0과 byte-identical하다.
- 전용 `P4_v1_4_seed1_r1.ipynb`, `P4_v1_4_seed2_r1.ipynb`와 공통
  `v1_4_p4_bundle_r1.zip`을 사용한다. 기존 P3 노트북의 seed 숫자만 바꾸지 않는다.
- 33 tests, CPU smoke, 두 seed의 CLI persistent resume 및 반환 감사, 421개 ZIP member
  checksum과 노트북 검증은 통과했다. 실제 seed 1·2 GPU 학습·gate는 아직 미실행이다.
- 두 반환물을 감사한 뒤 세 seed 전체의 학습·validation 결정을 동결하고 별도 frozen-test
  노트북으로 모든 학습 seed를 평가한다. 현 준비 완료를 P4 전체 완료로 표시하지 않는다.

## 16. 사용자 승인 디스크 정리 (2026-09-21)

- 최신 사용자 지시에 따라 과거 원본 전체 보존 규칙에 이번 삭제 목록만 예외를 적용했다.
- 현재 파일 가용성은 `maintenance/disk_cleanup_20260921/REPORT.md`와 `deleted.jsonl`을 먼저 확인한다.
- 이전 전달 ZIP, 완료된 smoke/resume 디버그 checkpoint, 거부된 최초 v1.4 corpus payload 일부는 해시·축약 기록을 남기고 삭제했다. 옛 문서의 원본 존재 문구만 믿지 않는다.
- 활성 rebuild_01, 동결 config, seed 0 반환 원본과 감사 증빙, P3 r3 및 P4 번들·노트북은 보존·해시 검증했다. 실험 상태와 gate는 변경하지 않았다.
- 삭제된 debug checkpoint는 본학습 입력이 아니다. 과거 검증을 재실행하려면 별도 경로에서 재생성하며 기존 성공 보고서를 수정하지 않는다.

## 17. v1.4 P4 seed 1·2 반환 감사 완료 (2026-09-21)

- 현재 기준은 `experiment_v1_4/results/p4_audit_20260921_01/REPORT.md`, `completion.json`,
  `validation_freeze.json`이다. seed 0·1·2 모두 validation gate 통과, checkpoint 선택 동결 완료다.
- Seed 1 선택 update 7,983, seed 2 update 8,399. 두 seed 모두 64,005,751 tokens / 8,399 updates.
- 로컬 Mac/PyTorch 2.10과 Colab Linux/PyTorch 2.11의 초기화는 tiny FP32 차이로
  bitwise 동일하지 않다. 반환된 Colab 검증은 통과했고 로컬은 다른 환경에 한해 atol=1e-7,
  rtol=0 및 동일 seed 비교를 적용했다. 최초 strict 실패와 최대 차이를 보존했다.
- 다음은 동결된 세 seed 전체의 one-time frozen test와 반환 검증이다. 아직 미실행이며
  P4 전체 완료 또는 P5 진입으로 표시하지 않는다. Gate/test 재채점은 하지 않았다.

## 18. v1.4 Frozen test 실행 전달 (2026-09-21)

- `experiment_v1_4/notebooks/P4_v1_4_frozen_test_r1.ipynb`와
  `experiment_v1_4/bundles/v1_4_frozen_test_bundle_r1.zip`이 전용 전달물이다.
- 기준은 `frozen_test_r1/contract.json`과 `results/frozen_test_preparation_r1/verification.json`이다
  (모두 `experiment_v1_4/` 기준). 세 seed × 7 suite를 동결 checkpoint에서 평가한다.
- 로컬 6 tests·미학습 CPU preflight와 입력 checksum 검증은 통과했다. 실제 test는 미실행이다.
- 완료 결과는 재사용하고 raw 저장 후 집계만 재개한다. 시작만 기록된 평가는 자동 재추론하지 않는다.
- 반환 검증 전 P4 완료/P5 진입으로 표시하지 않는다. Length는 P10 범위다.
