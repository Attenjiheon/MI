# 실험 세션 진입 규칙

이 파일은 프로젝트 세션을 시작할 때 가장 먼저 읽는 **라우터**다. 세부 실험값을 요약해 두는 문서가 아니며, 실제 규격은 아래 원문에서 확인한다.

## 1. `Pn단계를 수행하자` 요청을 받았을 때

1. 이 파일을 끝까지 읽는다.
2. [phase.md](./phase.md)의 `전체 순서와 우선순위`, 요청받은 `Pn` 절, 바로 앞 단계의 완료 조건을 읽는다. 계산 자원이나 실행 기록이 관련되면 `계산 예산과 운영 기록`도 읽는다.
3. 아래 표에 따라 필요한 원문 절을 읽는다. 구현값·표본 수·수식·gate를 기억에 의존해 적용하지 않는다.
4. `experiment_v1_4/` 아래의 해당 단계 상태 문서, config, manifest, run registry, README와 실제 산출물을 확인한다. 체크박스나 파일 존재만으로 통과를 가정하지 않는다.
5. 작업 전에 현재 단계의 선행 조건, 이번 목표, 완료 조건, 아직 없는 증빙을 짧게 정리한다.
6. 아직 충족되지 않은 첫 항목부터 실행하고 검증한다. 단계가 끝난 뒤에만 상태·체크박스·registry를 증빙 경로와 함께 갱신한다.

단계 번호가 모호하거나 현재 상태와 맞지 않으면, 안전한 읽기·검증은 먼저 수행하되 규격 변경이나 비용이 큰 실행 전에 불일치를 알린다.

## 2. 문서 역할과 우선순위

| 문서 | 역할 |
|---|---|
| [README.md](./README.md) | 저장소 전체 구조, 본 실험 구조, 과거 버전 archive, 실제 archive 경로와 동결 원문 보존 |
| [phase.md](./phase.md) | 실행 순서, 선행 조건, 단계별 산출물과 완료 gate |
| [01_experiment_design.md](./01_experiment_design.md) | 연구 목적, 가설, 해석 범위와 보고 원칙 |
| [02_language_and_corpus.md](./02_language_and_corpus.md) | 언어 문법, 토큰 ID, 실행 의미, 생성·split·metadata 규격 |
| [03_experiment_spec.md](./03_experiment_spec.md) | 모델·학습·probe·SAE·TC·인과 평가의 상세 실행 규격 |
| `experiment_v1_4/` | 본 실험의 동결 계약·상태·실행 기록·산출물. 과거 버전은 `archive/legacy/`를 따른다 |

규격 충돌의 적용 우선순위는 **사용자의 최신 명시 지시 → v1.4 동결 실행 계약·명시적 변경 → 03의 통합 상세 규격 → 02의 언어·코퍼스 규격 → 01의 연구 설계**다. 실행 순서와 단계 gate는 `phase.md`를 따르되, phase가 상세 규격을 바꾸는 근거로 쓰이지는 않는다. 충돌을 발견하면 조용히 절충하지 말고 해당 문서와 항목을 기록한다. 이 파일은 원문을 대체하지 않는다.

01–03에 공통으로 통합한 핵심 사항은 다음과 같다.

- Transcoder는 선택 확장이 아니라 필수 READ 분석이며 READ SAE 뒤에 수행한다.
- LM 데이터는 CPU에서 미리 생성한 shard를 모든 LM seed가 같은 순서로 한 번씩 소비한다. GPU에서 새로 생성하지 않는다.

## 3. 단계별 필수 읽기

아래 범위는 최소치다. 대상 코드나 결과가 다른 절을 참조하면 그 절까지 이어서 읽는다.

| 단계 | 먼저 읽을 원문 |
|---|---|
| P0 규격·현황 | 03 §1, §12–14; 01 §4–5, §11–12; `experiment_v1_4/CURRENT.md`와 동결 config/설계 계약 |
| P1 CPU 코퍼스 | 02 §2–16; 03 §5; 기존 corpus 코드·test·manifest·audit 결과 |
| P2 구현·smoke | 03 §2–4, §6–12, §14; 02 §10, §14–16; `experiment_v1_4/configs/`와 P2 환경·데이터 증빙 |
| P3 LM seed 0 64M | 03 §3–5, §10.1, §12–14; 01 §5–7; 02 §14; P1/P2 증빙 |
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
5. **범위:** 필수 범위는 gate 통과 LM의 block 0 READ probe·SAE·TC와 전 층 full probe 진단, `k={4,16}`, 후보 수 대조, 근사 대체와 인과 평가이며 LM seed 0의 sparse seed 1 반복을 포함한다. Update, block 1, length GPU 평가, `m→m` SAE는 P10 선택 분석이다. Crosscoder와 변수 8개 확장은 기본 범위 밖이다.
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

## 8. 현재 본 실험과 다음 단계 (2026-09-25)

- 본 실험은 **v1.4**다. [현재 안내](experiment_v1_4/CURRENT.md),
  [통합 실행 계획](phase.md), `experiment_v1_4/P1_STATUS.md`–`P5_STATUS.md`를 따른다.
- P1–P4 완료. Seed 0·1·2 모두 validation gate 통과, 전체 학습·선택 결정과 frozen test 반환 감사 완료.
  현재 근거는 `experiment_v1_4/results/frozen_test_audit_20260922_01/`의
  `REPORT.md`, `completion.json`, `frozen_lms.json`이다.
- P5 완료: 3,510개 probe와 원본 cache·전처리·예측·AUROC/CI 독립 검산을 마쳤다.
  근거는 `experiment_v1_4/results/p5_final_audit_20260925_01/`의 `REPORT.md`, `completion.json`, `p6_cache_manifest.json`이다.
- 다음은 P6 block 0 READ SAE, 세 LM × k=4/16이다. 아직 미실행이다.
  구현·검증은 `interp_v1_4/`, `tests_v1_4/`를 사용하고 P6 config를 실행 전에 고정한다.
- 활성 corpus는 `corpus_rebuild.json`에 지정된 `data/language_v1_4/rebuild_01/`이다.
  최초 거부 root를 학습하거나 완료한 gate/test를 다시 열지 않는다.
- `experiment_v1_4/DESIGN.md`, `README.md`, `design_config.json`은 동결 당시의 불변 기록이다.
  그 안의 readiness/미실행 문구를 현재 상태로 읽지 않는다. 현재 안내는 `CURRENT.md`다.

## 9. 과거 버전·동결 증빙 보존

- 사용자 승인 저장소 정리로 v1.0–v1.3 실험·코드·검증·데이터를 `archive/legacy/`에 모았다.
  최신 사용자 지시로 모든 바로가기를 제거했다. 실제 archive 경로를 사용한다.
  `interp_v1_4`의 과거 모듈 의존성은 `archive.legacy.interp_v1_2` 직접 import로 연결한다.
- 과거 감사에는 해당 버전의 원본 계약·코드·결과를 사용한다.
  당시 루트 01–03·phase·AGENTS·README는 `archive/specifications/pre_v1_4_integration/`에 보존한다.
  기존 manifest의 source_documents hash는 이 snapshot을 대상으로 검증한다.
- 이동 매핑·원문 hash·전후 검증은 `maintenance/repository_cleanup_20260922/`를 따른다.
  기존 config·manifest·증빙의 경로 문자열이나 hash를 새 경로로 재작성하지 않는다.
- 과거 삭제 목록과 가용성은 `maintenance/disk_cleanup_20260921/REPORT.md`, `deleted.jsonl`을 따른다.
  삭제된 debug checkpoint를 필요하면 별도 경로에서 재생성하고 옛 성공 보고서를 수정하지 않는다.

- 바로가기 제거 후 소스 경로 변경과 변경 전 원본은 `maintenance/remove_shortcuts_20260922/`에 기록했다.
  기존 GPU smoke의 code hash를 현재 소스 승인으로 재사용하지 않는다. 이후 GPU 작업은 새 소스 hash로 별도 검증한다.
