# v1.4 전체 범위 감사 및 CPU smoke — 2026-09-20

판정: **구현 fixture CPU smoke 통과, 전체 실험/코퍼스 감사 미완료, GPU 진입 불가**.
생성 중인 데이터와 기존 산출물은 수정하지 않았다. 실행 중인 생성기 소스도 변경하지 않았다.
이 보고서는 실패·누락을 포함한 현시점 감사이며 P1/P2/P3 완료 보고가 아니다.

## 실제 실행과 증빙

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/anaconda3/bin/python -m pytest tests_v1_4 -q --junitxml=experiment_v1_4/results/audit_20260920_02/tests.xml`: **15 passed / 3 failed**, 26.75초. 상세 `tests.log`, `tests.xml`.
- 실패 3개는 stage contract의 `configs/run.json`, frozen corpus의 `manifest.json`, input 검증의 `config_set_manifest.json` 부재다. Skip이나 성공으로 바꾸지 않았다.
- `/opt/anaconda3/bin/python -m interp_v1_4.smoke --device cpu --fixture-only --output experiment_v1_4/smoke/cpu_fixture_audit_02`: **passed**, 33.48초, env `3c50b28adae72d67`, Python 3.12.7 / torch 2.10.0 / CPU.
- 9,485,312 parameters, 12 blocks × 256, causal/PAD mask, RoPE, token-only read4, accumulation, 최대 길이 302 forward/backward, LM 2 updates와 복구 후 다음 update 일치 확인.
- 복구 후 update 3 / cursor 192 / 21,755 debug prediction tokens. Model tensor SHA-256 `aec488ea2820afb2964ac7b77bf1e9337cdfc2980a6c9f313c388872a3ff9320`.
- SAE/TC 각각 k=4/16 × 100 updates, 각 51,200 draws, probe fitting, 12층 hook, identity/full/sparse patch 통과. 이는 구현 검사이며 P5 이후 본실험이 아니다.
- `fixture_only=true`, `frozen_input_verified=false`. 정식 frozen-input CPU smoke와 real-CLI persistent-index 재개 검사를 대체하지 않는다.
- `/opt/anaconda3/bin/python experiment_v1_4/results/audit_20260920_02/check_snapshot.py`: 닫힌 파일의 독립 replay 및 hash 검사. `snapshot.json`, `snapshot.log`에 결과 저장.

## 코퍼스 및 설계 확인

설계 manifest에 등록된 설계/README/규격/제안/선행 corpus manifest와 audit/선행 evidence ZIP hash가 모두 일치한다.
복사된 v1.3 train 68 shards의 metadata/token 136개 파일은 모두 원본과 byte-identical하다.

현재 닫힌 select general 512, gate general 1,024, gate legacy diagnostic 각각 1,024,
select first/repeat 2,688 pairs(5,376 members, 42 cells × 64)를 검사했다.
합계 9,984개 고유 sequence, 149,607개 고유 READ prefix다. Token/metadata 정렬, 독립 replay,
실제 v1.4 seed, 진단 target 조건, first/repeat의 연산·입력·깊이 cell, 상태/답/깊이 동일성,
READ 1회 삽입 후 원본 복원과 quota가 통과했다. 읽기 전후 파일 checksum도 동일하다.

현재 검사한 sequence는 v1.0/v1.2/v1.3 all-sequence registry와 교집합 0이다.
현재 prefix는 v1.3 inherited reserved-prefix registry 3,148,422개 항목과 교집합 0이다.
v1.0/v1.2에는 해당 이름의 reserved-prefix 파일이 없다. **모든 과거 row에서 prefix를 다시
추출하는 독립 검사는 이번 snapshot에 포함되지 않았다. 전체 역사 누출 감사 통과로 확대하지 않는다.**

생성기 PID 50426, 후속 준비 PID 51223의 실행을 읽기 전용 확인했다. 관측 시점은 gate
first/repeat 생성 중이며 최종 manifest/postwrite audit가 없다. 새 train extension,
test/interpretation/causal 전체 quota와 replay, 최종 64M cursor는 아직 검증할 수 없다.
Test 모델 점수, GPU, 본학습은 실행하지 않았다.

## 발견 사항

1. **[P2] First/repeat target당 시도 상한이 registry 충돌 뒤 초기화된다.**
   `scripts/generate_v1_4_corpus.py:120–126`에서 `sample_pair(..., MAX_ATTEMPTS)`가 성공한 뒤
   `reserve_pair`가 충돌을 반환하면 새 100,000회 한도로 다시 호출한다.
   02 §9 및 03 §5.2.7의 target당 최대 100,000회가 전체 rejection 경로에 적용되지 않는다.
   Synthetic sample 100,000회 + prefix 충돌 + sample 100,000회로 단일 target이 성공 처리되는
   것을 `attempt_cap_reproduction.json`에 재현했다. 실제 생성물의 상한 초과를 증명한 것은 아니다.
   다음 수정에서는 수락 전 누적 시도를 유지하고 남은 한도만 전달해야 한다. 현재 실행 프로세스의
   코드를 중간에 바꾸면 기록된 code hash와 실제 실행 코드가 달라질 수 있어 수정하지 않았다.

2. **[P2] Postwrite first/repeat 감사가 cell 의미와 삽입 구조를 독립 검증하지 않는다.**
   `scripts/generate_v1_4_corpus.py:429–470`은 저장된 cell ID를 세고 replay/상태/답/READ 횟수를
   확인하지만 origin target의 마지막 update로부터 operator/input/depth cell을 재계산하지 않는다.
   Repeat의 READ 횟수도 `>=1`만 검사하며 READ 하나를 제거하면 원본과 동일한지 확인하지 않는다.
   따라서 잘못 저장된 cell 라벨/파생 구조를 놓칠 수 있다. 이번 snapshot은 **select 전체**에
   추가 검사를 실행해 통과했으나 gate/test와 최종 독립 audit에도 같은 검사가 필요하다.

3. **보고 산출물 누락:** `interp_v1_4/behavior.py`는 42-cell count/CE/accuracy를 저장하지만
   DESIGN §4가 요구하는 cell CI를 산출하지 않는다. 03 §10.1의 all-token CE, 세부 strata와
   행동 기준선도 현재 P3 요약에 포함되지 않는다. Gate/선택 기준 변경 없이 최종 보고 경로에서
   구현·검증해야 하며, 현재 smoke 통과로 보고 규격까지 완료됐다고 할 수 없다.

4. **문서 현황 불일치:** 루트 README는 v1.3을 활성 confirm 대기로 표시한다. 최신 AGENTS §11과
   v1.4 설계가 v1.3 gate 실패 및 v1.4 준비 상태를 명시한다. 설계 원본·기존 실패 증빙은 보존한다.

## 코드 감사 범위와 남은 gate

v1.4 model/training/behavior/runtime/CLI/smoke/integration/dictionary와 재사용되는 v1.2
runtime/persistence/probe/patching, corpus builder/first-repeat, config freeze, evidence verifier,
Colab builder를 읽어 계약과 대조했다. Forward 입력 token/mask, FP32/결정성, read4 분모,
LR/64M 경계, global near-tie 선택, seed 0 제한, one-time gate marker, immutable index 복구
구조를 확인했다. 동적 검증 범위는 위 tests와 fixture smoke에 한정된다.

남은 순서는 수정 corpus 완성 → 전체 독립 audit와 위 생성/감사 결함 처리 → frozen config →
전체 tests → frozen-input CPU smoke → real-CLI persistent resume/evidence 검증 → notebook/bundle
checksum 감사 → 실제 Colab GPU smoke다. 이후에만 P3 seed 0 학습을 허용한다.
P1/P2 완료 gate가 충족되지 않아 phase 완료 커밋·main push는 수행하지 않았다.
