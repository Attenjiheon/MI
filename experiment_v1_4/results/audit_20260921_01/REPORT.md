# v1.4 완성 코퍼스 감사와 정식 CPU 검증

2026-09-21. 최종 판정은 `completion.json`에 기록한다. 이 보고서는 P1 및 P2 CPU 범위이며
GPU smoke, P3 학습/행동 gate, test 모델 채점, replication, 표현 본실험을 실행하지 않았다.

## 검증 결과

- 수정 root: `data/language_v1_4/rebuild_01`. 거부된 원본은 원래 경로에 보존했다.
- 최종 학습량: **64,005,751 prediction tokens / 537,536 sequences / 133 shards**.
  마지막 완전 update는 **8,399**, nominal 64M 대비 overshoot **5,751**이다.
- `corpus/full_replay.json`: 전체 학습 및 새 split의 독립 parser replay, token/metadata 정렬,
  모든 manifest 파일 checksum, interpretation 위치 quota/join, 인과 pair replay를 검증했다.
  과거 sequence **344,948**, READ prefix **5,287,408**를 독립 열거한 loader로 대조했다.
  최종 registry는 sequence **659,092**, READ prefix **10,245,193**이다.
- 새 데이터와 과거 데이터 및 새 split 사이의 금지된 sequence/READ-prefix 교집합은 없다.
  의도된 v1.3 train prefix 복사와 pair 내부 공유 prefix는 규격의 예외로 분리했다.
- v1.3 train 68 shards의 136개 token/metadata 파일이 byte-identical하다.
- Select/gate 각각 42×64=2,688 pairs, test 42×128=5,376 pairs다.
- `policy.json`: 토큰 기반 holdout/길이 검사 **558,432 sequences**,
  train coverage **16 states / 48 commands / 12 truth-table / 25 adjacent-op patterns**.
  모든 10,752 first/repeat pair의 raw-token cell/depth와 단일 READ 삽입, 인과 3,072 pair의
  단일 초기화/SET bit 변경을 검증했다.
- `language_self_test.json`: 768 전이, NOT/XOR 복원, 고정 seed token/metadata 재현 통과.
- `tests_final.xml`, `tests_final.log`: **27 passed**. 종전 18개에 시도 상한 및 잘못된 pair
  label/index/삽입/target/READ 횟수/depth를 거부하는 회귀 검사를 추가했다.
- `existing_cpu_verified.json`: frozen config/corpus 및 현재 모델 코드 hash를 다시 계산해
  기존 `smoke/cpu_01/smoke.json`과 일치함을 확인했다. 정식 smoke는 `fixture_only=false`,
  `frozen_input_verified=true`, CPU 27.30초, env `3c50b28adae72d67`이다.
- 실제 CLI의 full/resumed run을 evidence verifier로 재감사했다. 모델·optimizer·RNG·cursor·
  microbatch·다음 평가 경계가 bitwise 동일하다. 3 debug updates / 21,755 tokens이며 본학습이 아니다.

## 두 결함의 처리

**Target당 시도 상한:** 생성 코드를 수정해 registry 충돌 뒤에도 누적 시도 횟수를 유지하고,
남은 한도만 sampler에 전달한다. 새 생성에서는 `attempts_per_target`도 저장한다.
이번 생성물은 원래 per-cell 합계만 저장했으므로, 기존 seed로 sampling stream을 재현해
저장된 origin tokens/target READ ID, 총 시도·거부 횟수, 최종 PCG64 상태를 모두 대조했다.
126 cell / 10,752 targets에서 전부 일치했고 최대는 **58,433 < 100,000**이었다.
따라서 상한 수정 때문에 이번 corpus를 재생성하거나 데이터 분포를 바꿀 필요가 없다.
증빙은 `attempts_02/summary.json`과 cell별 126 JSON이다.

재현은 CPU NumPy Generator를 Numba로 가속했다. 모델 compile을 사용한 것이 아니다.
42개 cell에서 원 sampler와 token/target/RNG/시도 횟수 동일성을 먼저 확인했고, 전체 stream의
최종 RNG 및 rejection 합계까지 일치함을 확인했다. 버전은 `attempts_environment.json`에 기록했다.
첫 JIT 실행의 argument typing 실패는 `attempts.log`에 보존했고, 성공으로 취급하지 않았다.

**Pair 의미 감사:** postwrite validator에 cell ID/index/operator/input/depth의 의미, 실제 token에서
계산한 target depth, 같은 상태/답/변수, 정확히 한 번의 이전 READ, 삽입 READ 삭제 시 원본 복원,
seed/target/ID 정합성을 추가했다. 출력 파일이 이미 있으면 거부하여 기존 감사 증빙을 보존한다.
전체 corpus replay가 시작된 뒤 추가한 raw-token depth 검사는 `policy.json`에서 별도로 전량 통과했다.

## 원본과 hash 보존

생성 완료 뒤에만 생성기 코드를 수정했다. `generator_at_creation.py`는 corpus manifest의 원래
생성기 SHA-256과 일치한다. `generator_full_replay.py`는 전체 replay 시작 당시 감사 코드다.
`provenance.json`에 원 생성 코드, 전체 replay 코드, 최종 강화 validator 코드의 hash를 각각 남겼다.
기존 corpus/manifest/config, 실패 기록, r1 전달물을 덮어쓰지 않았다. 모델/학습 코드 변경이 없어
정식 CPU smoke를 무의미하게 재실행하지 않고 현재 hash 및 실제 checkpoint로 증빙을 재검증했다.

## 명령과 범위

```text
python -u scripts/audit_v1_4_completed.py --output experiment_v1_4/results/audit_20260921_01/corpus
python -u scripts/audit_v1_4_attempts.py --output experiment_v1_4/results/audit_20260921_01/attempts_02
python -u scripts/audit_v1_4_policy.py --output experiment_v1_4/results/audit_20260921_01/policy.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests_v1_4 -q --junitxml=experiment_v1_4/results/audit_20260921_01/tests_final.xml
python experiment_v1_4/results/audit_20260921_01/verify_existing_cpu.py
```

위 `python`은 `/opt/anaconda3/bin/python`이다. 신규 model gate/test 점수는 계산하지 않았다.
Corpus test 파일의 token/hash/실행 의미를 확인한 것은 모델 test 평가와 다르다.
`corpus/summary.json`의 attempt-history unverified 표기는 해당 replay 검사만의 범위다.
별도 `attempts_02/summary.json`이 그 누락을 해결하며 `completion.json`에서 종합한다.

## 남은 실험 범위

P2 전체 완료에는 실제 Colab GPU smoke와 환경 증빙이 필요하다. P3는 미실행이다.
이전 보고서의 cell CI 및 03 §10.1 확장 행동 보고 지표는 아직 구현되지 않았으며, P3 결과를
규격상 완결된 보고로 승인하기 전에 보완해야 한다. 이번 CPU 검사 통과로 그 보고 범위까지
구현 완료했다고 주장하지 않는다. 원본 설계 README는 hash 보존을 위해 수정하지 않고
현재 실행 상태는 P1/P2/P3_STATUS와 루트 README에서 관리한다.
