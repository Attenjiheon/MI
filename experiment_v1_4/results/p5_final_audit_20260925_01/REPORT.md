# P5 완료 반환 감사 — 2026-09-25

**P5 READ cache·probe 및 전 12층 full probe 진단 완료. P6 진입 가능, 미실행.**
P4의 동결 LM seed 0·1·2를 유지했고, 2026-09-24 metadata 감사에 이어 원본 activation이 있는 Colab에서 독립 수치 검산을 수행한 반환물을 확인했다. 첨부 내용은 데이터로만 읽었으며 지시 또는 코드를 실행하지 않았다.

## 검증 결과

- 원본 ZIP SHA256 `adf5a3a80cdb915e28bb7a7f92cf8bb120922c36f59fa2a2a02d4209a985c0b3` 및 내부 **3,518개 파일** checksum 일치. Manifest를 포함하면 ZIP 항목은 3,519개다.
- 준비된 독립 감사 코드·정책·입력 reference manifest로 audit ID를 재계산하여 completion, 모든 task/cache receipt 및 실행 환경과 일치함을 확인했다.
- **3,510개 task 전부 passed**, 실패·누락 없음. Seed당 1,170개, 12층 h/u/m의 trained/init/full/shuffled 및 block 0 좌표/random/128후보/전이와 token/position 대조군 포함.
- **6개 seed/kind cache 묶음, 원본 NPZ 240개** 검증 증빙을 동결 metadata의 path/hash/position 수와 전수 대조했다. 각 seed/kind의 train/val/test 50k/10k/20k, READ key·row join·shape/dtype/유한성·전체 coverage 검사를 통과했다.
- **Dictionary train scalar 전처리 9개**와 probe train 전처리 검산 통과. 보고된 probe 평균·표준편차 최대 절대 오차는 모두 **0**이다.
- **선택 모델 3,750개**, 현재≠과거 subset을 포함한 **5,250개 평가**의 재계산 결과를 기존 동결 보고값과 비교해 일치했다.
- BA/F1: 5,250개 평가 각각 **1,000 valid bootstrap draws**. Binary AUROC: 4,500개 평가 각각 1,000 valid draws. 나머지 750개는 다중 클래스라 binary AUROC 적용 대상이 아니다.
- Numeric tolerance는 사전 고정 `atol=1e-9, rtol=1e-8`. 수치 불일치로 tolerance나 원본 결과를 변경한 기록은 없다.
- 독립 감사 task 시간 합계 **9,811.16초(2.73시간)**. Cache 검증·I/O·준비 등을 포함한 총 경과 시간은 아니다. 원래 CPU fitting 작업 시간 합계는 26.65시간이다.
- 로컬 동결 입력/code **59개 hash** 재검증과 관련 회귀검사 **28 passed**. 모델 재추론·probe 재학습·test 기반 선택 변경 없음.

## 의미 결과

READ 현재 값, full h, IID test balanced accuracy(%):

| Block | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| 0 | 59.78 | 63.82 | 60.62 |
| 5 | 99.15 | 99.89 | 99.05 |
| 11 | 99.93 | 99.92 | 99.99 |

[전 층·대조군 상세 보고](../p5_metadata_audit_20260924_01/REPORT.md)와
[전체 semantic CSV](../p5_metadata_audit_20260924_01/semantic_snapshot.csv)의 점수는 이번 독립 감사에서 재현되었다. 해당 과거 보고서의 ‘원본 검산 대기’는 당시 상태이며 이번 완료 증빙으로 해소됐다.

세 seed 모두 후반 층에서 현재 값 접근성이 높다. 주 sparse 분석 대상 block 0에서는 약한 신호와 seed 차이가 있으며, 과거 값은 현재≠과거 subset에서 우연 수준 부근이다. 높은 선형 접근성을 인과적 사용의 증거로 해석하지 않는다. 동결 block 0·checkpoint·k·분포·선택 규칙은 유지한다.

## 감사 범위와 완료 판정

원본 production NPZ를 로컬에 회수한 것은 아니다. 원본 Drive cache에서 실행한 별도 수치 감사의 코드/입력 identity와 전체 반환 receipt를 로컬에서 검증한 것이다. 독립 검산기는 production probe/cache/metrics 모듈을 import하지 않으며, GPU activation을 새로 추출하지 않는다.

선택 계수의 train 통계·validation 예측/threshold·test 예측·AUROC·CI와 train feature ranking을 원본 activation으로 검산했다. 선택되지 않은 lambda/prefix 계수는 원본에 저장되지 않아 후보 간 선택은 frozen validation trace로 확인했다. Optimizer의 모든 후보를 독립 재학습했다고 주장하지 않는다. 이 한계는 원래 감사 계획과 일치하며, 선택 모델의 수치 재현·gate 확인을 통과했으므로 P5를 완료한다.

반환 `completion.json`의 `p5_complete:false`는 로컬 반환 검토 전 phase 자동 완료를 방지하는 값이다. 원본은 수정하지 않으며, 이번 로컬 판정은 이 폴더의 별도 `completion.json`이다.

P6에서는 같은 검증된 block 0 READ train/val/test 위치와 scalar 통계를 사용한다. 원본 cache는 Drive `boolean_interp_v1_4/P5_r2`에 보존한다. P6–P9 필수 SAE·TC·인과 평가·sparse seed 반복은 미실행이며 전체 실험 완료가 아니다. Git 완료 절차는 커밋 후 원격 main SHA 일치까지 확인한다.

## 재현과 증빙

- `verify_return.py`: 외부/내부 checksum, audit ID, 3,510 task 및 6 cache group receipt, 원본 동결 결과 대조.
- `verification.json`: 정확한 수·시간·환경·policy·수치 오차.
- `returned/`: 원본 반환 파일. 기존 상태값을 변경하지 않는다.
- `p6_cache_manifest.json`: 다음 단계용 원본 cache hash/경로, 라벨 및 scalar 전처리 hash.
- 로컬 명령: `/opt/anaconda3/bin/python experiment_v1_4/results/p5_final_audit_20260925_01/verify_return.py`
- 회귀검사: `/opt/anaconda3/bin/python -m pytest tests_v1_4/test_p5.py tests_v1_4/test_p5_transfer.py tests_v1_4/test_p5_cpu_resume.py tests_v1_4/test_p5_independent_audit.py -q`
