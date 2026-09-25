# P5 원본 activation 독립 검산 준비 — 2026-09-24

**Colab 전달물 준비·로컬 검증 완료. Production activation 감사는 미실행이며 P5는 아직 미완료다.**

선행 증빙은 P4 동결 LM 3개와 `p5_metadata_audit_20260924_01`의 3,510개 CPU probe metadata 감사다. 로컬 저장소·Desktop·Downloads·CloudStorage·Volumes의 관련 파일 탐색에서는 production P5 NPZ를 찾지 못했다. 원본은 기존 Drive P5_r2 경로를 사용한다.

## 전달물

- `experiment_v1_4/notebooks/P5_v1_4_independent_audit_r1.ipynb`
- `experiment_v1_4/bundles/v1_4_p5_independent_audit_r1.zip` (11,925,457 bytes)
- 같은 ZIP의 `.sha256`

노트북은 CPU 런타임에서 처음부터 실행한다. 감사 ZIP을 Drive `내 드라이브/boolean_interp_v1_4/`에 넣거나 노트북의 작은 ZIP 업로드 셀로 넣는다. 기존 대용량 LM 입력 번들은 필요 없다. 원본 cache 기본 경로는 `내 드라이브/boolean_interp_v1_4/P5_r2`다. 경로가 다르면 첫 셀의 SOURCE를 수정한다.

읽기 전용 원본과 별도로 `P5_independent_audit_r1`에 감사 증빙을 저장한다. 실패·중단 시에도 다운로드 셀로 회수할 수 있다. 중단 후 같은 입력으로 재개하면 hash가 맞는 완료 작업은 건너뛴다. NPZ는 seed/kind 단위로 검증·전개하여 임시 디스크 약 3GB를 사용한다. 로컬 5GB 이상 여유를 요구한다.

## 독립 구현과 검사 범위

`scripts/p5_independent_audit.py` 감사기는 production probe/cache/metric 모듈을 import하지 않는다. NumPy 2.1.3, SciPy 1.16.3으로 원본과 별도 계산한다.

1. 반환 snapshot inventory에 고정한 계약·라벨·probe·directions·전처리·cache marker의 SHA256.
2. 동결 corpus 및 예약 위치에서 READ labels·순서를 독립 재구성하고 50k/10k/20k quota 확인.
3. 240개 NPZ의 bytes hash, checkpoint/config/label identity, sequence offset과 row join, 12층 × h/u/m shape·float32·유한성·중복·coverage.
4. 9개 dictionary scalar 전처리를 train 전체에서 재계산. Probe 차원별 mean/std는 IID/source-domain train에서 재계산.
5. 저장 random projection RNG·행렬·subset, train ANOVA ranking, prefix·상수 열 제거 확인.
6. 저장 선택 계수의 validation 예측·CE·threshold와 test confusion matrix·BA/F1/AUROC 재계산. Test로 선택하지 않음.
7. Sequence cluster 1,000 draw의 BA/F1/AUROC 95% CI·valid count·seed·cluster count 재계산. 현재≠과거 subset 포함. AUROC는 동점 처리를 포함한 ROC 사다리꼴 적분으로 독립 구현.
8. 전 3,510개 task별 receipt와 환경·실패 기록. 실행 code/source hash가 달라지면 이전 감사 receipt 재사용을 거부.

부동소수 비교는 atol=1e-9, rtol=1e-8로 고정했다. Count/seed/identity는 정확 일치를 요구한다. 실패 시 허용 오차나 원본 결과를 자동 수정하지 않는다.

## 실제 로컬 검증

- `python -m pytest tests_v1_4/test_p5_independent_audit.py -q`: **8 passed**.
  동점 AUROC, 클래스가 사라지는 cluster draw, 다중 클래스, source-domain 전처리·shuffle/ranking,
  평균/계수/CI 변조 탐지, cache checksum·행 검증을 포함한다.
- 실제 반환 **token/position 대조군 30개**의 독립 수치 감사 통과. 동결 corpus로 라벨 및 입력을 재구성해 저장 계수·선택 threshold·AUROC/CI를 검산했다. Activation을 사용한 3,480개 작업의 감사로 세지 않는다.
- 노트북 nbformat validation, 모든 코드 셀 compile, ZIP 내부 9개 파일 checksum, 번들 내 코드와 로컬 코드 동일성을 확인했다.
- `bundle_verification.json`, `local_token_verification.json`, `local_token_receipts/`에 근거를 보존했다.

로컬 token 재현 명령:
`/opt/anaconda3/bin/python experiment_v1_4/results/p5_independent_audit_preparation_r1/local_token_audit.py`
(기존 영속 결과와 다른 실행 시간 기록을 덮어쓰지 않으므로 새 증빙 경로에서 재실행하거나 기존 결과를 보존한다.)

## 남은 증빙과 해석 한계

본 activation cache가 있는 환경에서 노트북을 실행하고 결과 ZIP 및 SHA256 파일을 반환해야 한다. 독립 수치 감사 completion과 3,510 task / 6 cache group receipt를 로컬에서 검토한 다음 P5 완료 조건을 판정한다. 준비만으로 P6에 진입하거나 P5 완료 commit/push를 하지 않는다.

이번 요청 범위인 전처리·저장 계수 예측·AUROC/CI는 모두 독립 검산 대상으로 구현했다. 선택되지 않은 lambda/prefix의 계수는 원본 결과에 저장되지 않아 후보 간 선택은 frozen trace로 확인한다. Optimizer 재학습 또는 모든 후보 계수의 독립 재현을 수행했다고 주장하지 않는다. LM activation을 새로 추출해 비교하지도 않는다. 이 한계는 실제 감사 completion에도 기록된다.
