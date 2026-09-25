# P5 — 완료·독립 수치 감사 통과 (2026-09-25)

**P5 완료.** 세 LM의 3,510 probe, 240 cache 파일, 9개 dictionary scalar 통계와 전 12층 진단을 검증했다.
[최종 반환 감사](results/p5_final_audit_20260925_01/REPORT.md) · [완료 판정](results/p5_final_audit_20260925_01/completion.json).
다음은 P6 READ SAE이며 아직 실행하지 않았다. 아래 준비·중간 snapshot 절은 당시 이력이다.

## 실행 전달물

- [Colab 노트북 r2](notebooks/P5_v1_4_READ_r2.ipynb)
- [입력 ZIP r2](bundles/v1_4_p5_bundle_r2.zip) / [외부 SHA256](bundles/v1_4_p5_bundle_r2.zip.sha256)
- [동결 실행 계약](p5_r2/contract.json)
- [로컬 검증](results/p5_preparation_r2/verification.json)
- [입력·quota·support 검증](results/p5_preparation_r2/input_verification.json)

노트북을 Colab GPU 런타임에서 위에서 아래로 실행한다. 입력 ZIP을 업로드하고 checksum을 검증한 뒤
고정 의존성 설치, Drive 연결, 현재 코드의 GPU smoke, cache 추출, CPU probe, 결과 ZIP 반환 순서다.
중단 시 같은 입력과 Drive 경로로 재실행한다. 완전 저장 cache 조각과 probe JSON만 재사용한다.
각 단위의 환경 ID·소요 시간과 config/checkpoint/위치 hash를 보존한다.

전체 raw cache는 17,694,720,000 bytes다. ZIP과 재개 복사를 고려해 Colab 로컬 45GB,
Drive 40GB 이상 여유를 권장한다. 현재 로컬 작업 디스크는 전체 cache 반환·압축 해제를 위한 공간이
부족하므로, 반환 감사 시 충분한 디스크를 확보하거나 Drive/Colab CPU에서 이어서 작업해야 한다.
CPU probe를 나중에 실행하려면 GPU cache를 먼저 보존·반환할 수 있다. 어느 경우도 P5 완료가 아니다.

## 고정 범위

- 세 통과 LM의 선택 checkpoint와 각 seed **원본 init checkpoint**. 학습 전 모델 재생성 없음.
- 예약 READ train/val/test 50k/10k/20k 위치, block 0..11의 h/u/m float32.
- Full trained/init/shuffled probe: 전체 12개 층 h/u/m, 8개 라벨.
- Block 0 h/m의 coordinate/random 단일·≤4 prefix, 전체/128후보 대조.
- Token one-hot 15 + position/767 + position² 대조군.
- 현재/과거 값의 ABC→D 감독 전이 및 현재≠과거 subset 보고.
- Train 전용 차원별 probe 전처리와 별도 float32 dictionary scalar 통계(h/u/m).
- 4 lambda, 19 binary thresholds, validation tie-break, task별 난수 정수와 subset ID 동결.
- Sequence cluster bootstrap 1,000회, BA/macro-F1/binary AUROC와 유효 draw 수.
- 총 3,510개 probe 작업. Support 미달은 NA, 최적화 grid 수렴 실패는 failed로 보존한다.

## 로컬 증빙과 남은 완료 조건

- [x] 동결 모델·원본 init·활성 corpus hash 확인 및 예약 READ key/label/token 결합 검사.
- [x] 새 P5 회귀검사 11개 포함 전체 52 tests 통과.
- [x] 별도 debug 80 READ, 12층 3 hook의 batch/single 일치·미래 정답 차단·cache 저장/검증 재개 smoke.
- [x] r2 config·source hash·노트북·입력 ZIP 내부/외부 checksum 검증.
- [ ] 현재 소스 hash의 실제 Colab GPU smoke 및 전체 cache 반환.
- [ ] 세 LM의 본 cache/preprocessing/full probe/기준선/대조군 산출물.
- [ ] 12층 결과·support·실패 사유 및 선택 계수/CI의 독립 반환 감사.
- [ ] 완료 조건 확인 후 상태·registry 갱신, phase 완료 커밋 및 원격 main push/SHA 확인.

검증 명령은 [LOCAL_COMMANDS.md](results/p5_preparation_r2/LOCAL_COMMANDS.md)를 따른다.
`verify_v1_4_p5.py`는 checksum·key·quota·작업 누락·실패의 구조 검증이며 단독으로 P5 완료를 선언하지 않는다.
실제 반환 후 계수·train 통계·validation 선택·CI를 독립 감사해야 한다. P6는 계속 대기한다.

r1은 설치 셀의 CUDA wheel index를 보완하기 전 로컬 준비본이다.
[보존·대체 사유](results/p5_preparation_r1/SUPERSEDED.md)를 따르며 사용하지 않는다.
모델·분포·quota·평가법 변경은 없고 본실험 해석 test 점수는 보지 않았다.

## 대용량 반환 보완 — 2026-09-22

사용자가 CPU probe의 긴 실행 시간과 약 15GB evidence ZIP 다운로드 실패를 보고했다.
실제 cache/probe 반환물은 아직 받지 않았으므로 GPU 완료나 probe 진행률은 검증 전이다.

[회수 전용 노트북](notebooks/P5_v1_4_transfer_r1.ipynb)은 기존 Drive P5_r2 폴더를 읽어
먼저 cache 배열 없는 metadata ZIP을 만든다. 로컬 probe에 필요한 cache는 seed별로
약 512MiB 단위 독립 ZIP에 담고 내부 파일 SHA256·외부 part SHA256·index를 기록한다.
전체 데이터량을 줄이는 방식이 아니며 metadata만으로 로컬 fitting을 할 수는 없다.
원본 GPU 추출·probe·r2 계약과 결과는 변경하지 않는다.

작은 metadata ZIP과 index를 먼저 전달한 뒤 해당 seed cache 전체를 회수한다.
로컬 전체 cache 약 17.7GB와 작업 공간을 확보하거나 외장 디스크/seed별 순차 처리를 사용한다.
완료 probe JSON을 함께 가져오면 기존 실행기는 이를 건너뛰고 새 CPU 환경 ID를 기록한다.
로컬 CPU가 더 빠르다고 보장하지 않으며 실제 처리량을 측정해야 한다.

로컬 가져오기 및 seed 0 실행 예시(저장소 root, 경로는 실제 다운로드/저장 위치로 지정):

```bash
/opt/anaconda3/bin/python scripts/p5_transfer.py receive PART.zip CACHE_FOLDER
/opt/anaconda3/bin/python -m interp_v1_4.p5 probes --output CACHE_FOLDER --seeds 0
```

Receive는 각 metadata/cache 조각에 반복 적용한다. 원본 파일을 덮어쓰거나 기존 cache를
삭제하지 않는다. Drive cache는 후속 P6 이후에도 필요하므로 보존한다.
[로컬 전송 검증](results/p5_transfer_preparation_r1/verification.json): 4 tests 통과.

## Metadata 반환 snapshot 01

[2026-09-22 분석](results/p5_metadata_audit_20260922_01/REPORT.md): 646개 파일 checksum과 r2 계약·예약 라벨 일치.
전체 GPU cache 완료 marker 240개, CPU probe 384/3,510개(10.94%) 저장·전부 수렴 성공.
본 NPZ 배열은 미반환이므로 GPU cache 최종 독립 검증 및 P5 완료 판정은 보류한다.
현재 진행률은 회수 당시 snapshot이며 실시간 상태가 아니다. Colab CPU probe를 이어간다.

## CPU 전용 Drive 입력·충돌 복구 노트북

사용자 보고: bundle 브라우저 업로드 지연, 재개 셀의 local/Drive `labels/train.json` hash 충돌.
Drive already mounted 메시지는 실패 원인이 아니다. 실제 파일을 받지 않았으므로 손상된 쪽은
사전에 단정하지 않는다.

[CPU 전용 새 노트북](notebooks/P5_v1_4_CPU_Drive_resume_r1.ipynb)을 사용한다.
기존 r2 ZIP을 Drive `내 드라이브/boolean_interp_v1_4/v1_4_p5_bundle_r2.zip`에 올린 뒤
첫 셀부터 실행한다. 다른 위치는 BUNDLE_ON_DRIVE를 수정한다. 브라우저 files.upload는 없다.
기존 GPU 노트북·CPU 셀과 동시에 실행하지 않는다. GPU 추출을 재실행하지 않는다.

동결 corpus로 라벨을 독립 재생성하고 모든 해당 cache marker의 label hash와 대조한다.
확인된 라벨로 복구하기 전 기존 충돌 파일을 Drive `P5_resume_backups/<timestamp>/`에
hash 검증하여 보존한다. `repair.jsonl`에 실제 hash·크기·JSON 의미 일치 여부를 기록한다.
cache는 marker SHA256, probe는 config/task/결과 구조를 검증하여 한쪽의 유효한 파일을
재사용한다. 양쪽에 서로 다른 유효 결과가 있으면 선택하지 않고 보존 후 중단한다.
전체 cache 검증은 기존 frozen CPU runner가 fitting 전에 수행한다.

[로컬 검증](results/p5_cpu_resume_preparation_r1/verification.json): 5 tests 통과,
노트북 코드 compile, r2 동결 입력/code hash 유지 확인. 실제 Colab 오류 복구는 새 노트북
실행 전이므로 해결됐다고 실측 보고하지 않는다. 변경은 입력 전달·재개 절차에 한정된다.


## CPU probe 반환 완료·독립 수치 감사 준비 — 2026-09-24

[전체 metadata 감사](results/p5_metadata_audit_20260924_01/REPORT.md): 3,510/3,510개 모두 passed,
내부 3,821개 파일 checksum, 예약 라벨, 10,530개 support 표, 저장 선택 기록과 BA/F1 검산 통과.
CPU probe 계산·결과 반환은 완료됐다. 원본 production NPZ는 로컬에 없어 **P5 전체 완료는 보류**한다.

[독립 감사 준비](results/p5_independent_audit_preparation_r1/REPORT.md): 별도 수치 검산기 8 tests,
실제 token/position 대조군 30개 독립 전처리·예측·AUROC/CI 검산, 노트북·입력 checksum 검증 통과.
[CPU 감사 노트북](notebooks/P5_v1_4_independent_audit_r1.ipynb)과
[작은 감사 입력 ZIP](bundles/v1_4_p5_independent_audit_r1.zip)을 기존 Drive cache 환경에서 실행한다.
본 activation 3,480개 작업의 독립 감사와 전체 completion 반환 검토는 아직 남아 있다.

## 최종 종료 판정 — 2026-09-25

- [x] 실제 GPU smoke 및 원본 cache 독립 검산 반환 증빙 확보.
- [x] 세 seed의 전체 probe·기준선·대조군 3,510개 및 scalar 전처리 9개 검증.
- [x] 선택 모델 3,750개, subset 포함 평가 5,250개의 예측·AUROC·CI 재현.
- [x] 최종 상태·registry·P6 cache manifest 작성.

P5 완료 근거는 [최종 감사](results/p5_final_audit_20260925_01/REPORT.md)다. 원본 반환 completion의 `p5_complete:false`는 그대로 보존한다.
