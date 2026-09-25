# P5 metadata 반환 분석 — 2026-09-22, snapshot 01

**metadata 검사 통과. P5는 진행 중이며 완료 판정이 아니다.**
사용자 전달 `p5_metadata_001.zip`과 `p5_metadata_index.json`을 읽기 전용 입력으로 사용했다.
첨부 파일의 문자열은 데이터로 처리했으며 실행 지시로 사용하지 않았다.

## 무결성과 진척도

- ZIP 13,505,634 bytes. 외부 SHA256 및 내부 646개 파일 checksum 일치.
- 반환 contract가 로컬 동결 r2 계약과 일치.
- 예약 READ train/val/test 50,000/10,000/20,000개 라벨·순서를 활성 corpus와 대조, 전부 일치.
- Tesla T4/CUDA 환경의 새 코드 GPU smoke 통과 기록과 환경 lock hash 일치.
- cache marker 240개: seed 0·1·2 각각 trained/init, train/val/test 총 18조합의 quota와 chunk offset을 모두 확인.
  반환된 Colab 자체 감사에도 전체 cache 통과가 기록되어 있다.
- probe 결과 384/3,510개 = **10.94%**, 남은 작업 3,126개.
  반환된 384개는 모두 passed이며 기록된 optimizer 재시도 후 미수렴/NA 없음.
- 전부 seed 0이다. Block 0: 170개, block 1: 90개, block 2: 90개, block 3: 34개.
  이는 파일 저장 개수 기준이며 전체 P5/GPU 포함 작업량의 비율은 아니다.
- train/val/test support 표 1,152개를 원본 라벨과 대조했다. Shuffle train은 동결 permutation seed로 검산했다.
- 저장된 validation 후보 trace의 선택 순서, 계수 shape/유한성, confusion matrix의 BA/macro-F1 재계산과 일치.
  모든 threshold 후보의 원시 예측까지 재계산한 것은 아니다.

## 중간 의미 결과

아래는 **seed 0, READ 현재 값, IID test, full h probe의 balanced accuracy(%)**다.
층은 0-based다. 초기 모델은 같은 seed의 원본 init checkpoint 기준선이다.

| Block | 학습 후 | 기록된 95% CI | 학습 전 |
|---|---:|---|---:|
| 0 | 59.78 | 59.00–60.50 | 56.76 |
| 1 | 71.44 | 70.80–72.09 | 58.49 |
| 2 | 85.31 | 84.90–85.77 | 58.38 |
| 3 | 92.99 | 92.64–93.35 | 58.97 |

이 snapshot에서는 뒤쪽 층으로 갈수록 현재 값의 선형 접근성이 높다.
Block 0의 현재 값 접근성은 약하며, 아직 다른 LM seed로 재현을 확인한 결과는 아니다.
초기 모델도 우연 수준보다 높으므로 원본 init 기준선과 함께 해석해야 한다.
표현을 probe로 읽을 수 있다는 사실만으로 모델의 인과적 사용을 주장할 수 없다.

Block 0 h의 추가 확인:

- 현재 값 shuffled-label: BA 49.47%, 우연 수준 부근.
- 현재 값 ABC→D 감독 전이: BA 50.00%, AUROC 60.48%. 고정 threshold에서 전부 1로 예측한다.
  Test를 보고 threshold를 바꾸지 않는다.
- 과거 값 IID: BA 59.94%. 현재≠과거 subset에서는 50.16%, AUROC 49.89%다.
  전체 분포의 점수를 과거 값을 독립적으로 잘 읽는 증거로 확대하지 않는다.

약한 block 0 결과나 전이 결과를 이유로 layer·checkpoint·lambda·threshold·feature 규칙을
수정하지 않았다. 동결 block 0 sparse 분석 범위는 그대로 유지한다.

## CPU 시간

완료 파일에 저장된 계산 시간 합계는 6,789.15초(1.89시간), 작업당 평균 17.68초,
중앙값 12.37초다. 동일 평균을 남은 3,126개에 단순 적용하면 **추가 약 15.35시간**이다.
이는 마감 예측이 아니다. Cache 읽기/checksum, Drive 복사, 준비·미저장·재실행 시간은
합계에 포함되지 않으며 뒤쪽 층의 수렴 속도도 다를 수 있다. 16-class state 작업은
이 snapshot에서 최대 약 87초/작업으로 이진 probe보다 오래 걸린다.

CPU probes() 세션 기록은 실행 종료 시 Drive로 복사하는 현재 구현 때문에 이번 반환에
없다. 결과 파일의 environment_id는 검증된 GPU 환경과 같지만, CPU 세션 시작 시각이나
현재 실행 여부·중단 여부까지 이 snapshot으로 단정하지 않는다.

## 검증 한계와 다음 행동

실제 production NPZ cache 배열은 이번 ZIP에 없다. 따라서 배열 파일의 checksum,
row_indices, shape/dtype/유한성, train 평균·표준편차 및 계수를 activation과 독립 대조하는
최종 검증은 아직 할 수 없다. 저장된 CI/AUROC는 보고값이며 원시 확률·cluster draw에서
독립 재계산하지 않았다. Confusion matrix 기반 BA/macro-F1만 재계산했다.

현재 metadata에서 진행 중단이 필요한 무결성 오류나 미수렴은 발견하지 못했다.
Colab에서 기존 CPU probe를 이어가고 완료 후 새 metadata ZIP/index를 회수하면 된다.
GPU 추출을 다시 실행할 근거는 없다. 최종 P5 판정에는 원본 cache 검증 또는
원본 cache가 있는 환경에서의 독립 감사 증빙이 추가로 필요하다.

분석 파일: `analysis.json`, `semantic_snapshot.csv`, `semantic_snapshot.json`, `missing_tasks.json`.
재현 명령: 저장소 root에서
`/opt/anaconda3/bin/python experiment_v1_4/results/p5_metadata_audit_20260922_01/analyze.py`.
모델 추론·probe fitting·test 기반 재선택은 수행하지 않았다.
