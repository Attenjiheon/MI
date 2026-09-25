# P5 中간 metadata 분석 — 2026-09-23

**metadata 검사 통과, P5 진행 중.** 이번 snapshot은 550/3,510개(15.67%)이며 남은 작업은 2,960개다.
이전 384개에서 166개가 추가되었고, 이전 결과 파일은 바이트 단위로 전부 동일하다.
저장된 550개 모두 passed이며, 기록된 미수렴·NA는 없다. Passed는 계산 성공을 뜻하며 연구 가설 통과가 아니다.

## 검사 범위

- 외부 ZIP SHA256 `b5f7e6c6352ab7785a385b49e09b1887fb62ed84f44ea0c2a4a6dd04072ce010`과 내부 824개 파일 checksum 일치.
- 반환 r2 계약, train/val/test 예약 READ 50k/10k/20k의 라벨·순서를 로컬 동결 입력과 대조했다.
- GPU cache 240개 marker의 checkpoint·위치 수·offset·label hash 및 GPU smoke/환경 lock 기록 일치.
- Probe support 표 1,650개, 저장된 validation trace의 선택 규칙, 계수 shape/유한성,
  confusion matrix 기반 BA/macro-F1 재계산 일치.
- 새 CPU session 4개와 환경 ID `df8605acd754d4e2` 확인. GPU가 없는 CPU 환경이다.
  session 수를 오류 횟수로 해석하거나 현재 실행 상태를 단정하지 않는다.
- 여전히 production NPZ가 없어 배열 bytes/shape·계수·전처리 통계·원시 확률 기반 AUROC/CI의
  독립 최종 검증은 남아 있다. 이 보고서의 CI는 저장된 값이다.

## 진행 범위

모두 seed 0이다. Block 0: 170개, block 1–4: 각 90개, block 5: 20개.
Seed 1·2 결과는 이번 snapshot에 없다. 누락을 실패로 처리하지 않았다.

## 중간 표현 분석

READ 현재 값의 IID test, full h probe balanced accuracy(%):

| Block | 학습 후 | 저장된 95% CI | 학습 전 |
|---|---:|---|---:|
| 0 | 59.78 | 59.00–60.50 | 56.76 |
| 1 | 71.44 | 70.80–72.09 | 58.49 |
| 2 | 85.31 | 84.90–85.77 | 58.38 |
| 3 | 92.99 | 92.64–93.35 | 58.97 |
| 4 | 97.56 | 97.37–97.77 | 58.85 |
| 5 | 99.15 | 99.00–99.28 | 아직 미반환 |

현재까지 후반으로 갈수록 현재 값의 선형 접근성이 높아지는 패턴이다. Block 5까지의
seed 0 결과이며 전 seed/전 층 결론이나 인과적 사용의 증거는 아니다.

- ABC→D 현재 값 감독 전이 BA: block 4 97.68%, block 5 99.37%.
- 과거 값 IID BA: block 4 87.13%, block 5 88.45%.
- 과거 값 현재≠과거 subset BA: block 4 87.58%, block 5 88.71%.

층 번호는 0-based다. block 0 결과가 약하다는 이유로 동결 sparse 분석층을 바꾸지 않는다.
위 test 결과는 보고만 했으며 모델·threshold·lambda·선택 절차를 바꾸지 않았다.

## 시간과 대책에 대한 함의

저장된 계산 시간 합계는 3.14시간이다. 이는 실행 전체 경과 시간이 아니다.
이전 384개 평균은 17.68초/작업, 새 CPU 환경의 166개는 평균 27.13초,
중앙값 15.29초, 가장 긴 작업은 약 191초다. 두 집합의 층/라벨 구성이 달라
환경 변경만이 느려진 원인이라고 단정할 수 없다.

최근 166개 평균으로 남은 2,960개를 단순 외삽하면 추가 약 22.3시간의 계산이다.
I/O·checksum·Drive 저장·재시작 비용은 별도며, 실제 남은 작업의 수렴 시간도 다를 수 있다.
이전 15시간 단순 추정을 고정된 완료 시간으로 사용하면 안 된다.

성능 대책 검토에서 코드상 확인한 사항은 전체 NPZ의 SHA256을 hook/layer마다 반복 계산하는
구조, 완료된 층에서도 먼저 cache를 읽는 재개 구조, lambda마다 반복하는 상수 열 통계,
threshold 선택에 불필요한 AUROC 재계산이다. 최적화는 이 중복을 줄이는 쪽을 우선하며,
표본 수·lambda grid·float64·bootstrap 횟수는 아직 변경하지 않았다.
새 최적화 실행기는 아직 배포되지 않았다. 이미 계산된 550개를 재실행할 근거는 없다.

재현: `/opt/anaconda3/bin/python experiment_v1_4/results/p5_metadata_audit_20260923_01/analyze.py`.
원본 계수·상세 점수는 `semantic_snapshot.csv`, 진척은 `analysis.json`, 이전 대비는 `comparison.json`에 있다.
