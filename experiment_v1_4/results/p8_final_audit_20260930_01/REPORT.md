# P8 READ Transcoder 최종 감사 — 2026-09-30

**P8 완료.** LM seed 0·1·2 × block 0·3·7·11 × k=4/16의 TC 24개 학습·의미·fidelity·한 위치 대체·인과 평가 및 원본 재현 반환 감사를 통과했다. P9 sparse seed 1 반복 16 runs는 미실행이며 전체 실험은 미완료다.

## 증빙과 검증

- 학습: [학습 완료 판정](../p8_training_audit_20260929_01/completion.json). 120,000 updates, 61.44M draws, 504 checkpoint, 480 validation 수치/선택 검증.
- 평가 원시 자료: [로컬 감사](../p8_return_audit_20260929_01/status.json). 49,152 causal pairs와 49,152 replacement targets, 3,667,856 causal rows, 57,024 집계 점 추정치를 검사했다. 같은 origin을 여러 run에서 평가한 수이며 독립 origin 수가 아니다.
- 이번 반환: ZIP 1,866,163 bytes, manifest 외 871개 파일의 SHA-256과 외부 checksum 일치. 24 runs 모두 passed, 실패 기록 없음. 계약·코드·입력 hash를 확인했다.
- 원본 P5 cache를 사용하는 840개 고유 probe refit의 선택 feature/계수/threshold를 로컬에서 기존 결과와 대조했다. 재사용을 포함한 총 1,920 probe 작업, 3,360 의미 보고, 1,920,000 fidelity position 평가, 48 validation bin set의 재현 기록을 확인했다.
- 전체 test 예측·의미 bootstrap CI·fidelity·인과/대체 CI·paired 층간 차이·표/그림 재현은 hash 검증한 Colab 감사 반환에 근거한다. GPU 패칭 재실행은 사전 고정된 96 causal pairs 및 384 replacement targets다. 전체 인과 본평가를 두 번 수행했다는 의미는 아니다.
- TC는 u를 encode하고 m을 예측한다. 출력 패칭에 s_m을 적용하며 원본 residual skip을 보존한다. full u/full m, 좌표/random, 전체·128후보 대조와 단일/≤4 feature 결과를 유지했다. Test로 checkpoint나 feature를 재선택하지 않았다.
- 환경 f8912b7ab0ca620d, Tesla T4, torch 2.11.0+cu128. run별 감사 시간 합계 12,239.07초(약 3.40시간)는 CPU refit와 GPU replay 등이 섞인 경과시간이며 순수 GPU 계산시간이 아니다.
- 로컬 관련 테스트 30개 통과. 재검증 명령: `/opt/anaconda3/bin/python experiment_v1_4/results/p8_final_audit_20260930_01/verify_return.py`.

## READ 현재 값의 사후 감독 접근성

전체 후보에서 validation으로 고른 ≤4 TC feature의 test balanced accuracy다. 세 LM seed의 평균과 범위를 표시하며 3개 모델 이상의 독립 반복으로 해석하지 않는다.

| Block | k | 평균 | 최솟값–최댓값 |
|---|---|---|---|
| 0 | 4 | 0.5544 | 0.5358–0.5647 |
| 0 | 16 | 0.5890 | 0.5757–0.5976 |
| 3 | 4 | 0.8834 | 0.8232–0.9457 |
| 3 | 16 | 0.9022 | 0.8532–0.9596 |
| 7 | 4 | 0.9727 | 0.9393–0.9955 |
| 7 | 16 | 0.9912 | 0.9840–0.9980 |
| 11 | 4 | 0.9946 | 0.9890–0.9988 |
| 11 | 16 | 0.9987 | 0.9981–0.9993 |

층별·seed별 전체 수치는 [results_by_run.csv](results_by_run.csv), [층별 집계](results_by_layer.json), 원본 [표](../p8_return_audit_20260929_01/returned_metadata/tables/)와 [그림](../p8_return_audit_20260929_01/returned_metadata/figures/)에 있다. Probe 성능은 선형 접근성/사후 감독 성능이다. 모델의 유일한 계산 경로나 전 위치 회로 복원을 입증하지 않는다. SAE의 h 복원 NMSE와 TC의 m 예측 NMSE는 다른 target이므로 숫자만으로 우열을 판단하지 않는다.

## 다음 단계

필수 dictionary 64 runs 중 48개 학습·평가를 완료했다. 누적 학습 240,000 updates/122.88M draws. P9는 LM seed 0의 네 층 × SAE/TC × k=4/16에서 sparse seed 1의 16 runs(80,000 updates/40.96M draws)와 동일한 전체 평가다. 기존 결과·계약은 보존한다. 이전 보고서의 P8 미완료/감사 대기 상태는 당시 이력이다.
