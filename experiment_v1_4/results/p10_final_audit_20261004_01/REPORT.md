# P10 완료 — 2026-10-04

Update block 3의 LM seed 0/1/2 × SAE/TC × k=4/16, sparse seed 0의 **12개 학습·평가·원본 재현 반환 감사가 통과**했다. P10은 완료이며 P11 최종 집계는 남았다.

최종 반환 ZIP·checksum과 1,529개 파일 hash, 동결 감사/평가 계약 및 원래 반환 manifest 연결을 검증했다. 240개 GPU chunk의 위치 수를 예약 라벨의 sequence별 위치 수에서 독립 계산하여 대조했다. Trained/init 모델 6개 전체 480,000 position visits와 dictionary 12개의 960,000 latent positions를 확인했다. 반환 기록상 h/u/m 및 latent 최대 절대 오차는 모두 0이다.

1,248개 CPU refit의 task identity·원본 fit/report hash·선택 계수를 대조했고 총 2,016개 예측·CI 재현 보고의 누락이 없었다. 원본 감사 코드는 고정 계수 예측, sequence bootstrap 1,000회와 paired full-probe 차이까지 재현한다. 로컬 검토는 GPU를 재실행한 것이 아니라 동결 코드·입력에 연결된 실행 증빙의 coverage와 결과를 확인한 것이다.

과제별 기록 시간 합: GPU activation 90.75초, dictionary 77.24초, CPU refit·예측·CI 16,562.22초(약 4시간 36분). 설치·Drive I/O·대기·중복 재개를 포함한 총 경과 시간 또는 CU 소비량은 아니다. 환경 ID는 `6e859d41cc64e591`(GPU), `9ee13b7194181dfb`(CPU)다. CPU 시간 제한 정지 두 번 뒤 재개 완료했으며 최종 반환 로그에는 traceback이 없다. 과거 본평가의 조기 evaluate gate 오류와 중간 반환도 보존했다.

P10 학습량은 60,000 updates·30.72M draws이며 252 checkpoint·240 validation 수치 검산을 통과했다. 추가 층·length·m→m SAE·Update 인과 개입은 동결 결정대로 생략했다. Update는 READ 결과 후 선택된 탐색 분석이며 변수/연산 전이는 probe의 감독 전이다. 이 결과로 dictionary domain holdout이나 feature의 인과적 사용을 주장하지 않는다.

검증: `/opt/anaconda3/bin/python scripts/audit_v1_4_p10_source_return.py`.
[완료 판정·hash](completion.json), [원본 반환](returned/return_manifest.json), [학습 감사](../p10_training_audit_20261003_01/completion.json), [평가 반환 감사](../p10_return_audit_20261004_01/status.json).

로컬 회귀 검증: `python -m pytest -q tests_v1_4/test_p10.py tests_v1_4/test_p10_evaluation.py tests_v1_4/test_p10_source_audit.py` — 20 passed (4.36s).
