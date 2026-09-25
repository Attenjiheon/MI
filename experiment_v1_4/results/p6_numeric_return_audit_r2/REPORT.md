# P6 r2 수치 검산 반환 확인

24 runs의 480 validation checkpoint 수치 검산 증빙 통과. 모든 MSE 절대 오차는 0이다.
P6 전체 반환 감사는 미완료이며 전체 checkpoint ZIP이 필요하다.

- 원본 ZIP SHA256: `6494f0aaafd261f3ce88da3fc20c93f4927689a4597f23dde50126529d7eea59`.
- ZIP 내부 539개 항목의 경로/중복 검사 및 로컬 SHA256 목록을 기록했다. 외부 사전 제공 checksum은 없으며 이 hash는 수신본 식별값이다.
- 반환 검산 코드/배송 manifest를 로컬 검증한 r2 번들과 byte 대조했다. 원본 P6 실행 계약 hash도 일치했다.
- input manifest의 원본 P5 cache receipt 240개·labels·4층·36 statistics/24 prepared tensor 경로를 동결 입력과 대조했다. 실제 tensor/statistics bytes는 이 소형 ZIP에 없다.
- 모든 run의 5,000 updates·2.56M draws 보고, 21개 checkpoint hash 참조, 선택 규칙, run/result hash, checkpoint별 receipt와 completion 집계를 대조했다.
- 24 × 20 = 480개 validation MSE 전부 저장값과 정확히 일치했다. 선택 checkpoint는 저장된 validation MSE 최소/이른 update 규칙에 일치했다.
- Tesla T4, torch 2.11.0+cu128, 환경 ID e74fb1dcf8112ca0. 환경 lock hash/ID도 재계산했다. 검산 경과 시간 1,083.55초.

기존 실패 seed0/layer3/k16/update250에서 r2와 저장 MSE는 모두 0.030997336196491705다.
legacy 계산값 0.03099766511893298도 이전 오류 로그와 정확히 일치했다.
TopK 후보 집합이 달라지는 행 1개를 확인했다. 전체 480개 중 11개 checkpoint에서 legacy 후보 차이가 있었다.
이는 float32 연산 분해 차이가 TopK 선택에 영향을 준 검산 오류였으며, 학습 결과·checkpoint·허용 오차 변경 없이 해소됐다.

이 반환물은 Colab 검산 실행의 결과/receipt와 선택 보고다. 전체 checkpoint/optimizer/RNG,
initialization·curve·statistics 본문, 실제 GPU smoke 및 학습 환경/실패·재개 기록은 포함하지 않는다.
따라서 학습을 재실행할 필요는 없지만 P6 완료 판정·Git 완료 절차는 전체 반환 ZIP 검증 후에 수행한다.
현재 evidence 폴더에는 전체 ZIP의 `.sha256.json`만 있으며 본 ZIP은 없다.
필요 파일: `v1_4_p6_return_after_audit_r2_1790344845310707358.zip`.
예상 SHA256: `766e5b42e15262d9a354aa4aa2279211cb411d0c58ed9131cc25e719ca2d1c02`.

재현: `/opt/anaconda3/bin/python experiment_v1_4/results/p6_numeric_return_audit_r2/verify_return.py`.
첨부 원본의 코드는 실행하지 않았으며 반환 JSON을 데이터로 읽어 대조했다.
