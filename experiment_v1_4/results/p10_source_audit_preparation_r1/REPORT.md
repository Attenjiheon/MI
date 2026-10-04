# P10 원본 재현 감사 준비 — 2026-10-04

판정: **passed_preparation_only**. 독립 압축 해제 환경에서 기존 평가 번들과 추가 번들의 파일 hash, 테스트 7개, 노트북 2개의 schema·코드 구문을 검증했다. 실제 원본 재현 감사는 0/12이며 P10은 미완료다.

[동결 계약](../../p10_audit_r1/contract.json)은 GPU 허용오차와 원본 결과 식별자를 재현 실행 전에 고정한다. T4에서 trained/init 모델 6개의 예약 80,000 위치 전체(총 480,000 position visits·54개 raw arrays)를 재추출하고, 12개 dictionary의 960,000 latent positions·fidelity·train 통계를 독립 계산한다. CPU에서는 원래 train/validation domain으로 1,248개 refit을 대조하고 고정 계수의 2,016개 예측·sequence bootstrap CI·paired 차이를 재현한다. 기존 선택과 결과를 덮어쓰지 않는다.

새 입력은 302,942 bytes의 overlay ZIP이다. 기존 469MB 평가 ZIP과 Drive의 `P10_update_evaluation_r1` 원본 전체를 재사용한다. 새 감사 출력은 `P10_update_source_audit_r1`이며 완료된 chunk/dictionary/task 영수증부터 재개한다. 시간 제한은 기본 2시간이며 PAUSED이면 같은 실행 셀을 반복한다.

[검증 JSON](verification.json), [독립 테스트 로그](standalone_tests.txt), [실행 안내](../../P10_STATUS.md).

검증 명령은 verification.json에 기록했다. 로컬 테스트는 실제 CUDA replay나 전체 원본 refit의 완료 증빙이 아니다. 사용자 반환 증빙을 검증한 뒤에만 P10 완료와 Git 업데이트를 수행한다.
