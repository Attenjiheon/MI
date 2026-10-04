# P10 원본 감사 두 번째 CPU 중간 반환

2026-10-04. **GPU 통과·CPU 1,033/1,248개 완료 후 정상 일시정지. P10 미완료.**

ZIP과 checksum 파일의 번호는 다르지만 실제 SHA-256과 6,484,061 bytes가 일치한다. 새 반환의 1,307개 내부 파일 hash와 이전 두 반환의 공통 파일 동일성을 검증했다. GPU 240 chunks·480,000 visits·12개 dictionary 완료 기록은 그대로 유지된다.

CPU 계약 순서 첫 1,033개 task의 identity·원본 fit/report hash·refit 선택 계수와 1,586개 report 수를 검증했다. 과제별 계산시간 누계는 14,428.51초다. 최신 로그는 `PAUSED CPU source audit; rerun same cell`로 끝나며 `source_audit_complete.json`은 없다. 남은 task는 215개다.

기존 CPU 노트북의 감사 실행 셀을 반복한다. 새 런타임이면 노트북 위부터 실행하며 Drive 출력은 유지한다. T4 재실행은 필요 없다. 전체 완료 메시지를 확인한 뒤 새 ZIP/checksum을 반환한다.

검증 명령: `/opt/anaconda3/bin/python scripts/audit_v1_4_p10_source_partial_return_02.py`.
[판정·증빙 hash](status.json). P10 완료·Git 완료 commit/push는 대기한다.
