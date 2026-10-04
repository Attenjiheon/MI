# P10 원본 감사 중간 반환 — 2026-10-04

판정: **GPU 통과·CPU 일시정지**, P10 미완료.

두 반환 ZIP의 외부 checksum·262/764개 member hash와 공통 파일 동일성을 확인했다. 동결 audit/evaluation/return identity와 GPU 완료 manifest를 대조했다. GPU 240 chunks의 480,000 position visits, 6개 trained/init 모델의 split별 quota, dictionary 12개의 checkpoint·80,000 위치를 확인했다. 반환된 activation 및 latent 최대 오차는 모두 0이다.

CPU는 계약 순서의 첫 496/1,248개 task를 완료했고 752개 semantic reports를 재현했다. 각 task의 원본 fit/report hash와 refit 선택 계수를 동결 결과에 대조했다. 완료 task 계산 시간 합은 7,238.10초다. 로그는 `PAUSED CPU source audit; rerun same cell`로 끝나며 전체 완료 파일은 없다. 나머지 752개 task를 실행해야 한다.

초기 CPU에서의 GPU 노트북 실행은 파일 hash 검증까지만 진행된 로그로 보존했다. 이후 T4 GPU 실행이 완료되어 재실행할 필요가 없다.

기존 CPU 감사 노트북에서 같은 감사 실행 셀을 반복한다. 런타임이 종료되었으면 CPU 노트북 맨 위부터 실행한다. Drive 원본과 감사 출력은 유지한다. 전체 완료 뒤 새 ZIP/checksum을 반환한다. 아직 P10 완료·Git 완료 commit/push는 수행하지 않는다.

검증 명령: `/opt/anaconda3/bin/python scripts/audit_v1_4_p10_source_partial_return.py`.
[기계 판정과 증빙 hash](status.json).
