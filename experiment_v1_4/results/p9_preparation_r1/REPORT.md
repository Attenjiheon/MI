# P9 학습 전달물 준비 검증 — 2026-09-30

**로컬 준비 검증 통과. P9 미완료, 실제 학습 0/16·평가 0/16.**

P6/P7/P8 완료 판정에 연결된 증빙 SHA-256과 seed 0의 12개 scalar 통계 파일을 검증했다.
기존 P6/P8 학습 규칙을 계승하고 sparse seed 1의 초기화·sampler 난수를 고정했다.
총 16 runs/80,000 updates/40.96M draws이며 동일 layer/k의 SAE·TC draw seed는 같다.
평가 규칙·128후보·기준선·재구성 target은 기존 계약을 계승하며 feature와 probe는 dictionary별로 다시 선택한다.

- 로컬 19 tests 통과, 독립 압축해제 번들에서도 19 tests 통과.
- 네 층 hook·dictionary·패칭 CPU smoke와 16개 도구/층/k 조합의 정확한 재개 통과.
- SAE h→h 및 TC u→m의 독립 update 재현·validation MSE 계산 통과.
- 노트북 schema와 모든 코드 셀 문법, ZIP의 모든 계약 대상 hash 검증 통과.
- CUDA 본실행·Drive 원본 cache 검증은 아직 수행하지 않았다.

[검증 수치·hash·명령](verification.json), [사용자 실행 안내](../../P9_STATUS.md)를 따른다.
학습 결과와 독립 수치 검산을 반환받아 선택 checkpoint를 감사한 뒤 전체 평가 전달물을 연결한다.
P9 완료 gate가 남아 있으므로 완료 커밋/main push를 수행하지 않았다. 기존 용량 정리 변경은 보존했다.
