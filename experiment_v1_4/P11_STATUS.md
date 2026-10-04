# P11 완료 — 2026-10-05

**P1–P11 완료: v1.4 필수 분석 전체와 선택 Update 분석의 최종 집계·보고를 완료했다.**

[최종 보고서](results/p11_final_20261005_01/report.md) · [완료 판정](results/p11_final_20261005_01/completion.json) · [독립 검증](results/p11_final_20261005_01/verification.json)

- 필수 READ 64개, Update 12개 dictionary 및 LM 3개 결과를 모두 보존했다.
- 행동 3,945행, 의미 58,086행, fidelity/대체 2,436행, 인과 152,064행, 층간 차이 53,184행과 13개 그림을 작성했다.
- 1,000회 sequence/origin cluster CI와 기존 paired 층간 차이를 hash 대조 후 통합했다. 저장된 대체 출력으로 192개 층간 paired 차이를 추가했다. 모델 inference·fitting·선택·gate를 반복하지 않았다.
- 개별 LM seed와 평균/최소/최대, LM seed 0의 sparse 초기화 차이를 구분했다. Fidelity 점 추정과 sparse seed 점 차이에 존재하지 않는 CI를 붙이지 않았다.
- 19,362개 BA confusion-matrix 검산 최대 차이 1.11e-16, 인과 대조군 11종, matching 512행과 분모를 확인했다.
- 39 tests 통과. 동일 코드 재집계의 CSV byte identity는 [재현 검증](results/p11_final_20261005_01/reproduction.json)을 따른다.
- 추가 층·length GPU·m→m SAE·Update 인과 개입은 동결 자원 결정으로 생략했다. P10은 탐색적이며 READ 층 확장은 P5 test 관측 이후였다.

실행 계약은 [p11_r1/contract.json](p11_r1/contract.json), 실제 명령·입력/환경/hash는 최종 보고서와 manifest를 따른다. 새 Colab 실행·파일 업로드는 필요 없다.

Git 완료 커밋과 원격 main 일치는 Git 이력 및 [Git 전달 기록](results/p11_final_20261005_01/git_delivery.json)을 따른다. 기존 용량 정리의 미커밋 변경은 이번 phase 커밋에서 제외했다.
