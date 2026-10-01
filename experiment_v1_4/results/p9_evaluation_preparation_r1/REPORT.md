# P9 평가 전달물 준비 — 2026-09-30

**로컬 준비 검증 통과. 실제 평가 0/16, P9 미완료.**

16개 학습의 반환 감사로 고정한 checkpoint를 번들에 포함했다. P9 학습 전에 고정한 평가 규칙을 계승하며,
SAE 8개와 TC 8개의 source/target·통계·패칭 hook·계약·출력 경로를 분리했다.
기존 P7/P8 소스는 수정하지 않았다. P9 복사본은 seed 0만 집계하며 CSV 재개 시 줄바꿈 차이로 실패하지 않도록 검증했다.

- 로컬 24 tests 및 독립 압축해제 번들의 동일 24 tests 통과.
- 실제 선택 checkpoint의 hash/통계/latent 인코딩, 후보·bootstrap RNG 계승 확인.
- 두 도구 모두 feature 선택이 train/validation만 읽는지 확인.
- 네 층·두 k × 두 도구의 CPU smoke에서 identity·mean·approximation·full donor·선택·random latent·좌표·random 방향과 양방향 changed/unchanged 대조 통과.
- 노트북 형식과 모든 코드 셀, 332개 번들 member와 계약 source hash 검증.
- CUDA 본평가는 미실행. Drive 원본 P5 cache 검증과 새 CUDA smoke는 노트북에서 수행한다.

출력은 `P9_evaluation_r1/sae`와 `/tc`이며 전체 결과를 하나의 ZIP으로 반환한다.
모든 선택을 먼저 동결하고 test 의미·fidelity·대체·인과 평가를 수행한다.
기존 seed 0 표와 비교한 초기화 차이는 점 추정치이며 차이 CI 또는 LM 반복으로 해석하지 않는다.
동일 latent ID의 feature 의미를 seed 간 일치로 가정하지 않는다. 누락/지원 부족/미매칭은 NA로 보존한다.

[검증 상세](verification.json), [실행 안내](../../P9_STATUS.md).
평가 반환 및 원본 cache 재현 감사 뒤에만 P9 완료와 main push를 수행한다.
