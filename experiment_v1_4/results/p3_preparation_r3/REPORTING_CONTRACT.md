# P3 reporting revision 1 — 2026-09-21

학습 전 보고 구현 보완이다. `design_config.json`, 동결 config set, corpus, 선택 metric,
near-tie 규칙, 행동 gate와 표본 quota는 바꾸지 않는다. 모델·optimizer 구현도 변경하지 않는다.

- General과 legacy diagnostic: 모든 유효 prediction target의 unweighted CE 및 token 수,
  지정 READ의 full-vocabulary CE/accuracy, B0/B1 accuracy와 probability mass.
- Operator, query variable, answer, block count 8–24, update distance 0/1–2/3+,
  binary operator × truth pattern별 count, CE, accuracy, observed-strata macro, coverage.
  빈 strata의 점수는 JSON null (NA)이다.
- 같은 평가 target에 split majority (동률 B0), 독립 Bernoulli(0.5) 기대 정확도,
  query 변수의 최근 SET/초기화 값, 가장 최근 READ 답 (없으면 B0) 기준선을 적용한다.
  Target이 아닌 이전 READ도 복사 기준선의 문맥이며 미래 READ/SET은 사용하지 않는다.
- First/repeat: 기존 점추정과 gate group 집계를 유지하며 각 cell의 accuracy와 answer CE
  및 42-cell macro에 95% percentile CI를 추가한다. Cell 내부 independent origin을
  1,000회 복원 추출하며 first/repeat는 같은 draw를 공유한다. Macro gap CI도 보고한다.
- RNG: PCG64, SHA256 첫 8 bytes unsigned big endian;
  `20260920|experiment-spec-v1.4|behavior-bootstrap-r1|<cell_id>`.
  사용한 cell별 seed, 요청/유효 draw 수는 각 결과의 uncertainty에 저장한다.
  학습용 Python/NumPy/Torch 전역 RNG는 소비하지 않는다.
- 각 cell의 점수/CI는 추가 gate가 아니다. CI나 기준선으로 checkpoint를 선택하지 않는다.

새 runtime source hash에 맞는 CPU/GPU smoke가 필요하다. 이전 CPU/GPU 증빙과 r1/r2
번들은 그대로 보존한다. 새 노트북은 GPU smoke → fresh seed 0 → 64M → select 선택 →
one-time gate → 영속 index 반환 검증 순서다. Seed 1·2 및 test는 실행하지 않는다.
