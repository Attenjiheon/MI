# v1.2 P1 — passed

2026-09-17: 기존 모든 split과 3M train prefix를 byte-identical하게 복사하고 shard 00007부터 같은 분포의 새 표본을 추가했다. 기존 데이터는 수정하지 않았다.

- 35 shards, 138,240 sequences, 16,001,083 prediction tokens.
- 최종 update 2,160, cursor 138,240, overshoot 1,083. 마지막 완전 batch 직전은 16M 미만이다.
- 1M: update 136 / 1,002,099 tokens; 3M: update 407 / 3,004,531; 8M: update 1,082 / 8,001,583.
- 기존·신규 전체 독립 replay, token/metadata 정합성, global hash, holdout, causal prefix, dependency/sensitivity, interpretation quota 검사 통과.
- 768 전이와 복원 성질, 기존 예약 causal counterfactual origin 재계산, 실제 shard-7 seed의 64개 표본 2회 token/metadata/RNG 동일성 확인.
- Train coverage: 16 states / 48 commands / 12 truth-table cells / 25 adjacent operator pairs.
- 전체 생성·감사 289.98초. 실제 seed 정수·RNG state·수락/거부 수는 새 manifest에 저장했다.

증빙: `data/language_v1_2/{manifest.json,cpu_validation.json,postwrite_audit.json,corpus_statistics.json}`, `results/p1_generation.json`, `results/p1_base_reaudit.json`, `results/p1_frozen_training.json`, `results/p1_generation.log`.

재현: `.venv-p2/bin/python -u scripts/generate_p3_16m_corpus.py` (기존 출력 경로가 있으면 거부).
Manifest SHA-256: `87ec2e772f6f239c55b53a485ec861fe257ce15145727fb0b1f4bd9d3bc1af1c`.
GPU 학습이나 행동 gate 통과를 의미하지 않는다.

배포 정리: Finder `.DS_Store`를 manifest/번들 대상에서 제외했다. 원래 manifest/config는 `evidence/prepackaging_v1`에 보존했으며 모든 실제 데이터 leaf hash의 동일성은 `results/p1_release_validation.json`에 기록했다. 생성·전체 감사 결과와 학습 예산은 변하지 않았다.
