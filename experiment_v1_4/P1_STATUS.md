# v1.4 P1 — 수정 corpus 감사 통과

2026-09-21. 활성 root는 `corpus_rebuild.json`의 `data/language_v1_4/rebuild_01`이다.
최초 거부된 `data/language_v1_4/` 산출물은 원래 경로에 보존하며 사용하지 않는다.

- 64,005,751 prediction tokens / 537,536 sequences / 133 shards.
- 마지막 update 8,399 / cursor 537,536 / overshoot 5,751.
- v1.3 prefix 68 shards, 271,936 sequences, 32,004,917 tokens와 136개 파일 byte identity 통과.
- 전체 READ 독립 replay, checksum, token/metadata 정렬, seed, quota/position join 통과.
- 독립 historical loader로 과거 344,948 sequence / 5,287,408 READ prefixes와 누출 검사 통과.
- 토큰 기반 holdout/길이/coverage 및 모든 first/repeat cell·depth·단일 삽입, causal 단일 bit 검사 통과.
- target별 시도 횟수 재현: 10,752 targets 모두 한도 100,000 이내, 최대 58,433.
  원 생성 결과의 RNG 최종 상태·총 시도·거부 횟수와 전부 일치한다.

최종 증빙: [전체 감사 보고서](results/audit_20260921_01/REPORT.md),
[종합 판정](results/audit_20260921_01/completion.json),
[전체 replay](results/audit_20260921_01/corpus/full_replay.json),
[quota/cursor](results/audit_20260921_01/corpus/summary.json),
[토큰 의미·coverage](results/audit_20260921_01/policy.json),
[시도 상한 재현](results/audit_20260921_01/attempts_02/summary.json).

생성 시점 코드와 이후 감사 보완 코드는 별도 snapshot/hash로 보존했다. 생성 데이터나 manifest를
사후 수정하지 않았다. 이전 부분 감사와 실패 원인은 `results/audit_20260920_02/REPORT.md`,
`results/p1_initial_rejection.json`에 남아 있다. Test 파일의 데이터 무결성만 검사했으며 모델 점수는
계산하지 않았다. P2 GPU smoke와 P3는 별도 미완료다.
