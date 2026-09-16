# P3 v1.1 — 4-block seed 0 결과

상태: `failed` — 3M까지 실행 및 증빙 검증 후 행동 gate 미달로 종료.

- 실제 4-block / 797,184 parameters, GPU smoke passed.
- 1M 개선 조건 충족 → 동일 optimizer/RNG/cursor로 누적 3M까지 연장.
- 최종 407 updates / 3,004,531 tokens / overshoot 4,531 / cursor 26,048.
- 선택 checkpoint update 407: 일반 58.90%, 다른 변수 58.98%, 복수 갱신 59.96%, 첫 READ 55.47%. 일반 ≥99%, 세 진단 각각 ≥95% 기준 미달.
- CPU 재평가, 데이터·로그·checkpoint 감사 및 구현 테스트를 수행했다.

## 증빙

- `results/p3_stop_report.md`: 중단 보고와 2-block 비교.
- `results/p3_evidence_verification.json`: hash·실행 계약·CPU 재평가.
- `results/p3_cpu_corpus_reaudit.json`: 독립 CPU 코퍼스 재검산.
- `results/p3_postrun_pytest.txt`: 구현 테스트 25개.
- `evidence/p3_4block_evidence_20260916T085123549826.zip`: 원본 결과.
- `evidence/p3_colab_20260916T085123549826/`: 최종 checkpoint·로그·GPU smoke·환경.
- `results/p3_behavior_validation.csv`, `results/p3_depth_comparison.json`: 학습 곡선·비교.

1M checkpoint 바이트는 제출 ZIP에 없으므로 그 시점의 hash·로그·선택 수치만 검증했다. 최종 checkpoint는 직접 검증했다. 실패 판정으로 P4 및 SAE/TC에 진입하지 않는다. v1.0 실패와 분리 보존하며 새 버전 없이 추가 연장하지 않는다.
