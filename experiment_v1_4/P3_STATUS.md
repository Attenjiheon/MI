# v1.4 P3 — deepwide12_read4 seed 0 실행 전

2026-09-21 기준. `next_architecture_proposal.json`을 정식 v1.4 계약으로 승격했으나,
**GPU smoke 반환 검증은 통과했으며 seed 0 64M 학습은 아직 실행하지 않았다.** 이 문서는 P3 완료나 행동 gate 통과를
주장하지 않는다.

## 선행 조건

- [x] v1.3 confirm 반환 ZIP SHA-256과 내부 감사 `passed`를 확인했다.
- [x] v1.3 seed 0 행동 gate 실패와 test 미개봉을 확인했다.
- [x] v1.4 아키텍처·예산·선택·gate·replication 규칙을 test 열람 전에 동결했다.
- [x] v1.4 P1 corpus 생성과 전체 독립 audit가 통과했다 (`results/audit_20260921_01/completion.json`).
- [x] v1.4 P2 전체: 27 tests·CPU smoke·CLI resume·GPU 반환 검증 통과 (`evidence/gpu_smoke_20260921_01/REPORT.md`).
- [x] Colab r2 input bundle 전체·내부 checksum 및 notebook 7개 코드 셀 compile을 검증했다 (`results/audit_20260921_01/delivery.json`).

## P3 동결 계약

| 항목 | 값 |
|---|---|
| candidate / LM seed | `deepwide12_read4` / `0` |
| parameters | 9,485,312 |
| nominal budget | 64,000,000 prediction tokens + final-update overshoot |
| checkpoint selection | select first 42-cell macro answer CE 최소, near-tie `1e-4` |
| gate | 선택 checkpoint에서 1회, 재선택 금지 |
| failure branch | seed 1·2, test, P5 이후 중단 |
| pass branch | 같은 설정으로 seed 1·2; 최소 2 seeds 통과 시 해석 |

## 아직 없는 증빙

- P3 cell CI 및 확장 행동 보고 구현 보완·검증
- 실제 본학습 처리량·peak VRAM (smoke 측정은 완료)
- seed 0 init/milestone/last checkpoint와 config/code/data hash
- 선택 checkpoint, select 곡선, one-time gate, 실제 tokens/cursor/overshoot

위 증빙을 모두 회수·검증하기 전에는 체크박스나 registry를 `passed`로 갱신하지 않는다.

## 이전 준비 경과 (2026-09-20 기록)

최초 64,007,030-token 코퍼스는 historical READ-prefix 누락으로 승인하지 않는다.
원본을 보존한 수정본 재생성과 후속 로컬 검증이 진행 중이다.
현재 데이터 경로는 불변 설계 원본에 대한 운영 정정 `corpus_rebuild.json`을 따르며,
상세 근거는 `P1_STATUS.md`, 구현 부분 검증은 `P2_STATUS.md`에 기록한다.
최종 runtime config가 생성되고 GPU smoke가 통과하기 전에는 학습을 시작하지 않는다.

## 2026-09-21 CPU 감사 결과

수정 corpus 생성과 전체 CPU 감사가 통과했다. 실제 학습량은 64,005,751 tokens,
8,399 updates로 동결됐다. 상세 증빙은 `results/audit_20260921_01/REPORT.md`를 따른다.
기존 r1 notebook/bundle은 보존하며 감사 보완을 포함한 r2를 후속 전달물로 만든다.
GPU smoke와 학습은 미실행이다. Cell CI 및 확장 행동 보고 항목은 P3 결과의 최종 승인 전
보완해야 하며 현재 문서는 P3 완료나 모델 gate 통과를 의미하지 않는다.

## 2026-09-21 GPU 반환 검증

T4 GPU smoke 반환 증빙이 통과했다. Microbatch 16 / effective batch 64,
환경 `e74fb1dcf8112ca0`. 상세 증빙은 `evidence/gpu_smoke_20260921_01/REPORT.md`.
앞선 CPU 감사 절의 GPU 미실행 표기는 당시 상태다. P3 학습은 계속 미실행이며,
cell CI·확장 행동 보고 구현 보완과 검증이 다음 작업이다.
