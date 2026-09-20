# v1.4 P3 — deepwide12_read4 seed 0 실행 전

2026-09-20 기준. `next_architecture_proposal.json`을 정식 v1.4 계약으로 승격했으나,
**GPU smoke와 seed 0 64M 학습은 아직 실행하지 않았다.** 이 문서는 P3 완료나 행동 gate 통과를
주장하지 않는다.

## 선행 조건

- [x] v1.3 confirm 반환 ZIP SHA-256과 내부 감사 `passed`를 확인했다.
- [x] v1.3 seed 0 행동 gate 실패와 test 미개봉을 확인했다.
- [x] v1.4 아키텍처·예산·선택·gate·replication 규칙을 test 열람 전에 동결했다.
- [ ] v1.4 P1 corpus 생성과 postwrite audit가 통과했다.
- [ ] v1.4 P2 unit test와 CPU/GPU smoke가 통과했다.
- [ ] Colab input bundle과 notebook의 hash를 검증했다.

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

- 새 split과 64M train stream의 manifest/audit/hash
- 12×256 forward/backward, masking, resume, 평가 경로 CPU/GPU smoke
- 실제 GPU 모델·환경 ID·microbatch·처리량·peak VRAM
- seed 0 init/milestone/last checkpoint와 config/code/data hash
- 선택 checkpoint, select 곡선, one-time gate, 실제 tokens/cursor/overshoot

위 증빙을 모두 회수·검증하기 전에는 체크박스나 registry를 `passed`로 갱신하지 않는다.

## 준비 작업 계속 실행

최초 64,007,030-token 코퍼스는 historical READ-prefix 누락으로 승인하지 않는다.
원본을 보존한 수정본 재생성과 후속 로컬 검증이 진행 중이다.
현재 데이터 경로는 불변 설계 원본에 대한 운영 정정 `corpus_rebuild.json`을 따르며,
상세 근거는 `P1_STATUS.md`, 구현 부분 검증은 `P2_STATUS.md`에 기록한다.
최종 runtime config가 생성되고 GPU smoke가 통과하기 전에는 학습을 시작하지 않는다.
