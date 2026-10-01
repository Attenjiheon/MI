# P9 학습 반환 감사 — 2026-09-30

**16/16 학습 반환 감사 통과. 전체 평가는 0/16이며 P9는 미완료다.**

- 원본 ZIP SHA-256과 1,308개 내부 파일 checksum 일치. ZIP 총 1,309 members.
- LM seed 0 × block 0/3/7/11 × SAE/TC × k=4/16 × sparse seed 1의 16 runs.
- 80,000 updates, 40.96M draws, 336 production checkpoint를 직접 확인했다.
- 모든 checkpoint의 model·Adam 상태·단위 decoder norm·RNG cursor·학습 곡선·12개 scalar 통계를 검증했다.
- 320 validation MSE의 독립 CUDA 검산 반환 기록과 저장값 차이는 모두 0이다. 최소 validation MSE/이른 update 선택과 선택 checkpoint hash를 검증했다.
- 현재 소스 CUDA smoke: 네 층 patch/probe/dictionary와 16개 재개 조합의 model·optimizer·sampler bitwise 일치를 반환 payload로 재검증했다.
- 실패 기록 없음. 학습 update 계산 시간 합계 360.69초, validation 8.24초. 저장 I/O를 포함한 총 경과시간으로 해석하지 않는다.

검산 환경은 `e74fb1dcf8112ca0`이며 환경 lock과 GPU smoke를 연결해 확인했다.
원본 cache의 재계산은 Drive의 동결 준비·독립 CUDA 검산 증빙에 근거하며 로컬 재추론을 주장하지 않는다.
로컬 macOS/PyTorch와 Colab 초기화 byte 차이는 진단 기록으로 보존하고 환경 간 bitwise 동일성을 주장하지 않는다.

[검증 상세](verification.json), [선택 dictionary](selected_dictionary_manifest.json), [run별 수치](runs.csv).
재현 명령: `/opt/anaconda3/bin/python experiment_v1_4/results/p9_training_audit_20260930_01/verify_return.py`.
다음은 선택 16개에 대한 전체 의미·fidelity·대체·인과 평가와 초기화 민감도 비교다.
