# v1.4 P2 — 정식 CPU·GPU smoke 통과

2026-09-21. P1 통과 코퍼스와 동결 config를 사용한 CPU 범위가 통과했다.
**실제 GPU smoke 반환 증빙도 검증되어 P2 완료다.**

- `results/local_preparation/tests.xml`: 기존 전체 18 tests passed.
- `results/audit_20260921_01/tests_final.xml`: 감사 회귀 검사를 포함한 **27 tests passed**.
- `smoke/cpu_01/smoke.json`: 정식 CPU smoke passed, frozen input verified,
  fixture_only=false, 27.30초, env `3c50b28adae72d67`.
- 9,485,312 parameters, 12×256, masking/RoPE/read4 accumulation, 최대 길이 302,
  bitwise update resume, 전 층 hook, SAE/TC k=4/16 각 100 updates, probe/patch 경로 확인.
- `smoke/resume_01/resume_equivalence.json`: 실제 CLI full/pause/persistent recover/resume
  실행에서 모델·optimizer·RNG·cursor·microbatch·다음 평가 경계 bitwise 일치.
- `results/audit_20260921_01/existing_cpu_verified.json`: 현재 frozen config/corpus/code hash를
  다시 검증하고 두 실제 CLI run을 evidence verifier로 재검증했다. 모델 코드 변경이 없어
  기존 정식 smoke 증빙을 재사용했다. 추가 감사로 모델 gate/test를 채점하지 않았다.

[전체 감사 보고서](results/audit_20260921_01/REPORT.md)에 원본 code hash 보존, 생성 시도 상한
수정·전량 재현, pair 의미 검사 보강을 기록했다. 이전 fixture smoke와 corpus 부재 당시
15 passed / 3 failed 결과는 별도 경로에 보존하고 정식 통과로 바꾸지 않았다.

GPU 반환 검증: [보고서](evidence/gpu_smoke_20260921_01/REPORT.md),
`verification.json`, `smoke.json`, 원본 `return.zip`.
Tesla T4 / env `e74fb1dcf8112ca0` / microbatch 16 / effective batch 64,
25.54초 / peak allocated 1.36 GiB. 입력 330개·runtime code hash, 환경 lock,
checkpoint persistent 복사본 및 필수 smoke assertions가 일치했다.
P3의 cell CI 및 확장 행동 보고 구현도 별도 보완 대상이며 본 실험 완료를 주장하지 않는다.
