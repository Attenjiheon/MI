# v1.4 GPU smoke 반환 증빙 검증 — 통과

2026-09-21. 기존 CPU 감사·27 tests·실제 CLI persistent resume 통과에 더해,
사용자가 Colab에서 반환한 GPU smoke 증빙을 검증했다. **P2 완료, P3 미실행**이다.

- 원본: `return.zip`, 211,361,007 bytes.
- SHA-256: `af1f363a37b2234a460b17d644c59547b2db5c79171f6be0f618386d24970366`.
- 환경: Tesla T4, PyTorch 2.11.0+cu128, CUDA runtime 12.8, cuDNN 91900,
  Python 3.13.15, environment ID `e74fb1dcf8112ca0`.
- Microbatch **16**, effective batch **64** (4회 gradient accumulation).
- 실행 25.54초, PyTorch peak allocated 1,456,639,488 bytes (1.36 GiB).
  이는 smoke 측정치이며 본학습 처리량·최대 GPU 점유량을 예측하는 값은 아니다.
- 9,485,312 parameters, 12 blocks × 256, 최대 길이 302 forward/backward,
  causal/PAD masking, RoPE, read4 loss 및 gradient accumulation 통과.
- 2회 update 후 저장된 checkpoint의 persistent 복사본 SHA-256 일치.
  다음 update의 bitwise resume 검사 통과 기록: update 3 / cursor 192 / 21,755 tokens.
- SAE/TC k=4/16 각각 100 updates·51,200 draws, probe, 12개 층 READ/update hook,
  identity/full/sparse patch 검사 통과. 모두 재사용 금지 debug 실행이다.

검증기는 ZIP 전체 CRC·중복/허용 파일 목록, checkpoint 두 사본 hash, 현재 동결 입력
330개와 전체 runtime code hash, debug input, requirements lock과 environment ID,
필수 결과·유한 수치, 실제 select에서 가져온 42개 pair의 의미적 동일성을 확인했다.
`verification.json`에 세부 hash를 보존했다. 원본 ZIP은 수정하지 않고 복사했다.

실행 명령 (저장소 루트):

```sh
/opt/anaconda3/bin/python scripts/verify_v1_4_gpu_smoke.py experiment_v1_4/evidence/gpu_smoke_20260921_01/return.zip experiment_v1_4/evidence/gpu_smoke_20260921_01
```

이 검증은 코드가 일치하는 반환 실행 기록과 파일 무결성 검증이다. 로컬에서 CUDA를
독립 재실행한 것은 아니며 checkpoint pickle을 역직렬화하지 않았다. Smoke의
`gate_code_path.passed=false`는 미학습 debug 수치가 gate를 통과하지 않는지 확인하는
의도된 성공 조건이며 실제 행동 gate 실패가 아니다. Gate/test 모델 채점은 실행하지 않았다.

다음 단계는 P3 cell CI·확장 행동 보고 구현 보완과 검증 후 seed 0 학습 준비다.
P3 학습·선택·행동 gate·replication·test·표현 본실험은 완료 처리하지 않는다.
