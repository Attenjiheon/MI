# P9 평가 반환 로컬 감사

2026-10-01. **16/16 로컬 감사 통과. 원본 cache 재현 감사 대기로 P9 미완료, P10 진입 불가.**

반환 ZIP SHA-256: `7900644540f916183eaa30b957e2e8f62bfa9972c700364887dc9eff391b77a1`.
외부 checksum과 내부 38,080 파일 hash를 확인했다. 환경 ID는 `f8912b7ab0ca620d`다.

SAE 8개·TC 8개에서 validation 선택, 원시 의미 예측, 후보·대조군·bin, 저장 logits와 지표를 검산했다. 전체 causal pairs 32,768개, READ 대체 targets 32,768개, probe tasks 1,200개, causal rows 2,552,810개, 집계 point estimates 38,016개를 확인했다. 결과표 40,320행을 run 증빙에 대조하고 sparse seed 비교 44,880행을 재생성했다. 비교의 seed 1−0 방향 및 결측 처리를 확인했다.

초기화 비교는 동일 LM seed 0 내 점 추정치다. 별도 LM 반복이나 latent ID 대응, seed 차이 CI로 해석하지 않는다. 원본 cache에서 probe refit·전체 예측/fidelity/CI 및 고정 GPU replay는 아직 실행되지 않았다.

증빙은 `status.json`, `archive_verification.json`, `sae_verification.json`, `tc_verification.json`, `local_runs/`, `metadata_runs_r2/`다. 검증 명령:

```sh
python scripts/audit_v1_4_p9_return.py
python scripts/audit_v1_4_p9_return.py --tool sae
python scripts/audit_v1_4_p9_return.py --tool tc
python scripts/finalize_v1_4_p9_local_audit.py
```

원본 평가 입력 번들을 임시 root로 사용해 동결 hash를 보존하고, run별 임시 압축 해제로 디스크 사용을 제한했다. 다음 실행 파일은 `experiment_v1_4/notebooks/P9/P9_source_return_audit_r1.ipynb`와 대응 source audit ZIP이다.
