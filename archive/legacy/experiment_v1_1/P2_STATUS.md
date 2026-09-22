# 4-block P2 상태

상태: `passed` — CPU와 Colab GPU에서 새 4-block 코드의 smoke 통과.

CPU: `smoke/cpu_01/smoke.json`, 환경 `d7c07004dd4c1bd4`.
GPU: `evidence/p3_colab_20260916T085123549826/preflight/20260916T085123549826/smoke/smoke.json`, 환경 `e74fb1dcf8112ca0` (Tesla T4).

797,184 parameters, padding/causal mask, token-weighted accumulation, 네 block READ/update hook, dictionary/probe/patching 및 checkpoint 재개 검사가 통과했다. GPU smoke의 코드·config hash가 본실험 manifest와 일치함은 `results/p3_evidence_verification.json`에서 확인했다. Debug checkpoint는 본실험에 재사용하지 않는다.
