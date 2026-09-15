# P3 종료 보고 — 행동 학습 단계에서 중단

판정일: 2026-09-16 (KST). 규격: experiment-spec-v1.0.

## 결론

Seed 0을 규정대로 1M에서 누적 3M까지 한 번 연장했으나, 최소 일반 validation 답 CE checkpoint가 모든 행동 gate에 미달했다. P3 실행·판정은 종료했으며 **행동 gate 상태는 failed**다. P4 및 표현 분석으로 진행하지 않는다. SAE/TC 가설은 검증하지 않았다. 데이터·모델·예산을 변경하거나 다른 checkpoint/seed로 실패를 대체하지 않았다.

| 선택 checkpoint 평가 | 1M | 3M 최종 | gate |
|---|---:|---:|---:|
| 일반 validation 정확도 | 49.32% | 61.44% | ≥99% |
| 다른 변수 읽기 | 53.32% | 61.52% | ≥95% |
| 복수 갱신 | 54.88% | 64.26% | ≥95% |
| 최신 SET/초기화 이후 첫 READ | 53.12% | 61.33% | ≥95% |
| 일반 답 CE (nats) | 0.74522305 | 0.64695066 | checkpoint 선택 기준 |

일반 validation은 512 sequences / 8,171 READ, 진단은 각각 512개 독립 sequences / 지정 READ 512개다. 모든 정확도는 full-vocabulary argmax이며 test 점수는 사용하지 않았다.

## 연장 및 종료 근거

1M의 마지막 네 일반 답 CE는 `[0.85200430, 0.80378148, 0.76228187, 0.74522305]`다. 세 변화 모두 1e-4 nats 이상 개선하여 규정상 연장이 허용되었다. 초기 136 updates에서 1,002,099 tokens / cursor 8,704를 처리했다.

3M의 마지막 네 CE는 `[0.66104152, 0.65541271, 0.65472835, 0.64695066]`다. 개선은 계속됐지만 **규격은 3M 이후 추가 연장을 허용하지 않는다**. 최종 407 updates / 3,004,531 tokens / cursor 26,048, overshoot 4,531이며, 선택 update는 407이다. 데이터 중복 소비와 validation 누락은 발견되지 않았다.

선택 checkpoint SHA-256: `066126c223478c78d842ca52a2b30d4ee375c9687111619bcaf970add0e1936f`.

## 실행 비용·환경

- Tesla T4, float32, microbatch 16 / effective batch 64; 환경 ID `e74fb1dcf8112ca0`.
- Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900. P2와 동일 환경 ID이고 현재 실행 코드의 GPU smoke도 통과했다.
- 첫 50 updates: 371,385 tokens / 학습 전용 3.5349초 / 105,062 tokens/s / peak allocated VRAM 166,763,008 bytes (약 159.04 MiB).
- 두 학습 세션 합계 198.11초. 로그의 update 학습 시간 합계 28.87초, 경계 validation 합계 131.03초. 나머지는 선택 checkpoint 재평가·저장 등이며 별도 분해하지 않았다. GPU preflight와 업로드 시간은 세션 합계에 포함되지 않는다.

## 검증

`p3_evidence_verification.json`은 원본 ZIP checksum, 코드/config/data hash, GPU smoke, checkpoint model/optimizer/RNG/progress, token/cursor/update, warmup LR, 최소 CE 선택, 연장 조건과 첫 50-update 측정을 대조한다. CPU 재평가의 일반 답 CE는 0.6469506576443926으로 Colab과 약 3.6e-10 nats 차이고 정확도는 모두 동일하다. 일반 B0/B1 확률질량은 0.996513이며 binary 제한 정확도도 61.44%다.

`p3_cpu_corpus_reaudit.json`은 immutable 원본을 변경하지 않고 51,543 sequences와 3,072 causal pairs를 독립 replay·hash·누출·metadata·quota 규칙으로 재검사해 통과했다. 이 검사는 test 정답의 생성 무결성 확인이며 모델의 test 성능을 평가한 것이 아니다. 구현 테스트 23개도 통과했다 (`p3_postrun_pytest.txt`).

현재 확인한 범위에서 데이터·학습/평가 계약의 오류는 발견되지 않았다. 이것이 구현 오류의 완전한 부재를 보장하지는 않는다. 결과만으로 실패 원인을 모델 용량, 예산, 최적화 또는 과제 난도로 특정할 수 없다. 현재 결과가 뒷받침하는 결론은 지정 v1.0 설정·예산에서 seed 0이 행동 gate에 미달했다는 것이다.

## 보존과 한계

- 원본: `experiment_v1/evidence/p3_evidence_20260915T174256751247.zip`, SHA-256 `045e00dea98bbad101f18bfc2b4879ea806bf555a3e208fcfa83a19b6c8b96ed`.
- 추출본: `experiment_v1/evidence/p3_colab_20260915T174256751247/`; 원본 Colab 절대 경로는 manifest에 보존하고 로컬 경로는 registry로 연결했다.
- 최종 init/best/last를 보존했다. 1M 당시 best/last 바이트는 이 ZIP에 없고 1M 결정의 hash·수치·로그만 남아 있다. Drive snapshot 전체는 제출되지 않아 그 저장본의 원자적 복구는 이번 감사 범위 밖이다.
- 실패 seed를 교체하지 않았다. Seed 1/2, P5–P9 필수 해석, P10 선택 분석은 선행 행동 gate 미달로 미실행이다. 전체 실험 성공 또는 SAE/TC 분석 완료로 집계하지 않는다.
- 향후 v1.1은 별도 설계·config/version과 재검증이 필요하다. 이번 작업에서 새 실험을 시작하지 않았다.

## 재현 명령

프로젝트 루트에서 다음을 실행한다 (원본 데이터/학습 결과는 읽기 전용).

```bash
.venv-p2/bin/python scripts/verify_p3_evidence.py
.venv-p2/bin/python scripts/recheck_p3_corpus.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-p2/bin/python -m pytest tests -q
```

학습 원명령은 각 `manifest_1000000.json`, `manifest_3000000.json`에 있다. Colab 실행 문서는 `notebooks/01_colab_lm_upload_fix.ipynb`다. 중단 판정 이후 이 run을 추가 학습하지 않는다.
