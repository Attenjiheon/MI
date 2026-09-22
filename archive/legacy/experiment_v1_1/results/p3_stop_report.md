# 4-block P3 종료 보고 — 행동 gate 실패

판정일: 2026-09-16. 규격: experiment-spec-v1.1.

## 판정과 2-block 비교

4-block seed 0은 1M 개선 조건을 만족해 누적 3M까지 한 번 연장했지만, 최소 일반 validation 답 CE checkpoint가 일반 및 세 진단 gate 모두에 미달했다. **P3은 failed로 종료하며 P4 및 SAE/TC 분석으로 진행하지 않는다.**

| Validation | 2-block v1.0 | 4-block v1.1 | 기준 |
|---|---:|---:|---:|
| 일반 READ 정확도 | 61.44% | 58.90% | ≥99% |
| 다른 변수 읽기 | 61.52% | 58.98% | ≥95% |
| 복수 갱신 | 64.26% | 59.96% | ≥95% |
| 최신 SET/초기화 이후 첫 READ | 61.33% | 55.47% | ≥95% |
| 일반 답 CE (nats) | 0.646951 | 0.670175 | 최소값으로 선택 |

두 run은 같은 코퍼스·데이터 순서·예산·gate·GPU 환경에서 seed 0을 실행했다. 일반 validation은 512 sequences / 8,171 READ, 각 진단은 512개 독립 sequences / 지정 READ 512개다. 일반 정확도 차이는 −2.53 percentage points다. 모델 깊이와 깊이에 맞춘 출력 초기화 배율이 함께 달라졌고 모델별 반복은 하나이므로, 이것을 4-block의 일반적 열세나 깊이만의 인과 효과로 해석하지 않는다. Test 성능은 평가하지 않았다.

## 실행 및 선택 근거

- 실제 checkpoint: **4 blocks / 797,184 parameters**. 코드·config·코퍼스 hash가 전달한 v1.1과 일치한다.
- 1M: update 136, 1,002,099 tokens, cursor 8,704. 일반 정확도 50.35%, CE 0.73181313.
- 1M 최근 네 CE: `[0.84445811, 0.78661287, 0.75218140, 0.73181313]`. 세 변화 모두 1e-4 이상 개선하므로 3M 연장 규칙을 만족했다.
- 최종: update **407**, **3,004,531 tokens**, cursor **26,048**, overshoot **4,531**. Validation 30회; 모든 update가 64 sequences를 중복 없이 소비했다.
- 최소 답 CE 선택 update는 **407**. 최근 네 CE: `[0.67875483, 0.67629533, 0.67113652, 0.67017510]`.
- 선택 checkpoint SHA-256: `4be8ff4cb2fae15f9459711b647e29c273a934623d79c059435d5452490996e1`.

마지막까지 CE는 개선됐지만 3M 이후 추가 연장은 현 규격에 없으므로 중단한다. 이 사실만으로 추가 학습의 성공을 예측하거나 실패 원인을 학습량으로 확정하지 않는다.

## 환경·처리량

Tesla T4, 환경 ID `e74fb1dcf8112ca0`, torch 2.11.0+cu128 / CUDA 12.8 / cuDNN 91900. 현재 4-block 코드의 GPU smoke를 통과했다. Float32, microbatch 16, effective batch 64.

첫 50 updates는 371,385 tokens를 학습 전용 6.1796초에 처리했다: **60,098 tokens/s**, peak allocated VRAM **244,694,016 bytes** (약 233.36 MiB). 두 학습 세션 합계는 241.12초이며 로그상 학습 update 합계 45.73초, 경계 validation 합계 144.68초다. 선택 checkpoint 재평가·저장 등의 나머지 시간을 세부 분리하지 않았다. 업로드와 preflight 시간은 세션 합계에 포함되지 않는다.

## 검증과 해석 한계

`p3_evidence_verification.json`에서 모델/optimizer/RNG/cursor, 코드·config·코퍼스 checksum, GPU smoke, warmup LR, finite loss/gradient, 경계별 validation, 1M 연장 및 3M 종료 조건, 최소 CE 선택, 첫 50-update 측정을 검증했다. CPU 재평가의 정확도는 Colab과 모두 동일하고 일반 답 CE 차이는 약 9.9e-10 nats다. B0/B1 합산 확률질량은 0.997333이고 binary 제한 정확도도 58.90%이므로 다른 vocabulary token을 답으로 고르는 문제가 주된 오차는 아니다.

`p3_cpu_corpus_reaudit.json`은 51,543 sequences / 3,072 causal pairs의 독립 replay·metadata·holdout·hash·quota 검사를 통과한 증빙이다. 테스트 데이터의 생성 무결성도 검사하지만 모델 test 성능을 평가하거나 선택에 쓰지 않는다. 구현 테스트는 `p3_postrun_pytest.txt`의 25개 통과다.

검사 범위에서 계약·데이터 오류는 발견되지 않았으나 모든 구현 오류의 부재를 보장하지 않는다. **깊이를 4로 늘린 것만으로 현 설정의 gate 실패가 해결되지는 않았다.** 이전에 성공한 4-block 실험과의 차이는 해당 실험의 학습 예산·update 수·손실·데이터 길이·규칙을 대조해야 판단할 수 있다. 자동으로 예산을 늘리거나 새 학습을 시작하지 않았다.

## 보존과 다음 단계

원본 ZIP: `experiment_v1_1/evidence/p3_4block_evidence_20260916T085123549826.zip`, SHA-256 `4aa4d5c83af4afc1a53f4cd90a099934d4e347e303e4acdc50dca7bea9642e59`.

추출본은 `experiment_v1_1/evidence/p3_colab_20260916T085123549826/`다. Colab의 경로는 원본 manifest에 보존했고 로컬 경로는 registry에 연결했다. 최종 init/best/last 바이트를 검증했다. 1M 시점 best/last 바이트는 export에서 교체되어 없고 hash·로그·선택 수치만 남는다. 제출되지 않은 Drive snapshot의 복구 무결성은 이번 감사 범위 밖이다.

Seed 1/2와 표현·인과 분석은 선행 gate 실패로 미실행이다. 기존 2-block 실패를 포함해 두 버전 모두 보존하며 전체 실험 성공이나 SAE/TC 가설 검증 완료로 표시하지 않는다. 추가 변경은 별도 버전 규격과 재검증이 필요하다.

재현 (프로젝트 루트):

```bash
python scripts/verify_p3_4block_evidence.py
python scripts/recheck_p3_4block_corpus.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests_v1_1 -q
```
