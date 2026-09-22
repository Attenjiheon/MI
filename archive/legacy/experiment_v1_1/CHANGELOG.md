# experiment-spec-v1.1 — 4-block 변경 규격

승인: 사용자의 2026-09-16 명시 지시. 기존 2-block 모델보다 복잡한 인공어에서 4-block 모델의 성능이 좋았다는 이전 실험 경험을 근거로 깊이를 변경한다. 이전 실험의 세부 설정·수치는 제공되지 않았으므로 이번 실험의 성능을 예측하거나 통과 근거로 사용하지 않는다.

## 적용 범위와 우선순위

이 문서는 **v1.1에 한하여** 03_experiment_spec.md §4.1–4.3의 block 수·파라미터 수·출력 초기화 배율을 변경한다. 나머지 규격과 phase gate는 유지한다. 기존 `experiment_v1/`, `interp/` 및 2-block 결과·config는 변경하지 않는다. v1.0의 P3 failed는 그대로 보존한다.

| 항목 | v1.0 | v1.1 |
|---|---|---|
| Decoder block 수 | 2 | 4 |
| Trainable parameters | 400,640 | 797,184 |
| Attention/MLP output weight 초기화 배율 | 1/√(2×2)=0.5 | 1/√(2×4)≈0.35355339 |
| 모델 코드 | interp | interp_v1_1 |
| 설정·산출물 | experiment_v1 | experiment_v1_1 |

파라미터 수: embedding 1,920 + 4×198,272 + final LN 256 + unembedding 1,920 = 797,184. 모든 block의 attention output 및 MLP output에 깊이 기반 배율을 적용한다. 따라서 비교는 깊이와 그에 맞춘 초기화 변경의 결합 효과이며 깊이만의 인과 효과라고 단정하지 않는다.

## 유지 사항

Width 128, heads 4, MLP 512, vocabulary 15, RoPE/context, float32, optimizer, warmup, effective batch 64, microbatch 16부터 OOM 시 감소, validation 간격, checkpoint 선택 규칙과 행동 gate를 유지한다. 사전 생성한 `data/language_v1/`와 shard 순서·split을 그대로 사용한다. 학습 예산은 seed 0의 1M, 개선 조건 충족 시 누적 3M까지 한 번 연장이다. Test로 설정을 선택하지 않는다.

새 seed 0 초기화부터 시작한다. v1.0 checkpoint·optimizer·RNG를 v1.1에 불러오지 않는다. 코드/config hash와 파라미터 shape가 버전 혼합을 거부한다. 저장소·Drive 경로를 분리한다. v1.0 실패를 숨기거나 seed 교체로 취급하지 않는다.

## 선행 검증과 gate

P1 코퍼스는 버전·바이트가 같고 P3 종료 시 전체 재감사를 통과한 입력을 재사용한다. 새 코드에서 checksum을 다시 확인한다. 4-block CPU/GPU smoke에서 797,184개 파라미터, 네 block의 hook·mask·gradient, 저장·재개를 확인한다. 기존 2-block GPU smoke는 새 모델 통과 증빙이 아니다.

4-block GPU smoke 통과 후에만 P3을 실행한다. 일반 validation ≥99%, 진단 세 조건 각각 ≥95%를 최소 답 CE checkpoint에서 만족해야 P4로 진행한다. 실패 시 기존 중단 규칙을 따른다. 필수 READ 분석은 여전히 block 0이며 block 1은 선택 분석이다. block 2/3 표현 분석은 이번 변경으로 자동 추가하지 않는다.

## 재현 경로

- config: `experiment_v1_1/configs/` (config_set_manifest.json의 결합·개별 hash)
- 실행: `python -m interp_v1_1.smoke`, `python -m interp_v1_1.cli train_lm`
- Colab: `experiment_v1_1/notebooks/01_colab_lm.ipynb`
- 번들: `experiment_v1_1/bundles/p3_4block_bundle_v1.zip`
- Drive: `MyDrive/boolean_interp_p3_4block_v1`

2026-09-16 결과: GPU smoke는 passed, P3은 3M 행동 gate 미달로 failed 종료. 상세 증빙은 P3_STATUS.md와 results/p3_stop_report.md를 따른다.
