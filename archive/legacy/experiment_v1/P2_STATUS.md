# P2 구현·smoke 판정

판정일: 2026-09-10 (UTC)  
적용 규격: `experiment-spec-v1.0`  
상태: `passed`  
선행 조건: P1 `passed` (`data/language_v1/postwrite_audit.json`)

## 이번 단계의 목표

학습부터 패칭까지를 작은 debug 데이터로 연결하고, CPU와 Colab GPU 두 환경에서 phase.md §4의 8개 필수 검사를 통과시킨 뒤 통과 환경을 lock한다. 본실험 학습은 수행하지 않는다.

## 필수 검사 결과

두 환경 모두 같은 코드·config·corpus·debug fixture hash에서 실행했고, 8개 항목이 모두 통과했다.

| phase.md §4 항목 | CPU (`cpu_attempt_02`) | Colab GPU (`colab_gpu_01`) |
|---|---|---|
| LM 400,640 / SAE·TC 각 131,712 parameter | 400,640 / 131,712 ×4 | 동일 |
| 2 batch forward/backward, token-weighted accumulation·mask·target shift | 통과, 누적 gradient 최대 오차 1.34e-07 | 통과, 8.94e-08 |
| Padding batch와 개별 sequence 일치 (`atol=1e-5, rtol=1e-4`) | 최대 절대 오차 0.0 | 3.73e-07 |
| 미래 suffix 변경 시 이전 logits 불변 | 통과 (exact) | 통과 (exact) |
| READ/update hook 위치, `h=r_mid+m`, `u=LN_mlp(r_mid)` | 통과 (exact) | 통과 (exact) |
| 작은 pool에서 dictionary 100 updates + probe 1개 fitting | SAE/TC × k=4,16 각 100 updates, 51,200 draws; probe 통과 | 동일 |
| Identity/full/sparse patch 각 1회, identity가 원래 logits 재현 | 통과 (identity exact) | 통과 (identity exact) |
| Checkpoint 저장·복구 후 다음 update와 데이터/sampler 진행 일치 | LM `bitwise_equal`, cursor 192 / update 3; dictionary 4종 `bitwise_equal` | 동일 |

추가로 확인한 항목: RoPE 인접쌍 회전각의 독립 검산, PAD key masking이 padding token embedding과 무관함, top-k 동률 처리, `statistics()`의 상수 입력 거부, patch 예외 발생 시 hook 정리, SAE 입력 scale과 TC 출력 scale의 분리(`s_u=0.795`, `s_m=0.0196`).

두 환경의 차이는 모두 float 수준(≤3.7e-07)이며 구조적 값(parameter 수, cursor/update, draw 수, seed·key, dead 비율, L0)은 완전히 일치한다. Probe의 validation balanced accuracy는 CPU/GPU 모두 0.531843010676409로 동일하다(probe는 규격대로 CPU float64에서 fit한다).

측정값: 실제 21,755 prediction tokens, 993 unique READ positions, 206,848 draws. CPU 8.19s / peak RSS 304,152,576 B, GPU 8.02s / peak 할당 VRAM 167,518,208 B(`torch.cuda.max_memory_allocated()`; CPU 쪽은 `resource.getrusage`의 max RSS로 서로 다른 지표다).

## 통과 환경

| | CPU | Colab GPU |
|---|---|---|
| environment_id | `96a2bbce68883fdf` | `e74fb1dcf8112ca0` |
| Python | 3.13.4 | 3.13.15 |
| torch / numpy | 2.13.0 / 2.2.6 | 2.11.0+cu128 / 2.1.3 |
| CUDA / cuDNN | 없음 | 12.8 / 9.19.00 |
| GPU / VRAM | 없음 | Tesla T4 / 15,637,086,208 B |
| RAM / 여유 디스크 | 8,589,934,592 B / 18,823,163,904 B | 13,605,822,464 B / 201,835,798,528 B |
| 드라이버 | 해당 없음 | NVIDIA-SMI 580.82.07 (nvidia-smi는 CUDA 13.0 표기, torch는 12.8 빌드) |
| lock 파일 | `experiment_v1/environment/requirements-cpu.lock.txt` (`ec616cdc…4cbd`) | `experiment_v1/environment/requirements-colab.lock.txt` (`855ad8de…ffc7`) |

두 환경의 결합 기록은 `experiment_v1/environment/runtime_manifest.json`이다. GPU 작업 디렉터리는 규격대로 `/content/boolean_interp`이며, 그 안에서 corpus·config hash를 검증한 뒤 실행했다(107개 입력 파일).

공통 입력 hash: corpus manifest `bff7bc9b…5e26`, config 결합 `b00d015d…1bdf`, debug fixture `815194b8…6e50`. `interp/*.py` 11개 파일의 hash는 두 smoke 기록과 현재 작업 트리가 모두 일치한다.

## 실행 명령

```bash
# CPU
.venv-p2/bin/python -m pytest tests -q                     # 16 passed (experiment_v1/smoke/pytest_cpu.txt)
.venv-p2/bin/python -m interp.smoke --device cpu --output experiment_v1/smoke/cpu_attempt_02
.venv-p2/bin/python -m interp.cli train_lm --device cpu --debug --output experiment_v1/smoke/cli_lm_01

# Colab GPU: experiment_v1/notebooks/00_p2_colab_smoke.ipynb
python3 -m pytest tests -q
python3 -m interp.smoke --device cuda --output experiment_v1/smoke/colab_gpu_01
```

## 실패·수정 기록

1. **Colab 첫 시도 실패 (`pytest` exit 1).** `p2_colab_bundle_v1.zip`에 `experiment_v1/results/run_registry.csv`가 빠져 있어 `tests/test_p0_config.py::test_registry_uses_only_allowed_statuses`가 `FileNotFoundError`로 실패했다. 번들을 그대로 풀어 재현한 결과 이 항목 하나만 실패하고 나머지는 통과했다. 코드·규격 문제가 아니라 패키징 누락이다. 조치: 노트북에 registry 업로드 셀을 추가하고, 이번 GPU 실행의 입력을 "번들 v1(`91e1d91d…7251a`) + `run_registry.csv`(`5f06ae39…bdc5`)"로 기록한다. 번들 zip 자체는 덮어쓰지 않았다.
2. **`cpu_attempt_01`은 이전 버전 `interp/smoke.py`의 결과다.** 통과했으나 실제 READ 위치 probe 검사가 없어 `cpu_attempt_02`가 이를 대체한다. 두 기록 모두 보존한다.

## 규격 불일치 기록

`03 §7.2` 3항은 prefix 크기 간 동률에서 **작은 m(feature 수)을 먼저** 적용하고 그다음 §7.1 순서를 따르도록 정한다. 반면 P0의 `configs/probe.json`은 `validation_tie_break_order`에 `smaller_feature_count`를 마지막에 둔 단일 결합 순서를 기록하고 있다. `interp/probe.py`는 03의 명시적 상세 규격을 따른다(AGENTS.md §2 우선순위). P0 config 바이트는 수정하지 않고 보존했다.

## 산출물

| 경로 | 내용 |
|---|---|
| `interp/` | Transformer·학습·hook/cache·probe·SAE/TC·patching·저장/재개 구현 |
| `interp/cli.py`, `interp/smoke.py` | 실행 entry point (`experiment_v1/README.md`에 규격 인터페이스 대응 기록) |
| `experiment_v1/smoke/cpu_attempt_02/` | CPU smoke 로그·debug checkpoint·lock |
| `experiment_v1/smoke/colab_gpu_01/` | Colab GPU smoke 로그·debug checkpoint·lock·nvidia-smi |
| `experiment_v1/smoke/p2_colab_gpu_evidence.zip` | Colab에서 내려받은 원본 증빙 (`ea1c1498…b01b`) |
| `experiment_v1/smoke/cli_lm_01/` | CLI `train_lm` debug 실행 (init/best/last, 곡선, manifest) |
| `experiment_v1/environment/requirements-{cpu,colab}.lock.txt`, `runtime_manifest.json` | 통과 환경 lock과 런타임 기록 |
| `experiment_v1/bundles/p2_colab_bundle_v1.zip` | GPU 전송 번들 (CPU에서 생성, GPU에서 재생성하지 않음) |
| `experiment_v1/debug/sequences.json` | CPU에서 생성한 별도 debug fixture |

## 재사용 금지

이 단계의 checkpoint·feature·전처리 통계는 모두 `debug_only=true`, `reuse_in_experiment=false`다. 본실험에 로드하지 않는다.

## P2에서 판정하지 않은 것

- 행동 gate. `interp/cli.py train_lm`은 1M 파일럿 경계까지만 실행하며 `gate` 필드에 `not_adjudicated`를 기록한다. 3M 연장과 seed 1/2 동결 예산 실행은 P3의 결정 계약이 나온 뒤에 붙인다.
- 평가 entry point는 full-vocabulary 행동 지표와 정규화된 dictionary fidelity, 이미 동결된 patch plan 적용까지만 제공한다. Strata·기준선·의미 지표 표·causal matching 구성·cluster bootstrap은 P4–P11에서 통합·검증한다.
- GPU 처리량(tokens/sec)과 본실험 peak VRAM은 P3의 최초 50 updates에서 측정한다. 여기 기록한 8초짜리 debug 실행은 예산 산정 근거가 아니다.
- 본실험 학습은 어느 환경에서도 실행하지 않았다.
