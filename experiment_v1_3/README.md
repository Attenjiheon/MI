# experiment_v1_3

상태: **P1 완료 + 구현·CPU 검증 완료, GPU 실행 전**. config는 동결되었고 실행기·unit test·6-cell CPU smoke·Colab 전달물이 준비되었다. GPU smoke와 6-cell pilot은 아직 실행하지 않았다.

- [설계](./DESIGN.md)
- [기계 판독 계약](./design_config.json)
- [설계 hash manifest](./design_manifest.json)
- [동결 config set](./configs/config_set_manifest.json)
- [P1 생성 결과](./results/p1_generation.json) · [동결 train stream](./results/p1_frozen_training.json)
- [P2 구현·CPU 검증 상태](./P2_STATUS.md) · [실행 기록](./results/run_registry.csv)

v1.2는 16M 행동 gate 실패 실험으로 불변 보존한다. v1.3은 새 validation/test를 예약하고 제한된 3 architecture × 2 loss pilot으로 상태 전이 shortcut의 원인을 분리한 뒤, 사전 규칙으로 winner 하나만 32M 확인 단계에 올리는 별도 실험이다.

## 진행 상태

| 단계 | 상태 | 증빙 |
|---|---|---|
| 설계 동결 | 완료 | `DESIGN.md`, `design_config.json`, `design_manifest.json` |
| config 동결 | 완료 (`status: frozen`) | `configs/config_set_manifest.json` (`combined_sha256: acd92751…`) |
| 신규 split·train 생성 | 완료 | `results/p1_generation.json` (`status: passed`) |
| CPU 감사 | 통과 | `data/language_v1_3/cpu_validation.json` (`status: passed`) |
| 구현 (실행기·smoke·감사·선택 스크립트) | 완료 | `interp_v1_3/cli.py`, `interp_v1_3/smoke.py`, `scripts/verify_v1_3_evidence.py`, `scripts/select_v1_3_pilot_winner.py` |
| unit test | 13 passed | `results/p2_pytest.txt` |
| CPU smoke (6 cell 전부) | 통과 | `smoke/cpu_release/smoke.json` (환경 `f27a8fafc6f8eeaa`) |
| Colab 입력 번들·노트북 | 준비 완료 | `results/colab_delivery.json`, `notebooks/01_colab_pilot_8m.ipynb` |
| GPU smoke | **미실행** | — |
| 6-cell pilot | **미실행** | — |
| anchor bitwise 대조 (update 136/407/1082) | **미실행** | — |
| winner seed 0 confirm → seed 1·2 → frozen test | **미실행** | — |
| P5–P9 표현 분석 | 미실행 (gate 통과 LM 2개 이상일 때만) | — |

## P1 확정 수치

- 데이터 root: `data/language_v1_3`, shards 68개, train sequences 271,936
- 실제 train prediction tokens 32,004,917 (명목 32M, overshoot 4,917), final update 4,249, final cursor 271,936
- milestone cursor/update: 1M→136, 3M→407, 8M→1,082, 16M→2,160, 24M→3,205, 28.8M→3,832, 30.4M→4,041, 32M→4,249
- v1.2 train prefix 16,001,083 tokens / 138,240 sequences가 byte-identical로 보존됨 (`v1_2_train_prefix_byte_identical: true`)
- 예약 split: `select/`, `gate/`, `test/`, `first_repeat/`, `interpretation/`, `causal_pairs/` 생성 완료
- CPU 감사 통과 항목: 42 cell quota, first/repeat 의미 검사, NOT/XOR involution, 전역 exact sequence·READ prefix 격리(기존 exact hash 166,807 / read prefix 2,490,322 대조), 재현성, batch multiple 64
- audit hash `c72eb57a…`, manifest hash `7021b5f9…`

## 다음 작업

1. `notebooks/01_colab_pilot_8m.ipynb`를 Colab GPU 런타임에서 실행한다. 한 런타임에 `CELL` 하나씩,
   여섯 cell을 각각 돌리고 결과 ZIP과 SHA-256을 반환한다. GPU smoke가 통과해야 본학습이 시작된다.
2. 반환된 ZIP마다 `scripts/verify_v1_3_evidence.py`로 계약·보존·선택 규칙을 다시 대조한다.
3. `base4_uniform`의 update 136/407/1082 checkpoint가 v1.2 보존본과 bitwise 동일한지
   `--anchor-checkpoint`로 확인한다. 불일치 시 탐색을 중단하고 구현·환경을 감사한다.
4. 여섯 cell이 모두 끝난 뒤에만 `scripts/select_v1_3_pilot_winner.py`로 사전 규칙(§4.3)에 따라
   winner를 하나만 승격한다. Eligible cell이 없으면 pilot failure로 종료한다.

gate·test 결과는 선택이 모두 끝난 뒤에만 열람한다. 어느 단계든 quota·hash·재현·수치 검사가 실패하면 뒤 단계를 시작하지 않는다.
