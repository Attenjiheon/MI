# experiment_v1_3

상태: **6-cell pilot 선택 완료, winner seed 0 confirm 대기**. `wide4_read4`가 사전 규칙에 따라 승격되었고, confirm용 notebook과 최소 지원 번들을 검증했다. Colab 초기화 오류 두 건은 학습 전에 수정했으며, 32M 학습과 gate는 아직 시작하지 않았다.

- [설계](./DESIGN.md)
- [기계 판독 계약](./design_config.json)
- [설계 hash manifest](./design_manifest.json)
- [동결 config set](./configs/config_set_manifest.json)
- [P1 생성 결과](./results/p1_generation.json) · [동결 train stream](./results/p1_frozen_training.json)
- [P2 구현·CPU 검증 상태](./P2_STATUS.md) · [P3 pilot·confirm 상태](./P3_STATUS.md) · [실행 기록](./results/run_registry.csv)
- [pilot 선택 결과](./evidence/pilot_selection.json) · [증빙 경량화 기록](./evidence/COMPACTION.md)

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
| GPU smoke | 통과 | cell별 pilot audit JSON |
| 6-cell pilot | 완료·선택 완료 | `evidence/pilot_selection.json` |
| pilot winner | `wide4_read4` | 8M update 1,082, checkpoint `cef275da…` |
| seed 0 confirm 전달물 | 준비·로컬 검증 완료 | `results/confirm_colab_delivery.json`, `notebooks/02_colab_confirm_wide4_read4_seed0.ipynb` |
| winner seed 0 32M 학습·gate | **미실행** | `P3_STATUS.md` |
| seed 1·2 → frozen test | seed 0 gate 통과 전까지 대기 | — |
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

1. `notebooks/02_colab_confirm_wide4_read4_seed0.ipynb`를 Colab GPU 런타임에서 실행한다.
2. 입력은 `bundles/v1_3_pilot_bundle_v1.zip`과
   `bundles/v1_3_confirm_wide4_read4_seed0_support_v2.zip`이다.
3. 새 Tesla T4 런타임에서 `RESUME=False`로 시작한다. 입력 ZIP이 Drive에 있으면 다시 업로드하지 않는다.
4. seed 0을 동결된 32M 경계까지 실행하고 반환 증빙을 검증한 뒤에만 seed 1·2와 frozen test로 진행한다.

gate·test 결과는 선택이 모두 끝난 뒤에만 열람한다. 어느 단계든 quota·hash·재현·수치 검사가 실패하면 뒤 단계를 시작하지 않는다.
