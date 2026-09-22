# v1.3 P3 — 6-cell pilot 선택 완료, winner seed 0 confirm 대기

2026-09-20 기준. 여섯 pilot cell을 동일한 LM seed 0과 8M 경계에서 실행했고,
사전 고정한 eligibility·선택 규칙으로 `wide4_read4`를 32M confirm winner로 승격했다.
**32M seed 0 학습과 gate 평가는 아직 시작하지 않았다.** 따라서 이 문서는 행동 gate 통과,
seed 1·2 실행, frozen test 개봉 또는 P5 진입을 주장하지 않는다.

## 1. Pilot 완료 증빙

- 선택 기록: `evidence/pilot_selection.json` (`decision: promote`, `winner: wide4_read4`)
- 경량 보존 증빙: `evidence/pilot_summaries/`
- 원본 ZIP 정리·보존 범위: `evidence/COMPACTION.md`
- Colab 환경 ID: `e74fb1dcf8112ca0` (Tesla T4)
- 공통 pilot 경계: update 1,082 / 8,001,583 prediction tokens

`pilot_selection.json`의 고정 8M 지표는 다음과 같다. 정확도는 full-vocabulary answer accuracy다.

| cell | eligible | select general acc. | first 42-cell macro CE | first macro acc. | repeat macro acc. |
|---|---:|---:|---:|---:|---:|
| `base4_uniform` | no | 64.74% | 0.702383 | 48.96% | 54.02% |
| `base4_read4` | yes | 90.03% | 0.592987 | 68.15% | 98.66% |
| `wide4_uniform` | yes | 76.82% | 0.658746 | 59.82% | 78.57% |
| `wide4_read4` | yes | 92.61% | **0.493161** | **73.81%** | **100.00%** |
| `deep8_uniform` | yes | 70.35% | 0.740181 | 54.91% | 61.61% |
| `deep8_read4` | yes | 88.66% | 0.508258 | 71.73% | 99.85% |

Eligible cell 중 primary metric인 first-member 42-cell macro answer CE가 가장 낮은
`wide4_read4`가 tie-break 없이 선택됐다. 이 결과는 pilot 선택 결과이지 32M confirmatory
gate 결과가 아니다.

Winner의 동결 항목은 다음과 같다.

| 항목 | 값 |
|---|---|
| cell / LM seed | `wide4_read4` / `0` |
| pilot checkpoint SHA-256 | `cef275da1ca92377639be8b06f5f15a977a11ecd616fe855232a5ea68e665575` |
| model tensor digest | `c1c62d597e88192b53b6fdcaf9d08f60f5e26801a1bf124275e971e315ee5339` |
| next step | `confirm_32m_seed_0` |

## 2. Confirm 전달물

32M confirm은 `notebooks/02_colab_confirm_wide4_read4_seed0.ipynb`로만 실행한다. 노트북은
confirm stage를 0→8M까지 결정적으로 재생하고 pilot winner와 model·optimizer·RNG·cursor·milestone을
bitwise 대조한 뒤에만 8M→32M을 계속한다. 선택 checkpoint 하나에서 gate를 한 번만 열고
`test/`는 열지 않는다.

| 전달물 | SHA-256 / 상태 |
|---|---|
| `bundles/v1_3_pilot_bundle_v1.zip` | `77f6912217594f120c56de1ba30248a85e7eec86ac539b7df25cc105ef458644` |
| `bundles/v1_3_confirm_wide4_read4_seed0_support_v2.zip` | `0369862b29d9a8f03553e36ba9cbba102032c6dba3761c82c7cc537f7b55916c` |
| `notebooks/02_colab_confirm_wide4_read4_seed0.ipynb` | `0b9f75586481a169c6ae6bac6a6c9d2cc7c47a034b4a8bf44753602306ef2f34` |
| 전달 manifest | `results/confirm_colab_delivery.json` (`gpu_execution_completed: false`) |

로컬에서 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests_v1_3 -q`를 다시 실행해
14 tests passed를 확인했고, support ZIP을 실제로 풀어
`confirm_support/manifest.json`과 노트북의 `SUPPORT_ROOT`가 일치함을 확인했다.

## 3. Confirm 실행 시도와 수정 기록

Colab에서 두 번의 초기화 시도가 있었지만 모두 의존성 설치·GPU smoke·학습·gate 전에
중단됐다. 학습 checkpoint나 gate 결과로 계산하지 않는다.

1. `/content/boolean_interp`의 기존 파일/끊어진 symlink가 작업 디렉터리와 충돌했다.
   confirm 전용 `/content/boolean_interp_v1_3_confirm`으로 바꾸고 `lexists`로 사전 감지했다
   (commit `8756c25`).
2. support ZIP 내부 root `confirm_support/`와 노트북의 기대 경로가 달랐다.
   단일 `SUPPORT_ARCHIVE_ROOT`에서 경로를 생성하고 manifest member를 압축 해제 전에 확인하도록
   수정했다 (commit `bcc7436`).

## 4. 현재 gate와 다음 작업

- seed 0 confirm: **pending**
- seed 0 gate: **not evaluated**
- seed 1·2: **blocked on seed 0 gate pass**
- frozen test: **not opened**
- P5–P9: **blocked on at least two LM seeds passing the behavior gate**

다음 실행은 최신 노트북을 새 Tesla T4 런타임에서 `RESUME=False`로 시작한다.
Drive의 `boolean_interp_v1_3_confirm/inputs/`에 두 ZIP이 이미 있으면 다시 업로드하지 않는다.
완료 후 evidence ZIP과 `.sha256`를 회수해 `scripts/verify_v1_3_evidence.py`로 검증한 뒤에만
이 문서와 run registry를 gate 결과로 갱신한다.
