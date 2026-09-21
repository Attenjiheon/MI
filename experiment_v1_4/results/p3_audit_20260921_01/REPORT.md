# v1.4 P3 seed 0 — 반환 검증 통과

2026-09-21. **P3 완료 및 행동 gate 통과.** Seed 1·2, frozen test와 표현 분석은 미실행이다.

## 입력과 감사

원본 ZIP: `experiment_v1_4/evidence/v1_4_seed0_evidence_20260921T064114169595.zip`

SHA-256: `72403bb472955f88c2a12b3d60694570b3be70a044fb801077e07244678521a8`.
1,306,686,362 bytes, checksums.json의 8,449개 파일을 CRC/개별 checksum과 대조했다.
경로·중복·symlink를 검사하고 별도 임시 경로에 추출했다. 원본 ZIP은 보존한다.

- 기존 verifier를 로컬에서 실행해 동결 config/corpus/runtime hash, 실제 8,399개 update의
  token/cursor, 10개 milestone, init/선택/last checkpoint 및 optimizer/RNG 보존을 검증했다.
- 반환 GPU smoke는 r3 코드와 동결 입력에 일치한다. T4, microbatch 16, effective batch 64,
  환경 e74fb1dcf8112ca0, lock 및 필수 smoke assertions 검증 통과.
- Select 후보 전체에서 사전 규칙을 다시 계산해 update 7,983 선택이 일치함을 확인했다.
  Gate는 저장된 count·점수·threshold로 판정만 검산했다. 모델 gate/test는 재채점하지 않았다.
- 실제 기록에는 두 session이 있으며 두 번째는 update 4,249에서 재개했다. 두 session의
  환경 lock/ID가 동일하다. 최초 추가 감사의 단일 session 가정은 맞지 않아 이력 확인 뒤
  두 session의 resume 경로와 환경을 명시적으로 검증하도록 보완했다. 실험 결과는 수정하지 않았다.
- Read4 weight denominator, LR, microbatch, finite metrics, first-50 측정 합계,
  모든 pair quota와 CI/확장 행동 보고 필드가 확인됐다.

## 학습·선택 결과

| 항목 | 값 |
|---|---:|
| 실제 prediction tokens | 64,005,751 |
| overshoot | 5,751 |
| final update / cursor | 8,399 / 537,536 |
| 선택 update | 7,983 |
| 선택 checkpoint의 tokens | 60,801,363 |
| 선택 select first macro CE | 0.0032469069110952035 |

선택 checkpoint SHA-256:
`76d86f1a83bde7f8f6389684cd2a3913c923e21ccfbe42e40d3255385a9982b1`.
64M의 CE는 0.004460475431776227로 선택 후보보다 높다. Gate를 보고 checkpoint를 바꾸지 않았다.
선택 곡선은 `select_curve.csv`, 동결 정보는 `frozen_seed0.json`이다.

## 독립 gate

| 지표 | 정확도 | 기준 |
|---|---:|---:|
| 일반 READ (16,593 targets / 1,024 sequences) | 99.9277% | ≥99% |
| 다른 변수 읽기 | 99.9023% | ≥95% |
| 복수 갱신 | 99.9023% | ≥95% |
| SET 이후 첫 READ | 100% | ≥95% |
| First 42-cell macro | 99.8140% | ≥95% |
| Repeat 42-cell macro | 100% | ≥95% |

각 legacy diagnostic은 1,024 independent target sequences다. First/repeat는 42×64=2,688 origins.
First 연산별 최솟값 99.6094% (XOR), depth별 최솟값 99.4420% (4+),
answer별 최솟값 99.7768% (0). 모든 group threshold와 coverage/quota가 통과했다.
First macro 95% within-cell origin bootstrap CI는 [99.6280%, 99.9628%].
Cell CI는 보고 항목이며 추가 gate로 사용하지 않았다. 상세는 `gate.json`과 `gate_summary.csv`.

## 소요 시간과 한계

최종 재개 session: 13,069.19초 (3시간 37분 49초).
이 session의 학습 연산 합계 1,560.22초 (26분), select 평가 84.69초,
checkpoint serialization 5.22초. 나머지 약 11,419초 (3시간 10분)는 Drive 복사/checksum,
로컬 event 기록, gate 및 orchestration이 따로 측정되지 않아 원인별 분리가 불가능하다.
전체 보존 update의 학습 연산 합계는 3,124.15초 (52분 4초)다.
이 값은 첫 session의 미저장 재연산이나 세션 사이 downtime을 포함한 총 비용이 아니다.
첫 50 updates peak allocated VRAM은 972,558,848 bytes이며 전체 run peak라고 주장하지 않는다.

독립적인 본학습 CUDA 재실행은 하지 않았다. 반환 checkpoint/state/hash와 기록된 CUDA
결과를 검증한 것이며, 속도 측정만으로 wall time을 예측하는 데 한계가 확인됐다.

## 재현과 다음 단계

저장된 `audit_archive.py` → 기존 `scripts/verify_v1_4_evidence.py` (reevaluate 옵션 없이)
→ `scripts/verify_v1_4_gpu_smoke.py` → `audit_details.py` 순서로 검증했다.
`archive_verification.json`의 임시 추출 경로는 감사 당시 경로이며 원본 ZIP으로 재생성 가능하다.
완료 판정과 증빙 hash는 `completion.json`에 있다.

다음은 같은 동결 설정·train stream·예산의 P4 seed 1·2 재현이다. 최소 두 seed가 gate를
통과해야 표현 분석에 들어갈 수 있다. 모든 seed의 학습·validation 결정이 동결되기 전에는
frozen test를 열지 않는다. 이번 감사에서는 P4를 시작하지 않았다.
