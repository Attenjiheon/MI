# v1.3 pilot evidence compaction

2026-09-20 사용자 지시에 따라, 이후 실험에 필요한 상태만 남기고 로컬의 미추적 Colab 반환 ZIP을 정리했다. 원본 ZIP은 실행을 거듭할 때마다 이전 preflight, CPU/CUDA debug checkpoint, event 파일을 누적 포함해 총 12,957,597,679 bytes를 차지했다.

## 계속 보존하는 항목

- `pilot_selection.json`: 여섯 cell의 고정 8M 선택 지표, eligibility, winner(`wide4_read4`), 각 선택 checkpoint SHA-256.
- `pilot_summaries/`: 로컬에 실제 반환 ZIP이 남아 있던 run의 `LATEST.json`, `result.json`, 검증 audit JSON. 두 차례 반환된 `wide4_uniform`은 둘 다 보존한다.
- `../bundles/v1_3_confirm_wide4_read4_seed0_support_v2.zip`: confirm 재개에 필요한 winner 8M checkpoint, result, audit, environment, selection, 검증 스크립트를 포함한 18,299,110-byte 최소 지원 번들. SHA-256은 `0369862b29d9a8f03553e36ba9cbba102032c6dba3761c82c7cc537f7b55916c`이다.
- 동결 config, code, 32M corpus, confirm notebook은 기존 경로에 그대로 둔다.

`pilot_selection.json`에는 `deep8_read4`의 검증 결과와 checkpoint hash가 들어 있지만, 정리 당시 그 cell의 원본 반환 ZIP은 로컬에 없었다. 따라서 존재하지 않던 원본 증빙을 보존했다고 주장하지 않는다.

## 제거한 미추적 원본 ZIP

| 파일 | bytes | SHA-256 |
|---|---:|---|
| `v1_3_pilot_base4_uniform_evidence_20260918T172125514389.zip` | 391,938,046 | `04ceacf028491841ff12cf99cd9ad4cabce653bb1dee2fd17fa41a2f008886ed` |
| `v1_3_pilot_wide4_uniform_evidence_20260918T182113618135.zip` | 1,146,157,978 | `212a6fc0ba028c4062e9727f7851bf81750ecb425b69179a29d5439c03f35adf` |
| `v1_3_pilot_wide4_read4_evidence_20260918T183532331449.zip` | 1,507,560,499 | `41266cc2b3bd7d5db24b2ab9e059f616fd27f3d3adf0b3e5dbea551f8ceb2141` |
| `v1_3_pilot_base4_read4_evidence_20260919T023223589232.zip` | 2,922,224,946 | `c6e5bf2ec87292efd5efa779139ae4682f3aaa7906e47e947391efe2d1ff1e0b` |
| `v1_3_pilot_wide4_uniform_evidence_20260919T024833286858.zip` | 3,315,011,895 | `28017552b08e53095954a4db7ca674aa0c3a15d2009bea30e336834d8171a595` |
| `v1_3_pilot_deep8_uniform_evidence_20260919T030742947025.zip` | 3,674,704,315 | `e2aa8aa378050fcba7610409471afa46489d3773d1375b57708f9d768cb091d8` |

이 ZIP들은 Git에 추적되거나 LFS에 등록된 파일이 아니었다. 삭제 뒤의 다음 실행 진입점은 `../notebooks/02_colab_confirm_wide4_read4_seed0.ipynb`와 위 support bundle이다.
