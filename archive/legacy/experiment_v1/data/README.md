# 데이터 경로 매핑

`experiment_v1/` 아래에 153 MB 코퍼스를 복제하지 않는다. P0에서 검증하고 동결한 실제 입력은 프로젝트 루트의 `data/language_v1/`이다. 모든 경로는 프로젝트 루트 기준이며, 실행 전 `configs/language.json`에 기록된 SHA-256과 원본 manifest의 파일별 SHA-256을 확인한다.

| 03 §14 논리 경로 | 실제 경로 | 판정 |
|---|---|---|
| `data/corpus_manifest.json` | `data/language_v1/manifest.json` | 이름만 매핑, 재사용 |
| `data/cpu_validation.json` | `data/language_v1/cpu_validation.json` | 재사용 |
| `data/corpus_statistics.json` | `data/language_v1/corpus_statistics.json` | 재사용 |
| `data/train_shards/` | `data/language_v1/train_shards/` | 재사용, GPU에서 생성 금지 |
| `data/fixed_splits/` | `data/language_v1/{val_iid,test_iid,diagnostics,composition,test_length}*` | 재사용 |
| `data/interpretation/` | `data/language_v1/interpretation/` | 재사용 |
| `data/causal_pairs/` | `data/language_v1/causal_pairs/` | 재사용 |

`data/language_v1/`은 immutable input이다. 규격이나 생성 코드를 바꾸어 다시 만들 경우 `data/language_v1_1/`처럼 새 버전 디렉터리를 사용하며 기존 파일을 덮어쓰지 않는다.
