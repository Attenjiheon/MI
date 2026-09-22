# v1.4 본 실험 승격과 저장소 정리

2026-09-22 사용자 지시: 과거 실험을 한곳에 정리하고 v1.4 문서와 루트 문서를 충돌 없이 통합한다.
실험 수치·선택·gate·test 결과를 재계산하거나 학습을 실행하는 작업은 아니다.

## 이동과 원본 보존

- v1.0–v1.3의 실험·구현·테스트 12개와 과거 데이터 3개 디렉터리를 `archive/legacy/`로 이동했다.
- 기존 경로에 상대 symlink를 두어 frozen path, Python import 및 기존 notebook 참조를 유지했다.
  `interp_v1_4`가 사용하는 `interp_v1_2`도 동일 bytes로 참조한다.
- Archive 내부에도 원래 상대 문서 링크에 필요한 공용 경로를 제공한다.
- `.gitignore`와 Git LFS 규칙을 새 경로에 추가했다. 기본 pytest 대상은 `tests_v1_4`로 지정했다.
- 루트 원문 6개를 `archive/specifications/pre_v1_4_integration/`에 byte-identical하게 보관했다.
  동결 manifest의 source_documents hash는 이 snapshot으로 검증한다.
- `experiment_v1_4`의 기존 문서·계약·결과·번들 및 활성 데이터·코드는 수정하지 않았다.
  동결 README를 대체 안내할 `CURRENT.md`만 추가했다. 과거 문서의 상태는 당시 기록이다.
- 파일 삭제는 수행하지 않았다. 이전 삭제 범위는 `maintenance/disk_cleanup_20260921/`을 따른다.

이동 목록·기준 commit·문서 hash는 [migration.json](migration.json),
이동 전 전체 가용 실험 파일의 bytes/hash는 [preserved_files.jsonl](preserved_files.jsonl)에 있다.
캐시와 `.DS_Store`는 증빙 inventory에서 제외한다. 전체 검증은 로컬 전용 보존 파일도 포함하므로
새 clone에 없는 ignored payload는 `--full` 검사에서 명시적으로 누락으로 보고한다.

## 규격 충돌 해소

| 기존 충돌 | 통합 처리·근거 |
|---|---|
| 2-block/128, 400,640 parameters | 동결 v1.4 12-block/256, 9,485,312로 01·03 본문 통합 |
| 동일 가중 LM CE, 온라인 생성 | token-only read4 및 CPU 고정 shard 순차 1회 소비로 01·02·03 통합 |
| 1M→3M 연장, 일반 val 최소 CE | seed당 64M, 10 milestone, first 42-cell macro CE 전역 최소 및 near-tie로 교체 |
| 단일 validation과 이전 진단 quota | select/gate/test 분리, 64/64/128 pair quota와 v1.4 legacy 규모 반영 |
| 기존 gate만 안내 | general·legacy·first/repeat·group·coverage gate와 최소 두 통과 seed 조건 통합 |
| 최초 corpus root | `corpus_rebuild.json`의 활성 rebuild_01, historical READ-prefix 검증 반영 |
| 128차원 hook·probe·SAE/TC | 256차원, full binary probe 257 parameters, SAE/TC 262,912; dictionary width 512·예산 유지 |
| 128후보 “일치”인데 raw 좌표가 256개 | 128후보 대조 규모는 유지하고 raw 좌표도 사전 추출하도록 명시. 실제 subset·seed는 미실행 P5 config에 동결 |
| block 1 선택 분석과 전체 층 probe 진단 혼동 | 전체 12층 full probe는 동결 설계의 진단, block 1 sparse 학습·개입은 P10 선택 범위로 구분 |
| TC 선택 확장과 필수 범위 충돌 | READ SAE → 필수 READ TC → sparse seed 반복으로 01과 03 통일 |
| AGENTS·phase에 누적된 과거 실행 상태 | 원문을 archive에 보존하고 현재 P1–P4 완료·P5 미실행으로 정리 |

분석 RNG 파생식은 기존 규칙을 유지한다. Namespace에 남은 v1.0은 RNG 식별자이며 run key에
실제 v1.4 버전을 넣는다. Behavior bootstrap은 이미 동결된 v1.4 별도 계약을 유지한다.
새로 명시한 후보 수 대조 정합화는 관련 해석 test를 보기 전의 문서 통합 결정이다.
P5 실행 config·실제 결과를 작성하거나 P5 완료를 주장하지 않는다.

## 검증

검증 도구는 `scripts/verify_repository_layout.py`다. 기본 검사는 15개 호환 경로,
원문 snapshot, 동결 설계/config hash, 현재 문서의 로컬 링크를 확인한다.
`--full`은 이동 전 inventory 전체의 파일 수·bytes·SHA-256 동일성을 확인한다.

최종 보존 검증은 [verification.json](verification.json)에 기록했다.

| 검증 명령 | 결과 |
|---|---|
| `python3 scripts/verify_repository_layout.py --full --output maintenance/repository_cleanup_20260922/verification.json` | passed: 1,953 files / 13,910,350,697 bytes의 전후 SHA-256 일치; 15 aliases·6 snapshots·10 contract hashes 통과 |
| `/opt/anaconda3/bin/python -m pytest tests_v1_4 -q` | 40 passed, 22.73초 |
| `/opt/anaconda3/bin/python -m pytest --collect-only -q` | 기본 수집 40 tests; archive 중복 수집 없음 |
| `/opt/anaconda3/bin/python -m interp_v1_4.cli --help` | 정상 종료; archive 의존 import 호환 |

환경은 로컬 Python `/opt/anaconda3/bin/python`, PyTorch 2.10.0이다.
Model gate/test 재채점은 포함하지 않는다.
Git 변경은 내용이 동일한 이동과 현재 문서·검증 도구로 제한한다.

Git index 검사: 기존 추적 파일 **837개 모두** 이동 목적지에서 동일 blob ID와 mode를 유지했다.
[검증 기록](git_move_verification.json)에 남겼다. 동결 Markdown snapshot의 원래 두 칸 줄바꿈은
해시 보존을 위해 유지하며 해당 경로에만 Git whitespace 속성을 명시했다.
