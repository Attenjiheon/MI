# Boolean 프로그램의 상태 표현 해석 — v1.4 본 실험

Boolean 상태 추적 언어를 학습한 Transformer에서 READ 표현의 선형 접근성,
SAE·Transcoder feature의 의미와 인과적 효과를 분석한다.
**현재 P4 완료, 다음은 P5 READ cache·probe다. 전체 해석 실험은 아직 미완료다.**

## 1. 시작할 문서

| 문서 | 역할 |
|---|---|
| [AGENTS.md](AGENTS.md) | Codex 프로젝트 지침·세션 진입·단계별 필수 읽기·보존·실행 규칙 |
| [실험 설계](01_experiment_design.md) | v1.4 연구 질문·범위·해석 한계 |
| [언어·코퍼스](02_language_and_corpus.md) | 문법·metadata·split·누출 방지 |
| [상세 명세](03_experiment_spec.md) | 모델·read4 학습·선택·gate·probe·SAE·TC·인과 평가 |
| [실행 계획](phase.md) | 현재 P0–P11 순서·완료 조건·증빙·계산 예산 |
| [v1.4 현재 안내](experiment_v1_4/CURRENT.md) | 동결 계약·상태·결과로 연결되는 색인 |

루트 01–03은 v1.4 기준으로 통합했다. v1.4의 동결 설계·기계 판독 계약·실행 증빙이
우선하며, 문서 통합으로 이미 수행한 실험을 변경하지 않는다.
`experiment_v1_4/README.md`와 `DESIGN.md`는 hash가 동결된 **설계 당시 원문**이다.
그 안의 과거 상태 대신 위 현재 안내와 단계 상태 문서를 읽는다.

Codex에서 이 저장소를 열면 루트 `AGENTS.md`를 프로젝트 지침의 진입점으로 사용한다.
Codex 작업 방식은 `AGENTS.md` §0, 단계별 필수 문서는 §1–3, 완료 후 commit·push는 §6을 따른다.
현재 진행 상태는 `experiment_v1_4/CURRENT.md`와 해당 단계 상태·증빙에서 확인한다.
기존 미커밋 작업은 보존하고 요청한 변경만 커밋한다. 과거 archive와 동결 계약은 지침 전환을 위해 수정하지 않는다.

## 2. 저장소 구조

```text
MI/
├── AGENTS.md, README.md, phase.md, 01–03_*.md   현재 v1.4 문서
├── experiment_v1_4/                           본 실험 계약·상태·증빙·노트북·결과
├── interp_v1_4/                               본 실험 구현
├── tests_v1_4/                                본 실험 검증
├── data/language_v1_4/rebuild_01/              활성 동결 corpus
├── corpus/                                   공용·버전별 생성기 및 replay
├── scripts/                                  운영·번들·감사 도구
├── requirements-{corpus,interp}.txt            의존성 목록
├── maintenance/                              정리·이동·검증 기록
└── archive/
    ├── legacy/                               v1.0–v1.3 실험·코드·테스트·데이터 실체
    └── specifications/pre_v1_4_integration/   통합 전 루트 문서 원문
```

과거 버전은 `archive/legacy/`의 실제 경로로만 접근한다. 루트와 archive 내부의
바로가기는 모두 제거했다. `interp_v1_4`가 사용하는 과거 공용 모듈은
`archive.legacy.interp_v1_2`에서 직접 import한다.

## 3. 버전 계보와 현재 상태

| 버전 | 설정 | 상태·근거 |
|---|---|---|
| v1.0 | 2-block, 3M | [행동 gate 실패](archive/legacy/experiment_v1/results/p3_stop_report.md) |
| v1.1 | 4-block, 3M | [행동 gate 실패](archive/legacy/experiment_v1_1/results/p3_stop_report.md) |
| v1.2 | 4-block, 16M | [행동 gate 실패](archive/legacy/experiment_v1_2/results/p3_stop_report.md) |
| v1.3 | 6-cell pilot, 32M confirm | [Confirm gate 실패](archive/legacy/experiment_v1_3/P3_STATUS.md) |
| **v1.4** | **12×256, read4, 64M/seed** | [세 seed P4 완료](experiment_v1_4/P4_STATUS.md), P5 미실행 |

Seed 0/1/2 일반 READ test는 99.9410% / 99.9472% / 99.9659%이며,
선택 update는 7,983 / 7,983 / 8,399다. 세 seed 모두 해석 대상이다.
[동결 모델 목록](experiment_v1_4/results/frozen_test_audit_20260922_01/frozen_lms.json)을 따른다.
SAE·TC·인과 평가·sparse seed 반복까지 마쳐야 전체 실험 완료다.

## 4. 원본 보존과 검증

- [바로가기 제거 기록](maintenance/remove_shortcuts_20260922/REPORT.md)에 현재 경로·소스 변경·검증을 기록한다.
- [Archive 안내](archive/README.md)와 [정리 보고](maintenance/repository_cleanup_20260922/REPORT.md)에 이동 매핑과 검증을 기록한다.
- 기존 manifest의 루트 문서 hash는 [통합 전 snapshot](archive/specifications/pre_v1_4_integration/README.md)에 대응한다. 동결 manifest 자체를 수정하지 않는다.
- Corpus·config·checkpoint·번들·증빙 bytes를 보존한다. 기존 결과는 새 실행으로 덮어쓰지 않는다.
- [이전 디스크 정리](maintenance/disk_cleanup_20260921/REPORT.md)에서 삭제된 파일은 그 기록을 따른다. 이번 정리는 데이터 삭제 작업이 아니다.
- `.gitattributes`의 LFS 및 `.gitignore` 규칙은 archive 경로에도 적용한다.

## 5. 로컬 확인

프로젝트 의존성을 설치한 Python 환경에서 저장소 루트를 기준으로 실행한다.
현재 로컬 검증 환경은 `/opt/anaconda3/bin/python`이며 환경별 실제 lock은 실행 증빙을 따른다.

```bash
python -m pytest tests_v1_4 -q
python -m interp_v1_4.cli --help
python scripts/verify_repository_layout.py
```

기본 `pytest` 수집 대상도 `tests_v1_4/`다. 과거 검증은 해당 버전의 보존된 실행 환경과 `archive/legacy/tests_v1_*` 경로를 사용한다. 학습·gate/test 재실행은 위 확인 명령에 포함되지 않는다.
단계 완료 후 Git 갱신은 AGENTS §6을 따른다.
