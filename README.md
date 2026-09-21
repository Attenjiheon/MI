# Mechanistic Interpretability Project

상태 추적 인공어를 학습한 Transformer 내부에서 READ 연산의 표현과 인과적 역할을 분석하는
연구 프로젝트입니다. 코퍼스 생성부터 행동 평가, probe, SAE, transcoder, activation patching까지
재현 가능한 실험 절차를 담고 있습니다.

세션을 시작할 때는 [AGENTS.md](./AGENTS.md)를 먼저 읽습니다. 이 문서는 저장소의 **전체 구조와
명명 규칙**을 정의하는 지도이며, 실행 순서와 단계 gate는 [phase.md](./phase.md)가, 상세 규격은
`01`–`03` 원문이 담당합니다.

---

## 1. 저장소 구조

```
MI/
├── AGENTS.md                  세션 진입 라우터 (가장 먼저 읽는 문서)
├── README.md                  이 문서 — 전체 구조 지도
├── phase.md                   P0–P11 실행 순서, 선행 조건, 완료 gate
├── 01_experiment_design.md    연구 목적, 가설, 해석 범위와 보고 원칙
├── 02_language_and_corpus.md  언어 문법, 토큰 ID, 생성·split·metadata 규격
├── 03_experiment_spec.md      모델·학습·probe·SAE·TC·인과 평가 상세 규격
├── requirements-corpus.txt    코퍼스 생성 의존성 (CPU)
├── requirements-interp.txt    학습·해석 의존성
│
├── corpus/                    인공어 생성기 (버전 공용, 모듈로 버전 분기)
│   ├── language.py  generate.py  replay.py  audit.py
│   └── v1_3.py                v1.3 전용 생성 규칙
│
├── scripts/                   일회성 운영 스크립트 (번들 빌드·감사·선택·플롯)
│
├── data/                      생성 완료된 코퍼스 (불변)
│   ├── language_v1/           v1.0·v1.1 공용
│   ├── language_v1_2/         v1.2 (16M)
│   └── language_v1_3/         v1.3 (32M, 신규 split)
│
├── experiment_v1/    interp/        tests/        ← v1.0  (2-block, 보존)
├── experiment_v1_1/  interp_v1_1/   tests_v1_1/   ← v1.1  (4-block, 보존)
├── experiment_v1_2/  interp_v1_2/   tests_v1_2/   ← v1.2  (16M, 보존)
├── experiment_v1_3/  interp_v1_3/   tests_v1_3/   ← v1.3  (보존)
└── experiment_v1_4/  interp_v1_4/   tests_v1_4/   ← v1.4  (활성)
```

추적하지 않는 로컬 항목: `.venv-p2/`(로컬 파이썬 환경), `__pycache__/`, `.pytest_cache/`,
`.DS_Store`. 모두 `.gitignore` 대상이며 저장소 구조의 일부가 아닙니다.

## 2. 버전 3종 세트 규칙

실험 버전 하나는 항상 **세 디렉터리 + 코퍼스 하나**로 구성되며, 새 버전을 시작할 때도 같은
규칙을 따릅니다. 기존 버전의 코드·설정·결과를 재사용하거나 덮어쓰지 않습니다.

| 역할 | 디렉터리 | 내용 |
|---|---|---|
| 실행 산출물 | `experiment_vX/` | 설계·상태 문서, config, 번들, 노트북, smoke, evidence, 결과 |
| 구현 코드 | `interp_vX/` | 모델·학습·평가·probe·SAE/TC·patching·저장/재개, CLI |
| 검증 | `tests_vX/` | 해당 버전의 unit/contract test |
| 데이터 | `data/language_vX/` | 동결된 코퍼스, manifest, 감사 결과 |

`experiment_vX/` 내부의 표준 하위 구조는 다음과 같습니다.

| 하위 경로 | 내용 |
|---|---|
| `DESIGN.md`, `design_config.json`, `design_manifest.json` | 설계의 불변 기록과 기계 판독 계약 |
| `Pn_STATUS.md` | 단계별 진행 상태와 증빙 경로 |
| `configs/` | 동결된 config set과 `config_set_manifest.json` |
| `notebooks/` | Colab 실행용 `.ipynb` |
| `bundles/` | Colab 입력 번들 `.zip`과 `.sha256` |
| `smoke/` | CPU/GPU smoke 결과 |
| `evidence/` | Colab에서 회수해 검증한 실행 증빙 (불변) |
| `results/` | 감사 결과, run registry, 보고서 |

## 3. 버전 계보

| 버전 | 규격 | 상태 | 근거 문서 |
|---|---|---|---|
| v1.0 | 2-block, 400,640 params, 3M | **보존** — P3 행동 gate 실패 | `experiment_v1/results/p3_stop_report.md` |
| v1.1 | 4-block, 797,184 params, 3M | **보존** — P3 행동 gate 실패 | `experiment_v1_1/CHANGELOG.md`, `results/p3_stop_report.md` |
| v1.2 | 4-block 고정, 16M | **보존** — 감사된 행동 gate 실패 | `experiment_v1_2/DESIGN.md`, `results/p3_stop_report.md` |
| v1.3 | 3 architecture × 2 loss pilot, 32M | **보존** — 32M confirm 행동 gate 실패 | `experiment_v1_3/P3_STATUS.md` |
| v1.4 | 12 blocks × 256, read4, 64M | **활성** — P3 seed 0 행동 gate·반환 검증 통과, P4 seed 1·2 대기 | `experiment_v1_4/P1_STATUS.md`, `P2_STATUS.md`, `P3_STATUS.md` |

실패한 버전은 재현과 감사를 위해 원본 그대로 남깁니다. 과거 결과를 감사할 때는 그 버전의
코드(`interp_vX/`)와 config를 사용하며, 신규 버전의 구현으로 대체하지 않습니다.

## 4. 경로 이동 금지 원칙

`experiment_vX/evidence/`, `bundles/`, `data/*/manifest.json`, `provenance/`, run registry,
노트북에는 **디렉터리 경로가 문자열로 기록**되어 있습니다. 이 기록은 hash 검증의 대상이므로
디렉터리를 옮기거나 이름을 바꾸면 동결된 증빙이 깨집니다.

따라서 다음은 이동·개명하지 않습니다.

- `experiment_v1*/`, `interp*/`, `tests*/`, `data/`, `corpus/`, `scripts/`
- `interp_vX`는 `python -m interp_vX.cli` 형태로 호출되므로 패키지 위치가 곧 인터페이스입니다.

정리가 필요하면 구조를 바꾸는 대신 이 문서의 지도를 갱신하고, 새 산출물은 해당 버전
디렉터리 안에 둡니다.

## 5. 현재 활성 버전 (v1.4) 빠른 확인

```bash
.venv-p2/bin/python -m pytest tests_v1_4 -q
.venv-p2/bin/python -m interp_v1_4.cli --help
.venv-p2/bin/python -m interp_v1_4.smoke --device cpu --output experiment_v1_4/smoke/<새_경로>
```

출력 디렉터리는 항상 새로 만들고 기존 결과를 덮어쓰지 않습니다. 과거 버전을 감사할 때는
`tests`/`interp`(v1.0), `tests_v1_1`/`interp_v1_1`(v1.1), `tests_v1_2`/`interp_v1_2`(v1.2)를
같은 방식으로 사용합니다.

진행 상태와 다음 작업은 `experiment_v1_4/P1_STATUS.md`, `P2_STATUS.md`, `P3_STATUS.md`에서 확인합니다. 설계 당시 README는 원본 hash 보존을 위해 유지합니다.

## 6. 작업 위생

- 임시 파일·캐시(`__pycache__`, `.pytest_cache`, `.DS_Store`, `.ipynb_checkpoints`)는 커밋하지
  않으며, 발견하면 삭제합니다.
- 큰 산출물(`.zip`, 대용량 hash 목록)은 `.gitattributes`의 Git LFS 규칙을 따릅니다.
- 단계 완료 후 Git 갱신 절차는 [AGENTS.md](./AGENTS.md) §6을 따릅니다.
