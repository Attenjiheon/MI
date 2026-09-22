# P0 규격·현황 정리

판정일: 2026-09-10  
적용 규격: `experiment-spec-v1.0`  
상태: `passed`

## 우선순위와 범위

03 §1이 01보다 우선한다. 따라서 Transcoder는 선택 확장이 아니라 필수이며, LM 학습 입력은 GPU 온라인 생성이 아니라 CPU 사전 생성 shard다. 필수 순서는 행동 gate 통과 모델의 block 0 READ probe → SAE(k=4/16) 평가 → Transcoder(k=4/16) 평가 → LM seed 0 sparse seed 1 반복이다. Update, block 1, length, m→m SAE는 필수 READ 완료 후 자원에 따라 별도 run으로 수행한다. Crosscoder, 변수 8개, 대규모 model/seed sweep은 기본 범위에서 제외한다.

## 기존 산출물 판정

| 대상 | 존재 | 규격/검증 판정 | 조치 |
|---|---:|---|---|
| `corpus/` 생성·replay·audit 코드 | 예 | manifest에 생성/검증 코드 hash가 있고 단위 테스트 통과 | P1 입력으로 재사용 |
| `tests/test_corpus.py` | 예 | Python 3.13.4 + NumPy 2.2.6에서 5/5 통과 | 재사용. 생성 환경 lock과 동일하다는 주장은 하지 않음 |
| `data/language_v1/` | 예 | language-v1.0, 51,543 sequences, 3,004,531 train tokens, 모든 quota 충족 | immutable 입력으로 재사용 |
| CPU validation | 예 | `status=passed`, 768 transitions, 51,543 independent replays, 3,072 pair replays | 재사용 |
| post-write audit | 예 | 전체 checksum/alignment/replay/split/holdout/prefix audit 재실행 통과 | 재사용 |
| Language/LM/SAE/TC/probe/evaluation config | 아니오 | P0 시작 시 부재 | `experiment_v1/configs/`에 생성 |
| Transformer·학습·해석 entry point | 아니오 | 미구현·미검증 | P2 대상. 실행 가능하다고 간주하지 않음 |
| CPU/Colab 환경 lock | corpus 전용 요구사항 외 부재 | P2 smoke 전에는 확정 불가 | `experiment_v1/environment/`에 P2에서 기록 |

기본 `python3`은 Python 3.10.11이며 NumPy가 없어 검사를 시작하지 못했다. 재검사는 Python 3.13.4/NumPy 2.2.6에서 통과했다. 원본 생성 환경은 manifest의 Python 3.12.14/NumPy 2.3.5이므로, 현 재검사는 현재 산출물 무결성·독립 replay 증빙이지 생성 환경의 bitwise 재현 증빙은 아니다. P2에서 실제 CPU/Colab lock을 별도로 고정한다.

## 고정한 선택 규칙

- LM checkpoint와 행동 gate는 validation만 사용하며 test는 최종 1회 보고에만 쓴다.
- Dictionary checkpoint는 interpretation validation MSE로 선택한다.
- Probe의 전처리·ANOVA ranking·lambda·feature 수·threshold는 train/validation에서만 선택한다.
- Causal feature와 matching bin은 interpretation train/validation 및 causal validation에서 고정하고 causal test 효과로 수정하지 않는다.
- 선택 분석은 모든 필수 READ 결과가 끝난 뒤에만 시작하며 각각 별도 config/run ID를 사용한다.

## 버전·경로 정책

- 재사용 입력: `data/language_v1/` (쓰기 금지).
- 새 실행 루트: `experiment_v1/runs/`; LM별 루트는 `experiment_v1/runs/lm_seed_{0,1,2}/`.
- 결과와 상태: `experiment_v1/results/`; 초기 registry는 `run_registry.csv`.
- 새 코퍼스가 필요하면 기존 디렉터리를 수정하지 않고 새 semantic version 경로에 생성한다.
- 논리 경로와 실제 경로의 전체 대응은 `experiment_v1/data/README.md`, run ID·seed·manifest 형식은 `configs/run_protocol.json`에 고정했다.

## P0 증빙

- 규격 문서 결합 hash: `ed006e187df69e00bc41ff329dd6f8d3bbdec1b8efe53cea1648c5ac522318c3`
- 코퍼스 코드·검증 코드 결합 hash: `f484bba3817d501bfd75b74fa19be0a55a5e240133b1708e80a995e3af3a6167`
- 코퍼스 manifest hash: `bff7bc9bd4d13bc264c604ea9104210d3e3da64ff529f980c4f53589543e5b26`
- 검사 명령: `python3.13 -m unittest discover -s tests -v`
- audit 명령: `python3.13 -m corpus.audit data/language_v1`

미검증 항목은 모델·optimizer·hook·mask·patch·저장/재개의 실제 구현, 400,640/131,712 parameter count, GPU 결정성 및 런타임이다. 이들은 P0 완료 판정에 포함하지 않고 P2 smoke gate로 남긴다.
