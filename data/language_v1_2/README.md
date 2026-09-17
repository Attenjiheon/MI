# 인공어 코퍼스 language-v1.0

02_language_and_corpus.md의 문법·실행 의미·분포를 구현한 CPU 생성 코퍼스입니다. 02에는 구체적인 표본 수가 없어 03_experiment_spec.md의 5절 수량을 적용했습니다. LM 학습은 실행하지 않았습니다.

## 생성 결과

| 데이터 | 수량 |
|---|---:|
| LM train | 26,048 sequences / 3,004,531 예측 토큰 |
| IID validation / test | 512 / 2,048 sequences |
| 핵심 진단 validation / test | 3조건 × 512 / 3조건 × 1,024 sequences |
| 거리 진단 test | 3조건 × 1,024 sequences |
| 진리표 진단 test | 12조건 × 256 sequences |
| 답 균형 test | B0/B1 각각 512 sequences |
| 합성 holdout test | 2패턴 × 1,024 sequences |
| 길이 test | 1,024 sequences, 33~48블록 |
| 해석 train / val / test | 3,132 / 627 / 1,256 sequences |
| 해석 READ 위치 | 50,000 / 10,000 / 20,000 |
| 해석 update 위치 | 50,000 / 10,000 / 20,000 |
| 반사실적 validation | changed 512 + unchanged 512 pairs |
| 반사실적 test | changed 1,024 + unchanged 1,024 pairs |

총 51,543개 원본 시퀀스를 저장했습니다. 이 수에는 반사실적 pair의 origin 3,072개가 포함됩니다. 각 pair 유형은 memory/composition을 절반씩 포함합니다. Train은 4,096개 단위 6개 shard와 1,472개짜리 마지막 shard이며, 전체가 64-sequence batch로 나누어집니다. 3M budget을 처음 넘는 완전한 batch까지 생성했습니다.

Train READ는 414,348개이며 B1은 205,436개입니다. 정답 균형 rejection은 적용하지 않았습니다. IID validation/test의 관측 다수 클래스 기준선은 각각 약 50.679% / 50.161%입니다.

## 파일 읽기

- `*.jsonl.gz`: 시퀀스 전체 metadata. 각 행에 token_ids, token_roles, 상태, 갱신/READ events, 의존성 노드가 있습니다.
- `*.tokens.jsonl`: 같은 순서의 정수 토큰 배열만 저장한 모델 입력 파일입니다. 행 번호로 metadata와 대응합니다.
- `interpretation/*.read_positions.json`: READ 직후 변수 위치. 정답 토큰 직전 activation을 사용합니다.
- `interpretation/*.update_positions.json`: 갱신의 마지막 operand 위치입니다.
- `causal_pairs/*.jsonl.gz`: suffix와 target 답을 제외한 원본/반사실적 prefix 및 별도 정답입니다. `*.origins.jsonl.gz`는 추적·검산용 전체 원본입니다.
- `manifest.json`: 분포, seed/index 매핑, RNG 최종 상태, 버전, 생성 코드 해시, 수락·거부 수, 파일 SHA-256입니다.
- `corpus_statistics.json`: split별 길이·상태·명령·진리표·변수 값·깊이·거리·정답 통계입니다.
- `cpu_validation.json`, `postwrite_audit.json`: 생성 시 검증 및 저장 파일 재검증 결과입니다.
- `examples.md`: 실제 코퍼스에서 가져온 사람이 읽을 수 있는 예시 3개입니다.
- `reserved_hashes.txt`: train 생성 전에 예약된 원본/반사실적 전체 시퀀스 hash입니다.
- `all_sequence_hashes.txt`: train을 포함한 전체 등록 hash입니다. 저장되지 않은 반사실적 전체 시퀀스 hash도 포함됩니다.

```python
import gzip, json
from pathlib import Path

root = Path('data/language_v1')
with (root / 'train_shards/00000.tokens.jsonl').open() as f:
    token_ids = json.loads(next(f))
input_ids, labels = token_ids[:-1], token_ids[1:]

with gzip.open(root / 'val_iid.jsonl.gz', 'rt') as f:
    example = json.loads(next(f))
answer_positions = [r['answer_token_index'] for r in example['read_events']]
# 예측 logit 위치는 각 answer_token_index - 1입니다.
```

PAD를 제외한 모든 예측 토큰에 동일한 loss 가중치를 사용합니다. 역할·상태·answer mask는 모델 입력에 넣지 않습니다. 학습은 shard 이름 순서, 파일 행 순서로 진행합니다.

## Metadata 해석

상태 배열과 국소 감도 배열 순서는 A/B/C/D입니다. 모든 인덱스는 0-based입니다. `tokens_since_last_update`는 query 변수 위치에서 관련 갱신의 마지막 operand 위치를 뺀 값입니다. 일반 갱신이 없으면 해당 변수의 초기화 마지막 위치를 기준으로 합니다.

`previous_value_or_null`은 해당 query 변수의 가장 최근 일반 갱신 직전 값입니다. 초기화만 있으면 null입니다. READ 횟수는 현재 READ를 제외합니다.

국소 감도는 train에서 null, 나머지 split의 모든 READ에서 계산했습니다. target 블록 시작의 각 bit를 반전한 결과이며, 실제 감도 0과 null을 구분합니다. `holdout_start_dst_flip_survives`는 합성 test의 정보 보존 subset을 나타냅니다. 반복 갱신의 `previous_value_changed`, 첫 READ의 `logical_updates_since_latest_set`으로 규격의 추가 subset을 고를 수 있습니다.

진단 target은 조건을 만족하는 READ 중 균등 선택했습니다. `target_read_ids`만 해당 진단 점수에 사용합니다. 해석 위치는 시퀀스 후보 pool에서 종류별 균등 비복원 추출했으며 라벨에 따른 선택은 하지 않았습니다. Unchanged pair는 변경 SET의 변수와 다른 query 변수에서 target 답 불변을 확인했습니다.

## 검증 및 재현

768개 상태×명령 전이, NOT/XOR 복원, 고정 seed 토큰·metadata 재현성, 전체 READ 정답, role/index/거리 라벨, split 중복, holdout 정책, prefix 한 토큰 차이를 검증했습니다. 저장 파일을 다시 읽어 SHA-256, 토큰·metadata 대응, 의존성 그래프, 국소 감도 및 해석 위치 quota도 검사했습니다. 모두 통과했습니다. Train은 16상태·48명령·12진리표 strata와 25종 인접 2연산을 모두 포함합니다.

프로젝트 루트에서 Python 3.11 이상 및 `requirements-corpus.txt`의 NumPy 2.3.5를 사용합니다. 이 생성에 사용된 정확한 Python 버전은 manifest에 있습니다.

```bash
python3 -m unittest discover -s tests -v
python3 -m corpus.generate --output data/language_v1_reproduced
python3 -m corpus.audit data/language_v1_reproduced
```

작은 실행 확인은 `--smoke`를 추가합니다. 기존 output 디렉터리는 덮어쓰지 않습니다. 생성은 offline 일괄 방식이고, 중단 후 이어쓰기 기능은 없습니다. 실패한 실행은 새 디렉터리에서 같은 seed로 다시 시작합니다. 같은 NumPy 및 생성 코드로 token/metadata 내용과 순서가 재현되며, gzip 헤더의 생성 시각과 manifest 시각은 실행마다 달라질 수 있습니다. 저장 형식은 공간 절약을 위해 JSONL metadata를 gzip 압축했으며 문법과 샘플링 분포는 변경하지 않았습니다.
