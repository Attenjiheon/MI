# READ 네 층 분석 문서 검증

2026-09-25. **현재 안내·명세 12개 문서의 정합성 검증 통과.**

## 문서와 계약

- [문서 목록](updated_documents.json): 루트 01–03·phase·AGENTS·README, v1.4 DESIGN·README·CURRENT·P4/P5/P6 상태.
- [READ 분석 계약](../../experiment_v1_4/analysis_plan.json): block 0·3·7·11, SAE 24 runs, TC 24 runs, 초기화 반복 16 runs.
- 총 64 runs, 320,000 updates, 163.84M draws. 각 층에서 의미·후보 수 대조·fidelity·근사 대체·인과 평가를 수행한다.
- P6는 미실행이며 실행 config·입력·전처리·Colab/GPU smoke 준비가 필요하다.

## 검증 증빙

[검증 결과](verification.json)와 [현재 파일 hash](current_manifest.json)를 따른다.

- 현재 문서 12개의 네 층 범위와 마지막 변경 기록 위치 확인.
- 현재 문서의 로컬 링크 195개 확인.
- 수정 전 원본 12개 snapshot SHA256 확인.
- 실행 조합 64개의 유일성과 24/24/16 run 수·update/draw 산식 확인.
- 세 LM × 네 층 × 세 hook의 scalar 통계 36개와 준비 상태 확인.
- 동결 design manifest의 3개 파일 hash, LM 계약·기존 manifest·P5 cache manifest의 원본 bytes 확인. Registry의 과거 행은 보존하고 P6 준비 행을 추가했다.
- 저장소 검증 통과: archive 경로 15개, source snapshot 6개, 계약 hash 10개, 링크 145개.
- 활성 수정본의 `git diff --check` 통과. 수정 전 원본 snapshot의 Markdown 줄 끝 공백은 byte 보존을 위해 유지했다. LM 학습이나 해석 test 평가는 실행하지 않았다.

저장소 검증 명령:

```bash
python scripts/verify_repository_layout.py
git diff --cached --check -- . ':(exclude)maintenance/read_layers_20260925/originals/**'
```

현재 파일·snapshot hash 재검증:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
record = Path('maintenance/read_layers_20260925')
for name, digest in json.loads((record / 'current_manifest.json').read_text())['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
for name, item in json.loads((record / 'originals.json').read_text())['files'].items():
    assert hashlib.sha256(Path(item['snapshot']).read_bytes()).hexdigest() == item['sha256'], name
print('Current documents and preserved snapshots verified')
PY
```

## 변경 기록

P5 층별 결과 관측 후 12층 모델의 깊이에 따른 표현 차이를 평가하기 위해 필수 READ 범위를 block 0·3·7·11로 확정했다. 각 문서 본문은 현재 규격으로 작성하고 변경 내용·사유는 마지막에 기록했다. 동결 DESIGN/README를 포함한 수정 전 bytes는 [원본 목록](originals.json)에 보존했으며 저장소 검증기는 해당 snapshot에서 기존 hash를 확인한다. 과거 실행 결과와 동결 기계 계약은 보존하고 현재 해석 범위는 별도 READ 분석 계약으로 명시했다.

이후 문서 정합성 감사에서 네 가지를 보정했다. Registry에는 과거 P5 행을 유지하고 네 층 P6 준비 행을 추가했다. P5 cache receipt의 전 층 파일 hash와 block 0에만 있는 scalar 전처리를 구분하고, 나머지 27개 통계 및 네 층 실행 입력 manifest는 P6 미완료 gate로 표시했다. 상세 명세에는 P5 READ 80k와 선택 P10 update 80k의 용량을 분리했으며, 네 층 smoke의 시점을 P6 본학습 전으로 명시했다. [현재 파일 hash](current_manifest.json)와 [검증 결과](verification.json)는 이 보정 후 파일을 대상으로 갱신했다.
