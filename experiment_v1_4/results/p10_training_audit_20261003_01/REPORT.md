# P10 Update 학습 반환 감사

2026-10-03. **학습 반환 감사 12/12 통과. P10 전체는 미완료.**

## 확인한 내용

원본 ZIP은 `experiment_v1_4/evidence/P10/v1_4_p10_training_return_1791021261419206673.zip`이며 SHA-256은 `18b0dced48ca11d073f87e10fb8ced46b416a674ebe6bd0e82a16a825fedd67f`다.
외부 checksum·파일 크기와 ZIP의 중복 없는 member 집합을 확인했다. Manifest의 883개 파일 hash가 일치한다.

- block 3 UPDATE, LM seed 0/1/2 × SAE/TC × k=4/16 × sparse seed 0의 12 runs.
- 총 **60,000 updates·30.72M draws**, 252개 checkpoint, 240개 validation MSE 검산 기록.
- 각 checkpoint의 model/optimizer tensor, decoder column norm, draw cursor/PCG64 상태, run/config/input identity, train scalar 통계와 누적 curve를 대조했다.
- 반환 train/val 라벨·위치를 동결 corpus의 독립 replay와 비교했다. train/val 50k/10k, 9개 scalar 통계·18개 tensor receipt를 확인했다.
- 선택은 기존 규칙인 전체 validation MSE 최소·동률 이른 update다. 12개 모두 update 5,000이 선택됐다. 고정 예산 끝에서 선택된 것이 수렴을 입증하지 않으며 추가 학습으로 바꾸지 않는다.
- 선택한 원본 checkpoint 12개를 별도 보존하고 [manifest](selected_dictionary_manifest.json)에 연결했다. 나머지 checkpoint는 원본 ZIP에 유지했다. 기존 결과를 덮어쓰거나 삭제하지 않았다.
- CUDA smoke와 두 도구·두 k의 bitwise debug resume 기록이 같은 실행 계약·환경에 연결된다. 실패 디렉터리는 없다.

## 환경과 자원

Tesla T4, torch **2.11.0+cu130**, NumPy 2.1.3, 환경 ID `6e859d41cc64e591`다. 이전 단계의 cu128 환경과 혼동하지 않는다.
학습 시간 합은 259.83초, validation 시간 합은 5.89초, GPU 수치 검산은 53.72초다. 이 값은 cache 추출·Drive I/O·설치·대기·내보내기를 포함하는 전체 벽시계 시간이나 CU 소비량이 아니다.

## 판정의 범위

GPU 원본 activation cache/inputs는 Drive에 있어 로컬에서 모든 validation forward를 새로 계산하지 않았다.
동결 수치 감사 코드의 hash와 반환된 독립 재계산 240개를 원본 checkpoint·저장 curve·선택과 대조했다. 캐시 원본 재현은 후속 감사에서 수행한다.
Probe 의미·fidelity·변수/연산 전이 평가는 아직 0/12다. 이번 판정을 P10 완료 또는 Update의 인과적 사용 증거로 해석하지 않는다.

정확한 hash·집계는 [completion.json](completion.json)을 따른다. 감사 코드: `scripts/audit_v1_4_p10_training_return.py`.
