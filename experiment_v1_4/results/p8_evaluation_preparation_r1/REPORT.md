# P8 TC 평가 준비 검증

**로컬 검증 통과. 실제 평가 0/24, P8 미완료.**

학습 전에 동결한 평가 규칙을 유지하고 반환 감사를 통과한 TC 24개의 실제 checkpoint hash를 연결했다.
Full u/full m, m 좌표/random, TC latent의 전체/128후보, 단일/≤4, 변수 전이, fidelity, 한 READ 대체,
changed/unchanged 양방향과 모든 인과 대조, matching coverage, cluster CI 및 SAE/TC 비교 표·그림을 실행한다.
입력 u 정규화로 latent를 계산하고 출력 m의 scale으로 패칭하며 residual skip은 원본을 유지한다.
P7 h 복원과 TC m 예측은 서로 다른 target이라고 그림에 명시한다.

독립 압축해제 폴더에서 19 tests 통과, 노트북 schema·셀 문법 및 8개 layer/k CPU smoke 통과.
실제 semantic 함수의 세 split fidelity가 m target인지 별도 수치 검산을 통과했다.
CPU smoke 22.30초, 환경 `92a9339c663b9cd7`.

[노트북](../../notebooks/P8/P8_colab_read_tc_evaluation_r1.ipynb) · [입력 ZIP](../../bundles/P8/v1_4_p8_evaluation_r1.zip)

ZIP 418,555,326 bytes, SHA-256 `3a122c4fb1e28113e0349c80aa26da88968c5f47997b4c9f5ab97fa9f82c1b66`.
Drive `boolean_interp_v1_4/`에 올리고 GPU 런타임에서 순서대로 실행한다.
원본 `P5_r2`를 읽고 새 결과는 `P8_evaluation_r1`에 저장한다. 학습을 반복하지 않는다.
Fit·NPZ·causal pair 단위로 저장/재개하며 반환 ZIP 및 checksum을 생성한다.
원본 cache 기반 완전한 refit/fidelity/CI 재현은 결과 반환 후 별도 감사하며 준비만으로 이를 수행했다고 기록하지 않는다.
