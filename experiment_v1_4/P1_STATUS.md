# v1.4 P1 — 수정본 재생성 중, 미통과

최초 시도 `data/language_v1_4/`는 실제 shard 합계 64,007,030 prediction tokens,
537,600 sequences, 133 shards를 생성했다. 그러나 historical reserved READ-prefix 목록을
불완전하게 불러와 새 split 및 train extension에서 과거 prefix와의 충돌이 발견됐다.
기존 postwrite audit가 통과하더라도 같은 결함 있는 loader를 사용하므로 승인 근거가 아니다.

또한 실제 RNG는 v1.4였으나 일반 metadata의 seed 기록은 legacy namespace였다.
두 결함 모두 수정하고 `corpus_rebuild.json`에 지정한 별도 경로에 처음부터 재생성한다.
최초 산출물은 원래 경로에 보존한다. test의 토큰/hash를 누출 감사한 것이며 모델 test 점수는
계산하거나 선택에 사용하지 않았다.

아직 없는 증빙: 수정된 코퍼스의 독립 replay·historical prefix isolation·quota·seed 감사,
동결 runtime config, P2 smoke 및 Colab 전달물. P1/P2/P3를 완료로 표시하지 않는다.

## 계속 실행 중인 작업

`python -u scripts/generate_v1_4_corpus.py`가 `rebuild_01`을 생성 중이다.
최초 실패 시도의 오래된 감사 프로세스만 중단했으며, 원본 파일은 삭제하지 않았다.
수정본은 v1.0·v1.2·v1.3 파일과 inherited registry를 모두 합쳐 중복을 거부하고,
새 metadata의 RNG seed를 실제 v1.4 namespace와 대조한다.

`python -u scripts/finish_v1_4_preparation.py --wait-for-p1`가 수정본 감사 후
config 동결, 전체 test, CPU smoke, 실제 CLI 재개 검증, Colab 번들/노트북 생성 순으로 실행한다.
어느 검증이든 실패하면 즉시 중단한다. 현재 상태와 단계별 로그는
`results/local_preparation/`에서 확인한다. GPU 학습이나 Git push는 이 스크립트가 실행하지 않는다.
