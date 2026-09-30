# P8 TC 학습 반환 감사 — 2026-09-29

**학습 24/24 통과. P8 전체는 미완료이며 의미·fidelity·대체·인과 평가가 남았다.**

- 세 LM × block 0·3·7·11 × k=4/16, sparse seed 0.
- 총 120,000 updates, 61,440,000 position draws. 각 run 5,000 updates, 2,560,000 draws.
- ZIP 외부 SHA-256과 내부 1,748개 member hash 확인.
- 실제 504 checkpoint의 model·optimizer·RNG·sampler cursor·unit norm·입출력 통계를 제한된 deserialization으로 검증.
- 480개 validation MSE의 반환 독립 검산 통과. 저장값과 검산값의 최대 절대 오차 0.0.
- 모든 20개 validation 후보에서 최소 MSE/이른 update tie-break를 확인하고 선택 checkpoint 24개를 보존.
- 감사된 P6 scalar 통계 36개와 반환값 일치, u/m 입력 tensor manifest 48개 및 대응 SAE의 draw key/seed 일치.
- 현재 계약의 CUDA 네 층 smoke, TC 두 k의 bitwise model/optimizer/sampler 재개 증빙 확인.
- 실패 기록 0개. 환경 `e74fb1dcf8112ca0`. 학습 계산 시간 합계 546.45초, validation 12.61초.

첨부 checksum 파일명에는 선행 공백과 `sha356` 오타가 있지만 내용의 `sha256` 필드는 수신 ZIP과 일치했다.
원본 파일은 바꾸지 않았다. SHA-256: `bea70775b6d69ff4558b1b8f0d0b0e83ce20a8161dc4577d790f3da958490766`.

## 증빙과 한계

[검증 JSON](verification.json), [실행 로그](audit.log), [run별 학습 표](runs.csv),
[선택 TC manifest](selected_tc_manifest.json), [반환 감사 코드](verify_return.py)를 따른다.
원본 ZIP은 `experiment_v1_4/evidence/P8/v1_4_p8_training_return_1790613676530123020.zip`에 보존한다.
선택 checkpoint만 `experiment_v1_4/p8_selected/`에 별도 보존했고 나머지 대형 tensor는 ZIP에서 직접 검사했다.

원본 P5 activation은 Drive에 있어 로컬에서 다시 생성/학습하지 않았다. 수치 검산은 hash가 확인된 Colab 감사의 실제 480개 기록과 checkpoint/곡선을 로컬에서 대조한 것이다.
로컬 macOS/PyTorch 2.10과 Colab/PyTorch 2.11의 초기화 byte 재현은 진단으로만 기록하며 환경 간 bitwise 일치를 주장하지 않는다.
학습 계산·validation 시간은 checkpoint I/O나 전체 wall time을 포함하지 않는다.

## 다음 단계

학습 전에 고정한 `p8_r1/evaluation_rules.json`을 유지하고 선택 checkpoint hash를 연결한
[평가 노트북](../../notebooks/P8/P8_colab_read_tc_evaluation_r1.ipynb)과
[입력 번들](../../bundles/P8/v1_4_p8_evaluation_r1.zip)을 실행한다.
전체 TC 평가 및 최종 반환 감사 전에는 P8 완료/P9 진입/phase 완료 Git push를 하지 않는다.
