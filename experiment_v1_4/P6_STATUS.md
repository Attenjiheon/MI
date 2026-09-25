# P6 — block 0·3·7·11 READ SAE

2026-09-25. **실행 준비 전·미실행.** P5 완료 증빙을 입력으로 사용한다.
규격은 [READ 분석 계약](analysis_plan.json), [03 §8](../03_experiment_spec.md#8-sae-상세-명세),
[phase P6](../phase.md#8-p6--read-sae-학습)를 따른다.

## 고정 범위

- LM seed 0·1·2 × layer 0·3·7·11 × k=4/16 × sparse seed 0 = **24 SAE runs**.
- 각 run 5,000 updates, batch 512, dictionary width 512, Adam lr=1e-3.
- 매 250 updates 전체 validation MSE 평가·저장, 5,000 updates 후 최소 MSE checkpoint 선택.
- P6 합계 120,000 updates, 61.44M position draws. 라벨 loss·early stopping·dead 재초기화 없음.
- P7은 이 24개 모델의 평가·개입, P8은 같은 범위 TC 24 runs, P9는 LM seed 0의 네 층 sparse seed 1 반복 16 runs다.

## 입력과 준비 gate

- [x] [동결 LM 3개](results/frozen_test_audit_20260922_01/frozen_lms.json)와 [P5 완료 증빙](results/p5_final_audit_20260925_01/completion.json) 확인.
- [x] P5 전 12층 cache·READ key·quota·full probe의 독립 감사 완료. 원본은 Drive `boolean_interp_v1_4/P5_r2`에 보존.
- [x] [P5 cache receipt](results/p5_final_audit_20260925_01/p6_cache_manifest.json)의 240개 파일은 [추출 계약](p5_r2/contract.json)에 따라 전 12층을 담는다. Receipt의 `layer: 0`과 전처리 9개는 block 0에만 적용한다.
- [ ] 실행 환경에서 사용할 cache bytes와 source manifest hash를 대조한다.
- [ ] 네 층 h/u/m train scalar 통계 36개를 확보·검증한다. Block 0의 검증된 9개를 재사용하고 3·7·11의 27개를 계산한다.
- [ ] 네 층의 cache hash·READ key·전처리 36개를 묶은 P6 실행 입력 manifest를 작성·검증한다. P5 receipt만으로 이 gate를 통과 처리하지 않는다.
- [ ] 네 층용 P6 실행 config·sampler/init RNG·code/input hash·run manifest를 고정한다.
- [ ] 네 층의 hook·전처리·역변환·patch와 저장/재개를 debug 입력으로 검증한다.
- [ ] Colab 노트북·입력 번들·checksum·환경 기록·다운로드 절차를 준비한다.
- [ ] 현재 코드 hash의 실제 Colab GPU smoke를 통과한다.

## 학습 및 완료 증빙

- [ ] 최초 100 updates에서 seconds/update·peak VRAM을 측정하고 총 예산에 포함한다.
- [ ] 필수 24 runs 모두 5,000 updates를 완료한다.
- [ ] Best/last checkpoint, 학습 곡선, mu/s, optimizer·RNG·cursor·실제 draws·시간·환경/hash를 보존한다.
- [ ] 반환 산출물의 checksum·수치·validation checkpoint 선택·run 누락을 검증한다.
- [ ] 상태·registry·manifest를 증빙과 함께 갱신하고 phase 완료 Git 절차를 수행한다.

## 후속 평가 준비

P7/P8의 의미·fidelity·인과 평가는 네 층 모두에 적용한다.
각 평가 전에 block 3·7·11의 좌표/random 전체·128후보 기준선을 계산·검증한다.
모든 LM·layer·hook의 난수·후보 subset·feature 선택 규칙을 평가 전에 고정하고,
같은 READ target·causal origin을 사용해 한 층씩 개입한다.
준비 또는 중단된 run은 학습 완료로 집계하지 않는다.

## 변경 기록

2026-09-25: 필수 READ 분석을 block 0·3·7·11로 확정하고 관련 범위·예산·완료 조건을 정리했다. P5 층별 결과를 확인한 뒤, 12층 모델의 깊이에 따른 표현 차이를 평가하기 위한 변경이다. 이 확장을 P5 test 관측 전 사전등록으로 취급하지 않는다.
