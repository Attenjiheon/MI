# P6 — block 0·3·7·11 READ SAE

2026-09-25. **완료·전체 반환 감사 통과. 다음은 P7 SAE 평가·개입이다.** P5 완료 증빙을 입력으로 사용한다.
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
- [x] 실행 환경에서 사용할 cache bytes와 source manifest hash를 대조한다.
- [x] 네 층 h/u/m train scalar 통계 36개를 확보·검증한다. Block 0의 검증된 9개를 재사용하고 3·7·11의 27개를 계산한다.
- [x] 네 층의 cache hash·READ key·전처리 36개를 묶은 P6 실행 입력 manifest를 작성·검증한다. P5 receipt만으로 이 gate를 통과 처리하지 않는다.
- [x] 네 층용 [P6 실행 계약](p6_r1/contract.json)에 24 run·sampler/init RNG·code hash·P5 receipt hash를 고정했다. 실제 cache bytes와 36개 통계의 실행 입력 manifest는 Colab 검증 대기다.
- [x] 네 층의 hook·전처리·역변환·identity/full/sparse patch와 k=4/16 저장/재개를 CPU debug 입력으로 검증했다. [CPU smoke](smoke/p6_cpu_r1/smoke.json).
- [x] [Colab 노트북](notebooks/P6/P6_colab_read_sae_r1.ipynb)·[입력 번들](bundles/P6/v1_4_p6_bundle_r1.zip)·checksum·환경 기록·다운로드 절차를 준비하고 노트북 schema/셀 문법 및 격리된 번들의 import/hash를 검증했다.
- [x] 현재 코드 hash의 실제 Colab GPU smoke를 통과한다.

## 학습 및 완료 증빙

- [x] 최초 100 updates에서 seconds/update·peak VRAM을 측정하고 총 예산에 포함한다.
- [x] 필수 24 runs 모두 5,000 updates를 완료한다.
- [x] Best/last checkpoint, 학습 곡선, mu/s, optimizer·RNG·cursor·실제 draws·시간·환경/hash를 보존한다.
- [x] 반환 산출물의 checksum·수치·validation checkpoint 선택·run 누락을 검증한다.
- [x] 상태·registry·manifest를 검증 증빙으로 갱신했다. 이 기록을 포함하는 완료 커밋과 원격 main SHA 일치로 Git 반영을 확인한다.

## 후속 평가 준비

P7/P8의 의미·fidelity·인과 평가는 네 층 모두에 적용한다.
각 평가 전에 block 3·7·11의 좌표/random 전체·128후보 기준선을 계산·검증한다.
모든 LM·layer·hook의 난수·후보 subset·feature 선택 규칙을 평가 전에 고정하고,
같은 READ target·causal origin을 사용해 한 층씩 개입한다.
준비 또는 중단된 run은 학습 완료로 집계하지 않는다.

## 변경 기록

2026-09-25: 필수 READ 분석을 block 0·3·7·11로 확정하고 관련 범위·예산·완료 조건을 정리했다. P5 층별 결과를 확인한 뒤, 12층 모델의 깊이에 따른 표현 차이를 평가하기 위한 변경이다. 이 확장을 P5 test 관측 전 사전등록으로 취급하지 않는다.

## 2026-09-25 실행 준비 결과

[준비 보고서](results/p6_preparation_r1/REPORT.md)와 [검증 기록](results/p6_preparation_r1/verification.json)을 따른다.
Colab에서는 원본 Drive `P5_r2`를 읽고 `P6_r1`에 새 결과를 저장한다.
100-update 처리량 측정을 예산에 포함하고, 매 250 updates 전체 validation MSE를 평가한다.
완전 저장 checkpoint에서 optimizer·PCG64 sampler·global RNG·draw cursor를 복구한다.
`best_checkpoint`/`last_checkpoint`는 보존된 update 파일을 가리키며 선택에 test를 사용하지 않는다.
현재 실제 P6 본학습은 0/24 runs다. 36개 통계·실제 GPU smoke·본학습·반환 감사는 아직 미완료이며 phase 완료 커밋/push 조건을 충족하지 않았다.

## 사용자 GPU 실행 및 검산 오류 반환

반환 로그에 24개 run 결과 파일과 training_complete.json이 있고 원본 계약 file hash 불일치는 없다.
앞선 3 runs의 검산 후 seed0/layer3/k16/update250에서 validation MSE 불일치가 발생했다.
현재는 학습 완료 보고를 받은 상태이며 전체 반환 감사 완료로 간주하지 않는다.
[검산 보완 기록](results/p6_audit_correction_r2/REPORT.md),
[r2 재검산 노트북](notebooks/P6/P6_completed_run_audit_r2.ipynb),
[r2 입력 ZIP](bundles/P6/v1_4_p6_audit_r2.zip)을 따른다.
원본 계약·학습 코드·checkpoint·선택값·허용 오차를 유지하고 검산 연산 방식을 맞췄다.
실제 GPU에서 불일치가 해소되는지와 24개 run의 반환 검증은 아직 대기다.

## r2 수치 검산 반환 확인

[로컬 반환 검증](results/p6_numeric_return_audit_r2/REPORT.md): 24 runs/480 validation checkpoint의
MSE가 모두 정확히 일치했다. 기존 실패 수치는 legacy 연산으로 재현되고 r2에서 해소됐다.
소형 ZIP의 코드·배송 manifest·입력 identity·선택 규칙·환경·checkpoint hash 참조를 대조했다.
24 runs/120,000 updates/61.44M draws 보고가 있으나 실제 checkpoint·통계·smoke·학습 기록의
전체 반환 감사가 남아 P6 완료 체크박스와 Git 완료 절차는 보류한다. 재학습은 필요하지 않다.

## 최종 전체 반환 감사 — P6 완료

[완료 판정](results/p6_final_audit_20260925/completion.json), [감사 보고서](results/p6_final_audit_20260925/REPORT.md),
[선택 SAE 24개](results/p6_final_audit_20260925/selected_sae_manifest.json)를 기준으로 P6를 완료한다.
24 runs/120,000 updates/61.44M draws, 504개 checkpoint와 480개 validation 선택 검증을 마쳤다.
위 준비·오류·대기 절은 당시 기록이며 이 최종 판정으로 갱신한다. P7·P8·P9는 아직 미실행이다.
