# P8 — READ Transcoder 학습·평가

2026-09-30. **P8 완료. TC 24/24 학습·평가·원본 재현 반환 감사 통과.**

선행 조건은 [P7 최종 감사](results/p7_final_audit_20260928_01/completion.json)이다.
세 동결 LM × block 0·3·7·11 × k=4/16, sparse seed 0으로 24 TC를 학습한다.
각 run은 5,000 updates와 2.56M draws이며 P8 학습 예산은 총 120k updates/61.44M draws다.

## 실행 파일 이력

- [Colab 학습·진행 표시·수치 검산 노트북](notebooks/P8/P8_colab_read_tc_training_r2_progress.ipynb)
- [r2 입력 ZIP](bundles/P8/v1_4_p8_training_r2_progress.zip), [checksum](bundles/P8/v1_4_p8_training_r2_progress.sha256.json)
- [학습 계약](p8_r1/contract.json), [동결 평가 규칙](p8_r1/evaluation_rules.json)
- [준비 보고서](results/p8_preparation_r1/REPORT.md), [검증 증빙](results/p8_preparation_r1/verification.json)

ZIP을 내 드라이브 `boolean_interp_v1_4/`에 올리고 GPU 런타임에서 위부터 실행한다.
원본 입력은 `P5_r2`, 새 출력은 `P8_training_r1`이다. 중단 뒤 같은 노트북으로 checksum checkpoint에서 재개한다.
끝나면 결과 ZIP과 checksum JSON을 반환한다.

## 확인한 항목

15 tests 및 별도 압축해제 번들의 15 tests, 네 층 CPU smoke와 두 k의 정확한 TC 재개가 통과했다.
입력 u와 출력 m 통계를 분리했고 loss는 m 예측 MSE다. 대응 SAE의 position draw seed를 그대로 사용한다.
출력 패칭에 s_m을 곱하고 원본 residual skip을 보존하는 네 층 debug 검사를 통과했다.
이 결과는 CUDA 본실행 또는 필수 인과 평가의 완료 증빙이 아니다.

## 다음 순서와 완료 조건

1. Colab에서 원본 cache·전처리·현재 CUDA smoke를 검증한다.
2. 24 TC의 5,000 updates 및 모든 validation checkpoint 독립 수치 검산을 끝낸다.
3. 반환 학습 증빙을 감사하고 실제 선택 checkpoint hash를 연결한 평가 전달물을 준비한다.
4. 동결한 규칙으로 full u/m·좌표/random·TC 의미 평가와 128후보 대조, fidelity, 한 위치 대체, 모든 필수 인과 대조·표·그림을 실행한다.
5. 최종 반환 감사를 통과한 뒤에만 P8 완료와 Git main 업데이트를 처리한다. 이후 P9 반복 16 runs로 진행한다.

평가 실행기는 이번 학습 전달물의 범위에 포함되지 않는다. 학습 완료를 P8 완료로 간주하지 않는다.

## r2 진행 표시 추가

[진행 표시 노트북](notebooks/P8/P8_colab_read_tc_training_r2_progress.ipynb)과
[r2 입력 ZIP](bundles/P8/v1_4_p8_training_r2_progress.zip)을 사용한다.
전체/현재 run의 저장된 update 진행률, 경과시간, 초/update, 현재 run·선택 runs·전체 runs의 예상 남은 시간을 표시한다.
화면은 약 1초마다 갱신되며 update 수는 기존 100/250-update checkpoint 로그에서 갱신된다.
ETA는 관측된 학습·validation·저장 시간을 사용한다. 이후 입력 로드·최종 hash 검사·독립 검산·후속 평가는 제외한다.
전체 stdout/stderr는 Drive 출력의 `progress_logs/`에 보존한다.

r1 번들의 기존 파일은 모두 byte-identical이며 학습 계약·RNG·출력 `P8_training_r1`을 유지한다.
이미 실행한 r1 checkpoint를 그대로 재개할 수 있다. r1 노트북과 번들도 보존했다.
별도 압축해제 번들의 17 tests, 노트북 schema/셀 문법 및 subprocess 로그·실패 전달 검증을 통과했다.
[검증 증빙](results/p8_progress_preparation_r2/verification.json)을 따른다. CUDA 실행 여부나 P8 완료 상태는 바뀌지 않았다.

## 2026-09-29 학습 반환 감사 완료·평가 준비

[학습 완료 판정](results/p8_training_audit_20260929_01/completion.json), [감사 보고서](results/p8_training_audit_20260929_01/REPORT.md),
[선택 TC manifest](results/p8_training_audit_20260929_01/selected_tc_manifest.json)을 확인했다.
24 runs, 504 checkpoint, 480 validation 수치 검산 및 CUDA smoke·재개 감사 통과. 학습은 반복하지 않는다.

다음은 [TC 평가 노트북](notebooks/P8/P8_colab_read_tc_evaluation_r1.ipynb)과
[평가 입력 ZIP](bundles/P8/v1_4_p8_evaluation_r1.zip)이다.
ZIP을 내 드라이브 `boolean_interp_v1_4/`에 올리고 GPU 런타임에서 실행한다.
입력은 `P5_r2`, 새 출력은 `P8_evaluation_r1`이며 결과 ZIP과 checksum JSON을 반환한다.
[평가 준비 검증](results/p8_evaluation_preparation_r1/verification.json): 독립 번들 19 tests·노트북·네 층 두 k CPU smoke 통과.
기존 학습·평가 규칙을 바꾸지 않았다. 실제 평가 및 원본 재현 반환 감사 전에는 P8 완료나 P9 진입을 처리하지 않는다.
앞 절의 0/24 학습 상태·평가 실행기 미포함 문구는 당시 학습 전달물 이력이다.

## 2026-09-29 평가 반환·원본 재현 감사 대기

24 TC의 의미·fidelity·대체·인과 평가가 반환되었으며 로컬 검사를 통과했다.
[반환 감사 보고서](results/p8_return_audit_20260929_01/REPORT.md), [검증 및 집계](results/p8_return_audit_20260929_01/status.json)를 따른다.
원본 cache를 사용하는 refit·전체 예측/fidelity/CI 재현이 남아 P8 완료 판정은 보류한다.

다음은 [추가 감사 노트북](notebooks/P8/P8_source_return_audit_r1.ipynb)과
[감사 ZIP](bundles/P8/v1_4_p8_source_audit_r1.zip)이다. ZIP을 내 드라이브 `boolean_interp_v1_4/`에 업로드하고 T4 GPU에서 실행한다.
기존 `P5_r2`와 `P8_evaluation_r1`을 읽고 새 `P8_source_audit_r1`에 감사 결과만 저장한다.
독립 번들 13 tests 및 노트북 schema·셀 문법 검증 통과. [준비 검증](results/p8_source_audit_preparation_r1/verification.json)을 따른다.
반환 ZIP과 checksum JSON을 검증한 뒤에만 P8 완료/P9 진입을 판단한다. 앞 절의 평가 0/24는 당시 준비 상태다.

## 2026-09-30 P8 완료

[최종 보고서](results/p8_final_audit_20260930_01/REPORT.md), [완료 판정](results/p8_final_audit_20260930_01/completion.json)을 확인했다.
24 TC의 전체 필수 평가와 원본 재현 반환 감사가 통과했다. 840개 고유 refit, 1,920 probe 작업, 3,360 의미 보고와 fidelity/CI 재현 기록을 검증했다.
필수 dictionary 64개 중 48개 학습·평가 완료, P9 sparse seed 1 반복 16개는 미실행이다. 전체 실험은 미완료다. 앞선 대기·실행 안내는 당시 이력이다.
