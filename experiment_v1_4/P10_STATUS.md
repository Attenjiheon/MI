# P10 — Update block 3 선택 분석

2026-10-04. **학습·평가·원본 재현 반환 감사 12/12 통과. P10 완료. 다음은 P11 최종 집계.**

## 범위와 자원 결정

P9 완료 증빙의 연결 hash 7개, 반환 ZIP 594개 member와 16개 통과 run을 확인했다. 필수 READ dictionary 64개 완료 상태는 유지한다.
사용자가 T4와 잔여 24.59 컴퓨팅 단위를 알려주었다. phase의 권장 순서에 따라 에이전트가 Update 한 층을 기본 범위로 정했다.
사용자가 특정 층을 직접 선택한 것으로 기록하지 않는다.

| 분석 | 결정·이유 |
|---|---|
| Update | **block 3**, LM seed 0/1/2 × SAE/TC × k=4/16 × sparse seed 0 = 12 runs. 초기 층을 지난 기존 READ 관측 지점과 비교한다. 층별 test 점수로 선택하지 않았다. |
| 추가 층 | 생략. Update 한 층의 의미·fidelity·전이 평가에 자원을 우선 배정한다. |
| Length | 생략. phase의 Update 우선순위를 따른다. |
| m→m SAE | 생략. 서로 다른 h/m target의 NMSE만으로 도구 우열을 주장하지 않는다. |
| Update 인과 개입 | 별도 선택 실험으로 생략. 03 §11.2의 READ 개입을 그대로 Update에 적용하지 않는다. Update feature의 인과적 사용을 주장하지 않는다. |

이 결정은 READ 결과가 관측된 뒤의 탐색적 확장이다. P5/P7/P8/P9 test 관측 전 사전등록으로 취급하지 않는다.
새 [계약](p10_update_r1/contract.json)과 [평가 규칙](p10_update_r1/evaluation_rules.json)을 해당 Update test 성능 관측 전에 고정했다.
LM·기존 dictionary·기존 선택 규칙은 변경하지 않는다.

## 학습 전달물 — 완료 이력, 재실행하지 않음

- [T4 Colab 노트북](notebooks/P10/P10_colab_update_training_r1.ipynb)
- [입력 ZIP — 약 435 MB](bundles/P10/v1_4_p10_update_training_r1.zip)
- [SHA-256](bundles/P10/v1_4_p10_update_training_r1.sha256.json)

입력 ZIP을 내 드라이브 `boolean_interp_v1_4/`에 올리고 노트북을 T4 런타임에서 위부터 실행한다.
기존 P5 cache는 필요 없다. 번들에 동결 LM·예약 Update 입력·코드·계약을 포함했다.
새 출력은 Drive `boolean_interp_v1_4/P10_update_l3_r1`에 저장한다.

1. ZIP·구성 파일 checksum 확인, 고정 의존성 설치, 테스트.
2. 현재 GPU/환경 기록과 Update hook·미래 토큰 차단·SAE/TC 재개 smoke.
3. 초기화 SET을 제외한 train 50,000·validation 10,000 위치의 block 3 h/u/m 추출.
4. train 전용 9개 scalar 통계, 18개 train/val tensor와 위치 join·checksum 검증.
5. 12 runs × 5,000 updates × batch 512 = **60,000 updates, 30.72M draws**. SAE는 h→h, TC는 u→m. 같은 LM/k 두 도구는 동일 draw 순서를 사용한다.
6. **252 checkpoint·240 validation MSE** 및 선택, optimizer, RNG cursor, 통계 독립 검산.
7. 마지막 셀에서 반환 ZIP과 checksum JSON 다운로드. 오류·자원 중단 시에도 반환 파일을 만든다.

추출은 128 sequence chunk, 학습은 250-update 저장 경계에서 재개한다. 처음 100 updates의 처리량도 기록한다.
호출당 기본 시간 제한은 2시간이며 정지 경계까지 조금 초과할 수 있다. PAUSED이면 같은 셀을 다시 실행한다.
런타임 변경 시 처음부터 환경·CUDA smoke를 다시 기록하되 완전 저장 checkpoint의 optimizer·RNG·cursor를 복구한다.
환경이 달라진 재개를 동일 환경의 bitwise 재현이라고 주장하지 않는다.

컴퓨팅 단위의 시간당 차감률은 확인되지 않았다. 노트북의 `CU_PER_HOUR`는 Colab UI 실제 값이 있을 때만 입력하고 `REMAINING_CU`를 갱신한다.
24.59 CU로 전체 완료를 보장하거나 고정 시간으로 환산하지 않는다. 추출·학습·validation·검산·후속 CPU probe/CI 시간을 구분한다.

원본 cache/inputs와 smoke tensor는 Drive에 보존하며 반환 ZIP에는 전체 학습 checkpoint·선택·통계·환경·로그·smoke JSON·감사 결과를 넣는다.
재사용하는 P9 학습 엔진의 원시 result에 `p9_complete: false`가 남지만 run identity는 UPDATE/P10 계약이며 P9 실행이나 상태 변경을 뜻하지 않는다. P10 판정은 별도 `p10_status.json`과 반환 감사에서 관리한다.

## 로컬 검증

[검증 JSON](results/p10_preparation_r1/verification.json), [보고서](results/p10_preparation_r1/REPORT.md), [CPU smoke](smoke/p10_cpu_r1/smoke.json).

- ZIP 91개 member의 SHA-256, 독립 압축 해제 환경의 8개 테스트, 노트북 schema와 8개 코드 셀 구문 검증 통과.
- 예약 Update train/val/test 50k/10k/20k 위치를 토큰에서 독립 replay했다. 초기화 위치 제외, split 간 canonical hash 중복 없음.
- 3,138/631/1,255개 interpretation sequence를 확인했다. train/val의 각 16개 라벨·전이 과제는 클래스별 32 positions·16 sequences support를 충족했다. Test의 지표는 계산하지 않았다.
- Update h/u/m 추출 위치, residual 합, 실제 MLP input, 미래 토큰 불변성과 identity patch를 검증했다.
- SAE/TC × k=4/16의 checkpoint 재개 model·optimizer·sampler RNG·curve bitwise 일치와 독립 수치 계산을 확인했다.
- 실제 동결 LM의 GPU cache, 12개 학습 및 반환 검증은 아직 없다. 위 CPU 결과를 GPU 완료 증빙으로 재사용하지 않는다.

## 반환 후 진행

학습 반환의 파일 hash·252 checkpoint·240 validation 값·선택을 감사한 뒤 선택 manifest를 만든다.
그 다음 예약 test 20k 및 초기 LM cache를 연결하여, 고정 규칙으로 의미·fidelity·변수 및 AND/OR→XOR probe 전이를 평가한다.
단일/≤4 feature, 전체/128 후보 대조, full/좌표/random, 초기 LM·token/position·shuffled 대조와 sequence cluster CI를 포함한다.
전이는 probe의 감독 전이이며 dictionary 자체의 domain holdout으로 해석하지 않는다.

선택한 평가와 원본 재현 반환 감사까지 완료한 뒤에만 P10 완료 조건·체크박스·최종 registry를 갱신하고 main에 commit/push한다.
현재는 준비 상태만 기록한다. P11 최종 집계도 남아 전체 실험은 미완료다.


## 2026-10-03 학습 감사 완료·다음 평가 실행

[학습 감사](results/p10_training_audit_20261003_01/REPORT.md), [완료 판정](results/p10_training_audit_20261003_01/completion.json), [선택 모델 12개](results/p10_training_audit_20261003_01/selected_dictionary_manifest.json)를 확인했다.
252 checkpoint·240 validation 수치 검산과 RNG·optimizer·선택 감사가 통과했다. 학습을 다시 실행하지 않는다.
위의 0/12 학습·학습 노트북 안내는 최초 준비 당시 이력이다.

지금 실행할 파일:

1. [평가 입력 ZIP — 약 469MB](bundles/P10/v1_4_p10_update_evaluation_r1.zip), [checksum](bundles/P10/v1_4_p10_update_evaluation_r1.sha256.json)
2. [1/2 — T4 cache·latent 준비 노트북](notebooks/P10/P10_evaluation_1_T4_features_r1.ipynb)
3. [2/2 — CPU probe·전이·통계 노트북](notebooks/P10/P10_evaluation_2_CPU_probes_r1.ipynb)

ZIP을 내 드라이브 `boolean_interp_v1_4/`에 올린다. 기존 `P10_update_l3_r1` cache/inputs를 유지한다.
새 출력은 `P10_update_evaluation_r1`이다. T4 노트북의 `features_complete.json` 생성 후 **GPU 연결을 종료하고 CPU 런타임(하드웨어 가속기 없음)**에서 두 번째 노트북을 실행한다.
GPU 준비 직후의 중간 ZIP은 보관해도 되며, 최종적으로 CPU 평가의 마지막 셀에서 반환 ZIP과 checksum JSON을 전달한다.

1,248개 fitting과 2,016개 의미 보고는 과제별로 저장한다. PAUSED이면 같은 셀을 반복하고, 런타임이 끊기면 해당 노트북의 위부터 실행한다.
각 지표와 paired full-probe 차이에 sequence cluster CI를 기록한다. 초기 LM·token/position·shuffled·전체/128후보 대조를 포함한다.
평가 규칙은 [동결 계약](p10_eval_r1/contract.json), 로컬 검증은 [평가 준비 보고서](results/p10_evaluation_preparation_r1/REPORT.md)를 따른다.

반환 결과·원본 cache 재현 감사 이후에만 P10을 완료 처리한다. 전체 실험과 P11도 미완료다.


## 2026-10-04 평가 반환 감사·원본 재현 감사 실행

[평가 반환 로컬 감사](results/p10_return_audit_20261004_01/REPORT.md)가 통과했다. 1,248개 fit·2,016개 예측 보고·3,840개 paired 차이를 확인했다. 시간 제한 정지 두 번과 선택 완료 전 평가 호출의 gate 오류 두 번은 보존했으며 최종 재개로 모두 해결됐다. 앞선 평가 0/12 안내는 당시 이력이다. 학습과 본평가를 다시 실행하지 않는다.

현재 실행 순서:

1. [추가 감사 ZIP — 약 303KB](bundles/P10/v1_4_p10_source_audit_r1.zip)과 [checksum](bundles/P10/v1_4_p10_source_audit_r1.sha256.json)을 준비한다. ZIP을 Drive `boolean_interp_v1_4/`에 올린다. 같은 폴더의 기존 `v1_4_p10_update_evaluation_r1.zip`도 필요하다.
2. [1/2 — T4 원본 재현 감사](notebooks/P10/P10_source_audit_1_T4_r1.ipynb)를 실행한다. `gpu_complete.json` 생성 후 GPU 연결을 종료한다.
3. [2/2 — CPU refit·예측·CI 감사](notebooks/P10/P10_source_audit_2_CPU_r1.ipynb)를 하드웨어 가속기 없음으로 실행한다.
4. 마지막 셀에서 감사 반환 ZIP과 checksum JSON을 내려받아 전달한다.

Drive `P10_update_evaluation_r1`의 cache·arrays·latents·fit·prediction 원본 전체를 유지한다. 새 감사 출력은 `P10_update_source_audit_r1`에 저장한다. 기본 2시간 제한에서 PAUSED이면 같은 실행 셀을 반복한다. 런타임이 끊기면 해당 노트북의 위부터 실행하며 완료된 감사 기록부터 재개한다.

이전 CPU fitting의 과제 시간 합은 약 4시간 19분, 의미 평가는 약 29분이었다. 원본 refit 감사도 수 시간이 걸릴 수 있으며 실제 경과 시간·CU 소비를 보장하지 않는다.

[준비 검증](results/p10_source_audit_preparation_r1/REPORT.md): 독립 번들 테스트 7개와 노트북 2개 검증 통과. 실제 GPU replay·원본 refit/예측/fidelity/CI 감사는 미실행이다. 이 반환까지 검증한 뒤 P10 완료·Git commit/push를 수행한다. P11과 전체 실험도 미완료다.


## 2026-10-04 원본 감사 중간 반환 — CPU 재개 필요

[중간 반환 검증](results/p10_source_partial_audit_20261004_01/REPORT.md)을 통과했다. GPU 전체 replay 240 chunks·480,000 visits와 dictionary 12개 감사 완료이며 activation/latent 반환 최대 오차는 0이다. CPU는 496/1,248개 task·752개 semantic reports를 완료하고 기본 2시간 제한으로 정상 일시정지했다. 남은 task는 752개다.

T4를 다시 실행하거나 파일을 삭제하지 않는다. 기존 [CPU 감사 노트북](notebooks/P10/P10_source_audit_2_CPU_r1.ipynb)의 감사 실행 셀을 반복한다. 새 런타임이면 CPU 노트북 맨 위부터 실행하며 Drive `P10_update_source_audit_r1`의 완료 기록을 재사용한다. `source_audit_complete.json` 생성과 완료 메시지를 확인한 뒤 새 반환 ZIP/checksum을 전달한다. P10과 전체 실험은 미완료다.


## 2026-10-04 두 번째 CPU 중간 반환 — 215개 남음

[반환 검증](results/p10_source_partial_audit_20261004_02/REPORT.md)을 통과했다. ZIP/checksum 번호는 다르지만 실제 hash와 크기가 일치한다. CPU 1,033/1,248개 task·1,586개 reports가 완료됐으며 시간 제한으로 정상 일시정지했다. 남은 215개는 기존 CPU 감사 실행 셀을 반복하여 이어간다. Drive 결과를 유지하며 T4 재실행은 필요 없다. 전체 완료 파일은 아직 없어 P10은 미완료다.


## 2026-10-04 P10 완료

Update block 3 SAE/TC 12개 학습·평가·원본 재현 반환 감사를 통과했다. GPU 480,000 position visits·960,000 latent positions와 CPU 1,248 refits·2,016 예측/CI 보고를 확인했다. [최종 감사](results/p10_final_audit_20261004_01/REPORT.md), [완료 판정](results/p10_final_audit_20261004_01/completion.json)을 따른다. 앞선 재개 대기 안내는 당시 이력이며 추가 Colab 실행은 필요 없다. 다음은 P11 최종 집계다. 전체 실험은 아직 미완료다.
