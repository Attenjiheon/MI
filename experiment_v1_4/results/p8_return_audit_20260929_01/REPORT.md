# P8 TC 평가 반환 로컬 감사 — 2026-09-29

**24/24 평가가 반환되었고 로컬 수치 검사를 통과했다. 원본 재현 감사가 남아 P8 완료 판정은 보류한다.**

## 확인된 증빙

- 외부 checksum과 ZIP SHA-256 일치: `e17b8e2f9703772751a0dc952c0b5a518f6548d3229531f0367124803f9e4b03`.
- 반환 내부 56,950개 파일 hash와 동결 평가 계약을 확인했다. 실패 기록 없음.
- 선택 task 1,920개: validation trace winner, ANOVA prefix, lambda/threshold 선택, P5 기준선 계수 및 고정 128후보·projection seed 확인.
- 원시 test 예측에서 balanced accuracy/F1/AUROC/confusion matrix를 검산했다.
- 한 위치 대체 49,152 targets의 raw logits/CE 및 answer/READ ID를 대조했다.
- 인과 49,152 pair units, 3,667,856개 대조군 행의 logits 지표·후보 RNG·matching을 확인했다.
- both-correct/original-correct subset과 coverage 및 origin/condition 집계값 57,024개를 독립 검산했다.
- Fidelity의 position 수, activation rate, inactive 비율 및 L0 내부 일치 검사를 통과했다. 전체 원본 activation으로 다시 계산한 fidelity는 아직 아니다.
- CUDA T4, torch 2.11.0+cu128, 환경 `f8912b7ab0ca620d` 및 네 층·두 k smoke 확인.

확장 크기 8.56GB인 ZIP은 전체를 풀지 않고 hash를 streaming으로 확인한 뒤 한 run씩 임시 추출해 검사했다.
원본 ZIP과 기존 평가 결과는 변경하지 않았다. [검증 JSON](verification.json), [선택 identity 검증](selection_identity_verification.json), [원본 manifest](returned_metadata/return_manifest.json)을 따른다.
로컬 검산 가속 과정의 변수명 충돌 1회는 `local_tooling_attempt.log`에 남겼고 수정 후 해당 run을 처음부터 재검사했다. 실험 실패나 결과 변경이 아니다.
일부 중첩 metadata의 `p7_complete: false`는 P7 검증기에서 계승한 필드이며 단계 판정으로 사용하지 않는다. 최상위 `p8_complete: false`가 현재 상태다.

## 아직 필요한 확인

P5 원본 cache에 대한 모든 새 probe refit, 전체 예측·의미 CI, 전체 train/val/test fidelity 및 train dead 비율,
전체 validation bin 재현, 모든 causal/replacement CI·층간 paired contrast 재현과 고정 GPU replay 표본을 확인해야 한다.
P8을 완료하거나 P9로 진입하지 않았고 완료 Git commit/main push도 수행하지 않았다.

## 다음 실행

[원본 재현 감사 노트북](../../notebooks/P8/P8_source_return_audit_r1.ipynb)과
[감사 입력 ZIP](../../bundles/P8/v1_4_p8_source_audit_r1.zip)을 사용한다.
ZIP을 Drive `boolean_interp_v1_4/`에 업로드하고 T4 GPU에서 위부터 실행한다.
기존 `P5_r2`, `P8_evaluation_r1`은 읽기만 하며 새 출력은 `P8_source_audit_r1`이다.
모델 재학습이나 기존 feature/checkpoint 재선택은 없다. 검산 ZIP과 checksum JSON의 실제 반환 확인 뒤 P8 완료 여부를 판정한다.
