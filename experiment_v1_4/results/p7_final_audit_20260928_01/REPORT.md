# P7 최종 반환 감사 및 결과

**P7 완료. 24개 READ SAE 평가와 원본 재현 감사 통과. 다음은 P8 READ Transcoder다.**

2026-09-28. 범위: LM seed 0·1·2 × block 0·3·7·11 × k=4/16, sparse seed 0.

## 감사 근거

- 최초 평가 반환: 56,919개 파일 hash, 1,680개 probe 선택, 3,120개 의미 보고, 49,152개 대체 target, 49,152개 인과 pair 평가 및 4,061,712개 대조군 행의 로컬 검산 통과.
- 추가 원본 감사: 24/24 runs, 고유 refit 840개(중복 사용 포함 신규 fit task 1,200개), 전체 test 예측·의미 CI 및 1,920,000 position evaluations의 fidelity/dead 통계 재현.
- Causal validation bin 48개, 전체 causal/replacement CI와 paired layer contrast 재현. GPU 재실행은 사전에 정한 causal pair 96개와 대체 target 384개다. 전체 인과 실험을 두 번 실행했다는 뜻은 아니다.
- 반환 내부 872개 파일 hash, 감사/원본 계약, GPU 환경 및 refit 실제 계수·feature/lambda/threshold를 로컬에서 다시 대조했다. 실패 기록과 누락 run이 없다.
- 반환 ZIP의 별도 외부 checksum 파일은 제공되지 않았다. 수신 ZIP의 SHA-256을 계산하고 ZIP 내부 CRC/manifest hash 및 사전 동결 계약과 대조했다.
- T4 / torch 2.11.0+cu128 / 환경 `f8912b7ab0ca620d`. Run별 추가 검산 시간 합계 약 3.35시간이며 초기 검증·설치·최종 집계 시간은 별도다.

## READ 현재 값 접근성

아래는 전체 후보, ≤4 feature probe의 test balanced accuracy, LM 세 seed 평균이다.

| Block | SAE k=4 | SAE k=16 | 원래 좌표 ≤4 | Random 방향 ≤4 | Full h probe |
|---|---:|---:|---:|---:|---:|
| 0 | 54.70% | 59.01% | 56.48% | 57.44% | 61.40% |
| 3 | 76.29% | 86.31% | 92.80% | 92.95% | 93.81% |
| 7 | 91.87% | 98.41% | 99.76% | 99.77% | 99.82% |
| 11 | 94.99% | 99.92% | 99.95% | 99.93% | 99.95% |

후반 층에서 현재 값의 사후 감독 접근성이 높으며 k=16은 k=4보다 높다. 원래 좌표와 random 방향도 강한 기준선이므로, 이 표만으로 SAE 고유의 우월성을 주장하지 않는다. 두 k 모두 유지하며 결과로 선택하지 않았다.

## 해석 범위

선택 latent 패칭의 효과는 해당 layer·READ 위치의 지정 개입 증거다. Full donor나 좌표 대조와 함께, matching 성공 subset의 selected/random 효과 및 coverage를 구분해 해석한다. 층은 독립 모델 반복이 아니며 세 LM seed의 분포를 함께 보고한다. TC 및 sparse seed 1 반복은 아직 미실행이므로 전체 실험 완료가 아니다.

원시 값·seed별 범위는 [run별 결과](results_by_run.csv), [층별 평균/최소/최대](results_by_layer.json), 원래 모든 CI·128후보·전이·대조군은 [검증된 결과 표](../p7_return_audit_20260927_01/returned/tables/)를 따른다.

## 증빙

- [완료 판정](completion.json) · [최종 검증](verification.json) · [재현 스크립트](verify_return.py)
- [원본 감사 반환](returned/source_audit_complete.json) · [이전 전체 원시 검산](../p7_return_audit_20260927_01/verification.json)
- Git에는 ZIP 원본을 LFS로 보존한다. 중복인 causal/replacement/semantic prediction 압축 해제 payload는 로컬에 유지하며, 필요하면 원래 반환 ZIP을 `p7_return_audit_20260927_01/returned/` 아래로 복원한다.
- 완료 커밋과 원격 main 일치는 Git 기록으로 확인한다.
