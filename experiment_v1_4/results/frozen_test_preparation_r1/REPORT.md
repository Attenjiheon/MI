# v1.4 frozen test 전달 준비

2026-09-21. **로컬 준비 통과. 실제 GPU test 점수는 미관측이며 P4는 진행 중이다.**

세 seed의 학습·validation 결정을 고정한
`../p4_audit_20260921_01/validation_freeze.json`의 SHA256은
`9772cc4f4ae0467ce0343a1aa74cbcf70c2511cf562d5e9471649ab574f33b30`이다.
Seed 0/1의 update 7,983과 seed 2의 update 8,399 checkpoint를 원본 반환 ZIP에서
byte-identical하게 추출했다. 각 원본 ZIP 전체 해시와 선택 checkpoint 해시를 확인했고,
checkpoint payload의 seed별 hash 묶음·선택 update·microbatch 16·non-debug 상태를 확인했다.

## 사전 고정한 평가 범위

각 seed에서 아래 7 suite를 실행한다. 총 21개 평가이며 length는 P10에 남긴다.

| Suite | Sequence 수 | READ target 수 |
|---|---:|---:|
| General | 2,048 | 32,226 |
| Other variable | 2,048 | 2,048 |
| Repeated update | 2,048 | 2,048 |
| First READ after SET | 2,048 | 2,048 |
| First/repeat | 10,752 (5,376쌍) | 10,752 |
| Composition 0 | 1,024 | 1,024 |
| Composition 1 | 1,024 | 1,024 |

합계는 seed당 20,992 sequences / 51,170 READ targets다.
Test metadata에서 target ID·정답·표본 수·최대 길이만 확인했으며 모델 점수를 계산하지 않았다.
기존 full-vocabulary answer CE/accuracy, binary accuracy, bit mass, strata/coverage,
규칙 기준선과 all-token CE를 저장한다. First/repeat는 기존 paired within-cell bootstrap을
유지하고, 나머지는 사전 고정한 1,000회 sequence-cluster bootstrap CI를 추가한다.
모든 seed를 개별 보고하며 test로 checkpoint나 gate 판정을 바꾸지 않는다.

## 검증

- 6 tests 통과: 완료 결과의 추론 재호출 방지, binding 변경 거부, 원시 결과로 집계 재개,
  미저장 중단의 자동 재추론 거부, raw 변조 거부, target/cluster 검사 및 기존 pair 평가기와의 예측 일치.
- CPU preflight 통과: 미학습 모델로 일반 debug 80 targets, first/repeat 84 targets,
  microbatch 16 × 길이 302 인공 토큰 fixture의 유한 출력 확인.
- Test metadata 최대 길이는 201 tokens 이하로 위 preflight 범위 안이다.
- 기존 모델·행동·보고·runtime·corpus 코드는 seed 1 반환의 동결 code hash와 모두 일치한다.
- ZIP 내부 74개 파일 해시 전수검증, 노트북 5개 코드 셀 compile 및 nbformat schema 검증 통과.
- 원시 결과 저장 전 중단은 실패 증빙으로 보존한다. 완료 unit은 재사용하고 raw가 있으면
  모델 추론 없이 집계만 재개한다. 런타임 환경이 달라진 기존 unit을 자동 덮어쓰지 않는다.

실제 CUDA preflight·frozen test·반환 감사는 Colab 실행 후에 확인한다.
본 준비를 P4 완료나 P5 진입으로 해석하지 않는다.

## 전달

- [노트북](../../notebooks/P4_v1_4_frozen_test_r1.ipynb)
- [입력 ZIP](../../bundles/v1_4_frozen_test_bundle_r1.zip): 364,510,740 bytes (약 348 MiB)
- [준비 manifest](verification.json), [test 로그](tests.log), [CPU preflight](cpu/preflight.json),
  [metadata 확인](metadata_audit.json), [notebook 검증](notebook_validation.json)

ZIP SHA256: `80a4d30f607e7365bf3aed8bc22b4c8f43200aba6fb822e6dfdc22f27d4780a1`.
