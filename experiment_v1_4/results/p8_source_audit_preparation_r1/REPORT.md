# P8 원본 재현 감사 준비

**로컬 검증 통과. 실제 원본 재현 감사는 Colab 실행 대기다.**

P8 평가 반환 ZIP과 동결 평가 계약을 연결한 감사 전용 계약을 만들었다. 기존 평가 코드·선택 checkpoint·feature·threshold·k·test 절차는 변경하지 않는다.

- P5 원본 cache와 P8 반환 파일 전체 checksum을 대조한다.
- 새 probe의 모든 고유 train/validation refit를 기존 선택과 비교한다. 재계산 결과로 기존 선택을 덮어쓰지 않는다.
- 모든 test 예측·의미 CI, 전체 train/val/test fidelity 및 dead 비율을 검산한다.
- 전체 causal validation norm과 bin을 재현한다.
- 각 run에서 고정 corpus 순서의 suite별 첫 causal pair 4개와 첫 replacement 16개를 GPU로 재실행한다. 효과에 따라 표본을 고르지 않으며 전체 개입을 재실행했다고 주장하지 않는다.
- 모든 causal/replacement summary CI와 층간 paired contrast 및 비교 표를 재계산한다.

독립 압축해제 번들에서 13 tests, 모든 file hash, 노트북 schema·셀 문법을 검증했다.
TC u 인코딩과 m target 정규화를 실제 감사 함수 경로에서 합성 데이터로 검증했다.

[노트북](../../notebooks/P8/P8_source_return_audit_r1.ipynb), [입력 ZIP](../../bundles/P8/v1_4_p8_source_audit_r1.zip)을 사용한다.
ZIP을 Drive `boolean_interp_v1_4/`에 올린 뒤 T4 GPU에서 순서대로 실행한다.
기존 `P5_r2`, `P8_evaluation_r1`을 읽고 새 `P8_source_audit_r1`에만 결과를 쓴다.
각 refit/완료 run에서 재개할 수 있다. 끝나거나 중단되면 마지막 셀로 감사 ZIP과 checksum JSON을 내려받아 반환한다.

P8 완료는 로컬 반환 감사 및 이 원본 재현 감사의 실제 반환 확인 후 판정한다.
