# P7 원본 재현 감사 r2 준비

2026-09-27. **노트북·번들·로컬 테스트 준비 통과. 실제 Colab 감사는 아직 미실행.**

원래 P7 계약과 선택·평가 결과를 변경하지 않는다. P5 원본 cache를 확인하고 전체 새로운 probe refit,
전체 test 예측·의미 지표/CI, train/val/test fidelity·dead 비율, 전체 causal validation bin을 검산한다.
패칭은 각 run의 고정 corpus 순서 suite별 첫 pair 4개와 대체 target 첫 16개를 재현한다.
전체 raw 결과의 checksum·logit 지표·후보 RNG는 로컬 반환 검사에서 별도로 확인한다.
전체 causal/replacement CI와 층간 차이도 다시 계산하고 원래 표와 대조한다.
검산용 선택 계산을 새 실험 선택으로 적용하지 않으며 불일치 시 중단한다.

- [노트북](../../notebooks/P7/P7_source_return_audit_Drive_r2.ipynb)
- [입력 ZIP](../../bundles/P7/v1_4_p7_source_audit_r2.zip), 약 416MB
- [감사 계약](../../p7_audit_r2/contract.json)
- [준비 검증](verification.json)

T4 GPU 런타임에서 노트북을 위부터 실행한다. 첫 셀에 위 ZIP을 업로드한다.
기존 Drive `boolean_interp_v1_4/P5_r2`와 `P7_r1`을 읽고 `P7_source_audit_r2`에 새 검산 증빙을 저장한다.
CPU refit와 bootstrap 때문에 수 시간이 걸릴 수 있다. 같은 노트북으로 완료 refit/run부터 재개한다.
마지막 셀에서 검산 반환 ZIP·checksum JSON을 다운로드한다. 실패한 경우에도 마지막 셀로 로그를 반환한다.
독립 추출 번들의 기존 입력 hash 582개, 테스트 11개, 노트북 schema·셀 문법 검사를 통과했다.
본평가·재학습·test 기반 재선택은 수행하지 않았다. 실제 검산 반환 확인 전 P7 완료를 선언하지 않는다.

## Drive 번들 연결

2026-09-27 사용자 요청으로 첫 셀을 Drive 마운트 후 ZIP 연결 방식으로 변경했다.
기존 `v1_4_p7_source_audit_r2.zip`을 `내 드라이브/boolean_interp_v1_4/`에 올려 둔다.
다른 위치라면 첫 셀의 `BUNDLE_IN_DRIVE`를 수정한다. Colab 로컬로 복사하며 checksum·진행률을 확인하고,
같은 런타임에서는 checksum이 맞는 로컬 ZIP을 재사용한다. 원래 감사 계약·입력 ZIP과 이후 실행 셀은 동일하다.
