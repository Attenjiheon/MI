# P9 원본 재현 감사 준비 검증

2026-10-01. 준비 검증 통과, 실제 CUDA 원본 재현 감사 0/16. P9 미완료.

원래 평가 번들의 332개 member를 새 번들과 byte 단위로 대조해 모두 일치했다. 동결된 평가 코드·선택·계약을 변경하지 않고 별도 감사 코드와 반환 manifest를 추가했다. 독립 압축 해제 환경의 pytest 24개, 로컬 감사 단위 테스트 5개, 노트북 8개 셀의 schema·Python compile 검증을 통과했다.

범위: 신규 probe 560 refits, 재사용 포함 1,200 tasks, 16개 모델 각각 80,000 위치 fidelity, 전체 validation bins와 test 예측·의미/인과/대체 CI, 층간 대비 및 seed 비교. GPU replay는 효과와 무관한 고정 corpus 순서의 suite별 첫 pair 4개와 READ 대체 첫 16개/run이다. 모든 원시 인과 pair는 별도 로컬 반환 감사에서 확인했다.

실행: `P9_source_return_audit_r1.ipynb`, 입력 `v1_4_p9_source_audit_r1.zip`. Drive의 기존 `P5_r2`와 `P9_evaluation_r1`을 읽고 `P9_source_audit_r1`에 별도 저장한다. T4 런타임에서 실행 후 반환 ZIP·checksum을 검증해야 완료할 수 있다. 정확한 hash는 `verification.json`을 따른다.
