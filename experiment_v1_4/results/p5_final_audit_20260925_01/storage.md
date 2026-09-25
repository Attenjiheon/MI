# P5 저장 범위

활성 r2 실행 번들, 독립 감사 번들과 원본 init checkpoint는 Git LFS로 보존한다.
대체된 r1 416MB ZIP은 기존 위치에 그대로 두고 Git에서는 제외한다. r1 계약·checksum·대체 사유는 추적한다.
중간 snapshot과 최종 probe metadata, 독립 감사 반환 ZIP/receipt, 코드·노트북·검증 기록은 보존한다.
원본 production cache 약 17.7GB는 Drive `boolean_interp_v1_4/P5_r2`에 있고, Git에는 파일별 hash/위치/검증 증빙을 보존한다.
