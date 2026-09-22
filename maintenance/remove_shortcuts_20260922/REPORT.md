# 바로가기 제거

2026-09-22 사용자 요청에 따라 앞선 정리에서 만든 symlink 37개를 모두 제거했다.
과거 실험·코드·데이터의 실제 파일은 `archive/legacy/`에 그대로 보존했다.
제거 경로·target 및 소스 변경 전후 SHA-256은 [changes.json](changes.json)에 있다.

v1.4의 공용 모듈 import와 debug/proposal/번들 입력 경로를 실제 archive 경로로 바꿨다.
변경한 소스 14개의 원문은 `source_before/`에 보존했다. 모델·학습·평가 수식과
동결 config·corpus·checkpoint·반환 증빙은 변경하지 않았다. 소스 hash는 달라지므로
기존 GPU smoke를 새 소스 승인으로 쓰지 않으며 이후 GPU 작업 전에 새 검증이 필요하다.
기존 완료된 P4 학습·gate/test는 재실행하지 않는다.

루트 README·AGENTS와 archive 안내를 현재 구조에 맞췄다. 동결 과거 문서의 옛 경로는
당시 기록으로 보존하며, 실제 위치 조회에는 최초 migration.json 매핑을 사용한다.
과거 실행 환경을 복원할 때는 보존된 번들을 별도 디렉터리에서 사용한다.

현재 구조 검증 도구는 archive의 실제 경로와 제거된 링크의 부재를 확인한다.
전체 보존 검증은 과거 파일 경로를 이동 매핑으로 해석하며, 이번에 수정한 코드의
변경 전 bytes는 source_before와 대조하고 현재 코드도 after hash와 대조한다.

검증 이력: 첫 실행은 기존 재현 허가 테스트 1개가 새 source hash를 거부해 39 passed / 1 failed였다.
보호장치는 유지했다. 테스트의 과거 승인 사례는 보존된 원문을 임시 경로에 실제 파일로 복원해 검증하고,
현재 변경 소스를 기존 승인으로 실행할 수 없다는 검사를 추가했다.

최종 검증:

- `/opt/anaconda3/bin/python -m pytest tests_v1_4 -q`: **41 passed**, 25.69초.
- `python3 scripts/verify_repository_layout.py --full`: **passed**, 제거한 37개 링크 부재,
  15개 실제 archive 경로, 1,953개 기존 파일의 보존 bytes 및 수정 소스 전후 hash 확인.
  [전체 결과](verification_final.json).
- `/opt/anaconda3/bin/python -m interp_v1_4.cli --help`: 정상 종료.
- `git diff --check`: 통과.
