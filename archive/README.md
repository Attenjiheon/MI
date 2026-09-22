# 과거 실험 보관소

v1.0–v1.3의 실험·구현·검증·데이터 실체를 `legacy/`에 모았다.
버전별 폴더명과 내용은 유지한다. 루트와 archive 내부의 바로가기는 모두 제거했다.

| 버전 | 실험 | 구현 | 테스트 | 데이터 |
|---|---|---|---|---|
| v1.0 | [experiment_v1](legacy/experiment_v1/) | [interp](legacy/interp/) | [tests](legacy/tests/) | [language_v1](legacy/data/language_v1/) |
| v1.1 | [experiment_v1_1](legacy/experiment_v1_1/) | [interp_v1_1](legacy/interp_v1_1/) | [tests_v1_1](legacy/tests_v1_1/) | v1.0과 공유 |
| v1.2 | [experiment_v1_2](legacy/experiment_v1_2/) | [interp_v1_2](legacy/interp_v1_2/) | [tests_v1_2](legacy/tests_v1_2/) | [language_v1_2](legacy/data/language_v1_2/) |
| v1.3 | [experiment_v1_3](legacy/experiment_v1_3/) | [interp_v1_3](legacy/interp_v1_3/) | [tests_v1_3](legacy/tests_v1_3/) | [language_v1_3](legacy/data/language_v1_3/) |

[통합 전 루트 문서](specifications/pre_v1_4_integration/README.md)는 원본 bytes 그대로다.
기존 design manifest의 `source_documents` hash를 확인할 때 이 snapshot을 사용한다.
과거 문서의 미실행·실패·경로 이동 금지 문구는 당시 기록이며 현재 작업 지시는
[루트 AGENTS](../AGENTS.md)를 따른다. Archive를 수정해 현재 실험 문서를 갱신하지 않는다.

과거 원문의 상대 링크·manifest 경로는 당시 식별자로 보존되어 현재 경로와 다를 수 있다.
실제 경로는 `maintenance/repository_cleanup_20260922/migration.json`의 매핑을 따른다.
과거 결과 재현은 해당 버전의 보존된 입력 번들과 코드로 별도 작업 디렉터리에서 수행한다.
동결 원문을 링크 수정 목적으로 다시 쓰지 않는다.

현재 구조는 [바로가기 제거 기록](../maintenance/remove_shortcuts_20260922/REPORT.md),
최초 이동·보존 증빙은 [정리 보고서](../maintenance/repository_cleanup_20260922/REPORT.md)에 있다.
