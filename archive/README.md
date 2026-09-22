# 과거 실험 보관소

v1.0–v1.3의 실험·구현·검증·데이터 실체를 `legacy/`에 모았다.
버전별 폴더명과 내용은 유지하며 루트의 옛 경로는 상대 symlink다.

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

공용 코드와 v1.4를 가리키는 archive 내부 symlink는 상대 링크 호환용이다.
새 실행은 저장소 루트에서 수행하고 과거 bundle을 현재 문서로 재생성해 원본이라고 주장하지 않는다.
Symlink를 지원하는 Git checkout을 사용한다. 결과 재검증 시 기존 디스크 정리 삭제 목록도 확인한다.
이동·보존 증빙은 [정리 보고서](../maintenance/repository_cleanup_20260922/REPORT.md)에 있다.
