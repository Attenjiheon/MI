# P10 평가 반환 로컬 감사 — 2026-10-04

판정: **passed_local_source_reproduction_pending**. 평가 12/12 반환, 원본 재현 감사 대기이며 P10은 미완료다.

두 반환 ZIP의 외부 SHA-256과 각각 29·6,583개 manifest 파일 hash를 검증했다. 첫 반환의 29개 파일은 최종 반환과 동일하다. 동결 계약·선택·원본 식별자, 토큰 replay에 따른 split별 라벨, 1,248개 fit의 support·선택 trace·후보·계수, 2,016개 예측 배열의 확률과 BA/macro F1/AUROC/혼동행렬을 검산했다. Paired 차이 3,840개의 점 추정값과 CI seed·반복 수·범위를 확인했다. 원본 cache를 이용한 refit·fidelity·전체 bootstrap CI 재현은 아직 수행하지 않았다.

최종 fitting 1,248개는 모두 passed다. 두 번의 시간 제한 정지와 선택 완료 전 evaluate 호출 두 번은 로그에 보존했다. 후자는 selection gate에서 중단되었으며 최종 재개에서 선택·평가가 완료됐다. 환경은 GPU `6e859d41cc64e591`, CPU `9ee13b7194181dfb`이며 6개 세션을 기록했다.

과제별 계산시간 합은 fitting 15,544.42초, 의미 평가 1,746.37초, fidelity 65.75초다. 전체 경과 시간이나 CU 소비량으로 해석하지 않는다.

증빙: [status.json](status.json), [return manifest](return_manifest.json), [운영 이력](operational_events.json). 원본 예측은 반환 ZIP에 보존하고 나머지 감사용 metadata는 `returned_metadata/`에 추출했다.

검증 명령: `/opt/anaconda3/bin/python scripts/audit_v1_4_p10_evaluation_return.py`.
다음은 [원본 재현 감사 준비](../p10_source_audit_preparation_r1/REPORT.md)다. Update 인과 개입은 범위에 없으며 feature의 인과적 사용을 주장하지 않는다.
