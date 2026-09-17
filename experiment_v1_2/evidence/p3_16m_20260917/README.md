# v1.2 반환 증빙 요약 사본

이 폴더는 반환 ZIP의 JSON·환경·그림을 그대로 복사한 열람용 부분집합이다. `LATEST.json`이 참조하는 전체 indices, events 및 161개 checkpoint는 상위 `p3_16m_verified_contents_20260917.zip`에 모두 있다. 복구·검증은 전체 ZIP을 새 디렉터리에 풀고 수행한다.

원본 ZIP 최초 SHA는 `archive_inventory.json`, 재압축 전체 파일별 SHA와 이유는 `../../results/p3_repacked_archive.json`에 있다. 원래 입력은 `../../bundles/p3_16m_bundle_v1.zip`이다. 원본과 재압축 ZIP을 동일한 컨테이너로 표현하지 않는다.

`nvidia-smi.txt` 등 원본 도구 출력의 trailing whitespace도 변경하지 않고 보존한다. `.pt` debug 파일을 본실험 checkpoint로 사용하지 않는다. 실제 판정은 `../../results/p3_stop_report.md`의 **failed**다.
