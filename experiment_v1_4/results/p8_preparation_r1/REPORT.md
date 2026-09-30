# P8 첫 학습 전달물 검증 — 2026-09-29

**학습 준비 검증 통과. P8 미완료, 본학습 0/24, 본평가 0/24.**

P5/P6/P7 완료 증빙에 연결된 실제 파일 hash와 P6의 36개 scalar 통계를 확인했다.
P5 원본 cache는 Drive에서 재검증해야 하므로 로컬에서 240개 원본을 다시 확인했다고 주장하지 않는다.

## 이번 전달물

- TC 24 runs: u→m, 독립 입출력 통계 및 encoder/decoder 초기화.
- 대응 SAE와 동일한 draw key/seed, 5,000 updates, 512 positions, 매 250 updates 전체 validation MSE.
- 100-update benchmark, checksum checkpoint, optimizer/RNG 복구, 실패 기록, 별도 환경 ID.
- 독립 검산은 504 checkpoint와 480 validation MSE 및 선택을 검사한다. F.linear/float32 tensor scale/stable TopK를 사용한다.
- 향후 평가 규칙을 동결했다. full u/m, m 좌표/random, 전체/128후보, P7 공유 READ target·인과 quota·matching/CI를 유지한다.

## 로컬 검증

15 tests 통과, 별도 압축해제 폴더의 15 tests 통과. 노트북 schema 및 모든 코드 셀 compile 통과.
네 층의 hook·정규화·패칭·debug 학습과 TC 두 k의 bitwise model/optimizer/sampler 재개를 확인했다.
CPU smoke 환경: `92a9339c663b9cd7`, 소요 31.19초.
본학습 CUDA 증빙은 아직 없다. [검증 JSON](verification.json), [smoke 로그](cpu_smoke.log)을 따른다.

## 실행 파일

- [노트북](../../notebooks/P8/P8_colab_read_tc_training_r1.ipynb)
- [입력 ZIP](../../bundles/P8/v1_4_p8_training_r1.zip), 929,531 bytes
- SHA-256: `791c41d2bb5133455f845dd3b18a1d1799af95cdf3042146fae71fa584d43bd9`
- [학습 계약](../../p8_r1/contract.json), [평가 규칙](../../p8_r1/evaluation_rules.json)

ZIP을 Drive `boolean_interp_v1_4/`에 업로드한 뒤 GPU Colab에서 노트북을 순서대로 실행한다.
입력 `P5_r2`, 출력 `P8_training_r1`이다. 결과 ZIP과 checksum JSON을 반환받아 학습을 감사하고
실제 선택 checkpoint hash를 고정한 평가 전달물로 이어간다. 현재 전달물에 전체 의미·인과 평가 실행기는 포함되지 않는다.

## 보존과 상태

기존 P6/P7 코드·동결 계약·결과를 변경하지 않았다. 시작 시 존재한 용량 정리 관련 변경·삭제는 유지했다.
Phase 완료 조건이 충족되지 않아 완료 checkbox·완료 commit/main push를 수행하지 않았다.
