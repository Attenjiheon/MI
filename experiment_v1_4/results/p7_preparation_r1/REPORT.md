# P7 실행 준비 검증

2026-09-26. **준비 검증 통과. P7 본평가 0/24, GPU 및 반환 감사 대기.**

P6 완료 증빙의 파일별 checksum과 선택 SAE 24개·동결 LM 3개를 검증했다.
582개 입력/코드/계수/감사 파일의 hash를 실행 계약에 고정했다.
P5 기준선 원본 계수와 독립 감사 source hash를 대조했다.
층·표현별 projection/subset RNG 및 pair별 latent/direction RNG 정수를 평가 전에 저장했다.

기존 v1.4 테스트 85개 통과 후 interrupted-array 재개 회귀 테스트를 추가했다.
최종 P7 테스트 9개가 로컬과 독립 추출 번들에서 모두 통과했다.
네 층 × k=4/16, 8개 설정에서 memory/composition 및 changed/unchanged 양방향 대조군,
identity logits·padding 개별/배치 일치를 CPU debug 모델로 검사했다.
노트북 schema·모든 Python 셀 문법과 번들 독립 import·582개 hash를 검증했다.
이 smoke는 실제 CUDA 실행을 대신하지 않는다.

## 실행

[노트북](../../notebooks/P7/P7_colab_read_sae_evaluation_r1.ipynb)을 Colab GPU 런타임에서 열고
[입력 ZIP](../../bundles/P7/v1_4_p7_bundle_r1.zip)을 첫 셀에서 업로드한다.
Drive의 `boolean_interp_v1_4/P5_r2` 원본 cache를 읽고 `P7_r1`에 새 결과를 저장한다.
선택 → 의미/fidelity → causal validation bins → 대체/인과 test → 집계 순서를 지킨다.
중단 시 같은 노트북으로 재개하고, 마지막 셀의 반환 ZIP·checksum JSON을 전달한다.
설치 후 Colab 런타임 재시작이 필요하면 업로드 셀부터 다시 실행한다.

## 남은 검증

실제 GPU smoke·본평가·전체 반환 감사가 남았다. 원시 결과 검산 스크립트는
표본 수·feature/hash·후보 RNG·bin membership·probe 점수·logits의 CE/인과 점수를 검사한다.
전체 fitting·fidelity·CI 및 원래 activation에서의 패칭 재현은 cache 원본을 사용한 반환 감사 대상이다.
노트북/준비 통과로 P7 완료나 P8 진입을 선언하지 않는다.
P7 완료 조건을 충족한 뒤 완료 커밋·원격 main push·SHA 확인을 수행한다.
