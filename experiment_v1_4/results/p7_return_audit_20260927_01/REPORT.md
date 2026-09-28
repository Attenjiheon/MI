# P7 반환 감사 — 2026-09-27

**로컬 반환 검사 통과. GPU 평가 결과 24/24 확인. 원본 cache 재현 감사 대기, P7 완료 판정 보류.**

## 확인한 증빙

- 반환 ZIP SHA-256 `2d8d65b3bca7b6a0ec09d6f524d78c3495239ee2cf4a96fbc59a5187bf943449`, 1,748,886,790 bytes.
- Manifest 56,919개 파일의 hash·ZIP member 집합·안전한 상대경로·원래 계약 및 입력 identity 대조 통과.
- Tesla T4, torch 2.11.0+cu128, 환경 `f8912b7ab0ca620d`. 네 층 × 두 k GPU smoke와 24개 run의 단계 기록 확인.
- 1,680개 probe 선택 작업에서 수렴·validation tie-break·누적 prefix·후보 수·재사용 P5 계수 확인.
- 3,120개 의미 보고의 point metric, 49,152개 대체 target의 logits/CE, 49,152개 인과 pair 평가의 원시 logits·margin·flip·후보 RNG·bin membership 검산 통과.
- 인과 결과 4,061,712행에서 coverage와 origin/조건 macro 평균 57,024개를 독립 계산해 일치 확인.
- `original_correct`와 `both_correct` subset을 원래 15-token logits argmax와 corpus 정답에서 직접 재계산해 확인.
- P6 선택 validation MSE와 P7 val fidelity MSE의 최대 상대 차이 `2.2022e-08`. 이는 교차 일관성 검사이며 전체 cache 재현을 대신하지 않는다.
- 선택 10,763.81초, 의미 2,818.08초, 패칭 20,235.64초. 입력 검사·설치·집계 등은 이 단계 타이머 밖일 수 있다.

## 아직 남은 검사

반환 ZIP에는 P5 activation cache가 없다. 따라서 전체 probe refit·예측, 전체 train/val/test fidelity 및 dead latent,
semantic/causal/replacement와 층간 CI, 원본 activation 기반 validation bin 및 고정 GPU 패칭 재현은 아직 확인하지 않았다.
전체 선택이나 test 결과를 바꾸지 않고 원래 결과와 비교하는 별도 감사 노트북을 준비했다.
독립 추출 번들의 기존 입력 hash 582개·테스트 11개와 노트북 schema·셀 문법을 통과했다.

- [원본 재현 감사 노트북](../../notebooks/P7/P7_source_return_audit_r2.ipynb)
- [감사 입력 ZIP](../../bundles/P7/v1_4_p7_source_audit_r2.zip), 약 416MB
- [준비 검증](../p7_source_audit_preparation_r2/verification.json)

T4 GPU에서 실행한다. Drive `P5_r2`와 `P7_r1`은 읽기만 하고 `P7_source_audit_r2`에 검산 결과를 저장한다.
재학습은 없다. CPU refit·bootstrap에 수 시간이 걸릴 수 있으며 저장된 refit/run부터 재개한다.
완료 또는 오류 시 마지막 셀의 ZIP·checksum JSON을 반환한다. 이 반환을 확인하기 전 P7 완료/P8 진입이나 완료 Git push를 하지 않는다.

## 재현 명령과 상세 기록

- `verify_archive.py`: Python zipfile/hashlib로 수행한 전체 ZIP/member 검사의 재현 스크립트.
- `/opt/anaconda3/bin/python -u scripts/verify_v1_4_p7_return.py --root . --output experiment_v1_4/results/p7_return_audit_20260927_01/returned`
- `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/anaconda3/bin/python -u scripts/audit_v1_4_p7_metadata.py`
- [통합 검증](verification.json), [전수 원시 검산 로그](raw_verification.log), [독립 metadata/집계 검사](metadata_verification_r2.json), [P6 MSE 대조](p6_mse_crosscheck.json).

원본 ZIP·계약·선택·수치·그림을 수정하지 않았다. 앞선 `metadata_verification.json`은 subset 직접 검증을 추가하기 전 검사이며 r2 기록도 별도로 보존했다.
