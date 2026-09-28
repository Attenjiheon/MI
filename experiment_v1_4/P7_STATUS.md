# P7 — READ SAE 의미·fidelity·인과 평가

2026-09-28. **P7 완료. 24/24 평가 및 원본 재현 반환 감사 통과. 다음 단계는 P8이다.**

선행 조건은 [P6 완료 판정](results/p6_final_audit_20260925/completion.json)과
[선택 SAE manifest](results/p6_final_audit_20260925/selected_sae_manifest.json)이다.
LM seed 0·1·2 × block 0·3·7·11 × k=4/16, sparse seed 0의 24개를 모두 평가한다.

- 선택: train ANOVA 누적 1–4 features와 validation probe 선택. 전체 후보 및 고정 128후보를 별도로 평가한다.
- 기준선: P5에서 독립 검산된 full probe와 block 0 좌표/random probe를 재사용하고 다른 층 기준선을 계산한다.
- Fidelity: train 평균 분모 NMSE, eval 평균 R2, EV, L0 및 선택 checkpoint의 전체 train 재인코딩 dead 비율.
- 대체: general behavior test의 sequence별 고정 READ 한 위치를 공유한다.
- 인과: causal validation에서 matching bin을 고정하고, changed/unchanged 각 1,024 test pairs를 양방향 평가한다. Memory/composition, 전체/both-correct, matched/unmatched 및 coverage를 분리한다.
- 재개: 완료된 fit/JSON 및 checksum이 확정된 NPZ 단위로 재개한다. 실패·NA·미매칭을 숨기지 않는다.

## 완료 조건

24개 모두의 의미·후보 수 대조·fidelity·대체·필수 인과 대조·표와 그림 및 실제 반환 감사를 완료해야 한다.
노트북 작성이나 로컬 debug 통과는 P7 완료가 아니다. P8은 아직 시작하지 않는다.

## 실행 파일과 검증

[준비 보고서](results/p7_preparation_r1/REPORT.md) · [검증 JSON](results/p7_preparation_r1/verification.json) · [실행 계약](p7_r1/contract.json)

[Colab 노트북](notebooks/P7/P7_colab_read_sae_evaluation_r1.ipynb)과 [입력 ZIP](bundles/P7/v1_4_p7_bundle_r1.zip)을 사용한다.
별도 폴더의 번들 hash/import/9 tests, 노트북 schema/셀 문법, 네 층·두 k CPU smoke를 통과했다.

## 2026-09-27 GPU 반환

반환 ZIP `v1_4_p7_return_1790455082926962917.zip`의 SHA-256은
`2d8d65b3bca7b6a0ec09d6f524d78c3495239ee2cf4a96fbc59a5187bf943449`다.
24개 run의 선택·의미/fidelity·대체·인과·집계 결과와 T4 GPU 환경 증빙을 반환받았다.
[반환 감사 보고서](results/p7_return_audit_20260927_01/REPORT.md)과
[원본 재현 감사 준비](results/p7_source_audit_preparation_r2/verification.json)를 따른다.

[추가 감사 노트북](notebooks/P7/P7_source_return_audit_Drive_r2.ipynb)과
[감사 입력 ZIP](bundles/P7/v1_4_p7_source_audit_r2.zip)을 사용한다.
기존 Drive `P5_r2` cache와 `P7_r1` 결과를 읽고 `P7_source_audit_r2`에 검산 결과만 저장한다.
재학습·feature/checkpoint 재선택 없이 refit 결과·전체 예측·fidelity·CI·matching bin을 대조한다.
이 추가 검산의 실제 반환 확인 전에는 P7 완료/P8 진입 또는 완료 Git push를 처리하지 않는다.
앞 절의 0/24·준비 상태는 당시 기록이다.

로컬 반환 검사: 56,919개 파일 hash, 24개 run 원시 수치, 1,680개 선택 작업, 인과 평균 57,024개 및 정답 subset 검사를 통과했다.
전체 원본 cache 재현은 별도 대기이며 [통합 검증](results/p7_return_audit_20260927_01/verification.json)의 확인/미확인 범위를 따른다.

## Drive 번들 연결

2026-09-27 사용자 요청으로 첫 셀을 Drive 마운트 후 ZIP 연결 방식으로 변경했다.
기존 `v1_4_p7_source_audit_r2.zip`을 `내 드라이브/boolean_interp_v1_4/`에 올려 둔다.
다른 위치라면 첫 셀의 `BUNDLE_IN_DRIVE`를 수정한다. Colab 로컬로 복사하며 checksum·진행률을 확인하고,
같은 런타임에서는 checksum이 맞는 로컬 ZIP을 재사용한다. 원래 감사 계약·입력 ZIP과 이후 실행 셀은 동일하다.

## 2026-09-28 최종 판정

[완료 판정](results/p7_final_audit_20260928_01/completion.json)과 [결과·감사 보고서](results/p7_final_audit_20260928_01/REPORT.md)를 기준으로 P7를 완료한다.
24개 모든 run의 의미·fidelity·대체·인과 평가, 840개 고유 probe refit, 전체 예측·CI·bin 재현 및 고정 GPU replay 표본 검사를 통과했다.
위 준비·대기 문구는 당시 이력이다. 선택 규칙·test 절차를 변경하지 않았다. P8 TC와 P9 초기화 반복은 미실행이다.
