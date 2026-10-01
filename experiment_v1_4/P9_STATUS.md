# P9 — sparse seed 1 초기화 반복

2026-10-02. **P9 완료: 학습·전체 평가·원본 재현 반환 감사 16/16 통과.**

선행 조건은 [P8 최종 감사](results/p8_final_audit_20260930_01/completion.json)이며 연결된 증빙 hash를 확인했다.
LM seed 0 × block 0/3/7/11 × SAE/TC × k=4/16, sparse seed 1의 16 runs다.
각 5,000 updates/2.56M draws, 총 80,000 updates/40.96M draws를 사용한다.

## 실행 파일

- [Colab 학습·진행 표시·수치 검산 노트북](notebooks/P9/P9_colab_sparse_seed1_training_r1.ipynb)
- [입력 ZIP](bundles/P9/v1_4_p9_training_r1.zip), [SHA-256](bundles/P9/v1_4_p9_training_r1.sha256.json)
- [학습 계약](p9_r1/contract.json), [계승한 평가 규칙](p9_r1/evaluation_rules.json)
- [준비 검증](results/p9_preparation_r1/verification.json), [준비 보고서](results/p9_preparation_r1/REPORT.md)

ZIP을 내 드라이브 `boolean_interp_v1_4/`에 업로드하고 Colab GPU 런타임에서 위부터 실행한다.
기존 `P5_r2`의 seed 0 원본 cache를 읽고 새 결과를 `P9_training_r1`에 저장한다.
12개 h/u/m train scalar 통계를 감사된 P6 값과 대조한다. 학습·선택 tensor는 train/val만 사용한다.
입력 번들은 코드·계약·검증 증빙을 포함하며 원본 P5 cache는 기존 Drive 파일을 사용한다.

중단 후 같은 노트북을 다시 실행하면 마지막 checksum checkpoint에서 optimizer·RNG·draw cursor를 복구한다.
현재 환경에서 CUDA smoke를 실행하며 환경 ID·설치 기록·처리량·진행 로그를 저장한다.
16 runs 후 336 checkpoint와 320 validation MSE/선택을 독립 검산한다.
완료 또는 중단 후 마지막 셀에서 결과 ZIP과 checksum JSON을 다운로드해 반환한다.

## 동결한 규칙과 다음 단계

SAE는 h→h, TC는 u→m이며 초기화는 각각 기존 규칙을 따른다.
동일 layer/k의 SAE·TC는 같은 seed 1 position draw 순서를 사용한다.
Seed 0 run 및 완료한 LM gate/test를 재실행하거나 변경하지 않는다.

1. Colab 원본 cache 검증, CUDA smoke, 16개 학습과 독립 수치 검산.
2. 반환 파일·환경·실제 draw·전체 checkpoint·선택 hash 감사.
3. 감사된 선택 checkpoint를 연결한 평가 전달물로 기존 P7/P8과 같은 전체 평가.
4. 별도 원본 cache 재현 감사 및 sparse seed 0/1 비교 표·그림.
5. 필수 결과가 모두 통과한 뒤 P9 완료 기록·Git main 업데이트.

평가에는 단일/≤4 의미 probe, 전체·128후보 대조, full/좌표/random 기준선,
fidelity·한 위치 대체·모든 인과 대조가 포함된다. Latent ID가 같다는 이유로 같은 feature로 간주하지 않는다.
Sparse seed 반복은 LM seed 반복과 구분한다. 학습만으로 P9나 전체 실험을 완료하지 않는다.

## 2026-09-30 학습 반환 감사·평가 전달물

[학습 완료 판정](results/p9_training_audit_20260930_01/completion.json), [감사 보고서](results/p9_training_audit_20260930_01/REPORT.md),
[선택 dictionary 16개](results/p9_training_audit_20260930_01/selected_dictionary_manifest.json)를 확인했다.
80,000 updates/40.96M draws, 336 checkpoint, 320 validation 수치 검산과 CUDA smoke·재개 감사 통과.
학습은 다시 실행하지 않는다. 필수 64 dictionaries 학습은 모두 끝났지만 P9 16개 전체 평가는 아직 남아 있다.

다음 실행 파일:

- [P9 전체 평가 Colab 노트북](notebooks/P9/P9_colab_sparse_seed1_evaluation_r1.ipynb)
- [평가 입력 ZIP](bundles/P9/v1_4_p9_evaluation_r1.zip), [checksum](bundles/P9/v1_4_p9_evaluation_r1.sha256.json)
- [SAE 평가 계약](p9_eval_r1/sae_contract.json), [TC 평가 계약](p9_eval_r1/tc_contract.json), [초기화 비교 계약](p9_eval_r1/comparison.json)
- [로컬 평가 준비 검증](results/p9_evaluation_preparation_r1/verification.json)

ZIP을 내 드라이브 `boolean_interp_v1_4/`에 올리고 GPU 런타임에서 위부터 실행한다.
기존 `P5_r2` 원본 cache와 번들의 선택 모델을 사용하며 새 결과는 `P9_evaluation_r1/sae` 및 `/tc`에 저장한다.
SAE 8개·TC 8개의 모든 선택을 동결한 뒤 의미·fidelity·대체·인과 평가와 층간 CI·초기화 비교를 수행한다.
Seed 0 기존 집계표와 비교하며 원래 평가를 다시 실행하지 않는다. Sparse seed 간 비교는 점 추정치이며 차이 CI를 주장하지 않는다.
중단 시 같은 셀을 재실행하고 완료 또는 중단 후 마지막 셀의 반환 ZIP·checksum JSON을 전달한다.
평가 반환 및 원본 cache 재현 감사 전에는 P9 완료/P10 진입을 표시하지 않는다. 앞선 학습 안내는 당시 이력이다.


## 2026-10-01 사용자 승인 요약·단계별 압축 정리

현재 로컬 가용성은 [정리 보고서](../maintenance/disk_cleanup_20261001/REPORT.md)를 우선한다. P1 학습/생성 자료·완료 단계 상세 감사·P6/P8 선택 모델은 `experiment_v1_4/phase_archives/P*/`에 무손실 압축했고, P6/P8/P9 비선택·debug checkpoint는 수치 감사·hash·로그를 남기고 삭제했다. 기존 원본 전체 보존 문구는 이번 명시적 삭제 목록에 한해 예외다. P9 평가 노트북·입력 ZIP·직접 입력은 유지했다. 과거 세부 검증은 최신 `maintenance/disk_cleanup_20261001/restore.py`로 필요한 단계를 복원한 뒤 실행한다. 비선택 tensor 전체나 폐기한 전달 ZIP bytes는 요약만으로 복원할 수 없다. 동결 manifest의 과거 경로·hash와 phase 완료 상태는 변경하지 않았다.


## 2026-10-01 P9 평가 반환·원본 재현 감사 준비

P9 SAE 8개·TC 8개 전체 평가 반환의 로컬 감사가 통과했다. 38,080 파일 hash, causal pairs 32,768개, READ 대체 targets 32,768개, probe tasks 1,200개와 초기화 비교 44,880행을 확인했다. 원본 cache refit·예측/fidelity/CI 재현 및 고정 GPU replay 감사가 남아 P9와 전체 실험은 미완료이며 P10은 대기다.

- [로컬 감사 보고서](results/p9_return_audit_20261001_01/REPORT.md), [판정](results/p9_return_audit_20261001_01/status.json)
- [다음 Colab 감사 노트북](notebooks/P9/P9_source_return_audit_r1.ipynb)
- [감사 입력 ZIP](bundles/P9/v1_4_p9_source_audit_r1.zip), [checksum](bundles/P9/v1_4_p9_source_audit_r1.sha256.json)
- [감사 계약](p9_audit_r1/contract.json), [독립 번들 검증](results/p9_source_audit_preparation_r1/verification.json)

입력 ZIP(180,654,824 bytes)을 Drive `boolean_interp_v1_4/`에 올리고 T4 런타임에서 위부터 실행한다. 기존 `P5_r2`, `P9_evaluation_r1`을 유지한다. 결과는 별도 `P9_source_audit_r1`에 저장하며 중단 시 같은 셀을 재실행한다. 마지막 셀에서 다운로드한 ZIP과 checksum JSON을 반환한다. 본 학습·평가를 다시 실행할 필요는 없다.

신규 refit 560개, 재사용 포함 1,200 tasks, 모델당 80,000 위치 fidelity와 전체 예측·CI를 재현한다. GPU replay는 사전에 고정된 suite별 첫 pair 4개와 READ 대체 첫 16개/run이다. 기존 평가 번들 332개 member byte 동일, 독립 번들 테스트 24개와 노트북 검증 통과. 이는 준비 검증이며 실제 원본 재현 감사 통과를 의미하지 않는다. 앞선 평가 대기 안내는 당시 이력이다.


## 2026-10-02 P9 완료

16개 sparse seed 1 SAE/TC의 학습·전체 평가·원본 cache 재현 반환 감사를 통과했다. 필수 dictionary 64개 모두 학습·평가·감사 완료다. P10 선택 분석 결정과 P11 최종 집계는 남아 전체 실험은 미완료다. 앞선 P9 대기 안내는 당시 이력이다.

[완료 판정](results/p9_final_audit_20261002_01/completion.json), [최종 보고서](results/p9_final_audit_20261002_01/REPORT.md). 신규 refits 560개, probe tasks 1,200개, 반환 파일 hash 594개와 재개 identity를 확인했다. 추가 Colab 실행은 필요하지 않다.
