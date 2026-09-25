# P6 r1 실행 준비 — 2026-09-25

로컬 구현·검증과 Colab 전달물 준비 완료. 실제 P6 학습 0/24 runs이며 P6 완료가 아니다.

## 입력과 동결

P5 완료 manifest의 REPORT/verification/검산기/cache receipt hash를 재검증했다.
원본 cache는 Drive `boolean_interp_v1_4/P5_r2`에 있으며 로컬에 없다.
따라서 실제 240개 cache bytes 및 36개 scalar 통계 검증을 수행했다고 주장하지 않는다.
Colab 입력 준비 단계가 원본 contract/240개 NPZ/READ keys를 대조하고,
train만으로 통계 36개를 구해 block 0의 9개를 기존 검증값과 정확히 비교한다.
추가 27개 통계를 저장하고 분산의 별도 moment 항등식도 검산한다.
4층의 train/val h와 통계·원본 receipt를 새 input manifest로 묶는다.

실행 계약은 `experiment_v1_4/p6_r1/contract.json`이다.
3 LM × 4층 × 2 k = 24 SAE, 각 5,000 updates/batch 512/width 512,
총 120,000 updates·61,440,000 draws를 유지한다.
Init/PCG64 draw key와 정수를 고정했으며 이후 TC의 대응 draw key는 도구 및
도구별 hook을 제외한 동일 key를 사용해야 한다. 기존 LM RNG를 변경하지 않았다.

## 구현과 검증

- 네 층 모두 실제 12×256 debug LM에서 hook·READ 위치·정규화·역변환을 확인했다.
- Padding/causality, 두 LM forward/backward batch, 각 층 SAE/TC 두 k의 debug 100 updates,
  probe fitting 및 identity/full/sparse patch가 통과했다. 본 SAE/TC 학습과 구분한다.
- 네 층 × 두 k의 production SAE engine을 이용해 checkpoint 저장/재개 후 모델,
  optimizer, sampler state와 다음 loss/validation을 bitwise 비교했다. 8개 모두 통과했다.
- 손상 checkpoint/hash, 잘못된 resume identity, 중복 READ rows, TopK 동률 및
  encoder/decoder 독립성 검사와 기존 P5 관련 회귀검사 32개가 통과했다.
- 노트북 nbformat schema·모든 code cell 문법, 최종 code/config hash,
  번들을 별도 임시 디렉터리에 풀어 import 및 입력 hash 검증을 통과했다.

새 runner는 초기화 model digest/PCG64 state를 기록하고 첫 100 updates를 본 예산에 포함한다.
매 250 updates 전체 validation을 평가하고 모든 checkpoint와 checksum receipt를 보존한다.
마지막 완전 저장점부터 재개하며 미완성 payload는 별도 이름으로 보존한다.
5,000 updates 후 최소 validation MSE, 동률은 이른 update로 선택한다.
Best/last는 result manifest에서 보존된 update 파일을 참조한다.
실패는 환경 ID·이유를 별도 파일로 남기고 중단한다. 환경 변경 시 새 GPU smoke가 필요하다.

별도 검산기는 모든 480개 validation 평가를 직접 행렬 연산으로 재계산하고
선택 규칙·optimizer step·decoder norm·PCG64 draw cursor를 검증한다.
현재 이 production 수치 검산은 실행 전이다. 노트북에서 학습 완료 후 실행한다.

## 전달 및 다음 단계

1. `experiment_v1_4/notebooks/06_colab_read_sae_r1.ipynb`를 Colab에서 연다.
2. GPU 런타임을 선택하고 `experiment_v1_4/bundles/v1_4_p6_bundle_r1.zip`을 첫 셀에서 업로드한다.
3. Drive 연결 → 의존성·환경 → cache/통계 검증 → 현재 GPU smoke → 24 run 학습 순서로 실행한다.
4. 완료 후 독립 수치 검산과 결과 ZIP 다운로드를 실행한다. 중단 시에도 ZIP 다운로드는 가능하다.
5. 반환 ZIP을 로컬에서 검증한 뒤 상태·registry와 phase 완료 Git 절차를 수행한다.

출력은 Drive `boolean_interp_v1_4/P6_r1`이며 기존 P5 cache는 변경하지 않는다.
이 준비 작업은 새 phase 완료 판정이나 P7 진입 승인이 아니다. Git 완료 커밋/push는 미수행이다.

## 로컬 재현 명령

```sh
/opt/anaconda3/bin/python scripts/build_v1_4_p6.py
/opt/anaconda3/bin/python -m interp_v1_4.p6 smoke --root . --output experiment_v1_4/smoke/p6_cpu_r1 --device cpu
/opt/anaconda3/bin/python -m pytest tests_v1_4/test_p6.py tests_v1_4/test_p5.py tests_v1_4/test_p5_transfer.py tests_v1_4/test_p5_cpu_resume.py tests_v1_4/test_p5_independent_audit.py -q
```

Smoke 출력 경로는 새 경로를 사용한다. 기존 증빙을 덮어쓰지 않는다.
