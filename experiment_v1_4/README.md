# experiment_v1_4

상태: **v1.4 정식 설계 동결, 로컬 실행 준비 중**.

v1.3의 감사된 seed 0 행동 gate 실패 뒤, 후속 제안 `deepwide12_read4`를 독립 버전으로
승격했다. 단일 12-block × 256-width 모델을 fresh seed 0에서 고정 64M prediction tokens까지
학습한다. 새 select/gate/test를 사용하며 v1.3 test는 열지 않는다.

- [설계](./DESIGN.md)
- [기계 판독 계약](./design_config.json)
- [P3 상태](./P3_STATUS.md)
- [원 제안](../experiment_v1_3/results/next_architecture_proposal.json)

## 단계 상태

| 단계 | 상태 | 완료에 필요한 증빙 |
|---|---|---|
| 설계 | 동결 | `design_config.json`, `design_manifest.json` |
| P1 새 corpus | 준비 중 | manifest, CPU validation, postwrite audit, exact prefix hashes |
| P2 구현·smoke | 준비 중 | tests, CPU smoke, Colab GPU smoke, 환경 lock |
| P3 seed 0 64M | 미실행 | init/milestone/last, select, one-time gate, 반환 ZIP 감사 |
| seed 1·2 | seed 0 gate 통과 전 대기 | 동일 동결 설정과 stream |
| frozen test | validation 결정 동결 전 금지 | 학습한 seed 전체 평가 |
| P5 이후 | 통과 LM 2개 전 금지 | READ probe·SAE·TC·인과 평가 |

## 다음 작업

1. v1.4 corpus를 생성하고 전체 hash·replay·quota·v1.3 train prefix byte identity를 감사한다.
2. 동결 config를 만들고 unit test와 CPU smoke를 통과한다.
3. Colab notebook과 checksum 검증 가능한 입력 bundle을 생성한다.
4. 새 GPU 런타임에서 GPU smoke 통과 뒤 seed 0 64M 학습을 실행한다.
5. 반환 증빙을 로컬에서 감사한 뒤에만 이 문서와 P3 상태를 갱신한다.
