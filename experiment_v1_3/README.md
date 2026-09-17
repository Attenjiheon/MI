# experiment_v1_3

상태: **design only**. 구현·데이터 생성·CPU/GPU smoke·학습은 아직 시작하지 않았다.

- [설계](./DESIGN.md)
- [기계 판독 계약](./design_config.json)
- [설계 hash manifest](./design_manifest.json)

v1.2는 16M 행동 gate 실패 실험으로 불변 보존한다. v1.3은 새 validation/test를 예약하고 제한된 3 architecture × 2 loss pilot으로 상태 전이 shortcut의 원인을 분리한 뒤, 사전 규칙으로 winner 하나만 32M 확인 단계에 올리는 별도 실험이다.
