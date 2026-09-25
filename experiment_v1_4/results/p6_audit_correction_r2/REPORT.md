# P6 검산 r2 보완

사용자가 반환한 로그에 24개 result.json과 training_complete.json이 있으며 동결 파일 hash 불일치는 없다.
처음 3 runs 검산 후 seed0_l3_sae_k16_s0/update250 MSE 불일치로 중단했다.
이는 파일 존재와 부분 검산 증빙이며 24개 학습의 최종 완료 판정은 아니다.

원래 검산기는 encoder/decoder에서 `x @ W.T + b`를 사용했지만 학습은 `F.linear`를 사용했다.
정규화도 Python scalar divisor와 float32 tensor divisor로 달랐다.
수학적으로 같은 식이지만 float32 rounding 및 불연속 TopK 앞에서 동일한 수치 연산은 아니다.
CPU에서도 linear 두 표현의 차이를 관찰했다. 실제 GPU 실패의 원인인지 확인하려면 r2 반환물이 필요하다.

r2는 별도 함수로 production의 float32 tensor 정규화·F.linear·stable TopK·float64 squared-error 누적을 구현한다.
원본 Dictionary.forward/normalize/mse를 호출하지 않는다. 공통 원시 연산을 이용한 재구현 검산이며
서로 다른 GPU 수치 backend 사이의 검증을 주장하지 않는다.
허용 오차 1e-7/1e-6, 기존 checkpoint 선택, 학습 데이터·모델·계약은 변경하지 않았다.
이전 계산식의 MSE·TopK 후보 변경 행 수·activation 차이·경계 gap도 checkpoint별로 기록한다.
통과 기준을 완화하지 않으며 r2에서 실패해도 그대로 증빙을 남기고 중단한다.

k4/16 및 512-row chunk 경계의 독립 구현 대조와 기존 P6 회귀검사 8개 통과.
기존 r1 ZIP을 byte 그대로 포함한 별도 audit ZIP과 notebook을 제공하며,
격리 압축 해제 후 원본 계약의 모든 file hash 및 새 검산기 import를 확인했다.
새 노트북은 GPU에서 검산만 수행한다. 재학습은 필요하지 않다.
실제 r2 GPU 검산 결과와 전체 P6 반환 감사는 미완료다.
