# Run 저장 규칙

LM run은 `lm_seed_0/`, `lm_seed_1/`, `lm_seed_2/`에 저장한다. 각 run 내부의 checkpoint, log, activation, probe, dictionary, intervention 경로는 03 §14를 따른다. 실제 run ID와 manifest 필드는 `../configs/run_protocol.json`을 사용한다.

Checkpoint hash가 생기기 전 계획 단계에서는 run ID의 해당 필드를 `na`로 둘 수 있지만, 실행 시작 전에 실제 의존성 hash를 포함한 최종 ID로 registry를 갱신한다. 다른 dictionary run의 같은 latent ID를 같은 feature로 취급하지 않는다.
