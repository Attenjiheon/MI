# 로컬 준비 및 반환 검증 명령

저장소 root에서 `/opt/anaconda3/bin/python`으로 실행했다.

```bash
python scripts/build_v1_4_p5.py prepare
python -m pytest tests_v1_4 -q --junitxml=experiment_v1_4/results/p5_preparation_r2/tests.xml
python -m interp_v1_4.p5 preflight --device cpu --output experiment_v1_4/results/p5_preparation_r2/cpu
python scripts/build_v1_4_p5.py pack
```

Prepare/pack의 기존 동결 출력은 덮어쓰지 않는다. 동일 파일 검증은
`python -c 'from interp_v1_4.p5 import verify; verify(".")'`로 수행한다.
GPU/CPU 본실행 명령은 전달된 노트북에 포함했다. CPU probe는 동일 계약을 검증한
cache 폴더에 `python -m interp_v1_4.p5 probes --output CACHE_FOLDER`로 이어갈 수도 있다.

실제 결과 ZIP이 반환되면 외부 SHA256을 대조하고 새 디렉터리에 아래처럼 검증한다.

```bash
python scripts/verify_v1_4_p5.py NEW_RETURN_FOLDER --archive RETURN.zip --report NEW_REPORT.json
```

위 검증은 모델 추론을 하지 않으며 구조 검사까지만 수행한다. 완료 판정 전 원본 위치와
전처리 통계, validation 선택 계수, 집계/CI를 독립 감사하고 registry 및 summary를 작성한다.
현재 본 cache·probe가 없어 이 반환 명령은 실행하지 않았다.
