# Mechanistic Interpretability Project

상태 추적 인공어를 학습한 Transformer 내부에서 READ 연산의 표현과 인과적 역할을 분석하는 연구 프로젝트입니다. 코퍼스 생성부터 행동 평가, probe, SAE, transcoder, activation patching까지 재현 가능한 실험 절차를 담고 있습니다.

## 구성

- `corpus/`, `data/language_v1/`: 인공어 생성 코드와 검증된 코퍼스
- `interp/`: 모델 학습 및 해석 도구
- `experiment_v1/`: 설정, 단계별 상태, 실행 기록과 smoke 결과
- `tests/`: 코퍼스 및 구현 검증 테스트
- `01_experiment_design.md`~`03_experiment_spec.md`, `phase.md`: 연구 설계와 실행 규격

## 빠른 확인

```bash
.venv-p2/bin/python -m pytest tests -q
.venv-p2/bin/python -m interp.cli --help
```

현재 P2 CPU/Colab GPU smoke 검증까지 완료됐으며, 본 실험 학습은 아직 실행하지 않았습니다. 정확한 환경과 진행 상태는 `experiment_v1/README.md` 및 단계별 상태 문서를 확인하세요.
