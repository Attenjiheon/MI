# v1.4 P2 — 로컬 부분 검증, 최종 corpus 및 GPU smoke 대기

## 확인한 증빙

- `results/pre_corpus_unit_tests.xml`: 15 tests passed, 오류/실패 0.
- 실행 명령: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests_v1_4/test_contract.py -q -k 'not stage and not frozen_corpus and not verify_inputs' --junitxml=experiment_v1_4/results/pre_corpus_unit_tests.xml`
- 설계/아키텍처/read4/선택/gate/토큰 cursor/LR와 historical-prefix 및 RNG metadata 회귀 검사를 포함한다.
- 별도 debug 데이터에서 다음 LM update의 bitwise resume, width 256의 12층 hook,
  SAE/TC × k=4/16 각각 100 updates, probe fit, identity/full/sparse patch를 확인했다.
- Colab 초안 `notebooks/P3_v1_4_r1_DRAFT.ipynb`의 7개 코드 셀을 compile했다.
  이 파일은 READY=False이며 실행용 배포본이 아니다.

## 아직 통과하지 않은 항목

- 수정본 P1과 결합한 frozen config 및 전체 18 tests (3개는 수정 corpus가 있어야 실행 가능).
- frozen input/code hash에 결합한 정식 CPU smoke와 real-CLI persistent-index 복구 감사.
- input ZIP 전체 checksum 및 최종 notebook/bundle 검증.
- 실제 Colab GPU smoke, 환경 lock, GPU 종류/VRAM 및 선택 microbatch.

현재 P2 완료가 아니다. 로컬 준비 파이프라인은 위 CPU 항목을 순차 수행하고 결과를
`results/local_preparation/`에 남긴다. 실제 GPU 증빙은 별도로 회수·검증해야 한다.
