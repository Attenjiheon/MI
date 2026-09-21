# v1.4 P3 r3 로컬 준비 검증

2026-09-21. **로컬 준비 통과, 실제 P3 학습·gate는 미실행**.

P1 전체 독립 감사와 P2 T4 반환 검증을 확인하고, 남은 cell CI 및 확장 행동 보고를
[보고 계약](REPORTING_CONTRACT.md)에 따라 보완했다. 모델·학습·동결 config·데이터·gate는
그대로 유지했다. 보고 구현의 변경은 `source_hashes.json`에 별도 기록한다.

## 검증 결과

- 전체 tests_v1_4: 30 passed. 미래 SET/READ 누출, 지정 target 필터링, 전체 token CE,
  빈 strata/coverage, paired bootstrap 및 학습 RNG 보존 회귀 검사 포함.
- 새 정식 CPU smoke: passed, 34.96초, frozen inputs verified, fixture_only=false.
  증빙: `experiment_v1_4/smoke/cpu_p3_r3/smoke.json`.
- 실제 CLI full/pause/persistent-index recover/resume: 모델·optimizer·RNG·cursor·microbatch·
  다음 평가 경계 bitwise equal. Full/resumed 반환 검증 모두 통과.
  증빙: `experiment_v1_4/smoke/resume_p3_r3/resume_equivalence.json`.
- r3 ZIP: 397 files, 1,574,956,425 bytes; CRC와 모든 내부 SHA-256 검증.
  Notebook 7 code cells compile; notebook checksum 상수와 ZIP/sidecar SHA-256 일치.
- 기존 r1/r2 번들과 CPU/GPU 증빙은 보존했다. Test/gate 모델 점수는 열지 않았다.

## 재현 명령

다음 명령은 저장된 증빙을 만들 때 사용했다. 출력 경로는 불변이므로 재실행 시 새 경로를 쓴다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/anaconda3/bin/python -m pytest tests_v1_4 -q --junitxml=experiment_v1_4/results/p3_preparation_r3/tests.xml
/opt/anaconda3/bin/python -m interp_v1_4.smoke --root . --device cpu --output experiment_v1_4/smoke/cpu_p3_r3
/opt/anaconda3/bin/python scripts/check_v1_4_resume.py --output experiment_v1_4/smoke/resume_p3_r3
/opt/anaconda3/bin/python scripts/build_v1_4_colab.py --revision r3 --cpu-report experiment_v1_4/smoke/cpu_p3_r3/smoke.json
```

## 다음 실행

`experiment_v1_4/notebooks/P3_v1_4_r3.ipynb`를 Colab GPU에서 열고
`experiment_v1_4/bundles/v1_4_p3_bundle_r3.zip`을 업로드하여 위에서 아래로 실행한다.
새 runtime hash에 맞는 GPU smoke 통과 후 fresh seed 0을 64,005,751 tokens / 8,399 updates까지
학습한다. Init/milestone/last 및 15분 주기 완전 update를 Drive에 저장한다.
재개 시 RESUME=True, 마지막 완전 index에서 복구하며 gate를 다시 열지 않는다.

이 로컬 환경에서는 CUDA 본학습을 수행하지 않았다. 반환 ZIP의 로컬 감사 전에는 P3 완료나
행동 gate 통과로 표시하지 않는다. P3 완료 후 AGENTS.md §6의 commit/push/remote SHA 검증을
수행한다. 현재는 phase 완료 커밋·push를 수행하지 않았다.
