# 4-block 인공어 실험 v1.1

사용자가 승인한 4-block 변경은 `CHANGELOG.md`에 정의한다. 2-block v1.0은 failed 결과와 원본 코드·설정을 보존한다. 신규 실행은 `interp_v1_1`과 이 디렉터리를 사용한다.

## Colab 실행

1. 새 GPU 런타임에서 `notebooks/01_colab_lm.ipynb`를 연다.
2. `bundles/p3_4block_bundle_v1.zip`을 업로드한다. 기존 v4 2-block 번들은 사용하지 않는다.
3. 위부터 실행하면 checksum·패키지·GPU smoke를 확인한 뒤 새 seed 0으로 시작한다.
4. 결과 `p3_4block_evidence_*.zip`을 전달해 실제 P3 판정을 받는다.

입력 데이터는 기존 immutable language_v1와 같다. 새 모델은 4-block·797,184 parameters, residual output 초기화는 1/√8이다. 1M 파일럿과 조건부 3M 연장, validation gate를 유지한다. 2-block checkpoint를 새 실행의 초기값으로 사용하지 않는다. Drive는 `MyDrive/boolean_interp_p3_4block_v1`로 분리한다.

## 상태

- Config·버전 분리 완료. config hash: `039c0fad2d268f8321d481b645498005ed672711762592b29d3f4ee068ddf2e8`.
- CPU smoke 통과: `smoke/cpu_01/smoke.json`, 797,184 parameters, 환경 `d7c07004dd4c1bd4`.
- 테스트 25개 통과. 원래 23개 계약 검사에 네 block gradient·2-block state 거부·깊이 기반 초기화 검사를 추가했다.
- GPU smoke 및 P3 본실험은 pending. CPU 결과로 GPU 통과나 행동 gate 통과를 주장하지 않는다.
- 배포 번들 전체 checksum·격리 테스트 증빙: `smoke/bundle_validation.json`.

## 재현

프로젝트 루트에서 다음 명령을 사용한다.

```bash
python -m pytest tests_v1_1 -q
python -m interp_v1_1.smoke --device cpu --output experiment_v1_1/smoke/NEW_OUTPUT
```

Colab에 전달하는 실제 실행 절차는 노트북에 있다. 로컬 파일이 macOS dataless 상태여서, 준비·검증은 원격 main의 `b7c6308407c2f0cca9c5825883a81b60ac95ccb1`을 임시 디렉터리로 복제한 뒤 수행했다. 기존 가상환경 접근 지연을 피하려 시스템 Python의 torch 2.13.0 환경과 임시 pytest 9.1.1을 사용했다. CPU smoke 환경 lock과 notebook의 Colab 환경은 별도로 기록한다.
