# 제출본 검증과 재현 범위

## 바로 실행할 수 있는 검증

이 폴더는 별도 프로젝트 경로나 GPU 없이 결과를 검토할 수 있는 제출본이다.

```bash
python3 pipeline.py verify
python3 pipeline.py summary
```

`verify`는 모든 포함 파일의 SHA-256, 압축 해제한 결과표와 동결 P11 hash의 일치, registry의 64 READ·12 Update run 구성, 논문 수치의 원본 CSV 행·필터·집계를 검사한다. `summary`는 검증 후 README의 핵심 표를 다시 계산한다. 어느 명령도 학습·추론·feature 선택을 수행하지 않는다.

| 폴더 | 내용 |
|---|---|
| `code/` | 최종 v1.4 구현, 필요한 공용 함수, 동결 config, 핵심 수치 검증 tests |
| `results/` | 최종 집계 CSV 전체를 gzip으로 무손실 압축; 모든 대조군·음성 결과 포함 |
| `figures/` | 본문에 사용한 초기화 비교 그림 |
| `provenance/` | 원본 완료 판정·검증·입력 hash·수치 출처·파일 이동 기록 |

## 코드 검사

Python 환경에 아래 의존성을 설치한 뒤 제출 폴더에서 실행한다.

```bash
python3 -m pip install -r code/requirements-interp.txt
cd code
python3 -m pytest -q tests
```

동결 GPU/CPU 환경별 버전은 `results/environment_index.csv.gz`에 경로와 hash로 식별되어 있다. 의존성 목록은 설치 편의를 위한 것이며 과거 CUDA 환경을 동일하게 재구성하는 lock은 아니다. 패키지 검증 결과는 `provenance/package_validation.json`에 기록한다.

제출 코드의 기존 `archive.legacy.interp_v1_2` 공용 의존성은 `interp_v1_4/_support`에 포함했다. Import 경로 변경과 필요한 함수 이동은 `provenance/source_map.json`에 원본·제출본 hash로 기록했다. 제출본의 코드를 과거 GPU 실행 당시의 동일 bytes라고 주장하지 않는다. 원본 동결 config의 과거 경로와 hash는 식별 정보로 그대로 두었다.

## 전체 실험을 다시 실행하려면

대형 corpus·activation cache·checkpoint·원시 pair별 예측 배열·Colab 반환 ZIP은 이 검토 패키지에 포함하지 않았다. 학습과 원본 감사를 재현하려면 별도 보존 자료에서 다음 입력을 확보해야 한다.

1. 활성 `data/language_v1_4/rebuild_01/` corpus와 manifest, 이전 train prefix·중복 방지 registry.
2. 동결 LM 3개의 checkpoint와 선택·행동 gate 증빙.
3. P5 READ cache 및 층별 전처리 통계, 선택된 SAE/TC checkpoint.
4. 원시 의미·대체·인과 평가와 단계별 보존 압축본, 당시 환경 lock.

입력 식별은 `results/reproduction_inputs.csv.gz`, `provenance/input_manifest.json`, `code/experiment_v1_4/`의 계약을 따른다. 이 경로들은 **원본 저장소 기준의 출처**이며 모두 제출본 내부에 있다는 뜻이 아니다. 일부 비선택/debug checkpoint와 과거 전달 ZIP bytes는 기존 정리로 삭제되어 완전 복구할 수 없다.

학습 코드에는 입력·현재 환경·smoke gate가 남아 있다. 누락 입력을 채우지 않고 실행을 강행하거나, 과거 hash 검사를 끄면 같은 실험의 재현이 아니다. Colab에서 전체 학습을 진행할 경우 원본 단계별 실행 노트북·입력 번들과 업로드 checksum, 환경 기록, 저장·재개·다운로드 절차가 추가로 필요하다. 제출본에는 사용자가 즉시 실행해야 할 Colab 작업이 없다.

## 포함·제외 원칙

최종 v1.4의 방법과 결과 검증에 필요한 코드·설정·수치를 포함했다. 과거 버전의 별도 실험, 중간 전달 번들, 재개 로그, 용량 정리 기록, 반복된 상태 문서는 제출본에서 제외했다. 원본 MI 작업 폴더와 기존 증빙은 별도로 보존한다. 과거 architecture/budget pilot의 행동 gate 실패를 v1.4의 실패 seed 대체로 취급하지 않으며, 최종 모델과 분석을 다시 선택하지 않았다.
