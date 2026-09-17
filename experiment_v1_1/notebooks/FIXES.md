# 노트북 실행 오류 수정

## Environment 폴더 생성 (2026-09-16)

사용자 보고: 입력 107개 checksum 검증 이후 `ENV_DIR.mkdir()`에서 FileNotFoundError 발생. 번들은 파일만 포함하므로 비어 있던 `experiment_v1_1/environment/`가 생성되지 않은 것이 원인이다.

수정: `ENV_DIR.mkdir(parents=True)`로 상위 폴더까지 생성한다. 기존 세션을 덮어쓰지 않도록 exist_ok는 추가하지 않았다. 기본 노트북, `01_colab_lm_environment_fix.ipynb`, 생성 스크립트에 반영했다. 기존 입력 ZIP·code/config hash·학습 설정은 변경하지 않았다.

검증: 수정된 노트북의 실제 경로 생성 구문을 부모 폴더 없는 임시 디렉터리에서 실행해 생성 성공과 동일 세션 재생성 거부를 확인했다. 전체 Python 코드 셀 compile 통과. Colab 학습 완료를 의미하지 않는다.

현재 런타임에서는 오류 셀의 한 줄을 수정해 그 셀부터 이어 실행할 수 있다. 입력 ZIP을 다시 업로드하거나 데이터를 재생성할 필요는 없다.
