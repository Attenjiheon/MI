# 환경 기록

P2의 CPU/Colab smoke test를 실제로 통과한 환경만 기록한다. 통과 전에는 lock하지 않는다.

| 파일 | 내용 |
|---|---|
| `requirements-cpu.lock.txt` | 로컬 `.venv-p2` (`pip freeze`), sha256 `ec616cdc…4cbd`, environment_id `96a2bbce68883fdf` |
| `requirements-colab.lock.txt` | Colab GPU 런타임 (`pip freeze`), sha256 `855ad8de…ffc7`, environment_id `e74fb1dcf8112ca0` |
| `runtime-cpu.json` | CPU 런타임 상세 (P2 CPU smoke에서 기록) |
| `runtime_manifest.json` | 두 통과 환경의 결합 기록: 런타임, lock, 입력 hash, 실행 명령, 증빙 경로 |

Colab lock은 `experiment_v1/smoke/colab_gpu_01/requirements.lock.txt`를 바이트 그대로 복사한 것이며, 두 파일의 sha256이 같음을 확인했다. GPU 런타임은 Tesla T4 / CUDA 12.8 / cuDNN 9.19.00 / torch 2.11.0+cu128 / Python 3.13.15다.

코퍼스 생성 환경은 `data/language_v1/manifest.json`에 따로 기록되어 있으며(Python 3.12.14 / NumPy 2.3.5) 위 실험 환경 lock을 대신하지 않는다.

판정 근거는 `experiment_v1/P2_STATUS.md`에 있다.
