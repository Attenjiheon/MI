# P6 완료 반환 감사 — 2026-09-25

**P6 READ SAE 학습 완료. P7 의미·fidelity·인과 평가 진입 가능. 전체 실험은 미완료다.**

## 검증 결과

- 사용자 제공 전체 ZIP의 외부 checksum `cd858ec58d2b03883f425113229a28cdaccdba0062f83916aa48014ec480a068`과 내부 1,752개 파일 checksum을 확인했다. Manifest 포함 1,753개 항목이다.
- 세 LM × block 0·3·7·11 × k4/16 × sparse seed0 = 24 runs, 각각 5,000 updates/2.56M draws. 합계 120,000 updates/61.44M draws.
- 초기 100-update checkpoint를 포함한 504개 production checkpoint를 제한된 `weights_only` 로더로 읽었다. 모델 유한성·262,912 parameters·decoder norm·optimizer step/moment·동결 Adam 설정을 전수 확인했다.
- 각 저장점까지 PCG64 draw를 재생해 sampler state를 정확히 비교했다. Python/NumPy/Torch RNG 상태 형식, CUDA RNG bytes, cursor/통계/전체 curve prefix 및 환경 identity를 검증했다.
- 36개 train scalar 통계의 checksum/형식/유한성/표본 수를 확인했고 block0의 9개 값은 기존 P5 검증값과 정확히 일치했다. 네 층 입력 manifest는 검증된 P5 cache/READ keys를 가리킨다.
- 현재 source/config hash의 실제 Tesla T4 GPU smoke, 4층 SAE/TC debug/patch/probe, 8개 SAE 저장·재개 비교의 모델·optimizer·curve·sampler bitwise 일치를 반환 payload로 확인했다.
- r2 수치 감사의 24 runs/480 validation 결과와 이번 원본 checkpoint/curve/result hash를 연결했다. 모든 MSE 오차는 0이며 최소 validation MSE/동률 시 이른 update 선택 규칙과 일치한다.
- 24개의 선택 checkpoint를 `experiment_v1_4/p6_selected/`에 원본 bytes로 보존하고 [선택 manifest](selected_sae_manifest.json)를 작성했다. Run별 선택 update/MSE는 [runs.csv](runs.csv)를 따른다.
- 실패 run/실패 기록은 없다. 각 run의 세션·환경/재개점은 verification에 보존했다. 첫 100 updates의 처리량·peak VRAM은 본학습 예산에 포함됐다.
- 기록된 train 시간 합계 515.94초, validation 시간 합계 11.77초. 저장/전체 wall time과 구분한다.
- 마지막 로컬 P6 회귀검사 8개와 동결 입력/source file hash 검증 통과.

## 검산 보완과 해석 한계

r1 감사의 별도 행렬곱+bias 계산은 production F.linear와 float32 rounding이 달랐다.
실패 지점에서 legacy 값과 TopK 후보 변경 1개 행이 재현됐으며, r2의 동일 원시 연산 재구현에서 저장값과 정확히 일치했다.
허용 오차·학습·선택값을 바꾸지 않았다. r2 보완과 최초 실패 로그도 보존한다.

로컬 macOS/PyTorch 2.10의 초기화 재생성 hash는 Colab Linux/PyTorch 2.11의 기록과 다르다.
초기화 seed/코드/환경/model digest 및 sampler 초기 state는 보존·대조했고,
상이한 환경에서 초기 모델 bytes의 bitwise 재현을 주장하지 않는다. 이는 추가 로컬 진단이며 같은 GPU 환경의 저장/재개 검사는 통과했다.
원본 activation pool은 Drive에 있어 scalar fitting·val MSE를 로컬에서 다시 계산하지 않았다.
해당 bytes 검증과 scalar fitting은 동결 준비 코드 및 실행 입력 manifest,
480개 MSE는 고정 r2 GPU 감사 결과와 원본 checkpoint hash를 통해 확인했다.

P6는 학습 단계만 완료했다. SAE 의미·후보 수 대조·fidelity·인과 평가는 P7에 남아 있다.
TC 본학습/평가(P8)와 sparse seed1 반복(P9)도 남아 있다. Debug TC를 본실험으로 세지 않는다.

## 보존·재현·Git

원본 ZIP은 LFS로 보존하고, compact 원본 JSON/로그는 `returned_metadata/`에 보존했다.
원본 checkpoint는 선택본 이외에도 ZIP 내부에 모두 남아 있다. 반환물의 코드를 실행하지 않았다.
실행 명령: `/opt/anaconda3/bin/python experiment_v1_4/results/p6_final_audit_20260925/verify_return.py`.
이 보고서와 상태·registry를 포함하는 P6 완료 커밋을 원격 main에 반영하고 SHA 일치를 확인한다.
별도 진행된 용량 정리/과거 파일 이동 변경은 이 phase 커밋 범위에 포함하지 않는다.
