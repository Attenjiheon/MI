# P9 sparse 초기화 반복 최종 감사

2026-10-02. **P9 완료.** LM seed 0 × block 0/3/7/11 × SAE/TC × k=4/16 × sparse seed 1의 16 runs가 학습·전체 평가·원본 재현 반환 감사를 통과했다. 필수 dictionary 총 64개 학습·평가·감사 완료. P10 선택 분석 결정과 P11 최종 집계는 남아 전체 실험은 미완료다.

## 검증 증빙

- 학습: 80,000 updates, 40.96M draws, 336 checkpoint와 320 validation 수치 검산. `../p9_training_audit_20260930_01/completion.json`.
- 본 평가 로컬 감사: 파일 hash 38,080개, causal pairs 32,768개, READ 대체 targets 32,768개, probe tasks 1,200개. `../p9_return_audit_20261001_01/status.json`.
- 원본 재현: 반환 ZIP 외부 SHA-256과 594 member hashes, 감사 계약·코드 hash, 16개 run별 판정 및 aggregate 완료 기록을 확인했다. 신규 refits 560개를 반환 선택에 800회 대조했다. P5 재사용 probe를 포함한 1,200 tasks, 의미 보고 2,160개, 모델당 train/val/test 50k/10k/20k 위치 fidelity와 dead 비율, 전체 validation matching bins, 모든 예측·의미/인과/대체 CI 및 층간 대비의 재현 증빙이 있다.
- GPU 재생은 동결 corpus 순서 suite별 첫 pair 4개/run(총 64개), READ 대체 첫 16개/run(총 256개)이다. 전체 본평가를 GPU에서 다시 실행했다고 주장하지 않는다. 전체 원시 pair의 logits·대조·집계 검산은 앞선 로컬 감사 증빙이다.
- sparse seed 0/1 비교표 44,880행과 그림을 재현했다. 동일 LM seed 0 내 점 추정치이며 차이 CI나 latent ID의 dictionary 간 의미 대응을 주장하지 않는다.

## 환경과 재개

Tesla T4, torch 2.11.0+cu128, NumPy 2.1.3, 환경 ID `f8912b7ab0ca620d`. 3개 session의 config 및 dependency lock hash가 일치한다. 앞선 실행 로그와 16개 run의 identity를 검증한 재개 로그를 모두 보존했다. 실패 파일은 없으며 중간 종료 원인은 기록만으로 확정하지 않는다. 완료 run의 감사 시간 합은 7,894.35초로 checksum·집계·중단 대기까지 포함한 총 벽시계 시간이 아니다.

반환 ZIP: `experiment_v1_4/evidence/P9/v1_4_p9_source_audit_return_1790870630393490455.zip`, SHA-256 `f1d227c666e5b3b6986f073c0eb4b59097df36a1b3f84a48e05ec544bfd371b6`.

## 재검증

```sh
/opt/anaconda3/bin/python scripts/verify_v1_4_p9_source_return.py
```

정확한 hash와 집계는 `completion.json`을 따른다. GPU 원본 재현은 동결 `P9_source_return_audit_r1.ipynb`와 대응 번들을 사용한다. 최신 파일 가용성은 2026-10-01 용량 정리 기록을 따른다. P9 완료 이후 원래 LM·dictionary·평가 선택을 변경하지 않았다.
