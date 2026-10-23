# Boolean Transformer의 상태 표현과 인과 개입

**연구 질문:** 모델 내부에서 현재 상태를 잘 읽어 내는 sparse feature는, 그 상태에 대한 모델의 답도 바꿀 수 있는가?

Boolean 프로그램을 학습한 Transformer에서 선형 probe, Sparse Autoencoder(SAE), Transcoder(TC)를 비교했다. 핵심 관찰은 **높은 상태 판독 성능과 지정 위치에서의 개입 효과가 일치하지 않을 수 있다**는 것이다.

## 실험 흐름

```text
Boolean 프로그램 생성·독립 실행 검산
    → Transformer 3개 학습·행동 gate·모델 동결
    → 동일 READ 위치에서 activation 추출
    → full probe + SAE / TC 학습
    → 의미 접근성·복원·한 위치 대체·반사실 패칭
    → sparse 초기화 반복·불확실성 검증
    → 최종 결과 집계
```

| 핵심 설정 | 값 |
|---|---|
| 언어 | 변수 4개, 토큰 15개, SET / NOT / AND / OR / XOR / READ |
| 모델 | Decoder-only Transformer, 12 blocks, width 256, 4 heads, 약 9.49M parameters |
| LM 학습 | seed 0·1·2, 동일 순서의 데이터, 각 약 64M prediction tokens |
| 분석 위치 | READ 답 직전의 query-variable 토큰; block 0·3·7·11 |
| Dictionary | width 512, TopK 4·16, 각 5,000 updates |
| 반복 | 기본 READ 48개 + LM seed 0의 sparse 초기화 반복 16개 |
| 추가 분석 | Update block 3의 12개 dictionary; READ 이후 선택한 탐색 분석 |

## 핵심 결과

세 LM 모두 독립 validation 행동 gate를 통과했다. 일반 READ test 정확도는 **99.941% / 99.947% / 99.966%**였다.

아래는 기본 sparse seed 0, 최대 4개 feature의 현재 값 balanced accuracy(BA)와 changed-pair 답 전환율이다. 수치는 LM 3개의 산술평균이며, 패칭은 15-token argmax 기준이다.

<!-- CORE_TABLE_START -->
| 도구 | k | BA: block 0 | BA: block 3 | BA: block 7 | BA: block 11 | 답 전환율: block 11 |
|---|---:|---:|---:|---:|---:|---:|
| SAE | 4 | 0.5470 | 0.7629 | 0.9187 | 0.9499 | 0.3895 |
| SAE | 16 | 0.5901 | 0.8631 | 0.9841 | 0.9992 | 0.7987 |
| TC | 4 | 0.5544 | 0.8834 | 0.9727 | 0.9946 | 0.0000 |
| TC | 16 | 0.5890 | 0.9022 | 0.9912 | 0.9987 | 0.0000 |
<!-- CORE_TABLE_END -->

SAE는 residual 출력 `h`에, TC는 MLP 출력 `m`에 개입한다. 따라서 위 전환율은 **서로 다른 개입 위치의 결과**다. TC의 block 11 선택 패칭은 답을 전환하지 못했지만 반사실적 답 방향의 logit margin을 높였다. 이를 “TC에 인과 효과가 없다”로 해석하지 않는다. 좌표·random 방향도 강한 판독 성능을 보여, sparse feature의 독점적인 의미 분리를 주장하지 않는다.

같은 LM에서도 dictionary 초기화가 개입 결과를 크게 바꿨다. LM seed 0, block 11, k=16의 SAE는 sparse seed 0→1에서 **BA 0.9993→0.9991**, **답 전환율 0.8936→0.3857**이었다. 이 사례는 한 LM과 두 dictionary 초기화의 비교다.

![Dictionary 초기화에 따른 판독 성능과 개입 효과](figures/initialization_readout_and_patch.png)

## 읽는 순서와 확인 방법

1. [PIPELINE.md](PIPELINE.md): 데이터부터 개입까지의 방법·대조군·선택 규칙·코드 위치.
2. [results/](results/): 전체 최종 결과표. 모든 층·k·대조군·반복·탐색 분석을 압축 CSV로 보존했다.
3. [REPRODUCIBILITY.md](REPRODUCIBILITY.md): 제출본 검증, 코드 확인, 전체 재실행에 필요한 외부 입력.

Python 3 표준 라이브러리만으로 제출본의 파일 hash, 원본 결과표 hash, 결과의 출처와 핵심 집계를 확인할 수 있다.

```bash
python3 pipeline.py verify
python3 pipeline.py summary
```

## 해석 범위

- Probe는 라벨을 이용한 사후 감독 판독이다. Dictionary 학습에는 상태 라벨을 사용하지 않았다.
- 인과 결과는 한 층·한 READ 위치의 패칭에 한정된다. 전체 회로나 자연어 모델로 일반화하지 않는다.
- LM은 3개다. 층과 sparse 초기화 반복을 독립 LM 표본으로 세지 않는다.
- 네 층 분석 확장은 P5 test 관측 후 결정했고, Update는 READ 결과 후 선택했다. 사전등록으로 소급 표현하지 않는다.
- 의미·인과 CI는 sequence/origin 단위 cluster bootstrap이다. Fidelity와 sparse 초기화 차이에 없는 CI를 덧붙이지 않는다.

실험 v1.4의 최종 집계일은 2026-10-05다. 이 제출본은 2026-10-23에 정리한 코드·결과 검토 패키지이며, 대형 학습 데이터와 checkpoint는 별도 보존 자료다.
