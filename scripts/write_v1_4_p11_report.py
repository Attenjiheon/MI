"""Generate the final Korean report from frozen P11 tables, with explicit limits."""
from pathlib import Path
import argparse,json
import pandas as pd


def md(frame):return frame.to_markdown(index=False,floatfmt='.4f')


def report(root):
 p=Path(root);s=pd.read_csv(p/'semantic_metrics.csv',low_memory=False);f=pd.read_csv(p/'fidelity_metrics.csv');c=pd.read_csv(p/'causal_metrics.csv');b=pd.read_csv(p/'behavior.csv');cov=pd.read_csv(p/'matching_coverage.csv');ld=pd.read_csv(p/'layer_differences.csv',low_memory=False)
 for df in (s,f,c,b):df['value']=pd.to_numeric(df.value,errors='coerce')
 count=json.loads((p/'aggregation.json').read_text())['counts']
 q=b[(b.category=='frozen_test')&(b.metric=='accuracy')&b.stratum.isin(['overall','42-cell macro'])]
 behavior=q.pivot(index='lm_seed',columns='suite',values='value').reset_index()
 q=s[s.phase.isin(['P7','P8'])&(s.metric=='balanced_accuracy')&(s.label=='current')&(s.domain=='iid')&(s.subset=='all')&(s['size']=='up_to_four')&s.representation.isin(['sae','tc'])]
 means=q.groupby(['tool','k','layer']).value.mean().unstack('layer').reset_index()
 main=q.groupby(['tool','k','layer']).value.agg(['min','max']).reset_index()
 sem_example=s[(s.phase.isin(['P7','P8']))&(s.metric=='balanced_accuracy')&(s.label=='current')&(s.domain=='iid')&(s.subset=='all')&(s['size']=='up_to_four')&(s.k==16)&(s.layer==11)&s.representation.isin(['sae','tc'])]
 sem_example=sem_example[['tool','lm_seed','value','ci_low','ci_high','bootstrap_valid']]
 causal_example=c[(c.sparse_seed==0)&(c.k==16)&(c.layer==11)&(c.condition=='all')&(c.subset=='all')&(c['size']=='up_to_four')&(c.control=='selected')&(c.changed==True)&(c.metric=='full_flip')][['tool','lm_seed','value','ci_low','ci_high','origins']]
 cmean=c[(c.sparse_seed==0)&(c.condition=='all')&(c.subset=='all')&(c['size']=='up_to_four')&(c.control=='selected')&(c.changed==True)&(c.metric=='full_flip')].groupby(['tool','k','layer']).value.mean().unstack().reset_index()
 fidelity=f[(f.position_type=='READ')&(f.sparse_seed==0)&(f.split=='test')&(f.metric=='nmse')].groupby(['tool','target','k','layer']).value.mean().unstack().reset_index()
 transfer=s[s.phase.isin(['P7','P8'])&(s.metric=='balanced_accuracy')&(s.label=='current')&(s['size']=='up_to_four')&(s.k==16)&s.representation.isin(['sae','tc'])&(((s.domain=='iid')&(s.subset=='current_ne_previous'))|((s.domain=='transfer')&(s.subset=='all')))].groupby(['tool','domain','subset','layer']).value.mean().unstack().reset_index()
 coverage=cov.groupby(['tool','sparse_seed']).matched_fraction.agg(['min','mean','max']).reset_index()
 sparse=s[(s.lm_seed==0)&s.phase.isin(['P7','P8','P9'])&(s.metric=='balanced_accuracy')&(s.label=='current')&(s.domain=='iid')&(s.subset=='all')&(s['size']=='up_to_four')&s.representation.isin(['sae','tc'])].pivot(index=['tool','k','layer'],columns='sparse_seed',values='value').reset_index();sparse['difference_1_minus_0']=sparse[1]-sparse[0]
 maximum=sparse.loc[sparse.difference_1_minus_0.abs().idxmax()]
 # All three transfer domains and separate hooks, tools, and k values.
 update=s[(s.phase=='P10')&(s.metric=='balanced_accuracy')&(s.representation=='latent')&(s['size']=='up_to_four')&(s.subset=='all')].groupby(['tool','k','label','domain']).value.agg(['mean','min','max']).reset_index()
 update.to_csv(p/'update_summary.csv',index=False)
 read_registry=pd.read_csv(p/'dictionary_registry.csv');bs=pd.read_csv(p/'lm_seed_summary.csv',low_memory=False)
 text=f'''# v1.4 최종 실험 보고서 — P11

집계 시작 2026-10-04, 최종 검증 2026-10-05. 본 보고서는 기존 동결 결과의 통합 집계다.
최종 완료 판정은 [completion.json](completion.json), 검증 항목은 [verification.json](verification.json)을 따른다.

## 범위와 판정

LM seed 0·1·2가 모두 validation gate를 통과했다. 세 모델을 바꾸지 않고 block 0·3·7·11의 READ SAE/TC, k=4/16, sparse seed 0의 48개와 LM seed 0의 sparse seed 1 반복 16개를 완료했다.
선택 분석은 Update block 3의 12개다. 총 76개 dictionary가 5,000 updates씩 학습·평가·반환 감사를 통과했다.
필수 READ는 320,000 updates/163.84M draws, Update는 60,000 updates/30.72M draws이며 LM 학습은 총 192,017,253 tokens다.

이 보고서의 주요 관찰은 깊은 층에서 현재 값의 선형 접근성이 높아지는 반면, 높은 sparse probe 성능만으로 해당 위치의 패칭 효과를 예측할 수 없다는 것이다.
특히 TC는 깊은 층에서 높은 접근성을 보이지만 이번에 지정한 MLP 출력의 선택 feature 패칭에서는 flip이 거의 없었다. 이는 해당 개입에 대한 관찰이며 TC의 일반적 무용성이나 다른 위치의 인과적 부재를 의미하지 않는다.

## 행동 학습과 최종 test

아래는 정확도 비율이다. First/repeat는 42-cell macro이며 test로 checkpoint나 gate를 다시 고르지 않았다.

{md(behavior)}

선택 update는 seed 0·1이 7,983, seed 2가 8,399다. 실패 LM seed는 없다.
[behavior.csv](behavior.csv)는 선택용 학습 곡선, validation gate, frozen test의 CE·정확도·bit mass·조건별 표본 수·기준선을 구분한다.
행동 기준선은 같은 target에 적용한 majority, Bernoulli 기대값, 최근 SET/초기화, 최근 READ 복사다.
셀 CI·조건 macro와 원래 보고용 unweighted CE는 학습 read4 loss와 구분한다. 선택용 10개 milestone을 연속 매-update 평가처럼 해석하지 않는다.

![행동 곡선과 최종 test](figures/01_behavior.png)

## READ 의미 접근성

주 분석 라벨은 READ 현재 값이다. 아래는 ≤4 feature balanced accuracy의 LM 3개 평균이다. 열 0/3/7/11은 block이며 독립 모델 반복이 아니다. k=4와 k=16 모두 보존한다.

{md(means)}

Full h probe 평균은 block 0/3/7/11에서 각각 0.6140/0.9381/0.9982/0.9995였다.
좌표·random 방향도 같은 감독 선택을 사용하며, 본 결과는 SAE/TC만이 현재 값을 독점적으로 분리한다는 주장을 지지하지 않는다.
Full u/full m, 단일/≤4, 128후보 통제, 256 좌표/random 후보와 512 latent 후보를 [semantic_metrics.csv](semantic_metrics.csv)에 구분했다.

Block 11, k=16, ≤4 current IID의 개별 seed와 **sequence cluster 95% CI**:

{md(sem_example)}

현재≠과거 subset 및 query 변수 전이의 k=16·≤4 LM 평균:

{md(transfer)}

![현재 값 접근성](figures/02_semantic_iid.png)
![현재와 과거 값이 다른 subset](figures/03_current_not_previous.png)
![Query 변수 전이](figures/04_query_transfer.png)
![동일 128 후보 수 대조](figures/05_equal_candidate_control.png)
![전 12층 full probe와 학습 전 모델](figures/06_full_probe_diagnostic.png)

P5의 3,510개 과제는 모두 감사 passed이며 새 fitting 실패나 support 부족으로 누락된 과제는 없다.
다중 클래스의 binary AUROC는 정의되지 않아 NA/유효 bootstrap 0으로 보존했다. 이를 0점이나 실패 run으로 바꾸지 않았다.
Token/position·shuffled-label·init 및 보조 라벨 전체 결과도 통합표에 포함된다.

## Fidelity와 한 위치 대체

아래 NMSE는 **서로 다른 target**에 대한 점 추정치의 LM 평균이다. SAE는 h 복원, TC는 u에서 m 예측이며 두 표적의 수치를 같은 복원 과제의 우열로 비교하지 않는다.

{md(fidelity)}

R², EV, L0 평균·중앙값·분위수와 train 전체 재인코딩 dead 비율을 [fidelity_metrics.csv](fidelity_metrics.csv)에 보존했다.
Val/test inactivity는 train dead와 구분한다. Fidelity 원본은 점 추정치이며 여기에 bootstrap CI가 있는 것처럼 표시하지 않는다.
한 위치 대체의 CE·정확도 차이는 동일 sequence의 동일 READ target을 paired 비교했고, 해당 지표에는 1,000-draw CI가 있다.
추가 층간 대체 차이는 저장된 2,048개 sequence의 예측값만 사용해 재집계했다. 모델 추론은 반복하지 않았다.

![Fidelity와 한 위치 대체](figures/07_fidelity_replacement.png)

## 지정 위치에서의 인과 개입

Changed pair에 대한 선택 ≤4 feature의 **15-token argmax flip 비율**, 전체 origin·양방향 평균, LM 3개 평균:

{md(cmean)}

Block 11, k=16의 개별 seed와 **origin cluster 95% CI**:

{md(causal_example)}

이 결과는 SAE residual h 패칭과 TC MLP-output m 패칭의 서로 다른 개입을 비교한다.
SAE는 깊은 층에서 선택 패칭의 flip이 증가했으나, TC의 block 7·11 선택 패칭 flip은 이번 설정에서 0이었다.
Full donor도 보장된 상한으로 가정하지 않는다. TC residual skip 유지와 원본 오차 보존은 동결 수식을 따른다.
Unchanged 선택 패칭 오류 유발률은 기본 48개 run의 이 집계에서 0이었다. 표본에서의 0은 모집단 위험이 없다는 증명이 아니다.

[causal_metrics.csv](causal_metrics.csv)는 changed margin·full/binary flip, unchanged 오류·예측 변경, patch norm을 포함한다.
기억/합성/조건 macro, 전체 pair/양쪽 원래 정답 subset, 단일/≤4 및 모든 11개 대조군을 보존했다.
Matched random은 **selected_matched와 같은 성공 origin subset**에서만 비교해야 한다. 전체 selected와 coverage가 다른 random_matched를 직접 우열 판정에 사용하지 않는다.

{md(coverage)}

위 coverage 평균/범위는 suite/run별 기술통계다. 미매칭 origin을 성공 표본으로 대체하거나 숨기지 않았다.
[matching_coverage.csv](matching_coverage.csv)에 분모·matched·unmatched·NA를 모두 제공한다.

![인과 대조군](figures/08_causal_controls.png)
![Matching coverage](figures/09_matching_coverage.png)

## 층간 차이와 sparse 초기화

[layer_differences.csv](layer_differences.csv)는 높은 층−낮은 층 방향의 paired 차이 {len(ld):,}행이다.
의미·인과 차이는 이전에 감사한 동일 sequence/origin draw 결과를 재사용했다. Matched 층 비교는 두 층 matching 성공 origin의 교집합이다.
새 대체 차이 192행은 각 층에서 같은 sequence ID와 같은 draw를 공유한다. 서로 독립으로 구한 CI 끝점을 빼지 않았다.

![층간 paired CI](figures/11_paired_layer_intervals.png)
![Sparse 초기화 반복](figures/10_sparse_initialization.png)

Sparse seed 1−0 비교는 LM seed 0에 한정한다. 현재 IID ≤4 BA에서 가장 큰 절대 변화는 {maximum['tool']}, k={int(maximum['k'])}, block {int(maximum['layer'])}의 {maximum['difference_1_minus_0']:+.4f}다.
[sparse_seed_comparison.csv](sparse_seed_comparison.csv)의 44,880행은 기존 동결 점 추정 비교이며 차이 CI를 주장하지 않는다.
서로 다른 dictionary의 같은 latent ID를 같은 feature로 간주하지 않았다.
[lm_seed_summary.csv](lm_seed_summary.csv)는 층·도구·k별 LM 개별 값과 평균/최소/최대를 제공하며 sparse 반복을 LM 반복 수에 더하지 않는다.

## 선택 Update 분석과 생략

Update block 3의 12개 run은 READ 뒤에 선택된 탐색 분석이다.
[update_summary.csv](update_summary.csv)에 입력·관계·결과 라벨, 변수 전이, AND/OR→XOR 연산 전이의 LM별 범위를 구분했다.
전이는 감독 probe의 전이이며 dictionary 학습 분포 전체에 대한 holdout 주장이나 인과적 사용의 증거가 아니다.

![Update 의미·전이](figures/12_update.png)
![보조 라벨](figures/13_all_labels.png)

추가 dictionary 층, length GPU 평가, m→m SAE, Update 인과 개입은 동결 자원 결정대로 생략했다.
Crosscoder·변수 8개 확장은 기본 범위 밖이다. 생략한 결과를 0점으로 채우지 않았다.

## 불확실성·실패 이력·해석 한계

- 모든 저장된 의미/인과 CI는 1,000회 2.5/97.5 percentile cluster bootstrap이다. Interpretation은 sequence ID, causal은 origin ID를 표본 단위로 삼는다. 방향·후보를 독립 관측으로 세지 않는다.
- 고정 quota의 조건 안에서 재표집한 뒤 macro를 계산한다. 클래스 소실 등 불가능 draw는 제외하고 requested/valid를 기록한다. NA AUROC는 특히 valid=0이다.
- LM seed는 3개다. 평균·범위는 기술통계이며 세 seed를 정밀한 모집단 추정으로 포장하지 않는다. Fidelity 점 추정 및 sparse 비교에는 존재하지 않는 CI를 추가하지 않았다.
- 네 READ 층 확장은 P5 test 결과 관측 후 결정되었다. Update 역시 READ 결과 뒤 선택했다. 사전등록으로 소급 표현하지 않는다. 보조 라벨·다중 비교는 탐색적이다.
- 초기 거부 corpus, GPU/CPU 준비 실패, 정상 자원 중단·재개, P10 선택 완료 전 evaluate gate 오류는 기존 registry와 반환 감사에 남아 있다. 정상 재개 후 완료된 이력을 최종 실패 run 또는 미실행으로 오인하지 않는다.
- 필수 LM·dictionary의 최종 실패 또는 미완료는 없다. 행동 gate 실패 시 중단 경로를 택한 과거 버전은 이번 v1.4 성공 seed의 대체 실행이 아니다.
- 높은 probe 정확도는 선형 접근성/사후 감독 성능이다. 지정 위치의 패칭 결과를 전체 회로, 모든 문맥, 자연어 모델에 대한 인과적 설명으로 확대하지 않는다.
- P5 감사는 저장된 계수의 예측·지표 검산이며 모든 optimizer refit을 독립 반복한 것이 아니다. P7/P8/P9 원본 감사의 GPU 재연 범위는 각 원본 보고서의 고정 표본이며 전체 원시 지표 검산과 구분한다. P10은 별도 전체 GPU replay 감사를 마쳤다.

## 증빙·재현·가용성

- [dictionary_registry.csv](dictionary_registry.csv): 76개 실제 run, LM/checkpoint/config/선택 hash, 실제 update/draw 수.
- [run_registry.csv](run_registry.csv): 기존 phase별 실행·실패·중단 이력을 그대로 보존한 snapshot.
- [prerequisites.json](prerequisites.json): P1–P10 완료 및 GPU smoke 증빙.
- [input_manifest.json](input_manifest.json): 실제 읽은 파일·ZIP member hash, 압축본 전체 hash와 이전 반환 검증 연결.
- [reproduction_inputs.csv](reproduction_inputs.csv), [environment_index.csv](environment_index.csv): 동결 code/data/config 입력 hash와 환경/lock의 실제 보존 위치.
- [verification.json](verification.json), [completion.json](completion.json): coverage·대조군·CI·표·그림 및 소스 코드 검증.

현재 보존본을 사용한 재현 명령(저장소 루트, NumPy/pandas/matplotlib/pytest가 있는 Python):

```bash
python -m interp_v1_4.p11 --output /tmp/v1_4_p11_reproduction
MPLCONFIGDIR=/tmp/mi-p11-mpl python scripts/report_v1_4_p11.py --output /tmp/v1_4_p11_reproduction
python scripts/write_v1_4_p11_report.py --output /tmp/v1_4_p11_reproduction
python scripts/verify_v1_4_p11.py --output /tmp/v1_4_p11_reproduction
python -m pytest -q tests_v1_4/test_p11.py
```

집계 실행기는 기존 출력 경로 덮어쓰기를 거부한다. 학습·평가·feature 선택·원본 GPU 감사를 다시 수행하지 않는다.
과거 학습/평가 재현 명령은 실행 당시 환경·hash와 함께 원본 registry의 actual_command 및 phase 노트북에 연결된다.
P11은 로컬 CPU 보고 작업이므로 새 Colab 학습이나 사용자 업로드가 필요 없다.

완료 단계의 비선택/debug checkpoint와 일부 전달 ZIP bytes는 사용자 승인 정리로 삭제되었고 요약으로 복원할 수 없다.
선택 모델·원시 평가·단계별 압축 보존본과 원본 hash는 유지한다. 대형 원본을 모두 Git에서 다시 받을 수 있다는 뜻은 아니다.
정리 이후 과거 경로 복원이 필요하면 저장소의 `maintenance/disk_cleanup_20261001/restore.py --path 원래경로`와 후속 정리 보고를 먼저 따른다.
본 P11 재현은 manifest에 명시된 보존 압축본과 직접 파일을 읽으며 기존 실험 증빙을 덮어쓰지 않는다.

통합 행 수: 행동 {count['behavior']:,}, 의미 {count['semantic_metrics']:,}, fidelity/대체 {count['fidelity_metrics']:,}, 인과 {count['causal_metrics']:,}.
'''
 (p/'report.md').write_text(text)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);report(parser.parse_args().output)
