from pathlib import Path
import json,sys,statistics,csv
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from interp_v1_4.runtime import sha
OUT=Path(__file__).parent
a=json.loads((OUT/'analysis.json').read_text());v=json.loads((OUT/'verification.json').read_text())
freeze_path=ROOT/'experiment_v1_4/results/p4_audit_20260921_01/validation_freeze.json'
f=json.loads(freeze_path.read_text())
labels={'general':'일반 READ','other_variable':'다른 변수 업데이트','repeated_update':'반복 업데이트','first_read_after_set':'SET 뒤 첫 READ','first_repeat_first':'First READ (42-cell macro)','first_repeat_repeat':'Repeat READ (42-cell macro)','composition_0':'Composition 0','composition_1':'Composition 1'}
lines=['# v1.4 P4 frozen test 반환 감사 및 분석','', '2026-09-22. **세 seed의 21개 frozen test 평가와 반환 감사 완료. P4 완료, P5 진입 조건 충족.**','',
'Validation에서 통과한 seed 0·1·2를 모두 다음 표현 분석 대상으로 유지한다. Test는 최종 보고이며 새로운 합격 gate가 아니다. Test 점수에 따른 모델·checkpoint 재선택이나 학습 변경은 하지 않았다.','',
'## 정확도와 오류 수','', '각 셀은 정확도 % (오류 수 / READ target 수)다. 모든 seed는 같은 test 자료를 사용했다.', '', '| 평가 | Seed 0 | Seed 1 | Seed 2 | Seed 평균 % |','|---|---:|---:|---:|---:|']
for name,rows in a['by_suite'].items():
 vals=[f"{r['accuracy']*100:.4f} ({r['errors']}/{r['targets']:,})" for r in rows]
 lines.append('| '+labels[name]+' | '+' | '.join(vals)+f" | {a['seed_mean_min_max'][name]['accuracy']['mean']*100:.4f} |")
lines+=['','Seed당 20,992 sequences / 51,170 READ targets, 세 seed 총 153,510 target 평가다. 서로 다른 suite를 한 점수로 합쳐 합격 기준을 만들지 않는다. First/repeat는 42개 cell에 각각 128쌍이 있어 macro와 micro가 일치한다.','',
'## 핵심 해석','',
'1. 일반 READ는 평균 99.9514%, seed 범위 99.9410–99.9659%다. Validation의 높은 성능이 별도 test에서도 유지됐다. 일반 READ의 sequence-cluster bootstrap 95% CI는 seed 0 99.9100–99.9690%, seed 1 99.9162–99.9748%, seed 2 99.9442–99.9845%다.','',
'2. First READ는 seed별 3·3·2개 오류, repeat는 세 seed 모두 0개 오류다. First의 8개 오류는 전부 XOR이며 7개는 depth 4+다. `XOR:01:depth_4+` cell은 각 seed에서 2/128 오류로 98.4375%다. 세 seed가 모두 틀린 동일 pair는 1개이며, 세 seed의 first 오류를 합친 고유 pair는 5개다. 공통 취약 사례는 있지만 이를 모델 내부 계산 방식의 증거로 단정하지 않는다.','',
'3. First의 answer CE는 0.001772 / 0.002811 / 0.000989, repeat는 0.000190 / 0.000043 / 0.000016 nats다. First−repeat CE 차이는 세 seed 모두 paired bootstrap CI가 양수다. 정확도 차이 CI는 0을 포함하므로, 정확도 격차의 통계적 확정을 주장하지 않는다.','',
'4. Composition 0은 4·1·1개 오류(99.6094–99.9023%), Composition 1은 0·0·1개 오류다. 일반 평가보다 일부 조합에서 CE가 커지지만, 예약된 두 조합에서 성능 붕괴는 관측되지 않았다. 이는 평가한 두 holdout에 관한 결과이며 임의 조합·장문 일반화까지 보장하지 않는다.','',
'5. 일반 READ 기준선은 majority 50.8316%, Bernoulli 50%, 최근 SET/초기값 63.9949%, 최근 READ 답 복사 57.7949%다. First 평가의 최근 SET/초기값과 READ 복사는 각각 54.9107%, 52.8460%다. 모델은 이 규칙들만으로 설명할 수 없는 높은 성능을 보인다. 구체적인 상태 표현·회로·인과 기능은 P5 이후에 검증해야 한다.','',
'6. 일반 READ 오류는 XOR 26/47건이다. 연산별 분모도 확인하면 XOR 오류는 seed별 12·9·5 / 5,117 targets이고 SET 7,321 및 초기화 2,247 targets는 각 seed 모두 오류가 없다. 오류 유형 분석은 사후 기술 통계이며 새 모델 선택 규칙으로 사용하지 않는다.','',
'## CE와 seed 변동','', '| 평가 | Seed 0 CE | Seed 1 CE | Seed 2 CE | 평균 CE | 최소–최대 CE |','|---|---:|---:|---:|---:|---:|']
for name,rows in a['by_suite'].items():
 ag=a['seed_mean_min_max'][name]['answer_ce']
 lines.append('| '+labels[name]+' | '+' | '.join(f"{r['answer_ce']:.6f}" for r in rows)+f" | {ag['mean']:.6f} | {ag['minimum']:.6f}–{ag['maximum']:.6f} |")
lines+=['','CE는 full-vocabulary READ answer CE다. All-token CE, B0/B1 제한 정확도, bit mass, 각 strata·coverage·baseline은 `analysis.json`과 반환 `result.json`에 보존했다. 전체 153,510 targets에서 full-vocabulary와 binary 정확도는 일치한다.','',
'100% 셀의 경험적 bootstrap CI [100%,100%]는 모집단 오류 확률이 0이라는 뜻이 아니다. CI는 각 고정 모델의 test 표본 변동을 나타내며, 3개 seed를 대규모 독립 모델 표본처럼 해석하지 않는다. Seed 평균·최소·최대는 `analysis.json`에 모두 저장했다.','',
'## 반환 감사','',
'- 원본 ZIP 외부 SHA256 및 내부 115개 파일 checksum 통과.','- 동결 validation 기록, 3개 선택 checkpoint hash, 7개 데이터 hash, 원본 runtime source 일치 확인.','- 한 notebook attempt의 6 tests 및 Tesla T4 GPU preflight 통과. 환경 ID `e74fb1dcf8112ca0`, microbatch 16으로 학습 반환과 같은 환경이다.','- 21개 시작 기록·raw·결과·completion의 결합과 해시 확인. 저장된 raw에서 집계·CI를 재계산해 모든 결과와 정확히 일치했다. 모델 추론이나 gate/test 재채점은 하지 않았다.','- 원본 corpus metadata와 153,510개 행의 target ID·답·연산·변수·거리·기준선 입력 및 pair cell을 대조했다. 실제 prediction token 수도 원본 sequence 길이 합과 일치한다.','- 순수 저장된 평가 시간은 seed 0 72.87초, seed 1 70.65초, seed 2 71.81초, 총 215.32초다. 설치·업로드·preflight·후처리 시간은 이 합에 포함되지 않는다.','',
'## 완료 판정과 다음 단계','',
'P3 및 seed 1·2의 검증된 학습/validation 동결에 이번 전체 frozen test 반환 감사를 결합해 P4 완료로 기록한다. Gate 실패 seed는 없고 seed 0·1·2 모두 유지한다. 다음은 **P5 block 0 READ activation cache·probe**이며 아직 실행하지 않았다. Length는 P10, READ SAE·TC·인과 평가 및 sparse seed 반복도 남아 있으므로 전체 실험 완료는 아니다.','',
'근거: [검증 manifest](verification.json), [분석 수치](analysis.json), [오류 행](error_rows.json), [통과 모델 목록](frozen_lms.json), [P4 완료](completion.json).','']
(OUT/'REPORT.md').write_text('\n'.join(lines))
models=dict(status='frozen_for_P5',selection_basis='Previously frozen validation gates; no test-based model selection',passing_lm_seeds=[0,1,2],failed_lm_seeds=[],validation_freeze_sha256=sha(freeze_path),models=[dict(lm_seed=x['lm_seed'],selected_update=x['selected_update'],checkpoint=f"experiment_v1_4/frozen_test_r1/checkpoints/seed{x['lm_seed']}.pt",checkpoint_sha256=x['selected_checkpoint_sha256'],training_tokens=x['actual_prediction_tokens'],training_hashes=x['hashes']) for x in f['seeds']])
(OUT/'frozen_lms.json').write_text(json.dumps(models,indent=2)+'\n')
# Completion binds the immutable prior audits and all local final reports.
proofs=['archive_verification.json','verification.json','analysis.json','error_rows.json','REPORT.md','frozen_lms.json','audit.py','analyze.py','report.py']
completion=dict(status='passed_P4_complete',p4_complete=True,passing_lm_seeds=[0,1,2],failed_lm_seeds=[],p5_eligible=True,p5_started=False,test_used_for_selection=False,model_inference_repeated=False,validation_freeze_sha256=sha(freeze_path),prior_replication_audit_sha256=sha(ROOT/'experiment_v1_4/results/p4_audit_20260921_01/completion.json'),archive_sha256=v['archive_sha256'],evidence_sha256={p:sha(OUT/p) for p in proofs},git_update='Commit and push these records to main; verify remote HEAD. Commit identity is in repository history.')
(OUT/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
p=ROOT/'experiment_v1_4/results/run_registry.csv'
with p.open(newline='') as handle:reader=csv.DictReader(handle);fields=reader.fieldnames;records=list(reader)
if not any(r['run_id']=='frozen_test_return_20260922' for r in records):
 r={k:'' for k in fields};r.update(phase='P4',run_id='frozen_test_return_20260922',status='passed_P4_complete',scope='all_three_seed_frozen_test_and_return_audit',environment_id=v['environment_id'],seed='0;1;2',config_sha256=v['contract_sha256'],code_sha256=sha(ROOT/'interp_v1_4/frozen_test.py'),input_sha256=v['archive_sha256'],actual_tokens=0,elapsed_seconds=sum(a['inference_seconds_by_seed'].values()),evidence_paths='experiment_v1_4/results/frozen_test_audit_20260922_01/completion.json',failure_or_skip_reason='No failed LM seeds; test is not a gate; length remains P10',next_action='P5 block 0 READ cache and probes for all three frozen models',actual_command='python experiment_v1_4/results/frozen_test_audit_20260922_01/audit.py; python experiment_v1_4/results/frozen_test_audit_20260922_01/analyze.py; python experiment_v1_4/results/frozen_test_audit_20260922_01/report.py')
 with p.open('a',newline='') as handle:csv.DictWriter(handle,fieldnames=fields,lineterminator='\n').writerow(r)
print('Reports, frozen model list, P4 completion and run registry written.')
