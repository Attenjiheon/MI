# v1.3 P2 — 구현·CPU 검증 통과, GPU pilot 미실행

> 이 문서는 2026-09-18 P2 종료 시점의 불변 기록이다. 현재 pilot·confirm 진행 상태는
> [P3_STATUS.md](./P3_STATUS.md)를 따른다.

2026-09-18. v1.3 실행기와 여섯 pilot cell의 CPU smoke, unit test를 완료하고 Colab 입력 번들과 노트북을 준비했다.
**GPU smoke와 6-cell pilot은 아직 실행하지 않았다.** 이 문서는 준비 단계의 증빙이며 pilot 결과를 주장하지 않는다.

## 1. 선행 조건 재확인

작업 전에 동결된 입력을 다시 계산해 대조했다. `interp_v1_3.runtime.verify_inputs`가 corpus manifest의
**200개 파일 전부**를 바이트 hash로 검증하고, config set의 결합 hash와 P1 감사 상태를 확인한다.

| 항목 | 값 | 출처 |
|---|---|---|
| config set 결합 hash | `acd92751…` (`status: frozen`) | `configs/config_set_manifest.json` |
| corpus manifest hash | `7021b5f9…` | `data/language_v1_3/manifest.json` |
| 검증된 corpus 파일 수 | 200 | `verify_inputs` 반환값 |
| P1 post-write 감사 | `passed` | `data/language_v1_3/postwrite_audit.json` |

P1에서 확정된 수치(train 271,936 sequences / 32,004,917 prediction tokens, final update 4,249,
pilot 경계 8,001,583 tokens / update 1,082 / cursor 69,248)는 변경하지 않았고, 실행기가 이 값을
config에서 읽어 마지막 update에서 대조하도록 했다.

## 2. 이번 단계에서 새로 구현한 것

설계는 완료돼 있었지만 실행기가 없었다. `interp_v1_3/`에는 모델·loss·평가 모듈만 있었고,
학습 루프와 smoke가 비어 있어 pilot을 시작할 수 없는 상태였다.

- `interp_v1_3/cli.py` — 한 번에 frozen cell 하나를 학습한다. milestone에서만 평가하고(`select/`만 읽음),
  불변 checkpoint와 완료 index로 재개하며, 마지막 update에서 고정 cursor와 대조한다.
  `gate/`는 confirm 단계에서 선택된 checkpoint 하나에 대해 한 번만 열리고, `test/`는 열지 않는다.
- `interp_v1_3/smoke.py` — 여섯 cell 전부를 debug 고정 데이터로 검사한다.
- `interp_v1_3/runtime.py` — model tensor digest(`tensor_digest`)를 추가해 anchor bitwise 대조를 가능하게 했다.
- `interp_v1_3/persistence.py` — 감사된 v1.2 증분 영속 저장(`publish`)을 재노출했다.
- `scripts/verify_v1_3_evidence.py` — 반환된 run의 계약·보존·선택 규칙을 다시 계산해 대조한다.
- `scripts/select_v1_3_pilot_winner.py` — 여섯 run이 모두 끝난 뒤 §4.3 규칙만으로 winner를 판정한다.
- `scripts/build_v1_3_pilot_colab.py` — 입력 번들과 실행 노트북을 만든다.

설계값(architecture 3종, loss 2종, 예산, milestone, 선택·gate 규칙)은 모두 동결 config에서 읽으며
코드에 다시 적지 않았다. 여섯 cell 목록도 `run.json`의 값과 대조한다.

## 3. Unit test — 13 passed

증빙: [`results/p2_pytest.txt`](results/p2_pytest.txt) (13 passed, 9.17초).

기존 5개는 architecture별 파라미터 수와 forward shape, v1.2 초기화·uniform update의 bitwise 동일성,
token-only `read4` 가중, 42 cell first/repeat 의미 동치, §4.3 선택과 gate 규칙을 검사한다.
이번에 8개를 추가했다.

- 동결 예산·milestone·cursor 계약(pilot 8M / confirm 32M)이 config와 일치하고, 여섯 cell 외 값은 거부된다.
- checkpoint 선택이 first-member 42-cell macro CE 최소 → `1e-4` 이내 동률에서 select general CE → 이른 update 순을 따른다.
- 재개 시 cursor·update·token 누계 불일치를 거부한다.
- LR이 warmup 50k → 상수 → 28.8M부터 선형 감쇠 → 32M 이후 연장 없음을 따른다.
- 실행기와 평가 코드의 실행 문자열 중 `test/` 경로를 참조하는 것이 없다(AST 기반). 이 검사는 고의로
  `test/general.jsonl.gz` 문자열을 넣었을 때 실패함을 확인했다.
- production 실행이 CPU와 GPU smoke 부재를 거부한다.
- winner 동률 처리에서 uniform이 read4보다, 적은 파라미터가 많은 파라미터보다 우선한다.
- winner 보고 스크립트가 debug run을 거부한다.

## 4. 여섯 cell CPU smoke — passed

증빙: [`smoke/cpu_release/smoke.json`](smoke/cpu_release/smoke.json) (39.57초), 같은 디렉터리의
`requirements.lock.txt`와 `nvidia-smi.txt`(CPU 환경이므로 `unavailable`).

cell별로 다음을 확인했다.

| 검사 | 결과 |
|---|---|
| 파라미터 수 | base4 797,184 / wide4 1,640,544 / deep8 1,590,272, 동결 config와 일치 |
| padding 불변성 | 최대 절대 오차 0.0 (batch forward = 단일 forward) |
| causal·PAD key masking | 통과 (미래 토큰 변경과 PAD 토큰 치환이 결과를 바꾸지 않음) |
| RoPE | head width 32와 46 모두에서 인접 쌍 회전각을 해석적으로 대조 |
| gradient 누적 | 전체 batch와 microbatch 누적의 최대 절대 오차 ≤ 1.8e-7 |
| `read4` 가중 | answer 위치 145개, 분모 1,037 → 1,472 (= 1,037 + 3×145), 비 B0/B1 target은 예외 |
| checkpoint 재개 | 여섯 cell 모두 복원 후 다음 update가 bitwise 동일, log도 동일 |
| 평가 경로 | `select/`의 실제 pair로 42 cell 전부를 채워 macro/coverage 경로 실행 |

anchor 대조로 `base4` 초기화와 첫 uniform update가 감사된 v1.2 구현과 bitwise 동일함을 확인했다
(`anchor_v1_2_equivalence.status = passed`). 이는 구현 동치성 확인이며, **실제 1M/3M/8M checkpoint의
v1.2 보존본 대조는 GPU 실행 뒤에 수행한다**(§6).

smoke의 선택·gate 코드 경로는 학습되지 않은 debug 수치로 실행된다. `smoke.json`의
`selection_code_path`와 `gate_code_path`는 `debug_only: true`이며 실험 결과가 아니다. gate는
42-cell coverage만 만족하고 모든 정확도 조건에서 실패하는 것을 확인했다.

### 실행 환경

이번 CPU 검증은 사용자 Mac이 아니라 클라우드 Linux 컨테이너에서 실행했다.
환경 ID `f27a8fafc6f8eeaa`, Linux x86_64 / Python 3.11.15 / torch 2.14.0+cu130 / numpy 2.4.4.
Colab 환경 ID는 노트북 실행 시 새로 기록되며, 본학습은 그 GPU smoke 환경 ID와 일치해야만 시작된다.
v1.2 anchor 환경 ID `e74fb1dcf8112ca0`와의 일치 여부는 Colab 실행 기록으로 확인한다.

debug checkpoint(`*_debug.pt`, 약 186MB)는 재개 검사에만 쓰이는 일회용 산출물이라 git에 포함하지 않는다.
각 파일의 SHA-256은 `smoke.json`의 cell별 `resume.checkpoint_sha256`에 기록돼 있다.

## 5. Colab 전달물

- 입력 번들: `bundles/v1_3_pilot_bundle_v1.zip` (+ `.sha256`)
- 노트북: `notebooks/01_colab_pilot_8m.ipynb`
- 전달 기록: [`results/colab_delivery.json`](results/colab_delivery.json) — 번들 hash, 크기, 파일 수, 노트북 hash

노트북은 한 런타임에서 `CELL` 하나만 실행한다. 순서는 입력 checksum → 의존성·환경 기록 →
CPU 테스트 → CPU smoke → GPU smoke → 해당 cell 8M 학습 → 계약 감사·곡선 → 증빙 ZIP 다운로드다.
마지막 셀은 여섯 cell이 모두 끝난 뒤에만 winner를 판정한다. 모든 code cell은 번들 생성 시 컴파일 검사를 통과했다.

번들 안의 `P2_STATUS.md`는 번들을 만든 시점의 스냅숏이라 아래 §7(Git 갱신 기록)이 없다. 기준 문서는 저장소의 이 파일이며, 번들 내부 manifest는 자기 자신과 일치한다(260개 항목 재계산 확인). `run_registry.csv`는 번들 생성 이후에 만든 파일이라 번들에 없다. 둘 다 Colab runner가 읽지 않는 문서다.

## 6. 아직 확인되지 않은 것

1. **GPU smoke와 6-cell pilot 미실행.** 처리량·VRAM·실제 GPU 수치는 아직 없다.
2. **anchor bitwise 대조 미완.** `base4_uniform`의 update 136 / 407 / 1082 checkpoint를 v1.2 보존본과
   대조해야 한다. v1.2와 v1.3의 milestone update 번호가 같음(136/407/1082)은 양쪽 config에서 확인했다.
   대조는 `scripts/verify_v1_3_evidence.py --anchor-checkpoint <v1.2 checkpoint>`로 수행하며,
   v1.2 checkpoint는 `experiment_v1_2/evidence/p3_16m_verified_contents_20260917.zip`에서 꺼낸다.
   불일치하면 §4.3에 따라 탐색을 중단하고 구현·환경을 감사한다.
3. **confirm 단계 경로 미실행.** 32M 학습, gate 평가, seed 1·2는 winner 승격 이후의 작업이다.
   실행기에는 구현돼 있으나 CPU debug 경로로만 검증했다.
4. 이 문서의 어떤 항목도 행동 gate 통과나 SAE/TC 가설 검증을 의미하지 않는다.

## 7. Git 갱신 — 미완료 (환경 제약)

AGENTS.md §6의 Git 갱신은 **이 세션에서 완료하지 못했다.** 숨기지 않고 사유를 기록한다.

이 세션의 셸은 사용자 Mac에 직접 붙은 셸이 아니라 프로젝트 폴더가 마운트된 별도 Linux VM이다.
두 가지가 걸렸다.

1. 해당 VM에 `git-lfs`가 없다. `.gitattributes`는 `experiment_v1_3/bundles/v1_3_pilot_bundle_v1.zip`을
   LFS로 추적하도록 지정하지만, LFS 필터 없이 커밋하면 831MB blob이 그대로 들어가 원격에서 거부된다.
2. 프로젝트가 iCloud Drive에 있고 디스크 여유가 5.9GB(98% 사용)까지 떨어지면서, 마운트 경유 읽기가
   간헐적으로 `Errno 35 Resource deadlock avoided`로 실패했다. 대상 파일을 개별적으로 내려받아 재시도해도
   다음 시도에서 다른 파일이 같은 상태가 되었고, 마지막에는 `.git/objects/pack/*.pack` 자체가 읽히지 않아
   `git add`가 진행되지 않았다.

2026-09-19에 디스크 여유를 48GB(80% 사용)까지 확보한 뒤 다시 시도했으나 결과는 같았다. 원인은 공간이 아니라
git이 195MB pack 파일을 mmap한다는 점이며, 브리지 파일시스템은 그 mmap을 감당하지 못해 `git status`가
`Bus error`(SIGBUS)로 죽는다. 원격은 익명 fetch로 도달 가능하지만(`git ls-remote` 성공) push 자격증명은
Mac 키체인에 있어 이 VM에서 쓸 수 없고, git-lfs도 여전히 없다. 세 가지 모두 Mac에서는 문제가 되지 않는다.

따라서 **사용자 Mac의 터미널에서** 아래 한 줄을 실행한다. 스크립트가 iCloud placeholder 내려받기,
staging, LFS pointer 확인, 커밋, push, 로컬/원격 SHA 대조를 수행하고 전 과정을
`experiment_v1_3/results/git_push.log`에 남긴다.

```sh
bash ~/Desktop/MI/scripts/push_v1_3.sh
```

스크립트가 하는 일을 직접 확인하거나 수동으로 실행하려면 아래와 같다.

```sh
cd ~/Desktop/MI
git lfs install
git add .gitattributes .gitignore \
        interp_v1_3/cli.py interp_v1_3/smoke.py interp_v1_3/runtime.py interp_v1_3/persistence.py \
        tests_v1_3/test_contract.py \
        scripts/verify_v1_3_evidence.py scripts/select_v1_3_pilot_winner.py scripts/build_v1_3_pilot_colab.py \
        experiment_v1_3/P2_STATUS.md experiment_v1_3/README.md \
        experiment_v1_3/results/p2_pytest.txt experiment_v1_3/results/run_registry.csv \
        experiment_v1_3/results/colab_delivery.json \
        experiment_v1_3/smoke/cpu_release/ experiment_v1_3/notebooks/01_colab_pilot_8m.ipynb \
        experiment_v1_3/bundles/v1_3_pilot_bundle_v1.sha256 \
        experiment_v1_3/bundles/v1_3_pilot_bundle_v1.zip
git status            # 번들이 LFS pointer로 잡히는지, 무관한 변경이 섞이지 않았는지 확인
git commit -m "Implement the v1.3 runner and pass the six-cell CPU smoke"
git push origin main
git rev-parse HEAD && git rev-parse origin/main   # 두 SHA가 같아야 한다
```

push가 실패하면 이 단계를 완료로 표시하지 말고 사유를 기록한 뒤 해결해서 다시 push한다.
