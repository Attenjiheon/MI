# P4 replication preparation r1

2026-09-21. Scope: execution delivery for seeds 1 and 2, not P4 completion.

The audited seed-0 completion SHA-256 is pinned in `interp_v1_4/replication.py`.
All evidence referenced by that completion is verified before a replication run.
Frozen config/corpus hashes, final budget/cursor, and seed-0 microbatch 16 must match.
The original P3 config and numerical/model/evaluation modules remain byte-identical.
Changes are restricted to CLI stage/seed authorization, evidence verification and delivery.
P3 r3 bundle and returned evidence remain preserved at their original paths.

`--stage p3` still permits only seed 0; `--stage p4` permits only seeds 1 and 2.
The same training loop, milestone selector and one-time gate are used. Resume hashes
bind the stage, seed and audited authorization. Debug runs are explicitly marked and
cannot serve as production evidence. The verifier independently reconstructs the
seed's initialization and checks its tensor digest, without reevaluating gate or test.

Two notebooks share one self-contained input ZIP. Each seed has its own Drive path.
Seed-0's recorded primary package versions and full version constraints are included,
with the original lock retained. Current GPU smoke and per-session environment records
remain mandatory; different environments are not represented as bitwise GPU replication.
A completed persistent run can be recovered for export without retraining or reopening gate.

Validation commands (local Python: `/opt/anaconda3/bin/python`):

- `-m pytest tests_v1_4 -q --junitxml=experiment_v1_4/results/p4_preparation_r1/tests.xml`
- `-m interp_v1_4.smoke --device cpu --output experiment_v1_4/smoke/cpu_p4_r1`
- `scripts/check_v1_4_resume.py --stage p4 --lm-seed 1 --output experiment_v1_4/smoke/resume_p4_seed1_r1`
- `scripts/check_v1_4_resume.py --stage p4 --lm-seed 2 --output experiment_v1_4/smoke/resume_p4_seed2_r1`
- `scripts/build_v1_4_p4_colab.py`

Final outcomes and file hashes are in `verification.json` and `delivery.json`.
The initial `.venv-p2/bin/python` command was unavailable (see `tests.log`); testing
used the existing Anaconda environment also used for prior CPU evidence.

Outstanding: actual CUDA smoke/64M training/one-time gate for both seeds; local return
audits; all-seed validation/checkpoint freeze; separate frozen-test delivery and execution
for all three trained seeds including failures; final return audit, behavior aggregation,
passing-model list and phase completion Git update. No model test scores were observed.

## Validated local outcome

33 tests passed; current frozen-input CPU smoke passed; each seed full/resumed run
and both evidence audits passed; cross-seed resume rejected; independent initial model
tensor digests differ. ZIP 421-member CRC/SHA-256 and all 14 notebook code cells passed.
Bundle size 1,575,034,876 bytes; SHA-256 `75a435a7da71fcd3cf954cc3c12cc7d0c66db693aacd37ffb4105fc97b092bb3`.
P4 remains incomplete. Production tokens executed in this preparation: 0.

Disk exhaustion briefly blocked final status writes and approval-review initialization.
Only byte-identical duplicate .pt files in the two newly created P4 debug resume trees
were hardlinked after full SHA-256 checks (912,109,612 bytes reclaimed). All artifact
paths and contents are retained; prior experiment artifacts were not removed.

## After user-authorized disk cleanup

The later user-authorized cleanup deleted completed debug checkpoints while retaining
their hashes and successful validation records. Current availability is documented in
`maintenance/disk_cleanup_20260921/REPORT.md`; the preceding preservation statements
describe the earlier preparation state. No debug checkpoint was a production input.
The P4 ZIP, notebooks, tested code and preparation evidence were rechecked successfully;
see `post_cleanup_verification.json`. Production GPU execution remains pending.
