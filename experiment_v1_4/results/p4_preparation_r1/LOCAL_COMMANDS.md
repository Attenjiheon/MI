# Local preparation commands and boundaries

Executed with `/opt/anaconda3/bin/python` after the documented `.venv-p2` path was absent.
See `tests.log` for that failed launch; `tests_anaconda.log` is the actual passing test run.

1. `-m pytest tests_v1_4 -q --junitxml=experiment_v1_4/results/p4_preparation_r1/tests.xml`
2. `-m interp_v1_4.smoke --device cpu --output experiment_v1_4/smoke/cpu_p4_r1`
3. `scripts/check_v1_4_resume.py --stage p4 --lm-seed 1 --output experiment_v1_4/smoke/resume_p4_seed1_r1`
4. `scripts/check_v1_4_resume.py --stage p4 --lm-seed 2 --output experiment_v1_4/smoke/resume_p4_seed2_r1`
5. `scripts/build_v1_4_p4_colab.py`
6. `scripts/verify_v1_4_p4_preparation.py`

Each production seed still requires 64,005,751 tokens / 8,399 updates, independent
selection and a one-time gate. The local 3-update debug runs do not count toward that budget.
All-seed freeze and final test delivery follow audited returns, not notebook preparation.

No P4 completion commit or push is claimed: the phase gates and production returns remain pending.
