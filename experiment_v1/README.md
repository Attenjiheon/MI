# P2 implementation

Run commands from the project root. The local environment is `.venv-p2`; exact
versions and runtime details are recorded with each passing smoke result.

```bash
.venv-p2/bin/python -m pytest tests -q
.venv-p2/bin/python -m interp.smoke --device cpu --output experiment_v1/smoke/cpu_new_run
.venv-p2/bin/python -m interp.cli --help
```

Every output directory must be new. Debug results are explicitly marked and cannot
be loaded by production CLI commands without `--debug`. The immutable corpus and
P0 configs are checked against the recorded SHA-256 values before CLI execution.

| Spec interface | Executable interface |
|---|---|
| smoke_test.py | `python -m interp.smoke --device cpu\|cuda --output PATH` |
| train_lm.py | `python -m interp.cli train_lm --device cuda --output PATH` |
| eval_behavior.py | `python -m interp.cli eval_behavior --checkpoint LM --input METADATA --output PATH` |
| cache_activations.py | `python -m interp.cli cache_activations --checkpoint LM --input METADATA --positions POSITIONS_JSON --split train\|val\|test --output PATH` |
| fit_probes.py | `python -m interp.cli fit_probes --input TRAIN_NPZ --validation VAL_NPZ --output PATH` |
| train_dictionary.py | `python -m interp.cli train_dictionary --input TRAIN_CACHE --validation VAL_CACHE --kind sae\|transcoder --k 4\|16 --output PATH` |
| evaluate_dictionary.py | `python -m interp.cli evaluate_dictionary --checkpoint DICTIONARY --input CACHE --output PATH` |
| run_patching.py | `python -m interp.cli run_patching --checkpoint LM --input FROZEN_PATCH_PLAN --output PATH` |
| aggregate_results.py | `python -m interp.cli aggregate_results --input RUNS_DIRECTORY --output PATH` |

Use `--device cuda` on the GPU, `--seed` for the LM identity, `--sparse-seed` for
dictionary initialization repeats, and `--resume` for a trusted saved checkpoint.
`--persistent-dir` copies training checkpoints through a temporary file and verifies
the copied checksum. Train sequence cursor is a global row offset in sorted shard
order; no shuffle or repetition is performed.

P2 passed on both environments: local CPU (`experiment_v1/smoke/cpu_attempt_02/`)
and Colab GPU (`experiment_v1/smoke/colab_gpu_01/`, Tesla T4). Locks and runtime
records are in `experiment_v1/environment/`; the status document is `experiment_v1/P2_STATUS.md`.
The GPU bundle `bundles/p2_colab_bundle_v1.zip` does not contain
`experiment_v1/results/run_registry.csv`, which `tests/test_p0_config.py` reads, so
the Colab notebook uploads that file separately. Do not treat the bundle alone as a
complete test workspace.

P2 implements and checks the underlying training/cache/probe/dictionary/patch
interfaces. P3 still must adjudicate the selected LM checkpoint on all three 512
sequence diagnostics, record extension or freeze decisions, and implement the
decision-specific continuation. `train_lm` currently stops at the initial 1M pilot
boundary. Do not use it as a 3M extension or seed 1/2 frozen-budget runner until P3
provides the decision contract. No main-experiment training has been executed here.

The evaluation entry points provide basic full-vocabulary behavior and normalized
dictionary fidelity, and apply an already frozen explicit patch plan. Full strata,
baselines, semantic metric tables, causal matching/control construction and cluster
bootstrap reporting must be integrated and verified under P4–P11 before those
phases are declared complete. The aggregate command collects manifests; it does not
declare experimental completion or generate inferential conclusions.

Cache `.pt` files contain `keys`, metadata-only `labels`, and float32 `tensors` under
the prescribed hook names. Probe `.npz` input contains `x`, `y`, `sequence_id`, and
scalar `classes`. Patch plans contain `checkpoint_sha256`, `prefix_token_ids`
(target answer/suffix excluded), `hook`, `positions`, and replacement `values`.
Torch checkpoint files are trusted local artifacts and use full pickle loading to
restore Python and NumPy RNG state. Do not load an untrusted downloaded checkpoint.

`probe.py` follows 03 §7.2: across prefix sizes, balanced-accuracy ties favor smaller
feature count before the within-count §7.1 rules. The P0 probe config lists feature
count last in a combined order; this discrepancy is recorded in P2_STATUS.md and
the explicit detailed specification takes precedence. P0 config bytes are preserved.
