# Package Verification and Scope of Reproduction

## Checks you can run right away

This folder is a submission package. You can review the results with it without a separate project path or a GPU.

```bash
python3 pipeline.py verify
python3 pipeline.py summary
```

`verify` checks the following:

- the SHA-256 of every included file;
- that the decompressed result tables match the frozen P11 hashes;
- that the registry contains 64 READ runs and 12 Update runs;
- the source CSV rows, filters, and aggregations behind each number in the paper.

`summary` first runs these checks and then recomputes the main table in the README. Neither command runs training, inference, or feature selection.

| Folder | Contents |
|---|---|
| `code/` | The final v1.4 implementation, the shared functions that it needs, the frozen configs, and tests that check the key numbers |
| `results/` | All final aggregated CSVs, compressed losslessly with gzip; every control and negative result is included |
| `figures/` | The figure that compares initializations, used in the main text |
| `provenance/` | The original completion decision, verification records, input hashes, sources for each number, and records of moved files |

## Checking the code

Install the dependencies below in a Python environment. Then run the tests from the package folder.

```bash
python3 -m pip install -r code/requirements-interp.txt
cd code
python3 -m pytest -q tests
```

The package versions for each frozen GPU or CPU environment are identified by path and hash in `results/environment_index.csv.gz`. The dependency list only makes installation easier. It is not a lock file, so it does not recreate the earlier CUDA environment exactly. The results of the package validation are recorded in `provenance/package_validation.json`.

The submitted code originally depended on shared code in `archive.legacy.interp_v1_2`. This code is now included in `interp_v1_4/_support`. The changes to import paths and the functions that had to be moved are recorded in `provenance/source_map.json`, together with hashes of the original and packaged files. This package does not claim that its code is byte-for-byte identical to the code used in the original GPU runs. The original paths and hashes in the frozen configs are left unchanged, because they identify the original files.

## Rerunning the full experiment

This review package does not include the large corpus, the activation caches, the checkpoints, the raw per-pair prediction arrays, or the ZIP files returned from Colab. To reproduce the training and the original audits, you need to obtain the following inputs from the separately archived materials.

1. The active `data/language_v1_4/rebuild_01/` corpus and its manifest, and the earlier registry of train prefixes that is used to prevent duplicates.
2. The checkpoints of the 3 frozen LMs, and the records of their selection and behavioral gate.
3. The P5 READ cache, the preprocessing statistics for each layer, and the selected SAE and TC checkpoints.
4. The raw semantic, substitution, and causal evaluations, the compressed archives saved at each stage, and the environment lock from that time.

The inputs are identified in `results/reproduction_inputs.csv.gz`, `provenance/input_manifest.json`, and the contracts in `code/experiment_v1_4/`. These paths **refer to locations in the original repository**. They do not mean that every file is inside this package. Some non-selected and debug checkpoints, and the bytes of ZIP files delivered earlier, were deleted during an earlier cleanup. Therefore, they cannot be fully recovered.

The training code still contains its checks on the inputs and the current environment, as well as its smoke gate. If you force a run without supplying the missing inputs, or if you turn off the checks against past hashes, the result is not a reproduction of the same experiment. Running the full training on Colab would also require the original notebooks for each stage, the input bundles with their upload checksums, the environment records, and the procedures for saving, resuming, and downloading. This package does not contain any Colab work that you need to run right away.

## What is included and what is excluded

The package includes the code, configs, and numbers that are needed to verify the methods and results of the final v1.4 experiment. It excludes separate experiments from earlier versions, intermediate delivery bundles, resume logs, records of storage cleanup, and repeated status documents. The original MI working folder and the existing evidence are archived separately. Earlier architecture and budget pilots failed the behavioral gate. These failures are not treated as failed v1.4 seeds that were replaced, and the final models and analyses were not selected again.
