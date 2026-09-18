"""Freeze executable v1.3 configs against the audited CPU corpus."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus.generate import file_hash, write_json


def run():
    design_path = ROOT / "experiment_v1_3/design_config.json"
    design = json.loads(design_path.read_text())
    data = ROOT / design["data"]["root"]
    audit = json.loads((data / "postwrite_audit.json").read_text())
    corpus = json.loads((data / "manifest.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("CPU corpus audit has not passed")
    out = ROOT / "experiment_v1_3/configs"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Executable config set is immutable once created")
    write_json(
        out / "data.json",
        {
            "schema": "data-contract-v1.3",
            "root": design["data"]["root"],
            "generation_order": design["data"]["generation_order"],
            "seed_namespace": design["data"]["seed_namespace"],
            "frozen_training": corpus["frozen_training"],
            "manifest_sha256": file_hash(data / "manifest.json"),
            "audit_sha256": file_hash(data / "postwrite_audit.json"),
        },
    )
    write_json(
        out / "transformer.json",
        {
            "schema": "transformer-contract-v1.3",
            "architectures": design["architectures"],
            "losses": design["losses"],
            "training": design["training"],
            "pilot": design["pilot"],
            "checkpoint_selection": design["checkpoint_selection"],
        },
    )
    write_json(
        out / "evaluation.json",
        {
            "schema": "evaluation-contract-v1.3",
            "splits": design["evaluation_splits"],
            "gate": design["gate"],
            "test_policy": design["test_policy"],
        },
    )
    run_config = {
        "schema": "pilot-run-v1.3",
        "data_root": design["data"]["root"],
        "data_hashes": {
            name: file_hash(data / name)
            for name in ("manifest.json", "cpu_validation.json", "postwrite_audit.json", "corpus_statistics.json")
        },
        "design_config_sha256": file_hash(design_path),
        "pilot_cells": design["pilot"]["cells"],
        "pilot_seed": design["pilot"]["lm_seeds"][0],
        "pilot_nominal_prediction_tokens": design["pilot"]["nominal_prediction_tokens"],
        "pilot_milestones": design["pilot"]["milestones_prediction_tokens"],
        "effective_batch": design["training"]["effective_batch_sequences"],
        "dtype": design["training"]["dtype"],
        "frozen_training": corpus["frozen_training"],
        "anchor": {
            "cell": design["pilot"]["anchor_cell"],
            "environment_id": design["pilot"]["anchor_environment_id"],
            "microbatch": design["pilot"]["anchor_microbatch_sequences"],
            "v1_2_checkpoint_sha256": design["predecessor"]["selected_checkpoint_sha256"],
        },
    }
    write_json(out / "run.json", run_config)
    files = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in sorted(out.glob("*.json"))
        if path.name != "config_set_manifest.json"
    }
    combined = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(
        out / "config_set_manifest.json",
        {
            "schema": "config-set-v1.3",
            "status": "frozen",
            "design_config_sha256": file_hash(design_path),
            "files": files,
            "combined_sha256": combined,
        },
    )
    print(json.dumps({"status": "frozen", "combined_sha256": combined}, indent=2))


if __name__ == "__main__":
    run()
