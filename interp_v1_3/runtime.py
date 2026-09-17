"""v1.3 hashes, deterministic execution, and frozen input verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from interp_v1_2.runtime import (
    deterministic,
    environment,
    restore,
    restore_rng,
    rng_state,
    save,
    sha,
    verified_copy,
)


def seed(purpose, key):
    payload = f"20260917|experiment-spec-v1.3|{purpose}|{key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def verify_inputs(root):
    root = Path(root)
    config_dir = root / "experiment_v1_3/configs"
    manifest = json.loads((config_dir / "config_set_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(root / name) != digest:
            raise ValueError("Changed config: " + name)
    combined = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if combined != manifest["combined_sha256"]:
        raise ValueError("Config manifest mismatch")
    run = json.loads((config_dir / "run.json").read_text())
    if run["schema"] != "pilot-run-v1.3" or run["effective_batch"] != 64:
        raise ValueError("Unsupported v1.3 run contract")
    if run["pilot_cells"] != [
        "base4_uniform",
        "base4_read4",
        "wide4_uniform",
        "wide4_read4",
        "deep8_uniform",
        "deep8_read4",
    ]:
        raise ValueError("Frozen pilot cells changed")
    data = root / run["data_root"]
    for name, digest in run["data_hashes"].items():
        if sha(data / name) != digest:
            raise ValueError("Changed data contract: " + name)
    corpus = json.loads((data / "manifest.json").read_text())
    for name, info in corpus["files"].items():
        path = data / name
        if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
            raise ValueError("Changed corpus: " + name)
    if corpus["frozen_training"] != run["frozen_training"]:
        raise ValueError("Frozen train cursor mismatch")
    if json.loads((data / "postwrite_audit.json").read_text())["status"] != "passed":
        raise ValueError("P1 audit missing")
    return {
        "corpus_manifest": sha(data / "manifest.json"),
        "configs": combined,
        "verified_files": len(corpus["files"]),
    }


__all__ = [
    "deterministic",
    "environment",
    "restore",
    "restore_rng",
    "rng_state",
    "save",
    "seed",
    "sha",
    "verified_copy",
    "verify_inputs",
]
