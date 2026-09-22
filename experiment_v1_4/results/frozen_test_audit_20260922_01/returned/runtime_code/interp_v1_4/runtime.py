"""v1.4 hashes, deterministic execution, and frozen input verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

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
    payload = f"20260920|experiment-spec-v1.4|{purpose}|{key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def tensor_digest(state_dict):
    """Order-independent byte digest of model tensors for bitwise anchor comparison."""
    value = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def checkpoint_tensor_digest(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return tensor_digest(payload["model"] if "model" in payload else payload)


def verify_inputs(root):
    root = Path(root)
    config_dir = root / "experiment_v1_4/configs"
    manifest = json.loads((config_dir / "config_set_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(root / name) != digest:
            raise ValueError("Changed config: " + name)
    combined = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if combined != manifest["combined_sha256"]:
        raise ValueError("Config manifest mismatch")
    run = json.loads((config_dir / "run.json").read_text())
    if sha(root / "experiment_v1_4/design_config.json") != run["design_config_sha256"]:
        raise ValueError("Changed design contract")
    if sha(root / "experiment_v1_4/corpus_rebuild.json") != run["corpus_rebuild_sha256"]:
        raise ValueError("Changed corpus rebuild provenance")
    if run["schema"] != "p3-run-v1.4" or run["effective_batch"] != 64:
        raise ValueError("Unsupported v1.4 run contract")
    if run["candidate"] != "deepwide12_read4" or run["architecture"] != "deepwide12" or run["loss"] != "read4":
        raise ValueError("Frozen v1.4 candidate changed")
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
    audit = json.loads((data / "postwrite_audit.json").read_text())
    if audit["status"] != "passed" or not audit.get("historical_reserved_registry_loaded"):
        raise ValueError("P1 audit missing")
    return {
        "corpus_manifest": sha(data / "manifest.json"),
        "configs": combined,
        "verified_files": len(corpus["files"]),
    }


__all__ = [
    "checkpoint_tensor_digest",
    "deterministic",
    "environment",
    "restore",
    "restore_rng",
    "rng_state",
    "save",
    "seed",
    "sha",
    "tensor_digest",
    "verified_copy",
    "verify_inputs",
]
