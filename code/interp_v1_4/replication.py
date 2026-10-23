"""P4 authorization from the locally audited, immutable P3 completion record."""
import json

from .runtime import sha

AUDIT = "experiment_v1_4/results/p3_audit_20260921_01"
COMPLETION_SHA256 = "3634653751ca64d655ce71a60aa79ac3709803b824cfcd8194754342f6f8d6da"


def validate_seed(stage, lm_seed):
    if (stage == "p3" and lm_seed == 0) or (stage == "p4" and lm_seed in (1, 2)):
        return
    raise ValueError("P3 permits seed 0 only; P4 permits seeds 1 and 2 only")


def replication_authorization(root, contract, hashes, microbatch, *, debug=False):
    folder = root / AUDIT
    completion_path = folder / "completion.json"
    if sha(completion_path) != COMPLETION_SHA256:
        raise ValueError("P3 audited completion changed")
    completion = json.loads(completion_path.read_text())
    for name, digest in completion["evidence_sha256"].items():
        if sha(folder / name) != digest:
            raise ValueError("P3 audit evidence changed: " + name)
    frozen = json.loads((folder / "frozen_seed0.json").read_text())
    if completion["status"] != "passed" or completion["gate_passed"] is not True or frozen["status"] != "passed":
        raise ValueError("An audited passing seed 0 is required")
    for key in ("configs", "corpus_manifest"):
        if hashes[key] != frozen["hashes"][key]:
            raise ValueError("Replication changed frozen " + key)
    expected = contract["expected"]
    actual = frozen["final_state"]
    if (actual["update"], actual["next_data_cursor"], actual["prediction_tokens"]) != (
        expected["update"], expected["cursor"], expected["prediction_tokens"]
    ):
        raise ValueError("Replication budget differs from seed 0")
    if contract["budget"] != frozen["nominal_budget"] or contract["config"]["candidate"] != frozen["candidate"]:
        raise ValueError("Replication candidate/budget changed")
    if not debug and microbatch != frozen["microbatch"]:
        raise ValueError("P4 must preserve seed 0 microbatch 16")
    # Only orchestration may change; numerical/model/reporting dependencies stay byte-identical.
    for name, digest in frozen["hashes"]["code"].items():
        if name != "interp_v1_4/cli.py" and sha(root / name) != digest:
            raise ValueError("Frozen numerical dependency changed: " + name)
    return dict(
        completion_sha256=COMPLETION_SHA256,
        frozen_seed0_sha256=sha(folder / "frozen_seed0.json"),
        selected_seed0_checkpoint_sha256=frozen["selected_checkpoint_sha256"],
        debug_only=debug,
    )
