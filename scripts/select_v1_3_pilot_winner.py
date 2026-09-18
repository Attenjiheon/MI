"""Apply the frozen §4.3 promotion rule to the six verified pilot runs. No new criteria.

Each cell contributes its 8M milestone `select/` metrics only. A cell counts as validated
only when its own evidence verification passed. If no cell is eligible, v1.3 stops before
confirmatory training instead of promoting a best-of-six.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interp_v1_3.behavior import select_pilot
from interp_v1_3.persistence import atomic_json


def candidate(run, verification):
    result = json.loads((Path(run) / "result.json").read_text())
    if result["stage"] != "pilot" or result["debug_only"] or result["lm_seed"] != 0:
        raise ValueError("Not a production pilot seed 0 run: " + str(run))
    budget = result["nominal_budget"]
    milestone = result["milestones"][str(budget)]["current"]
    if milestone["checkpoint"] != result["selected"]["checkpoint"]:
        raise ValueError("Pilot must compare the final milestone checkpoint: " + str(run))
    audit = json.loads(Path(verification).read_text()) if verification else None
    validated = bool(audit and audit["status"] == "passed" and audit["cell"] == result["cell"])
    return result["cell"], dict(
        validated=validated,
        run=str(run),
        verification=str(verification) if verification else None,
        checkpoint=milestone["checkpoint"],
        checkpoint_sha256=result["selected"]["sha256"],
        prediction_tokens=milestone["prediction_tokens"],
        update=milestone["update"],
        select_general=milestone["validation"]["general"],
        select_pairs=milestone["validation"]["pairs"],
    )


def report(pairs):
    design = json.loads((ROOT / "experiment_v1_3/design_config.json").read_text())
    candidates = dict(candidate(run, verification) for run, verification in pairs)
    decision = select_pilot(candidates, design)
    table = {
        cell: dict(
            validated=value["validated"],
            update=value["update"],
            prediction_tokens=value["prediction_tokens"],
            select_general_answer_ce=value["select_general"]["answer_ce"],
            select_general_accuracy=value["select_general"]["accuracy"],
            first_macro_answer_ce=value["select_pairs"]["first"]["cell"]["macro_answer_ce"],
            first_macro_accuracy=value["select_pairs"]["first"]["cell"]["macro_accuracy"],
            repeat_macro_accuracy=value["select_pairs"]["repeat"]["cell"]["macro_accuracy"],
            paired_answer_ce_gap=value["select_pairs"]["paired_gap"]["answer_ce_first_minus_repeat"],
        )
        for cell, value in candidates.items()
    }
    return dict(
        schema="pilot-selection-v1.3",
        primary_metric=design["pilot"]["primary_metric"],
        eligibility=design["pilot"]["eligibility"],
        anchor_cell=design["pilot"]["anchor_cell"],
        decision=decision["decision"],
        winner=decision["winner"],
        next_step="confirm_32m_seed_0" if decision["decision"] == "promote" else design["pilot"]["no_eligible_cell_action"],
        eligibility_by_cell=decision["eligibility"],
        cells=table,
        sources={cell: dict(run=value["run"], verification=value["verification"], checkpoint_sha256=value["checkpoint_sha256"]) for cell, value in candidates.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("RUN_DIR", "VERIFICATION_JSON"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = report(args.run)
    atomic_json(args.output, value, immutable=True)
    print(json.dumps({k: value[k] for k in ("decision", "winner", "next_step")}, indent=2))
