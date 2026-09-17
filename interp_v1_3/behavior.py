"""Frozen v1.3 behavior metrics, pair macros, gates, and pilot selection."""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import batch


LEGACY_DIAGNOSTICS = ("other_variable", "repeated_update", "first_read_after_set")


def metadata(path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def _target_row(logit, event, **extra):
    answer = event["answer"]
    probabilities = logit.softmax(-1)
    return {
        **extra,
        "answer": answer,
        "answer_ce": float(-logit.log_softmax(-1)[13 + answer]),
        "correct": int(logit.argmax() == 13 + answer),
        "binary_correct": int(logit[13:15].argmax() == answer),
        "bit_mass": float(probabilities[13:15].sum()),
    }


@torch.no_grad()
def evaluate_records(model, records, *, target_only=False, microbatch=16):
    model.eval()
    rows = []
    records = list(records)
    device = next(model.parameters()).device
    for offset in range(0, len(records), microbatch):
        examples = records[offset : offset + microbatch]
        ids, mask, _ = batch([example["token_ids"] for example in examples], device)
        logits = model(ids, mask)
        for row, example in enumerate(examples):
            targets = set(example["target_read_ids"])
            for event in example["read_events"]:
                if target_only and event["read_id"] not in targets:
                    continue
                rows.append(
                    _target_row(
                        logits[row, event["query_token_index"]],
                        event,
                        sequence_id=example["sequence_id"],
                        read_id=event["read_id"],
                    )
                )
    if not rows:
        raise ValueError("No READ answer targets")
    return summarize(rows)


def summarize(rows):
    return {
        "answer_count": len(rows),
        "sequence_count": len({row["sequence_id"] for row in rows}),
        "answer_ce": sum(row["answer_ce"] for row in rows) / len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "binary_accuracy": sum(row["binary_correct"] for row in rows) / len(rows),
        "bit_mass": sum(row["bit_mass"] for row in rows) / len(rows),
        "rows": rows,
    }


@torch.no_grad()
def evaluate_first_repeat(model, path, *, microbatch=16):
    pairs = list(metadata(Path(path)))
    device = next(model.parameters()).device
    rows = []
    for offset in range(0, len(pairs), microbatch):
        chunk = pairs[offset : offset + microbatch]
        examples = [member for pair in chunk for member in (pair["origin"], pair["repeat"])]
        ids, mask, _ = batch([example["token_ids"] for example in examples], device)
        logits = model(ids, mask)
        for index, pair in enumerate(chunk):
            for member_offset, member in enumerate(("first", "repeat")):
                example = pair["origin" if member == "first" else "repeat"]
                rid = pair["origin_target_read_id" if member == "first" else "repeat_target_read_id"]
                event = example["read_events"][rid]
                rows.append(
                    _target_row(
                        logits[2 * index + member_offset, event["query_token_index"]],
                        event,
                        pair_id=pair["pair_id"],
                        sequence_id=example["sequence_id"],
                        member=member,
                        cell_id=pair["cell_id"],
                        operator=pair["operator"],
                        depth_bin=pair["depth_bin"],
                        input_pattern=pair["input_pattern"],
                    )
                )
    return pair_summary(rows)


def _macro(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    cells = {
        value: {
            "count": len(group),
            "answer_ce": sum(row["answer_ce"] for row in group) / len(group),
            "accuracy": sum(row["correct"] for row in group) / len(group),
        }
        for value, group in sorted(groups.items())
    }
    return {
        "cells": cells,
        "macro_answer_ce": sum(cell["answer_ce"] for cell in cells.values()) / len(cells),
        "macro_accuracy": sum(cell["accuracy"] for cell in cells.values()) / len(cells),
    }


def pair_summary(rows):
    first = [row for row in rows if row["member"] == "first"]
    repeat = [row for row in rows if row["member"] == "repeat"]
    if not first or len(first) != len(repeat):
        raise ValueError("Incomplete first/repeat rows")
    paired = defaultdict(dict)
    for row in rows:
        paired[row["pair_id"]][row["member"]] = row
    if any(set(value) != {"first", "repeat"} for value in paired.values()):
        raise ValueError("Broken pair join")
    gaps = [
        {
            "pair_id": key,
            "ce_first_minus_repeat": value["first"]["answer_ce"] - value["repeat"]["answer_ce"],
            "accuracy_first_minus_repeat": value["first"]["correct"] - value["repeat"]["correct"],
        }
        for key, value in paired.items()
    ]
    return {
        "pair_count": len(first),
        "first": {
            "cell": _macro(first, "cell_id"),
            "operator": _macro(first, "operator"),
            "depth": _macro(first, "depth_bin"),
            "answer": _macro(first, "answer"),
            "micro": summarize(first),
        },
        "repeat": {
            "cell": _macro(repeat, "cell_id"),
            "operator": _macro(repeat, "operator"),
            "depth": _macro(repeat, "depth_bin"),
            "answer": _macro(repeat, "answer"),
            "micro": summarize(repeat),
        },
        "paired_gap": {
            "answer_ce_first_minus_repeat": sum(row["ce_first_minus_repeat"] for row in gaps) / len(gaps),
            "accuracy_first_minus_repeat": sum(row["accuracy_first_minus_repeat"] for row in gaps) / len(gaps),
            "rows": gaps,
        },
        "rows": rows,
    }


def without_rows(value):
    if isinstance(value, dict):
        return {key: without_rows(item) for key, item in value.items() if key != "rows"}
    if isinstance(value, list):
        return [without_rows(item) for item in value]
    return value


def gate(general, diagnostics, pairs, thresholds):
    if set(diagnostics) != set(LEGACY_DIAGNOSTICS):
        raise ValueError("Missing legacy diagnostic")
    checks = {
        "general": general["accuracy"] >= thresholds["general_full_vocabulary_accuracy_min"],
        "legacy": all(value["accuracy"] >= thresholds["each_legacy_diagnostic_accuracy_min"] for value in diagnostics.values()),
        "first_cell_macro": pairs["first"]["cell"]["macro_accuracy"] >= thresholds["first_member_42_cell_macro_accuracy_min"],
        "first_operator": all(cell["accuracy"] >= thresholds["each_first_member_operator_macro_accuracy_min"] for cell in pairs["first"]["operator"]["cells"].values()),
        "first_depth": all(cell["accuracy"] >= thresholds["each_first_member_depth_macro_accuracy_min"] for cell in pairs["first"]["depth"]["cells"].values()),
        "first_answer": all(cell["accuracy"] >= thresholds["each_first_member_answer_accuracy_min"] for cell in pairs["first"]["answer"]["cells"].values()),
        "repeat_cell_macro": pairs["repeat"]["cell"]["macro_accuracy"] >= thresholds["repeat_member_42_cell_macro_accuracy_min"],
        "coverage": len(pairs["first"]["cell"]["cells"]) == 42 and len(pairs["repeat"]["cell"]["cells"]) == 42,
    }
    return {"passed": all(checks.values()), "checks": checks}


def select_pilot(candidates, design):
    ids = design["pilot"]["cells"]
    if set(candidates) != set(ids):
        raise ValueError("Exactly six frozen pilot cells are required")
    anchor = candidates[design["pilot"]["anchor_cell"]]
    eligibility = design["pilot"]["eligibility"]
    anchor_accuracy = anchor["select_general"]["accuracy"]
    anchor_ce = anchor["select_pairs"]["first"]["cell"]["macro_answer_ce"]
    anchor_pair_accuracy = anchor["select_pairs"]["first"]["cell"]["macro_accuracy"]
    eligible = []
    decisions = {}
    for candidate_id in ids:
        value = candidates[candidate_id]
        finite = all(
            math.isfinite(metric)
            for metric in (
                value["select_general"]["answer_ce"],
                value["select_pairs"]["first"]["cell"]["macro_answer_ce"],
            )
        )
        general_ok = value["select_general"]["accuracy"] >= anchor_accuracy - eligibility["general_accuracy_max_regression_absolute"]
        improved = (
            value["select_pairs"]["first"]["cell"]["macro_answer_ce"] <= anchor_ce - eligibility["first_macro_ce_min_improvement_nats"]
            or value["select_pairs"]["first"]["cell"]["macro_accuracy"] >= anchor_pair_accuracy + eligibility["or_first_macro_accuracy_min_improvement_absolute"]
        )
        ok = finite and value.get("validated", False) and general_ok and improved
        decisions[candidate_id] = {"eligible": ok, "finite": finite, "general_ok": general_ok, "improved": improved}
        if ok:
            eligible.append(candidate_id)
    if not eligible:
        return {"decision": "pilot_failure", "winner": None, "eligibility": decisions}

    architecture_order = {entry["id"]: index for index, entry in enumerate(design["architectures"])}
    parameter_count = {entry["id"]: entry["expected_trainable_parameters"] for entry in design["architectures"]}

    minimum = min(candidates[candidate_id]["select_pairs"]["first"]["cell"]["macro_answer_ce"] for candidate_id in eligible)
    contenders = [
        candidate_id
        for candidate_id in eligible
        if candidates[candidate_id]["select_pairs"]["first"]["cell"]["macro_answer_ce"] - minimum <= 1e-4
    ]

    def key(candidate_id):
        value = candidates[candidate_id]
        architecture, loss = candidate_id.rsplit("_", 1)
        return (
            value["select_general"]["answer_ce"],
            0 if loss == "uniform" else 1,
            parameter_count[architecture],
            architecture_order[architecture],
        )

    winner = min(contenders, key=key)
    return {"decision": "promote", "winner": winner, "eligibility": decisions}
