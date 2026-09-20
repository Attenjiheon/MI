"""Frozen v1.4 behavior metrics, pair macros, and gates."""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import batch
from corpus.v1_4 import cells as pair_cells


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
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    model.eval()
    pairs = list(metadata(Path(path)))
    device = next(model.parameters()).device
    rows = []
    members = [(pair, member) for pair in pairs for member in ("first", "repeat")]
    # microbatch counts sequences, not pairs (including the microbatch=1 case).
    for offset in range(0, len(members), microbatch):
        chunk = members[offset : offset + microbatch]
        examples = [pair["origin" if member == "first" else "repeat"] for pair, member in chunk]
        ids, mask, _ = batch([example["token_ids"] for example in examples], device)
        logits = model(ids, mask)
        for index, (pair, member) in enumerate(chunk):
            example = examples[index]
            rid = pair["origin_target_read_id" if member == "first" else "repeat_target_read_id"]
            event = example["read_events"][rid]
            rows.append(
                _target_row(
                    logits[index, event["query_token_index"]], event,
                    pair_id=pair["pair_id"], sequence_id=example["sequence_id"],
                    member=member, cell_id=pair["cell_id"], operator=pair["operator"],
                    depth_bin=pair["depth_bin"], input_pattern=pair["input_pattern"],
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
        "coverage": valid_pair_coverage(pairs),
    }
    return {"passed": all(checks.values()), "checks": checks}


def valid_pair_coverage(pairs, quota=None):
    expected = {c.cell_id for c in pair_cells()}
    if quota is None:
        quota = pairs["pair_count"] // 42
    if quota <= 0 or pairs["pair_count"] != 42 * quota:
        return False
    group_counts = {"operator": {"NOT": 6, "AND": 12, "OR": 12, "XOR": 12},
                    "depth": {"1": 14, "2-3": 14, "4+": 14}, "answer": {"0": 21, "1": 21}}
    for member in ("first", "repeat"):
        values = pairs[member]["cell"]["cells"]
        if set(values) != expected or any(v["count"] != quota for v in values.values()):
            return False
        for metric in ("accuracy", "answer_ce"):
            macro = pairs[member]["cell"]["macro_" + metric]
            if not math.isfinite(macro) or not math.isclose(macro, sum(v[metric] for v in values.values()) / 42, abs_tol=1e-10):
                return False
        for group, counts in group_counts.items():
            values = pairs[member][group]["cells"]
            if set(values) != set(counts) or any(values[k]["count"] != n * quota for k, n in counts.items()):
                return False
    return True
