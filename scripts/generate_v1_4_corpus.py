"""Build the immutable v1.4 reserved splits and 64M ordered train stream."""
from __future__ import annotations

import copy
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus.generate_v1_4_base import Builder, Stats, dump, file_hash, self_test, write_json
from corpus.language import OPS, PATTERNS, digest
from corpus.replay import replay, validate
from corpus.v1_4 import cells, rng as v14_rng, sample_pair, seed as v14_seed


MAX_ATTEMPTS = 100_000


def metadata_rows(path: Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def load_prior_registry(base: Path) -> tuple[set[str], set[str]]:
    hashes = set((base / "all_sequence_hashes.txt").read_text().splitlines())
    # The registry also contains v1.0-v1.2 reservations absent from v1.3 files.
    # Loading only actual v1.3 rows silently drops those historical prefixes.
    registry = base / "reserved_prefix_hashes.txt"
    prefixes = set(registry.read_text().splitlines()) if registry.exists() else set()
    predecessor = {"language_v1_3": "language_v1_2", "language_v1_2": "language_v1"}.get(base.name)
    if predecessor:
        older_hashes, older_prefixes = load_prior_registry(base.parent / predecessor)
        hashes.update(older_hashes)
        prefixes.update(older_prefixes)
    for path in sorted(base.rglob("*.jsonl.gz")):
        if (
            "provenance" in path.parts
            or "first_repeat" in path.parts
            or ("causal_pairs" in path.parts and not path.name.endswith(".origins.jsonl.gz"))
        ):
            continue
        for example in metadata_rows(path):
            for event in example["read_events"]:
                prefixes.add(digest(example["token_ids"][: event["answer_token_index"]]))
    for path in sorted((base / "first_repeat").glob("*.jsonl.gz")):
        for pair in metadata_rows(path):
            for example in (pair["origin"], pair["repeat"]):
                for event in example["read_events"]:
                    prefixes.add(digest(example["token_ids"][: event["answer_token_index"]]))
    for path in sorted((base / "causal_pairs").glob("*.jsonl.gz")):
        if path.name.endswith(".origins.jsonl.gz"):
            continue
        with gzip.open(path, "rt") as handle:
            for line in handle:
                pair = json.loads(line)
                prefixes.update(pair["prefix_hashes"])
    return hashes, prefixes


class V14Builder(Builder):
    def __init__(self, out: Path, hashes: set[str], prefixes: set[str]):
        super().__init__(out)
        self.hashes = hashes
        self.prefixes = prefixes
        self.causal_prefixes = set()

    def rng(self, purpose, index):
        return v14_rng(purpose, index)

    def accept(self, example, causal=False):
        if example["canonical_hash"] in self.hashes:
            return "duplicate_sequence"
        read_prefixes = [digest(example["token_ids"][: r["answer_token_index"]]) for r in example["read_events"]]
        if len(read_prefixes) != len(set(read_prefixes)) or any(value in self.prefixes for value in read_prefixes):
            return "read_prefix_collision"
        validate(example)
        self.validated += 1
        self.hashes.add(example["canonical_hash"])
        self.prefixes.update(read_prefixes)
        return None

    def reserve_pair(self, pair: dict) -> str | None:
        examples = (pair["origin"], pair["repeat"])
        hashes = [e["canonical_hash"] for e in examples]
        if len(set(hashes)) != 2 or any(value in self.hashes for value in hashes):
            return "duplicate_sequence"
        prefixes = [digest(e["token_ids"][: r["answer_token_index"]]) for e in examples for r in e["read_events"]]
        # The repeat shares pre-insertion READ prefixes with its paired origin by
        # construction. Permit only those within-pair duplicates; reject history.
        if any(value in self.prefixes for value in prefixes):
            return "read_prefix_collision"
        for example in examples:
            validate(example)
        self.validated += 2
        self.hashes.update(hashes)
        self.prefixes.update(prefixes)
        return None

    def first_repeat(self, split: str, quota: int):
        path = self.out / f"first_repeat/{split}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {}
        total_attempts = 0
        answer_support = Counter()
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as handle:
            for cell in cells():
                generator = v14_rng(f"{split}_first_repeat", cell.index)
                accepted = attempts = 0
                rejects = Counter()
                while accepted < quota:
                    pair, used, local_rejects = sample_pair(generator, split, cell, MAX_ATTEMPTS)
                    attempts += used
                    rejects.update(local_rejects)
                    reason = self.reserve_pair(pair)
                    if reason:
                        rejects[reason] += 1
                        continue
                    handle.write(dump(pair) + "\n")
                    accepted += 1
                    answer_support[pair["answer"]] += 1
                total_attempts += attempts
                summary[cell.cell_id] = {
                    "cell_index": cell.index,
                    "operator": cell.operator,
                    "input_pattern": cell.input_pattern,
                    "depth_bin": cell.depth,
                    "quota": quota,
                    "accepted_pairs": accepted,
                    "independent_origins": accepted,
                    "attempts": attempts,
                    "rejections": dict(rejects),
                    "rng_seed": v14_seed(f"{split}_first_repeat", cell.index),
                    "rng_final_state": generator.bit_generator.state,
                }
                print(f"first_repeat/{split} {cell.cell_id}: {accepted} pairs / {attempts} attempts", flush=True)
        self.manifest[f"first_repeat/{split}"] = {
            "purpose": f"{split}_first_repeat",
            "pairs_per_cell": quota,
            "cell_count": len(summary),
            "pairs": quota * len(summary),
            "members": 2 * quota * len(summary),
            "attempts": total_attempts,
            "answer_support": dict(answer_support),
            "cells": summary,
        }


def all_train_lengths(root: Path) -> list[int]:
    return [
        len(json.loads(line)) - 1
        for path in sorted((root / "train_shards").glob("*.tokens.jsonl"))
        for line in path.open()
    ]


def file_inventory(root: Path) -> dict:
    return {
        str(path.relative_to(root)): {"sha256": file_hash(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "postwrite_audit.json", ".DS_Store"}
    }


def run():
    started = time.monotonic()
    design_path = ROOT / "experiment_v1_4/design_config.json"
    design = json.loads(design_path.read_text())
    base = ROOT / design["data"]["base_root"]
    amendment_path = ROOT / "experiment_v1_4/corpus_rebuild.json"
    amendment = json.loads(amendment_path.read_text())
    out = ROOT / amendment["active_data_root"]
    if out.exists():
        raise FileExistsError("Preserve an existing v1.4 corpus; never overwrite it")
    if file_hash(base / "manifest.json") != json.loads((ROOT / "experiment_v1_4/design_manifest.json").read_text())["v1_3_inputs"]["data/language_v1_3/manifest.json"]:
        raise ValueError("v1.3 base manifest differs from frozen design input")
    base_manifest = json.loads((base / "manifest.json").read_text())
    base_frozen = base_manifest["frozen_training"]
    assert (base_frozen["actual_prediction_tokens"], base_frozen["final_cursor"], base_frozen["shards"]) == (
        design["data"]["base_prefix_prediction_tokens"],
        design["data"]["base_prefix_sequences"],
        design["data"]["base_prefix_shards"],
    )
    print("loading all v1.0-v1.3 exact-sequence and READ-prefix hashes", flush=True)
    prior_hashes, prior_prefixes = load_prior_registry(base)
    out.mkdir(parents=True)
    (out / "train_shards").mkdir()
    inherited = {}
    for path in sorted((base / "train_shards").glob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        destination = out / "train_shards" / path.name
        shutil.copyfile(path, destination)
        if file_hash(path) != file_hash(destination):
            raise IOError(f"Prefix copy checksum failed: {path.name}")
        inherited[str(path.relative_to(base))] = file_hash(path)
    provenance = out / "provenance/v1_3"
    provenance.mkdir(parents=True)
    for name in ("manifest.json", "cpu_validation.json", "postwrite_audit.json", "corpus_statistics.json", "all_sequence_hashes.txt"):
        shutil.copyfile(base / name, provenance / name)
    builder = V14Builder(out, prior_hashes.copy(), prior_prefixes.copy())
    # New selection validation is generated before gate validation and test.
    builder.produce("select/general", "select_general", 0, design["evaluation_splits"]["select"]["general_sequences"])
    builder.first_repeat("select", design["evaluation_splits"]["select"]["first_repeat_pairs_per_cell"])
    conditions = [
        ("other_variable", lambda e, r: not r["same_as_last_dst"]),
        ("repeated_update", lambda e, r: r["query_update_count"] >= 2),
        ("first_read_after_set", lambda e, r: r["reads_of_query_var_since_latest_set"] == 0),
    ]
    for split in ("gate", "test"):
        spec = design["evaluation_splits"][split]
        builder.produce(f"{split}/general", f"{split}_general", 0, spec["general_sequences"])
        for index, (name, predicate) in enumerate(conditions):
            builder.produce(
                f"{split}/legacy_diagnostics/{name}",
                f"{split}_legacy_diagnostic",
                index,
                spec["legacy_diagnostic_targets_each"],
                predicate,
            )
        builder.first_repeat(split, spec["first_repeat_pairs_per_cell"])
    # Preserve the inherited composition/length and interpretation/causal scope,
    # but regenerate it under the new seed namespace before extending train.
    for pattern in range(2):
        builder.produce(f"test/composition/pattern_{pattern}", "test_composition", pattern, 1024, pattern=pattern)
    builder.produce("test/length", "test_length", 0, 1024, length=True)
    for split, quota in (("train", 50_000), ("val", 10_000), ("test", 20_000)):
        builder.produce(f"interpretation/{split}", f"interpretation_{split}", 0, position_quota=quota)
    for split, number in (("val", 256), ("test", 512)):
        for index, (changed, condition) in enumerate(itertools.product((True, False), ("memory", "composition"))):
            builder.pairs(
                f"causal_pairs/{split}_{'changed' if changed else 'unchanged'}_{condition}",
                f"causal_{split}",
                index,
                number,
                changed,
                condition,
            )
    reserved_hash_count = len(builder.hashes)
    reserved_prefix_count = len(builder.prefixes)
    (out / "reserved_hashes.txt").write_text("".join(value + "\n" for value in sorted(builder.hashes)))
    (out / "reserved_prefix_hashes.txt").write_text("".join(value + "\n" for value in sorted(builder.prefixes)))
    base_lengths = all_train_lengths(base)
    assert len(base_lengths) == design["data"]["base_prefix_sequences"]
    assert sum(base_lengths) == design["data"]["base_prefix_prediction_tokens"]
    total = sum(base_lengths)
    sequences = len(base_lengths)
    shard = design["data"]["base_prefix_shards"]
    budget = design["data"]["nominal_prediction_tokens"]
    while total < budget:
        tokens, count = builder.produce(
            f"train_shards/{shard:05d}",
            "train_extension",
            shard,
            4096,
            prediction_budget=budget - total,
        )
        total += tokens
        sequences += count
        shard += 1
    lengths = all_train_lengths(out)
    assert lengths[: len(base_lengths)] == base_lengths
    assert len(lengths) == sequences and sum(lengths) == total and sequences % 64 == 0
    assert total - sum(lengths[-64:]) < budget <= total
    train_stats = [value for key, value in builder.statistics.items() if key.startswith("train_shards/")]
    # Copied prefix statistics are loaded from the v1.3 manifest for coverage.
    base_stats = json.loads((base / "corpus_statistics.json").read_text())
    def train_statistics(value):
        for key, item in value.items():
            if key.startswith("train_shards/"):
                yield item
            elif key in ("new_splits", "base_train"):
                yield from train_statistics(item)
    prior_train_stats = list(train_statistics(base_stats))
    combined_stats = prior_train_stats + train_stats
    coverage = {
        key: len(set().union(*(set(value["histograms"][key]) for value in combined_stats)))
        for key in ("states", "commands", "truth_table", "adjacent_operators")
    }
    assert coverage == {"states": 16, "commands": 48, "truth_table": 12, "adjacent_operators": 25}
    milestones = {}
    cumulative = 0
    for cursor in range(64, sequences + 1, 64):
        cumulative += sum(lengths[cursor - 64 : cursor])
        for boundary in design["checkpoint_selection"]["candidate_milestones_prediction_tokens"]:
            if cumulative >= boundary and str(boundary) not in milestones:
                milestones[str(boundary)] = {"update": cursor // 64, "cursor": cursor, "prediction_tokens": cumulative}
    frozen = {
        "nominal_prediction_tokens": budget,
        "actual_prediction_tokens": total,
        "overshoot": total - budget,
        "final_cursor": sequences,
        "final_update": sequences // 64,
        "shards": shard,
        "prefix_sequences": len(base_lengths),
        "prefix_prediction_tokens": sum(base_lengths),
        "milestones": milestones,
    }
    validation = {
        **self_test(),
        "status": "passed",
        "schema": "language-v1.4-cpu-validation",
        "train_coverage": coverage,
        "train_prediction_tokens": total,
        "train_sequences": sequences,
        "batch_multiple_64": True,
        "prior_exact_hashes_loaded": len(prior_hashes),
        "prior_read_prefixes_loaded": len(prior_prefixes),
        "reserved_hash_count_before_train": reserved_hash_count,
        "reserved_prefix_count_before_train": reserved_prefix_count,
        "all_new_sequences_replayed": builder.validated,
        "causal_pairs_replayed": builder.pair_count,
        "first_repeat_cell_count": 42,
        "first_repeat_quota_checks": "passed",
        "first_repeat_semantics": "passed",
        "v1_3_train_prefix_byte_identical": True,
        "global_exact_sequence_and_read_prefix_isolation": "passed",
        "frozen_training": frozen,
    }
    shutil.copyfile(base / "vocab.json", out / "vocab.json")
    write_json(out / "corpus_statistics.json", {"new_splits": builder.statistics, "base_train": base_stats})
    write_json(out / "cpu_validation.json", validation)
    (out / "all_sequence_hashes.txt").write_text("".join(value + "\n" for value in sorted(builder.hashes)))
    manifest = {
        "schema_version": "language-v1.4",
        "generator_version": "1.4.1",
        "corpus_rebuild_sha256": file_hash(amendment_path),
        "experiment_version": "v1.4",
        "profile": "full",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "design_config_sha256": file_hash(design_path),
        "base_manifest_sha256": file_hash(base / "manifest.json"),
        "seed_derivation": "SHA256(20260920|language-v1.4|<purpose>|<index>)[:8], unsigned big endian",
        "rng": "numpy.random.Generator(PCG64)",
        "numpy_version": __import__("numpy").__version__,
        "holdout_patterns": PATTERNS,
        "splits": builder.manifest,
        "frozen_training": frozen,
        "prefix_contract": {
            "shards": design["data"]["base_prefix_shards"],
            "sequences": design["data"]["base_prefix_sequences"],
            "prediction_tokens": design["data"]["base_prefix_prediction_tokens"],
            "files": inherited,
        },
        "hash_registry": {
            "prior_exact_sequences": len(prior_hashes),
            "prior_read_prefixes": len(prior_prefixes),
            "all_exact_sequences": len(builder.hashes),
            "all_read_prefixes": len(builder.prefixes),
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in list((ROOT / "corpus").glob("*.py")) + [Path(__file__)]
        },
    }
    manifest["files"] = file_inventory(out)
    write_json(out / "manifest.json", manifest)
    # The v1.0 auditor does not understand nested first/repeat pair records, so
    # v1.4 performs a versioned audit below instead of weakening that auditor.
    audit_v1_4(out)
    results = ROOT / "experiment_v1_4/results"
    results.mkdir(parents=True, exist_ok=True)
    write_json(results / "p1_frozen_training.json", frozen)
    write_json(
        results / "p1_generation.json",
        {
            "status": "passed",
            "manifest_sha256": file_hash(out / "manifest.json"),
            "audit_sha256": file_hash(out / "postwrite_audit.json"),
            "elapsed_seconds": time.monotonic() - started,
            **frozen,
        },
    )
    print(json.dumps(frozen, indent=2), flush=True)


def audit_v1_4(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        path = root / name
        assert path.stat().st_size == info["bytes"] and file_hash(path) == info["sha256"], name
    base = ROOT / "data/language_v1_3"
    prior_hashes, prior_prefixes = load_prior_registry(base)
    seen_hashes = set(prior_hashes)
    seen_prefixes = set(prior_prefixes)
    new_hashes = set()
    new_prefixes = set()

    def reserve_stored(example):
        validate(example)
        value = example["canonical_hash"]
        assert value == digest(example["token_ids"]) and value not in seen_hashes
        prefixes = [digest(example["token_ids"][: event["answer_token_index"]]) for event in example["read_events"]]
        assert not (set(prefixes) & seen_prefixes)
        seen_hashes.add(value)
        seen_prefixes.update(prefixes)
        new_hashes.add(value)
        new_prefixes.update(prefixes)

    standard_paths = []
    for path in sorted(root.rglob("*.jsonl.gz")):
        if "provenance" in path.parts or "first_repeat" in path.parts or "causal_pairs" in path.parts:
            continue
        if "train_shards" in path.parts and int(path.name[:5]) < manifest["prefix_contract"]["shards"]:
            continue
        standard_paths.append(path)
    standard_sequences = 0
    for path in standard_paths:
        token_path = Path(str(path).replace(".jsonl.gz", ".tokens.jsonl"))
        with gzip.open(path, "rt") as metadata, token_path.open() as tokens:
            for left, right in zip(metadata, tokens, strict=True):
                example = json.loads(left)
                assert json.loads(right) == example["token_ids"]
                name = str(path.relative_to(root)).removesuffix(".jsonl.gz")
                info = manifest["splits"][name]
                assert example["rng_seed"] == info["rng_seed"] == v14_seed(info["purpose"], info["index"])
                reserve_stored(example)
                standard_sequences += 1

    pair_counts = {}
    pair_hashes = set()
    for split, quota in (("select", 64), ("gate", 64), ("test", 128)):
        counts = Counter()
        members = set()
        answers = set()
        with gzip.open(root / f"first_repeat/{split}.jsonl.gz", "rt") as handle:
            for line in handle:
                pair = json.loads(line)
                origin, repeat = pair["origin"], pair["repeat"]
                cell = next(c for c in cells() if c.cell_id == pair["cell_id"])
                assert origin["rng_seed"] == repeat["rng_seed"] == v14_seed(f"{split}_first_repeat", cell.index)
                validate(origin)
                validate(repeat)
                assert origin["split"] == repeat["split"] == split
                assert origin["canonical_hash"] == pair["origin_hash"]
                assert repeat["canonical_hash"] == pair["repeat_hash"]
                assert origin["canonical_hash"] not in members and repeat["canonical_hash"] not in members
                hashes = {origin["canonical_hash"], repeat["canonical_hash"]}
                assert not (hashes & seen_hashes)
                prefixes = {
                    digest(example["token_ids"][: event["answer_token_index"]])
                    for example in (origin, repeat)
                    for event in example["read_events"]
                }
                assert not (prefixes & seen_prefixes)
                seen_hashes.update(hashes)
                seen_prefixes.update(prefixes)
                new_hashes.update(hashes)
                new_prefixes.update(prefixes)
                members.update((origin["canonical_hash"], repeat["canonical_hash"]))
                first = origin["read_events"][pair["origin_target_read_id"]]
                repeated = repeat["read_events"][pair["repeat_target_read_id"]]
                assert first["answer"] == repeated["answer"] == pair["answer"]
                assert first["state_at_read"] == repeated["state_at_read"]
                assert first["reads_of_query_var_since_last_update"] == 0
                assert repeated["reads_of_query_var_since_last_update"] >= 1
                counts[pair["cell_id"]] += 1
                answers.add(pair["answer"])
        assert len(counts) == 42 and set(counts.values()) == {quota} and answers == {0, 1}
        pair_counts[split] = {"pairs": sum(counts.values()), "members": len(members), "cells": len(counts)}
        pair_hashes.update(members)
    causal_pairs = 0
    for pair_path in sorted((root / "causal_pairs").glob("*.jsonl.gz")):
        if pair_path.name.endswith(".origins.jsonl.gz"):
            continue
        origin_path = Path(str(pair_path).replace(".jsonl.gz", ".origins.jsonl.gz"))
        with gzip.open(pair_path, "rt") as pairs, gzip.open(origin_path, "rt") as origins:
            for pair_line, origin_line in zip(pairs, origins, strict=True):
                pair, origin = json.loads(pair_line), json.loads(origin_line)
                name = str(pair_path.relative_to(root)).removesuffix(".jsonl.gz")
                info = manifest["splits"][name]
                assert pair["rng_seed"] == origin["rng_seed"] == v14_seed(info["purpose"], info["index"])
                reserve_stored(origin)
                assert origin["canonical_hash"] == pair["origin_hash"]
                original, counter = pair["original_prefix_ids"], pair["counterfactual_prefix_ids"]
                assert [index for index, values in enumerate(zip(original, counter)) if values[0] != values[1]] == pair["prefix_diff_indices"]
                first, second = replay(original, partial=True), replay(counter, partial=True)
                assert first["answer"] == pair["original_answer"] and second["answer"] == pair["counterfactual_answer"]
                assert (first["answer"] != second["answer"]) == pair["changed_target"]
                original_prefix, counter_prefix = map(digest, (original, counter))
                assert [original_prefix, counter_prefix] == pair["prefix_hashes"]
                assert original_prefix in seen_prefixes and counter_prefix not in seen_prefixes
                counter_hash = pair["counterfactual_origin_hash"]
                assert counter_hash not in seen_hashes
                seen_hashes.add(counter_hash)
                seen_prefixes.add(counter_prefix)
                new_hashes.add(counter_hash)
                new_prefixes.add(counter_prefix)
                causal_pairs += 1
    # Independent full-train replay and prefix-byte checks.
    for path in sorted((base / "train_shards").glob("*")):
        if path.is_file() and path.name != ".DS_Store":
            assert file_hash(path) == file_hash(root / "train_shards" / path.name)
    train_sequences = train_tokens = 0
    for path in sorted((root / "train_shards").glob("*.jsonl.gz")):
        token_path = Path(str(path).replace(".jsonl.gz", ".tokens.jsonl"))
        with gzip.open(path, "rt") as metadata, token_path.open() as tokens:
            for left, right in zip(metadata, tokens, strict=True):
                example = json.loads(left)
                assert json.loads(right) == example["token_ids"]
                validate(example)
                train_sequences += 1
                train_tokens += len(example["token_ids"]) - 1
    frozen = manifest["frozen_training"]
    assert (train_sequences, train_tokens) == (frozen["final_cursor"], frozen["actual_prediction_tokens"])
    expected_hashes = set((root / "all_sequence_hashes.txt").read_text().splitlines())
    assert seen_hashes == expected_hashes
    reserved_hashes = set((root / "reserved_hashes.txt").read_text().splitlines())
    assert reserved_hashes <= expected_hashes and prior_hashes <= reserved_hashes
    reserved_prefixes = set((root / "reserved_prefix_hashes.txt").read_text().splitlines())
    assert prior_prefixes <= reserved_prefixes <= seen_prefixes
    # Position quota files are independently joined against stored metadata.
    for split, quota in (("train", 50_000), ("val", 10_000), ("test", 20_000)):
        available = {"read": set(), "update": set()}
        for example in metadata_rows(root / f"interpretation/{split}.jsonl.gz"):
            available["read"].update((example["sequence_id"], event["query_token_index"], event["read_id"]) for event in example["read_events"])
            available["update"].update((example["sequence_id"], event["end_token_index"], event["update_id"]) for event in example["update_events"])
        for kind in ("read", "update"):
            positions = json.loads((root / f"interpretation/{split}.{kind}_positions.json").read_text())
            keys = [(row["sequence_id"], row["token_index"], row["event_id"]) for row in positions]
            assert len(keys) == quota and keys == sorted(keys) and len(set(keys)) == quota and set(keys) <= available[kind]
    result = {
        "status": "passed",
        "schema": "language-v1.4-postwrite-audit",
        "file_checksums": "passed",
        "train_token_metadata_alignment": "passed",
        "train_independent_replay": "passed",
        "all_new_fixed_splits_independent_replay": "passed",
        "global_sequence_and_read_prefix_isolation": "passed",
        "historical_reserved_registry_loaded": True,
        "rng_metadata_matches_v14_namespace": True,
        "interpretation_position_quota_and_join": "passed",
        "causal_pair_replay": "passed",
        "v1_3_train_prefix_byte_identical": "passed",
        "first_repeat_replay_and_quota": "passed",
        "first_repeat": pair_counts,
        "first_repeat_unique_member_hashes": len(pair_hashes),
        "standard_new_sequences": standard_sequences,
        "causal_pairs": causal_pairs,
        "prior_sequence_hashes": len(prior_hashes),
        "prior_read_prefix_hashes": len(prior_prefixes),
        "all_sequence_hashes": len(expected_hashes),
        "all_read_prefix_hashes": len(seen_prefixes),
        "train_sequences": train_sequences,
        "train_prediction_tokens": train_tokens,
    }
    write_json(root / "postwrite_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    run()
