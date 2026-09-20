"""v1.4 seed namespace and grammar-preserving first/repeat pair construction."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .language import OPS, digest, pattern_ok, render, sample, select_target
from .replay import validate


SEED_NAMESPACE = "20260920|language-v1.4"
DEPTH_BINS = ("1", "2-3", "4+")
BINARY_OPS = ("AND", "OR", "XOR")
TRUTHS = ("00", "01", "10", "11")


def seed(purpose: str, index: int = 0) -> int:
    payload = f"{SEED_NAMESPACE}|{purpose}|{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def rng(purpose: str, index: int = 0) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed(purpose, index)))


def depth_bin(depth: int) -> str:
    if depth == 1:
        return "1"
    if 2 <= depth <= 3:
        return "2-3"
    if depth >= 4:
        return "4+"
    raise ValueError(f"Logical target has invalid depth {depth}")


@dataclass(frozen=True)
class Cell:
    index: int
    operator: str
    input_pattern: str
    depth: str

    @property
    def cell_id(self) -> str:
        return f"{self.operator}:{self.input_pattern}:depth_{self.depth}"


def cells() -> list[Cell]:
    result: list[Cell] = []
    for op in ("NOT",) + BINARY_OPS:
        patterns = ("0", "1") if op == "NOT" else TRUTHS
        for pattern in patterns:
            for depth in DEPTH_BINS:
                result.append(Cell(len(result), op, pattern, depth))
    assert len(result) == 42 and len({x.cell_id for x in result}) == 42
    return result


def _event_cell(example: dict, read: dict) -> tuple[str, str, str] | None:
    uid = read["last_update_id_for_query_var_or_null"]
    if uid is None or read["reads_of_query_var_since_last_update"] != 0:
        return None
    update = example["update_events"][uid]
    if update["op"] not in ("NOT",) + BINARY_OPS or update["dst"] != read["query_var"]:
        return None
    pattern = str(update["dst_before"]) if update["op"] == "NOT" else update["input_truth_pattern_or_null"]
    return update["op"], pattern, depth_bin(read["structural_depth"])


def eligible_targets(example: dict, cell: Cell) -> list[int]:
    """Return targets whose last relevant update can be split into a valid repeat block."""
    result = []
    if len(example["blocks"]) >= 24:
        return result
    for read in example["read_events"]:
        if _event_cell(example, read) != (cell.operator, cell.input_pattern, cell.depth):
            continue
        uid = read["last_update_id_for_query_var_or_null"]
        update = example["update_events"][uid]
        block = example["blocks"][update["block_id"]]
        offset = block["update_ids"].index(uid)
        # Both halves need 1--3 updates. The original block READ must not query the
        # target variable, otherwise the origin was not a first READ at the target.
        if offset + 1 >= len(block["update_ids"]):
            continue
        block_read = example["read_events"][block["read_id"]]
        if block_read["query_var"] == read["query_var"]:
            continue
        result.append(read["read_id"])
    return result


def make_pair(initial: list[int], program: list, split: str, rng_seed: int, cell: Cell, chooser: np.random.Generator) -> dict | None:
    """Create two valid programs differing only by an intervening correct READ block."""
    origin = render(initial, program, split, rng_seed, True)
    candidates = eligible_targets(origin, cell)
    if not candidates:
        return None
    rid = candidates[int(chooser.integers(len(candidates)))]
    target = origin["read_events"][rid]
    uid = target["last_update_id_for_query_var_or_null"]
    update = origin["update_events"][uid]
    bid = update["block_id"]
    block_commands, block_query = program[bid]
    offset = origin["blocks"][bid]["update_ids"].index(uid)
    query_index = "ABCD".index(target["query_var"])
    repeat_program = list(program[:bid]) + [
        (list(block_commands[: offset + 1]), query_index),
        (list(block_commands[offset + 1 :]), block_query),
    ] + list(program[bid + 1 :])
    repeat = render(initial, repeat_program, split, rng_seed, True)
    repeat_rid = rid + 1
    repeat_target = repeat["read_events"][repeat_rid]
    select_target(origin, rid)
    select_target(repeat, repeat_rid)
    validate(origin)
    validate(repeat)
    assert origin["canonical_hash"] == digest(origin["token_ids"])
    assert repeat["canonical_hash"] == digest(repeat["token_ids"])
    assert target["answer"] == repeat_target["answer"]
    assert target["query_var"] == repeat_target["query_var"]
    assert target["structural_depth"] == repeat_target["structural_depth"]
    assert target["state_at_read"] == repeat_target["state_at_read"]
    assert target["reads_of_query_var_since_last_update"] == 0
    assert repeat_target["reads_of_query_var_since_last_update"] == 1
    first_update = origin["update_events"][uid]
    repeat_update = repeat["update_events"][uid]
    for key in ("op", "dst", "src_or_null", "literal_or_null", "dst_before", "src_before_or_null", "dst_after", "input_truth_pattern_or_null", "structural_depth"):
        assert first_update[key] == repeat_update[key]
    return {
        "schema_version": "language-v1.4-first-repeat-pair",
        "split": split,
        "pair_id": f"{cell.cell_id}:{origin['canonical_hash']}",
        "cell_id": cell.cell_id,
        "cell_index": cell.index,
        "operator": cell.operator,
        "input_pattern": cell.input_pattern,
        "depth_bin": cell.depth,
        "answer": target["answer"],
        "origin_hash": origin["canonical_hash"],
        "repeat_hash": repeat["canonical_hash"],
        "origin": origin,
        "repeat": repeat,
        "origin_target_read_id": rid,
        "repeat_target_read_id": repeat_rid,
        "inserted_read_id": origin["blocks"][bid]["read_id"],
    }


def sample_pair(generator: np.random.Generator, split: str, cell: Cell, max_attempts: int = 100_000) -> tuple[dict, int, dict]:
    rejections: dict[str, int] = {}
    for attempt in range(1, max_attempts + 1):
        initial, program, _ = sample(generator)
        if not pattern_ok(program):
            rejections["holdout"] = rejections.get("holdout", 0) + 1
            continue
        pair = make_pair(initial, program, split, seed(f"{split}_first_repeat", cell.index), cell, generator)
        if pair is None:
            rejections["no_eligible_target"] = rejections.get("no_eligible_target", 0) + 1
            continue
        return pair, attempt, rejections
    raise RuntimeError(f"{split}/{cell.cell_id}: MAX_ATTEMPTS exceeded: {rejections}")
