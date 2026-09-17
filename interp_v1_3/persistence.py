"""Immutable artifact persistence primitives shared with the audited v1.2 runner."""
from interp_v1_2.persistence import atomic_json, read_index, recover, safe_path

__all__ = ["atomic_json", "read_index", "recover", "safe_path"]
