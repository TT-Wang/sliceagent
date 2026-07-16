"""Canonical tool-call identity and safe same-wave read deduplication."""
from __future__ import annotations

import json
from typing import Any, Mapping


DEDUP_SAFE_TOOL_NAMES = frozenset({"read_file", "list_files", "search_history", "grep", "glob"})


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON with model commentary excluded from physical call identity."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return json.dumps(
        {key: value for key, value in args.items() if key != "note"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


__all__ = ["DEDUP_SAFE_TOOL_NAMES", "canonical_tool_args"]
