"""Tiny shared text helpers — ONE home for the whitespace/timestamp normalizers that were copy-pasted
across regions.py, skills.py, pagetable.py and history.py. Leaf module (no intra-package imports) so any
module can use it without an import cycle. Behavior is byte-identical to the expressions it replaces.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize_ws(s) -> str:
    """Collapse all runs of whitespace to a single space and strip. `None`/non-str -> ''."""
    return _WS.sub(" ", str(s or "")).strip()


def one_line(s, n: int = 80) -> str:
    """`normalize_ws` truncated to `n` chars — a one-line, bounded rendering of arbitrary text."""
    return normalize_ws(s)[:n]


def format_ts(ts) -> str:
    """ISO timestamp -> compact 'MM-DD HH:MM' (e.g. '06-16 12:30'); '' for empty/None."""
    return (ts or "")[5:16].replace("T", " ")
