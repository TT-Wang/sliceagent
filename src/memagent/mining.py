"""Lesson-mining HELPERS — shared by the CACHE-sourced distill path.

The WRITE side of the memory loop is CACHE-ONLY: distillation happens at session end in
`memory.consolidate` → `consolidate.promote_episodes` / `promote_procedures`, which read the episodic
CACHE (`read_episodes`), never the live slice. The old per-turn `LessonMiner` that read slice state
(`s.edited_files`, `s.last_error`) was removed so distill no longer couples to L1 — the layering is
now strictly  L1 slice ─seal→ L2 cache ─distill→ L3 memem ─recall→ L1 slice.

These pure helpers — error-signature dedup, self-inflicted-error filtering, and pitfall titling — are
reused by that cache path (and the tests). No slice import; no event sink.
"""
from __future__ import annotations

import re

from .text_utils import one_line
from .tools import HOST_ERROR_SENTINELS


def _err_key(err: str) -> str:
    """A stable-ish signature for an error, for dedup."""
    return one_line(err, 100).lower()


def is_self_inflicted(pitfall: str) -> bool:
    """D2 — True when `pitfall` is the agent hitting the HOST's own guard rail (confinement,
    permission), not a real engineering pitfall. Such an error teaches a future agent nothing, so it
    must mine NOTHING. Task-agnostic substring match against the host's error sentinels (tools.py)."""
    low = (pitfall or "").lower()
    return any(sentinel in low for sentinel in HOST_ERROR_SENTINELS)


# leading boilerplate the host prepends to a tool error — stripped so the lesson TITLE is the actual
# pitfall, not "Error: ". Task-agnostic (no tool/language names).
_ERR_PREFIX_RE = re.compile(r"^\s*(?:error|exit code \d+)\s*[:\-]?\s*", re.I)


def pitfall_signature(pitfall: str, n: int = 60) -> str:
    """D1 — a short, readable lesson TITLE from the PITFALL itself (never the user's goal). Strips the
    host's 'Error:'/'Exit code:' prefix so the title leads with the real failure."""
    sig = _ERR_PREFIX_RE.sub("", one_line(pitfall, 200)).strip()
    return one_line(sig or pitfall, n)
