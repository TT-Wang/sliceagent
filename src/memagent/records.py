"""Append-only records journal (borrowed from Kimi agent-core/records + usage).

A durable, per-session, TYPED event log that sits ABOVE the kernel: replay/resume and the cron /
background subsystems read it. It NEVER feeds the live slice — replay rebuilds state on RESUME only,
never mid-turn (preserving the Markov boundary; cf. the records-replay moat-conflict note). Reuses the
per-session JSONL pattern of the episodic cache rather than inventing a new store.

`UsageRecorder` is the first consumer: it journals per-turn token usage as a durable cost log — distinct
from the in-memory `CostMetrics` summary (metrics.py), which measures the moat curve within a run.
"""
from __future__ import annotations

import json
import os

from .events import Event, TurnEnd

RECORDS_ROOT = "scratch/records"


def _records_path(session_id: str, root: str = RECORDS_ROOT) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (session_id or "default"))
    return os.path.join(root, f"{safe}.jsonl")


class Journal:
    """A per-session append-only typed-record log. `record(type, **data)` appends one line;
    `read(type=None)` reads them back (optionally filtered by type). Robust by construction: a malformed
    line is skipped, a missing file reads as empty — a journal hiccup never breaks the caller."""

    def __init__(self, session_id: str, root: str = RECORDS_ROOT):
        self.path = _records_path(session_id, root)

    def record(self, rtype: str, **data) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": rtype, **data}, ensure_ascii=False) + "\n")

    def read(self, rtype: str | None = None) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001 — a corrupt line never breaks replay
                    continue
                if rtype is None or rec.get("type") == rtype:
                    out.append(rec)
        return out


class UsageRecorder:
    """Event sink that journals per-turn token usage (durable cost log). Records on TurnEnd. Pure
    observer — off the moat, like CostMetrics; the difference is this PERSISTS for cross-run analysis."""

    def __init__(self, journal: Journal, model: str = ""):
        self.journal = journal
        self.model = model
        self._turn = 0

    def __call__(self, e: Event) -> None:
        if isinstance(e, TurnEnd):
            self._turn += 1
            u = e.usage or {}
            self.journal.record(
                "usage", turn=self._turn, model=self.model,
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                input_other=u.get("input_other", 0),
                input_cache_read=u.get("input_cache_read", 0),
            )


def total_usage(journal: Journal) -> dict:
    """Aggregate the journal's usage records into per-model + grand totals (a simple cost report)."""
    by_model: dict[str, dict] = {}
    for r in journal.read("usage"):
        m = by_model.setdefault(r.get("model") or "?", {"prompt_tokens": 0, "completion_tokens": 0, "turns": 0})
        m["prompt_tokens"] += r.get("prompt_tokens", 0)
        m["completion_tokens"] += r.get("completion_tokens", 0)
        m["turns"] += 1
    return by_model
