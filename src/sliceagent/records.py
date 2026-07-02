"""Append-only records journal.

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

from .events import Event, StepEnd, TurnEnd, TurnInterrupted
from .recovery import state_dir

# Records live in the sliceagent STATE dir (~/.sliceagent/records), NOT scratch/ in the user's workspace —
# the session_id is already in each filename, so a flat per-session journal needs no per-workspace key.
RECORDS_ROOT = state_dir("records")


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
        with open(self.path, encoding="utf-8", errors="replace") as f:   # truncated multibyte → replacement char (then json.loads skips it); never crash replay
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
        self._acc = {"input_other": 0, "input_cache_read": 0, "input_cache_creation": 0, "output": 0}

    def __call__(self, e: Event) -> None:
        # #55: the TYPED breakdown (input_other/cache_read/…) lives on StepEnd, not on TurnEnd (whose usage
        # is just the prompt/completion totals). Accumulate per step and snapshot at turn close, so the
        # journalled cache fields are real, not always 0. Snapshot on BOTH clean and parked turn-ends.
        if isinstance(e, StepEnd):
            u = e.usage or {}
            for k in self._acc:
                self._acc[k] += u.get(k, 0) or 0
            if "output" not in u:   # legacy usage dicts: fall back to completion_tokens for output
                self._acc["output"] += u.get("completion_tokens", 0) or 0
        elif isinstance(e, (TurnEnd, TurnInterrupted)):
            self._turn += 1
            u = getattr(e, "usage", None) or {}   # TurnInterrupted carries no usage; accumulator has it
            # prefer the per-step accumulator; fall back to a typed field carried on the TurnEnd usage
            # itself (back-compat for callers that pass the full breakdown there).
            typed = {k: (self._acc[k] or u.get(k, 0) or 0) for k in self._acc}
            # On a PARKED turn (TurnInterrupted carries no usage) the prompt/completion totals would record
            # as 0 — fall back to the per-step accumulator so the journal isn't undercounted.
            acc_prompt = self._acc["input_other"] + self._acc["input_cache_read"] + self._acc["input_cache_creation"]
            self.journal.record(
                "usage", turn=self._turn, model=self.model,
                prompt_tokens=u.get("prompt_tokens") or acc_prompt,
                completion_tokens=u.get("completion_tokens") or self._acc["output"],
                **typed,
            )
            self._acc = {k: 0 for k in self._acc}


def total_usage(journal: Journal) -> dict:
    """Aggregate the journal's usage records into per-model + grand totals (a simple cost report)."""
    fields = ("prompt_tokens", "completion_tokens", "input_other", "input_cache_read",
              "input_cache_creation", "output")   # #55: aggregate the cache breakdown, not just prompt/compl
    by_model: dict[str, dict] = {}
    for r in journal.read("usage"):
        m = by_model.setdefault(r.get("model") or "?", {**{f: 0 for f in fields}, "turns": 0})
        for f in fields:
            m[f] += r.get(f, 0) or 0
        m["turns"] += 1
    return by_model
