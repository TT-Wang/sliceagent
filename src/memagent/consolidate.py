"""Consolidation — the cache→long-term step (MEMORY-SPEC step 4).

Reads the lossless episodic cache for a session and PROMOTES durable lessons into long-term memory,
per the policy: promote CORRECTIVE episodes (a pitfall hit, then the task ended clean), deduped by
pitfall signature, declarative phrasing, secrets excluded. `promote_episodes(records)` is pure (no
I/O, no LLM) so it's testable offline; `MememMemory.consolidate` wires it to the JSONL + remember().

Deliberately NOT here yet (designed in MEMORY-SPEC, later steps): routing procedures→skills,
frequency/retrieval weighting, and the decay pass. This is the facts→memory core.
"""
from __future__ import annotations

import os
import re

from .slice import one_line

_SECRET_RE = re.compile(r"(api[_-]?key|secret|token|password|credential|bearer)\s*[:=]", re.I)
_EXT_TAG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust",
            ".java": "java", ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".sh": "shell"}


def _tags(files) -> str:
    tags = {"memagent"}
    for f in files:
        t = _EXT_TAG.get(os.path.splitext(f)[1])
        if t:
            tags.add(t)
    return ",".join(sorted(tags))


def _is_secret(text: str) -> bool:
    return bool(_SECRET_RE.search(text or ""))


def _failing_obs(record: dict) -> str:
    for st in record.get("steps", []):
        for o in st.get("observation", []):
            if isinstance(o, str) and (o.startswith("Error") or o.startswith("Exit code")):
                return o
    return ""


def promote_episodes(records: list[dict]) -> list[dict]:
    """Promote one durable lesson per CORRECTIVE episode (a task that hit an error and then ended
    clean). Deduped by pitfall signature; secrets excluded; declarative phrasing. Returns a list of
    {title, content, tags}. Pure — feed it the parsed episodic JSONL records."""
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r.get("task_id", ""), []).append(r)
    out, seen = [], set()
    for recs in by_task.values():
        recs = sorted(recs, key=lambda r: r.get("turn", 0))
        rmeta = [r.get("record", {}) for r in recs]
        had_fail = any(m.get("meta", {}).get("failing") for m in rmeta)
        last_meta = rmeta[-1].get("meta", {})
        ended_clean = last_meta.get("stop_reason") == "end_turn" and not last_meta.get("failing")
        if not (had_fail and ended_clean):           # corrective episode only
            continue
        pitfall = next((p for m in reversed(rmeta) if (p := _failing_obs(m))), "")
        if not pitfall or _is_secret(pitfall):        # need a pitfall; never store secrets
            continue
        sig = one_line(pitfall, 100).lower()
        if sig in seen:                               # dedupe by pitfall signature
            continue
        seen.add(sig)
        note = rmeta[-1].get("note") or next((m.get("note") for m in reversed(rmeta) if m.get("note")), "")
        files = sorted({f for m in rmeta for f in m.get("meta", {}).get("files", [])})
        content = (f"Pitfall: {one_line(pitfall, 200)}\n"
                   f"Resolution: {one_line(note, 200) or 'resolved'} (files: {', '.join(files) or 'n/a'})")
        title = "Lesson: " + (one_line(note, 60) or one_line(pitfall, 60))
        out.append({"title": title, "content": content, "tags": _tags(files)})
    return out
