"""Turn write-ahead log (WAL) for crash recovery.

memagent is cache-not-log by design (no transcript), but a HARD process crash mid-turn (kill -9, OOM, power
loss) would otherwise lose the in-flight turn entirely. The WAL is a RECOVERY-ONLY artifact: the accumulating
turn messages are written after each step and DELETED on any clean/parked exit. It is never read during
normal operation — only on the NEXT startup in the same workspace, to surface what was interrupted. Keyed by
workspace root so a restart-in-place finds it. Entirely best-effort: a WAL failure must never affect a turn.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time


def _wal_dir() -> str:
    base = os.environ.get("MEMAGENT_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".memagent")
    d = os.path.join(base, "wal")
    os.makedirs(d, exist_ok=True)
    return d


def _path(root: str) -> str:
    key = hashlib.sha1(os.path.realpath(root).encode("utf-8")).hexdigest()[:16]
    return os.path.join(_wal_dir(), key + ".json")


def _sanitize(messages: list) -> list:
    """Strip heavy image base64 from WAL messages — recovery only needs text + structure, and writing a
    per-step b64 image would bloat the WAL (and add nothing to the recovery notice). Replace image_url parts
    with a small placeholder; keep everything else."""
    out = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            parts = [{"type": "text", "text": "[image attached]"} if isinstance(p, dict)
                     and p.get("type") == "image_url" else p for p in c]
            out.append({**m, "content": parts})
        else:
            out.append(m)
    return out


def record(root: str, *, goal: str, messages: list, step: int) -> None:
    """Atomically write the in-flight turn. Best-effort — never raises into the loop."""
    tmp = None
    try:
        body = json.dumps({"goal": goal, "step": step, "ts": time.time(),
                           "root": os.path.realpath(root), "messages": _sanitize(messages)},
                          ensure_ascii=False)
        p = _path(root)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), prefix=".wal-", suffix=".tmp")  # mkstemp → 0600
        try:
            os.write(fd, body.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001 — the WAL must never destabilize a turn
        if tmp is not None:            # tmp may be unbound if json.dumps / _path / mkstemp itself failed
            try:
                os.remove(tmp)
            except OSError:
                pass


def pending(root: str) -> dict | None:
    """The interrupted turn for this workspace, or None. Its mere existence means the last turn never
    reached a clean/parked exit (i.e. a hard crash)."""
    try:
        with open(_path(root), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def clear(root: str) -> None:
    """Remove the WAL — called on every clean or parked turn exit (so a leftover WAL == a crash)."""
    try:
        os.remove(_path(root))
    except OSError:
        pass


def last_assistant(wal: dict) -> str:
    """The most recent assistant text in the interrupted turn (what the agent was last saying)."""
    for m in reversed((wal or {}).get("messages", []) or []):
        c = m.get("content") if isinstance(m, dict) else None
        if m.get("role") == "assistant" if isinstance(m, dict) else False:
            if isinstance(c, str) and c.strip():       # assistant content is text; list-safe by construction
                return c
    return ""
