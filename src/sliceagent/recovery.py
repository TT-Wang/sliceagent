"""Turn write-ahead log (WAL) for crash recovery.

sliceagent is cache-not-log by design (no transcript), but a HARD process crash mid-turn (kill -9, OOM, power
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

from .private_state import private_dir, private_file


def state_dir(*parts: str) -> str:
    """The sliceagent STATE root (~/.sliceagent, or $SLICEAGENT_CACHE_DIR) — internal logs / records / WAL live
    HERE, never in the user's workspace. Joins `parts`, creates the dir, returns it. One source of truth so
    nothing scribbles scratch/ into the project being worked on."""
    base = os.environ.get("SLICEAGENT_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".sliceagent")
    current = private_dir(base)
    for part in parts:
        current = private_dir(os.path.join(current, part))
    return current


def root_key(root: str) -> str:
    """A stable short key for a workspace path (so per-workspace state files don't collide)."""
    return hashlib.sha1(os.path.realpath(root or ".").encode("utf-8")).hexdigest()[:16]


def _wal_dir() -> str:
    return state_dir("wal")


def _path(root: str) -> str:
    return os.path.join(_wal_dir(), root_key(root) + ".json")


def _sanitize(messages: list) -> list:
    """Strip heavy image base64 from WAL messages AND redact secrets — the WAL persists in-flight tool
    output to disk after a hard crash, so it must honor the same redact-on-persist boundary as the episodic
    cache / debug log (every other durable store redacts). Replace image_url parts with a placeholder."""
    from .safety import redact_text

    def _redact_tool_calls(tcs):
        # assistant.tool_calls[*].function.arguments is a JSON STRING carrying the same secrets as content
        # (edit_file bodies, tokens/paths in run_command) — it was NOT redacted, so a hard-crash WAL persisted
        # it in the clear (external review H-13). Redact it on the same persist boundary as content.
        if not isinstance(tcs, list):
            return tcs
        red = []
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                red.append({**tc, "function": {**fn, "arguments": redact_text(fn["arguments"])}})
            else:
                red.append(tc)
        return red

    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        new = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    parts.append({"type": "text", "text": "[image attached]"})
                elif isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append({**p, "text": redact_text(p["text"])})
                else:
                    parts.append(p)
            new["content"] = parts
        elif isinstance(c, str):
            new["content"] = redact_text(c)
        if "tool_calls" in new:
            new["tool_calls"] = _redact_tool_calls(new["tool_calls"])
        out.append(new)
    return out


def record(root: str, *, goal: str, messages: list, step: int) -> None:
    """Atomically write the in-flight turn. Best-effort — never raises into the loop."""
    tmp = None
    try:
        from .safety import redact_text
        body = json.dumps({"goal": redact_text(goal or ""), "step": step, "ts": time.time(),
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
        private_file(p)  # repair an older permissive target on unusual rename implementations
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
        path = _path(root)
        private_file(path)  # repair state created by older releases before reading it
        with open(path, encoding="utf-8") as f:
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
