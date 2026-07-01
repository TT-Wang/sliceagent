"""Deterministic tool-result one-liners (item 14b) — '[tool] action -> outcome, N lines'.

Replaces a large tool output with an informative line instead of a zero-information
placeholder. memagent has no transcript to compress, but the SAME shape makes the episodic
TRACE (history.render_trace) far more legible: a compact line that says WHAT the tool did and
HOW IT CAME OUT, replacing the raw head+tail tail-snippet.

Adapted to memagent's tool names (file_operations / tools.py: read_file, write_file,
edit_file/str_replace, append_to_file, run_command, list_files, execute_code, plus the
memagent built-ins skill/new_topic/switch_topic/recall_history). Pure + deterministic →
testable offline, no LLM.

NO-TRANSCRIPT INVARIANT: this only formats already-stored episodic records for read-back; it
produces no new context and is never injected into the slice.

PUBLIC SIGNATURE (pinned):
    summarize_tool_result(name: str, args: dict, output: str, *, failing: bool = False) -> str
"""
from __future__ import annotations

import re

_MAX_TGT = 70


def _line_count(text: str) -> int:
    return (text.count("\n") + 1) if (text or "").strip() else 0


def _clip(s, n: int = _MAX_TGT) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _first_code_line(code: str) -> str:
    for ln in str(code or "").splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def summarize_tool_result(name: str, args: dict, output: str, *, failing: bool = False) -> str:
    """One informative line: '[name] action -> outcome, N lines'. Never raises — a bad arg
    shape falls through to the generic branch. `failing` flags the outcome with ✗."""
    args = args if isinstance(args, dict) else {}
    out = output or ""
    n = _line_count(out)
    fail = " ✗" if failing else ""

    if name in ("run_command", "execute_code"):
        if name == "execute_code":
            tgt = _clip(_first_code_line(args.get("code", "")))
        else:
            tgt = _clip(args.get("command", ""))
        exit_m = re.search(r"[Ee]xit code[:\s]+(-?\d+)", out)
        outcome = f"exit {exit_m.group(1)}" if exit_m else ("error" if failing else "ok")
        return f"[{name}] `{tgt}` -> {outcome}, {n} lines{fail}"

    if name == "read_file":
        return f"[read_file] {_clip(args.get('path', '?'))} -> {len(out):,} chars, {n} lines{fail}"

    if name in ("write_file", "append_to_file"):
        wl = _line_count(args.get("content", ""))
        verb = "wrote" if name == "write_file" else "appended"
        return f"[{name}] {verb} {_clip(args.get('path', '?'))} ({wl} lines){fail}"

    if name in ("edit_file", "str_replace"):
        # "0 " anywhere in the first 20 chars false-matched the byte count of a normal write ("Wrote 100
        # bytes" contains "0 bytes"); use precise no-op signals so a real edit is never summarized as no-op.
        ok = "no-op" if (not failing and ("No changes" in out or "Wrote 0 bytes" in out)) else \
            ("failed" if failing else "applied")
        return f"[{name}] {_clip(args.get('path', '?'))} -> {ok}{fail}"

    if name == "list_files":
        return f"[list_files] {_clip(args.get('path', '.'))} -> {n} entries{fail}"

    if name == "skill":
        return f"[skill] loaded {_clip(args.get('name', '?'))}{fail}"

    if name in ("new_topic", "switch_topic"):
        tgt = args.get("goal") or args.get("task_id") or ""
        return f"[{name}] {_clip(tgt)}{fail}"

    if name == "recall_history":
        return f"[recall_history] -> {n} lines{fail}"

    # generic fallback: first one or two args + size
    hint = ""
    for k, v in list(args.items())[:2]:
        if k == "note":
            continue
        hint += f" {k}={_clip(v, 30)}"
    return f"[{name}]{hint} -> {len(out):,} chars{fail}"
