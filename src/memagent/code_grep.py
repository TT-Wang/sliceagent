"""Guarded `grep` tool — Kimi-style ripgrep pagination + Hermes consecutive-search guard.

The single discovery-on-demand seam (W7 deleted `code_index.snippets`; the model now
greps for content instead). Two borrowed mechanisms:

- Kimi grep pagination (packages/agent-core/src/tools/builtin/file/grep.ts:277): run
  ripgrep, then slice the result lines by offset/limit and append an explicit
  "[truncated; use offset=N to see more]" notice when more remain.
- Hermes consecutive-search guard (tools/file_tools.py:1399, the `{last_key, consecutive}`
  tracker): the 4th identical-in-a-row call returns a BLOCKED message instead of re-running
  the same search forever. The key INCLUDES offset, so paging through truncated results
  (a *different* offset each call) never trips the guard.

Moat note: GREP_GUARD is keyed by host identity (not a transcript). Every call sources
live from the workspace via ripgrep; no growing message history is assumed.
"""
from __future__ import annotations

import shutil
import subprocess
import threading

from .access import FileAccess
from .registry import ToolEntry, ToolText

# {host_id: {"last_key": tuple | None, "consecutive": int}} — module-level, per-host.
# Mirrors Hermes' per-task _read_tracker shape. Not a transcript: a tiny durable counter.
GREP_GUARD: dict = {}

_BLOCK_AFTER = 4          # 4th identical-in-a-row call is blocked (mirrors Hermes count>=4)
_DEFAULT_LIMIT = 50
_RG_MAX_FILESIZE = "300K"
_RG_MAX_COLUMNS = "400"


_GREP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search file CONTENTS for a regex pattern (ripgrep) under the workspace. Returns "
            "line-numbered matches (file:line:text). Results are paginated: pass offset/limit "
            "to page through a large result set. Use this to find code instead of reading whole "
            "files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for."},
                "path": {"type": "string", "description": "Subdirectory to search under (default: workspace root)."},
                "glob": {"type": "string", "description": "Optional file glob filter, e.g. '*.py'."},
                "offset": {"type": "integer", "description": "Number of leading result lines to skip (default 0)."},
                "limit": {"type": "integer", "description": "Max result lines to return (default 50)."},
            },
            "required": ["pattern"],
        },
    },
}


def _norm_int(value, default: int) -> int:
    """Coerce a model-supplied arg to a non-negative int, falling back to default."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n >= 0 else default


def _bump_guard(host_id, key) -> int:
    """Update the consecutive-identical counter for this host; return the run length."""
    if host_id not in GREP_GUARD and len(GREP_GUARD) > 256:
        GREP_GUARD.clear()   # bound the per-host map (no host-teardown hook + id() reuse) — resetting the
        #                      consecutive counter is harmless (worst case: one missed back-to-back warning)
    data = GREP_GUARD.setdefault(host_id, {"last_key": None, "consecutive": 0})
    if data["last_key"] == key:
        data["consecutive"] += 1
    else:
        data["last_key"] = key
        data["consecutive"] = 1
    return data["consecutive"]


def make_grep_tool(host) -> ToolEntry:
    """Build the guarded grep ToolEntry bound to a host (LocalToolHost-like: root()/_resolve)."""

    def handler(args: dict) -> str:
        pattern = args.get("pattern") or ""
        if not pattern:
            return "grep: no pattern given — pass a 'pattern' to search for."
        path = args.get("path") or "."
        glob = args.get("glob") or ""
        offset = _norm_int(args.get("offset"), 0)
        limit = _norm_int(args.get("limit"), _DEFAULT_LIMIT)
        if limit <= 0:
            limit = _DEFAULT_LIMIT

        # Consecutive-identical guard. Key includes offset so paging never trips it.
        key = (pattern, path, glob, limit, offset)
        count = _bump_guard((id(host), threading.get_ident()), key)   # per-THREAD: parallel explorers share the host but must not cross-contaminate the consecutive-grep counter
        if count >= _BLOCK_AFTER:
            return (
                f"BLOCKED: you have run this exact grep {count} times in a row and the results "
                "have NOT changed. STOP re-searching — use what you already have, or change the "
                "pattern/path/glob/offset."
            )

        # Confine the search target under the workspace root (rejects escapes).
        try:
            target = host._resolve(path)
        except (PermissionError, ValueError) as e:
            return ToolText(f"Error: {e}", ok=False)   # ok=False so a repeated boundary-escape is seen by the failure guardrail

        rg = shutil.which("rg")
        if not rg:
            # Quiet, non-failing: degrade gracefully when ripgrep is absent.
            return "grep: ripgrep (rg) is not available in this environment; no results."

        cmd = [
            rg, "-n",
            "--max-filesize", _RG_MAX_FILESIZE,
            "--max-columns", _RG_MAX_COLUMNS,
        ]
        if glob:
            cmd += ["--glob", glob]
        cmd += ["--", pattern, target]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=host.root(), timeout=30
            )
        except (OSError, subprocess.SubprocessError) as e:
            return f"grep: search failed ({e}); no results."

        # rg exit codes: 0 = matches, 1 = no matches (not an error), 2 = real error.
        if proc.returncode == 1:
            return "grep: no matches found."
        if proc.returncode not in (0, 1):
            err = (proc.stderr or "").strip()
            return f"grep: no matches found.{(' (' + err + ')') if err else ''}"

        lines = [ln for ln in proc.stdout.splitlines() if ln]
        if not lines:
            return "grep: no matches found."

        total = len(lines)
        window = lines[offset:offset + limit]
        if not window:
            return f"grep: no results at offset={offset} ({total} total matches)."

        body = "\n".join(window)
        if offset + limit < total:
            next_offset = offset + limit
            body += f"\n\n[truncated; use offset={next_offset} to see more]"
        return body

    return ToolEntry(
        name="grep",
        schema=_GREP_SCHEMA,
        handler=handler,
        accesses=lambda args: [FileAccess("search", args.get("path") or ".", recursive=True)],
        source="builtin",
    )
