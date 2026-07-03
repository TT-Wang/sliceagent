"""Guarded `grep` tool — ripgrep pagination + a consecutive-search guard.

The single discovery-on-demand seam (W7 deleted `code_index.snippets`; the model now
greps for content instead). Two mechanisms:

- Grep pagination: run ripgrep, then slice the result lines by offset/limit and
  append an explicit "[truncated; use offset=N to see more]" notice when more
  remain.
- Consecutive-search guard (the `{last_key, consecutive}` tracker): the 4th
  identical-in-a-row call returns a BLOCKED message instead of re-running
  the same search forever. The key INCLUDES offset, so paging through truncated results
  (a *different* offset each call) never trips the guard.

Moat note: GREP_GUARD is keyed by host identity (not a transcript). Every call sources
live from the workspace via ripgrep; no growing message history is assumed.
"""
from __future__ import annotations

from .platform_compat import IS_WINDOWS, norm_rel
import shutil
import subprocess
import threading

from .access import FileAccess
from .registry import ToolEntry, ToolText

# {host_id: {"last_key": tuple | None, "consecutive": int}} — module-level, per-host.
# A per-task tracker shape. Not a transcript: a tiny durable counter.
GREP_GUARD: dict = {}
_GREP_LOCK = threading.Lock()   # parallel explorers share GREP_GUARD; serialize the check-then-update

_BLOCK_AFTER = 4          # 4th identical-in-a-row call is blocked (count>=4)
_DEFAULT_LIMIT = 50
_RG_MAX_FILESIZE = "300K"
_RG_MAX_COLUMNS = "400"


_GREP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search file CONTENTS for a regex pattern (ripgrep) under the workspace. `output_mode` shapes the "
            "result: 'content' (default — line-numbered file:line:text matches), 'files_with_matches' (just the "
            "matching file paths, newest-modified first), or 'count' (per-file match counts) — use the latter "
            "two to locate code cheaply before reading. Results paginate via offset/limit. Prefer this over "
            "reading whole files or bash grep."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for."},
                "path": {"type": "string", "description": "Subdirectory to search under (default: workspace root)."},
                "glob": {"type": "string", "description": "Optional file glob filter, e.g. '*.py' or '*.{ts,tsx}'."},
                "type": {"type": "string", "description": "Optional ripgrep file type, e.g. 'py', 'js', 'rust'."},
                "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"],
                                "description": "content (default) | files_with_matches | count."},
                "context": {"type": "integer", "description": "Lines of context around each match (content mode; like rg -C)."},
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
    """Update the consecutive-identical counter for this host; return the run length. Locked: parallel
    explorers share GREP_GUARD, so the check-then-clear and the counter update must be atomic (else a
    concurrent clear() blows away another thread's run length)."""
    with _GREP_LOCK:
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
        ftype = (args.get("type") or "").strip()
        mode = (args.get("output_mode") or "content").strip().lower()
        if mode not in ("content", "files_with_matches", "count"):
            mode = "content"
        context = _norm_int(args.get("context"), 0)
        offset = _norm_int(args.get("offset"), 0)
        limit = _norm_int(args.get("limit"), _DEFAULT_LIMIT)
        if limit <= 0:
            limit = _DEFAULT_LIMIT

        # Consecutive-identical guard. Key includes offset so paging never trips it.
        key = (pattern, path, glob, ftype, mode, context, limit, offset)
        count = _bump_guard((id(host), threading.get_ident()), key)   # per-THREAD: parallel explorers share the host but must not cross-contaminate the consecutive-grep counter
        if count >= _BLOCK_AFTER:
            return (
                f"BLOCKED: you have run this exact grep {count} times in a row and the results "
                "have NOT changed. STOP re-searching — use what you already have, or change the "
                "pattern/path/glob/type/output_mode/offset."
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

        cmd = [rg] + (["--path-separator", "/"] if IS_WINDOWS else [])  # model-facing paths: '/' on all platforms
        if mode == "files_with_matches":
            cmd += ["-l", "--sortr", "modified"]      # just the files, newest-changed first (cheap relevance)
        elif mode == "count":
            cmd += ["-c"]
        else:
            cmd += ["-n"]
            if context > 0:
                cmd += ["-C", str(context)]
        cmd += ["--max-filesize", _RG_MAX_FILESIZE, "--max-columns", _RG_MAX_COLUMNS]
        if glob:
            cmd += ["--glob", glob]
        if ftype:
            cmd += ["--type", ftype]
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


# ── glob: find files by NAME (the discovery companion to grep's find-by-CONTENT) ──────────────────────
_GLOB_DEFAULT_LIMIT = 100
_GLOB_IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ruff_cache", ".pytest_cache",
                     ".next", ".turbo", ".parcel-cache", ".nuxt", ".svelte-kit", ".output", "dist", "build"}

_GLOB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "glob",
        "description": (
            "Find files AND directories by NAME pattern under the workspace (for CONTENTS use grep). Matching "
            "folders (e.g. a 'hunter/' project dir) are returned too, listed first. Supports glob wildcards incl. "
            "brace sets, e.g. '*.py', 'src/**/*.{ts,tsx}', '*hunter*'. Paths most-recently-modified first, capped. "
            "Use to locate a file or project folder before opening it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "File-name glob, e.g. '*.py' or '**/*.{ts,tsx}'."},
                "path": {"type": "string", "description": "Subdirectory to search under (default: workspace root)."},
                "limit": {"type": "integer", "description": "Max paths to return (default 100)."},
            },
            "required": ["pattern"],
        },
    },
}


def _expand_braces(pat: str) -> list:
    """Expand ONE level of brace alternation ('*.{ts,tsx}' -> ['*.ts','*.tsx']); recurse for nested."""
    import re as _re
    m = _re.search(r"\{([^{}]*)\}", pat)
    if not m:
        return [pat]
    pre, post = pat[:m.start()], pat[m.end():]
    out: list = []
    for opt in m.group(1).split(","):
        out.extend(_expand_braces(pre + opt + post))
    return out


def _glob_walk(root: str, pattern: str, cap: int) -> list:
    """ripgrep-free fallback: os.walk + brace-expanded fnmatch, newest-modified first."""
    import fnmatch
    import os as _os
    pats = _expand_braces(pattern)
    hits: list = []
    for dp, dns, fns in _os.walk(root):
        dns[:] = [d for d in dns if d not in _GLOB_IGNORE_DIRS]
        for fn in fns:
            full = _os.path.join(dp, fn)
            rel = _os.path.relpath(full, root)
            if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(fn, p) for p in pats):
                hits.append(norm_rel(full))
        if len(hits) >= cap * 4:                       # gather generously, then mtime-sort + cap
            break
    hits.sort(key=lambda f: -(_os.path.getmtime(f) if _os.path.exists(f) else 0.0))
    return hits


def _dir_matches(root: str, pats: list, cap: int, maxdepth: int = 12) -> list:
    """Find DIRECTORIES whose name (or relpath) matches the pattern. `rg --files` lists FILES only, so a
    project FOLDER named like the pattern (e.g. a 'hunter/' dir) is otherwise invisible to glob. BREADTH-
    first so a shallow match (Desktop/hunter) is found before descending huge deep trees (Library/…), and
    bounded by depth + a node budget so it can't run away on a big home directory."""
    import fnmatch
    import os as _os
    from collections import deque
    hits: list = []
    q: deque = deque([(root, 0)])
    budget = 40000
    while q and budget > 0:
        base, depth = q.popleft()
        budget -= 1
        try:
            entries = list(_os.scandir(base))
        except OSError:
            continue
        for e in entries:
            try:
                if not e.is_dir(follow_symlinks=False):   # no symlink loops; dirs only
                    continue
            except OSError:
                continue
            if e.name in _GLOB_IGNORE_DIRS:
                continue
            rel = _os.path.relpath(e.path, root)
            if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(e.name, p) for p in pats):
                hits.append(norm_rel(e.path) + "/")
                if len(hits) >= cap:
                    return hits
            if depth + 1 < maxdepth:
                q.append((e.path, depth + 1))
    return hits


def make_glob_tool(host) -> ToolEntry:
    """Build the file-name `glob` ToolEntry bound to a host (uses ripgrep --files, falls back to os.walk)."""

    def handler(args: dict) -> str:
        pattern = (args.get("pattern") or "").strip()
        if not pattern:
            return "glob: no pattern given — pass a 'pattern' like '*.py'."
        path = args.get("path") or "."
        limit = _norm_int(args.get("limit"), _GLOB_DEFAULT_LIMIT) or _GLOB_DEFAULT_LIMIT
        try:
            target = host._resolve(path)
        except (PermissionError, ValueError) as e:
            return ToolText(f"Error: {e}", ok=False)

        rg = shutil.which("rg")
        files: list = []
        if rg:
            cmd = [rg] + (["--path-separator", "/"] if IS_WINDOWS else []) + ["--files", "--sortr", "modified", "-g", pattern, target]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=host.root(), timeout=30)
                if proc.returncode in (0, 1):          # 0 = files, 1 = none (not an error)
                    files = [ln for ln in proc.stdout.splitlines() if ln]
            except (OSError, subprocess.SubprocessError):
                files = []
        else:
            files = _glob_walk(target, pattern, limit)   # graceful degrade when rg is absent

        # rg --files lists FILES only — so a DIRECTORY named like the pattern (a 'hunter/' project folder)
        # would be invisible. Always add matching directories so "glob *hunter*" finds folders too; show
        # them FIRST (a name search is usually after the folder, not files buried beneath it).
        dirs = _dir_matches(target, _expand_braces(pattern), limit)
        results = dirs + files
        if not results:
            return f"glob: nothing matches {pattern!r} (no files or directories)."
        total = len(results)
        body = "\n".join(results[:limit])
        if total > limit:
            body += f"\n\n[{total - limit} more not shown; narrow the pattern or path]"
        return body

    return ToolEntry(
        name="glob",
        schema=_GLOB_SCHEMA,
        handler=handler,
        accesses=lambda args: [FileAccess("search", args.get("path") or ".", recursive=True)],
        source="builtin",
    )
