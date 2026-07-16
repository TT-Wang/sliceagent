"""Project/focus-root `grep` tool with bounded ripgrep pagination.

The single discovery-on-demand seam (W7 deleted `code_index.snippets`; the model now
greps for content instead). It runs ripgrep, slices result lines by offset/limit, and
appends an explicit continuation hint when more results remain. Repeating a search is
an ordinary live observation; the host never turns repetition into a refusal.

Every call sources live from the current project or a grounded absolute focus root; no transcript or
cross-call admission state is involved.
"""
from __future__ import annotations

from .platform_compat import IS_WINDOWS, norm_rel
import shutil
import subprocess

from .access import FileAccess
from .execution import ToolStatus
from .reach import ReachSteer
from .registry import ToolEntry, ToolText

_DEFAULT_LIMIT = 50
_RG_MAX_FILESIZE = "300K"
_RG_MAX_COLUMNS = "400"


_GREP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search file CONTENTS for a regex pattern (ripgrep) in the current project or a grounded absolute "
            "focus root. `output_mode` shapes the "
            "result: 'content' (default — line-numbered file:line:text matches), 'files_with_matches' (just the "
            "matching file paths, newest-modified first), or 'count' (per-file match counts) — use the latter "
            "two to locate code cheaply before reading. Results paginate via offset/limit. Prefer this over "
            "reading whole files or bash grep."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for."},
                "path": {"type": "string", "description": (
                    "Project-relative subdirectory or grounded absolute focus root (default: project root)."
                )},
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


def make_grep_tool(host) -> ToolEntry:
    """Build the paginated grep ToolEntry bound to a LocalToolHost-like object."""

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

        # Virtual history/ namespace: no files on disk to ripgrep — scan the sealed turn docs in Python.
        hf = host._history_route(path) if hasattr(host, "_history_route") else None
        if hf is not None:
            virtual_path = host._archive_handle(path) if hasattr(host, "_archive_handle") else path
            return hf.grep(pattern, path=virtual_path, output_mode=mode, context=context,
                           offset=offset, limit=limit)

        # Resolve through the live ReachSet: project-relative paths plus grounded
        # narrow absolute focus roots, while blanket/sensitive escapes stay denied.
        try:
            target = host.resolve_read(path) if hasattr(host, "resolve_read") else host._resolve(path)
        except ReachSteer as e:
            return ToolText(str(e), status=ToolStatus.STEERED)
        except (PermissionError, ValueError) as e:
            return ToolText(f"Error: {e}", ok=False)

        rg = shutil.which("rg")
        if not rg:
            return ToolText(
                "grep: ripgrep (rg) is not available in this environment; search was not run.",
                status=ToolStatus.FAILED,
            )

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
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",  # H10: UTF-8, not locale
                cwd=host.root(), timeout=30
            )
        except (OSError, subprocess.SubprocessError) as e:
            return ToolText(f"grep: search failed ({e}); search was not completed.",
                            status=ToolStatus.FAILED)

        # rg exit codes: 0 = matches, 1 = no matches (not an error), 2 = real error.
        if proc.returncode == 1:
            return "grep: no matches found."
        if proc.returncode not in (0, 1):
            err = (proc.stderr or "").strip()
            detail = f" ({err})" if err else ""
            return ToolText(
                f"grep: ripgrep failed with exit code {proc.returncode}{detail}",
                status=ToolStatus.FAILED,
            )

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
            "Find files AND directories by NAME pattern in the current project or a grounded absolute focus "
            "root (for CONTENTS use grep). Matching "
            "folders (e.g. a 'hunter/' project dir) are returned too, listed first. Supports glob wildcards incl. "
            "brace sets, e.g. '*.py', 'src/**/*.{ts,tsx}', '*hunter*'. Paths most-recently-modified first, capped. "
            "Use to locate a file or project folder before opening it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "File-name glob, e.g. '*.py' or '**/*.{ts,tsx}'."},
                "path": {"type": "string", "description": (
                    "Project-relative subdirectory or grounded absolute focus root (default: project root)."
                )},
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
            target = host.resolve_read(path) if hasattr(host, "resolve_read") else host._resolve(path)
        except ReachSteer as e:
            return ToolText(str(e), status=ToolStatus.STEERED)
        except (PermissionError, ValueError) as e:
            return ToolText(f"Error: {e}", ok=False)

        rg = shutil.which("rg")
        files: list = []
        if rg:
            cmd = [rg] + (["--path-separator", "/"] if IS_WINDOWS else []) + ["--files", "--sortr", "modified", "-g", pattern, target]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",   # H10: UTF-8, not locale
                                       errors="replace", cwd=host.root(), timeout=30)
                if proc.returncode in (0, 1):          # 0 = files, 1 = none (not an error)
                    files = [ln for ln in proc.stdout.splitlines() if ln]
                else:
                    err = (proc.stderr or "").strip()
                    detail = f" ({err})" if err else ""
                    return ToolText(
                        f"glob: ripgrep failed with exit code {proc.returncode}{detail}",
                        status=ToolStatus.FAILED,
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                return ToolText(
                    f"glob: search failed ({exc}); search was not completed.",
                    status=ToolStatus.FAILED,
                )
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
