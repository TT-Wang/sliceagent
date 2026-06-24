"""LocalToolHost — the default ToolHost.

Safe execution lives here: file ops are confined to the workspace root (no path
traversal out of it), and shell runs through a Sandbox backend (sandbox.py) — so
swapping in a container later never touches the loop. Authorization (which calls
are allowed at all) is separate: policy.py via the PermissionHook.

Note: Python's str.replace is literal, so str_replace has no $-pattern footgun
(unlike JS).
"""
from __future__ import annotations

import os
import re
import shlex
import tempfile

from .access import AllAccess, FileAccess
from .binsniff import looks_binary
from .fuzzy import fuzzy_find_unique
from .procman import ProcManager
from .registry import ToolEntry, ToolRegistry, ToolText
from .sandbox import LocalSandbox
from .terminal import SessionManager

# I1 PROVENANCE — host SELF-INFLICTED error sentinels. These name failures caused by the HOST's own
# guard rails (file-tool confinement, permission denial), NOT by a real bug in the user's code. Lesson
# mining filters pitfalls whose signature contains one of these so a turn whose only error was the
# agent hitting its OWN sandbox mines nothing (D2). Lower-cased substrings, matched task-agnostically;
# defined HERE (the source of these strings) so the denylist tracks the actual error messages.
HOST_ERROR_SENTINELS = (
    "path escapes workspace",
    "file tools are confined",
    "permission denied",
    "operation not permitted",
)

# Prepended to every execute_code script: the in-sandbox tool helpers (code-as-action,
# Hermes pattern). No imports needed by the model. The workspace is cwd and on sys.path,
# Strip a leading "cat -n" line-number prefix ("   123\t") from a str_replace snippet pasted back from the
# numbered OPEN FILES render. Only fires when EVERY non-blank line has one (clearly cat -n output, not real
# source), so a genuine match is never altered; used as a fallback in _t_str_replace.
_LINENO_PREFIX = re.compile(r"^[ \t]*\d+\t")


def _strip_line_numbers(text: str) -> str:
    lines = text.split("\n")
    nonblank = [ln for ln in lines if ln.strip()]
    if not nonblank or not all(_LINENO_PREFIX.match(ln) for ln in nonblank):
        return text
    return "\n".join(_LINENO_PREFIX.sub("", ln) if ln.strip() else ln for ln in lines)


def _numbered(text: str) -> str:
    """cat -n line numbers for read_file's RETURN, so the model gets file:line evidence IMMEDIATELY in-turn
    (same format as the OPEN FILES render). The number is a display prefix, NOT file content — str_replace
    strips a pasted prefix via _strip_line_numbers, so editing from a numbered read still matches."""
    return "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(text.splitlines(), 1))


# so `import <workspace_module>` works for testing freshly-written code.
_CODE_PRELUDE = '''\
import os as _os, sys as _sys, subprocess as _sp
_sys.path.insert(0, _os.getcwd())

def _confine(path):
    # Confine code-as-action file helpers to the workspace (cwd = workspace root in the sandbox). Without
    # this, an absolute path or ../ escape let execute_code read/write outside allowed_roots, bypassing the
    # file-tool boundary. Shell (run_command) stays unconfined by design; these in-code helpers do not.
    _p = _os.path.realpath(path)
    _root = _os.path.realpath(_os.getcwd())
    if _p != _root and not _p.startswith(_root + _os.sep):
        raise PermissionError(f"path escapes workspace: {path} (use run_command for paths outside it)")
    return path

def read_file(path):
    with open(_confine(path), encoding="utf-8") as _f: return _f.read()

def write_file(path, content):
    path = _confine(path)
    _d = _os.path.dirname(path)
    if _d: _os.makedirs(_d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as _f: _f.write(content)
    if content[:2] == "#!":  # a shebang script should be runnable (parity with the edit_file tool)
        try: _os.chmod(path, _os.stat(path).st_mode | 0o111)
        except OSError: pass
    return f"wrote {len(content)} bytes to {path}"

def append_file(path, content):
    path = _confine(path)
    _d = _os.path.dirname(path)
    if _d: _os.makedirs(_d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as _f: _f.write(content)
    return f"appended {len(content)} bytes to {path}"

def str_replace(path, old, new):
    path = _confine(path)
    with open(path, encoding="utf-8") as _f: _cur = _f.read()
    _n = _cur.count(old)
    if _n != 1: return (f"error: old_string occurs {_n}x in {path} (need exactly 1) — "
                        f"add surrounding lines to make it unique, or write_file the whole file")
    with open(path, "w", encoding="utf-8") as _f: _f.write(_cur.replace(old, new, 1))
    return f"replaced 1 occurrence in {path}"

def list_files(path="."):
    return sorted(_os.listdir(_confine(path)))

def run(cmd, timeout=60):
    _r = _sp.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    _o = (_r.stdout or "") + (_r.stderr or "")
    return _o if _r.returncode == 0 else f"[exit {_r.returncode}]\\n{_o}"
'''


def _fn(name: str, desc: str, props: dict, req: list[str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": desc,
                     "parameters": {"type": "object", "properties": props, "required": req}},
    }


# The FINDINGS-capture seam. Every tool call carries a 'note' — the model's distilled conclusion
# for this turn. It rides on the call the model is ALREADY making (no extra round-trip, unlike a
# dedicated note tool) and is folded into the slice's FINDINGS tier. This is how a Markov/slice
# agent gives a REASONING model its own prior conclusions back: the slice has no transcript, so
# without it the model re-derives the situation each turn (big reasoning bursts → slow). Reasoning
# models (e.g. deepseek) emit empty message content while tool-calling, so a tool ARG — not message
# text — is the only reliable capture point.
NOTE_PROP = {
    "note": {
        "type": "string",
        "description": ("Optional — usually leave EMPTY. Fill ONLY when this call established a NEW durable FACT "
                        "(root cause, a confirmed fix, a ruled-out hypothesis, or 'task done'), in <=15 words — a "
                        "conclusion, NOT the action you're taking. Saved across turns so you never re-derive it; "
                        "routine reads/edits need no note."),
    }
}


def with_note(schema: dict) -> dict:
    """Inject the 'note' arg (first, OPTIONAL) into a tool schema — the FINDINGS capture seam.
    Applied to EVERY tool the model sees, regardless of source (builtin/MCP/plugin/skill).
    Optional, not required: the model writes it only when it has a genuine durable fact, so the
    tier fills with conclusions — not the action-narration that forcing a note on every call
    produces (and which can self-reinforce loops)."""
    fn = schema.get("function") or {}
    params = fn.get("parameters") or {"type": "object", "properties": {}, "required": []}
    props = {**NOTE_PROP, **(params.get("properties") or {})}
    req = [r for r in (params.get("required") or []) if r != "note"]
    return {**schema, "function": {**fn, "parameters": {**params, "properties": props, "required": req}}}


# Build/VCS/cache directories that are pure noise to LIST — and FLOOD context on a real repo (the reason a
# whole-repo "review" derailed: find/list surfaced thousands of .venv/.ruff_cache paths). list_files prunes
# these so the model gets a clean map and doesn't fall back to raw `find`. Task-agnostic denylist (not a full
# .gitignore parse): covers the universal offenders. ripgrep (grep tool) is already .gitignore-aware natively.
_IGNORE_NAMES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".env", "node_modules", "__pycache__", ".ruff_cache",
    ".pytest_cache", ".mypy_cache", ".tox", ".idea", ".vscode", ".cache", "dist", "build", ".eggs", "htmlcov",
    ".DS_Store",
})
_IGNORE_SUFFIX = (".egg-info", ".pyc")
_LIST_CAP = 600   # bound recursive output so a huge tree can't flood the slice
# Credential/secret dirs the shell-path auto-grant (#31) must never widen file-tool reach into.
_SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".gpg", ".kube", ".docker", ".config", "keyrings", ".password-store"}


def _is_ignored(name: str) -> bool:
    return name in _IGNORE_NAMES or any(name.endswith(s) for s in _IGNORE_SUFFIX)


# Asset/binary/log files are noise in a structural MAP (they crowd out source); skipped from repo_map
# only (list_files still shows them). Generic, not task-specific.
_MAP_SKIP_SUFFIX = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".log", ".lock", ".bin",
                    ".so", ".dylib", ".o", ".class", ".woff", ".woff2", ".ttf", ".mp4", ".mov", ".zip",
                    ".tar", ".gz", ".whl", ".pyc", ".jsonl", ".csv", ".parquet")
# Code extensions — used ONLY to RANK directories by source-density so the map shows the real source
# tree first (a generic signal, identical across task types; never a task-category switch).
_CODE_SUFFIX = (".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".c", ".h", ".cc",
                ".cpp", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala", ".sh", ".lua", ".ml", ".ex",
                ".exs", ".clj", ".r", ".jl", ".vue", ".sql")


def repo_map(root: str, *, max_entries: int = 300, max_per_dir: int = 25) -> str:
    """A compact, ignore-aware STRUCTURAL MAP of the project (the world-state cache's tier-B resident
    store): directories with their files, pruned of VCS/venv/cache + asset/log noise, RANKED by source-
    density so the real code tree shows first and never gets starved by asset/log dirs. This is what
    kills cold-start — a 'review/understand the repo' task sees the structure RESIDENT instead of re-
    listing with find. Built ONCE per session (stable → prompt-cache warm); new files created mid-task
    surface via the LIVE worktree region. Over budget, late dirs collapse to a count (structure stays
    COMPLETE). '' if root is unusable; never raises."""
    if not root or not os.path.isdir(root):
        return ""
    rows: list[tuple[str, list[str], int, int]] = []  # (rel, files, total, code_count)
    try:
        for dirpath, dirnames, filenames in os.walk(root):  # symlinks not followed
            dirnames[:] = sorted(d for d in dirnames if not _is_ignored(d))
            files = sorted(f for f in filenames
                           if not _is_ignored(f) and not f.endswith(_MAP_SKIP_SUFFIX))
            if not files:
                continue
            rel = os.path.relpath(dirpath, root)
            code_count = sum(1 for f in files if f.endswith(_CODE_SUFFIX))
            rows.append((rel, files, len(files), code_count))
    except OSError:
        return ""
    if not rows:
        return ""
    # rank source-dense dirs first (so src/ beats docs/ assets), ties broken by path for stability
    rows.sort(key=lambda r: (-r[3], r[0]))
    lines, shown = [], 0
    for rel, files, total, _code in rows:
        prefix = "./" if rel == "." else rel + "/"
        if shown < max_entries:                       # detailed: list files (per-dir capped)
            take = files[:max_per_dir]
            shown += len(take)
            extra = f" (+{total - len(take)} more)" if total > len(take) else ""
            lines.append(f"{prefix} — {', '.join(take)}{extra}")
        else:                                         # over budget: keep the dir, collapse to a count
            lines.append(f"{prefix} — ({total} files)")
    return "\n".join(lines)


TOOL_SCHEMAS = [
    _fn("read_file",
        "Read and return a file's FULL contents with cat -n line numbers for reference (whole file — no "
        "line-range args; the leading number is NOT part of the file, so don't include it in a str_replace "
        "old_string). To list a directory use list_files; to SEARCH file contents use the `grep` tool "
        "(ripgrep-backed, paginated) — not bash grep. "
        "Arg `path` is workspace-relative or absolute but confined to the workspace — for outside paths use "
        "run_command. A binary file returns a hexdump preview, not editable text.",
        {"path": {"type": "string"}}, ["path"]),
    _fn("list_files",
        "List directory entries (ignore-aware: skips .git/.venv/caches/build/node_modules noise). Use to "
        "discover what exists; use read_file for a file's CONTENTS and the `grep` tool (ripgrep-backed) to "
        "SEARCH text. Pass recursive=true to map a whole subtree in ONE call (flat file paths, capped at 600 — "
        "pass a subdir to narrow) — PREFER this over shell `find` for a clean cache-free map.",
        {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, []),
    _fn("edit_file",
        "Create a new file, or OVERWRITE an existing file's ENTIRE contents with `content` (the complete text); "
        "parent dirs are auto-created and a leading `#!` shebang makes it executable. To change PART of an "
        "existing file use str_replace; to add to its end use append_to_file. Do NOT use edit_file to tweak a "
        "file — it discards all current content.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("append_to_file",
        "Append `content` verbatim to the END of a file (creates it + parent dirs if missing) — the only writer "
        "that ADDS without touching existing content. Use str_replace to modify text already in the file, "
        "edit_file to replace the whole file. No newline is added — include a leading '\\n' yourself if needed.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("code_review",
        "Review code changes: returns the `git diff` for the workspace (default vs HEAD; pass `ref` for a "
        "branch / commit / range like 'main', 'HEAD~3', or 'main...HEAD') so you can audit the changes for "
        "correctness, security, and edge cases — cite file:line for each issue you find. Read-only; needs a "
        "git repo. Prefer this over piecing a review together from many read_file calls.",
        {"ref": {"type": "string"}}, []),
    _fn("str_replace",
        "Make a SURGICAL edit to an EXISTING file — replace one snippet, leave the rest. The default for "
        "changing a file you've read. `old_string` should be the SMALLEST unique snippet — usually 2-4 adjacent "
        "lines, not 10+. It must identify exactly ONE place: more than one occurrence is rejected (add "
        "surrounding context); an exact match is used, else a unique whitespace-tolerant fuzzy "
        "match. Prefer over edit_file (whole-file overwrite). If a replace fails to match, don't retry it "
        "identically — rewrite the file with edit_file.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"]),
    _fn("run_command",
        "Run a shell command (blocking, cwd=workspace root); returns combined stdout+stderr (exit code on "
        "failure). Pass timeout (seconds, default 30, max 600) for slow builds. Use for one-shot commands that "
        "finish; for a process that must STAY alive use proc_start, for an interactive REPL use terminal_open, "
        "to chain several edits + a test in one turn use execute_code. No cwd arg — prepend `cd DIR &&`. The "
        "shell is unconfined (can reach outside the workspace, unlike the file tools).",
        {"command": {"type": "string"}, "timeout": {"type": "number"}}, ["command"]),
    _fn("execute_code",
        "Run a Python script that does SEVERAL file/shell steps in ONE turn (e.g. multiple edits + a test). Use "
        "over run_command when you'd chain many calls; over proc_start when it's one-shot (blocking, ~30s). "
        "Helpers (no imports): read_file(path), write_file(path, content), append_file(path, content), "
        "str_replace(path, old, new), list_files(path='.'), run(shell_cmd). Workspace is cwd + on sys.path. ONLY "
        "what you print() is returned. The file helpers are workspace-confined — use run() (shell) for outside paths.",
        {"code": {"type": "string"}}, ["code"]),
    _fn("ask_user",
        "Ask the user a concise follow-up question and WAIT for their answer (returned to you). Use this "
        "whenever you are UNSURE or the request is AMBIGUOUS, or when you have FAILED / been blocked and don't "
        "know how to proceed — instead of guessing or repeating a failing action; prefer just answering in text "
        "when you can infer intent. Give a few short 'options' for multiple-choice, or omit for open-ended. In "
        "headless/eval runs there is no interactive user — it returns a fallback telling you to proceed with a "
        "stated assumption, so never loop waiting on it.",
        {"question": {"type": "string"},
         "options": {"type": "array", "items": {"type": "string"}}}, ["question"]),
    _fn("proc_start",
        "Start a LONG-RUNNING / background process (a server, a watcher, a multi-minute build) and return a "
        "handle (p1, p2, …) immediately; it keeps running across turns. Use over run_command when the process "
        "must outlive the turn, over terminal_open when you only launch-and-probe (it gets no stdin). It does "
        "NOT confirm the process started — one that instantly dies still returns a handle — so "
        "proc_poll/proc_tail to check status and proc_kill to stop.",
        {"command": {"type": "string"}}, ["command"]),
    _fn("proc_poll", "Check a background process by handle: 'running' or 'exited <code>'.",
        {"handle": {"type": "string"}}, ["handle"]),
    _fn("proc_tail", "Read recent output (stdout+stderr) of a background process.",
        {"handle": {"type": "string"}, "lines": {"type": "number"}}, ["handle"]),
    _fn("proc_wait",
        "Wait up to timeout seconds for a background process to exit; returns its status + recent output.",
        {"handle": {"type": "string"}, "timeout": {"type": "number"}}, ["handle"]),
    _fn("proc_kill", "Terminate a background process and its child group.",
        {"handle": {"type": "string"}}, ["handle"]),
    _fn("terminal_open",
        "Open a persistent interactive PTY session for anything needing a LIVE terminal across turns: a "
        "REPL/text-game/TUI, answering successive prompts, or holding shell state (cd/export/venv). Unlike "
        "proc_start (no stdin) or run_command (one-shot), you drive it with terminal_send/terminal_wait/"
        "terminal_read and end with terminal_close. Omit command for a shell, or pass one (e.g. 'python3 -i -q'); "
        "'session' names it (default 'main'). Don't reopen an already-open session name — close it first.",
        {"session": {"type": "string"}, "command": {"type": "string"}}, []),
    _fn("terminal_send",
        "Send input to a terminal session. By default a newline is appended (sends a line). Set "
        "enter=false to send raw keys without a newline (e.g. a control char like '\\u0003' for Ctrl-C, "
        "or an escape sequence). Returns the immediate echo/output.",
        {"session": {"type": "string"}, "input": {"type": "string"}, "enter": {"type": "boolean"}},
        ["input"]),
    _fn("terminal_read", "Read the output a terminal session has produced (drains the live stream).",
        {"session": {"type": "string"}, "timeout": {"type": "number"}}, []),
    _fn("terminal_wait",
        "Wait until a regex pattern appears in a terminal session's output (or timeout) — the reliable "
        "way to sync: send a command, then wait for its prompt/result before sending the next.",
        {"session": {"type": "string"}, "until": {"type": "string"}, "timeout": {"type": "number"}},
        ["until"]),
    _fn("terminal_close", "Close a terminal session and kill its process group.",
        {"session": {"type": "string"}}, []),
    _fn("world_set",
        "Save DURABLE task state to your WORLD MODEL under a key (overwrites that key). Use it to maintain "
        "non-code state across turns: an explored maze map, a game's rooms+inventory, a system "
        "inventory, a running plan. It appears in the WORLD MODEL section of your context from your NEXT "
        "turn on; within THIS turn, re-read a value from your own world_set call above. value may be multiline.",
        {"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]),
    _fn("world_clear", "Remove a key from your WORLD MODEL (omit key to clear all of it).",
        {"key": {"type": "string"}}, []),
    _fn("require",
        "Record a STANDING REQUIREMENT that must HOLD when the task is done — an exact name/signature, an "
        "output format, a stated rule, or a constraint the user adds. It joins your STANDING REQUIREMENTS "
        "contract (shown every turn from your next turn on, and the bar for 'done'). Record only DURABLE "
        "constraints, never transient sub-steps or chit-chat; re-recording the same one is a no-op.",
        {"text": {"type": "string"}}, ["text"]),
    _fn("requirement_done",
        "Mark a STANDING REQUIREMENT satisfied (after verifying it against the real end-state). It stays "
        "shown as '[x] done' so it is not re-flagged but not forgotten. `text` must match the requirement.",
        {"text": {"type": "string"}}, ["text"]),
    _fn("drop_requirement",
        "Remove a STANDING REQUIREMENT the user RETRACTED or that no longer applies. `text` must match.",
        {"text": {"type": "string"}}, ["text"]),
    _fn("update_plan",
        "Maintain an ordered PLAN (a TODO list) for a multi-step task. Pass the COMPLETE list of steps "
        "every time — it REPLACES the previous plan. Keep exactly ONE step 'in_progress'; mark each 'done' "
        "as you finish it. The plan shows in your PLAN section across turns so progress survives and the "
        "user can follow along. Use it for non-trivial multi-step work; skip it for a single action.",
        {"steps": {"type": "array", "description": "the full ordered step list (replaces the prior plan)",
                   "items": {"type": "object", "properties": {
                       "step": {"type": "string", "description": "one concrete step, imperative"},
                       "status": {"type": "string", "enum": ["pending", "in_progress", "done"]}},
                       "required": ["step", "status"]}}},
        ["steps"]),
    _fn("set_mission",
        "Set your MISSION — the overarching NORTH-STAR objective for a long multi-step task (the 'why'), "
        "shown at the top of your context every turn so you stay oriented across many steps. Set it once at "
        "the start of a substantial task; it is ABOVE the literal task and your step plan. Re-setting "
        "replaces it. Skip it for quick one-off requests.",
        {"text": {"type": "string"}}, ["text"]),
    _fn("mission_done", "Clear your MISSION once the overarching objective is achieved (it stops showing).",
        {}, []),
]


def _default_ask_user(question: str, options) -> str:
    """Fallback when no interactive user is wired (headless/eval) — never hangs."""
    return ("(no interactive user is available to answer; proceed with your best assumption and "
            "STATE it explicitly, or stop with a clear summary of what you need)")


def _sniff_image_mime(raw: bytes) -> str | None:
    """Identify an image by MAGIC BYTES (not extension). Returns the MIME type or None if not an image."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:2] == b"BM":
        return "image/bmp"
    return None


class LocalToolHost:
    def __init__(self, root: str | None = None, *, sandbox=None, timeout: int = 30,
                 registry: ToolRegistry | None = None):
        # root=None → confine to the *current* working directory, resolved per call
        # (so the eval runner, which chdirs into a temp workdir after construction,
        # is confined to that workdir). Pass an explicit root to pin it.
        self._root = root
        self.timeout = timeout
        self.sandbox = sandbox or LocalSandbox()
        # Background/long-running processes — the live-handle registry the one-shot sandbox can't
        # express (servers, multi-minute builds). Scrubs secrets like the sandbox; cleanup() at exit.
        _scrub = getattr(self.sandbox, "scrub_secrets", True)
        self.procs = ProcManager(scrub_secrets=_scrub)
        # Interactive PTY sessions — drive REPLs/TUIs/games, hold shell+env across turns.
        self.terminals = SessionManager(scrub_secrets=_scrub)
        # I2 — RE-OBSERVATION REACH = ACTION REACH. File tools and shell must reach the
        # SAME places, or the agent writes (via shell, unconfined) files its file tools can
        # never read back, and OPEN FILES lies "(not created yet)" about real on-disk files.
        # `_extra_roots` holds dirs the goal/user EXPLICITLY targets (added via add_root):
        # _resolve accepts a path under the workspace root OR any extra root. Explicit and
        # bounded — never a blanket '/'; the workspace stays the default and only the launch
        # dir is implicit. Task-agnostic (we don't parse the goal) and safe (opt-in).
        self._extra_roots: list[str] = []
        # ask_user (the "come back and ask" capability): a host callback that prompts the real user and
        # returns their answer. Defaults to a non-interactive fallback so headless/eval never hangs; the
        # CLI overrides it with a TUI/plain prompt. Injected (not a core dependency) — task/LLM-agnostic.
        self.on_ask_user = _default_ask_user
        self._edit_journal: list = []   # (rel, full, prev_bytes|None) per write — powers /undo
        self.pending_images: list = []  # images @-attached for the NEXT seed build (vision models only)
        # The registry is the single source of tools; MCP/plugin/skill tools register
        # into this same object later (Step ③). The host just projects from it.
        self.registry = registry or ToolRegistry()
        self._register_builtins()
        import atexit
        atexit.register(self.cleanup)   # leaked background procs / PTYs must not survive exit/abort/crash

    def cleanup(self) -> None:
        """Tear down background processes + PTY sessions (idempotent; never raises). Wired to atexit AND
        called by the CLI on exit/abort, so leaked servers/shells/PTYs don't outlive the agent (#5)."""
        for _mgr in (getattr(self, "procs", None), getattr(self, "terminals", None)):
            try:
                if _mgr is not None:
                    _mgr.cleanup()
            except Exception:  # noqa: BLE001
                pass

    def _register_builtins(self) -> None:
        handlers = {
            "read_file": self._t_read_file, "list_files": self._t_list_files,
            "edit_file": self._t_edit_file, "append_to_file": self._t_append,
            "str_replace": self._t_str_replace, "run_command": self._t_run_command,
            "execute_code": self._t_execute_code, "ask_user": self._t_ask_user,
            "proc_start": self._t_proc_start, "proc_poll": self._t_proc_poll,
            "proc_tail": self._t_proc_tail, "proc_wait": self._t_proc_wait,
            "proc_kill": self._t_proc_kill,
            "terminal_open": self._t_terminal_open, "terminal_send": self._t_terminal_send,
            "terminal_read": self._t_terminal_read, "terminal_wait": self._t_terminal_wait,
            "terminal_close": self._t_terminal_close,
            "world_set": self._t_world_set, "world_clear": self._t_world_clear,
            "require": self._t_require, "requirement_done": self._t_requirement_done,
            "drop_requirement": self._t_drop_requirement, "update_plan": self._t_update_plan,
            "set_mission": self._t_set_mission, "mission_done": self._t_mission_done,
            "code_review": self._t_code_review,
        }
        for schema in TOOL_SCHEMAS:
            name = schema["function"]["name"]
            self.registry.register(ToolEntry(
                name=name, schema=schema, handler=handlers[name],
                accesses=(lambda args, n=name: self._builtin_accesses(n, args)),
                source="builtin",
            ))

    def root(self) -> str:
        return os.path.realpath(self._root or os.getcwd())

    def add_root(self, path: str) -> str | None:
        """Mark a directory the goal/user EXPLICITLY targets as in-reach for file tools.

        The minimal, safe, task-agnostic mechanism for "explicitly-targeted dir" (I2): a
        SETTABLE root, not goal-parsing heuristics. After this, read_file/edit_file/list_files
        resolve paths under `path` exactly as the shell already does (shell is unconfined),
        so a shell-written file is always readable back through OPEN FILES — reach matches.
        Refuses a blanket root ('/' or '~') so the workspace boundary is never erased.
        Returns the realpath added (idempotent), or None if rejected/unusable."""
        if not path:
            return None
        full = os.path.realpath(os.path.expanduser(path))
        # never widen reach to the whole filesystem or the bare home dir
        if full == os.sep or full == os.path.realpath(os.path.expanduser("~")):
            return None
        if full == self.root() or full in self._extra_roots:
            return full
        self._extra_roots.append(full)
        return full

    def allowed_roots(self) -> list[str]:
        """The set of dirs file tools may reach: the workspace root ∪ explicitly-targeted dirs.
        Honored by `_resolve`; matches where the shell already acts (I2: reach = action reach)."""
        roots = [self.root()]
        for r in self._extra_roots:
            if r not in roots:
                roots.append(r)
        return roots

    def _grant_shell_paths(self, text: str) -> None:
        """I2 — reach FOLLOWS action. When the shell acts on a path outside the allowed roots,
        grant file-tool reach to its directory so a shell-written file is ALWAYS readable back via
        OPEN FILES. No NEW capability — the shell already reaches there; this only lets the file
        tools observe it (the original split-brain: writes it could never read back). Restricted to
        the user's HOME subtree, never HOME itself or an ancestor of the workspace (add_root also
        refuses '/' and '~'). Pure path detection — task/LLM-agnostic, no command parsing."""
        if not text:
            return
        home = os.path.realpath(os.path.expanduser("~"))
        root = self.root()
        # quoted paths (may contain spaces) OR bare ~/-rooted tokens up to a shell metachar/space
        for q, uq in re.findall(
                r"""['"]([^'"]*/[^'"]*)['"]|(?<![\w'"])((?:~|/)[^\s'"|&;<>()]+)""", text):
            cand = (q or uq).strip()
            if not (cand.startswith("/") or cand.startswith("~")):
                continue
            full = os.path.realpath(os.path.expanduser(cand))
            d = full if os.path.isdir(full) else os.path.dirname(full)
            if not d or not os.path.isdir(d):
                continue
            if not d.startswith(home + os.sep):          # only the user's own subtree (excludes HOME itself)
                continue
            if d == root or root.startswith(d + os.sep):  # never an ancestor of the workspace
                continue
            # #31: never auto-widen file-tool reach into credential/secret dirs, even inside HOME — a path
            # merely MENTIONED in an allowed shell command must not make ~/.ssh etc. readable by the tools.
            if any(part in _SECRET_DIRS for part in d.split(os.sep)):
                continue
            self.add_root(d)

    def _resolve(self, path: str) -> str:
        """Resolve a tool path under an ALLOWED root (workspace ∪ explicitly-targeted dirs);
        reject escapes. expanduser FIRST so '~' behaves like the shell (P2) instead of
        silently creating a literal '~' dir inside the workspace."""
        if not path:
            raise ValueError("empty path")
        path = os.path.expanduser(path)  # P2 — '~' → $HOME before any join/realpath
        roots = self.allowed_roots()
        primary = roots[0]
        full = path if os.path.isabs(path) else os.path.join(primary, path)
        full = os.path.realpath(full)
        for root in roots:
            if full == root or full.startswith(root + os.sep):
                return full
        # P3 — prescriptive error: name the boundary AND the escape hatch so a no-transcript
        # model recovers instead of re-deriving the dead end (and looping into shell fallback).
        raise PermissionError(
            f"path escapes workspace ({primary}): {path} — File tools are confined to the "
            "workspace. To act on paths outside it, use run_command/execute_code (shell is "
            "unconfined), or re-run memagent with the workspace set to that directory.")

    # --- ToolHost projection: everything comes from the registry now ---
    def schemas(self) -> list[dict]:
        # inject the 'note' arg into every tool so the model's per-turn conclusion rides on the
        # call it already makes and lands in the slice's FINDINGS tier (anti-re-derivation)
        return [with_note(s) for s in self.registry.schemas()]

    def accesses(self, name: str, args: dict) -> list:
        return self.registry.accesses(name, args)

    def run(self, name: str, args: dict) -> str:
        return self.registry.run(name, args)  # registry wraps the handler in try/except

    def read_text(self, path: str) -> str:
        # Read bytes first so the binary gate runs BEFORE we trust the file as text.
        # A NUL byte / mostly-control-char head means "not text" — feeding it through
        # OPEN FILES would corrupt the slice and burn tokens. ValueError flows through
        # the registry try/except so both read_file and str_replace degrade gracefully.
        full = self._resolve(path)
        with open(full, "rb") as f:
            raw = f.read()
        sample = raw[:8192].decode("utf-8", errors="replace")
        if looks_binary(path, sample):
            raise ValueError(f"{path} appears to be binary; not shown")
        return raw.decode("utf-8")

    def _builtin_accesses(self, name: str, args: dict) -> list:
        """Declare what each builtin call touches so the scheduler can safely parallelize."""
        p = args.get("path")
        if name == "read_file":
            return [FileAccess("read", p)] if p else []
        if name == "list_files":
            return [FileAccess("search", args.get("path") or ".", recursive=True)]
        if name in ("edit_file", "append_to_file", "str_replace"):
            return [FileAccess("readwrite", p)] if p else [AllAccess()]
        if name in ("run_command", "execute_code", "proc_start", "proc_poll",
                    "proc_tail", "proc_wait", "proc_kill", "terminal_open", "terminal_send",
                    "terminal_read", "terminal_wait", "terminal_close"):
            return [AllAccess()]  # arbitrary / stateful execution → globally exclusive
        return [AllAccess()]

    # --- builtin tool handlers (args) -> str (the registry catches exceptions) ---
    def _t_read_file(self, args: dict) -> str:
        # Text files: return the content. Binary files: instead of refusing (which blanks the
        # agent on forensics/media/archive tasks), return a hexdump + size + magic so it can
        # inspect structure and pick the right CLI. str_replace still uses read_text() (which
        # raises on binary) — you can't text-edit a binary, so that path stays a hard error.
        path = args["path"]
        full = self._resolve(path)
        with open(full, "rb") as f:
            raw = f.read()
        sample = raw[:8192].decode("utf-8", errors="replace")
        if looks_binary(path, sample):
            return self._binary_view(path, raw)
        # Return WITH cat -n line numbers so the model has file:line evidence immediately this turn (matching
        # the OPEN FILES render). Safe for editing: str_replace strips a pasted line-number prefix.
        return _numbered(raw.decode("utf-8"))

    @staticmethod
    def _binary_view(path: str, raw: bytes, head_bytes: int = 256) -> str:
        head = raw[:head_bytes]
        rows = []
        for off in range(0, len(head), 16):
            chunk = head[off:off + 16]
            hexpart = " ".join(f"{b:02x}" for b in chunk)
            asciipart = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            rows.append(f"{off:08x}  {hexpart:<47}  {asciipart}")
        return (f"{path}: binary file, {len(raw)} bytes — text tools can't edit it; inspect/convert "
                f"it with run_command/execute_code (the right CLI).\n"
                f"magic: {head[:8].hex()}\n"
                f"hexdump (first {len(head)} bytes):\n" + "\n".join(rows))

    @staticmethod
    def _detect_crlf(full: str) -> bool:
        """True if the existing file uses Windows CRLF line endings (sample the head). Used to PRESERVE
        line endings on edit: the model emits '\\n', and writing that to a CRLF file rewrites every line
        ending — a huge spurious diff / corruption on Windows-authored repos. Borrowed from Kimi kaos."""
        try:
            with open(full, "rb") as f:
                return b"\r\n" in f.read(65536)
        except OSError:
            return False

    @staticmethod
    def _preserve_eol(text: str, crlf: bool) -> str:
        """Convert `text` to CRLF iff the target file is CRLF (normalize first → idempotent, handles
        mixed input). No-op for the common LF case, so LF files never gain spurious '\\r'."""
        return text.replace("\r\n", "\n").replace("\n", "\r\n") if crlf else text

    def _t_list_files(self, args: dict) -> str:
        base = self._resolve(args.get("path") or ".")
        if not args.get("recursive"):
            entries = sorted(os.listdir(base))
            shown = [e + "/" if os.path.isdir(os.path.join(base, e)) else e
                     for e in entries if not _is_ignored(e)]
            hidden = [e for e in entries if _is_ignored(e)]
            body = "\n".join(shown) or "(empty)"
            if hidden:  # name them so the model KNOWS they exist (recoverable), without flooding
                body += f"\n(+{len(hidden)} ignored: {', '.join(hidden[:6])})"
            return body
        # recursive: a clean, ignore-pruned, bounded repo MAP — the native alternative to shell `find`
        rels: list[str] = []
        capped = False
        for dirpath, dirnames, filenames in os.walk(base):  # symlinks not followed (no .venv loops)
            dirnames[:] = sorted(d for d in dirnames if not _is_ignored(d))  # prune in place → don't descend
            rel = os.path.relpath(dirpath, base)
            for f in sorted(filenames):
                if _is_ignored(f):
                    continue
                rels.append(f if rel == "." else os.path.join(rel, f))
                if len(rels) >= _LIST_CAP:
                    capped = True
                    break
            if capped:
                break
        body = "\n".join(sorted(rels)) or "(empty)"
        if capped:
            body += f"\n(+more — capped at {_LIST_CAP}; pass a subdirectory path to narrow)"
        return body

    def _t_edit_file(self, args: dict) -> str:
        full = self._resolve(args["path"])
        self._mkparent(full)
        content = args["content"]
        if os.path.exists(full):                      # preserve the file's existing line endings (CRLF)
            content = self._preserve_eol(content, self._detect_crlf(full))
        self._journal(args["path"], full)
        self._atomic_write(full, content)
        if content[:2] == "#!":          # a shebang script should be runnable (general, task-agnostic)
            self._make_executable(full)
        return f"Wrote {len(content)} bytes to {args['path']}"

    def _make_executable(self, full: str) -> None:
        """chmod +x a freshly-written shebang script (a script the agent declared executable via '#!'
        should run without a separate chmod). Best-effort; never fails the write."""
        try:
            import stat as _stat
            os.chmod(full, os.stat(full).st_mode | _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH)
        except OSError:
            pass

    def _t_append(self, args: dict) -> str:
        full = self._resolve(args["path"])
        self._mkparent(full)
        self._journal(args["path"], full)
        with open(full, "a", encoding="utf-8") as f:
            f.write(args["content"])
        return f"Appended {len(args['content'])} bytes to {args['path']}"

    def _t_str_replace(self, args: dict) -> str:
        full = self._resolve(args["path"])
        cur = self.read_text(args["path"])
        crlf = self._detect_crlf(full)                # preserve the file's line endings on write-back
        old = args["old_string"]
        new = args["new_string"]
        # OPEN FILES renders with cat -n line numbers; if the model pasted a numbered snippet back into
        # old_string, strip the "  N\t" prefixes so it still matches the real (unnumbered) file. Tried only
        # as a FALLBACK after the raw text, and only when EVERY line carried a number (clearly cat -n output,
        # not source) — so a real match is never altered.
        candidates = [old]
        stripped = _strip_line_numbers(old)
        if stripped != old:
            candidates.append(stripped)
        # PRIMARY: exact match (raw first, then de-numbered). A >1 count is a real ambiguity — report it.
        for cand in candidates:
            n = cur.count(cand)
            if n == 1:
                updated = self._preserve_eol(cur.replace(cand, new, 1), crlf)
                self._journal(args["path"], full)
                self._atomic_write(full, updated)
                return f"Replaced 1 occurrence in {args['path']} ({len(cur)} → {len(updated)} bytes)"
            if n > 1:
                return ToolText(f"Error: old_string occurs {n} times in {args['path']}; add context to make it unique", ok=False)
        # FALLBACK: whitespace-tolerant UNIQUE fuzzy span (raw first, then de-numbered). fuzzy_find_unique
        # returns None on 0/>1 candidates, so uniqueness is preserved — we never replace an ambiguous match.
        for cand in candidates:
            span = fuzzy_find_unique(cur, cand)
            if span is not None:
                updated = self._preserve_eol(cur[:span[0]] + new + cur[span[1]:], crlf)
                self._journal(args["path"], full)
                self._atomic_write(full, updated)
                return (f"Replaced 1 occurrence (normalized/fuzzy match) in {args['path']} "
                        f"({len(cur)} → {len(updated)} bytes)")
        return ToolText(f"Error: old_string not found in {args['path']} — your snippet does not match "
                f"the file. Copy the EXACT text from OPEN FILES (the live content, WITHOUT the line-number "
                f"prefix), or rewrite the whole file with edit_file. Do NOT retry the same str_replace.", ok=False)

    # --- edit journal (powers /undo) -----------------------------------------
    def _journal(self, rel: str, full: str) -> None:
        """Record a file's pre-image (or None if it didn't exist) just before a write, so /undo can revert
        the most recent edit. Bounded ring — recent edits only, never an unbounded history."""
        try:
            prev = open(full, "rb").read() if os.path.exists(full) else None
        except OSError:
            prev = None
        self._edit_journal.append((rel, full, prev))
        if len(self._edit_journal) > 50:
            del self._edit_journal[:-50]

    def undo_last(self) -> str:
        """Revert the most recent journaled edit. Returns a human-readable result for the UI."""
        if not self._edit_journal:
            return "Nothing to undo."
        rel, full, prev = self._edit_journal.pop()
        try:
            if prev is None:
                if os.path.exists(full):
                    os.remove(full)
                return f"Undid: removed {rel} (it did not exist before that edit)."
            with open(full, "wb") as f:
                f.write(prev)
            return f"Undid the last edit to {rel} ({len(prev)} bytes restored)."
        except OSError as e:
            return f"Undo failed for {rel}: {e}"

    def attach_image(self, path: str) -> str:
        """Stash a workspace image for the NEXT seed build as a vision content part. Returns a status line.
        Gated by the caller (only called for a vision-capable model). Confined to the workspace like reads.
        The MIME type is sniffed from MAGIC BYTES (not the extension), so a spoofed extension can't smuggle a
        non-image through as image/png."""
        import base64
        try:
            full = self._resolve(path)
            raw = open(full, "rb").read()
        except OSError as e:
            return f"Error: cannot read image {path}: {e}"
        if len(raw) > 8 * 1024 * 1024:
            return f"Error: image {path} is {len(raw)} bytes (cap 8MB) — too large to attach"
        mime = _sniff_image_mime(raw)
        if mime is None:
            return f"Error: {path} is not a recognized image (png/jpeg/gif/webp/bmp) — not attached"
        self.pending_images.append({"path": path, "b64": base64.b64encode(raw).decode("ascii"), "mime": mime})
        return f"attached image {path} ({len(raw)} bytes, {mime})"

    def _t_code_review(self, args: dict) -> str:
        """Return the git diff for the workspace so the model can review it (read-only; task-agnostic)."""
        import subprocess
        ref = (args.get("ref") or "HEAD").strip() or "HEAD"
        try:
            p = subprocess.run(["git", "-C", self.root(), "diff", ref],
                               capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            return ToolText("Error: git is not installed.", ok=False)
        except subprocess.SubprocessError as e:
            return ToolText(f"Error: git diff failed ({type(e).__name__}: {e}).", ok=False)
        if p.returncode != 0:
            return ToolText(f"Error: `git diff {ref}` failed — {p.stderr.strip()[:300]} "
                            "(is this a git repo? is the ref valid?)", ok=False)
        diff = p.stdout
        if not diff.strip():
            return f"No changes vs {ref} — the working tree matches it. Nothing to review."
        cap = 20000
        body = diff if len(diff) <= cap else diff[:cap] + f"\n… (diff truncated at {cap} of {len(diff)} chars)"
        return (f"git diff {ref} ({len(diff)} chars). Review for correctness, security, and edge cases; "
                f"cite file:line per issue.\n\n{body}")

    def _t_ask_user(self, args: dict) -> str:
        q = (args.get("question") or "").strip()
        if not q:
            return ToolText("Error: ask_user requires a non-empty 'question'.", ok=False)
        opts = args.get("options")
        opts = [str(o) for o in opts] if isinstance(opts, list) and opts else None
        try:
            ans = (self.on_ask_user or _default_ask_user)(q, opts)
        except (EOFError, KeyboardInterrupt):
            ans = "(no answer)"
        return f"User answered: {str(ans).strip()}"

    def _t_run_command(self, args: dict) -> str:
        # Optional per-call timeout (default self.timeout, hard ceiling 600s) so slow builds don't
        # die at the 30s default and come back as exit 124. Long-lived processes use proc_start.
        try:
            t = float(args.get("timeout") or self.timeout)
        except (TypeError, ValueError):
            t = float(self.timeout)
        t = max(1.0, min(t, 600.0))
        code, out = self.sandbox.run(args["command"], cwd=self.root(), timeout=t)
        self._grant_shell_paths(args.get("command", ""))  # I2 reach=action: dirs the shell touched
        out = out.strip()
        if code != 0:
            return ToolText(f"Exit code {code}\n{out or '(no output)'}", ok=False)
        return out or "(command produced no output)"

    # --- background / long-running processes (procman) ---
    def _host_only_note(self) -> str:
        # #4: background procs + PTY sessions run on the HOST, not through self.sandbox. Under a non-local
        # sandbox (e.g. docker) that defeats container isolation — surface it instead of silently bypassing.
        return ("[warning: this runs on the HOST, NOT inside the configured sandbox — "
                f"{type(self.sandbox).__name__} isolation does not apply]\n"
                if type(self.sandbox).__name__ != "LocalSandbox" else "")

    def _t_proc_start(self, args: dict) -> str:
        h = self.procs.start(args["command"], cwd=self.root())
        return (f"{self._host_only_note()}Started background process {h}: {args['command']}\n"
                f"Use proc_tail/proc_poll/proc_wait/proc_kill with handle {h}.")

    def _t_proc_poll(self, args: dict) -> str:
        return self.procs.poll(args["handle"])

    def _t_proc_tail(self, args: dict) -> str:
        # #26: cap requested lines so a huge `lines` can't dump a chatty server's whole log into the slice.
        return self.procs.tail(args["handle"], max(1, min(int(args.get("lines") or 40), 2000)))

    def _t_proc_wait(self, args: dict) -> str:
        try:
            t = float(args.get("timeout") or 30.0)
        except (TypeError, ValueError):
            t = 30.0
        # proc_wait is a poll-with-timeout — allow sub-second waits (unlike run_command's 1s floor).
        return self.procs.wait(args["handle"], max(0.05, min(t, 600.0)))

    def _t_proc_kill(self, args: dict) -> str:
        return self.procs.kill(args["handle"])

    # --- interactive PTY sessions (terminal) ---
    def _t_terminal_open(self, args: dict) -> str:
        name = args.get("session") or "main"
        self.terminals.open(name, cwd=self.root(), command=args.get("command") or None)
        banner = self.terminals.peek(name, timeout=0.6)  # peek, not read — don't eat the first prompt
        return f"{self._host_only_note()}Opened terminal session {name!r}.\n{banner}"

    def _t_terminal_send(self, args: dict) -> str:
        name = args.get("session") or "main"
        enter = args.get("enter")
        enter = True if enter is None else bool(enter)
        return self.terminals.send(name, args["input"], enter=enter)

    def _t_terminal_read(self, args: dict) -> str:
        name = args.get("session") or "main"
        try:
            t = float(args.get("timeout") or 1.0)
        except (TypeError, ValueError):
            t = 1.0
        return self.terminals.read(name, timeout=max(0.05, min(t, 120.0)))

    def _t_terminal_wait(self, args: dict) -> str:
        name = args.get("session") or "main"
        try:
            t = float(args.get("timeout") or 10.0)
        except (TypeError, ValueError):
            t = 10.0
        return self.terminals.wait(name, args["until"], timeout=max(0.1, min(t, 600.0)))

    def _t_terminal_close(self, args: dict) -> str:
        return self.terminals.close(args.get("session") or "main")

    # --- world model (durable agent scratchpad; state lives in the Slice, folded by slice_sink) ---
    def _t_world_set(self, args: dict) -> str:
        k = (args.get("key") or "").strip()
        if not k:
            return ToolText("Error: world_set requires a non-empty 'key'.", ok=False)
        v = " ".join(str(args.get("value", "")).split())   # one-line echo so the value is readable THIS turn
        if len(v) > 200:
            v = v[:200] + "…"
        return (f"WORLD MODEL: saved {k!r} = {v} (in your WORLD MODEL section from your NEXT turn; "
                f"this turn, re-read it from this call).")

    def _t_world_clear(self, args: dict) -> str:
        k = (args.get("key") or "").strip()
        return f"WORLD MODEL: cleared {repr(k) if k else '(all keys)'}."

    # --- standing requirements (the durable contract; state lives in the Slice, folded by slice_sink) ---
    def _t_require(self, args: dict) -> str:
        t = " ".join((args.get("text") or "").split())
        if not t:
            return ToolText("Error: require needs a non-empty 'text'.", ok=False)
        return f"REQUIREMENT recorded: {t} (in your STANDING REQUIREMENTS from your next turn until done/dropped)."

    def _t_requirement_done(self, args: dict) -> str:
        t = " ".join((args.get("text") or "").split())
        if not t:
            return ToolText("Error: requirement_done needs the requirement 'text'.", ok=False)
        return f"REQUIREMENT marked done: {t} (stays shown as [x], no longer flagged outstanding)."

    def _t_drop_requirement(self, args: dict) -> str:
        t = " ".join((args.get("text") or "").split())
        if not t:
            return ToolText("Error: drop_requirement needs the requirement 'text'.", ok=False)
        return f"REQUIREMENT dropped: {t}."

    def _t_update_plan(self, args: dict) -> str:
        # The STATE lives in the slice's PLAN tier (folded by slice_sink from this event); the handler
        # only validates + confirms (the world_set/require pattern).
        steps = args.get("steps")
        if not isinstance(steps, list) or not steps:
            return ToolText("Error: update_plan requires a non-empty 'steps' list "
                            "(each {step, status: pending|in_progress|done}).", ok=False)
        n = len(steps)
        done = sum(1 for s in steps if isinstance(s, dict) and s.get("status") == "done")
        doing = sum(1 for s in steps if isinstance(s, dict) and s.get("status") == "in_progress")
        return f"PLAN updated: {n} steps ({done} done, {doing} in progress) — shown in your PLAN section."

    def _t_set_mission(self, args: dict) -> str:
        t = " ".join((args.get("text") or "").split())
        if not t:
            return ToolText("Error: set_mission needs a non-empty 'text'.", ok=False)
        return f"MISSION set: {t} (shown at the top of your context until you call mission_done)."

    def _t_mission_done(self, args: dict) -> str:
        return "MISSION cleared (achieved — no longer shown)."

    def _t_execute_code(self, args: dict) -> str:
        out = self._execute_code(args["code"])
        self._grant_shell_paths(args.get("code", ""))  # I2 reach=action: dirs code-as-action touched
        return out

    def _execute_code(self, code: str) -> str:
        """Code-as-action: run the model's script (prelude + code) in the sandbox, cwd=workspace.
        Only stdout returns. The script is written INSIDE the workspace as a hidden temp file
        (so it's mounted/available in every backend) and deleted right after; cwd is on sys.path
        so workspace imports resolve. `sandbox.python_cmd` keeps it backend-portable."""
        script = _CODE_PRELUDE + "\n# --- agent code ---\n" + code
        root = self.root()
        fd, path = tempfile.mkstemp(suffix=".py", prefix=".memagent-exec-", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(script)
            cmd = f"{shlex.quote(self.sandbox.python_cmd)} {shlex.quote(os.path.basename(path))}"
            code_n, out = self.sandbox.run(cmd, cwd=root, timeout=self.timeout)
            out = out.strip()
            if code_n != 0:
                return ToolText(f"Exit code {code_n}\n{out or '(no output)'}", ok=False)
            return out or "(execute_code produced no output)"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def _mkparent(path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _atomic_write(full: str, content: str) -> None:
        """Write `content` to `full` atomically: write a temp file in the SAME directory,
        then os.replace() it over the target. A crash/error mid-write leaves the original
        intact (the rename is atomic on POSIX); the temp is unlinked on any failure. The
        temp must share the target's filesystem for os.replace to be atomic, hence
        dir=os.path.dirname(full) (full is already _resolve()'d)."""
        d = os.path.dirname(full)
        fd, tmp = tempfile.mkstemp(prefix=".memagent-tmp-", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, full)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
