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
import shlex
import tempfile

from .access import AllAccess, FileAccess
from .registry import ToolEntry, ToolRegistry
from .sandbox import LocalSandbox

# Prepended to every execute_code script: the in-sandbox tool helpers (code-as-action,
# Hermes pattern). No imports needed by the model. The workspace is cwd and on sys.path,
# so `import <workspace_module>` works for testing freshly-written code.
_CODE_PRELUDE = '''\
import os as _os, sys as _sys, subprocess as _sp
_sys.path.insert(0, _os.getcwd())

def read_file(path):
    with open(path, encoding="utf-8") as _f: return _f.read()

def write_file(path, content):
    _d = _os.path.dirname(path)
    if _d: _os.makedirs(_d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as _f: _f.write(content)
    return f"wrote {len(content)} bytes to {path}"

def append_file(path, content):
    _d = _os.path.dirname(path)
    if _d: _os.makedirs(_d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as _f: _f.write(content)
    return f"appended {len(content)} bytes to {path}"

def str_replace(path, old, new):
    with open(path, encoding="utf-8") as _f: _cur = _f.read()
    _n = _cur.count(old)
    if _n != 1: return (f"error: old_string occurs {_n}x in {path} (need exactly 1) — "
                        f"add surrounding lines to make it unique, or write_file the whole file")
    with open(path, "w", encoding="utf-8") as _f: _f.write(_cur.replace(old, new, 1))
    return f"replaced 1 occurrence in {path}"

def list_files(path="."):
    return sorted(_os.listdir(path))

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
        "description": ("Optional. A durable FACT you established this turn — root cause, a confirmed fix, a "
                        "ruled-out hypothesis, or 'task done' — in <=15 words. A conclusion, NOT the action "
                        "you're taking. Saved across turns so you never re-derive it. Empty if nothing new."),
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


TOOL_SCHEMAS = [
    _fn("read_file", "Read a file and return its contents.", {"path": {"type": "string"}}, ["path"]),
    _fn("list_files", "List files in a directory (defaults to current directory).", {"path": {"type": "string"}}, []),
    _fn("edit_file", "Create or OVERWRITE an entire file (replaces ALL content). Use only for brand-new files.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("append_to_file", "Append content to the end of a file (creates it if missing). Use to ADD without overwriting.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("str_replace", "Replace a unique snippet in an existing file. old_string must match exactly and occur once.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"]),
    _fn("run_command", "Run a shell command; returns combined output (and exit code on failure).",
        {"command": {"type": "string"}}, ["command"]),
    _fn("execute_code",
        "Run a Python script that does MULTIPLE file/shell actions in ONE turn, then prints a "
        "short result. Helpers (no imports needed): read_file(path), write_file(path, content), "
        "append_file(path, content), str_replace(path, old, new), list_files(path='.'), "
        "run(shell_cmd). The workspace is the cwd and on sys.path. ONLY what you print() is "
        "returned. Prefer this to fire several edits AND a test in a single turn.",
        {"code": {"type": "string"}}, ["code"]),
]


class LocalToolHost:
    def __init__(self, root: str | None = None, *, sandbox=None, timeout: int = 30,
                 registry: ToolRegistry | None = None):
        # root=None → confine to the *current* working directory, resolved per call
        # (so the eval runner, which chdirs into a temp workdir after construction,
        # is confined to that workdir). Pass an explicit root to pin it.
        self._root = root
        self.timeout = timeout
        self.sandbox = sandbox or LocalSandbox()
        # The registry is the single source of tools; MCP/plugin/skill tools register
        # into this same object later (Step ③). The host just projects from it.
        self.registry = registry or ToolRegistry()
        self._register_builtins()

    def _register_builtins(self) -> None:
        handlers = {
            "read_file": self._t_read_file, "list_files": self._t_list_files,
            "edit_file": self._t_edit_file, "append_to_file": self._t_append,
            "str_replace": self._t_str_replace, "run_command": self._t_run_command,
            "execute_code": self._t_execute_code,
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

    def _resolve(self, path: str) -> str:
        """Resolve a tool path under the workspace root; reject escapes."""
        if not path:
            raise ValueError("empty path")
        root = self.root()
        full = path if os.path.isabs(path) else os.path.join(root, path)
        full = os.path.realpath(full)
        if full != root and not full.startswith(root + os.sep):
            raise PermissionError(f"path escapes workspace ({root}): {path}")
        return full

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
        with open(self._resolve(path), encoding="utf-8") as f:
            return f.read()

    def _builtin_accesses(self, name: str, args: dict) -> list:
        """Declare what each builtin call touches so the scheduler can safely parallelize."""
        p = args.get("path")
        if name == "read_file":
            return [FileAccess("read", p)] if p else []
        if name == "list_files":
            return [FileAccess("search", args.get("path") or ".", recursive=True)]
        if name in ("edit_file", "append_to_file", "str_replace"):
            return [FileAccess("readwrite", p)] if p else [AllAccess()]
        if name in ("run_command", "execute_code"):
            return [AllAccess()]  # arbitrary execution → globally exclusive
        return [AllAccess()]

    # --- builtin tool handlers (args) -> str (the registry catches exceptions) ---
    def _t_read_file(self, args: dict) -> str:
        return self.read_text(args["path"])

    def _t_list_files(self, args: dict) -> str:
        d = self._resolve(args.get("path") or ".")
        return "\n".join(sorted(os.listdir(d))) or "(empty)"

    def _t_edit_file(self, args: dict) -> str:
        full = self._resolve(args["path"])
        self._mkparent(full)
        with open(full, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return f"Wrote {len(args['content'])} bytes to {args['path']}"

    def _t_append(self, args: dict) -> str:
        full = self._resolve(args["path"])
        self._mkparent(full)
        with open(full, "a", encoding="utf-8") as f:
            f.write(args["content"])
        return f"Appended {len(args['content'])} bytes to {args['path']}"

    def _t_str_replace(self, args: dict) -> str:
        full = self._resolve(args["path"])
        cur = self.read_text(args["path"])
        old = args["old_string"]
        n = cur.count(old)
        if n == 0:
            return (f"Error: old_string not found in {args['path']} — your snippet does not match "
                    f"the file. Copy the EXACT text from OPEN FILES (the live content), or rewrite "
                    f"the whole file with edit_file. Do NOT retry the same str_replace.")
        if n > 1:
            return f"Error: old_string occurs {n} times in {args['path']}; add context to make it unique"
        updated = cur.replace(old, args["new_string"], 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(updated)
        return f"Replaced 1 occurrence in {args['path']} ({len(cur)} → {len(updated)} bytes)"

    def _t_run_command(self, args: dict) -> str:
        code, out = self.sandbox.run(args["command"], cwd=self.root(), timeout=self.timeout)
        out = out.strip()
        if code != 0:
            return f"Exit code {code}\n{out or '(no output)'}"
        return out or "(command produced no output)"

    def _t_execute_code(self, args: dict) -> str:
        return self._execute_code(args["code"])

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
                return f"Exit code {code_n}\n{out or '(no output)'}"
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
