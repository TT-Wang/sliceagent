"""LocalToolHost — the default ToolHost (no sandbox yet).

P1 will wrap execution in a container; this keeps the same interface so the loop
never changes. Note: Python's str.replace is literal, so str_replace has no
$-pattern footgun (unlike JS).
"""
from __future__ import annotations

import os
import subprocess

from .access import AllAccess, FileAccess


def _fn(name: str, desc: str, props: dict, req: list[str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": desc,
                     "parameters": {"type": "object", "properties": props, "required": req}},
    }


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
]


class LocalToolHost:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def accesses(self, name: str, args: dict) -> list:
        """Declare what each call touches so the scheduler can safely parallelize."""
        p = args.get("path")
        if name == "read_file":
            return [FileAccess("read", p)] if p else []
        if name == "list_files":
            return [FileAccess("search", args.get("path") or ".", recursive=True)]
        if name in ("edit_file", "append_to_file", "str_replace"):
            return [FileAccess("readwrite", p)] if p else [AllAccess()]
        if name == "run_command":
            return [AllAccess()]  # shell can do anything → globally exclusive
        return [AllAccess()]

    def read_text(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    def run(self, name: str, args: dict) -> str:
        try:
            if name == "read_file":
                return self.read_text(args["path"])
            if name == "list_files":
                return "\n".join(sorted(os.listdir(args.get("path") or "."))) or "(empty)"
            if name == "edit_file":
                self._mkparent(args["path"])
                with open(args["path"], "w", encoding="utf-8") as f:
                    f.write(args["content"])
                return f"Wrote {len(args['content'])} bytes to {args['path']}"
            if name == "append_to_file":
                self._mkparent(args["path"])
                with open(args["path"], "a", encoding="utf-8") as f:
                    f.write(args["content"])
                return f"Appended {len(args['content'])} bytes to {args['path']}"
            if name == "str_replace":
                cur = self.read_text(args["path"])
                old = args["old_string"]
                n = cur.count(old)
                if n == 0:
                    return f"Error: old_string not found in {args['path']}"
                if n > 1:
                    return f"Error: old_string occurs {n} times in {args['path']}; add context to make it unique"
                updated = cur.replace(old, args["new_string"], 1)
                with open(args["path"], "w", encoding="utf-8") as f:
                    f.write(updated)
                return f"Replaced 1 occurrence in {args['path']} ({len(cur)} → {len(updated)} bytes)"
            if name == "run_command":
                r = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=self.timeout)
                out = (r.stdout or "") + (r.stderr or "")
                if r.returncode != 0:
                    return f"Exit code {r.returncode}\n{out.strip() or '(no output)'}"
                return out.strip() or "(command produced no output)"
            return f'Error: unknown tool "{name}"'
        except Exception as e:  # errors come back as strings so the model can react
            return f"Error: {e}"

    @staticmethod
    def _mkparent(path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
