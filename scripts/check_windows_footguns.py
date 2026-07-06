#!/usr/bin/env python3
"""check_windows_footguns — grep-lint that kills the Windows-breakage bug CLASS (ported from
Hermes' scripts/check-windows-footguns.py, trimmed to sliceagent's actual surfaces).

Rules over src/sliceagent/*.py (suppress a line with `# windows-footgun: ok`):
  1. text-mode builtin open() without encoding=  (Windows defaults to cp1252 — mojibake)
  2. pathlib read_text()/write_text() without encoding=
  3. os.kill(pid, 0) existence-probe  (on Windows it CTRL_C's or kills the target — bpo-14484)
  4. bare os.setsid / os.killpg / os.getpgid / start_new_session outside the seam
  5. signal.SIGKILL by attribute outside the seam  (doesn't exist on win32)
  6. subprocess shell=True outside the seam  (agent commands must go through platform_compat.sh())
  7. unguarded top-level `import fcntl|pty|termios` outside the allowed guarded files

Exemptions: platform_compat.py IS the seam (all branches live there). terminal.py is POSIX-only BY
CONSTRUCTION (SessionManager.open() refuses on Windows before any of its killpg/PTY code can run) —
exempt from 4/5/6 until the Phase-2 pywinpty bridge. Docstrings are excluded via ast.

Exit 0 = clean, 1 = violations (printed one per line).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "sliceagent"
SEAM = "platform_compat.py"
POSIX_ONLY = {"terminal.py", SEAM}            # gated at entry; Phase 2 adds the win bridge
GUARDED_UNIX_IMPORTS = {"terminal.py", SEAM}  # fcntl/pty imports sit in try/except there

_BINARY_MODE = re.compile(r"['\"][rwax+]*b[rwax+]*['\"]")
_BUILTIN_OPEN = re.compile(r"(?<![\w.])open\(")           # builtin only — not .open() methods
_RT_WT = re.compile(r"\.(read_text|write_text)\(")

RULES: list[tuple[str, re.Pattern, set[str]]] = [
    ("os.kill(pid, 0) existence probe (bpo-14484: kills the target on Windows)",
     re.compile(r"os\.kill\([^,\n]+,\s*0\s*\)"), set()),
    ("bare setsid/killpg/getpgid/start_new_session (win32: no-op or AttributeError)",
     re.compile(r"os\.(setsid|killpg|getpgid)\b|start_new_session\s*="), POSIX_ONLY),
    ("signal.SIGKILL attribute (missing on win32 — use platform_compat.SIG_KILL)",
     re.compile(r"signal\.SIGKILL\b"), POSIX_ONLY),
    ("shell=True (agent commands must route through platform_compat.sh())",
     re.compile(r"shell\s*=\s*True"), POSIX_ONLY),
    ("unguarded top-level import of fcntl/pty/termios",
     re.compile(r"^import (fcntl|pty|termios)\b|^from (fcntl|pty|termios) import"), GUARDED_UNIX_IMPORTS),
]

# Individually-reviewed sites the line rules can't see into:
ALLOW = {
    # tools.py holds the sandbox-toolkit TEMPLATE STRING for code-as-action workers; its inline
    # `shell=True` executes under the worker sandbox on the same host — tracked for Phase 2.
    ("tools.py", "shell=True (agent commands must route through platform_compat.sh())"),
}


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Line numbers covered by docstrings (module/class/function) — excluded from linting."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                lines.update(range(body[0].value.lineno, (body[0].value.end_lineno or body[0].value.lineno) + 1))
    return lines


def _check_encoding_rules(fname: str, i: int, line: str, bad: list[str]) -> None:
    stripped = line.strip()
    if _BUILTIN_OPEN.search(line) and not stripped.startswith("def "):
        if "encoding=" not in line and not _BINARY_MODE.search(line) and "open()" not in line:
            bad.append(f"{fname}:{i}: [open() without encoding= (cp1252 on Windows)] {stripped[:100]}")
    if _RT_WT.search(line) and "encoding=" not in line:
        # sliceagent's own tool-host read_text() methods are not pathlib — skip those receivers
        if not re.search(r"(self|tools|inner)\.(read_text|write_text)\(", line):
            bad.append(f"{fname}:{i}: [read_text/write_text without encoding=] {stripped[:100]}")


def main() -> int:
    bad: list[str] = []
    for f in sorted(SRC.glob("*.py")):
        text = f.read_text(encoding="utf-8")
        try:
            doc_lines = _docstring_lines(ast.parse(text))
        except SyntaxError:
            doc_lines = set()
        for i, line in enumerate(text.splitlines(), 1):
            if i in doc_lines or "windows-footgun: ok" in line:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            _check_encoding_rules(f.name, i, line, bad)
            for label, rx, exempt in RULES:
                if f.name in exempt or (f.name, label) in ALLOW:
                    continue
                if rx.search(line):
                    bad.append(f"{f.name}:{i}: [{label}] {stripped[:100]}")
    if bad:
        print(f"{len(bad)} Windows footgun(s):")
        for b in bad:
            print(" ", b)
        return 1
    print("windows-footguns: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
