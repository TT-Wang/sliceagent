"""Workspace grep tool — pagination, repeatable live observation, and path confinement.
No model, no pytest. Run: python tests/test_code_grep.py

Covers sec 5 test_code_grep.py: line-numbered matches; pagination (offset/limit +
truncation hint); repeated identical observations remain available; missing/broken rg is
reported as a real failure; path escaping root is rejected by host._resolve.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent import code_grep                               # noqa: E402
from sliceagent.code_grep import make_glob_tool, make_grep_tool  # noqa: E402
from sliceagent.execution import ToolStatus                   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Host:
    """Minimal LocalToolHost-shaped host: root() + _resolve() with escape rejection."""
    def __init__(self, root):
        self._root = os.path.realpath(root)
    def root(self):
        return self._root
    def _resolve(self, path):
        if not path:
            raise ValueError("empty path")
        full = path if os.path.isabs(path) else os.path.join(self._root, path)
        full = os.path.realpath(full)
        if full != self._root and not full.startswith(self._root + os.sep):
            raise PermissionError(f"path escapes the boundary ({self._root}): {path}")
        return full


def _workspace(files: dict):
    d = tempfile.mkdtemp(prefix="sliceagent-grep-")
    for name, content in files.items():
        p = os.path.join(d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(name) else None
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
    return d


def _fresh_tool(files):
    host = _Host(_workspace(files))
    return make_grep_tool(host).handler, host


_HAS_RG = shutil.which("rg") is not None


@check
def entry_shape_is_grep():
    host = _Host(tempfile.mkdtemp())
    entry = make_grep_tool(host)
    assert entry.name == "grep"
    assert entry.schema["function"]["name"] == "grep"
    assert entry.source == "builtin"
    # accesses declares a recursive search on the path arg
    acc = entry.accesses({"path": "sub"})
    assert acc[0].operation == "search" and acc[0].path == "sub" and acc[0].recursive


@check
def line_numbered_matches():
    if not _HAS_RG:
        return  # degradation covered by no_rg_degrades_quietly
    handler, _ = _fresh_tool({"a.py": "alpha\nbeta\nalpha again\n"})
    out = handler({"pattern": "alpha"})
    # ripgrep -n yields file:line:text
    assert "1:alpha" in out
    assert "3:alpha again" in out
    assert "BLOCKED" not in out


@check
def pagination_offset_limit_and_truncation_hint():
    if not _HAS_RG:
        return
    body = "".join(f"hit line {i}\n" for i in range(10))
    handler, _ = _fresh_tool({"big.txt": body})
    out = handler({"pattern": "hit", "limit": 3, "offset": 0})
    assert out.count("\n") >= 2  # 3 lines + hint
    assert "[truncated; use offset=3 to see more]" in out
    # next page: no truncation hint (fewer than limit remain), still real matches
    out2 = handler({"pattern": "hit", "limit": 3, "offset": 9})
    assert "hit" in out2 and "[truncated" not in out2


@check
def repeated_identical_searches_remain_live_observations():
    if not _HAS_RG:
        return
    handler, _ = _fresh_tool({"a.py": "needle\n"})
    results = [handler({"pattern": "needle"}) for _ in range(6)]
    assert all("needle" in result for result in results)
    assert len(set(results)) == 1


@check
def no_rg_is_a_real_failure(monkeypatch_which=None):
    # force the no-rg path regardless of environment
    orig = code_grep.shutil.which
    code_grep.shutil.which = lambda name: None
    try:
        handler, _ = _fresh_tool({"a.py": "x\n"})
        out = handler({"pattern": "x"})
        assert "ripgrep" in out and "not run" in out
        assert out.status is ToolStatus.FAILED
    finally:
        code_grep.shutil.which = orig


@check
def rg_launch_failure_is_a_real_failure():
    orig_which = code_grep.shutil.which
    orig_run = code_grep.subprocess.run
    code_grep.shutil.which = lambda name: "/usr/bin/rg"
    code_grep.subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(OSError("cannot execute"))
    try:
        handler, _ = _fresh_tool({"a.py": "x\n"})
        out = handler({"pattern": "x"})
        assert "search failed" in out and out.status is ToolStatus.FAILED
    finally:
        code_grep.shutil.which = orig_which
        code_grep.subprocess.run = orig_run


@check
def rg_exit_two_is_a_real_failure_not_no_matches():
    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "regex parse error"

    orig_which = code_grep.shutil.which
    orig_run = code_grep.subprocess.run
    code_grep.shutil.which = lambda name: "/usr/bin/rg"
    code_grep.subprocess.run = lambda *a, **kw: _Proc()
    try:
        handler, _ = _fresh_tool({"a.py": "x\n"})
        out = handler({"pattern": "["})
        assert "exit code 2" in out and "regex parse error" in out
        assert "no matches" not in out and out.status is ToolStatus.FAILED
    finally:
        code_grep.shutil.which = orig_which
        code_grep.subprocess.run = orig_run


@check
def glob_rg_launch_and_exit_errors_are_real_failures():
    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "bad glob"

    host = _Host(_workspace({"a.py": "x\n"}))
    handler = make_glob_tool(host).handler
    orig_which = code_grep.shutil.which
    orig_run = code_grep.subprocess.run
    code_grep.shutil.which = lambda name: "/usr/bin/rg"
    try:
        code_grep.subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(OSError("cannot execute"))
        launch = handler({"pattern": "*.py"})
        assert launch.status is ToolStatus.FAILED and "search failed" in launch

        code_grep.subprocess.run = lambda *a, **kw: _Proc()
        exit_two = handler({"pattern": "["})
        assert exit_two.status is ToolStatus.FAILED and "exit code 2" in exit_two
        assert "bad glob" in exit_two
    finally:
        code_grep.shutil.which = orig_which
        code_grep.subprocess.run = orig_run


@check
def path_escaping_root_blocked():
    handler, _ = _fresh_tool({"a.py": "x\n"})
    out = handler({"pattern": "x", "path": "../../etc"})
    assert out.startswith("Error:") and "escapes the boundary" in out


@check
def empty_pattern_is_quiet():
    handler, _ = _fresh_tool({"a.py": "x\n"})
    out = handler({"pattern": ""})
    assert "no pattern" in out and "Error" not in out and "BLOCKED" not in out


@check
def no_matches_is_quiet_non_failing():
    if not _HAS_RG:
        return
    handler, _ = _fresh_tool({"a.py": "alpha\n"})
    out = handler({"pattern": "zzzznotpresent"})
    assert "no matches" in out and "Error" not in out and "BLOCKED" not in out


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
