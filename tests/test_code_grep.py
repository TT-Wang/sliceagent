"""Guarded grep tool — pagination, consecutive-search block, path confinement.
No model, no pytest. Run: python tests/test_code_grep.py

Covers sec 5 test_code_grep.py: line-numbered matches; pagination (offset/limit +
truncation hint); 4th identical-in-a-row BLOCKED; different pattern resets the counter;
paging (increasing offset) does NOT trip the guard; no-rg degrades quietly; path escaping
root blocked via host._resolve.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent import code_grep                               # noqa: E402
from memagent.code_grep import GREP_GUARD, make_grep_tool    # noqa: E402

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
            raise PermissionError(f"path escapes workspace ({self._root}): {path}")
        return full


def _workspace(files: dict):
    d = tempfile.mkdtemp(prefix="memagent-grep-")
    for name, content in files.items():
        p = os.path.join(d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(name) else None
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
    return d


def _fresh_tool(files):
    GREP_GUARD.clear()
    host = _Host(_workspace(files))
    return make_grep_tool(host).handler, host


_HAS_RG = shutil.which("rg") is not None


@check
def entry_shape_is_grep():
    GREP_GUARD.clear()
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
def fourth_identical_in_a_row_blocked():
    if not _HAS_RG:
        return
    handler, _ = _fresh_tool({"a.py": "needle\n"})
    a = handler({"pattern": "needle"})
    b = handler({"pattern": "needle"})
    c = handler({"pattern": "needle"})
    d = handler({"pattern": "needle"})
    assert "BLOCKED" not in a and "BLOCKED" not in b and "BLOCKED" not in c
    assert "BLOCKED" in d


@check
def different_pattern_resets_counter():
    if not _HAS_RG:
        return
    handler, _ = _fresh_tool({"a.py": "needle\nhaystack\n"})
    handler({"pattern": "needle"})
    handler({"pattern": "needle"})
    handler({"pattern": "needle"})  # streak of 3
    other = handler({"pattern": "haystack"})  # resets
    assert "BLOCKED" not in other
    again = handler({"pattern": "needle"})  # streak restarts at 1
    assert "BLOCKED" not in again


@check
def paging_does_not_trip_guard():
    if not _HAS_RG:
        return
    body = "".join(f"row {i}\n" for i in range(20))
    handler, _ = _fresh_tool({"big.txt": body})
    # same pattern/limit but increasing offset each time — key differs, never blocks
    for off in (0, 2, 4, 6, 8):
        out = handler({"pattern": "row", "limit": 2, "offset": off})
        assert "BLOCKED" not in out, f"blocked at offset={off}: {out!r}"


@check
def no_rg_degrades_quietly(monkeypatch_which=None):
    # force the no-rg path regardless of environment
    orig = code_grep.shutil.which
    code_grep.shutil.which = lambda name: None
    try:
        handler, _ = _fresh_tool({"a.py": "x\n"})
        out = handler({"pattern": "x"})
        assert "ripgrep" in out and "Error" not in out and "BLOCKED" not in out
    finally:
        code_grep.shutil.which = orig


@check
def path_escaping_root_blocked():
    handler, _ = _fresh_tool({"a.py": "x\n"})
    out = handler({"pattern": "x", "path": "../../etc"})
    assert out.startswith("Error:") and "escapes workspace" in out


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
