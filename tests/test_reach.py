"""I2 — RE-OBSERVATION REACH = ACTION REACH.

File-tool reach (allowed_roots, expanduser, prescriptive escape error) must match where the
agent acts; OPEN FILES must tell the TRUTH per exception type instead of the "(not created yet)"
lie; a failed read must not be pinned into the working set; and the slice must OBSERVE the
environment (platform/HOME/cwd/git) rather than remember it.

No model, no pytest. Run: PYTHONPATH=src python3 tests/test_reach.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import ToolResult                         # noqa: E402
from memagent.slice import (                                   # noqa: E402
    Slice, build_artifacts, make_build_slice, slice_sink, touch_file,
)
from memagent.tools import LocalToolHost                       # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _host():
    d = tempfile.mkdtemp(prefix="memagent-reach-test-")
    return LocalToolHost(root=os.path.realpath(d)), os.path.realpath(d)


def _write(d, rel, content, *, mode="w"):
    p = os.path.join(d, rel)
    os.makedirs(os.path.dirname(p) or d, exist_ok=True)
    with open(p, mode, encoding="utf-8") as f:
        f.write(content)
    return p


# --- P2: ~ expansion is the FIRST step in _resolve -----------------------------

@check
def tilde_expands_to_home_first():
    h, _ = _host()
    home = os.path.realpath(os.path.expanduser("~"))
    # '~/x' must resolve UNDER $HOME, never to a literal '~' dir inside the workspace.
    # It escapes the workspace (correct) -> PermissionError, not a silent literal-'~' write.
    raised = False
    try:
        h._resolve("~/some_unlikely_dir_xyz/file.txt")
    except PermissionError as e:
        raised = True
        # the message proves it expanded ~ to the real home before checking reach
        assert home in str(e), f"~ not expanded before escape check: {e!r}"
    assert raised, "~ path outside workspace must raise (after expansion), not silently write a literal ~"


@check
def tilde_resolves_when_home_is_allowed():
    # add_root($HOME/<sub>) brings a ~-path into reach: expanduser + allowed_roots cooperate.
    h, _ = _host()
    sub = tempfile.mkdtemp(prefix="memagent-reach-home-", dir=os.path.expanduser("~"))
    try:
        h.add_root(sub)
        rel = os.path.join("~", os.path.relpath(sub, os.path.expanduser("~")), "f.txt")
        full = h._resolve(rel)
        assert full == os.path.realpath(os.path.join(sub, "f.txt"))
    finally:
        os.rmdir(sub)


# --- P3: prescriptive escape error naming the shell escape hatch ---------------

@check
def escape_error_is_prescriptive():
    h, _ = _host()
    raised = False
    try:
        h._resolve("/etc/hosts")
    except PermissionError as e:
        raised = True
        msg = str(e)
        assert "run_command" in msg or "execute_code" in msg, f"escape error names no shell hatch: {msg!r}"
        assert "escapes workspace" in msg
    assert raised


# --- allowed_roots: file tools resolve a targeted EXTERNAL dir ------------------

@check
def add_root_makes_external_dir_reachable():
    h, root = _host()
    ext = tempfile.mkdtemp(prefix="memagent-reach-ext-")
    try:
        # before: a path under ext escapes the workspace
        before = False
        try:
            h._resolve(os.path.join(ext, "out.txt"))
        except PermissionError:
            before = True
        assert before, "external path should escape before add_root"
        # after add_root: it resolves, and so does the workspace path (reach is the UNION)
        added = h.add_root(ext)
        assert added == os.path.realpath(ext)
        assert h._resolve(os.path.join(ext, "out.txt")) == os.path.realpath(os.path.join(ext, "out.txt"))
        assert h._resolve("inside.txt") == os.path.join(root, "inside.txt")
        assert os.path.realpath(ext) in h.allowed_roots() and root in h.allowed_roots()
    finally:
        os.rmdir(ext)


@check
def shell_written_file_is_readable_after_add_root():
    # the core I2 promise: a file the SHELL writes outside the workspace is readable back by file tools
    h, _ = _host()
    ext = tempfile.mkdtemp(prefix="memagent-reach-shellext-")
    try:
        _write(ext, "made_by_shell.txt", "hello from shell")
        h.add_root(ext)
        assert h.read_text(os.path.join(ext, "made_by_shell.txt")) == "hello from shell"
    finally:
        import shutil
        shutil.rmtree(ext, ignore_errors=True)


@check
def add_root_refuses_blanket_roots():
    # safety: never widen reach to the whole filesystem or the bare home dir (no blanket '/').
    h, _ = _host()
    assert h.add_root("/") is None
    assert h.add_root("~") is None
    assert h.add_root(os.path.expanduser("~")) is None
    # those must not have entered allowed_roots
    roots = h.allowed_roots()
    assert os.sep not in [r for r in roots if r == os.sep]
    assert os.path.realpath(os.path.expanduser("~")) not in roots


# --- OF1: OPEN FILES branches on exception TYPE (truthful per case) -------------

@check
def open_files_not_created_for_missing():
    h, _ = _host()
    s = Slice(); s.reset("t")
    s.active_files = ["ghost.txt"]                       # never written
    out = build_artifacts(s, h)
    assert "(not created yet)" in out
    assert "outside file-tool reach" not in out


@check
def open_files_outside_reach_for_escape():
    # a file that EXISTS on disk but is outside file-tool reach must NOT read "(not created yet)"
    h, _ = _host()
    ext = tempfile.mkdtemp(prefix="memagent-reach-of-")
    try:
        ext_file = _write(ext, "real.txt", "exists on disk")  # exists, but outside allowed_roots
        s = Slice(); s.reset("t")
        s.active_files = [ext_file]
        out = build_artifacts(s, h)
        assert "(not created yet)" not in out, "lied about an existing out-of-reach file"
        assert "outside file-tool reach" in out
        assert "run_command" in out or "execute_code" in out
    finally:
        import shutil
        shutil.rmtree(ext, ignore_errors=True)


@check
def open_files_binary_exists_but_not_shown():
    h, root = _host()
    _write(root, "blob.png", "not really a png but binary by ext")
    s = Slice(); s.reset("t")
    s.active_files = ["blob.png"]
    out = build_artifacts(s, h)
    assert "(not created yet)" not in out
    assert "outside file-tool reach" not in out
    assert "exists but not shown" in out and "binary" in out


@check
def open_files_shows_real_file_in_full():
    h, root = _host()
    _write(root, "a.py", "print('hi')\n")
    s = Slice(); s.reset("t")
    s.active_files = ["a.py"]
    out = build_artifacts(s, h)
    assert "print('hi')" in out and "(not created yet)" not in out


# --- WS1: a failed read is NOT pinned into the working set ----------------------

@check
def failed_read_not_pinned():
    h, _ = _host()
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    # a read_file that FAILED in _resolve (path escapes workspace) -> failing=True
    sink(ToolResult("read_file", {"path": "/etc/hosts"},
                    "Error: path escapes workspace ...", True))
    assert s.active_files == [], f"failed read pinned the path: {s.active_files}"


@check
def successful_read_still_pinned():
    h, root = _host()
    _write(root, "ok.txt", "content")
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    sink(ToolResult("read_file", {"path": "ok.txt"}, "content", False))
    assert s.active_files == ["ok.txt"]


@check
def failed_edit_not_pinned_but_success_is():
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    sink(ToolResult("edit_file", {"path": "out/x.py", "content": "..."},
                    "Error: path escapes workspace ...", True))
    assert s.active_files == [], "failed edit pinned a path"
    sink(ToolResult("edit_file", {"path": "in.py", "content": "..."}, "Wrote 3 bytes to in.py", False))
    assert s.active_files == ["in.py"] and "in.py" in s.edited_files


# --- ENVIRONMENT tier: re-observed ground truth in the system message -----------

class _NullRetriever:
    def retrieve(self, query, k=6):
        return []


@check
def environment_tier_present_and_observed():
    h, root = _host()
    s = Slice(); s.reset("do a thing")
    build = make_build_slice(s, h, _NullRetriever(), None, "do a thing")
    msgs = build()
    assert len(msgs) == 2 and msgs[0]["role"] == "system"
    sysmsg = msgs[0]["content"]
    assert "# ENVIRONMENT" in sysmsg and "OBSERVED ground truth" in sysmsg
    assert f"- Platform: {sys.platform}" in sysmsg
    assert os.path.expanduser("~") in sysmsg          # real HOME, not a generic /home/user
    assert root in sysmsg                              # real cwd


@check
def environment_tier_is_cache_stable():
    # the system message (level 0) is byte-identical across turns: per-session compute, not per-turn
    h, _ = _host()
    s = Slice(); s.reset("g")
    build = make_build_slice(s, h, _NullRetriever(), None, "g")
    sys1 = build()[0]["content"]
    # mutate volatile state the way a turn would; the SYSTEM tier must not move
    touch_file(s, "x.py")
    s.last_error = "boom"
    s.findings.append("a fact")
    sys2 = build()[0]["content"]
    assert sys1 == sys2, "ENVIRONMENT/system tier drifted across turns (cache-busting)"


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
