"""World-state container (the "cache") — Phase 1 (live worktree state) + Phase 2 (resident repo map).
Deterministic, no model. Builds a real temp git repo and runs the actual slice reconstruction.
Run: PYTHONPATH=src python tests/test_world_state.py
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.tools import LocalToolHost, repo_map                     # noqa: E402
from memagent.workspace import git_worktree_state, workspace_facts     # noqa: E402
from memagent.slice import Slice, make_build_slice                     # noqa: E402
from memagent.memory import NullMemory                                 # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _git(wd, *args):
    subprocess.run(["git", "-C", wd, *args], capture_output=True, text=True)


def _mk_repo() -> str:
    wd = tempfile.mkdtemp(prefix="ws-")
    os.makedirs(os.path.join(wd, "pkg"))
    os.makedirs(os.path.join(wd, ".venv", "lib"))          # junk that MUST be pruned
    os.makedirs(os.path.join(wd, "__pycache__"))
    open(os.path.join(wd, "pyproject.toml"), "w").write("[project]\nname='x'\n")
    open(os.path.join(wd, "pkg", "core.py"), "w").write("def a():\n    return 1\n")
    open(os.path.join(wd, "pkg", "util.py"), "w").write("def b():\n    return 2\n")
    open(os.path.join(wd, ".venv", "lib", "junk.py"), "w").write("# noise\n")
    open(os.path.join(wd, "__pycache__", "x.pyc"), "w").write("noise")
    _git(wd, "init"); _git(wd, "add", "-A"); _git(wd, "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-m", "init")
    return wd


# ── Phase 1: live worktree state ──────────────────────────────────────────────
@check
def worktree_clean_then_dirty():
    wd = _mk_repo()
    ws = git_worktree_state(wd)
    assert "working tree clean" in ws, ws
    open(os.path.join(wd, "pkg", "core.py"), "a").write("# edit\n")     # make it dirty
    open(os.path.join(wd, "new.py"), "w").write("x=1\n")               # untracked
    ws2 = git_worktree_state(wd)
    assert "changed file" in ws2 and "modified: pkg/core.py" in ws2 and "untracked: new.py" in ws2, ws2


@check
def worktree_bounded():
    wd = _mk_repo()
    for i in range(40):
        open(os.path.join(wd, f"f{i}.py"), "w").write("x\n")
    ws = git_worktree_state(wd, max_files=20)
    assert "more" in ws and ws.count("\n") <= 23, f"not bounded: {ws.count(chr(10))} lines"


@check
def worktree_empty_outside_repo():
    wd = tempfile.mkdtemp(prefix="norepo-")
    assert git_worktree_state(wd) == ""


# ── Phase 2: resident repo map ────────────────────────────────────────────────
@check
def repo_map_ignore_aware_and_source_ranked():
    wd = _mk_repo()
    m = repo_map(wd)
    assert ".venv" not in m and "__pycache__" not in m and ".pyc" not in m and ".git/" not in m, m
    assert "pkg/" in m and "core.py" in m and "util.py" in m, m
    # source dir ranked at/near the top (code-density)
    assert m.splitlines()[0].startswith("pkg/") or m.splitlines()[0].startswith("./"), m.splitlines()[0]


@check
def repo_map_empty_for_bad_root():
    assert repo_map("") == "" and repo_map("/nonexistent/xyz") == ""


# ── end-to-end: the cache regions appear in the reconstructed slice ────────────
def _build(wd, goal="review the repo"):
    state = Slice(); state.reset(goal)
    tools = LocalToolHost(wd)
    build = make_build_slice(state, tools, None, NullMemory(), goal)
    return state, build


@check
def slice_contains_repo_map_and_live_worktree():
    wd = _mk_repo()
    open(os.path.join(wd, "pkg", "core.py"), "a").write("# edit\n")
    _, build = _build(wd)
    msgs = build()
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "# REPO MAP" in user and "core.py" in user, "repo map missing from slice"
    assert "# WORKSPACE STATE (LIVE" in user and "modified: pkg/core.py" in user, "live worktree missing"
    # PROJECT facts (static) live in the SYSTEM message; live git does NOT (cache stability)
    assert "# PROJECT" in system, "static project facts missing from system msg"
    # the live git DATA must not be in the cacheable system msg (the disclaimer may NAME the region)
    assert "modified: pkg/core.py" not in system and "changed file(s)" not in system, \
        "live worktree DATA leaked into the cacheable system message"


@check
def system_message_byte_stable_but_worktree_is_live():
    wd = _mk_repo()
    _, build = _build(wd)
    sys1 = build()[0]["content"]
    open(os.path.join(wd, "new2.py"), "w").write("x=1\n")              # change the world
    msgs2 = build()
    sys2, user2 = msgs2[0]["content"], msgs2[1]["content"]
    assert sys1 == sys2, "system message must stay byte-stable across builds (prompt cache)"
    assert "untracked: new2.py" in user2, "live worktree must reflect the new file (freshness)"


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
