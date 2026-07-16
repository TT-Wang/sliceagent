"""ACTIVE FOCUS — cross-turn continuity for EXTERNAL work (the hunter 'index.ts' miss): the dir the
agent works in via run_command is auto-granted to the file tools, and now SURFACED so the model resolves
follow-up referents there instead of cold-searching the workspace or re-asking.
No model, no pytest. Run: PYTHONPATH=src python tests/test_active_focus.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.tools import LocalToolHost                          # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn

HOME = os.path.realpath(os.path.expanduser("~"))


def _home_dirs():
    """A workspace + a sibling external project, BOTH under HOME (so _grant_shell_paths will grant them)."""
    return (tempfile.mkdtemp(prefix="focus-ws-", dir=HOME), tempfile.mkdtemp(prefix="focus-proj-", dir=HOME))


# ---- render (pure) -----------------------------------------------------------------------------
@check
def render_focus_renders_and_suppresses():
    from sliceagent.regions import render_focus
    ws, proj = HOME + "/x/sliceagent", HOME + "/x/hunter"
    out = render_focus(proj, [proj], home=HOME, workspace=ws)
    assert "~/x/hunter" in out and "read_file" in out and "RECENT CONVERSATION" in out, out
    assert render_focus(None, [], home=HOME, workspace=ws) == "", "single-workspace → suppressed"
    assert render_focus(ws, [ws], home=HOME, workspace=ws) == "", "focus == workspace → suppressed"


# ---- focus tracking + reach --------------------------------------------------------------------
@check
def grant_shell_paths_sets_focus_and_file_reach():
    ws, proj = _home_dirs()
    try:
        host = LocalToolHost(root=ws)
        assert host.focus() == (None, []), "no external work yet → no focus"
        host._grant_shell_paths(f"ls {proj}/lib")          # shell works on an external HOME-subtree dir
        focus, extra = host.focus()
        assert focus == os.path.realpath(proj), (focus, proj)
        assert os.path.realpath(proj) in host.allowed_roots(), "reach must follow the shell"
        # the file tools can now read inside the focus (the reach the slice now advertises)
        open(os.path.join(proj, "index.ts"), "w").write("export const x = 1\n")
        assert "export const x" in host._t_read_file({"path": os.path.join(proj, "index.ts")})
    finally:
        shutil.rmtree(ws, ignore_errors=True); shutil.rmtree(proj, ignore_errors=True)


@check
def focus_is_host_level_and_survives_a_seal():
    ws, proj = _home_dirs()
    try:
        from sliceagent.pfc import Slice
        host = LocalToolHost(root=ws)
        host._grant_shell_paths(f"cat {proj}/package.json")
        before = host.focus()
        assert before[0] == os.path.realpath(proj)
        Slice().seal()                                     # a turn-boundary seal must NOT drop host focus
        assert host.focus() == before, "focus persists across turns (it's host-level, like the granted reach)"
    finally:
        shutil.rmtree(ws, ignore_errors=True); shutil.rmtree(proj, ignore_errors=True)


# ---- integration: the focus tier renders in the built slice ------------------------------------
@check
def active_focus_renders_in_the_built_slice():
    ws, proj = _home_dirs()
    try:
        from sliceagent.memory import NullMemory
        from sliceagent.retriever import NullRetriever
        from sliceagent.pfc import Slice
        from sliceagent.seed import make_build_slice
        host = LocalToolHost(root=ws)
        host._grant_shell_paths(f"ls {proj}")
        s = Slice(); s.reset("look into index.ts")
        msgs = make_build_slice(s, host, NullRetriever(), NullMemory(), "look into index.ts")()
        user = msgs[1]["content"]
        text = user if isinstance(user, str) else " ".join(
            p.get("text", "") for p in user if isinstance(p, dict))
        assert "CURRENT PROJECT" in text, "the current-project tier is missing from the slice"
        assert os.path.basename(proj) in text, "the focus path is missing from the slice"
    finally:
        shutil.rmtree(ws, ignore_errors=True); shutil.rmtree(proj, ignore_errors=True)


# ---- the framing rule --------------------------------------------------------------------------
@check
def system_prompt_resolves_referents_before_asking():
    from sliceagent.prompt import SYSTEM_PROMPT
    low = SYSTEM_PROMPT.lower()
    assert "current project" in low, "prompt must reference the CURRENT PROJECT tier"
    assert "resolve before asking" in low, "prompt must teach resolve-before-asking"
    assert "history/" in low, "prompt must point to the history/ files before re-asking"


@check
def system_prompt_is_autonomous_except_at_material_or_consequential_forks():
    from sliceagent.prompt import MEMORY_ACCUMULATE, SYSTEM_PROMPT
    low = SYSTEM_PROMPT.lower()
    memory = MEMORY_ACCUMULATE.lower()
    assert "autonomy first" in low
    assert "material ambiguity" in low and "consequential external action" in low
    assert "routine observation, task-local edits, tests" in low
    assert "clarify before committing" not in low
    assert "host-enforced effect ceiling" not in memory
    assert "reasonable judgment" in memory
    # Autonomy changes action selection, never evidence standards.
    assert "canonical execution receipts" in memory
    assert "never reconstruct what the prior answer said from plausibility" in memory
    for stale_host_promise in (
        "host checks and removes", "host replaces", "host-checked against", "before publication",
    ):
        assert stale_host_promise not in memory


# ---- bounded-C: the current project (relative base) follows focus; the boundary floor never moves -----
@check
def resolution_base_follows_current_project_but_floor_stays():
    ws, proj = _home_dirs()
    try:
        host = LocalToolHost(root=ws)
        assert host.resolution_base() == os.path.realpath(ws), "no focus → base is the boundary root"
        assert host._resolve("a.py") == os.path.realpath(os.path.join(ws, "a.py")), "bare rel → boundary root"
        host._grant_shell_paths(f"ls {proj}")                       # move into the other project
        assert host.resolution_base() == os.path.realpath(proj), "after move → base is the current project"
        assert host._resolve("a.py") == os.path.realpath(os.path.join(proj, "a.py")), "bare rel now follows project"
        assert os.path.realpath(ws) in host.allowed_roots(), "the boundary floor never moved (ws still reachable)"
    finally:
        shutil.rmtree(ws, ignore_errors=True); shutil.rmtree(proj, ignore_errors=True)


@check
def open_files_stay_truthful_after_the_project_moves():
    # I2 guard: a workspace pin must NOT re-resolve against the moved base and lie '(not created yet)'.
    from sliceagent.pfc import Slice, touch_file
    from sliceagent.seed import build_artifacts
    ws, proj = _home_dirs()
    try:
        host = LocalToolHost(root=ws)
        open(os.path.join(ws, "core.py"), "w").write("print('in ws')\n")
        s = Slice(); s.reset("t"); touch_file(s, "core.py")        # pinned by relative name, base = ws
        host._grant_shell_paths(f"ls {proj}")                       # current project moves to proj
        out = build_artifacts(s, host)
        assert "in ws" in out and "(not created yet)" not in out, \
            "workspace pin must stay truthful after the project moved (locate is base-stable)"
    finally:
        shutil.rmtree(ws, ignore_errors=True); shutil.rmtree(proj, ignore_errors=True)


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
