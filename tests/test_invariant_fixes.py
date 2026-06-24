"""Post-redesign adversary-fix coverage: shell-grant reach (I2 wired on the real path),
bounded action_log (no-transcript), and the per-turn call-budget floor (I3 backstop).
No model, no pytest.  Run: PYTHONPATH=src python tests/test_invariant_fixes.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.guardrails import ToolCallGuardrail                      # noqa: E402
from memagent.slice import (MAX_ACTION_LOG, MAX_ACTION_SHOWN, Slice,    # noqa: E402
                            action_sig, record_action, render_action_history)
from memagent.tools import LocalToolHost                                # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- I2: reach FOLLOWS action — a shell-touched external dir becomes file-tool reachable ----
@check
def shell_grant_makes_external_dir_reachable():
    home = os.path.realpath(os.path.expanduser("~"))
    ext = tempfile.mkdtemp(dir=home, prefix=".memagent-test-")   # under HOME, outside the workspace
    ws = tempfile.mkdtemp(prefix="memagent-ws-")
    try:
        host = LocalToolHost(ws)
        threw = False
        try:
            host._resolve(os.path.join(ext, "x.txt"))
        except PermissionError:
            threw = True
        assert threw, "external dir must be OUT of reach before any shell action"
        host._grant_shell_paths(f'mkdir -p "{ext}"')             # shell acts there → grant
        host._resolve(os.path.join(ext, "x.txt"))                # no raise now
        assert os.path.realpath(ext) in host.allowed_roots()
    finally:
        shutil.rmtree(ext, ignore_errors=True)
        shutil.rmtree(ws, ignore_errors=True)


@check
def shell_grant_refuses_home_and_ancestors():
    home = os.path.realpath(os.path.expanduser("~"))
    ws = tempfile.mkdtemp(prefix="memagent-ws-")
    try:
        host = LocalToolHost(ws)
        host._grant_shell_paths(f'ls "{home}"')                  # HOME itself must NOT be granted
        host._grant_shell_paths("cat /etc/hosts")               # outside HOME → not granted
        assert home not in host.allowed_roots()
        assert "/etc" not in host.allowed_roots()
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# ---- no-transcript: the anti-loop tally is bounded ----
@check
def action_log_is_bounded():
    s = Slice(); s.reset("t")
    for i in range(MAX_ACTION_LOG + 30):
        record_action(s, "read_file", {"path": f"f{i}.py"}, "ok")   # distinct one-shot non-failing
    assert len(s.action_log) <= MAX_ACTION_LOG, len(s.action_log)


@check
def failing_actions_survive_eviction():
    s = Slice(); s.reset("t")
    record_action(s, "run_command", {"command": "boom"}, "Error: boom")  # failing — high signal
    fsig = action_sig("run_command", {"command": "boom"})
    for i in range(MAX_ACTION_LOG + 10):
        record_action(s, "read_file", {"path": f"g{i}.py"}, "ok")
    assert fsig in s.action_log, "failing entry must survive eviction (anti-loop signal)"


@check
def render_action_history_caps_rendered():
    s = Slice(); s.reset("t")
    for i in range(MAX_ACTION_SHOWN + 6):
        for _ in range(2):                                          # count>=2 so it qualifies to show
            record_action(s, "list_files", {"path": f"d{i}"}, "x")
    out = render_action_history(s.action_log)
    assert "more repeated/failing (omitted)" in out, out
    assert out.count("\n- ") <= MAX_ACTION_SHOWN, "rendered entries must be capped"


# ---- I3 backstop: per-turn call budget ----
@check
def call_budget_blocks_a_nonprogress_spree():
    # The floor backstops genuine FLAILING: calls that make NO progress — each FAILS (or re-observes an
    # already-seen result). Distinct FAILING reads isolate the floor: they don't trip the per-signature
    # exact-failure axis (each path is unique), the result axis (each error is unique), or the no-edit
    # mutation axis (read_file is non-mutating). 18 of them = exploring in circles → block.
    g = ToolCallGuardrail()
    n = g.config.call_budget_warn_after
    for i in range(n):
        d = g.before_call("read_file", {"path": f"missing{i}.py"})
        assert not d.block, f"should not block before the budget (call {i})"
        g.after_call("read_file", {"path": f"missing{i}.py"}, f"Error: no such file missing{i}.py", failed=True)
    d = g.before_call("read_file", {"path": "one-more.py"})
    assert d.block and d.code == "call_budget", (d.block, d.code)


@check
def distinct_successful_reads_are_progress_not_circling():
    # REGRESSION (the analysis/review-task bug): a read-only task makes MANY distinct successful reads and
    # never edits. Each distinct read returns NEW information = progress, so the call-budget floor must NOT
    # fire — analysis / review / debugging-by-reading is legitimate work, not "exploring in circles". Drive
    # 2x the old floor to prove distinct reads never accumulate it.
    g = ToolCallGuardrail()
    for i in range(g.config.call_budget_warn_after * 2):
        d = g.before_call("read_file", {"path": f"src/mod{i}.py"})
        assert not d.block, f"distinct read #{i} wrongly blocked as a no-progress spree (code={d.code})"
        g.after_call("read_file", {"path": f"src/mod{i}.py"}, f"contents of module {i}: def f{i}(): return {i}")


@check
def repeated_reads_still_trip_the_floor():
    # but re-observing the SAME output IS non-progress: distinct ARGS returning an identical result (so the
    # per-signature idempotent axis can't see it) must still accumulate the floor. (Belt-and-suspenders with
    # the result axis, which trips earlier at result_repeat_block_after.)
    g = ToolCallGuardrail()
    blocked = False
    for i in range(g.config.call_budget_warn_after + 2):
        d = g.before_call("read_file", {"path": f"alias{i}.py"})   # distinct args …
        if d.block:
            blocked = True
            break
        g.after_call("read_file", {"path": f"alias{i}.py"}, "IDENTICAL OUTPUT EVERY TIME")  # … same result
    assert blocked, "re-observing the same result via distinct args must still trip a no-progress block"


@check
def successful_edit_resets_the_budget():
    g = ToolCallGuardrail()
    for i in range(g.config.call_budget_warn_after - 1):
        g.after_call("read_file", {"path": f"f{i}.py"}, f"c{i}")
    g.after_call("edit_file", {"path": "x.py", "content": "y"}, "Wrote 1 bytes to x.py")  # change lands
    for i in range(5):
        d = g.before_call("read_file", {"path": f"g{i}.py"})
        assert not d.block, "budget must reset after a successful change"
        g.after_call("read_file", {"path": f"g{i}.py"}, f"d{i}")


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
