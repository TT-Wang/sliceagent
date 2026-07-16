"""Post-redesign adversary-fix coverage: shell-grant reach (I2 wired on the real path)
and bounded action_log (no-transcript).
No model, no pytest.  Run: PYTHONPATH=src python tests/test_invariant_fixes.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.pfc import Slice  # noqa: E402
from sliceagent.regions import MAX_ACTION_LOG, MAX_ACTION_SHOWN, action_sig, record_action, render_action_history  # noqa: E402
from sliceagent.tools import LocalToolHost                                # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- I2: reach FOLLOWS action — a shell-touched external dir becomes file-tool reachable ----
@check
def shell_grant_makes_external_dir_reachable():
    home = os.path.realpath(os.path.expanduser("~"))
    ext = tempfile.mkdtemp(dir=home, prefix=".sliceagent-test-")   # under HOME, outside the workspace
    ws = tempfile.mkdtemp(prefix="sliceagent-ws-")
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
    ws = tempfile.mkdtemp(prefix="sliceagent-ws-")
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


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
