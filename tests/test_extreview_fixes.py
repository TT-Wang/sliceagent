"""Regression tests for the EXTERNAL security-review fixes: H-13 (WAL redacts tool_call arguments),
H-14 (proc_kill reaps a background child that outlived the shell leader), H-06 (git write/exec via
diff helpers is not auto-approved / is disabled in code_review). No model, no network.
Run: PYTHONPATH=src python tests/test_extreview_fixes.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- H-13: WAL sanitization must redact secrets in tool_calls[*].function.arguments -------------------

@check
def h13_wal_redacts_tool_call_arguments():
    from sliceagent.recovery import _sanitize
    secret = "sk-ant-api03-" + "A" * 40
    msgs = [
        {"role": "assistant", "content": "calling a tool",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "run_command",
                                      "arguments": '{"command": "deploy --key ' + secret + '"}'}}]},
        {"role": "tool", "content": f"used {secret}"},
    ]
    out = _sanitize(msgs)
    assert secret not in str(out), "secret leaked through _sanitize (tool_calls args or content)"
    tc = out[0]["tool_calls"][0]
    assert tc["id"] == "c1" and tc["type"] == "function" and isinstance(tc["function"]["arguments"], str)
    assert out[0]["content"] == "calling a tool"
    assert "tool_calls" not in out[1]


@check
def h13_handles_malformed_tool_calls_without_crashing():
    from sliceagent.recovery import _sanitize
    weird = [
        {"role": "assistant", "tool_calls": "not-a-list"},
        {"role": "assistant", "tool_calls": [{"no_function": True}, {"function": {"arguments": None}}, "junk"]},
        "not-a-dict",
    ]
    out = _sanitize(weird)  # must not raise
    assert out[0]["tool_calls"] == "not-a-list"
    assert out[2] == "not-a-dict"


# ---- H-06: "read-only" git must not auto-approve write/exec options ----------------------------------

@check
def h06_code_review_disables_git_helpers():
    import inspect
    from sliceagent.tools import LocalToolHost
    src = inspect.getsource(LocalToolHost._t_code_review)
    assert '"--no-ext-diff"' in src and '"--no-textconv"' in src, \
        "code_review git diff must pass --no-ext-diff --no-textconv"


# ---- H-14: killing a process must reach a background child that outlived the shell leader ------------

@check
def h14_proc_kill_reaps_orphaned_background_child():
    if os.name != "posix":
        return
    from sliceagent.procman import ProcManager
    pm = ProcManager()
    marker = f"/tmp/.sliceagent-h14-{os.getpid()}-{int(time.time() * 1000)}"
    # leader spawns a background sleep (recording its pid) then EXITS → the sleep is orphaned in the group.
    # Before the fix, kill() saw poll()!=None and never signalled the group, so the sleep survived.
    handle = pm.start(f"sleep 120 & echo $! > {marker}; exit 0", cwd="/tmp")
    child_pid = None
    for _ in range(60):
        try:
            v = open(marker).read().strip()
        except OSError:
            v = ""
        if v:
            child_pid = int(v); break
        time.sleep(0.05)
    assert child_pid, "background child never recorded its pid"
    os.kill(child_pid, 0)                     # alive now
    pm.kill(handle)
    dead = False
    for _ in range(80):
        try:
            os.kill(child_pid, 0); time.sleep(0.05)
        except ProcessLookupError:
            dead = True; break
    for p in (marker,):
        try:
            os.remove(p)
        except OSError:
            pass
    pm.cleanup()
    assert dead, f"orphaned background child {child_pid} survived proc_kill (H-14 regression)"


def main():
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
