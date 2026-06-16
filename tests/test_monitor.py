"""Slice-monitor tests — store/sink shape + a live-server smoke check. No model, no pytest.
Run: python tests/test_monitor.py
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import (  # noqa: E402
    AssistantText, SliceBuilt, StepEnd, ToolResult, TurnEnd, TurnInterrupted)
from memagent.monitor import SliceMonitor, serve  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def sb(system, user):
    return SliceBuilt(user, [{"role": "system", "content": system}, {"role": "user", "content": user}])


@check
def captures_full_messages():
    m = SliceMonitor()
    m.sink(sb("SYS", "USER-SLICE"))
    s = m.snapshot()["steps"][0]
    assert s["system"] == "SYS" and s["user"] == "USER-SLICE"
    assert s["turn"] == 1 and s["step"] == 1 and s["i"] == 0


@check
def multi_step_single_turn():
    m = SliceMonitor()
    m.sink(sb("S", "u1")); m.sink(sb("S", "u2"))
    steps = m.snapshot()["steps"]
    assert [s["step"] for s in steps] == [1, 2]
    assert all(s["turn"] == 1 for s in steps)


@check
def turnend_starts_new_turn():
    m = SliceMonitor()
    m.sink(sb("S", "u1")); m.sink(TurnEnd("end_turn", 1, {}))
    m.sink(sb("S", "u2"))
    steps = m.snapshot()["steps"]
    assert steps[0]["turn"] == 1 and steps[1]["turn"] == 2 and steps[1]["step"] == 1


@check
def captures_assistant_tools_usage_stop():
    m = SliceMonitor()
    m.sink(sb("S", "u"))
    m.sink(AssistantText("thinking..."))
    m.sink(ToolResult("read_file", {"path": "a.py"}, "contents", False))
    m.sink(ToolResult("run_command", {"command": "pytest"}, "Error: boom", True))
    m.sink(StepEnd(1, {"prompt_tokens": 100, "completion_tokens": 20}, "tool_use"))
    s = m.snapshot()["steps"][0]
    assert s["assistant"] == "thinking..."
    assert [t["name"] for t in s["tools"]] == ["read_file", "run_command"]
    assert s["tools"][1]["failing"] is True and "path" in s["tools"][0]["args"]
    assert s["usage"]["prompt_tokens"] == 100 and s["stop_reason"] == "tool_use"


@check
def fallback_when_no_messages():
    m = SliceMonitor()
    m.sink(SliceBuilt("just-the-user-text"))    # legacy positional build, no messages
    s = m.snapshot()["steps"][0]
    assert s["user"] == "just-the-user-text" and s["system"] == ""


@check
def interrupted_tagged_and_closes_turn():
    m = SliceMonitor()
    m.sink(sb("S", "u")); m.sink(TurnInterrupted("max_steps"))
    m.sink(sb("S", "u2"))                        # next slice → new turn
    steps = m.snapshot()["steps"]
    assert steps[0]["interrupted"] == "max_steps"
    assert steps[1]["turn"] == 2


@check
def context_fn_captured_per_step():
    box = {"goal": "task A", "topic": "t-aaa"}
    m = SliceMonitor(context_fn=lambda: dict(box))
    m.sink(sb("S", "u1"))
    box["goal"], box["topic"] = "task B", "t-bbb"
    m.sink(TurnEnd("end_turn", 1, {})); m.sink(sb("S", "u2"))
    steps = m.snapshot()["steps"]
    assert steps[0]["goal"] == "task A" and steps[0]["topic"] == "t-aaa"
    assert steps[1]["goal"] == "task B" and steps[1]["topic"] == "t-bbb"


@check
def context_fn_failure_is_safe():
    def boom():
        raise RuntimeError("nope")
    m = SliceMonitor(context_fn=boom)
    m.sink(sb("S", "u"))                          # must not raise
    assert m.snapshot()["steps"][0]["goal"] == ""


@check
def snapshot_totals_and_version():
    m = SliceMonitor()
    v0 = m.snapshot()["version"]
    m.sink(sb("S", "u")); m.sink(StepEnd(1, {"prompt_tokens": 10, "completion_tokens": 5}, "end_turn"))
    m.sink(TurnEnd("end_turn", 1, {}))
    snap = m.snapshot()
    assert snap["tokens"] == 15 and snap["turns"] == 1 and snap["steps_total"] == 1
    assert snap["version"] > v0


@check
def large_output_clipped():
    m = SliceMonitor()
    m.sink(sb("S", "u"))
    m.sink(ToolResult("run_command", {}, "x" * 20000, False))
    out = m.snapshot()["steps"][0]["tools"][0]["output"]
    assert len(out) < 20000 and "chars]" in out


@check
def snapshot_independent_of_live_mutation():
    # the snapshot must not share the live step's mutable tools list — else json.dumps (outside the
    # lock) can race with the loop thread appending a tool result mid-poll.
    m = SliceMonitor()
    m.sink(sb("S", "u"))
    m.sink(ToolResult("read_file", {}, "one", False))
    snap = m.snapshot()
    assert len(snap["steps"][0]["tools"]) == 1
    m.sink(ToolResult("run_command", {}, "two", False))   # live mutation AFTER the snapshot
    assert len(snap["steps"][0]["tools"]) == 1             # snapshot frozen, not retro-mutated
    assert len(m.snapshot()["steps"][0]["tools"]) == 2     # fresh snapshot sees both


@check
def live_server_smoke():
    m = SliceMonitor()
    m.sink(sb("SYSTEM-PROMPT", "ACTIVE SLICE TEXT"))
    srv, url = serve(m, port=7790)
    try:
        page = urllib.request.urlopen(url + "/", timeout=3).read().decode()
        assert "active memory slice" in page and "/api/state" in page
        state = json.loads(urllib.request.urlopen(url + "/api/state", timeout=3).read().decode())
        assert state["steps_total"] == 1 and state["steps"][0]["user"] == "ACTIVE SLICE TEXT"
    finally:
        srv.shutdown()


@check
def file_sink_persists_snapshot():
    import os
    import tempfile
    from memagent.monitor import _session_files, make_file_monitor_sink
    d = tempfile.mkdtemp()
    sink = make_file_monitor_sink("sess-A", dir=d)
    sink(sb("SYS", "USER-SLICE"))
    sink(StepEnd(1, {"prompt_tokens": 10, "completion_tokens": 5}, "end_turn"))
    p = os.path.join(d, "sess-A.json")
    assert os.path.exists(p)
    snap = json.load(open(p))
    assert snap["session"] == "sess-A" and snap["steps"][0]["user"] == "USER-SLICE"
    assert [s for s, _ in _session_files(d)] == ["sess-A"]


@check
def persistent_server_idle_and_sessions():
    import os
    import tempfile
    import threading
    import time
    import urllib.request
    from http.server import ThreadingHTTPServer
    from memagent.monitor import IDLE_SECONDS, _PersistentHandler, make_file_monitor_sink
    d = tempfile.mkdtemp()
    srv = ThreadingHTTPServer(("127.0.0.1", 7793), _PersistentHandler)
    srv.monitor_dir = d
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        # empty dir → idle, no session
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:7793/api/state", timeout=3).read())
        assert st["idle"] is True and st["session"] is None and st["sessions"] == []
        # a fresh session → live (not idle), shows up
        make_file_monitor_sink("live-1", dir=d)(sb("S", "U"))
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:7793/api/state", timeout=3).read())
        assert st["session"] == "live-1" and st["idle"] is False and "live-1" in st["sessions"]
        # a stale file → idle (age past threshold), but still served (doesn't die)
        old = time.time() - IDLE_SECONDS - 50
        os.utime(os.path.join(d, "live-1.json"), (old, old))
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:7793/api/state", timeout=3).read())
        assert st["idle"] is True and st["session"] == "live-1" and st["steps"][0]["user"] == "U"
    finally:
        srv.shutdown()


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
