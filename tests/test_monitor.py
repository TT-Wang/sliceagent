"""Slice-monitor tests — store/sink shape + a live-server smoke check. No model, no pytest.
Run: python tests/test_monitor.py
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import (  # noqa: E402
    AssistantText, ModelCallPrepared, SliceBuilt, StepEnd, ToolResult, TurnEnd, TurnInterrupted)
from sliceagent.monitor import SliceMonitor, serve  # noqa: E402

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
def physical_attempts_do_not_create_fake_monitor_steps():
    m = SliceMonitor()
    m.sink(sb("SYS", "INITIAL-SLICE"))
    first = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "FULL"}]
    second = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "LOCATOR"}]
    m.sink(ModelCallPrepared(1, 1, first, "roomy", "compatibility-unknown"))
    m.sink(ModelCallPrepared(1, 2, second, "critical", "compatibility-unknown"))
    first[1]["content"] = "MUTATED-AFTER-DISPATCH"

    snap = m.snapshot()
    assert len(snap["steps"]) == 1 and snap["steps_total"] == 1
    step = snap["steps"][0]
    assert step["user"] == "INITIAL-SLICE", "attempt inspection must not replace the lifecycle slice"
    assert [(call["step"], call["attempt"]) for call in step["model_calls"]] == [(1, 1), (1, 2)]
    assert step["model_calls"][0]["messages"][1]["content"] == "FULL"
    assert step["model_calls"][1]["messages"] == second


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
    import stat
    import tempfile
    from sliceagent.monitor import _session_files, make_file_monitor_sink
    d = tempfile.mkdtemp()
    sink = make_file_monitor_sink("sess-A", dir=d)
    sink(sb("SYS", "USER-SLICE"))
    sink(StepEnd(1, {"prompt_tokens": 10, "completion_tokens": 5}, "end_turn"))
    sink.writer.drain()                          # MON2: disk I/O is async — wait for it to land
    p = os.path.join(d, "sess-A.json")
    assert os.path.exists(p)
    snap = json.load(open(p))
    assert snap["session"] == "sess-A" and snap["steps"][0]["user"] == "USER-SLICE"
    assert [s for s, _ in _session_files(d)] == ["sess-A"]
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600


@check
def file_sink_repairs_modes_and_sanitizes_session_filename():
    import stat
    import tempfile
    from sliceagent.monitor import make_file_monitor_sink
    d = tempfile.mkdtemp()
    os.chmod(d, 0o755)
    sink = make_file_monitor_sink("../escape", dir=d)
    sink(sb("SYS", "sensitive")); sink.writer.drain()
    expected = os.path.join(d, "___escape.json")
    assert os.path.exists(expected) and not os.path.exists(os.path.join(os.path.dirname(d), "escape.json"))
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(expected).st_mode) == 0o600


@check
def relative_snapshot_writer_does_not_chmod_cwd():
    if os.name == "nt":
        return
    import stat
    import tempfile
    from sliceagent.monitor import _SnapshotWriter
    with tempfile.TemporaryDirectory() as cwd:
        os.chmod(cwd, 0o755)
        previous = os.getcwd()
        try:
            os.chdir(cwd)
            writer = _SnapshotWriter("state.json", debounce_ms=0)
            writer.submit({"private": True}, flush=True)
            assert writer.drain()
            writer.close()
            assert stat.S_IMODE(os.stat(cwd).st_mode) == 0o755
            assert stat.S_IMODE(os.stat("state.json").st_mode) == 0o600
        finally:
            os.chdir(previous)


@check
def persistent_server_idle_and_sessions():
    import os
    import tempfile
    import threading
    import time
    import urllib.request
    from http.server import ThreadingHTTPServer
    from sliceagent.monitor import IDLE_SECONDS, _PersistentHandler, make_file_monitor_sink
    d = tempfile.mkdtemp()
    srv = ThreadingHTTPServer(("127.0.0.1", 7793), _PersistentHandler)
    srv.monitor_dir = d
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        # empty dir → idle, no session
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:7793/api/state", timeout=3).read())
        assert st["idle"] is True and st["session"] is None and st["sessions"] == []
        # a fresh session → live (not idle), shows up
        s1 = make_file_monitor_sink("live-1", dir=d)
        s1(sb("S", "U")); s1.writer.drain()       # MON2: async write — wait for the file to land
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:7793/api/state", timeout=3).read())
        assert st["session"] == "live-1" and st["idle"] is False and "live-1" in st["sessions"]
        # a stale file → idle (age past threshold), but still served (doesn't die)
        old = time.time() - IDLE_SECONDS - 50
        os.utime(os.path.join(d, "live-1.json"), (old, old))
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:7793/api/state", timeout=3).read())
        assert st["idle"] is True and st["session"] == "live-1" and st["steps"][0]["user"] == "U"
    finally:
        srv.shutdown()


# --- Invariant 0: BOUND THE MONITOR (MON1 ring cap, MON2 decoupled writes, MON3 stale prune) ---

@check
def ring_is_capped_to_last_n_steps():
    # MON1: the in-memory ring serves only the last `cap` steps, no matter how many ran.
    m = SliceMonitor(cap=5)
    for k in range(20):
        m.sink(sb("S", f"u{k}"))
    snap = m.snapshot()
    assert len(snap["steps"]) == 5                       # ring trimmed to the cap
    assert [s["user"] for s in snap["steps"]] == [f"u{k}" for k in range(15, 20)]  # the LAST 5


@check
def counters_accurate_despite_ring_trim():
    # MON1: turns/steps_total/tokens come from FULL tallies, not the trimmed ring.
    m = SliceMonitor(cap=3)
    for k in range(10):                                  # 10 steps in one turn, ring holds only 3
        m.sink(sb("S", f"u{k}"))
        m.sink(StepEnd(1, {"prompt_tokens": 100, "completion_tokens": 10}, "tool_use"))
    snap = m.snapshot()
    assert snap["steps_total"] == 10                     # accurate total despite cap=3
    assert snap["turns"] == 1
    assert snap["tokens"] == 10 * 110                    # full token tally survives trimming
    assert len(snap["steps"]) == 3                       # but only 3 served


@check
def step_id_is_monotonic_not_list_index():
    # MON1: `i` must stay a unique monotonic id across trims so the UI can still select a step.
    m = SliceMonitor(cap=3)
    for k in range(8):
        m.sink(sb("S", f"u{k}"))
    served = m.snapshot()["steps"]
    ids = [s["i"] for s in served]
    assert ids == [5, 6, 7]                              # last 3 monotonic ids, not 0/1/2
    assert len(set(ids)) == len(ids)                     # unique


@check
def live_step_survives_trim_and_keeps_growing():
    # MON1: the live (current) step is always the last element — trimming the front never drops it.
    m = SliceMonitor(cap=2)
    for k in range(5):
        m.sink(sb("S", f"u{k}"))
    m.sink(AssistantText("after-trim"))                 # mutate the live step post-trim
    m.sink(ToolResult("read_file", {}, "ok", False))
    cur = m.snapshot()["steps"][-1]
    assert cur["user"] == "u4" and cur["assistant"] == "after-trim"
    assert [t["name"] for t in cur["tools"]] == ["read_file"]


@check
def file_write_is_off_the_hot_path_and_flushes():
    # MON2: the sink publishes to a background writer; StepEnd flushes. After drain the file is current.
    import tempfile
    from sliceagent.monitor import make_file_monitor_sink
    d = tempfile.mkdtemp()
    sink = make_file_monitor_sink("sess-w", dir=d)
    for k in range(30):                                  # a burst of hot-path events
        sink(sb("S", f"u{k}"))
    sink(StepEnd(1, {"prompt_tokens": 5, "completion_tokens": 5}, "end_turn"))
    assert sink.writer.drain() is True
    snap = json.load(open(os.path.join(d, "sess-w.json")))
    assert snap["session"] == "sess-w"
    assert snap["steps_total"] == 30                     # full count even though writes coalesced
    assert snap["steps"][-1]["user"] == "u29"           # freshest slice is what's served


@check
def file_snapshot_ring_is_bounded_on_disk():
    # MON1+MON2: the persisted snapshot is O(cap) — a long session never grows the file unboundedly.
    import tempfile
    from sliceagent.monitor import _RING_CAP, make_file_monitor_sink
    d = tempfile.mkdtemp()
    sink = make_file_monitor_sink("sess-big", dir=d)
    for k in range(_RING_CAP + 25):
        sink(sb("S", f"u{k}"))
    sink(StepEnd(1, {}, "end_turn")); sink.writer.drain()
    snap = json.load(open(os.path.join(d, "sess-big.json")))
    assert len(snap["steps"]) == _RING_CAP              # bounded on disk
    assert snap["steps_total"] == _RING_CAP + 25       # but the counter is honest


@check
def prune_drops_stale_keeps_newest():
    # MON3: files older than the TTL are dropped, but the freshest is NEVER deleted (even if stale).
    import tempfile
    import time as _t
    from sliceagent.monitor import _prune_sessions
    d = tempfile.mkdtemp()
    for sid in ("old-a", "old-b", "fresh"):
        with open(os.path.join(d, f"{sid}.json"), "w") as f:
            json.dump({"steps": []}, f)
    old = _t.time() - (48 * 3600)                        # 2 days old → past the 24h TTL
    os.utime(os.path.join(d, "old-a.json"), (old, old))
    os.utime(os.path.join(d, "old-b.json"), (old, old))
    # leave "fresh" with a current mtime, but also age it to prove the freshest is kept regardless
    survivors = [s for s, _ in _prune_sessions(d, ttl=24 * 3600)]
    assert survivors == ["fresh"]                        # stale ones gone, freshest kept
    assert os.path.exists(os.path.join(d, "fresh.json"))
    assert not os.path.exists(os.path.join(d, "old-a.json"))


@check
def prune_keeps_freshest_even_when_all_stale():
    # MON3 guarantee: even if EVERY file is stale, the most-recent one stays so the active session shows.
    import tempfile
    import time as _t
    from sliceagent.monitor import _prune_sessions
    d = tempfile.mkdtemp()
    for i, sid in enumerate(("s0", "s1", "s2")):
        p = os.path.join(d, f"{sid}.json")
        with open(p, "w") as f:
            json.dump({"steps": []}, f)
        old = _t.time() - (48 * 3600) - i               # all stale; s0 most recent
        os.utime(p, (old, old))
    survivors = [s for s, _ in _prune_sessions(d, ttl=1)]
    assert survivors == ["s0"]                           # freshest survives despite being stale


@check
def prune_caps_to_most_recent_m():
    # MON3: keep only the most-recent M files (newest never deleted).
    import tempfile
    import time as _t
    from sliceagent.monitor import _prune_sessions
    d = tempfile.mkdtemp()
    for i in range(6):
        p = os.path.join(d, f"s{i}.json")
        with open(p, "w") as f:
            json.dump({"steps": []}, f)
        t = _t.time() - i                                # s0 newest … s5 oldest, all within TTL
        os.utime(p, (t, t))
    survivors = [s for s, _ in _prune_sessions(d, ttl=10 * 3600, keep=2)]
    assert survivors == ["s0", "s1"]                     # only the 2 most-recent kept
    assert not os.path.exists(os.path.join(d, "s5.json"))


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
