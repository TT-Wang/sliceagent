"""recall_history tool + read_episodes tests. No model, no pytest.
Run: python tests/test_history.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.history import make_history_tool, render_index, render_trace  # noqa: E402
from memagent.memory import NullMemory  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _rec(title, note, steps, failing=False):
    return {"title": title, "note": note, "steps": steps,
            "meta": {"failing": failing, "stop_reason": "end_turn", "files": []}}


def _memem():
    from memagent.memory import MememMemory
    m = MememMemory()
    m._vault = tempfile.mkdtemp()
    return m


@check
def append_and_read_roundtrip_with_ts_topic_title():
    try:
        m = _memem()
    except Exception:
        print("  (skip: memem not importable)"); return
    m.append_episode("s1", "t-aaa", 1, _rec("fix the parser", "root cause: off-by-one",
                                            [{"slice": "SLICE1", "action": [{"name": "read_file", "args": {"path": "p.py"}, "failing": False}],
                                              "observation": ["file contents"]}]))
    m.append_episode("s1", "t-bbb", 2, _rec("add range query", "done", [{"slice": "S2", "action": [], "observation": []}]))
    lines = m.read_episodes("s1")
    assert len(lines) == 2
    assert lines[0]["task_id"] == "t-aaa" and lines[0]["ts"] and lines[0]["record"]["title"] == "fix the parser"
    assert m.read_episodes("s1", limit=1)[0]["turn"] == 2          # most-recent N
    assert m.read_episodes("nope") == []                           # missing session → []


@check
def index_shows_breadcrumbs():
    lines = [{"task_id": "t-aaa", "turn": 1, "ts": "2026-06-16T12:30:00",
              "record": _rec("fix the parser", "off-by-one in tokenizer", [{"slice": "", "action": [], "observation": []}])}]
    idx = render_index(lines)
    assert "turn 1" in idx and "fix the parser" in idx and "off-by-one" in idx and "t-aaa" in idx and "12:30" in idx


@check
def tool_no_args_returns_index():
    try:
        m = _memem()
    except Exception:
        print("  (skip)"); return
    m.append_episode("s1", "t1", 1, _rec("task one", "did a thing", [{"slice": "", "action": [], "observation": []}]))
    tool = make_history_tool(m, "s1")
    out = tool.handler({})
    assert "CACHED HISTORY" in out and "task one" in out


@check
def tool_fetch_trace_and_specific_turn():
    try:
        m = _memem()
    except Exception:
        print("  (skip)"); return
    m.append_episode("s1", "t1", 1, _rec("t-one", "n1",
                     [{"slice": "SL1", "action": [{"name": "run_command", "args": {"command": "pytest"}, "failing": True}],
                       "observation": ["FAILED 3 tests"]}]))
    m.append_episode("s1", "t1", 2, _rec("t-two", "n2", [{"slice": "SL2", "action": [], "observation": []}]))
    tool = make_history_tool(m, "s1")
    last = tool.handler({"last": 1})
    assert "turn 2" in last and "turn 1" not in last              # only the most recent
    one = tool.handler({"turns": [1]})
    assert "run_command" in one and "FAILED 3 tests" in one and "n1" in one   # action+obs+note
    full = tool.handler({"turns": [1], "full": True})
    assert "SL1" in full                                          # full mode returns the stored slice


@check
def trace_is_bounded():
    lines = [{"task_id": "t", "turn": i, "ts": "2026-06-16T12:00:00",
              "record": _rec(f"turn{i}", "", [{"slice": "", "action": [{"name": "run", "args": {}, "failing": False}],
                                               "observation": ["x" * 5000]}])} for i in range(1, 20)]
    out = render_trace(lines, cap=2000)
    assert len(out) < 2600 and "truncated" in out                # capped regardless of N requested


class _FakeMem:
    def __init__(self, lines):
        self._lines = lines
    def read_episodes(self, session_id, *, limit=None):
        return self._lines


@check
def repeat_redirected_but_distinct_search_allowed():
    lines = [{"task_id": "t", "turn": i, "ts": "2026-06-16T12:00:00",
              "record": _rec(f"turn{i}", "note", [{"slice": f"S{i}", "action": [], "observation": []}])}
             for i in range(1, 15)]
    fake = _FakeMem(lines)
    tool = make_history_tool(fake, "s")
    a = tool.handler({"turns": [1]})
    assert "turn 1" in a and "Record what you need" in a            # served + capture-back
    b = tool.handler({"turns": [1]})                                # EXACT repeat → redirect, no re-dump
    assert "already pulled" in b and "S1" not in b
    for i in range(2, 9):                                           # 7 more DISTINCT drills (never blocked)
        assert f"turn {i}" in tool.handler({"turns": [i]}), f"distinct fetch {i} must be served"
    over = tool.handler({"turns": [9]})                            # past the generous backstop → to index
    assert "index" in over.lower() and "turn 9" not in over
    assert "CACHED HISTORY" in tool.handler({})                    # index is free (not counted as a drill)
    assert "already pulled" in tool.handler({})                    # repeat index → redirect
    fake._lines = lines + [dict(lines[0], turn=99)]               # cache grew → new turn → rein resets
    assert "turn 1" in tool.handler({"turns": [1]})               # can drill again


@check
def ratchet_folds_lookback_into_slice():
    from memagent.events import ToolResult
    from memagent.slice import Slice, render_reviewed, slice_sink
    s = Slice(); s.reset("task")
    sink = slice_sink(s)
    sink(ToolResult("recall_history", {}, "index", False))            # index lookback
    sink(ToolResult("recall_history", {"turns": [3]}, "trace", False))  # a drill
    assert "index" in s.reviewed and "turns=[3]" in s.reviewed         # advanced the slice state
    sink(ToolResult("recall_history", {}, "index", False))            # repeat → deduped
    assert s.reviewed.count("index") == 1
    rev = render_reviewed(s)
    assert "HISTORY REVIEWED" in rev and "do NOT re-fetch" in rev and "turns=[3]" in rev
    s2 = Slice(); s2.reset("t")
    slice_sink(s2)(ToolResult("recall_history", {}, "boom", True))    # failing lookback → not recorded
    assert s2.reviewed == [] and render_reviewed(s2) == ""


@check
def reviewed_is_temporal_not_permanent():
    # the ratchet must clear between directives/turns, or a past lookback contaminates future moves
    from memagent.memory import NullMemory
    from memagent.session import Session
    from memagent.slice import Slice
    s = Slice(); s.reset("task A"); s.reviewed = ["index", "turns=[3]"]
    s.reset("task B")
    assert s.reviewed == []                              # new_topic / reset → clean slate
    sess = Session(NullMemory(), "s")
    sess.new_topic("do X"); sess.active().reviewed = ["index"]
    sess.continue_topic("now do Y")
    assert sess.active().reviewed == []                  # a new directive clears it


@check
def nullmemory_has_no_history():
    m = NullMemory()
    assert m.read_episodes("s1") == []
    assert "No cached history" in make_history_tool(m, "s1").handler({})


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
