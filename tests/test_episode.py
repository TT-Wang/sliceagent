"""Offline tests for the episodic cache (MEMORY-SPEC step 1). No model, no pytest.
Run: python tests/test_episode.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.episode import EpisodeSink, make_episode_sink   # noqa: E402
from memagent.events import (AssistantText, SliceBuilt, ToolResult,   # noqa: E402
                             TurnEnd, TurnInterrupted)
from memagent.memory import NullMemory   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class DurableDouble:
    is_durable = True
    def __init__(self):
        self.records = []
    def append_episode(self, session_id, task_id, turn, record):
        self.records.append((session_id, task_id, turn, record))


def _sink(mem):
    return EpisodeSink(mem, session_id="s1", task_id_fn=lambda: "t1")

def tr(name="run_command", out="ok", failing=False, **args):
    return ToolResult(name, dict(args), out, failing)


@check
def gating():
    assert make_episode_sink(NullMemory(), session_id="s", task_id_fn=lambda: "t") is None
    assert make_episode_sink(DurableDouble(), session_id="s", task_id_fn=lambda: "t") is not None

@check
def one_turn_one_record():
    d = DurableDouble(); s = _sink(d)
    s(SliceBuilt("SLICE-A"))
    s(tr("edit_file", "Wrote 8 bytes to a.py", path="a.py"))   # an EDIT → meta['files'] captures it
    s(tr("run_command", "1 passed", note="tests pass"))
    s(TurnEnd("end_turn", 2, {"prompt_tokens": 100, "completion_tokens": 20}))
    assert len(d.records) == 1
    sid, tid, turn, rec = d.records[0]
    assert (sid, tid, turn) == ("s1", "t1", 1)
    assert len(rec["steps"]) == 1
    st = rec["steps"][0]
    assert st["slice"] == "SLICE-A"
    assert len(st["action"]) == 2 and len(st["observation"]) == 2
    assert rec["note"] == "tests pass"
    assert rec["meta"]["ptok"] == 100 and rec["meta"]["ctok"] == 20   # from TurnEnd, not StepEnd
    assert rec["meta"]["stop_reason"] == "end_turn"
    assert "a.py" in rec["meta"]["files"]

@check
def per_turn_reset():
    d = DurableDouble(); s = _sink(d)
    s(SliceBuilt("A")); s(tr("edit_file", out="x", path="a.py")); s(TurnEnd("end_turn", 1, {}))
    s(SliceBuilt("B")); s(tr("edit_file", out="y", path="b.py")); s(TurnEnd("end_turn", 1, {}))
    assert len(d.records) == 2
    r2 = d.records[1][3]
    assert r2["meta"]["files"] == ["b.py"]              # no a.py bleed
    assert r2["steps"][0]["slice"] == "B"

@check
def lossless_observation():
    d = DurableDouble(); s = _sink(d)
    big = "X" * 5000
    s(SliceBuilt("A")); s(tr(out=big)); s(TurnEnd("end_turn", 1, {}))
    assert d.records[0][3]["steps"][0]["observation"][0] == big   # verbatim, not observe()-truncated

@check
def aborted_turn_flushes():
    d = DurableDouble(); s = _sink(d)
    s(SliceBuilt("A")); s(tr("edit_file", out="x", path="a.py"))
    s(TurnInterrupted("aborted"))                       # loop returns WITHOUT TurnEnd
    assert len(d.records) == 1 and d.records[0][3]["meta"]["stop_reason"] == "aborted"
    s(SliceBuilt("B")); s(tr("edit_file", out="y", path="b.py")); s(TurnEnd("end_turn", 1, {}))
    assert len(d.records) == 2 and d.records[1][3]["meta"]["files"] == ["b.py"]

@check
def max_steps_double_flush_guard():
    d = DurableDouble(); s = _sink(d)
    s(SliceBuilt("A")); s(tr(out="x"))
    s(TurnInterrupted("max_steps"))                     # loop emits this...
    s(TurnEnd("max_steps", 40, {}))                     # ...then this (empty) → must NOT add a record
    assert len(d.records) == 1

@check
def multi_step_pairing():
    d = DurableDouble(); s = _sink(d)
    s(SliceBuilt("S1")); s(tr("read_file", "r1", path="a.py"))
    s(SliceBuilt("S2")); s(tr("str_replace", "ok", path="a.py"))
    s(TurnEnd("end_turn", 2, {}))
    steps = d.records[0][3]["steps"]
    assert len(steps) == 2
    assert steps[0]["slice"] == "S1" and steps[0]["action"][0]["name"] == "read_file"
    assert steps[1]["slice"] == "S2" and steps[1]["action"][0]["name"] == "str_replace"

@check
def note_from_assistant_text():
    d = DurableDouble(); s = _sink(d)
    s(SliceBuilt("A")); s(AssistantText("root cause X")); s(tr(out="ok")); s(TurnEnd("end_turn", 1, {}))
    assert d.records[0][3]["note"] == "root cause X"

@check
def nullmemory_noop():
    tmp = tempfile.mkdtemp()
    os.environ["MEMAGENT_VAULT"] = tmp
    m = NullMemory()
    assert m.is_durable is False
    assert m.append_episode("s", "t", 1, {}) is None
    assert m.load_task("x") is None and m.list_session_tasks("s") == []
    assert os.listdir(tmp) == []                        # wrote nothing


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
