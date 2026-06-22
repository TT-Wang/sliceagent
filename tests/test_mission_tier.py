"""MISSION tier (Kimi goal mode): a north-star objective set via set_mission, cleared via mission_done.
Self-suppressing (no bloat when unset), carried by seal(), wiped by reset(), round-trips through TaskState.
No model, no pytest. Run: PYTHONPATH=src python tests/test_mission_tier.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import ToolResult  # noqa: E402
from memagent.slice import Slice, render_slice, slice_sink  # noqa: E402
from memagent.taskstate import slice_to_task_state, task_state_to_slice  # noqa: E402
from memagent.tools import LocalToolHost  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def set_and_clear_mission():
    s = Slice(); s.reset("x"); sink = slice_sink(s)
    sink(ToolResult("set_mission", {"text": "ship the v2 release"}, "", failing=False))
    assert s.mission == "ship the v2 release"
    sink(ToolResult("mission_done", {}, "", failing=False))
    assert s.mission == ""


@check
def mission_self_suppresses_when_empty_and_renders_when_set():
    s = Slice(); s.reset("do a thing")
    out_empty = render_slice(s, "(no files)")
    assert "# MISSION" not in out_empty, "unset mission must render NOTHING (no bloat)"
    s.mission = "land the refactor cleanly"
    out_set = render_slice(s, "(no files)")
    assert "# MISSION" in out_set and "land the refactor cleanly" in out_set


@check
def seal_carries_mission_reset_wipes():
    s = Slice(); s.reset("x")
    slice_sink(s)(ToolResult("set_mission", {"text": "north star"}, "", failing=False))
    s.seal()
    assert s.mission == "north star", "seal() carries the mission across the turn boundary"
    s.reset("new task")
    assert s.mission == "", "reset() (new task) wipes the mission"


@check
def mission_roundtrips_through_taskstate():
    s = Slice(); s.reset("x")
    slice_sink(s)(ToolResult("set_mission", {"text": "the objective"}, "", failing=False))
    s2 = task_state_to_slice(slice_to_task_state(s, "tid", session_id="sid"))
    assert s2.mission == "the objective"


@check
def tool_handlers_validate_and_confirm():
    host = LocalToolHost("/tmp")
    assert not host.run("set_mission", {"text": "   "}).ok
    ok = host.run("set_mission", {"text": "win"})
    assert ok.ok and "MISSION set" in ok
    assert host.run("mission_done", {}).ok


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
