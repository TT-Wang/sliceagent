"""PLAN / TodoWrite tier (Kimi/Claude borrow): the model maintains an ordered step list with status via
update_plan (replace-all); it renders in the slice, is carried through seal(), wiped by reset(), and
round-trips through TaskState. Distinct from STANDING REQUIREMENTS (acceptance criteria). No model, no
pytest. Run: PYTHONPATH=src python tests/test_plan_tier.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import ToolResult  # noqa: E402
from memagent.regions import render_plan  # noqa: E402
from memagent.slice import Slice, slice_sink  # noqa: E402
from memagent.taskstate import slice_to_task_state, task_state_to_slice  # noqa: E402
from memagent.tools import LocalToolHost  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _steps(*pairs):
    return [{"step": s, "status": st} for s, st in pairs]


@check
def update_plan_folds_validates_and_bounds():
    s = Slice(); s.reset("build a feature")
    sink = slice_sink(s)
    sink(ToolResult("update_plan", {"steps": _steps(("write code", "in_progress"),
                                                    ("add tests", "pending"),
                                                    ("ship", "bogus_status"))},
                    "", failing=False))
    assert [p["step"] for p in s.plan] == ["write code", "add tests", "ship"]
    assert s.plan[0]["status"] == "in_progress"
    assert s.plan[2]["status"] == "pending", "unknown status normalizes to pending"


@check
def update_plan_is_replace_all():
    s = Slice(); s.reset("x"); sink = slice_sink(s)
    sink(ToolResult("update_plan", {"steps": _steps(("a", "in_progress"))}, "", failing=False))
    sink(ToolResult("update_plan", {"steps": _steps(("a", "done"), ("b", "in_progress"))}, "", failing=False))
    assert [(p["step"], p["status"]) for p in s.plan] == [("a", "done"), ("b", "in_progress")]


@check
def render_marks_status():
    out = render_plan(_steps(("done step", "done"), ("now", "in_progress"), ("later", "pending")))
    assert "1. [x] done step" in out and "2. [~] now" in out and "3. [ ] later" in out, out
    assert render_plan([]) == ""


@check
def seal_carries_plan_reset_wipes_it():
    s = Slice(); s.reset("x"); sink = slice_sink(s)
    sink(ToolResult("update_plan", {"steps": _steps(("a", "in_progress"))}, "", failing=False))
    s.seal()
    assert s.plan and s.plan[0]["step"] == "a", "seal() must CARRY the plan across the turn boundary"
    s.reset("new task")
    assert s.plan == [], "reset() (new task) wipes the plan"


@check
def plan_roundtrips_through_taskstate():
    s = Slice(); s.reset("x"); slice_sink(s)(
        ToolResult("update_plan", {"steps": _steps(("a", "done"), ("b", "pending"))}, "", failing=False))
    ts = slice_to_task_state(s, "tid", session_id="sid")
    s2 = task_state_to_slice(ts)
    assert [(p["step"], p["status"]) for p in s2.plan] == [("a", "done"), ("b", "pending")]


@check
def tool_handler_validates_and_confirms():
    host = LocalToolHost("/tmp")
    bad = host.run("update_plan", {"steps": []})
    assert not bad.ok and "non-empty" in bad
    ok = host.run("update_plan", {"steps": _steps(("a", "done"), ("b", "in_progress"))})
    assert ok.ok and "PLAN updated" in ok and "1 done" in ok and "1 in progress" in ok, ok


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
