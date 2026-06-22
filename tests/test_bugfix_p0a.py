"""Regression tests for verified P0 bugs (review #1,#2,#3,#15). No model, no pytest.
Run: PYTHONPATH=src python tests/test_bugfix_p0a.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.regions import MAX_FINDINGS, record_action, record_note  # noqa: E402
from memagent.slice import Slice  # noqa: E402
from memagent.memory import _parse_task_md, _render_task_md  # noqa: E402
from memagent.interfaces import TaskState  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def record_action_uses_authoritative_failing_flag():  # #2
    s = Slice(); s.reset("x")
    # grep/log output that legitimately starts with "Error" but the tool SUCCEEDED (failing=False)
    record_action(s, "grep", {"pattern": "Error"}, "Error: connection refused (log line match)", failing=False)
    assert s.last_error == "", "a successful tool whose OUTPUT starts with 'Error' must NOT set last_error"
    sig = next(iter(s.action_log))
    assert s.action_log[sig]["failing"] is False, "must trust event.failing, not the prose prefix"
    # a genuine failure (failing=True) still records
    record_action(s, "run_command", {"command": "false"}, "Exit code 1", failing=True)
    assert s.last_error, "a real failure must set last_error"
    # back-compat: no flag → prose heuristic still works
    s2 = Slice(); s2.reset("y")
    record_action(s2, "x", {}, "Error: boom")
    assert s2.last_error, "fallback heuristic still flags when no flag is passed"


@check
def seal_bounds_findings_carry_and_clears_pre_defs():  # #1 + #15
    s = Slice(); s.reset("x")
    for i in range(MAX_FINDINGS + 12):
        record_note(s, f"fact number {i} established", source="observed")
    assert len(s.findings) > MAX_FINDINGS, "within a loop, findings are NOT cut (carry whole)"
    s.pre_defs = {"a.py": {"foo"}}
    s.seal()
    assert len(s.findings) == MAX_FINDINGS, "seal bounds the cross-loop carry to MAX_FINDINGS"
    assert all(k in set(s.findings) for k in s.finding_source), "finding_source pruned to live findings"
    assert s.pre_defs == {}, "seal clears transient pre_defs"


@check
def task_state_markdown_roundtrips_new_tiers():  # #3
    ts = TaskState(
        task_id="t1", session_id="s1", goal="do the thing",
        requirements=[{"text": "public API must stay stable", "done": False}],
        plan=[{"step": "write code", "status": "done"}, {"step": "test", "status": "in_progress"}],
        mission="ship the v2 release",
        world={"port": "8137", "map": "room A -> room B"},
    )
    md = _render_task_md(ts, created="2026-06-23", updated="2026-06-23")
    p = os.path.join(tempfile.mkdtemp(prefix="ts-"), "task.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(md)
    back = _parse_task_md(p)
    assert back.requirements == [{"text": "public API must stay stable", "done": False}], back.requirements
    assert [x["step"] for x in back.plan] == ["write code", "test"], back.plan
    assert back.plan[1]["status"] == "in_progress"
    assert back.mission == "ship the v2 release", back.mission
    assert back.world == {"port": "8137", "map": "room A -> room B"}, back.world


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
