"""Nested Slice ownership and region-owned lifecycle. No model/network."""
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import ToolResult  # noqa: E402
from sliceagent.pfc import Slice, slice_sink  # noqa: E402
from sliceagent.session import apply_turn_continuation  # noqa: E402
from sliceagent.slice_state import MAX_PROGRESS_SIGNALS, TurnRuntime  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def a_move_on_directive_clears_a_stale_open_user_report_but_a_follow_up_keeps_it():
    # The bug: a stale OPEN USER REPORT from a prior concern was answered first on an unrelated request.
    # A move-on cue ("anyways do a bug hunt") abandons it → cleared; a benign follow-up must KEEP a real
    # "it's broken" report so it survives the debugging thread.
    s = Slice(); s.reset("prior task"); s.open_report = "the 12-vs-11 explorer self-correction gap"
    apply_turn_continuation(s, "anyways do a bug hunting for this project")
    assert s.open_report == "", s.open_report

    s2 = Slice(); s2.reset("build task"); s2.open_report = "the build is red"
    apply_turn_continuation(s2, "also add a logging line")     # a follow-up, not a move-on → report survives
    assert s2.open_report == "the build is red", s2.open_report

    s3 = Slice(); s3.reset("t"); s3.open_report = "still broken"   # a move-on that IS a fresh report keeps one
    apply_turn_continuation(s3, "never mind — it still doesn't work")
    assert s3.open_report, "a fresh report in the same turn must not be cleared"


@check
def slice_has_exactly_six_authoritative_regions():
    assert [f.name for f in fields(Slice)] == [
        "intent", "task", "evidence", "work", "continuity", "runtime",
    ]
    assert not hasattr(__import__("sliceagent.pfc", fromlist=["x"]), "_SLICE_SEAL_POLICY"), \
        "lifecycle must live on regions, not in a parallel policy table"


@check
def legacy_aliases_are_live_views_not_copies():
    s = Slice(); s.reset("task")
    assert s.plan is s.task.plan and s.findings is s.evidence.findings
    assert s.active_files is s.work.active_files and s.conversation is s.continuity.conversation
    s.plan.append({"step": "one", "status": "pending"})
    s.active_files.append("a.py")
    s.goal = "renamed"
    assert s.task.plan[0]["step"] == "one"
    assert s.work.active_files == ["a.py"] and s.task.goal == "renamed"


@check
def reset_delegates_to_every_region():
    s = Slice(); s.reset("old")
    s.intent.add_exact("constraint")
    s.plan.append({"step": "x"}); s.world["k"] = "v"; s.task.add_progress("edit", "a.py")
    s.findings.append("fact"); s.last_error = "boom"; s.open_report = "broken"
    s.active_files.append("a.py"); s.active_skills.append({"name": "x", "body": "y"})
    s.conversation.append({"user": "u", "assistant": "a"})
    s.runtime.step = 9; s.turn_actions = 4; s.explore_mode = True
    s.reset("new")
    expected = Slice(); expected.reset("new")
    assert s == expected


@check
def seal_preserves_semantic_state_and_resets_runtime():
    s = Slice(); s.reset("task")
    s.intent.add_exact("keep API")
    s.plan = [{"step": "ship", "status": "in_progress"}]
    s.action_log = {"sig": {"count": 2}}
    s.world = {"phase": "verify"}
    s.task.add_progress("edit", "a.py")
    s.findings = [f"finding {i}" for i in range(12)]
    s.finding_source = {f: "observed" for f in s.findings}
    s.last_error = "still failing"; s.open_report = "user says broken"
    s.active_files = ["read.py", "edit.py"]
    s.edited_files = {"edit.py"}; s.edit_anchor = {"edit.py": "anchor", "read.py": "x"}
    s.active_skills = [{"name": "skill", "body": "body"}]
    s.pre_defs = {"edit.py": {"old"}, "read.py": {"other"}}
    s.ghosts = [{"kind": "file", "ref": "old.py"}]; s.hot = {"read.py": 2}
    s.protected_deps = {"dep.py"}; s.stale_deps = {"dep.py"}
    s.io["refault"] = 3; s.read_budget = 9
    s.conversation = [{"user": "u", "assistant": "a"}]; s.turns = 1
    s.runtime.step = 5; s.runtime.usage = {"prompt_tokens": 10}
    s.runtime.recent_calls = [{"name": "read_file"}]; s.runtime.blocked_calls = 2
    s.runtime.applied_effect_ids = {"effect-1"}
    s.since_edit = 7; s.turn_actions = 8; s.explore_mode = True

    s.seal()

    assert s.intent.entries and s.plan and not s.action_log and s.world
    assert s.task.progress_signals and s.task.progress_signals[0].detail == "a.py"
    assert s.findings == [f"finding {i}" for i in range(12)]
    assert set(s.finding_source) == set(s.findings)
    assert s.last_error == "still failing" and s.open_report == "user says broken"
    assert s.active_files == ["read.py", "edit.py"] and s.edited_files == {"edit.py"}
    assert s.edit_anchor == {"edit.py": "anchor", "read.py": "x"}
    assert s.pre_defs == {"edit.py": {"old"}, "read.py": {"other"}}
    assert s.active_skills and s.ghosts and s.hot == {"read.py": 2}
    assert s.protected_deps == set() and s.stale_deps == set()
    assert s.io["refault"] == 3 and s.read_budget == 9
    assert s.conversation and s.turns == 1
    assert s.runtime == TurnRuntime(), "detailed turn state must not cross the seal"


@check
def progress_signal_ring_is_bounded_coalesced_and_task_scoped():
    s = Slice(); s.reset("task")
    for i in range(MAX_PROGRESS_SIGNALS + 4):
        s.task.add_progress("evidence", f"fact {i}")
    assert len(s.progress_signals) == MAX_PROGRESS_SIGNALS
    latest = s.progress_signals[-1]
    s.task.add_progress(latest.kind, latest.detail)
    assert len(s.progress_signals) == MAX_PROGRESS_SIGNALS and s.progress_signals[-1].count == 2
    s.seal()
    assert s.progress_signals[-1].count == 2
    s.reset("new task")
    assert s.progress_signals == []


@check
def reducer_emits_semantic_progress_not_raw_output():
    s = Slice(); s.reset("task")
    slice_sink(s)(ToolResult(
        "edit_file", {"path": "app.py", "note": "root cause confirmed"},
        "very large raw output that must not become a progress record", False,
    ))
    records = s.task.progress_records()
    assert {r["kind"] for r in records} == {"edit", "evidence"}
    assert all("very large raw output" not in r["detail"] for r in records)
    s.seal()
    assert s.task.progress_records() == records


@check
def typed_effect_ids_make_reduction_idempotent():
    from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus

    s = Slice(); s.reset("task")
    invocation = ToolInvocation("call-1", "world_set", {"key": "n", "value": "1"}, 0)
    outcome = ToolOutcome(
        invocation, ToolStatus.SUCCEEDED, "ok",
        (ToolEffect("effect-1", "tool_outcome", {"name": "world_set"}),),
    )
    event = ToolResult("world_set", {"key": "n", "value": "1"}, "ok", False, outcome=outcome)
    sink = slice_sink(s)
    sink(event); first = (dict(s.world), dict(s.action_log), s.since_edit)
    sink(event)
    assert (dict(s.world), dict(s.action_log), s.since_edit) == first


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
