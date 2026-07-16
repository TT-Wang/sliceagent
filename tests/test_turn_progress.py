"""Truthful, renderer-independent turn progress state.

These checks pin the lifecycle semantics shared by every terminal renderer.  They intentionally import no
Rich or prompt_toolkit code: presentation may change, but a late SliceBuilt event must never rewind a turn,
physical retries must remain visible, tool waves must settle by invocation, and a model-loop TurnEnd must
not masquerade as a durably committed completion.

No model, no pytest. Run: PYTHONPATH=src python tests/test_turn_progress.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import (  # noqa: E402
    ApiRetry,
    AssistantText,
    ModelCallPrepared,
    SliceBuilt,
    StepBegin,
    StepEnd,
    SubagentProgress,
    ToolResult,
    ToolStarted,
    TurnCommitted,
    TurnEnd,
    TurnInterrupted,
    TurnPhaseChanged,
    TurnStarted,
)
from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus  # noqa: E402
from sliceagent.progress import ProgressPhase, TurnProgress  # noqa: E402
from sliceagent.regions import MAX_PLAN_CHARS, MAX_PLAN_ITEMS  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


class _Clock:
    """Deterministic monotonic clock; reducer tests must never sleep."""

    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _machine(start: float = 100.0):
    clock = _Clock(start)
    return TurnProgress(clock=clock, await_commit=True), clock


def _start(machine: TurnProgress):
    return machine.reduce(TurnStarted(
        "fix the parser", task_title="Parser fix", task_id="task-1", turn_id="turn-1",
    ))


def _status(snapshot) -> str:
    return snapshot.status_text().casefold()


@check
def production_event_order_never_rewinds_the_turn():
    """run_turn emits StepBegin before its once-per-turn SliceBuilt observation."""
    machine, clock = _machine()
    started = _start(machine)
    assert started.phase is ProgressPhase.PREPARING
    assert started.started_at == 100.0

    clock.advance(2.0)
    first_pass = machine.reduce(StepBegin(1))
    assert first_pass.model_pass == 1

    clock.advance(3.0)
    built = machine.reduce(SliceBuilt("CURRENT REQUEST: fix the parser"))
    assert built.model_pass == 1, "late SliceBuilt must not reset the already-started model pass"
    assert built.started_at == started.started_at, "late SliceBuilt must not restart the whole-turn clock"

    prepared = machine.reduce(ModelCallPrepared(
        step=1, attempt=1, messages=[{"role": "user", "content": "fix the parser"}],
    ))
    assert prepared.phase is ProgressPhase.THINKING
    assert prepared.model_pass == 1 and prepared.provider_attempt == 1
    assert not prepared.turn_complete and not prepared.committed
    assert "thinking" in _status(prepared) and "waiting for model" in _status(prepared), _status(prepared)
    assert "parser fix" not in _status(prepared), "status text must not pin the task's first prompt/title"


@check
def tool_wave_settles_by_invocation_and_then_integrates():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    machine.reduce(SliceBuilt("seed"))
    machine.reduce(ModelCallPrepared(1, 1, []))

    first = ToolInvocation("read-a", "read_file", {"path": "a.py"}, 0)
    second = ToolInvocation("read-b", "read_file", {"path": "b.py"}, 1)
    one_active = machine.reduce(ToolStarted(first.name, dict(first.args), first))
    two_active = machine.reduce(ToolStarted(second.name, dict(second.args), second))
    assert one_active.phase is ProgressPhase.INSPECTING
    assert len(one_active.active_tools) == 1 and len(two_active.active_tools) == 2

    # Pure reads may complete in a different order than they were announced.  The reducer must settle the
    # matching invocation, not blindly pop whichever tool happens to be last.
    one_left = machine.reduce(ToolResult(
        second.name, dict(second.args), "b", False, invocation_id=second.id,
    ))
    assert len(one_left.active_tools) == 1
    settled = machine.reduce(ToolResult(
        first.name, dict(first.args), "a", False, invocation_id=first.id,
    ))
    assert settled.active_tools == ()
    assert settled.counts.get("read") == 2, settled.counts

    integrating = machine.reduce(StepEnd(1, {}, "tool_use"))
    assert integrating.phase is ProgressPhase.INTEGRATING
    assert integrating.counts.get("read") == 2

    machine.reduce(StepBegin(2))
    next_pass = machine.reduce(ModelCallPrepared(2, 1, []))
    assert next_pass.phase is ProgressPhase.THINKING
    assert next_pass.model_pass == 2 and next_pass.provider_attempt == 1
    assert next_pass.counts.get("read") == 2, "cumulative work must survive model-pass churn"
    # Snapshots returned earlier must not be live views into later reducer mutation.
    assert len(one_active.active_tools) == 1 and one_active.counts.get("read", 0) == 0


@check
def retry_remains_live_and_prepared_call_owns_the_attempt_number():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    machine.reduce(SliceBuilt("seed"))
    machine.reduce(ModelCallPrepared(1, 1, []))

    retrying = machine.reduce(ApiRetry(
        attempt=1, error="temporary provider timeout", delay_s=1.5, max_attempts=3,
    ))
    assert retrying.phase is ProgressPhase.RETRYING
    assert retrying.provider_attempt == 1, "ApiRetry identifies the failed attempt, not fabricated future I/O"
    assert not retrying.turn_complete
    retry_status = _status(retrying)
    assert "retry" in retry_status and "1.5" in retry_status, retry_status

    prepared = machine.reduce(ModelCallPrepared(1, 2, []))
    assert prepared.phase is ProgressPhase.THINKING
    assert prepared.provider_attempt == 2, "ModelCallPrepared is authoritative for physical attempt count"
    assert "thinking" in _status(prepared) and "attempt 2" in _status(prepared), _status(prepared)


@check
def terminal_open_and_wait_use_truthful_command_progress():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))

    opened = machine.reduce(ToolStarted(
        "terminal_open", {"session": "server", "command": "python -m app"},
    ))
    assert opened.phase is ProgressPhase.RUNNING
    assert opened.active_tools[0].bucket == "cmd"
    assert "python -m app" in _status(opened), _status(opened)
    after_open = machine.reduce(ToolResult(
        "terminal_open", {"session": "server", "command": "python -m app"},
        "started", False,
    ))
    assert after_open.counts.get("cmd") == 1, after_open.counts

    waiting = machine.reduce(ToolStarted(
        "terminal_wait", {"session": "server", "until": "ready"},
    ))
    assert waiting.phase is ProgressPhase.WAITING
    assert waiting.active_tools[0].bucket == "cmd"
    assert "waiting for server" in _status(waiting), _status(waiting)
    settled = machine.reduce(ToolResult(
        "terminal_wait", {"session": "server", "until": "ready"}, "ready", False,
    ))
    assert settled.counts.get("cmd") == 2, settled.counts


@check
def plan_updates_replace_all_and_failed_updates_do_not_mutate():
    machine, _ = _machine()
    carried = machine.reduce(TurnStarted("continue", task_title="Parser fix", plan=[
        {"step": "inspect existing behavior", "status": "done"},
        {"step": "fix the parser", "status": "in_progress"},
    ]))
    assert carried.plan.total == 2 and carried.plan.current == "fix the parser", \
        "a persisted task plan must be visible before the model calls update_plan again"
    first = machine.reduce(ToolResult("update_plan", {"steps": [
        {"step": "inspect the parser", "status": "done"},
        {"step": "fix error handling", "status": "in_progress"},
        {"step": "run focused tests", "status": "pending"},
    ]}, "PLAN updated", False))
    assert first.plan.total == 3 and first.plan.done == 1
    assert first.plan.current == "fix error handling" and first.plan.current_index == 2
    assert "2/3" in _status(first) and "fix error handling" in _status(first), _status(first)

    replacement = machine.reduce(ToolResult("update_plan", {"steps": [
        {"step": "implement fix", "status": "done"},
        {"step": "verify regression", "status": "in_progress"},
    ]}, "PLAN updated", False))
    assert replacement.plan.total == 2 and replacement.plan.done == 1
    assert replacement.plan.current == "verify regression" and replacement.plan.current_index == 2

    failed = machine.reduce(ToolResult("update_plan", {"steps": [
        {"step": "corrupt replacement", "status": "in_progress"},
    ]}, "Error: invalid plan", True))
    assert failed.plan == replacement.plan, "a failed update_plan call must not alter displayed task progress"

    malformed = machine.reduce(ToolResult("update_plan", {"steps": [
        {"step": "   ", "status": "done"},
        {"step": "  normalized   whitespace  ", "status": "INVALID"},
        *({"step": "x" * (MAX_PLAN_CHARS + 20), "status": "pending"}
          for _ in range(MAX_PLAN_ITEMS + 5)),
    ]}, "PLAN updated", False))
    assert malformed.plan.total == MAX_PLAN_ITEMS - 1, malformed.plan
    assert malformed.plan.current == "normalized whitespace" and malformed.plan.current_index == 1


@check
def assistant_and_loop_end_are_not_durable_completion():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    machine.reduce(SliceBuilt("seed"))
    machine.reduce(ModelCallPrepared(1, 1, []))

    writing = machine.reduce(AssistantText("I found the parser defect.", final=False))
    assert writing.phase is ProgressPhase.WRITING
    assert not writing.turn_complete and not writing.committed

    response_ready = machine.reduce(AssistantText("The parser is fixed."))
    assert response_ready.phase is ProgressPhase.FINALIZING
    assert not response_ready.turn_complete and not response_ready.committed

    verifying = machine.reduce(TurnPhaseChanged("verifying", "running focused tests"))
    assert verifying.phase is ProgressPhase.VERIFYING
    assert verifying.detail == "running focused tests"
    assert "running focused tests" in _status(verifying), _status(verifying)

    machine.reduce(StepEnd(1, {}, "end_turn"))
    loop_finished = machine.reduce(TurnEnd("end_turn", 1, {}))
    assert loop_finished.phase is ProgressPhase.FINALIZING
    assert not loop_finished.turn_complete and not loop_finished.committed, \
        "TurnEnd closes the model loop; it does not prove the host checkpoint was saved"

    saving = machine.reduce(TurnPhaseChanged("saving", "saving checkpoint"))
    assert saving.phase is ProgressPhase.SAVING and not saving.turn_complete
    committed = machine.reduce(TurnCommitted(
        True, "end_turn", artifact_id="artifact-1", detail="checkpoint saved",
    ))
    assert committed.phase is ProgressPhase.COMPLETE
    assert committed.turn_complete and committed.committed
    assert "saved" in _status(committed) or "complete" in _status(committed), _status(committed)


@check
def sealed_indeterminate_receipt_overrides_a_preseal_end_turn():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    machine.reduce(TurnEnd("end_turn", 1, {}))
    event = TurnCommitted(
        True, "end_turn", artifact_id="artifact-uncertain", detail="checkpoint saved",
        receipt={
            "turn_status": "indeterminate", "disposition": "indeterminate",
            "counts": {"requested": 1, "execution_started": 1, "indeterminate": 1},
            "agents": {},
        },
    )
    assert event.stop_reason == "indeterminate"
    assert event.detail == "indeterminate state saved"
    committed = machine.reduce(event)
    assert committed.phase is ProgressPhase.INTERRUPTED
    assert committed.stop_reason == "indeterminate"
    assert committed.committed and not committed.turn_complete


@check
def interruption_is_terminal_without_claiming_success():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    edit = ToolInvocation("edit-a", "str_replace", {"path": "a.py"}, 0)
    active = machine.reduce(ToolStarted(edit.name, dict(edit.args), edit))
    assert active.phase is ProgressPhase.EDITING and len(active.active_tools) == 1

    interrupted = machine.reduce(TurnInterrupted(
        "indeterminate", "the edit was interrupted after it started",
    ))
    assert interrupted.phase is ProgressPhase.INTERRUPTED
    assert interrupted.active_tools == ()
    assert not interrupted.turn_complete and not interrupted.committed
    assert interrupted.detail == "the edit was interrupted after it started"
    assert "interrupt" in _status(interrupted), _status(interrupted)


@check
def parallel_subagents_are_identity_safe_and_terminal_state_is_monotonic():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    first = ToolInvocation("spawn-1", "spawn_agent", {"agent": "explorer", "task": "audit storage"}, 0)
    second = ToolInvocation("spawn-2", "spawn_agent", {"agent": "explorer", "task": "audit TUI"}, 1)
    machine.reduce(ToolStarted(first.name, dict(first.args), first))
    started = machine.reduce(ToolStarted(second.name, dict(second.args), second))
    assert started.phase is ProgressPhase.DELEGATING and "2 agents running" in started.detail

    machine.subagent_activity(SubagentProgress(
        "child-1", "turn-1", 1, "explorer", "storage", 1,
        "running", "read_file store.py", 3, 3,
    ))
    active = machine.subagent_activity(SubagentProgress(
        "child-2", "turn-1", 2, "explorer", "ui", 1,
        "running", "grep renderer", 2, 2,
    ))
    assert len(active.subagents) == 2 and "2 agents running" in active.detail
    nested = SubagentProgress(
        "child-2-1", "turn-1", 1, "explorer", "widgets", 2,
        "running", "read widget.py", 1, 1, parent_agent_id="child-2",
    )
    active = machine.subagent_activity(nested)
    nested_state = next(item for item in active.subagents if item.agent_id == "child-2-1")
    assert nested_state.parent_agent_id == "child-2" and nested_state.depth == 2
    assert "ui → widgets" in active.detail, active.detail
    leaf = machine.subagent_activity(SubagentProgress(
        "child-2-1-1", "turn-1", 1, "explorer", "leaf", 3,
        "failed", "provider timeout", 1, 2, parent_agent_id="child-2-1",
    ))
    assert "ui → widgets → leaf · provider timeout" in leaf.detail, leaf.detail

    settled = machine.subagent_activity(SubagentProgress(
        "child-1", "turn-1", 1, "explorer", "storage", 1,
        "report_ready", "report ready", 3, 999,
    ))
    assert "1 sealed" in settled.detail
    # A late callback cannot resurrect a terminal child, even with a later sequence.
    late = machine.subagent_activity(SubagentProgress(
        "child-1", "turn-1", 1, "explorer", "storage", 1,
        "running", "late stale read", 4, 1000,
    ))
    child_one = next(item for item in late.subagents if item.agent_id == "child-1")
    assert child_one.phase == "report_ready" and "late stale" not in late.detail

    first_effect = ToolEffect("child-1:effect", "child_artifact", {"artifact_id": "child-1"})
    first_outcome = ToolOutcome(first, ToolStatus.SUCCEEDED, "report", (first_effect,))
    after_parent_result = machine.reduce(ToolResult(
        "spawn_agent", dict(first.args), "report", False, status="succeeded",
        invocation_id=first.id, outcome=first_outcome,
    ))
    assert "1 sealed" in after_parent_result.detail and "agents running" in after_parent_result.detail
    retired = machine.subagent_activity(SubagentProgress(
        "child-1", "turn-1", 1, "explorer", "storage", 1,
        "report_ready", "late terminal callback", 3, 2_000,
    ))
    retained = next(item for item in retired.subagents if item.agent_id == "child-1")
    assert retained.phase == "report_ready" and retained.detail == "sealed", \
        "settled calls must tombstone callbacks while retaining the terminal matrix row through StepEnd"
    integrated = machine.reduce(StepEnd(1, {}, "tool_use"))
    assert integrated.subagents == (), "the durable settled group replaces the transient matrix at StepEnd"

    # An old physical turn cannot contaminate a new active projection.
    stale = machine.subagent_activity(SubagentProgress(
        "old-child", "turn-0", 1, "explorer", "old", 1,
        "running", "wrong turn", 1, 1,
    ))
    assert all(item.agent_id != "old-child" for item in stale.subagents)


@check
def unmatched_typed_result_cannot_retire_a_concurrent_same_name_sibling():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    real = ToolInvocation("spawn-real", "spawn_agent", {"agent": "explorer", "task": "real"}, 0)
    machine.reduce(ToolStarted(real.name, dict(real.args), real))
    machine.subagent_activity(SubagentProgress(
        "child-real", "turn-1", 1, "explorer", "real", 1,
        "running", "read real.py", 1, 1,
    ))
    rejected = machine.reduce(ToolResult(
        "spawn_agent", {"agent": "explorer", "task": "rejected"}, "not started", True,
        status="failed", invocation_id="spawn-never-started",
    ))
    assert [tool.invocation_id for tool in rejected.active_tools] == ["spawn-real"], rejected
    assert [child.agent_id for child in rejected.subagents] == ["child-real"], rejected


@check
def exact_spawn_identity_binds_reverse_callbacks_without_duplicate_rows():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    calls = [
        ToolInvocation("same-a", "spawn_agent", {"agent": "explorer", "task": "same"}, 0),
        ToolInvocation("same-b", "spawn_agent", {"agent": "explorer", "task": "same"}, 1),
    ]
    for call in calls:
        machine.reduce(ToolStarted(call.name, dict(call.args), call))
    for index in (2, 1):
        call = calls[index - 1]
        machine.subagent_activity(SubagentProgress(
            f"child-{index}", "turn-1", index, "explorer", "duplicate", 1,
            "running", f"work {index}", index, index,
            invocation_id=call.id, request_ordinal=index, objective="same",
        ))
    rows = machine.snapshot().subagents
    assert len(rows) == 2 and {row.invocation_id for row in rows} == {"same-a", "same-b"}, rows
    assert {row.request_ordinal for row in rows} == {1, 2}
    assert not any(row.agent_id.startswith("invocation:") for row in rows), rows


@check
def equal_sequence_conflict_cannot_rewrite_a_child_row():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    call = ToolInvocation("spawn-one", "spawn_agent", {"agent": "explorer", "task": "ui"}, 0)
    machine.reduce(ToolStarted(call.name, dict(call.args), call))
    machine.subagent_activity(SubagentProgress(
        "child-one", "turn-1", 1, "explorer", "ui", 1,
        "running", "read tui.py", 1, 4, invocation_id=call.id, request_ordinal=1,
    ))
    conflict = machine.subagent_activity(SubagentProgress(
        "child-one", "turn-1", 1, "explorer", "ui", 1,
        "failed", "fabricated conflict", 1, 4, invocation_id=call.id, request_ordinal=1,
    ))
    row = next(item for item in conflict.subagents if item.agent_id == "child-one")
    assert row.phase == "running" and row.detail == "read tui.py", row


@check
def terminal_child_hint_is_monotonic_even_at_a_higher_sequence():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    call = ToolInvocation("spawn-terminal", "spawn_agent", {
        "agent": "explorer", "task": "ui",
    }, 0)
    machine.reduce(ToolStarted(call.name, dict(call.args), call))
    machine.subagent_activity(SubagentProgress(
        "child-terminal", "turn-1", 1, "explorer", "ui", 1,
        "failed", "provider timeout", 2, 4,
        invocation_id=call.id, request_ordinal=1,
    ))
    later = machine.subagent_activity(SubagentProgress(
        "child-terminal", "turn-1", 1, "explorer", "ui", 1,
        "report_ready", "late success claim", 3, 99,
        invocation_id=call.id, request_ordinal=1,
    ))
    row = next(item for item in later.subagents if item.agent_id == "child-terminal")
    assert row.phase == "failed" and row.detail == "provider timeout", row


@check
def nested_invocation_identity_is_scoped_to_its_parent_branch():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    calls = [
        ToolInvocation("root-a", "spawn_agent", {"agent": "explorer", "task": "A"}, 0),
        ToolInvocation("root-b", "spawn_agent", {"agent": "explorer", "task": "B"}, 1),
    ]
    for call in calls:
        machine.reduce(ToolStarted(call.name, dict(call.args), call))
    for index, call in enumerate(calls, 1):
        machine.subagent_activity(SubagentProgress(
            f"parent-{index}", "turn-1", index, "explorer", f"parent-{index}", 1,
            "running", "parent work", 1, 1,
            invocation_id=call.id, request_ordinal=index,
        ))
        machine.subagent_activity(SubagentProgress(
            f"nested-{index}", "turn-1", 1, "explorer", f"nested-{index}", 2,
            "running", "nested work", 1, 1,
            parent_agent_id=f"parent-{index}", invocation_id="call_1_0", request_ordinal=1,
        ))
    rows = {item.agent_id: item for item in machine.snapshot().subagents}
    assert {"nested-1", "nested-2"}.issubset(rows), rows
    assert rows["nested-1"].invocation_id == rows["nested-2"].invocation_id == "call_1_0"
    assert rows["nested-1"].parent_agent_id != rows["nested-2"].parent_agent_id


@check
def malformed_nested_cycle_settles_once_instead_of_freezing_the_ui():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    call = ToolInvocation("cycle", "spawn_agent", {"agent": "explorer", "task": "cycle"}, 0)
    machine.reduce(ToolStarted(call.name, dict(call.args), call))
    machine.subagent_activity(SubagentProgress(
        "A", "turn-1", 1, "explorer", "A", 2, "running", "A", 0, 1,
        parent_agent_id="B", invocation_id=call.id, request_ordinal=1,
    ))
    machine.subagent_activity(SubagentProgress(
        "B", "turn-1", 1, "explorer", "B", 2, "running", "B", 0, 1,
        parent_agent_id="A",
    ))
    effect = ToolEffect("cycle:effect", "child_artifact", {"artifact_id": "A"})
    outcome = ToolOutcome(call, ToolStatus.SUCCEEDED, "report", (effect,))
    settled = machine.reduce(ToolResult(
        call.name, dict(call.args), "report", False, status="succeeded",
        invocation_id=call.id, outcome=outcome,
    ))
    phases = {item.agent_id: item.phase for item in settled.subagents}
    assert phases == {"A": "report_ready", "B": "indeterminate"}, phases


@check
def authoritative_result_rekeys_a_missing_callback_and_settles_its_nested_tree():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    call = ToolInvocation("call-A", "spawn_agent", {"agent": "explorer", "task": "A"}, 0)
    machine.reduce(ToolStarted(call.name, dict(call.args), call))
    machine.subagent_activity(SubagentProgress(
        "nested", "turn-1", 1, "explorer", "nested", 2,
        "running", "nested work", 1, 1, parent_agent_id="child-A",
    ))
    effect = ToolEffect("A:effect", "child_artifact", {"artifact_id": "child-A"})
    outcome = ToolOutcome(call, ToolStatus.SUCCEEDED, "report", (effect,))
    settled = machine.reduce(ToolResult(
        call.name, dict(call.args), "report", False, status="succeeded",
        invocation_id=call.id, outcome=outcome,
    ))
    rows = {item.agent_id: item for item in settled.subagents}
    assert set(rows) == {"child-A", "nested"}, rows
    assert rows["child-A"].phase == "report_ready" and rows["nested"].phase == "indeterminate"


@check
def result_before_first_callback_keeps_terminal_alias_and_rejects_late_activity():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    call = ToolInvocation("call-A", "spawn_agent", {"agent": "explorer", "task": "A"}, 0)
    machine.reduce(ToolStarted(call.name, dict(call.args), call))
    failed = machine.reduce(ToolResult(
        call.name, dict(call.args), "provider timeout", True, status="failed", invocation_id=call.id,
    ))
    assert len(failed.subagents) == 1 and failed.subagents[0].phase == "failed", failed
    late = machine.subagent_activity(SubagentProgress(
        "child-A", "turn-1", 1, "explorer", "A", 1,
        "running", "late callback", 1, 1, invocation_id=call.id, request_ordinal=1,
    ))
    assert len(late.subagents) == 1 and late.subagents[0].phase == "failed", late


@check
def invocation_identity_wins_over_a_contradictory_artifact_or_callback_alias():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    calls = {
        name: ToolInvocation(name, "spawn_agent", {"agent": "explorer", "task": name}, index)
        for index, name in enumerate(("A", "B"))
    }
    for index, (name, call) in enumerate(calls.items(), 1):
        machine.reduce(ToolStarted(call.name, dict(call.args), call))
        machine.subagent_activity(SubagentProgress(
            f"child-{name}", "turn-1", index, "explorer", name, 1,
            "running", f"work {name}", 1, 1, invocation_id=name, request_ordinal=index,
        ))
    # A malformed artifact payload points A's result at B. Exact physical invocation A remains authoritative.
    effect = ToolEffect("bad", "child_artifact", {"artifact_id": "child-B"})
    outcome = ToolOutcome(calls["A"], ToolStatus.SUCCEEDED, "report", (effect,))
    settled = machine.reduce(ToolResult(
        "spawn_agent", dict(calls["A"].args), "report", False, status="succeeded",
        invocation_id="A", outcome=outcome,
    ))
    rows = {item.agent_id: item for item in settled.subagents}
    assert rows["child-A"].phase == "report_ready" and rows["child-B"].phase == "running", rows
    # A later callback cannot hijack child-A by presenting sibling invocation B.
    hijack = machine.subagent_activity(SubagentProgress(
        "child-A", "turn-1", 1, "explorer", "A", 1,
        "running", "hijack", 2, 2, invocation_id="B", request_ordinal=1,
    ))
    rows = {item.agent_id: item for item in hijack.subagents}
    assert rows["child-A"].invocation_id == "A" and rows["child-A"].phase == "report_ready"
    assert rows["child-B"].invocation_id == "B"


@check
def report_ready_children_are_not_also_counted_as_running():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    for index in (1, 2):
        invocation = ToolInvocation(
            f"spawn-{index}", "spawn_agent", {"agent": "explorer", "task": f"area {index}"}, index - 1,
        )
        machine.reduce(ToolStarted(invocation.name, dict(invocation.args), invocation))
        machine.subagent_activity(SubagentProgress(
            f"child-{index}", "turn-1", index, "explorer", f"area-{index}", 1,
            "running", "reading", 1, 1,
        ))
    one_ready = machine.subagent_activity(SubagentProgress(
        "child-1", "turn-1", 1, "explorer", "area-1", 1,
        "report_ready", "report ready", 1, 2,
    ))
    assert "1 agent running" in one_ready.detail and "1 sealed" in one_ready.detail
    assert "2 agents running" not in one_ready.detail, one_ready.detail
    all_ready = machine.subagent_activity(SubagentProgress(
        "child-2", "turn-1", 2, "explorer", "area-2", 1,
        "report_ready", "report ready", 1, 2,
    ))
    assert "agents running" not in all_ready.detail and "2 sealed" in all_ready.detail


@check
def a_new_model_step_clears_prior_unlinked_agent_outcomes():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    old = ToolInvocation("old", "spawn_agent", {"agent": "explorer", "task": "old"}, 0)
    machine.reduce(ToolStarted(old.name, dict(old.args), old))
    machine.subagent_activity(SubagentProgress(
        "turn-1:agent:1", "turn-1", 1, "explorer", "old", 1,
        "failed", "provider timeout", 1, 2,
    ))
    machine.reduce(ToolResult(
        old.name, dict(old.args), "failed", True, status="failed", invocation_id=old.id,
    ))
    machine.reduce(StepBegin(2))
    new = ToolInvocation("new", "spawn_agent", {"agent": "explorer", "task": "new"}, 0)
    fresh = machine.reduce(ToolStarted(new.name, dict(new.args), new))
    assert "failed" not in fresh.detail and "new" in fresh.detail, fresh.detail


@check
def unknown_explicit_spawn_status_is_live_uncertainty_not_success():
    machine, _ = _machine()
    _start(machine)
    machine.reduce(StepBegin(1))
    first = ToolInvocation("first", "spawn_agent", {"agent": "explorer", "task": "first"}, 0)
    second = ToolInvocation("second", "spawn_agent", {"agent": "explorer", "task": "second"}, 1)
    machine.reduce(ToolStarted(first.name, dict(first.args), first))
    machine.reduce(ToolStarted(second.name, dict(second.args), second))
    uncertain = machine.reduce(ToolResult(
        first.name, dict(first.args), "provider extension", False,
        status="timed_out", invocation_id=first.id,
    ))
    assert "1 state unknown" in uncertain.detail and "sealed" not in uncertain.detail, uncertain.detail


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
