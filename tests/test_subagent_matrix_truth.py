"""Regression coverage for the authoritative subagent progress matrix."""
from __future__ import annotations

from rich.console import Console

from sliceagent.events import (ModelCallPrepared, StepBegin, StepEnd, SubagentProgress, ToolResult,
                               ToolStarted, TurnInterrupted, TurnStarted)
from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
from sliceagent.progress import ProgressPhase, TurnProgress
from sliceagent.subagent import _nested_sink
from sliceagent.tui import _agent_matrix_plain_lines, _render_agent_batch
from sliceagent.tui_projection import project_agent_result


class Clock:
    def __init__(self, now: float = 100.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def started_machine(*, turn_id: str = "turn-matrix") -> tuple[TurnProgress, Clock]:
    clock = Clock()
    machine = TurnProgress(clock=clock, await_commit=True)
    machine.reduce(TurnStarted("delegate the review", turn_id=turn_id))
    machine.reduce(StepBegin(1))
    return machine, clock


def spawn(machine: TurnProgress, index: int = 1) -> ToolInvocation:
    invocation = ToolInvocation(
        f"spawn-{index}", "spawn_agent",
        {"agent": "explorer", "name": f"area-{index}", "task": f"inspect area {index}"},
        index - 1,
    )
    machine.reduce(ToolStarted(invocation.name, dict(invocation.args), invocation))
    return invocation


def update(machine: TurnProgress, invocation: ToolInvocation, phase: str, sequence: int, **fields):
    return machine.subagent_activity(SubagentProgress(
        agent_id=f"child-{invocation.provider_index + 1}",
        parent_turn_id="turn-matrix",
        launch_ordinal=invocation.provider_index + 1,
        kind="explorer",
        name=f"area-{invocation.provider_index + 1}",
        phase=phase,
        sequence=sequence,
        invocation_id=invocation.id,
        request_ordinal=invocation.provider_index + 1,
        objective=f"inspect area {invocation.provider_index + 1}",
        **fields,
    ))


def rendered(machine: TurnProgress, *, now: float | None = None, width: int = 120) -> str:
    return "\n".join(
        line for _style, line in _agent_matrix_plain_lines(machine.snapshot(), width, now=now)
    )


def failed_result(invocation: ToolInvocation, *, cause: str = "provider_timeout") -> ToolResult:
    artifact_id = f"child-{invocation.provider_index + 1}"
    effect = ToolEffect(f"{artifact_id}:artifact", "child_artifact", {
        "artifact_id": artifact_id,
        "kind": "explorer",
        "launch_ordinal": invocation.provider_index + 1,
        "status": "failed",
        "stop_reason": "error",
        "stop_cause": cause,
        "partial": True,
    })
    outcome = ToolOutcome(invocation, ToolStatus.FAILED, "child failed", (effect,))
    return ToolResult(
        invocation.name, dict(invocation.args), outcome.text, True,
        status="failed", invocation_id=invocation.id, outcome=outcome,
    )


def succeeded_result(invocation: ToolInvocation, *, evidence_status: str = "not_assessed",
                     evidence_account: dict | None = None) -> ToolResult:
    artifact_id = f"child-{invocation.provider_index + 1}"
    effect = ToolEffect(f"{artifact_id}:artifact", "child_artifact", {
        "artifact_id": artifact_id,
        "kind": "explorer",
        "launch_ordinal": invocation.provider_index + 1,
        "status": "ok",
        "stop_reason": "end_turn",
        "stop_cause": "complete",
        "explorer_evidence_status": evidence_status,
        **({"explorer_evidence": evidence_account} if evidence_account is not None else {}),
    })
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "report", (effect,))
    return ToolResult(
        invocation.name, dict(invocation.args), outcome.text, False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    )


def test_matrix_exposes_typed_child_phases_without_false_model_wait():
    machine, clock = started_machine()
    invocation = spawn(machine)
    initial = rendered(machine, now=clock.now)
    assert "starting" in initial
    assert "awaiting model" not in initial and "model wait" not in initial

    clock.advance(1)
    awaiting = update(machine, invocation, "awaiting_model", 1, attempt=1, max_attempts=3)
    row = awaiting.subagents[0]
    assert (row.phase, row.attempt, row.max_attempts) == ("awaiting_model", 1, 3)
    assert "awaiting model" in rendered(machine, now=clock.now)

    clock.advance(1)
    active = update(machine, invocation, "model_active", 2, attempt=1, max_attempts=3)
    assert active.subagents[0].phase == "model_active"
    assert "model responding" in rendered(machine, now=clock.now)
    assert "awaiting model" not in rendered(machine, now=clock.now)

    clock.advance(1)
    reasoning = update(machine, invocation, "reasoning", 3, attempt=1, max_attempts=3)
    assert reasoning.subagents[0].phase == "reasoning"
    assert "reasoning" in rendered(machine, now=clock.now)
    assert "awaiting model" not in rendered(machine, now=clock.now)

    clock.advance(1)
    writing = update(machine, invocation, "writing", 4, attempt=1, max_attempts=3)
    assert writing.subagents[0].phase == "writing"
    assert "writing report" in rendered(machine, now=clock.now)

    clock.advance(1)
    using_tool = update(
        machine, invocation, "running_tool", 5, tool_name="read_file", tool_count=7,
    )
    row = using_tool.subagents[0]
    assert row.phase == "running_tool" and row.tool_name == "read_file" and row.tool_count == 7
    assert "running read_file" in rendered(machine, now=clock.now)

    clock.advance(1)
    retrying = update(
        machine, invocation, "retry_wait", 6,
        attempt=2, max_attempts=3, retry_delay_s=1.25,
    )
    row = retrying.subagents[0]
    assert row.phase == "retry_wait" and row.retry_delay_s == 1.25
    text = rendered(machine, now=clock.now)
    assert "retry wait" in text and "attempt 2/3" in text and "1.2s" in text


def test_child_interruption_moves_to_nonterminal_settling_until_outer_result():
    machine, clock = started_machine()
    invocation = spawn(machine)
    sink = _nested_sink(
        machine.subagent_activity, 1,
        agent_id="child-1", parent_turn_id="turn-matrix", launch_ordinal=1,
        kind="explorer", name="area-1", invocation_id=invocation.id,
        request_ordinal=1, objective="inspect area 1",
    )

    sink(ModelCallPrepared(step=1, attempt=1, messages=[]))
    assert machine.snapshot().subagents[0].phase == "awaiting_model"
    clock.advance(1)
    sink(TurnInterrupted("max_steps", "final attempt exhausted"))

    row = machine.snapshot().subagents[0]
    assert row.phase == "settling"
    assert row.detail == "sealing partial outcome"
    assert row.finished_at is None, "child interruption is not the outer spawn's terminal result"
    text = rendered(machine, now=clock.now)
    assert "sealing" in text and "sealing partial outcome" in text
    assert "awaiting model" not in text and "responding" not in text

    settled = machine.reduce(succeeded_result(invocation))
    assert settled.subagents[0].phase == "report_ready", \
        "the outer typed ToolResult remains the sole terminal authority"


def test_transport_activity_distinguishes_capacity_first_byte_and_live_stream():
    machine, _clock = started_machine()
    invocation = spawn(machine)
    sink = _nested_sink(
        machine.subagent_activity, 1,
        agent_id="child-1", parent_turn_id="turn-matrix", launch_ordinal=1,
        kind="explorer", name="area-1", invocation_id=invocation.id,
        request_ordinal=1, objective="inspect area 1",
    )

    sink(ModelCallPrepared(step=1, attempt=1, messages=[]))
    sink.on_activity("provider_queue", {"queue_ms": 2400, "active": 4, "capacity": 4})
    row = machine.snapshot().subagents[0]
    assert row.phase == "awaiting_model" and row.detail == "provider queue 2.4s · 4/4 active"

    sink.on_activity("provider_admitted", {"queue_ms": 2500, "remaining_ms": 57500})
    row = machine.snapshot().subagents[0]
    assert row.phase == "awaiting_model" and row.detail == "provider admitted · queue 2.5s"

    sink.on_activity("stream_heartbeat", {
        "state": "awaiting_first_byte", "elapsed_ms": 7600, "idle_ms": 5100, "chunks": 0,
    })
    row = machine.snapshot().subagents[0]
    assert row.phase == "awaiting_model" and row.detail == "awaiting first byte · 7.6s"

    sink.on_activity("first_byte", {"ttfb_ms": 7900, "elapsed_ms": 10400})
    row = machine.snapshot().subagents[0]
    assert row.phase == "model_active" and row.detail == "first byte · TTFT 7.9s"

    sink.on_activity("stream_heartbeat", {
        "state": "receiving", "elapsed_ms": 15400, "idle_ms": 900, "chunks": 17,
    })
    row = machine.snapshot().subagents[0]
    assert row.phase == "model_active" and row.detail == "stream live · 17 chunks · idle 0.9s"


def test_late_model_callbacks_cannot_rewind_visible_writing_progress():
    machine, clock = started_machine()
    invocation = spawn(machine)

    update(machine, invocation, "awaiting_model", 1, attempt=1)
    update(machine, invocation, "model_active", 2, attempt=1)
    update(machine, invocation, "reasoning", 3, attempt=1)
    writing = update(machine, invocation, "writing", 4, attempt=1)
    assert writing.subagents[0].phase == "writing"

    # Provider activity and delta callbacks can race at the SDK boundary. Their
    # sequence records delivery order, not permission to semantically move backward.
    clock.advance(1)
    late_reasoning = update(machine, invocation, "reasoning", 5, attempt=1)
    assert late_reasoning.subagents[0].phase == "writing"
    late_first_byte = update(machine, invocation, "model_active", 6, attempt=1)
    assert late_first_byte.subagents[0].phase == "writing"
    assert "writing report" in rendered(machine, now=clock.now)
    assert "reasoning" not in rendered(machine, now=clock.now)

    # The guard is scoped to consecutive model phases: a real next pass first
    # publishes `starting`, after which awaiting-model is truthful again.
    restarted = update(machine, invocation, "starting", 7)
    assert restarted.subagents[0].phase == "starting"
    awaiting_next_pass = update(machine, invocation, "awaiting_model", 8, attempt=2)
    assert awaiting_next_pass.subagents[0].phase == "awaiting_model"


def test_terminal_wrapper_update_preserves_cumulative_child_tool_count():
    machine, _clock = started_machine()
    invocation = spawn(machine)
    running = update(
        machine, invocation, "running_tool", 1, tool_name="grep", tool_count=7,
    )
    assert running.subagents[0].tool_count == 7

    terminal = update(
        machine, invocation, "report_ready", 2, terminal_reason="complete", tool_count=0,
    )
    row = terminal.subagents[0]
    assert row.phase == "report_ready" and row.tool_count == 7
    assert "7" in rendered(machine), "the ready row must retain its real cumulative tool count"


def test_matrix_shows_last_activity_separately_from_total_elapsed():
    machine, clock = started_machine()
    invocation = spawn(machine)
    clock.advance(5)
    update(machine, invocation, "reasoning", 1, attempt=1)

    text = rendered(machine, now=clock.now + 7, width=120)
    assert "last" in text and "time" in text
    assert "00:07" in text, text
    assert "00:12" in text, text


def test_typed_timeout_settles_partial_row_and_late_callback_cannot_resurrect_it():
    machine, clock = started_machine()
    invocation = spawn(machine)
    update(machine, invocation, "awaiting_model", 1, attempt=1, max_attempts=3)

    settled = machine.reduce(failed_result(invocation))
    row = settled.subagents[0]
    assert row.phase == "timed_out"
    assert row.terminal_reason == "provider timeout" and row.partial
    assert settled.phase is ProgressPhase.INTEGRATING
    text = rendered(machine, now=clock.now)
    assert "timed out" in text and "provider timeout" in text and "partial report" in text

    late = update(machine, invocation, "reasoning", 99, attempt=2)
    assert late.subagents[0].phase == "timed_out"

    view = project_agent_result(failed_result(invocation), duration_s=5)
    assert view.timed_out and view.partial and view.terminal_reason == "provider_timeout"
    console = Console(record=True, width=100, force_terminal=False, color_system=None)
    console.print(_render_agent_batch([view], 100))
    durable = console.export_text()
    assert "1 timed out" in durable and "partial report" in durable


def test_reverse_fanout_settlement_transitions_to_results_and_step_end_clears_matrix():
    machine, _clock = started_machine()
    first, second = spawn(machine, 1), spawn(machine, 2)
    update(machine, first, "running_tool", 1, tool_name="grep", tool_count=2)
    update(machine, second, "reasoning", 1, attempt=1)

    one_left = machine.reduce(succeeded_result(second))
    assert one_left.phase is ProgressPhase.DELEGATING
    assert {row.phase for row in one_left.subagents} == {"running_tool", "report_ready"}

    all_settled = machine.reduce(succeeded_result(first))
    assert all_settled.phase is ProgressPhase.INTEGRATING
    assert {row.phase for row in all_settled.subagents} == {"report_ready"}
    assert "2 sealed" in rendered(machine)

    cleared = machine.reduce(StepEnd(1, {}, "tool_use"))
    assert cleared.subagents == () and cleared.phase is ProgressPhase.INTEGRATING


def test_typed_evidence_status_and_counts_get_a_separate_live_column():
    machine, _clock = started_machine()
    navigation, empty = spawn(machine), spawn(machine, 2)
    navigation_result = succeeded_result(
        navigation,
        evidence_status="navigation_only",
        evidence_account={"navigation_success_count": 3, "content_success_count": 0},
    )
    machine.reduce(navigation_result)
    settled = machine.reduce(succeeded_result(empty, evidence_status="none"))
    by_id = {row.invocation_id: row for row in settled.subagents}
    assert by_id[navigation.id].phase == "report_ready"
    assert by_id[navigation.id].evidence_status == "navigation_only"
    assert dict(by_id[navigation.id].evidence_account)["navigation_success_count"] == 3
    assert by_id[empty.id].evidence_status == "none"

    lines = _agent_matrix_plain_lines(settled, 120, now=100.0)
    rendered_text = "\n".join(line for _style, line in lines)
    row_styles = [style for style, line in lines if line.strip()[:1].isdigit()]
    assert "2 sealed" in rendered_text and "evidence" in rendered_text
    assert "nav 3" in rendered_text and "no evidence" in rendered_text
    assert "source complete" not in rendered_text
    assert row_styles == ["warn", "warn"], \
        "navigation-only/absent evidence are amber, not red execution failures"
