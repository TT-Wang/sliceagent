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
    ToolResult,
    ToolStarted,
    TurnCommitted,
    TurnEnd,
    TurnInterrupted,
    TurnPhaseChanged,
    TurnStarted,
)
from sliceagent.execution import ToolInvocation  # noqa: E402
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
    return machine.reduce(TurnStarted("fix the parser", task_title="Parser fix", task_id="task-1"))


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
