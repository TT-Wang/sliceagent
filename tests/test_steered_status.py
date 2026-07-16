"""Focused contract tests for benign, conclusive tool steering."""
from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace as NS

from rich.console import Console

from sliceagent.cli import cli_sink
from sliceagent.events import (ToolExecutionStarted, ToolRejected, ToolResult,
                               ToolStarted, TurnStarted)
from sliceagent.execution import ToolInvocation, ToolOutcome, ToolStatus
from sliceagent.hippocampus import _files_of, turn_markdown
from sliceagent.hooks import Hooks
from sliceagent.loop import run_tool_batch
from sliceagent.monitor import SliceMonitor
from sliceagent.persistence import JournalSnapshot
from sliceagent.pfc import Slice, slice_sink
from sliceagent.progress import TurnProgress
from sliceagent.receipts import (compact_receipt_projection, receipt_completion_label,
                                 receipt_has_adverse_lifecycle, receipt_summary_parts)
from sliceagent.registry import ToolText
from sliceagent.runtime_persistence import LocalTurnStore
from sliceagent.tui import _render_tool_result


class _SteeringHost:
    def accesses(self, _name, _args):
        return []

    def preflight_run(self, _name, _args):
        return None, ToolText("use a child work item instead", status=ToolStatus.STEERED)

    def run_preflighted(self, *_args):
        raise AssertionError("a steered preflight must not enter the handler")

    def run(self, *_args):
        raise AssertionError("a steered preflight must not use the legacy run path")


def _steered_result(name: str = "update_plan", args: dict | None = None) -> ToolResult:
    args = args or {}
    invocation = ToolInvocation("steer-1", name, args, 0)
    outcome = ToolOutcome(invocation, ToolStatus.STEERED, "choose a valid target")
    return ToolResult(
        name, args, outcome.text, outcome.failing,
        status=outcome.status.value, invocation_id=invocation.id, outcome=outcome,
    )


def test_steered_is_conclusive_non_success_and_non_failing():
    invocation = ToolInvocation("steer-1", "example", {}, 0)
    outcome = ToolOutcome(invocation, ToolStatus.STEERED, "try another route")

    assert ToolStatus.STEERED.conclusive
    assert not ToolStatus.STEERED.failing
    assert ToolStatus.STEERED is not ToolStatus.SUCCEEDED
    assert outcome.failing is False
    assert ToolText("try another route", status="steered").ok is False


def test_host_steer_is_rejected_and_settled_before_start_without_warning():
    store = LocalTurnStore(
        tempfile.mkdtemp(prefix="steer-workspace-"), "session-1",
        store_root=tempfile.mkdtemp(prefix="steer-store-"),
    )
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="delegate")
    events = []

    def sink(event):
        events.append(event)
        store.observe_event(event)
        if isinstance(event, ToolResult):
            store.observe_reduction(event)

    _, rows = run_tool_batch(
        [NS(name="spawn_agent", id="spawn-1", args={"task": "inspect"})],
        _SteeringHost(), sink, Hooks(), turn_id="turn-1",
    )
    store.seal(state={}, record={}, status="end_turn")
    artifact = store.coordinator.artifacts.get(active.artifact_id)
    receipt = artifact.to_dict()["structured_body"]["turn_receipt"]
    compact = compact_receipt_projection(receipt)

    assert rows[0]["status"] == "steered" and not rows[0]["failing"]
    assert any(isinstance(event, ToolRejected) and event.kind == "steered" for event in events)
    assert not any(isinstance(event, (ToolExecutionStarted, ToolStarted)) for event in events)
    assert receipt["counts"]["requested"] == 1
    assert receipt["counts"]["rejected_before_execution"] == 1
    assert receipt["counts"]["steered_before_execution"] == 1
    assert receipt["counts"]["execution_started"] == 0
    assert receipt["counts"]["settled"] == 1
    assert receipt["counts"]["steered"] == 1
    assert receipt["disposition"] == "completed" and receipt["warnings"] == []
    assert not receipt_has_adverse_lifecycle(compact)
    assert receipt_completion_label(compact, "end_turn") == "turn saved"
    summary = receipt_summary_parts(compact)
    assert any("1 steered" in part for part in summary)
    assert not any("rejected before start" in part for part in summary)


def test_steered_does_not_mutate_slice_or_progress_plan_and_gets_own_tally():
    args = {"steps": [{"step": "replace existing plan", "status": "done"}]}
    event = _steered_result(args=args)

    state = Slice()
    state.reset("keep current plan")
    state.plan = [{"step": "current plan", "status": "in_progress"}]
    slice_sink(state)(event)
    assert state.plan == [{"step": "current plan", "status": "in_progress"}]
    assert state.runtime.recent_calls[-1]["status"] == "steered"
    assert state.runtime.blocked_calls == 0 and state.action_log == {}

    progress = TurnProgress(await_commit=True)
    progress.reduce(TurnStarted("keep current plan", turn_id="turn-1"))
    progress.reduce(ToolStarted(event.name, event.args, event.outcome.invocation))
    snapshot = progress.reduce(event)
    assert snapshot.counts == {"steer": 1}
    assert snapshot.plan.total == 0


def test_steered_renders_dim_route_in_rich_plain_and_monitor_surfaces():
    event = _steered_result("read_file", {"path": "outside.txt"})

    rich_buffer = io.StringIO()
    console = Console(file=rich_buffer, force_terminal=False, color_system=None, width=100)
    console.print(_render_tool_result(event, 100))
    rich_text = rich_buffer.getvalue()
    assert "↷" in rich_text and "✗" not in rich_text

    plain_buffer = io.StringIO()
    with redirect_stdout(plain_buffer):
        cli_sink()(event)
    plain_text = plain_buffer.getvalue()
    assert "↷ read_file" in plain_text and "✗" not in plain_text

    monitor = SliceMonitor()
    from sliceagent.events import SliceBuilt
    monitor.sink(SliceBuilt("current slice"))
    monitor.sink(event)
    assert monitor.snapshot()["steps"][0]["tools"][0]["status"] == "steered"
    assert _files_of(_steered_result("str_replace", {"path": "a.py"})) == []
    history = turn_markdown("turn", [{
        "action": [{
            "name": "str_replace", "args": {"path": "a.py"},
            "status": "steered", "failing": False,
        }],
        "observation": ["old_string was not found"],
    }], "", {})
    assert "↷ [str_replace] a.py -> steered" in history and "applied" not in history


def test_steered_is_conclusive_in_legacy_crash_recovery():
    snapshot = JournalSnapshot({}, (
        {"type": "tool-invocation", "payload": {
            "invocation_id": "steer-1", "name": "read_file", "args": {},
        }},
        {"type": "tool-outcome", "payload": {
            "invocation_id": "steer-1", "outcome": {
                "status": "steered", "text": "choose another target",
            },
        }},
    ))
    assert snapshot.unresolved_invocations == ()


if __name__ == "__main__":
    checks = tuple(
        value for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    passed = 0
    for check in checks:
        try:
            check()
            passed += 1
            print(f"PASS {check.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(checks)} passed")
    raise SystemExit(0 if passed == len(checks) else 1)
