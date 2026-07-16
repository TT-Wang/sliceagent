"""Focused invariants for the narrow tool-preflight seam. No network or pytest."""
from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import ToolRejected, ToolResult, ToolStarted, TurnEnd, TurnInterrupted  # noqa: E402
from sliceagent.execution import ToolPurity, ToolStatus, reconciliation_targets  # noqa: E402
from sliceagent.hooks import CatastrophicSafeguardHook, Hooks, ToolPreflight  # noqa: E402
from sliceagent.loop import run_tool_batch, run_turn  # noqa: E402
from sliceagent.pfc import Slice, slice_sink  # noqa: E402
from sliceagent.registry import ToolEntry, ToolRegistry  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _call(name: str, call_id: str = "call", **args):
    return NS(name=name, id=call_id, args=args)


class _Host:
    def __init__(self):
        self.ran = []

    def schemas(self):
        return []

    def accesses(self, _name, _args):
        return []

    def run(self, name, _args):
        self.ran.append(name)
        return "ran"


class _ScriptLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, _messages, _schemas):
        self.calls += 1
        return self.responses.pop(0)


def _tool_response(call):
    return NS(content="", tool_calls=[call], finish_reason="tool_calls", usage={})


def _done_response():
    return NS(content="done", tool_calls=[], finish_reason="stop", usage={})


@check
def preflight_hook_exception_proceeds():
    class Broken(Hooks):
        def preflight_tool(self, _name, _args):
            raise RuntimeError("extension crashed")

    host = _Host()
    _, rows = run_tool_batch([_call("ordinary")], host, lambda _event: None, Broken())
    assert host.ran == ["ordinary"]
    assert rows[0]["status"] == ToolStatus.SUCCEEDED.value
    assert "rejection_kind" not in rows[0]


@check
def normal_shell_work_runs_without_a_permission_or_reconciliation_gate():
    host = _Host()
    _, rows = run_tool_batch(
        [_call("run_command", command="find . -type f")],
        host,
        lambda _event: None,
        CatastrophicSafeguardHook(),
    )
    assert host.ran == ["run_command"]
    assert rows[0]["status"] == ToolStatus.SUCCEEDED.value
    assert "policy" not in rows[0]["output"].casefold()


@check
def interrupted_readonly_explorer_has_no_workspace_effect_target():
    assert reconciliation_targets("spawn_explore", {"task": "inspect"}) == ()
    assert reconciliation_targets("spawn_agent", {"agent": "explorer", "task": "inspect"}) == ()
    assert reconciliation_targets("spawn_agent", {"agent": "reviewer", "task": "edit"}) == (
        "workspace:*", "opaque:spawn_agent",
    )


@check
def lifecycle_stop_is_a_typed_neutral_cancellation():
    class LifecycleStop(Hooks):
        def preflight_tool(self, _name, _args):
            return ToolPreflight(True, "lifecycle moved on", kind="lifecycle")

    events = []
    host = _Host()
    blocked, rows = run_tool_batch([_call("ordinary")], host, events.append, LifecycleStop())
    assert blocked == 0 and host.ran == []
    assert rows[0]["status"] == ToolStatus.CANCELLED.value
    assert rows[0]["output"] == "Not run: lifecycle moved on"
    assert rows[0]["rejection_kind"] == "lifecycle"
    assert rows[0]["rejection_reason"] == "lifecycle moved on"
    assert rows[0]["rejected_before_execution"] is False
    assert rows[0]["not_run_before_execution"] is True
    assert not any(isinstance(event, ToolStarted) for event in events)
    assert sum(isinstance(event, ToolRejected) for event in events) == 1


@check
def plain_cli_renders_cancellation_neutrally():
    from sliceagent.cli import cli_sink

    output = StringIO()
    with redirect_stdout(output):
        cli_sink()(ToolResult(
            "ordinary", {}, "Not run: lifecycle moved on", True, status="cancelled",
        ))
    rendered = output.getvalue()
    assert "↷ ordinary" in rendered and "✗" not in rendered, rendered


@check
def cancelled_not_run_result_does_not_mutate_task_evidence_or_clear_a_blocker():
    state = Slice(); state.reset("work"); state.last_error = "earlier edit failed"
    slice_sink(state)(ToolResult(
        "str_replace", {"path": "app.py", "note": "fixed everything"},
        "Not run: lifecycle moved on", True, status="cancelled",
    ))
    assert state.last_error == "earlier edit failed"
    assert not state.findings and not state.action_log and not state.progress_signals
    assert not state.active_files and not state.edited_files


@check
def lifecycle_stop_does_not_park_the_turn():
    class LifecycleStop(Hooks):
        def preflight_tool(self, _name, _args):
            return ToolPreflight(True, "lifecycle moved on", kind="lifecycle")

    llm = _ScriptLLM(_tool_response(_call("ordinary")), _done_response())
    events = []
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "go"}],
        llm=llm, tools=_Host(), dispatch=events.append, hooks=LifecycleStop(), max_steps=3,
    )
    assert result.stop_reason == "end_turn" and llm.calls == 2
    assert not any(isinstance(event, TurnInterrupted) for event in events)
    assert sum(isinstance(event, TurnEnd) for event in events) == 1


@check
def usage_observer_exception_does_not_invent_a_budget_stop():
    class BrokenUsageObserver(Hooks):
        def record_step_usage(self, _usage):
            raise RuntimeError("observer crashed")

    llm = _ScriptLLM(_done_response())
    events = []
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "go"}],
        llm=llm, tools=_Host(), dispatch=events.append, hooks=BrokenUsageObserver(), max_steps=2,
    )
    assert result.stop_reason == "end_turn" and llm.calls == 1
    assert not any(isinstance(event, TurnInterrupted) for event in events)


@check
def catastrophic_rejection_parks_once_with_exact_safety_message():
    safety = "Safety stop: refused a potentially catastrophic command (test fixture)."

    class Catastrophic(Hooks):
        def preflight_tool(self, _name, _args):
            return ToolPreflight(True, safety, kind="catastrophic")

    llm = _ScriptLLM(_tool_response(_call("run_command", command="shutdown -h now")))
    events = []
    host = _Host()
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "go"}],
        llm=llm, tools=host, dispatch=events.append, hooks=Catastrophic(), max_steps=3,
    )
    interruptions = [event for event in events if isinstance(event, TurnInterrupted)]
    results = [event for event in events if isinstance(event, ToolResult)]
    assert result.stop_reason == "blocked" and llm.calls == 1 and host.ran == []
    assert len(interruptions) == 1 and interruptions[0].message == safety
    assert len(results) == 1 and results[0].status == ToolStatus.CANCELLED.value
    assert results[0].output == safety


@check
def workspace_handoff_tail_is_cancelled_without_parking():
    registry = ToolRegistry()
    ran = []
    schema = lambda name: {"type": "function", "function": {"name": name, "parameters": {}}}
    registry.register(ToolEntry(
        "change_workspace", schema("change_workspace"),
        lambda _args: ran.append("change_workspace") or "switched",
        purity=ToolPurity.EFFECTFUL, capabilities=frozenset({"workspace_handoff"}),
    ))
    registry.register(ToolEntry(
        "ordinary", schema("ordinary"), lambda _args: ran.append("ordinary") or "ran",
        purity=ToolPurity.EFFECTFUL,
    ))

    class Host:
        def __init__(self):
            self.registry = registry

        def schemas(self):
            return registry.schemas()

        def accesses(self, name, args):
            return registry.accesses(name, args)

        def run(self, name, args):
            return registry.run(name, args)

    batch = NS(
        content="", finish_reason="tool_calls", usage={},
        tool_calls=[_call("change_workspace", "switch"), _call("ordinary", "tail")],
    )
    llm = _ScriptLLM(batch, _done_response())
    events = []
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "go"}],
        llm=llm, tools=Host(), dispatch=events.append, hooks=Hooks(), max_steps=3,
    )
    results = [event for event in events if isinstance(event, ToolResult)]
    assert result.stop_reason == "end_turn" and llm.calls == 2
    assert ran == ["change_workspace"]
    assert [event.status for event in results] == ["succeeded", "cancelled"]
    assert results[1].output.startswith("Not run:")
    assert not any(isinstance(event, TurnInterrupted) for event in events)


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {error!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
