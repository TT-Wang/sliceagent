"""Truthful subagent admission: benign request corrections never become live children."""
from __future__ import annotations

from types import SimpleNamespace as NS
from unittest import mock

import pytest

from sliceagent.agents import BUILTIN_AGENTS
from sliceagent.events import SubagentProgress, ToolExecutionStarted, ToolRejected, ToolStarted
from sliceagent.execution import ToolStatus
from sliceagent.hooks import Hooks
from sliceagent.loop import run_tool_batch
from sliceagent.progress import TurnProgress
from sliceagent.registry import ToolText
from sliceagent.subagent import SubagentHost


class _Inner:
    def __init__(self):
        self.ran = []

    def schemas(self):
        return []

    def accesses(self, _name, _args):
        return []

    def run(self, name, args):
        self.ran.append((name, dict(args)))
        return "inner"

    def root(self):
        return "."


class _ConflictMemory:
    def roster_get(self, _name):
        return {"kind": "explorer", "jobs": 1}


class _MissingWork:
    def get(self, _identity):
        return None


def _call(name: str, identity: str, args: dict):
    return NS(name=name, id=identity, args=args)


def _settle(host: SubagentHost, name: str, args: dict):
    events = []
    progress = TurnProgress(await_commit=False)

    def dispatch(event):
        events.append(event)
        progress.reduce(event)

    _, rows = run_tool_batch([_call(name, "call-1", args)], host, dispatch, Hooks())
    return rows[0]["outcome"], events, progress.snapshot()


def _benign_cases():
    # 1–3: grant-shape corrections.
    yield SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None,
        spec=BUILTIN_AGENTS["general"], depth=0, max_depth=2,
    ), "spawn_agent", {"agent": "general", "task": "reduce", "grants": ["subagents/sub-1.md"]}
    yield SubagentHost(_Inner(), llm=None, retriever=None, memory=None), "spawn_agent", {
        "agent": "explorer", "task": "inspect", "grants": "subagents/sub-1.md",
    }
    yield SubagentHost(_Inner(), llm=None, retriever=None, memory=None), "spawn_agent", {
        "agent": "explorer", "task": "inspect", "grants": ["subagents/sub-1.md"] * 17,
    }

    # 4–5: child-only parent capability/private-memory requests.
    yield SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None,
        spec=BUILTIN_AGENTS["general"], depth=1, max_depth=2,
    ), "ask_user", {"question": "what next?"}
    yield SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None,
        spec=BUILTIN_AGENTS["general"], depth=1, max_depth=2,
    ), "read_file", {"path": "history/turn-1.md"}

    # 6–11: delegation request-shape corrections.
    yield SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None, depth=1, max_depth=1,
    ), "spawn_agent", {"agent": "explorer", "task": "inspect"}
    yield SubagentHost(_Inner(), llm=None, retriever=None, memory=None), "spawn_agent", {
        "agent": "explorer", "task": "  ",
    }
    yield SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None, core_mode=True,
    ), "spawn_agent", {"agent": "explorer", "task": "inspect", "name": "standing"}
    yield SubagentHost(_Inner(), llm=None, retriever=None, memory=None), "spawn_agent", {
        "agent": "explorer", "task": "inspect", "name": "sub-7",
    }
    yield SubagentHost(_Inner(), llm=None, retriever=None, memory=None), "spawn_agent", {
        "agent": "unknown", "task": "inspect",
    }
    yield SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None, core_mode=True,
    ), "spawn_subagent", {"task": "edit"}

    # 12: an already-known standing specialist kind conflict.
    yield SubagentHost(
        _Inner(), llm=None, retriever=None, memory=_ConflictMemory(),
    ), "spawn_agent", {"agent": "general", "task": "edit", "name": "auth"}


@pytest.mark.parametrize("host,name,args", tuple(_benign_cases()))
def test_benign_subagent_rejections_are_steered_before_start_without_a_matrix_row(host, name, args):
    outcome, events, snapshot = _settle(host, name, args)

    assert outcome.status is ToolStatus.STEERED
    assert not outcome.failing
    rejections = [event for event in events if isinstance(event, ToolRejected)]
    assert len(rejections) == 1 and rejections[0].kind == "steered"
    assert rejections[0].outcome is not None
    assert rejections[0].outcome.status is ToolStatus.STEERED
    assert not any(isinstance(event, (ToolExecutionStarted, ToolStarted, SubagentProgress)) for event in events)
    assert snapshot.subagents == ()
    assert getattr(host.inner, "ran", []) == []


def test_capability_escalation_and_missing_work_binding_stay_loud_but_do_not_launch():
    explorer = SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None,
        spec=BUILTIN_AGENTS["explorer"], depth=0, max_depth=2,
    )
    escalation, escalation_events, _ = _settle(
        explorer, "spawn_agent", {"agent": "general", "task": "write files"},
    )
    bound = SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None,
        active_work_provider=_MissingWork(),
    )
    missing, missing_events, _ = _settle(
        bound, "spawn_agent",
        {"agent": "explorer", "task": "inspect", "work_item_id": "missing"},
    )

    assert escalation.status is ToolStatus.FAILED
    assert missing.status is ToolStatus.FAILED
    assert not any(isinstance(event, (ToolExecutionStarted, ToolStarted))
                   for event in escalation_events + missing_events)


def test_runtime_child_failure_is_loud_and_occurs_after_started(monkeypatch):
    import sliceagent.subagent as module

    progress_events = []
    monkeypatch.setattr(
        module, "run_subagent",
        lambda *_args, **_kwargs: ToolText("Error: provider timed out", status=ToolStatus.FAILED),
    )
    host = SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None, notify=progress_events.append,
    )
    outcome, events, _ = _settle(
        host, "spawn_agent", {"agent": "explorer", "task": "inspect"},
    )

    assert outcome.status is ToolStatus.FAILED
    assert any(isinstance(event, ToolExecutionStarted) for event in events)
    assert any(isinstance(event, ToolStarted) for event in events)
    assert any(isinstance(event, SubagentProgress) and event.phase == "starting"
               for event in progress_events)


@pytest.mark.parametrize(
    "agent,expected",
    (("explorer", ToolStatus.FAILED), ("general", ToolStatus.INDETERMINATE)),
)
def test_unexpected_child_crash_is_typed_by_effect_risk(monkeypatch, agent, expected):
    import sliceagent.subagent as module

    def crash(*_args, **_kwargs):
        raise RuntimeError("child runtime broke")

    monkeypatch.setattr(module, "run_subagent", crash)
    host = SubagentHost(_Inner(), llm=None, retriever=None, memory=None)
    outcome, events, _ = _settle(
        host, "spawn_agent", {"agent": agent, "task": "inspect"},
    )

    assert outcome.status is expected
    assert outcome.failing
    assert any(isinstance(event, ToolExecutionStarted) for event in events)
    if agent == "general":
        assert "may have applied task-local effects" in outcome.text


def test_result_sink_binding_failure_is_a_typed_started_failure():
    class SinkOwner:
        def record(self, _artifact_id):
            return None

        def bind_artifact_ref_sink(self, **_kwargs):
            raise OSError("turn seal unavailable")

    owner = SinkOwner()
    host = SubagentHost(
        _Inner(), llm=None, retriever=None, memory=None,
        artifact_ref_sink=owner.record,
    )
    outcome, events, _ = _settle(
        host, "spawn_agent", {"agent": "explorer", "task": "inspect"},
    )

    assert outcome.status is ToolStatus.FAILED
    assert "could not bind subagent result" in outcome.text
    assert any(isinstance(event, ToolExecutionStarted) for event in events)


class _StandaloneMonkeyPatch:
    def __init__(self):
        self._patches = []

    def setattr(self, target, name, value):
        patch = mock.patch.object(target, name, value)
        patch.start()
        self._patches.append(patch)

    def undo(self):
        while self._patches:
            self._patches.pop().stop()


if __name__ == "__main__":
    checks = []
    for ordinal, (host, name, args) in enumerate(_benign_cases(), 1):
        checks.append((
            f"benign_subagent_rejection_{ordinal}",
            lambda host=host, name=name, args=args:
                test_benign_subagent_rejections_are_steered_before_start_without_a_matrix_row(
                    host, name, args,
                ),
        ))
    checks.append((
        "capability_escalation_and_missing_work_binding",
        test_capability_escalation_and_missing_work_binding_stay_loud_but_do_not_launch,
    ))

    def runtime_failure():
        patch = _StandaloneMonkeyPatch()
        try:
            test_runtime_child_failure_is_loud_and_occurs_after_started(patch)
        finally:
            patch.undo()

    checks.append(("runtime_child_failure", runtime_failure))
    for agent, expected in (("explorer", ToolStatus.FAILED), ("general", ToolStatus.INDETERMINATE)):
        def crash_case(agent=agent, expected=expected):
            patch = _StandaloneMonkeyPatch()
            try:
                test_unexpected_child_crash_is_typed_by_effect_risk(patch, agent, expected)
            finally:
                patch.undo()
        checks.append((f"unexpected_{agent}_crash", crash_case))
    checks.append(("result_sink_binding_failure", test_result_sink_binding_failure_is_a_typed_started_failure))

    passed = 0
    for name, check in checks:
        try:
            check()
            passed += 1
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(checks)} passed")
    raise SystemExit(0 if passed == len(checks) else 1)
