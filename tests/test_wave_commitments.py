"""Regression coverage for staged delegation commitments and Active Work reconciliation.

Run: PYTHONPATH=src python tests/test_wave_commitments.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.active_work import WorkDelta, WorkItem  # noqa: E402
from sliceagent.events import AssistantText, TurnEnd  # noqa: E402
from sliceagent.execution import ToolInvocation, ToolStatus  # noqa: E402
from sliceagent.hooks import ActiveWorkContinuationHook, CompositeHooks  # noqa: E402
from sliceagent.interfaces import AssistantMessage, ToolCall  # noqa: E402
from sliceagent.loop import run_turn  # noqa: E402
from sliceagent.pfc import Slice, record_user, slice_sink  # noqa: E402
from sliceagent.prompt import render_delegation_guidance  # noqa: E402
from sliceagent.tools import LocalToolHost, TOOL_SCHEMAS  # noqa: E402


def _state(*statuses: tuple[str, str]):
    state = Slice()
    state.reset("review the whole project")
    record_user(
        state, "review the whole project", source_event_id="event-1",
        logical_id="logical-1", workspace_epoch=0,
    )
    root = state.active_work.request_roots[-1]
    children = tuple(
        WorkItem(
            id=item_id, root_id=root.id, source_refs=root.source_refs,
            description=f"Review {item_id}", status=status, logical_id=root.logical_id,
        )
        for item_id, status in statuses
    )
    if children:
        state.active_work = state.active_work.apply_delta(WorkDelta(
            expected_revision=state.active_work.revision, creates=children,
        ))
    return state


def _hook(state, logical_id="logical-1"):
    return ActiveWorkContinuationHook(lambda: (state.active_work, logical_id))


def test_open_frontier_gets_one_bounded_reconciliation_pass():
    state = _state(("wave-1-backend", "ready"), ("wave-2-ui", "open"))
    hook = _hook(state)
    first = hook.should_continue_after_stop("end_turn")
    assert first and first["continue"] and first["exclusive"]
    assert "wave-2-ui [open]" in first["feedback"]
    assert hook.should_continue_after_stop("end_turn") is None, \
        "an unchanged stale frontier must not spin forever"
    hook.reset_for_turn()
    assert hook.should_continue_after_stop("end_turn") is not None


def test_metadata_only_revision_does_not_rearm_an_unchanged_frontier():
    state = _state(("wave-2-ui", "open"))
    hook = _hook(state)
    assert hook.should_continue_after_stop("end_turn") is not None
    item = state.active_work.get("wave-2-ui")
    state.active_work = state.active_work.apply_delta(WorkDelta(
        expected_revision=state.active_work.revision,
        updates=(WorkItem(
            id=item.id, root_id=item.root_id, source_refs=item.source_refs,
            description="Review the complete UI layer", status=item.status,
            logical_id=item.logical_id,
        ),),
    ))
    assert hook.should_continue_after_stop("end_turn") is None


def test_ready_waiting_and_no_child_frontiers_can_finish():
    for state in (
        _state(),
        _state(("wave-1", "ready")),
        _state(("needs-user", "waiting_user")),
        _state(("discarded", "cancelled")),
    ):
        assert _hook(state).should_continue_after_stop("end_turn") is None


def test_pending_child_from_an_older_root_does_not_block_current_request():
    state = _state(("old-wave", "open"))
    record_user(
        state, "answer a new question", source_event_id="event-2",
        logical_id="logical-2", workspace_epoch=0,
    )
    assert _hook(state, "logical-2").should_continue_after_stop("end_turn") is None


def test_update_work_result_reprojects_the_remaining_frontier_within_the_turn():
    state = _state(("wave-1-backend", "in_progress"), ("wave-2-ui", "open"))
    host = LocalToolHost()
    host.bind_active_work(lambda: (state.active_work, "logical-1", 0))
    outcome = host.registry.invoke(ToolInvocation(
        "ready-wave-1", "update_work", {
            "expected_revision": state.active_work.revision,
            "changes": [{"id": "wave-1-backend", "status": "ready"}],
        }, 0,
    ))
    assert outcome.status is ToolStatus.SUCCEEDED
    assert "wave-2-ui [open]" in outcome.text
    assert "settled batch does not retire" in outcome.text


def test_two_wave_loop_defers_the_first_final_until_wave_two_is_reconciled():
    state = _state(("wave-1-backend", "ready"), ("wave-2-ui", "open"))
    host = LocalToolHost()
    host.bind_active_work(lambda: (state.active_work, "logical-1", 0))

    class LLM:
        def __init__(self):
            self.calls = 0
            self.seen = []

        def complete(self, messages, _schemas):
            self.calls += 1
            self.seen.append([dict(message) for message in messages])
            if self.calls == 1:
                return AssistantMessage("premature full review", finish_reason="stop", usage={})
            if self.calls == 2:
                return AssistantMessage(
                    "", [ToolCall(
                        "ready-ui", "update_work", {
                            "expected_revision": state.active_work.revision,
                            "changes": [{"id": "wave-2-ui", "status": "ready"}],
                        },
                    )], usage={}, finish_reason="tool_calls",
                )
            return AssistantMessage("complete after UI wave", finish_reason="stop", usage={})

    llm = LLM()
    events = []
    reducer = slice_sink(state)

    def dispatch(event):
        events.append(event)
        reducer(event)

    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "review the whole project"}],
        llm=llm, tools=host, dispatch=dispatch,
        hooks=CompositeHooks(_hook(state)), max_steps=5,
    )
    finals = [event.content for event in events if isinstance(event, AssistantText) and event.final]
    assert result.stop_reason == "end_turn" and llm.calls == 3
    assert finals == ["complete after UI wave"], finals
    assert len([event for event in events if isinstance(event, TurnEnd)]) == 1
    assert "wave-2-ui [open]" in llm.seen[1][-1]["content"]
    assert state.active_work.get("wave-2-ui").status == "ready"


def test_contract_requires_complete_frontier_in_one_logical_batch():
    spawn_schema = {"type": "function", "function": {
        "name": "spawn_agent",
        "parameters": {"type": "object", "properties": {
            "agent": {"type": "string", "enum": ["explorer"]},
            "task": {"type": "string"}, "work_item_id": {"type": "string"},
            "scope": {"type": "array"},
        }},
    }}
    guidance = render_delegation_guidance([spawn_schema])
    assert "COMPLETE declared coverage frontier" in guidance
    assert "submit every independent partition in one logical delegation batch" in guidance
    assert "scheduler owns those physical waves" in guidance
    update_work = next(
        row["function"] for row in TOOL_SCHEMAS
        if row["function"]["name"] == "update_work"
    )
    assert "complete promised frontier" in update_work["description"]
    assert "future coverage lives in prose" in update_work["description"]


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"wave commitment tests: {len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
