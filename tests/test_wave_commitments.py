"""Regression coverage for delegation without WorkGraph wave bookkeeping.

Run: PYTHONPATH=src python tests/test_wave_commitments.py
"""
from __future__ import annotations

from sliceagent.access import ReadAllAccess
from sliceagent.active_work import WorkDelta, WorkItem
from sliceagent.events import AssistantText, TurnEnd
from sliceagent.execution import ToolEffect
from sliceagent.hooks import Hooks
from sliceagent.interfaces import AssistantMessage, ToolCall
from sliceagent.loop import run_turn
from sliceagent.pfc import Slice, record_user, slice_sink
from sliceagent.prompt import SYSTEM_PROMPT, render_delegation_guidance
from sliceagent.registry import ToolText
from sliceagent.subagent import SubagentHost


def _state(*statuses: tuple[str, str]):
    state = Slice()
    state.reset("review the whole project")
    record_user(
        state,
        "review the whole project",
        source_event_id="event-1",
        logical_id="logical-1",
        workspace_epoch=0,
    )
    root = state.active_work.request_roots[-1]
    children = tuple(
        WorkItem(
            id=item_id,
            root_id=root.id,
            source_refs=root.source_refs,
            description=f"Review {item_id}",
            status=status,
            logical_id=root.logical_id,
        )
        for item_id, status in statuses
    )
    if children:
        state.active_work = state.active_work.apply_delta(WorkDelta(
            expected_revision=state.active_work.revision,
            creates=children,
        ))
    return state


class _Inner:
    def schemas(self):
        return []

    def run(self, _name, _args):
        return "inner"

    def accesses(self, _name, _args):
        return []


def _spawn_schema():
    host = SubagentHost(_Inner(), llm=None, retriever=None, memory=None)
    return next(
        row for row in host.schemas()
        if row["function"]["name"] == "spawn_agent"
    )


def test_delegation_guidance_describes_direct_reports_without_bookkeeping():
    schema = _spawn_schema()
    guidance = render_delegation_guidance([schema])

    properties = schema["function"]["parameters"]["properties"]
    assert "work_item_id" not in properties
    assert "returns one complete normalized report directly as this tool result" in guidance
    assert "parent owns the final synthesis" in guidance
    assert "use every returned report" in guidance
    assert "scheduler owns those physical waves" in guidance
    assert "ACTIVE WORK" not in guidance
    assert "work_item_id" not in guidance
    assert "FAN-IN" not in guidance.upper()
    assert "create the complete declared coverage frontier" not in guidance


def test_system_contract_keeps_active_work_optional_and_out_of_child_lifecycle():
    assert "ACTIVE WORK is optional" in SYSTEM_PROMPT
    assert "not a second user, a scheduler, a transcript, or a prerequisite for tool use" in SYSTEM_PROMPT
    assert "Do not create work items merely to launch children" in SYSTEM_PROMPT
    assert "mirror tool lifecycle" in SYSTEM_PROMPT


def test_open_frontier_does_not_rewrite_or_block_a_terminal_answer():
    state = _state(("wave-1-backend", "ready"), ("wave-2-ui", "open"))
    before = state.active_work.to_dict()

    class LLM:
        def __init__(self):
            self.calls = 0

        def complete(self, _messages, _schemas):
            self.calls += 1
            return AssistantMessage("the requested review", finish_reason="stop", usage={})

    llm = LLM()
    events = []

    def dispatch(event):
        events.append(event)
        slice_sink(state)(event)

    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "review the whole project"}],
        llm=llm,
        tools=_Inner(),
        dispatch=dispatch,
        hooks=Hooks(),
        max_steps=3,
    )

    finals = [event.content for event in events if isinstance(event, AssistantText) and event.final]
    assert result.stop_reason == "end_turn"
    assert llm.calls == 1
    assert finals == ["the requested review"]
    assert len([event for event in events if isinstance(event, TurnEnd)]) == 1
    assert state.active_work.to_dict() == before


def test_two_child_batch_synthesizes_from_direct_results_without_update_work():
    state = _state(("legacy-wave-marker", "open"))
    graph_before = state.active_work.to_dict()

    class LLM:
        def __init__(self):
            self.calls = 0
            self.synthesis_messages = []

        def complete(self, messages, _schemas):
            self.calls += 1
            if self.calls == 1:
                return AssistantMessage(
                    "launching children",
                    [
                        ToolCall("child-1", "spawn_agent", {
                            "agent": "explorer", "task": "backend",
                        }),
                        ToolCall("child-2", "spawn_agent", {
                            "agent": "explorer", "task": "ui",
                        }),
                    ],
                    usage={},
                    finish_reason="tool_calls",
                )
            self.synthesis_messages = [dict(message) for message in messages]
            return AssistantMessage("combined backend and UI review", finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return [_spawn_schema()]

        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, args):
            area = args["task"]
            report = (
                f"[child · explorer · succeeded]\n\n"
                f"BEGIN CHILD REPORT\nfull {area} evidence\nEND CHILD REPORT"
            )
            return ToolText(
                report,
                effects=(ToolEffect(f"{area}:outcome", "child_outcome", {
                    "status": "succeeded",
                    "operational_status": "succeeded",
                    "kind": "explorer",
                    "report_completion": "complete",
                    "report_sha256": area[0] * 64,
                    "report_bytes": len(report.encode("utf-8")),
                    "source_coverage_status": "source_complete",
                }),),
            )

    llm = LLM()
    events = []

    def dispatch(event):
        events.append(event)
        slice_sink(state)(event)

    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "review backend and UI"}],
        llm=llm,
        tools=Host(),
        dispatch=dispatch,
        hooks=Hooks(),
        max_steps=3,
    )

    assert result.stop_reason == "end_turn"
    assert llm.calls == 2
    trajectory = llm.synthesis_messages[1:]
    assert [message["role"] for message in trajectory] == ["assistant", "tool", "tool"]
    assert trajectory[0]["content"] == ""
    assert "full backend evidence" in trajectory[1]["content"]
    assert "full ui evidence" in trajectory[2]["content"]
    combined = "\n".join(str(message.get("content") or "") for message in trajectory)
    assert "HOST FAN-IN" not in combined
    assert "update_work" not in combined
    assert state.active_work.to_dict() == graph_before
    finals = [event.content for event in events if isinstance(event, AssistantText) and event.final]
    assert finals == ["combined backend and UI review"]


def main():
    tests = [
        value for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"wave commitment tests: {len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
