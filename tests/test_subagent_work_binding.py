from __future__ import annotations

from sliceagent.active_work import WorkDelta, WorkGraph, WorkItem, attach_child_artifacts
from sliceagent.context_compiler import render_active_work
from sliceagent.events import ToolResult
from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
from sliceagent.pfc import Slice, record_user, slice_sink
from sliceagent.subagent import SubagentHost
from sliceagent.subagent_contract import SubagentBrief


class Inner:
    def schemas(self):
        return []

    def run(self, _name, _args):
        return "inner"

    def accesses(self, _name, _args):
        return []


def graph_state():
    state = Slice(); state.reset("delegate")
    record_user(state, "review the parser", source_event_id="event", logical_id="logical")
    root = state.active_work.request_roots[0]
    child = WorkItem(
        id="parser-review", root_id=root.id, source_refs=root.source_refs,
        description="Review parser", status="in_progress",
    )
    state.active_work = state.active_work.apply(WorkDelta(expected_revision=1, creates=(child,)))
    return state


def test_spawn_schema_and_brief_carry_an_immutable_work_binding():
    state = graph_state()
    host = SubagentHost(
        Inner(), llm=None, retriever=None, memory=None,
        active_work_provider=lambda: state.active_work,
    )
    schema = next(row for row in host.schemas() if row["function"]["name"] == "spawn_agent")
    assert "work_item_id" in schema["function"]["parameters"]["properties"]
    brief = host._brief("Review parser", {"work_item_id": "parser-review"}, frozenset())
    assert brief.work_item_id == "parser-review"
    assert SubagentBrief.from_dict(brief.to_dict()) == brief
    assert "PARENT ACTIVE WORK ITEM" in brief.render()


def test_active_work_bound_parent_requires_existing_child_binding_in_schema():
    state = graph_state()
    host = SubagentHost(
        Inner(), llm=None, retriever=None, memory=None,
        active_work_provider=lambda: state.active_work,
    )
    schema = next(row for row in host.schemas() if row["function"]["name"] == "spawn_agent")
    required = schema["function"]["parameters"]["required"]
    description = schema["function"]["parameters"]["properties"]["work_item_id"]["description"]
    assert "work_item_id" in required
    assert "existing nonterminal child" in description and "never invent" in description


def test_active_work_bound_parent_steers_a_missing_runtime_binding_before_launch():
    state = graph_state()
    host = SubagentHost(
        Inner(), llm=None, retriever=None, memory=None,
        active_work_provider=lambda: (state.active_work, "logical"),
    )
    result = host.run("spawn_agent", {"agent": "explorer", "task": "inspect"})
    assert result.status is ToolStatus.STEERED
    assert "requires an existing ACTIVE WORK child ID" in str(result)


def test_spawn_rejects_a_nonexistent_or_request_root_binding_before_launch():
    state = graph_state()
    host = SubagentHost(
        Inner(), llm=None, retriever=None, memory=None,
        active_work_provider=lambda: state.active_work,
    )
    missing = host.run("spawn_agent", {
        "agent": "explorer", "task": "inspect", "work_item_id": "missing",
    })
    assert "no active child work item" in str(missing)
    root_id = state.active_work.request_roots[0].id
    request_root = host.run("spawn_agent", {
        "agent": "explorer", "task": "inspect", "work_item_id": root_id,
    })
    assert "no active child work item" in str(request_root)


def test_spawn_rejects_a_child_owned_by_an_older_request_root():
    state = graph_state()
    record_user(
        state, "review the current parser", source_event_id="event-2",
        logical_id="logical-2",
    )
    host = SubagentHost(
        Inner(), llm=None, retriever=None, memory=None,
        active_work_provider=lambda: (state.active_work, "logical-2"),
    )
    result = host.run("spawn_agent", {
        "agent": "explorer", "task": "inspect", "work_item_id": "parser-review",
    })
    assert result.status is ToolStatus.FAILED
    assert "does not belong to the current request" in str(result)


def test_typed_child_effect_preserves_work_item_binding_in_parent_runtime():
    state = graph_state()
    invocation = ToolInvocation(
        "spawn-1", "spawn_agent",
        {"agent": "explorer", "task": "inspect", "work_item_id": "parser-review"}, 0,
    )
    outcome = ToolOutcome(
        invocation, ToolStatus.SUCCEEDED, "child returned",
        (ToolEffect("child-1", "child_artifact", {
            "artifact_id": "subagent-artifact-1", "work_item_id": "parser-review",
            "source_coverage_status": "source_partial",
            "required_ref_count": 2, "consumed_refs": ["subagents/sub-1.md"],
            "cited_refs": ["subagents/sub-1.md"], "covered_refs": ["subagents/sub-1.md"],
            "source_gaps": ["one report was not read"],
        }),),
    )
    slice_sink(state)(ToolResult(
        name="spawn_agent", args=dict(invocation.args), output=outcome.text,
        failing=False, status="succeeded", invocation_id=invocation.id, outcome=outcome,
    ))
    call = state.runtime.recent_calls[-1]
    assert call["child_artifact_id"] == "subagent-artifact-1"
    assert call["child_work_item_id"] == "parser-review"
    assert call["child_source_coverage_status"] == "source_partial"
    assert call["child_covered_ref_count"] == 1
    assert "one report was not read" not in str(call)


def test_child_binding_promotion_is_shared_idempotent_and_does_not_claim_completion():
    state = graph_state()
    calls = [{
        "child_artifact_id": "subagent-artifact-1",
        "child_work_item_id": "parser-review",
        "child_source_coverage_status": "source_partial",
    }]
    first = attach_child_artifacts(state.active_work, calls, workspace_epoch=4)
    second = attach_child_artifacts(first, calls, workspace_epoch=4)
    item = second.get("parser-review")
    assert item.status == "in_progress"
    assert [(ref.kind, ref.ref) for ref in item.evidence_refs] == [
        ("child_artifact", "subagent-artifact-1"),
    ]
    assert item.evidence_refs[0].qualifier == "source_partial"
    assert [(ref.kind, ref.ref, ref.workspace_epoch) for ref in item.resource_refs] == [
        ("subagent", "subagent-artifact-1", 4),
    ]
    assert second.digest == first.digest


def test_source_status_and_locator_survive_active_work_roundtrip_into_the_next_turn():
    state = graph_state()
    attached = attach_child_artifacts(state.active_work, ({
        "child_artifact_id": "subagent-artifact-1",
        "child_work_item_id": "parser-review",
        "child_source_coverage_status": "source_complete",
    },), workspace_epoch=4)

    restored = WorkGraph.from_records(attached.to_records())
    next_turn = attach_child_artifacts(restored, (), workspace_epoch=4)
    evidence = next_turn.get("parser-review").evidence_refs
    assert len(evidence) == 1
    assert (evidence[0].kind, evidence[0].ref, evidence[0].qualifier) == (
        "child_artifact", "subagent-artifact-1", "source_complete",
    )
    seed = render_active_work(
        next_turn, sources={"event": "review the parser"}, current_logical_id="next-logical",
    )
    assert "child_artifact:subagent-artifact-1 [source complete]" in seed
