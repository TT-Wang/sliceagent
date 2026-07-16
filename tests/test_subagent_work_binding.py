from __future__ import annotations

"""Regression coverage for the retired delegation/Active Work coupling.

Active Work remains a user-commitment model.  A child launch and its terminal
outcome are ordinary tool execution and must not require or mutate that model.
Legacy artifact fields remain readable, but are not part of the live spawn API.
"""

from sliceagent.active_work import WorkDelta, WorkItem
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
    state = Slice()
    state.reset("delegate")
    record_user(state, "review the parser", source_event_id="event", logical_id="logical")
    root = state.active_work.request_roots[0]
    child = WorkItem(
        id="parser-review",
        root_id=root.id,
        source_refs=root.source_refs,
        description="Review parser",
        status="in_progress",
    )
    state.active_work = state.active_work.apply(
        WorkDelta(expected_revision=state.active_work.revision, creates=(child,))
    )
    return state


def _host(state):
    # Keep the former provider seam populated: even an embedding that still
    # supplies it must receive the simplified, unbound public contract.
    return SubagentHost(
        Inner(),
        llm=None,
        retriever=None,
        memory=None,
        active_work_provider=lambda: (state.active_work, "logical"),
    )


def test_spawn_schema_has_no_active_work_binding_or_bookkeeping_field():
    state = graph_state()
    schema = next(
        row for row in _host(state).schemas()
        if row["function"]["name"] == "spawn_agent"
    )["function"]

    properties = schema["parameters"]["properties"]
    assert "work_item_id" not in properties
    assert schema["parameters"]["required"] == ["agent", "task"]
    assert "complete normalized report" in schema["description"]
    assert "directly in this tool result" in schema["description"]
    assert "create the complete declared coverage frontier" not in schema["description"]
    assert "ACTIVE WORK" not in schema["description"]


def test_live_brief_ignores_a_legacy_work_item_argument():
    state = graph_state()
    host = _host(state)
    before = state.active_work.digest

    # A stale caller may still send the retired field.  It is not copied into
    # the new child contract and it cannot alter the user's work model.
    brief = host._brief(
        "Review parser",
        {"work_item_id": "parser-review", "scope": ["src/parser.py"]},
        frozenset(),
    )
    assert brief.work_item_id == ""
    assert brief.scope == ("src/parser.py",)
    assert "PARENT ACTIVE WORK ITEM" not in brief.render()
    assert state.active_work.digest == before


def test_legacy_brief_records_still_round_trip_for_archive_compatibility():
    legacy = SubagentBrief.create("Review parser", work_item_id="parser-review")
    restored = SubagentBrief.from_dict(legacy.to_dict())
    assert restored == legacy
    assert "PARENT ACTIVE WORK ITEM" in restored.render()


def test_direct_child_outcome_does_not_mutate_active_work():
    state = graph_state()
    graph_before = state.active_work.to_dict()
    revision_before = state.active_work.revision
    invocation = ToolInvocation(
        "spawn-1", "spawn_agent", {"agent": "explorer", "task": "inspect"}, 0,
    )
    outcome = ToolOutcome(
        invocation,
        ToolStatus.SUCCEEDED,
        "[child 1 · explorer · succeeded]\n\nBEGIN CHILD REPORT\nP2: parser issue\nEND CHILD REPORT",
        (
            ToolEffect("child-1:outcome", "child_outcome", {
                "status": "succeeded",
                "operational_status": "succeeded",
                "kind": "explorer",
                "launch_ordinal": 1,
                "report_completion": "complete",
                "report_sha256": "a" * 64,
                "report_bytes": 16,
                "source_coverage_status": "source_complete",
            }),
            # Old artifacts may still carry the historical join.  The reducer
            # may expose it as compatibility metadata, but must not fold it
            # into the live graph or advance the item to ready.
            ToolEffect("child-1:artifact", "child_artifact", {
                "artifact_id": "subagent-artifact-1",
                "work_item_id": "parser-review",
                "operational_status": "succeeded",
                "source_coverage_status": "source_complete",
            }),
        ),
    )

    slice_sink(state)(ToolResult(
        name="spawn_agent",
        args=dict(invocation.args),
        output=outcome.text,
        failing=False,
        status="succeeded",
        invocation_id=invocation.id,
        outcome=outcome,
    ))

    assert state.active_work.revision == revision_before
    assert state.active_work.to_dict() == graph_before
    assert state.active_work.get("parser-review").status == "in_progress"
    call = state.runtime.recent_calls[-1]
    assert call["child_operational_status"] == "succeeded"
    assert call["child_artifact_id"] == "subagent-artifact-1"
    assert call["child_work_item_id"] == "parser-review"  # compatibility projection only


def test_preflight_does_not_validate_or_reject_against_work_graph():
    state = graph_state()
    host = _host(state)
    before = state.active_work.digest

    admission, error = host._preflight_spawn(
        "spawn_agent",
        {
            "agent": "explorer",
            "task": "inspect parser",
            "work_item_id": "does-not-exist",
        },
    )
    assert error is None
    assert admission is not None
    assert admission.brief.work_item_id == ""
    assert state.active_work.digest == before
