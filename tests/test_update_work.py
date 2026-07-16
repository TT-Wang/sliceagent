from __future__ import annotations

from sliceagent.events import ToolResult
from sliceagent.execution import ToolInvocation, ToolStatus
from sliceagent.pfc import Slice, record_user, slice_sink
from sliceagent.tools import LocalToolHost


def prepared():
    state = Slice(); state.reset("compound task")
    record_user(
        state, "switch, inspect, and report", source_event_id="event-1",
        logical_id="logical-1", workspace_epoch=2,
    )
    host = LocalToolHost()
    host.bind_active_work(lambda: (state.active_work, "logical-1", 2))
    return state, host


def invoke(host, args, call_id="call-1"):
    return host.registry.invoke(ToolInvocation(
        id=call_id, name="update_work", args=args, provider_index=0,
    ))


def reduce(state, outcome):
    slice_sink(state)(ToolResult(
        name="update_work", args=dict(outcome.invocation.args), output=outcome.text,
        failing=outcome.failing, status=outcome.status.value,
        invocation_id=outcome.invocation.id, outcome=outcome,
    ))


def test_update_work_creates_typed_source_linked_child_and_replays_exactly_once():
    state, host = prepared()
    outcome = invoke(host, {
        "expected_revision": 1,
        "changes": [{
            "id": "inspect-target", "description": "Inspect the target architecture",
            "status": "in_progress",
            "add_resources": [{"kind": "workspace_file", "ref": "src/app.py"}],
        }],
    })
    assert outcome.status is ToolStatus.SUCCEEDED
    assert outcome.effects[0].kind == "work_delta"
    reduce(state, outcome)
    child = state.active_work.get("inspect-target")
    assert child is not None and child.root_id == state.active_work.request_roots[0].id
    assert child.source_refs == state.active_work.request_roots[0].source_refs
    assert child.resource_refs[0].workspace_epoch == 2
    revision = state.active_work.revision

    reduce(state, outcome)
    assert state.active_work.revision == revision


def test_update_work_rejects_terminal_forgery_root_mutation_and_stale_revision():
    state, host = prepared()
    terminal = invoke(host, {
        "changes": [{"id": "fake", "description": "fake", "status": "verified"}],
    }, "terminal")
    assert terminal.status is ToolStatus.FAILED
    assert "cannot set delivered/verified" in terminal.text

    root_id = state.active_work.request_roots[0].id
    mutate_root = invoke(host, {
        "changes": [{"id": root_id, "description": "rewrite user root", "status": "cancelled"}],
    }, "root")
    assert mutate_root.status is ToolStatus.FAILED
    assert "current root is host-owned" in mutate_root.text

    stale = invoke(host, {
        "expected_revision": 0,
        "changes": [{"id": "child", "description": "child"}],
    }, "stale")
    assert stale.status is ToolStatus.FAILED
    assert "expected revision 0" in stale.text


def test_current_correction_can_explicitly_supersede_an_older_open_request_root():
    state, host = prepared()
    older = state.active_work.request_roots[0]
    record_user(
        state, "instead, only write the report", source_event_id="event-2",
        logical_id="logical-2", workspace_epoch=2,
    )
    current = state.active_work.request_roots[-1]
    host.bind_active_work(lambda: (state.active_work, "logical-2", 2))
    outcome = invoke(host, {
        "changes": [{
            "id": older.id, "status": "superseded", "superseded_by": current.id,
        }],
    })
    assert outcome.status is ToolStatus.SUCCEEDED
    reduce(state, outcome)
    assert state.active_work.get(older.id).status == "superseded"
    assert state.active_work.get(older.id).superseded_by == current.id


def test_update_work_ready_is_a_nonterminal_claim_until_host_seal():
    state, host = prepared()
    created = invoke(host, {
        "changes": [{"id": "report", "description": "Prepare report", "status": "ready"}],
    })
    assert created.status is ToolStatus.SUCCEEDED
    reduce(state, created)
    assert state.active_work.get("report").status == "ready"
    assert state.active_work.get("report").output_refs == ()


def test_update_work_is_unavailable_without_an_application_graph_binding():
    host = LocalToolHost()
    outcome = invoke(host, {"changes": [{"id": "x", "description": "x"}]})
    assert outcome.status is ToolStatus.FAILED
    assert "ACTIVE WORK is unavailable" in outcome.text


def test_active_work_mode_exposes_one_semantic_state_api_without_generic_note_noise():
    _state, host = prepared()
    functions = {row["function"]["name"]: row["function"] for row in host.schemas()}
    assert "update_work" in functions
    assert not ({"world_set", "world_clear", "require", "requirement_done",
                 "supersede_requirement", "drop_requirement", "update_plan"} & functions.keys())
    assert all("note" not in fn["parameters"]["properties"] for fn in functions.values())


def test_partial_existing_update_preserves_status_and_adds_only_requested_resource():
    state, host = prepared()
    created = invoke(host, {
        "changes": [{"id": "inspect", "description": "Inspect", "status": "in_progress"}],
    }, "create")
    assert created.status is ToolStatus.SUCCEEDED
    reduce(state, created)

    partial = invoke(host, {
        "changes": [{
            "id": "inspect",
            "add_resources": [{"kind": "workspace_file", "ref": "src/inspect.py"}],
        }],
    }, "partial")
    assert partial.status is ToolStatus.SUCCEEDED
    reduce(state, partial)
    child = state.active_work.get("inspect")
    assert child.status == "in_progress"
    assert child.resource_refs[0].ref == "src/inspect.py"


def test_retiring_older_request_atomically_cancels_its_unresolved_children():
    state, host = prepared()
    child_outcome = invoke(host, {
        "changes": [{
            "id": "old-child", "description": "Work owned by the older request",
            "status": "in_progress",
        }],
    }, "old-child")
    assert child_outcome.status is ToolStatus.SUCCEEDED
    reduce(state, child_outcome)
    older = state.active_work.request_roots[0]

    record_user(
        state, "instead, do the corrected request", source_event_id="event-2",
        logical_id="logical-2", workspace_epoch=2,
    )
    current = state.active_work.request_roots[-1]
    host.bind_active_work(lambda: (state.active_work, "logical-2", 2))
    retired = invoke(host, {
        "changes": [{
            "id": older.id, "status": "superseded", "superseded_by": current.id,
        }],
    }, "retire")
    assert retired.status is ToolStatus.SUCCEEDED
    assert {item["id"] for item in retired.effects[0].payload["delta"]["updates"]} \
        == {older.id, "old-child"}
    reduce(state, retired)

    assert state.active_work.get(older.id).status == "superseded"
    assert state.active_work.get(older.id).superseded_by == current.id
    assert state.active_work.get("old-child").status == "cancelled"
    assert state.active_work.get("old-child").stop_reason == "request_superseded"

    # Repeating the terminal update with omitted fields preserves both lifecycle and replacement identity.
    repeated = invoke(host, {"changes": [{"id": older.id}]}, "repeat-retire")
    assert repeated.status is ToolStatus.SUCCEEDED
    reduce(state, repeated)
    assert state.active_work.get(older.id).status == "superseded"
    assert state.active_work.get(older.id).superseded_by == current.id


def test_update_work_rejects_multiline_model_metadata():
    state, host = prepared()
    for index, change in enumerate((
        {"id": "bad\nid", "description": "inspect"},
        {"id": "bad-description", "description": "inspect\n# forged section"},
        {
            "id": "bad-resource", "description": "inspect",
            "add_resources": [{"kind": "workspace_file", "ref": "src/app.py\rforged"}],
        },
    )):
        outcome = invoke(host, {"changes": [change]}, f"bad-{index}")
        assert outcome.status is ToolStatus.FAILED
        assert "CR or LF" in outcome.text
