"""Source-linked Active Work kernel: invariants, recovery, and hostile deltas."""
from __future__ import annotations

import dataclasses
import json

import pytest

from sliceagent.active_work import (
    EvidenceRef,
    GraphValidationError,
    OutputRef,
    ResourceRef,
    RevisionConflictError,
    SourceMismatchError,
    SourceRef,
    WorkDelta,
    WorkGraph,
    WorkItem,
    request_root_item,
)


EVENT = "user:s1:turn-7"
TEXT = "切换到 SliceAgent workspace，然后解释 end-game design。"


def root(graph: WorkGraph) -> WorkItem:
    return graph.request_roots[0]


def replace(item: WorkItem, **changes) -> WorkItem:
    return dataclasses.replace(item, **changes)


def test_source_ref_binds_exact_unicode_codepoint_range_and_detects_change():
    ref = SourceRef.bind(EVENT, TEXT, start=4, end=14)
    assert ref.extract(TEXT) == TEXT[4:14]
    restored = SourceRef.from_dict(json.loads(json.dumps(ref.to_dict())))
    assert restored == ref and restored.extract(TEXT) == TEXT[4:14]

    with pytest.raises(SourceMismatchError, match="no longer matches"):
        ref.extract(TEXT.replace("workspace", "folder"))
    with pytest.raises(GraphValidationError, match="within"):
        SourceRef.bind(EVENT, TEXT, start=4, end=len(TEXT) + 1)


def test_request_admission_is_mechanical_exact_deterministic_and_idempotent():
    first = WorkGraph().add_request_root(EVENT, TEXT)
    second = first.add_request_root(EVENT, TEXT)
    item = root(first)

    assert second is first
    assert first.revision == 1
    assert item.description == ""  # host did not invent a semantic paraphrase
    assert item.source_refs[0].extract(TEXT) == TEXT
    assert request_root_item(EVENT, TEXT) == item
    assert item.id == request_root_item(EVENT, "different words").id  # identity belongs to the event

    with pytest.raises(GraphValidationError, match="already owns"):
        first.add_request_root(EVENT, TEXT + " changed")


def test_terminal_request_history_does_not_accumulate_in_the_active_graph():
    graph = WorkGraph()
    for index in range(20):
        graph = graph.open_request(
            f"event-{index}", f"request {index}", logical_id=f"logical-{index}",
        )
        graph = graph.seal_current(
            "end_turn", OutputRef("response", f"answer-{index}"),
            logical_id=f"logical-{index}",
        )
        assert len(graph.request_roots) == 1
        assert len(graph.items) == 1
    assert graph.revision == 40


def test_graph_and_nested_records_are_immutable():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        graph.revision = 9
    with pytest.raises(dataclasses.FrozenInstanceError):
        root(graph).status = "verified"
    with pytest.raises(TypeError, match="assignment|immutable"):
        graph._by_id["x"] = root(graph)


def test_delta_can_create_dependency_and_update_root_atomically():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    child = WorkItem(
        id="inspect-design",
        root_id=request.id,
        description="Inspect the current context machinery",
        source_refs=request.source_refs,
        status="in_progress",
    )
    changed_root = replace(request, status="in_progress", dependencies=(child.id,))
    graph = graph.apply(WorkDelta(
        expected_revision=graph.revision,
        creates=(child,),
        updates=(changed_root,),
    ))

    assert graph.revision == 2
    assert [item.id for item in graph.dependency_closure()] == [request.id, child.id]
    assert graph.get(child.id) == replace(child, logical_id=request.logical_id)


def test_unknown_dependencies_self_dependencies_and_cycles_are_rejected():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)

    with pytest.raises(GraphValidationError, match="unknown dependency"):
        graph.apply(WorkDelta(
            expected_revision=graph.revision,
            updates=(replace(request, dependencies=("missing",)),),
        ))

    other_graph = graph.open_request("other-event", "other", logical_id="other-logical")
    other = other_graph.request_roots[-1]
    with pytest.raises(GraphValidationError, match="across request roots"):
        other_graph.upsert(replace(request, dependencies=(other.id,)))
    with pytest.raises(GraphValidationError, match="depends on itself"):
        graph.apply(WorkDelta(
            expected_revision=graph.revision,
            updates=(replace(request, dependencies=(request.id,)),),
        ))

    child = WorkItem(
        id="child", root_id=request.id, source_refs=request.source_refs,
        dependencies=(request.id,),
    )
    with pytest.raises(GraphValidationError, match="cycle"):
        graph.apply(WorkDelta(
            expected_revision=graph.revision,
            creates=(child,),
            updates=(replace(request, dependencies=(child.id,)),),
        ))


def test_dependencies_are_append_only_until_an_item_is_explicitly_retired():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    child = WorkItem(id="child", root_id=request.id, source_refs=request.source_refs)
    graph = graph.apply(WorkDelta(
        expected_revision=1,
        creates=(child,),
        updates=(replace(request, dependencies=(child.id,)),),
    ))
    with pytest.raises(GraphValidationError, match="erase dependency"):
        graph.upsert(replace(root(graph), dependencies=()))


def test_compare_and_swap_rejects_stale_delta_and_duplicate_identity():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    with pytest.raises(RevisionConflictError, match="expected revision 0"):
        graph.apply(WorkDelta(expected_revision=0, updates=(replace(request, status="in_progress"),)))
    with pytest.raises(GraphValidationError, match="existing work item"):
        graph.apply(WorkDelta(expected_revision=1, creates=(request,)))
    with pytest.raises(GraphValidationError, match="both create and update"):
        WorkDelta(expected_revision=1, creates=(request,), updates=(request,))


def test_delivery_and_verification_require_the_right_evidence_families():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    with pytest.raises(GraphValidationError, match="delivered output"):
        replace(request, status="delivered")

    delivered = replace(
        request,
        status="delivered",
        output_refs=(OutputRef("response", "turn-7:answer"),),
    )
    graph = graph.apply(WorkDelta(expected_revision=1, updates=(delivered,)))
    with pytest.raises(GraphValidationError, match="verification evidence"):
        replace(delivered, status="verified")

    verified = replace(
        delivered,
        status="verified",
        evidence_refs=(EvidenceRef("test", "artifact:test-active-work"),),
    )
    graph = graph.apply(WorkDelta(expected_revision=2, updates=(verified,)))
    assert root(graph).status == "verified"
    assert graph.unresolved_roots == ()


def test_terminal_statuses_cannot_silently_reopen():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    cancelled = replace(request, status="cancelled")
    graph = graph.apply(WorkDelta(expected_revision=1, updates=(cancelled,)))
    with pytest.raises(GraphValidationError, match="cancelled -> in_progress"):
        graph.apply(WorkDelta(
            expected_revision=2,
            updates=(replace(cancelled, status="in_progress"),),
        ))


def test_delivered_work_can_reopen_but_cannot_erase_outputs_or_provenance():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    delivered = replace(
        request,
        status="delivered",
        output_refs=(OutputRef("response", "answer-1"),),
        evidence_refs=(EvidenceRef("observation", "obs-1"),),
    )
    graph = graph.apply(WorkDelta(expected_revision=1, updates=(delivered,)))
    reopened = replace(delivered, status="in_progress")
    graph = graph.apply(WorkDelta(expected_revision=2, updates=(reopened,)))
    assert root(graph).status == "in_progress"

    with pytest.raises(GraphValidationError, match="erase output"):
        graph.apply(WorkDelta(
            expected_revision=3,
            updates=(replace(reopened, output_refs=()),),
        ))
    with pytest.raises(GraphValidationError, match="erase source"):
        graph.apply(WorkDelta(
            expected_revision=3,
            updates=(replace(reopened, source_refs=(SourceRef.bind("user:new", "new"),)),),
        ))


def test_supersession_requires_a_real_replacement():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    original = WorkItem(
        id="original", root_id=request.id, source_refs=request.source_refs,
        description="Original child work",
    )
    graph = graph.apply(WorkDelta(expected_revision=1, creates=(original,)))
    original = graph.get(original.id)
    assert original is not None
    replacement = WorkItem(
        id="replacement", root_id=request.id, source_refs=request.source_refs,
        description="Updated work after correction",
    )
    superseded = replace(original, status="superseded", superseded_by=replacement.id)
    graph = graph.apply(WorkDelta(
        expected_revision=2,
        creates=(replacement,),
        updates=(superseded,),
    ))
    assert graph.get(original.id).superseded_by == replacement.id

    bad = WorkGraph().add_request_root("user:bad", "bad")
    with pytest.raises(GraphValidationError, match="unknown replacement"):
        bad.apply(WorkDelta(
            expected_revision=1,
            updates=(replace(root(bad), status="superseded", superseded_by="ghost"),),
        ))


def test_supersession_cycles_and_cross_request_replacements_are_rejected():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    a = WorkItem(id="a", root_id=request.id, source_refs=request.source_refs)
    b = WorkItem(id="b", root_id=request.id, source_refs=request.source_refs)
    with pytest.raises(GraphValidationError, match="supersession cycle"):
        graph.apply(WorkDelta(
            expected_revision=1,
            creates=(
                replace(a, status="superseded", superseded_by="b"),
                replace(b, status="superseded", superseded_by="a"),
            ),
        ))

    other = request_root_item("user:other", "other")
    first_child = WorkItem(id="first-child", root_id=request.id, source_refs=request.source_refs)
    with pytest.raises(GraphValidationError, match="different request root"):
        WorkGraph(items=(
            request,
            replace(first_child, status="superseded", superseded_by=other.id),
            other,
        ))


def test_source_event_cannot_be_rebound_to_conflicting_content():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    conflicting = WorkItem(
        id="conflict",
        root_id=request.id,
        source_refs=(SourceRef.bind(EVENT, TEXT + " altered"),),
    )
    with pytest.raises(GraphValidationError, match="conflicting immutable content"):
        graph.apply(WorkDelta(expected_revision=1, creates=(conflicting,)))


def test_external_source_validation_catches_missing_and_changed_ledger_events():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    graph.validate_sources({EVENT: TEXT})
    with pytest.raises(SourceMismatchError, match="unavailable"):
        graph.validate_sources({})
    with pytest.raises(SourceMismatchError, match="no longer matches"):
        graph.validate_sources({EVENT: TEXT + "!"})


def test_json_round_trip_is_canonical_lossless_and_detached_from_wire_dict():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    delivered = replace(
        request,
        status="delivered",
        evidence_refs=(EvidenceRef("receipt", "receipt-1"),),
        output_refs=(OutputRef("response", "response-1"),),
    )
    graph = graph.apply(WorkDelta(expected_revision=1, updates=(delivered,)))
    wire = graph.to_dict()
    encoded = json.dumps(wire, sort_keys=True, ensure_ascii=False)
    restored = WorkGraph.from_dict(json.loads(encoded))

    assert restored == graph
    assert restored.digest == graph.digest
    wire["items"][0]["status"] = "cancelled"
    wire["items"][0]["source_refs"][0]["event_id"] = "mutated"
    assert root(graph).status == "delivered"
    assert root(graph).source_refs[0].event_id == EVENT


def test_delta_round_trip_preserves_typed_records():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    request = root(graph)
    delta = WorkDelta(
        expected_revision=graph.revision,
        updates=(replace(request, status="waiting_user"),),
    )
    assert WorkDelta.from_dict(json.loads(json.dumps(delta.to_dict()))) == delta


def test_clean_integration_api_tracks_logical_request_and_workspace_epoch():
    graph = WorkGraph().open_request(
        EVENT,
        TEXT,
        workspace_epoch=7,
        logical_id="logical-turn-42",
    )
    request = root(graph)
    assert request.logical_id == "logical-turn-42"
    assert request.workspace_epoch == 7
    assert graph.active_frontier() == (request,)
    assert graph.dependency_closure(item.id for item in graph.active_frontier()) == (request,)

    # Same logical request cannot be accidentally reopened under another source event.
    with pytest.raises(GraphValidationError, match="already owns"):
        graph.open_request("user:duplicate", "duplicate", workspace_epoch=8, logical_id="logical-turn-42")


def test_resources_are_typed_epoch_scoped_and_append_only():
    graph = WorkGraph().open_request(EVENT, TEXT, workspace_epoch=3, logical_id="logical")
    request = root(graph)
    resource = ResourceRef("file", "src/app.py", workspace_epoch=3, revision="sha256:abc")
    updated = replace(request, resource_refs=(resource,))
    graph = graph.upsert(updated)
    restored = WorkGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert root(restored).resource_refs == (resource,)

    with pytest.raises(GraphValidationError, match="erase resource"):
        graph.upsert(replace(root(graph), resource_refs=()))


def test_upsert_and_apply_delta_are_idempotent_for_identical_snapshots():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    assert graph.upsert(root(graph)) is graph
    assert graph.apply_delta(WorkDelta(expected_revision=graph.revision, updates=(root(graph),))) is graph


def test_seal_current_distinguishes_delivery_waiting_recovery_and_transport():
    delivered = WorkGraph().open_request(EVENT, TEXT, logical_id="deliver")
    delivered = delivered.seal_current("completed", OutputRef("response", "answer"))
    assert root(delivered).status == "delivered"
    assert root(delivered).stop_reason == "completed"

    waiting = WorkGraph().open_request("u:waiting", "choose one", logical_id="waiting")
    waiting = waiting.seal_current("waiting_user", OutputRef("response", "question"))
    assert root(waiting).status == "waiting_user"

    interrupted = WorkGraph().open_request("u:interrupt", "do work", logical_id="interrupt")
    interrupted = interrupted.seal_current("interrupted")
    assert root(interrupted).status == "in_progress"

    transported = WorkGraph().open_request("u:switch", "switch and inspect", logical_id="switch")
    transported = transported.seal_current(
        "workspace_transition",
        OutputRef("response", "progress-only"),
        transitioned=True,
    )
    assert root(transported).status == "in_progress"
    assert root(transported).output_refs[-1].ref == "progress-only"


def test_response_cannot_hide_unfinished_child_from_the_frontier():
    graph = WorkGraph().open_request(EVENT, TEXT, logical_id="compound")
    request = root(graph)
    child = WorkItem(
        id="inspect-target", root_id=request.id, source_refs=request.source_refs,
        description="Inspect the target workspace", status="in_progress",
    )
    graph = graph.apply(WorkDelta(expected_revision=1, creates=(child,)))
    graph = graph.seal_current("end_turn", OutputRef("response", "progress-answer"))

    assert root(graph).status == "in_progress"
    assert {item.id for item in graph.active_frontier()} == {request.id, child.id}
    assert {item.id for item in graph.dependency_closure()} == {request.id, child.id}


def test_ready_child_is_delivered_only_when_the_host_attaches_the_real_response():
    graph = WorkGraph().open_request(EVENT, TEXT, logical_id="ready")
    request = root(graph)
    child = WorkItem(
        id="prepared-report", root_id=request.id, source_refs=request.source_refs,
        description="Prepare the architecture report", status="ready",
    )
    graph = graph.apply(WorkDelta(expected_revision=1, creates=(child,)))
    interrupted = graph.seal_current("interrupted")
    assert interrupted.get(child.id).status == "ready"
    assert root(interrupted).status == "in_progress"

    delivered = interrupted.seal_current("end_turn", OutputRef("response", "answer"))
    assert delivered.get(child.id).status == "delivered"
    assert delivered.get(child.id).output_refs == (OutputRef("response", "answer"),)
    assert root(delivered).status == "delivered"


def test_checkpoint_records_preserve_revision_and_typed_graph():
    graph = WorkGraph().open_request(EVENT, TEXT, workspace_epoch=2, logical_id="logical")
    graph = graph.seal_current("waiting_user", OutputRef("response", "question"))
    records = json.loads(json.dumps(graph.to_records(), ensure_ascii=False))
    assert records[0]["record_type"] == "active_work_graph"
    restored = WorkGraph.from_records(records)
    assert restored == graph and restored.digest == graph.digest
    assert WorkGraph.from_records([]) == WorkGraph()  # pre-Active-Work checkpoint

    with pytest.raises(GraphValidationError, match="header"):
        WorkGraph.from_records(records[1:])


def test_child_items_must_share_the_roots_logical_identity():
    graph = WorkGraph().open_request(EVENT, TEXT, logical_id="logical")
    request = root(graph)
    child = WorkItem(
        id="bad-child",
        root_id=request.id,
        logical_id="other-logical-turn",
        source_refs=request.source_refs,
    )
    with pytest.raises(GraphValidationError, match="logical_id differs"):
        graph.upsert(child)


@pytest.mark.parametrize(
    "corrupt",
    [
        {"v": 99, "revision": 0, "items": []},
        {"v": 1, "revision": -1, "items": []},
        {"v": 1, "revision": 0, "items": {}},
        {"v": 1, "revision": 0, "items": ["not-an-object"]},
    ],
)
def test_corrupt_serialized_graphs_fail_closed(corrupt):
    with pytest.raises(GraphValidationError):
        WorkGraph.from_dict(corrupt)


def test_non_iterable_hostile_wire_values_raise_domain_errors():
    graph = WorkGraph().add_request_root(EVENT, TEXT)
    item = root(graph).to_dict()
    item["source_refs"] = 7
    with pytest.raises(GraphValidationError, match="source_refs must be a sequence"):
        WorkItem.from_dict(item)
    with pytest.raises(GraphValidationError, match="work_delta.creates must be a sequence"):
        WorkDelta.from_dict({"expected_revision": 1, "creates": 7})
    with pytest.raises(GraphValidationError, match="active-work records must be a sequence"):
        WorkGraph.from_records(7)
    with pytest.raises(GraphValidationError, match="sequence of item IDs"):
        graph.dependency_closure(root(graph).id)


def test_request_admission_retry_is_idempotent_after_lifecycle_progress():
    graph = WorkGraph().open_request(EVENT, TEXT, workspace_epoch=2, logical_id="logical-retry")
    graph = graph.seal_current("interrupted", logical_id="logical-retry")
    revision = graph.revision

    retried = graph.open_request(EVENT, TEXT, workspace_epoch=2, logical_id="logical-retry")
    assert retried is graph
    assert retried.revision == revision
    assert root(retried).status == "in_progress"

    with pytest.raises(GraphValidationError, match="already owns request root"):
        graph.open_request(EVENT, TEXT + " changed", workspace_epoch=2, logical_id="logical-retry")


def test_terminal_request_root_cannot_retain_unresolved_child_work():
    graph = WorkGraph().open_request(EVENT, TEXT, logical_id="compound-terminal")
    request = root(graph)
    child = WorkItem(
        id="still-open", root_id=request.id, source_refs=request.source_refs,
        logical_id=request.logical_id, description="Still required", status="in_progress",
    )
    graph = graph.apply_delta(WorkDelta(graph.revision, creates=(child,)))

    with pytest.raises(GraphValidationError, match="terminal request root.*unresolved child work"):
        graph.transition(
            request.id, "delivered", output_refs=(OutputRef("turn_artifact", "turn-1"),),
        )


def test_active_work_metadata_is_single_line_but_exact_user_source_may_be_multiline():
    multiline = "inspect this\nthen report"
    graph = WorkGraph().open_request(EVENT, multiline, logical_id="multiline-source")
    request = root(graph)
    assert request.source_refs[0].extract(multiline) == multiline

    invalid_records = (
        lambda: SourceRef.bind("event\nforged", TEXT),
        lambda: EvidenceRef("receipt\n# injected", "receipt-1"),
        lambda: EvidenceRef("receipt", "receipt-1\rforged"),
        lambda: OutputRef("response\n# injected", "turn-1"),
        lambda: ResourceRef("workspace_file", "src/app.py\n# injected"),
        lambda: ResourceRef("workspace_file", "src/app.py", revision="sha\rforged"),
        lambda: WorkItem(
            id="child\nforged", root_id=request.id, source_refs=request.source_refs,
            logical_id=request.logical_id, description="child",
        ),
        lambda: WorkItem(
            id="child-description", root_id=request.id, source_refs=request.source_refs,
            logical_id=request.logical_id, description="inspect\n# injected",
        ),
        lambda: WorkItem(
            id="child-dependency", root_id=request.id, source_refs=request.source_refs,
            logical_id=request.logical_id, description="inspect", dependencies=("dep\rforged",),
        ),
    )
    for factory in invalid_records:
        with pytest.raises(GraphValidationError, match="CR or LF"):
            factory()
