from __future__ import annotations

from sliceagent.active_work import WorkDelta, WorkItem, attach_child_artifacts
from sliceagent.context import ResourceKind, ResourceRef
from sliceagent.events import ToolResult
from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
from sliceagent.fan_in import (
    MAX_BUNDLE_RENDER_CHARS,
    MAX_FAN_IN_CHILDREN,
    artifact_read_coverage,
    artifact_view_kind,
    build_fan_in_bundle,
    build_fan_in_manifest,
    canonical_artifact_id,
    canonical_evidence_index_handle,
    canonical_report_handle,
    normalize_evidence_account,
    normalize_evidence_status,
)
from sliceagent.hooks import ActiveWorkContinuationHook
from sliceagent.memory import NullMemory
from sliceagent.pfc import Slice, record_user, slice_sink
from sliceagent.regions import build_context_blocks, render_delegation_fan_in
from sliceagent.seed import _slice_context, make_build_slice
from sliceagent.tools import LocalToolHost


def _state_with_child():
    state = Slice()
    state.reset("delegate one review")
    record_user(state, "delegate one review", source_event_id="event", logical_id="logical")
    root = state.active_work.request_roots[-1]
    child = WorkItem(
        id="review-child", root_id=root.id, source_refs=root.source_refs,
        description="Review the parser", status="ready",
    )
    state.active_work = state.active_work.apply(
        WorkDelta(expected_revision=state.active_work.revision, creates=(child,))
    )
    return state


def _result(invocation_id, name, args, output, effects):
    invocation = ToolInvocation(invocation_id, name, args, 0)
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, output, tuple(effects))
    return ToolResult(
        name, args, output, False, status="succeeded",
        invocation_id=invocation_id, outcome=outcome,
    )


def test_page_backed_evidence_keeps_root_provenance_without_counting_as_report_consumption():
    handle = "artifacts/sub-1/evidence/obs-001-page-001.md"
    assert canonical_artifact_id("artifact", handle) == "sub-1"
    assert artifact_view_kind("artifact", handle) == "evidence"
    assert artifact_view_kind("artifact", "artifacts/sub-1.md") == "report"
    report_with_marker_prose = "Finding: code emits [truncated; bytes omitted] and says paged out."
    assert artifact_read_coverage(
        {}, report_with_marker_prose, resource_kind="artifact", handle="artifacts/sub-1.md",
    ) == "complete"

    state = _state_with_child()
    calls = ({
        "id": "spawn-1", "child_artifact_id": "sub-1",
        "child_work_item_id": "review-child", "child_digest_delivered": True,
        "child_operational_status": "ok", "child_integration_policy": "report_required",
    }, {
        "id": "read-evidence", "observed_artifact_id": "sub-1",
        "observed_artifact_view": "evidence", "observed_read_coverage": "complete",
    })
    child = build_fan_in_manifest(calls, graph=state.active_work).children[0]
    assert child.artifact_opened == "unopened"


def test_runtime_manifest_distinguishes_digest_delivery_from_full_report_consumption():
    state = _state_with_child()
    child_effect = ToolEffect("child", "child_artifact", {
        "artifact_id": "sub-1", "work_item_id": "review-child",
        "operational_status": "succeeded", "integration_policy": "report_required",
        "explorer_evidence_status": "content_retained",
        "explorer_evidence": {
            "status": "content_retained", "content_success_count": 3,
            "content_paths": ["src/a.py", "src/b.py"],
        },
        "source_coverage_status": "source_partial",
    })
    slice_sink(state)(_result(
        "spawn-1", "spawn_agent", {"agent": "explorer"}, "bounded digest", (child_effect,),
    ))
    manifest = state.runtime.fan_in_manifest
    assert manifest.children[0].digest_delivered is True
    assert manifest.children[0].artifact_opened == "unopened"
    assert manifest.children[0].evidence_status == "content_retained"
    assert "scope=0, content=3" in manifest.render()
    assert manifest.report_required_unread == manifest.children

    read_effect = ToolEffect("read", "resource_observed", {
        "resource_kind": "artifact", "handle": "artifacts/sub-1.md",
        "artifact_id": "sub-1", "read_coverage": "complete",
    })
    slice_sink(state)(_result(
        "read-1", "read_file", {"path": "artifacts/sub-1.md"}, "full report", (read_effect,),
    ))
    consumed = state.runtime.fan_in_manifest.children[0]
    assert consumed.digest_delivered is True
    assert consumed.artifact_opened == "complete"
    assert state.runtime.fan_in_manifest.report_required_unread == ()


def test_successful_bound_child_is_host_settled_ready_without_model_bookkeeping():
    state = _state_with_child()
    current = state.active_work.get("review-child")
    state.active_work = state.active_work.transition(
        current.id, "in_progress", expected_revision=state.active_work.revision,
    )
    child_effect = ToolEffect("child", "child_artifact", {
        "artifact_id": "sub-ready", "work_item_id": "review-child",
        "operational_status": "ok", "explorer_evidence_status": "content_retained",
    })
    slice_sink(state)(_result(
        "spawn-ready", "spawn_agent", {"agent": "explorer", "work_item_id": "review-child"},
        "bounded digest", (child_effect,),
    ))
    item = state.active_work.get("review-child")
    assert item.status == "ready"
    assert any(ref.kind == "child_artifact" and ref.ref == "sub-ready" for ref in item.evidence_refs)


def test_evidence_account_preserves_the_child_contract_but_is_bounded_and_tolerant():
    assert normalize_evidence_status("none") == "none"
    assert normalize_evidence_status("navigation_only") == "navigation_only"
    assert normalize_evidence_status("future-new-value") == "not_assessed"
    account = normalize_evidence_account({
        "status": "content_partial", "content_success_count": 999999,
        "truncated_content_view_count": 2,
        "scope_paths": [f"src/{i}.py" for i in range(100)],
        "content_paths": "malformed",
    })
    assert account["status"] == "content_partial"
    assert account["content_success_count"] == 10_000
    assert len(account["scope_paths"]) == 16
    assert "content_paths" not in account
    assert normalize_evidence_account("not a mapping") == {}


def test_read_effect_proves_exact_artifact_identity_and_complete_or_partial_coverage(tmp_path):
    host = LocalToolHost(str(tmp_path))
    host.resource_ref = lambda _path: ResourceRef(
        ResourceKind.ARTIFACT, "artifacts/sub-1.md",
    )
    try:
        full = host._read_resource_effects(
            ToolInvocation("read-full", "read_file", {"path": "artifacts/sub-1.md"}, 0),
            ToolStatus.SUCCEEDED, "whole report",
        )[0].payload
        partial = host._read_resource_effects(
            ToolInvocation(
                "read-page", "read_file", {"path": "artifacts/sub-1.md", "limit": 10}, 0,
            ),
            ToolStatus.SUCCEEDED, "first page",
        )[0].payload
    finally:
        host.cleanup()
    assert full["artifact_id"] == "sub-1" and full["read_coverage"] == "complete"
    assert len(full["content_sha256"]) == 64 and full["content_bytes"] == len("whole report")
    assert partial["read_coverage"] == "partial"


def test_seal_fold_appends_parent_use_relations_without_changing_source_coverage_semantics():
    state = _state_with_child()
    calls = ({
        "id": "spawn-1", "child_artifact_id": "sub-1",
        "child_work_item_id": "review-child", "child_digest_delivered": True,
        "child_operational_status": "ok",
        "child_source_coverage_status": "source_partial",
        "child_evidence_status": "content_retained",
        "child_integration_policy": "report_required",
    }, {
        "id": "read-1", "observed_artifact_id": "sub-1",
        "observed_read_coverage": "complete",
    })
    folded = attach_child_artifacts(state.active_work, calls, workspace_epoch=3)
    item = folded.get("review-child")
    refs = {(ref.kind, ref.ref, ref.qualifier) for ref in item.evidence_refs}
    assert ("child_artifact", "sub-1", "source_partial") in refs
    assert ("child_operational_status", "sub-1", "ok") in refs
    assert ("child_digest_delivered", "sub-1", "") in refs
    assert ("child_artifact_opened", "sub-1", "complete") in refs
    assert ("child_evidence_status", "sub-1", "content_retained") in refs
    assert ("child_integration_policy", "sub-1", "report_required") in refs
    assert attach_child_artifacts(folded, calls, workspace_epoch=3).digest == folded.digest
    restored = build_fan_in_manifest((), graph=folded)
    assert restored.children[0].operational_status == "ok"


def test_evidence_account_survives_work_graph_round_trip_and_complete_reads_are_monotonic():
    state = _state_with_child()
    account = normalize_evidence_account({
        "v": 1, "status": "content_partial", "scope_path_count": 3,
        "content_success_count": 2, "retained_content_view_count": 2,
        "omitted_content_view_count": 1, "truncated_content_view_count": 1,
        "scope_paths": ["src/a.py", "src/b.py", "src/c.py"],
        "content_paths": ["src/a.py", "src/b.py"], "gap_paths": ["src/c.py"],
    })
    calls = ({
        "id": "spawn-1", "child_artifact_id": "sub-1",
        "child_work_item_id": "review-child", "child_digest_delivered": True,
        "child_evidence_status": "content_partial", "child_evidence_account": account,
        "child_integration_policy": "report_required",
    }, {
        "id": "read-complete", "observed_artifact_id": "sub-1",
        "observed_read_coverage": "complete",
    })
    folded = attach_child_artifacts(state.active_work, calls, workspace_epoch=3)
    restored_graph = type(folded).from_dict(folded.to_dict())
    restored = build_fan_in_manifest((), graph=restored_graph).children[0]
    assert dict(restored.evidence_account) == account
    assert restored.artifact_opened == "complete"

    # A later tail/page read cannot downgrade previously proven complete consumption.
    reread = build_fan_in_manifest(({
        "id": "read-partial", "observed_artifact_id": "sub-1",
        "observed_read_coverage": "partial",
    },), graph=restored_graph).children[0]
    assert reread.artifact_opened == "complete"
    assert reread.needs_report_advisory is False

    partial_graph = attach_child_artifacts(state.active_work, ({
        **calls[0], "child_artifact_opened": "partial",
    },), workspace_epoch=3)
    later_complete = ({
        "id": "read-complete-later", "observed_artifact_id": "sub-1",
        "observed_read_coverage": "complete",
    },)
    upgraded = build_fan_in_manifest(later_complete, graph=partial_graph).children[0]
    assert upgraded.artifact_opened == "complete"
    resealed = attach_child_artifacts(partial_graph, later_complete, workspace_epoch=3)
    reloaded = type(resealed).from_dict(resealed.to_dict())
    assert build_fan_in_manifest((), graph=reloaded).children[0].artifact_opened == "complete"


def test_report_required_is_compatibility_metadata_not_a_continuation_gate():
    state = _state_with_child()
    calls = ({
        "id": "spawn-1", "child_artifact_id": "sub-1",
        "child_work_item_id": "review-child", "child_digest_delivered": True,
        "child_integration_policy": "report_required",
    },)
    hook = ActiveWorkContinuationHook(lambda: (state.active_work, "logical", calls))
    assert hook.should_continue_after_stop("end_turn") is None

    ordinary = ({
        **calls[0], "child_integration_policy": "digest_ok",
    },)
    digest_hook = ActiveWorkContinuationHook(lambda: (state.active_work, "logical", ordinary))
    assert digest_hook.should_continue_after_stop("end_turn") is None


def test_fan_in_region_is_high_signal_and_physically_bounded():
    state = _state_with_child()
    state.runtime.recent_calls = [{
        "id": f"spawn-{index}", "child_artifact_id": f"sub-{index}",
        "child_digest_delivered": True,
        "child_evidence_status": "navigation_only",
        "child_integration_policy": "digest_ok",
    } for index in range(MAX_FAN_IN_CHILDREN + 7)]
    rendered = render_delegation_fan_in(state)
    assert "deterministic terminal-child synthesis bundle" in rendered
    assert "sub-0.md" not in rendered and f"sub-{MAX_FAN_IN_CHILDREN + 6}.md" in rendered
    assert "+7 older delegation record(s) omitted" in rendered
    block = next(
        item for item in build_context_blocks(_slice_context(state, "(no files opened yet)"))
        if item.item_id == "region:fan_in"
    )
    assert block.priority == 100 and block.mandatory is True


def _bundle_call(index: int, disposition: str) -> dict:
    if disposition == "complete":
        return {
            "id": f"spawn-{index}", "status": "succeeded",
            "child_artifact_id": f"sub-{index}",
            "child_work_item_id": f"review-{index}",
            "child_operational_status": "succeeded",
            "child_evidence_declared": True,
            "child_evidence_status": "content_retained",
            "child_source_coverage_status": "source_complete",
        }
    if disposition == "partial":
        return {
            "id": f"spawn-{index}", "status": "succeeded",
            "child_artifact_id": f"sub-{index}",
            "child_work_item_id": f"review-{index}",
            "child_operational_status": "succeeded",
            "child_evidence_declared": True,
            "child_evidence_status": "content_partial",
            "child_source_coverage_status": "source_partial",
        }
    return {
        "id": f"spawn-{index}", "status": "failed",
        "child_artifact_id": f"sub-{index}",
        "child_work_item_id": f"review-{index}",
        "child_operational_status": "failed",
        "child_evidence_declared": True,
        "child_evidence_status": "content_partial",
    }


def test_six_terminal_children_form_one_resident_canonical_synthesis_bundle():
    calls = tuple(
        _bundle_call(index, disposition)
        for index, disposition in enumerate(
            ("complete", "partial", "failed", "complete", "partial", "failed"), 1,
        )
    )
    reports = {
        canonical_report_handle(f"sub-{index}"): f"FULL REPORT {index}\nFinding {index} with qualifiers."
        for index in range(1, 7)
    }
    opened = []

    def load(handle):
        opened.append(handle)
        return reports[handle]

    bundle = build_fan_in_bundle(calls, report_loader=load)
    assert bundle.census.terminal == 6
    assert (bundle.census.complete, bundle.census.partial, bundle.census.failed) == (2, 2, 2)
    assert bundle.census.reports_resident == 6
    assert [entry.report for entry in bundle.entries] == list(reports.values())
    assert opened == list(reports)

    rendered = bundle.render()
    for index in range(1, 7):
        assert reports[canonical_report_handle(f"sub-{index}")] in rendered
        assert canonical_report_handle(f"sub-{index}") in rendered
        assert canonical_evidence_index_handle(f"sub-{index}") in rendered
    assert "census: terminal=6; complete=2; partial=2; failed=2" in rendered


def test_bundle_keeps_full_material_immutable_but_bounds_the_mandatory_render():
    huge = "load-bearing report line\n" * 20_000
    handle = canonical_report_handle("sub-huge")
    bundle = build_fan_in_bundle(
        (_bundle_call(1, "complete") | {"child_artifact_id": "sub-huge"},),
        report_loader=lambda requested: huge if requested == handle else "",
    )
    assert bundle.entries[0].report == huge
    rendered = bundle.render()
    assert len(rendered) <= MAX_BUNDLE_RENDER_CHARS
    assert "report material paged" in rendered
    assert handle in rendered


def test_region_uses_supplied_loader_and_exposes_full_reports_as_model_visible_data():
    state = _state_with_child()
    state.runtime.recent_calls = [
        _bundle_call(1, "complete") | {"child_work_item_id": "review-child"}
    ]
    report = "exact full child report for synthesis"
    ctx = _slice_context(state, "(no files opened yet)")
    ctx["fan_in_report_loader"] = lambda _handle: report
    block = next(
        item for item in build_context_blocks(ctx) if item.item_id == "region:fan_in"
    )
    assert report in block.content
    assert "@sliceagent/evidence/children/sub-1.md" in block.content
    assert block.mandatory is True


def test_production_seed_loader_reconstructs_fan_in_from_contextfs(tmp_path):
    from sliceagent.contextfs import ArtifactContextProvider
    from sliceagent.persistence import Artifact, ArtifactStore
    from sliceagent.runtime_persistence import CoreArtifactFS

    state = _state_with_child()
    state.runtime.recent_calls = [
        _bundle_call(1, "complete") | {"child_work_item_id": "review-child"}
    ]
    report = "FULL CANONICAL CHILD REPORT\nP2: parser rejects a valid empty input."
    store = ArtifactStore(str(tmp_path / "artifacts"))
    store.put(Artifact(
        id="sub-1", kind="subagent", workspace_id="workspace", session_id="session",
        task_id="task", status="ok", summary=report,
        structured_body={"report": report, "observations": [], "claims": []},
    ))
    core = CoreArtifactFS(store)
    host = LocalToolHost(str(tmp_path))
    host._artifacts = core
    host._contextfs.mount(
        "evidence/children",
        ArtifactContextProvider(
            core, kinds=("subagent",),
            canonical_mount="@sliceagent/evidence/children", title="CHILDREN",
        ),
    )
    try:
        seed = make_build_slice(
            state, host, None, NullMemory(), state.goal, session_id="session",
        )()
    finally:
        host.cleanup()
    rendered = "\n".join(str(message.get("content") or "") for message in seed)
    assert report in rendered
    assert "census: terminal=1; complete=1" in rendered
