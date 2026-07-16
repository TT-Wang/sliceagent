from __future__ import annotations

"""Legacy fan-in compatibility plus proof it is no longer a live context path.

Old seals and WorkGraph records remain readable through ``sliceagent.fan_in``.
New child computation returns directly in its tool result, so neither the seed
nor the elastic region compiler mounts a synthetic fan-in bundle.
"""

from sliceagent.active_work import WorkDelta, WorkItem, attach_child_artifacts
from sliceagent.context import ResourceKind, ResourceRef
from sliceagent.fan_in import (
    MAX_BUNDLE_RENDER_CHARS,
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
from sliceagent.memory import NullMemory
from sliceagent.pfc import Slice, record_user
from sliceagent.regions import build_context_blocks, render_delegation_fan_in
from sliceagent.seed import _slice_context, make_build_slice
from sliceagent.tools import LocalToolHost


def _state_with_child():
    state = Slice()
    state.reset("delegate one review")
    record_user(state, "delegate one review", source_event_id="event", logical_id="logical")
    root = state.active_work.request_roots[-1]
    child = WorkItem(
        id="review-child",
        root_id=root.id,
        source_refs=root.source_refs,
        description="Review the parser",
        status="ready",
    )
    state.active_work = state.active_work.apply(
        WorkDelta(expected_revision=state.active_work.revision, creates=(child,))
    )
    return state


def _bundle_call(index: int, disposition: str) -> dict:
    if disposition == "complete":
        return {
            "id": f"spawn-{index}",
            "status": "succeeded",
            "child_artifact_id": f"sub-{index}",
            "child_work_item_id": f"review-{index}",
            "child_operational_status": "succeeded",
            "child_evidence_declared": True,
            "child_evidence_status": "content_retained",
            "child_source_coverage_status": "source_complete",
        }
    if disposition == "partial":
        return {
            "id": f"spawn-{index}",
            "status": "succeeded",
            "child_artifact_id": f"sub-{index}",
            "child_work_item_id": f"review-{index}",
            "child_operational_status": "succeeded",
            "child_evidence_declared": True,
            "child_evidence_status": "content_partial",
            "child_source_coverage_status": "source_partial",
        }
    return {
        "id": f"spawn-{index}",
        "status": "failed",
        "child_artifact_id": f"sub-{index}",
        "child_work_item_id": f"review-{index}",
        "child_operational_status": "failed",
        "child_evidence_declared": True,
        "child_evidence_status": "content_partial",
    }


def test_legacy_artifact_helpers_preserve_exact_identity_and_view_kind():
    handle = "artifacts/sub-1/evidence/obs-001-page-001.md"
    assert canonical_artifact_id("artifact", handle) == "sub-1"
    assert artifact_view_kind("artifact", handle) == "evidence"
    assert artifact_view_kind("artifact", "artifacts/sub-1.md") == "report"
    report_with_marker_prose = "Finding: code emits [truncated; bytes omitted] and says paged out."
    assert artifact_read_coverage(
        {},
        report_with_marker_prose,
        resource_kind="artifact",
        handle="artifacts/sub-1.md",
    ) == "complete"


def test_legacy_evidence_account_is_bounded_and_tolerant():
    assert normalize_evidence_status("none") == "none"
    assert normalize_evidence_status("navigation_only") == "navigation_only"
    assert normalize_evidence_status("future-new-value") == "not_assessed"
    account = normalize_evidence_account({
        "status": "content_partial",
        "content_success_count": 999999,
        "truncated_content_view_count": 2,
        "scope_paths": [f"src/{index}.py" for index in range(100)],
        "content_paths": "malformed",
    })
    assert account["status"] == "content_partial"
    assert account["content_success_count"] == 10_000
    assert len(account["scope_paths"]) == 16
    assert "content_paths" not in account
    assert normalize_evidence_account("not a mapping") == {}


def test_read_effect_keeps_legacy_artifact_coverage_proofs(tmp_path):
    host = LocalToolHost(str(tmp_path))
    host.resource_ref = lambda _path: ResourceRef(ResourceKind.ARTIFACT, "artifacts/sub-1.md")
    try:
        from sliceagent.execution import ToolInvocation, ToolStatus

        full = host._read_resource_effects(
            ToolInvocation("read-full", "read_file", {"path": "artifacts/sub-1.md"}, 0),
            ToolStatus.SUCCEEDED,
            "whole report",
        )[0].payload
        partial = host._read_resource_effects(
            ToolInvocation(
                "read-page",
                "read_file",
                {"path": "artifacts/sub-1.md", "limit": 10},
                0,
            ),
            ToolStatus.SUCCEEDED,
            "first page",
        )[0].payload
    finally:
        host.cleanup()
    assert full["artifact_id"] == "sub-1"
    assert full["read_coverage"] == "complete"
    assert len(full["content_sha256"]) == 64
    assert full["content_bytes"] == len("whole report")
    assert partial["read_coverage"] == "partial"


def test_legacy_work_graph_fan_in_records_round_trip_without_losing_evidence():
    state = _state_with_child()
    account = normalize_evidence_account({
        "v": 1,
        "status": "content_partial",
        "scope_path_count": 3,
        "content_success_count": 2,
        "retained_content_view_count": 2,
        "omitted_content_view_count": 0,
        "truncated_content_view_count": 1,
        "scope_paths": ["src/a.py", "src/b.py", "src/c.py"],
        "content_paths": ["src/a.py", "src/b.py"],
        "gap_paths": ["src/c.py"],
    })
    calls = ({
        "id": "spawn-1",
        "child_artifact_id": "sub-1",
        "child_work_item_id": "review-child",
        "child_digest_delivered": True,
        "child_operational_status": "ok",
        "child_evidence_status": "content_partial",
        "child_evidence_account": account,
        "child_source_coverage_status": "source_partial",
        "child_integration_policy": "report_required",
    }, {
        "id": "read-complete",
        "observed_artifact_id": "sub-1",
        "observed_read_coverage": "complete",
    })

    folded = attach_child_artifacts(state.active_work, calls, workspace_epoch=3)
    restored_graph = type(folded).from_dict(folded.to_dict())
    child = build_fan_in_manifest((), graph=restored_graph).children[0]
    assert child.artifact_id == "sub-1"
    assert child.operational_status == "ok"
    assert child.artifact_opened == "complete"
    assert dict(child.evidence_account) == account
    assert child.source_coverage_status == "source_partial"
    assert attach_child_artifacts(folded, calls, workspace_epoch=3).digest == folded.digest

    # A later partial page read cannot downgrade a complete legacy proof.
    reread = build_fan_in_manifest(({
        "id": "read-partial",
        "observed_artifact_id": "sub-1",
        "observed_read_coverage": "partial",
    },), graph=restored_graph).children[0]
    assert reread.artifact_opened == "complete"


def test_legacy_bundle_reconstructs_six_reports_in_canonical_order():
    calls = tuple(
        _bundle_call(index, disposition)
        for index, disposition in enumerate(
            ("complete", "partial", "failed", "complete", "partial", "failed"),
            1,
        )
    )
    reports = {
        canonical_report_handle(f"sub-{index}"): (
            f"FULL REPORT {index}\nFinding {index} with qualifiers."
        )
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


def test_legacy_bundle_keeps_full_material_but_bounds_its_compatibility_render():
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


def test_legacy_renderer_is_not_registered_as_a_live_context_region():
    state = _state_with_child()
    state.runtime.recent_calls = [
        _bundle_call(1, "complete") | {"child_work_item_id": "review-child"}
    ]

    # The explicit function stays available to inspect old records.
    assert "deterministic terminal-child synthesis bundle" in render_delegation_fan_in(state)

    blocks = build_context_blocks(_slice_context(state, "(no files opened yet)"))
    ids = {block.item_id for block in blocks}
    assert "region:fan_in" not in ids
    rendered = "\n".join(block.content for block in blocks)
    assert "# DELEGATION FAN-IN" not in rendered
    assert "HOST FAN-IN" not in rendered


def test_production_seed_does_not_reload_or_inject_old_child_reports(tmp_path):
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
        id="sub-1",
        kind="subagent",
        workspace_id="workspace",
        session_id="session",
        task_id="task",
        status="ok",
        summary=report,
        structured_body={"report": report, "observations": [], "claims": []},
    ))
    core = CoreArtifactFS(store)
    host = LocalToolHost(str(tmp_path))
    host._artifacts = core
    host._contextfs.mount(
        "evidence/children",
        ArtifactContextProvider(
            core,
            kinds=("subagent",),
            canonical_mount="@sliceagent/evidence/children",
            title="CHILDREN",
        ),
    )
    try:
        seed = make_build_slice(
            state,
            host,
            None,
            NullMemory(),
            state.goal,
            session_id="session",
        )()
    finally:
        host.cleanup()

    rendered = "\n".join(str(message.get("content") or "") for message in seed)
    assert report not in rendered
    assert "# DELEGATION FAN-IN" not in rendered
    assert "HOST FAN-IN" not in rendered
