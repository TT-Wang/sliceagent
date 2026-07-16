from __future__ import annotations

import hashlib
from dataclasses import asdict
from types import SimpleNamespace

from sliceagent.active_work import ResourceRef, WorkDelta, WorkItem
from sliceagent.cli import _hydrate_workspace_tasks, _use_chitchat_fast_path
from sliceagent.context_compiler import compile_active_context, dependency_resource_paths
from sliceagent.events import ToolResult
from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
from sliceagent.intent import TurnAdmission
from sliceagent.memory import NullMemory
from sliceagent.persistence import Artifact, ArtifactStore, Checkpoint
from sliceagent.pfc import Slice, record_user
from sliceagent.runtime_persistence import CoreArtifactFS, LocalTurnStore
from sliceagent.seed import make_build_slice
from sliceagent.session import Session
from sliceagent.taskstate import slice_to_task_state
from sliceagent.tools import LocalToolHost


def test_latest_compact_receipt_is_reconstructed_from_artifacts_after_restart(tmp_path):
    state = Slice(); state.reset("task")
    task_state = slice_to_task_state(
        state, "task", session_id="session", workspace_epoch=7,
    )
    checkpoint = Checkpoint.create(
        workspace_id="workspace", session_id="session", task_id="task", generation=1,
        state=asdict(task_state), order_ns=1,
    )
    artifacts = ArtifactStore(str(tmp_path / "core"))
    artifact = Artifact.create(
        kind="turn", workspace_id="workspace", session_id="session", task_id="task",
        logical_id="logical", status="end_turn",
        structured_body={
            "meta": {"order_ns": 2},
            "turn_receipt": {
                "turn_status": "end_turn", "disposition": "completed_with_warnings",
                "counts": {"requested": 1, "execution_started": 1, "failed": 1},
                "operations": [{
                    "name": "run_command", "requested": True, "execution_started": True,
                    "settled": True, "disposition": "failed",
                }],
                "warnings": ["test failed"],
            },
        },
    )
    artifacts.put(artifact)
    store = SimpleNamespace(
        checkpoints=lambda: [checkpoint], coordinator=SimpleNamespace(artifacts=artifacts),
    )
    session = Session(NullMemory(), "session")
    _hydrate_workspace_tasks(store, session, lambda _message: None)

    restored = session.tasks["task"].continuity
    assert session.workspace_epoch == 7
    assert restored.last_receipt_artifact_id == artifact.id
    assert restored.last_receipt["counts"]["failed"] == 1
    assert restored.last_receipt["warning_count"] == 1


def test_crash_recovery_exposes_journaled_child_report_to_real_resumed_seed(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = LocalTurnStore(
        str(workspace), "session-before-crash", store_root=str(tmp_path / "core"),
    )
    active = store.begin(
        task_id="task-review", logical_id="logical-crash", user_request="Review the project.",
    )
    admission = TurnAdmission(request_text="Review the project.")
    store.record_admission({
        "action": "continue", "task_id": "task-review", "logical_turn_id": "logical-crash",
        "source_event_id": "event-crash", "source_event_text": "Review the project.",
        "workspace_epoch": 0, "admission": admission.to_dict(),
    })
    report = "REPORT-FIRST " + ("full-child-evidence " * 90) + "REPORT-LAST"
    invocation = ToolInvocation(
        "spawn-crash", "spawn_agent", {"agent": "explorer", "task": "review core"}, 0,
    )
    effect = ToolEffect("child-outcome-crash", "child_outcome", {
        "status": "succeeded", "operational_status": "succeeded", "kind": "explorer",
        "launch_ordinal": 1, "report_completion": "complete",
        "report_bytes": len(report.encode("utf-8")),
        "report_sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
    })
    text = (
        "[child 1 · explorer · succeeded]\n\nBEGIN CHILD REPORT\n"
        + report + "\nEND CHILD REPORT"
    )
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, text, (effect,))
    # This is the audited crash window: the complete ToolResult is fsynced, but the process dies before the
    # reducer transition or the parent's next model call can synthesize it.
    store.observe_event(ToolResult(
        invocation.name, dict(invocation.args), text, False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    ))
    store.close()

    recovered = LocalTurnStore(
        str(workspace), "session-after-crash", store_root=str(tmp_path / "core"),
    )
    result = recovered.recover_pending()[0]
    assert result.status == "attached"
    recovered_artifact = recovered.coordinator.artifacts.get(active.artifact_id)
    assert recovered_artifact.status == "interrupted"

    session = Session(NullMemory(), "session-after-crash")
    _hydrate_workspace_tasks(recovered, session, lambda _message: None)
    state = session.active()
    assert state.continuity.recovery_child_artifact_id == active.artifact_id
    assert state.continuity.recovery_child_report_count == 1

    record_user(
        state, "Continue and synthesize the review.", source_event_id="event-resume",
        source_text="Continue and synthesize the review.", logical_id="logical-resume",
    )

    class Ledger:
        @staticmethod
        def resolve_user_sources(ids):
            sources = {
                "event-crash": "Review the project.",
                "event-resume": "Continue and synthesize the review.",
            }
            return {event_id: sources[event_id] for event_id in ids if event_id in sources}

    host = LocalToolHost(str(workspace))
    host._artifacts = CoreArtifactFS(recovered.coordinator.artifacts)
    messages = make_build_slice(
        session, host, None, NullMemory(), "Continue and synthesize the review.",
        session.session_id, event_ledger=Ledger(), model_id="test-model",
    )()
    provider_text = "\n".join(
        str(message.get("content") or "") for message in messages if message.get("role") == "user"
    )
    locator = f'artifacts/{active.artifact_id}.md'
    assert "crash recovery: 1 returned child report" in provider_text
    assert f'read_file("{locator}")' in provider_text
    assert "HOST FAN-IN" not in provider_text
    assert "REPORT-FIRST" not in provider_text, \
        "recovery should advertise canonical evidence, not reload report bodies into every seed"

    archived = host.run("read_file", {"path": locator})
    assert report in archived, "the advertised immutable locator must retain the full child result"
    state.seal()
    assert state.continuity.recovery_child_artifact_id == "", \
        "the recovery locator is a one-turn repair seam, not recurring context"
    recovered.close()


def test_latest_seed_keeps_operational_success_and_source_coverage_separate_without_gap_prose():
    state = Slice(); state.reset("task")
    record_user(
        state, "continue", source_event_id="event", source_text="continue", logical_id="logical",
    )
    state.continuity.last_receipt = {
        "turn_status": "end_turn", "disposition": "completed",
        "counts": {"requested": 1, "execution_started": 1, "settled": 1, "succeeded": 1},
        "agents": {
            "requested": 1, "execution_started": 1, "settled": 1, "succeeded": 1,
            "source_coverage": {
                "source_complete": 0, "source_partial": 1, "source_unsupported": 0,
                "not_assessed": 0, "required_refs": 2, "covered_refs": 1,
                "source_gaps": 1,
            },
        },
        "warning_count": 0,
    }
    state.continuity.last_receipt_artifact_id = "turn-receipt"

    compiled = compile_active_context(
        state, (), source_texts={"event": "continue"}, current_logical_id="logical",
    )
    receipt = next(item for item in compiled if item.item_id == "active-receipt")
    assert "agents · 1/1 succeeded" in receipt.content
    assert "agents · 1 source partial · 1/2 granted reports covered" in receipt.content
    assert "grounded" not in receipt.content and "verified" not in receipt.content
    assert "source_gaps" not in receipt.content, "only the bounded gap count belongs in compact state"


def test_restart_restores_the_epoch_that_selects_current_workspace_resources(tmp_path):
    state = Slice(); state.reset("task")
    record_user(
        state, "inspect current workspace", source_event_id="event",
        source_text="inspect current workspace", logical_id="logical", workspace_epoch=0,
    )
    root = state.active_work.request_roots[0]
    child = WorkItem(
        id="inspect", root_id=root.id, source_refs=root.source_refs,
        description="Inspect files", status="in_progress",
        resource_refs=(
            ResourceRef("workspace_file", "a-old.py", workspace_epoch=0),
            ResourceRef("workspace_file", "b-current.py", workspace_epoch=1),
        ),
    )
    state.active_work = state.active_work.apply(WorkDelta(
        expected_revision=state.active_work.revision, creates=(child,),
    ))
    task_state = slice_to_task_state(
        state, "task", session_id="session", workspace_epoch=1,
    )
    checkpoint = Checkpoint.create(
        workspace_id="workspace-b", session_id="session", task_id="task", generation=1,
        state=asdict(task_state), order_ns=1,
    )
    artifacts = ArtifactStore(str(tmp_path / "core-epoch"))
    store = SimpleNamespace(
        checkpoints=lambda: [checkpoint], coordinator=SimpleNamespace(artifacts=artifacts),
    )
    session = Session(NullMemory(), "new-session")
    _hydrate_workspace_tasks(store, session, lambda _message: None)

    assert session.workspace_epoch == 1
    assert dependency_resource_paths(
        session.active().active_work, workspace_epoch=session.workspace_epoch,
    ) == ("b-current.py",)


def test_latest_sealed_exchange_is_rehydrated_as_the_only_restart_adjacency(tmp_path):
    state = Slice(); state.reset("task")
    task_state = slice_to_task_state(state, "task", session_id="session")
    checkpoint = Checkpoint.create(
        workspace_id="workspace", session_id="session", task_id="task", generation=1,
        state=asdict(task_state), order_ns=1,
    )
    artifacts = ArtifactStore(str(tmp_path / "core"))
    older = Artifact.create(
        kind="turn", workspace_id="workspace", session_id="session", task_id="task",
        logical_id="older", status="end_turn", brief={"request": "old user"},
        structured_body={
            "assistant": "old assistant", "assistant_provenance": "final_response",
            "meta": {"order_ns": 2},
        },
    )
    latest = Artifact.create(
        kind="turn", workspace_id="workspace", session_id="session", task_id="task",
        logical_id="latest", status="end_turn", brief={"request": "Which option should I choose?"},
        structured_body={
            "assistant": "Choose option two.", "assistant_provenance": "final_response",
            "meta": {"order_ns": 3},
        },
    )
    artifacts.put(older); artifacts.put(latest)
    store = SimpleNamespace(
        checkpoints=lambda: [checkpoint], coordinator=SimpleNamespace(artifacts=artifacts),
    )
    session = Session(NullMemory(), "new-process-session")
    _hydrate_workspace_tasks(store, session, lambda _message: None)

    assert session.tasks["task"].conversation == [{
        "user": "Which option should I choose?",
        "assistant": "Choose option two.",
        "artifact_id": latest.id,
    }]
    assert not _use_chitchat_fast_path("okay", session.tasks["task"]), \
        "a restart-hydrated assent must reach the model with its paired adjacency"
    restored = session.tasks["task"]
    record_user(
        restored, "yes", source_artifact="current-artifact", source_event_id="current-event",
        logical_id="current-logical",
    )
    compiled = compile_active_context(
        restored, (), source_texts={"current-event": "yes"},
        current_logical_id="current-logical",
    )
    adjacency = next(
        item for item in compiled
        if item.item_id == "active-adjacency" and item.fidelity.value == "full"
    )
    assert "> Which option should I choose?" in adjacency.content
    assert "> Choose option two." in adjacency.content
