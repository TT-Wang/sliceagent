"""Runtime adapter over the local seal protocol. No model, no pytest."""
import os
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.runtime_persistence import CoreArtifactFS, LocalTurnStore  # noqa: E402
from sliceagent.events import ToolResult, ToolStarted  # noqa: E402
from sliceagent.execution import (  # noqa: E402
    ToolEffect, ToolInvocation, ToolOutcome, ToolStatus,
)

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def turn_identity_is_captured_and_execution_is_journaled():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="fix it")
    invocation = ToolInvocation("call-1", "read_file", {"path": "a.py"}, 0)
    effect = ToolEffect("transition-1", "finding", {"text": "observed"})
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "bytes", (effect,))
    store.observe_event(ToolStarted("read_file", {"path": "a.py"}, invocation))
    store.observe_event(ToolResult(
        "read_file", {"path": "a.py"}, "bytes", False,
        status="succeeded", invocation_id="call-1", outcome=outcome,
    ))
    store.observe_reduction(ToolResult(
        "read_file", {"path": "a.py"}, "bytes", False,
        status="succeeded", invocation_id="call-1", outcome=outcome,
    ))
    result = store.seal(
        state={"intent": "fix it"}, record={"trajectory": ["done"]}, status="end_turn",
        title="fix", summary="done", files=("a.py",),
    )
    artifact = store.coordinator.artifacts.get(active.artifact_id)
    checkpoint = store.coordinator.checkpoints.load(store.workspace_id, "task-A")
    assert result.artifact_id == artifact.id and artifact.task_id == "task-A"
    assert checkpoint.artifact_refs == (artifact.id,)
    assert checkpoint.applied_transition_ids == ("transition-1",)


@check
def transitions_are_not_marked_applied_before_reduction():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(workspace, "session-1", store_root=tempfile.mkdtemp(prefix="core-store-"))
    store.begin(task_id="task-A", logical_id="turn-1", user_request="fix it")
    invocation = ToolInvocation("call-1", "edit_file", {"path": "a.py"}, 0)
    outcome = ToolOutcome(
        invocation, ToolStatus.SUCCEEDED, "wrote", (ToolEffect("effect-1", "edit", {"path": "a.py"}),),
    )
    event = ToolResult("edit_file", {"path": "a.py"}, "wrote", False, outcome=outcome)
    store.observe_event(event)
    result = store.seal(state={"edited": False}, record={}, status="error")
    checkpoint = store.coordinator.checkpoints.load(store.workspace_id, "task-A")
    assert result.artifact_id and checkpoint.applied_transition_ids == ()


@check
def local_durability_does_not_import_semantic_memory():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    old = sys.modules.get("memem", "absent")
    sys.modules["memem"] = None
    try:
        store = LocalTurnStore(workspace, "session-1", store_root=store_root)
        active = store.begin(task_id="task-A", logical_id="turn-1", user_request="hello")
        store.seal(state={"current": True}, record={"messages": []}, status="end_turn")
        store.close()
        reopened = LocalTurnStore(workspace, "session-1", store_root=store_root)
        assert reopened.coordinator.artifacts.get(active.artifact_id).status == "end_turn"
    finally:
        if old == "absent":
            sys.modules.pop("memem", None)
        else:
            sys.modules["memem"] = old


@check
def durable_records_are_redacted_before_storage():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(workspace, "session-1", store_root=tempfile.mkdtemp(prefix="core-store-"))
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="token sk-test-secret")
    store.record_invocation("call-1", name="run", args={"token": "sk-test-secret"})
    store.record_outcome("call-1", status="failed", text="sk-test-secret")
    store.seal(state={"token": "sk-test-secret"}, record={"token": "sk-test-secret"}, status="failed")
    artifact = store.coordinator.artifacts.get(active.artifact_id)
    assert "sk-test-secret" not in str(artifact.to_dict())


@check
def active_turn_cannot_be_overwritten_in_memory():
    store = LocalTurnStore(tempfile.mkdtemp(), "session-1", store_root=tempfile.mkdtemp())
    store.begin(task_id="task-A", logical_id="turn-1", user_request="one")
    try:
        store.begin(task_id="task-B", logical_id="turn-2", user_request="two")
        assert False, "starting another turn must not replace the recovery journal"
    except RuntimeError as exc:
        assert "already active" in str(exc)


@check
def failed_seal_keeps_turn_owned_for_retry():
    from sliceagent.persistence import SealCoordinator

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")

    def crash(stage):
        if stage == "artifact":
            raise OSError("simulated crash")

    store = LocalTurnStore(
        workspace, "session-1", store_root=store_root,
        coordinator=SealCoordinator(store_root, on_stage=crash),
    )
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="one")
    try:
        store.seal(state={"ready": True}, record={}, status="end_turn")
        assert False, "injected seal crash must surface"
    except OSError:
        pass
    assert store.active == active
    try:
        store.begin(task_id="task-A", logical_id="turn-2", user_request="two")
        assert False, "a failed prior seal must retain generation ownership"
    except RuntimeError:
        pass


@check
def failed_prepared_seal_freezes_child_reference_set_through_recovery():
    from sliceagent.persistence import SealCoordinator

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    crashed = {"once": False}

    def crash_once(stage):
        if stage == "artifact" and not crashed["once"]:
            crashed["once"] = True
            raise OSError("simulated crash after seal intent was prepared")

    store = LocalTurnStore(
        workspace, "session-failed-seal-child", store_root=store_root,
        coordinator=SealCoordinator(store_root, on_stage=crash_once),
    )
    active = store.begin(task_id="task-A", logical_id="turn-A", user_request="inspect A")
    publish = store.bind_artifact_ref_sink(task_id=active.task_id, parent_id=active.artifact_id)
    child = _linearization_child(store, active, "subagent-after-failed-seal")
    try:
        store.seal(state={}, record={}, status="end_turn")
        assert False, "injected prepared-seal crash must surface"
    except OSError:
        pass
    assert active.journal.snapshot().seal_intent is not None
    try:
        publish.commit_artifact(store.coordinator.artifacts, child)
        assert False, "a frozen prepared checkpoint must reject a late child commit"
    except RuntimeError as error:
        assert "prepared seal" in str(error)
    assert not store.coordinator.artifacts.exists(child.id)

    recovered = store.recover_active_seal()
    assert recovered is not None and store.active is None
    parent = store.coordinator.artifacts.get(active.artifact_id)
    checkpoint = store.coordinator.checkpoints.load(store.workspace_id, active.task_id)
    assert child.id not in parent.refs and child.id not in checkpoint.artifact_refs
    store.close()


@check
def seal_cannot_publish_an_unmatched_started_invocation_as_clean():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(workspace, "session-1", store_root=tempfile.mkdtemp(prefix="core-store-"))
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="run it")
    store.record_invocation("call-interrupted", name="run_command", args={"command": "deploy"})
    store.seal(
        state={"status": "parked", "reconciliation_required": "", "reconciliation_targets": []},
        record={}, status="aborted",
    )
    checkpoint = store.coordinator.checkpoints.load(store.workspace_id, "task-A")
    artifact = store.coordinator.artifacts.get(active.artifact_id)
    assert checkpoint.state["status"] == "indeterminate"
    assert checkpoint.state["reconciliation_targets"] == ("workspace:*", "opaque:run_command")
    assert "call-interrupted" in checkpoint.state["reconciliation_required"]
    assert artifact.status == "indeterminate" and artifact.uncertainty


@check
def exclusive_workspace_lease_blocks_a_second_live_owner():
    from sliceagent.persistence import LeaseBusyError

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    first = LocalTurnStore(workspace, "session-1", store_root=store_root)
    try:
        try:
            LocalTurnStore(workspace, "session-2", store_root=store_root)
            assert False, "a second live owner must not recover or append this workspace's journals"
        except LeaseBusyError:
            pass
    finally:
        first.close()
    third = LocalTurnStore(workspace, "session-3", store_root=store_root)
    try:
        first.begin(task_id="task-A", logical_id="late", user_request="must fail")
        assert False, "a closed former owner must not write after another process acquires the lease"
    except RuntimeError as exc:
        assert "closed" in str(exc)
    third.close()


@check
def workspace_lease_excludes_another_process_and_releases_on_process_death():
    from sliceagent.persistence import LeaseBusyError

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    source_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src"))
    script = (
        "import sys,time; "
        "from sliceagent.runtime_persistence import LocalTurnStore; "
        "store=LocalTurnStore(sys.argv[1], 'child', store_root=sys.argv[2]); "
        "print('READY', flush=True); time.sleep(30)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", script, workspace, store_root],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        assert child.stdout.readline().strip() == "READY"
        try:
            LocalTurnStore(workspace, "parent", store_root=store_root)
            assert False, "the OS lease must exclude a writer in another process"
        except LeaseBusyError:
            pass
    finally:
        child.terminate()
        child.wait(timeout=5)
    reopened = LocalTurnStore(workspace, "parent", store_root=store_root)
    reopened.close()


@check
def checkpoints_are_discoverable_without_semantic_memory():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    first = LocalTurnStore(workspace, "session-1", store_root=store_root)
    first.begin(task_id="task-A", logical_id="turn-1", user_request="one")
    first.seal(state={"task_id": "task-A"}, record={}, status="end_turn")
    first.close()
    reopened = LocalTurnStore(workspace, "session-2", store_root=store_root)
    checkpoints = reopened.checkpoints()
    assert len(checkpoints) == 1 and checkpoints[0].task_id == "task-A"


@check
def corrupt_authoritative_checkpoint_is_a_startup_conflict_not_a_missing_task():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    store.close()
    path = os.path.join(store_root, "checkpoints", "damaged.json")
    with open(path, "wb") as stream:
        stream.write(b"not-json\n")
    reopened = LocalTurnStore(workspace, "session-2", store_root=store_root)
    try:
        try:
            reopened.checkpoints()
            assert False, "checkpoint discovery must not silently skip damaged authoritative state"
        except Exception as exc:  # exact type lives in the persistence layer; startup reports it generically
            assert type(exc).__name__ == "CheckpointCorruptError"
    finally:
        reopened.close()


@check
def live_checkpoint_artifact_refs_are_revalidated_on_startup():
    from sliceagent.persistence import ArtifactNotFoundError

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    first = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = first.begin(task_id="task-A", logical_id="turn-1", user_request="one")
    first.seal(state={"task_id": "task-A"}, record={}, status="end_turn")
    artifact_path = first.coordinator.artifacts.path_for(active.artifact_id)
    first.close()
    os.unlink(artifact_path)

    reopened = LocalTurnStore(workspace, "session-2", store_root=store_root)
    try:
        try:
            reopened.checkpoints()
            assert False, "a dangling live artifact reference must fail closed"
        except ArtifactNotFoundError:
            pass
    finally:
        reopened.close()


@check
def corrupt_live_checkpoint_artifact_is_a_startup_conflict():
    from sliceagent.persistence import ArtifactCorruptError

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    first = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = first.begin(task_id="task-A", logical_id="turn-1", user_request="one")
    first.seal(state={"task_id": "task-A"}, record={}, status="end_turn")
    artifact_path = first.coordinator.artifacts.path_for(active.artifact_id)
    first.close()
    with open(artifact_path, "wb") as stream:
        stream.write(b"not-json\n")

    reopened = LocalTurnStore(workspace, "session-2", store_root=store_root)
    try:
        try:
            reopened.checkpoints()
            assert False, "a corrupt live artifact dependency must fail closed"
        except ArtifactCorruptError:
            pass
    finally:
        reopened.close()


@check
def unprepared_recovery_replays_confirmed_semantic_state_without_rerunning_tool():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="set x")
    invocation = ToolInvocation("call-1", "world_set", {"key": "x", "value": "1"}, 0)
    effect = ToolEffect("effect-1", "tool_outcome", {"name": "world_set", "status": "succeeded"})
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "WORLD[x] set", (effect,))
    event = ToolResult(
        "world_set", dict(invocation.args), outcome.text, False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    )
    store.observe_event(ToolStarted("world_set", dict(invocation.args), invocation))
    store.observe_event(event)
    store.observe_reduction(event)

    store.close()  # simulate process death: journal remains, writer ownership does not
    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    results = recovered.recover_pending()
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-A")
    artifact = recovered.coordinator.artifacts.get(active.artifact_id)
    assert results[0].status == "attached" and artifact.status == "interrupted"
    assert checkpoint.state["world"] == {"x": "1"}
    assert checkpoint.applied_transition_ids == ("effect-1",)
    assert checkpoint.artifact_refs == (active.artifact_id,)


@check
def checkpoint_resume_deep_thaws_typed_state_and_memory_mirror():
    from dataclasses import asdict

    from sliceagent.memory import _now_iso, _render_task_md
    from sliceagent.pfc import Slice, record_user
    from sliceagent.taskstate import (slice_to_task_state, task_state_from_checkpoint,
                                      task_state_to_slice)

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = store.begin(
        task_id="task-intent", logical_id="turn-1",
        user_request="You must never log passwords.",
    )
    state = Slice(); state.reset("Repair authentication safely")
    record_user(
        state, "You must never log passwords.", source_artifact=active.artifact_id,
    )
    state.intent.mark_provisional(
        "You must never log passwords.", evidence_refs=("invocation:read-auth",),
    )
    state.task.add_progress("read", "Inspected src/auth.py")
    state.task.add_progress("read", "Inspected src/auth.py")
    expected_entry = state.intent.entries[0]
    store.seal(
        state=asdict(slice_to_task_state(
            state, "task-intent", session_id="session-1", status="active",
        )),
        record={}, status="end_turn",
    )
    store.close()

    reopened = LocalTurnStore(workspace, "session-2", store_root=store_root)
    checkpoint = reopened.checkpoints()[0]
    mutable = checkpoint.thawed_state()
    assert isinstance(mutable["intent_entries"], list)
    assert isinstance(mutable["intent_entries"][0], dict)
    assert isinstance(mutable["progress_signals"], list)
    assert isinstance(mutable["progress_signals"][0], dict)

    task_state = task_state_from_checkpoint(checkpoint)
    restored = task_state_to_slice(task_state)
    assert restored.intent.entries == [expected_entry]
    assert len(restored.task.progress_signals) == 1
    assert restored.task.progress_signals[0].detail == "Inspected src/auth.py"
    assert restored.task.progress_signals[0].count == 2

    # The committed-state merge passes this TaskState to the optional markdown mirror. Frozen nested mappings
    # used to make json.dumps fail here even though the core checkpoint itself had committed successfully.
    rendered = _render_task_md(task_state, created=_now_iso(), updated=_now_iso())
    assert "You must never log passwords." in rendered
    assert "Inspected src/auth.py" in rendered
    reopened.close()


@check
def unprepared_crash_recovery_preserves_base_intent_and_progress():
    from dataclasses import asdict

    from sliceagent.pfc import Slice, record_user
    from sliceagent.taskstate import (slice_to_task_state, task_state_from_checkpoint,
                                      task_state_to_slice)

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    base_turn = store.begin(
        task_id="task-intent", logical_id="turn-base",
        user_request="Only modify src/auth.py.",
    )
    state = Slice(); state.reset("Repair authentication safely")
    record_user(state, "Only modify src/auth.py.", source_artifact=base_turn.artifact_id)
    state.task.add_progress("read", "Inspected src/auth.py")
    expected_entry = state.intent.entries[0]
    store.seal(
        state=asdict(slice_to_task_state(
            state, "task-intent", session_id="session-1", status="active",
        )),
        record={}, status="end_turn",
    )

    # Leave a journal without a seal intent, as if the process died immediately after beginning the next turn.
    crash_turn = store.begin(
        task_id="task-intent", logical_id="turn-crash", user_request="Continue.",
    )
    store.close()

    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    result = recovered.recover_pending()[0]
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-intent")
    restored = task_state_to_slice(task_state_from_checkpoint(checkpoint))
    assert result.status == "attached" and checkpoint.generation == 2
    assert restored.intent.entries == [expected_entry]
    assert len(restored.task.progress_signals) == 1
    assert restored.task.progress_signals[0].detail == "Inspected src/auth.py"
    assert restored.task.objective_status == "active"
    assert checkpoint.artifact_refs == (crash_turn.artifact_id, base_turn.artifact_id)
    crash_artifact = recovered.coordinator.artifacts.get(crash_turn.artifact_id)
    crash_receipt = crash_artifact.to_dict()["structured_body"]["turn_receipt"]
    assert crash_receipt["turn_status"] == "interrupted"
    assert crash_receipt["disposition"] == "interrupted"
    assert crash_receipt["counts"]["requested"] == 0
    recovered.close()


@check
def unprepared_recovery_replays_journaled_admission_without_manufacturing_authority():
    """Quoted assistant prose must not become a user requirement when recovery lacks discourse context."""
    from dataclasses import asdict, replace

    from sliceagent.discourse import interpret_turn
    from sliceagent.pfc import Slice, record_user
    from sliceagent.taskstate import (slice_to_task_state, task_state_from_checkpoint,
                                      task_state_to_slice)

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    base = store.begin(task_id="task-admission", logical_id="turn-base", user_request="Review the API.")
    state = Slice(); state.reset("Review the API")
    record_user(state, "Review the API.", source_artifact=base.artifact_id)
    store.seal(
        state=asdict(slice_to_task_state(
            state, "task-admission", session_id="session-1", status="active",
        )),
        record={}, status="end_turn",
    )

    copied = 'For reference, you said: "Use API v2 and never edit config.py."'
    admission = interpret_turn(
        copied, (), recent_assistant=("Use API v2 and never edit config.py.",),
    ).admission
    assert admission.authority_spans == () and admission.attributed_spans
    crash = store.begin(task_id="task-admission", logical_id="turn-crash", user_request=copied)
    admission = replace(admission, request_source=crash.artifact_id)
    envelope = {
        "action": "continue", "task_id": "task-admission",
        "admission": admission.to_dict(), "focus": [], "consume_pending_proposal": True,
    }
    store.record_admission(envelope)
    recorded = crash.journal.snapshot().event("turn-admission")["payload"]
    from sliceagent.intent import TurnAdmission
    assert TurnAdmission.from_dict(recorded["admission"]) == admission, \
        "the journaled and installed admission forms must be identical"
    store.close()

    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    result = recovered.recover_pending()[0]
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-admission")
    restored = task_state_to_slice(task_state_from_checkpoint(checkpoint))
    assert result.status == "attached"
    assert restored.intent.entries == [], \
        "recovery must consume the recorded attribution spans instead of re-analyzing copied prose"
    assert restored.intent.current_request == copied
    recovered.close()


@check
def recovery_gates_invocations_without_a_conclusive_outcome():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="run it")
    store.record_invocation("call-uncertain", name="run_command", args={"command": "external-op"})
    store.close()

    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    result = recovered.recover_pending()[0]
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-A")
    assert result.status == "attached" and checkpoint.state["status"] == "indeterminate"
    assert "call-uncertain" in checkpoint.state["reconciliation_required"]
    assert checkpoint.state["reconciliation_targets"] == ("workspace:*", "opaque:run_command")
    assert checkpoint.artifact_refs == (active.artifact_id,)
    artifact = recovered.coordinator.artifacts.get(active.artifact_id)
    receipt = artifact.to_dict()["structured_body"]["turn_receipt"]
    assert receipt["disposition"] == "indeterminate"
    assert receipt["operations"][0]["execution_started"] is True
    assert receipt["operations"][0]["settled"] is False


@check
def recovery_treats_a_statusless_outcome_as_unresolved():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="edit")
    store.record_invocation("call-statusless", name="edit_file", args={"path": "critical.py"})
    active.journal.record_outcome("call-statusless", {})
    store.close()

    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    assert recovered.recover_pending()[0].status == "attached"
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-A")
    assert checkpoint.state["status"] == "indeterminate"
    assert checkpoint.state["reconciliation_targets"] == ("path:critical.py",)


@check
def recovery_treats_a_non_scalar_status_as_unresolved():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="edit")
    store.record_invocation("call-invalid", name="edit_file", args={"path": "critical.py"})
    active.journal.record_outcome("call-invalid", {"status": {}})
    store.close()

    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    assert recovered.recover_pending()[0].status == "attached"
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-A")
    assert checkpoint.state["status"] == "indeterminate"
    assert checkpoint.state["reconciliation_targets"] == ("path:critical.py",)


@check
def recovery_salvages_only_a_torn_final_append_and_gates_the_invocation():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="edit")
    store.record_invocation("call-torn", name="edit_file", args={"path": "critical.py"})
    with open(active.journal.path, "ab") as stream:
        stream.write(b'{"v":1,"seq":2,"type":"tool-outcome"')
        stream.flush(); os.fsync(stream.fileno())
    store.close()

    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    result = recovered.recover_pending()[0]
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-A")
    assert result.status == "attached" and checkpoint.state["status"] == "indeterminate"
    assert checkpoint.state["reconciliation_targets"] == ("path:critical.py",)
    assert os.path.isfile(active.journal.path + ".torn")
    assert not os.path.exists(active.journal.path)


@check
def later_crash_uncertainty_survives_replayed_reconciliation():
    from dataclasses import asdict
    from sliceagent.pfc import Slice
    from sliceagent.taskstate import slice_to_task_state

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    state = Slice(); state.reset("repair")
    state.reconciliation_required = "an earlier edit may still write a.py"
    state.reconciliation_targets = ["path:a.py"]
    base = asdict(slice_to_task_state(
        state, "task-A", session_id="session-1", status="indeterminate",
    ))
    store.begin(task_id="task-A", logical_id="turn-base", user_request="repair")
    store.seal(state=base, record={}, status="indeterminate")

    store.begin(task_id="task-A", logical_id="turn-reconcile", user_request="reconcile")
    invocation = ToolInvocation(
        "call-reconcile", "reconcile_execution", {"resolution": "a.py is settled"}, 0,
    )
    effect = ToolEffect("effect-reconcile", "tool_outcome", {
        "name": "reconcile_execution", "status": "succeeded",
    })
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "reconciled", (effect,))
    event = ToolResult(
        invocation.name, dict(invocation.args), outcome.text, False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    )
    store.observe_event(event)
    store.observe_reduction(event)
    store.record_invocation("call-new-edit", name="edit_file", args={"path": "b.py"})
    store.close()

    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    assert recovered.recover_pending()[0].status == "attached"
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-A")
    assert checkpoint.state["status"] == "indeterminate"
    assert "call-new-edit" in checkpoint.state["reconciliation_required"]
    assert checkpoint.state["reconciliation_targets"] == ("path:b.py",)


@check
def partial_multi_effect_journal_never_claims_a_rolled_back_transition():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(workspace, "session-1", store_root=tempfile.mkdtemp(prefix="core-store-"))
    store.begin(task_id="task-A", logical_id="turn-1", user_request="one")
    invocation = ToolInvocation("call-1", "world_set", {"key": "x", "value": "1"}, 0)
    effects = (
        ToolEffect("effect-1", "first", {}), ToolEffect("effect-2", "second", {}),
    )
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "ok", effects)
    event = ToolResult("world_set", dict(invocation.args), "ok", False, outcome=outcome)
    store.observe_event(event)
    store.record_transition("effect-1", {"kind": "first"})
    store.seal(state={"world": {}}, record={}, status="error")
    checkpoint = store.coordinator.checkpoints.load(store.workspace_id, "task-A")
    assert checkpoint.applied_transition_ids == ()


@check
def crash_between_child_put_and_parent_ref_recovers_downward_dependency():
    from sliceagent.persistence import Artifact

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    parent = LocalTurnStore(workspace, "session-1", store_root=store_root)
    active = parent.begin(task_id="task-A", logical_id="turn-1", user_request="delegate")
    child = Artifact(
        id="subagent-child-window", kind="subagent", workspace_id=parent.workspace_id,
        session_id="session-1", task_id="task-A", parent_id=active.artifact_id,
    )
    parent.coordinator.artifacts.put(child)  # hard crash before record_artifact_ref(child.id)

    parent.close()
    recovered = LocalTurnStore(workspace, "session-2", store_root=store_root)
    assert recovered.recover_pending()[0].status == "attached"
    artifact = recovered.coordinator.artifacts.get(active.artifact_id)
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-A")
    assert artifact.refs == (child.id,)
    assert checkpoint.artifact_refs == (active.artifact_id, child.id)


@check
def crash_replay_preserves_child_artifact_active_work_binding():
    from dataclasses import asdict

    from sliceagent.active_work import EvidenceRef, ResourceRef, WorkDelta, WorkGraph, WorkItem
    from sliceagent.intent import TurnAdmission
    from sliceagent.persistence import Artifact
    from sliceagent.pfc import Slice, record_user
    from sliceagent.taskstate import slice_to_task_state

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-old", store_root=store_root)

    state = Slice(); state.reset("delegate")
    record_user(
        state, "delegate", source_event_id="event-old", source_text="delegate",
        logical_id="logical-base",
    )
    root = state.active_work.request_roots[0]
    child = WorkItem(
        id="review", root_id=root.id, source_refs=root.source_refs,
        description="Review the parser", status="in_progress",
    )
    state.active_work = state.active_work.apply(WorkDelta(
        expected_revision=state.active_work.revision, creates=(child,),
    ))
    store.begin(task_id="task-A", logical_id="logical-base", user_request="delegate")
    store.seal(
        state=asdict(slice_to_task_state(state, "task-A", session_id="session-old")),
        record={}, status="end_turn",
    )

    active = store.begin(
        task_id="task-A", logical_id="logical-crash", user_request="continue",
    )
    admission = TurnAdmission(request_text="continue")
    store.record_admission({
        "action": "continue", "task_id": "task-A", "logical_turn_id": "logical-crash",
        "source_event_id": "event-new", "source_event_text": "continue", "workspace_epoch": 0,
        "admission": admission.to_dict(),
    })
    artifact = Artifact(
        id="child-artifact", kind="subagent", workspace_id=store.workspace_id,
        session_id="session-old", task_id="task-A", parent_id=active.artifact_id,
    )
    store.coordinator.artifacts.put(artifact)
    invocation = ToolInvocation(
        "spawn-1", "spawn_agent", {"task": "review", "work_item_id": "review"}, 0,
    )
    effect = ToolEffect(
        "child-effect", "child_artifact",
        {
            "artifact_id": artifact.id, "work_item_id": "review",
            "source_coverage_status": "source_partial",
            "required_ref_count": 2, "consumed_refs": ["subagents/sub-1.md"],
            "cited_refs": ["subagents/sub-1.md"], "covered_refs": ["subagents/sub-1.md"],
            "source_gaps": ["raw source gap must remain in the sealed child artifact"],
        },
    )
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "child returned", (effect,))
    result = ToolResult(
        "spawn_agent", dict(invocation.args), outcome.text, False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    )
    store.observe_event(result)
    store.observe_reduction(result)
    store.close()  # crash before the normal turn seal attaches child testimony to Active Work

    recovered = LocalTurnStore(workspace, "session-new", store_root=store_root)
    assert recovered.recover_pending()[0].status == "attached"
    checkpoint = recovered.coordinator.checkpoints.load(recovered.workspace_id, "task-A")
    graph = WorkGraph.from_records(checkpoint.thawed_state()["active_work"])
    restored = graph.get("review")
    assert restored.evidence_refs == (
        EvidenceRef("child_artifact", artifact.id, qualifier="source_partial"),
        EvidenceRef("child_digest_delivered", artifact.id),
    )
    assert restored.resource_refs == (ResourceRef("subagent", artifact.id, workspace_epoch=0),)
    assert artifact.id in checkpoint.artifact_refs
    recovered.close()


@check
def child_and_source_artifact_refs_are_retained_by_parent_checkpoint():
    from sliceagent.persistence import Artifact

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(workspace, "session-1", store_root=tempfile.mkdtemp(prefix="core-store-"))
    for artifact_id, kind in (("subagent-child-1", "subagent"), ("turn-source-1", "turn")):
        store.coordinator.artifacts.put(Artifact(
            id=artifact_id, kind=kind, workspace_id=store.workspace_id,
            session_id="session-1", task_id="task-A",
        ))
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="delegate")
    store.record_artifact_ref("subagent-child-1")
    store.record_artifact_ref("subagent-child-1")
    store.seal(state={"task_id": "task-A"}, record={}, status="end_turn",
               refs=("turn-source-1", "subagent-child-1"))
    checkpoint = store.coordinator.checkpoints.load(store.workspace_id, "task-A")
    assert checkpoint.artifact_refs == (
        active.artifact_id, "subagent-child-1", "turn-source-1",
    )


@check
def authoritative_artifacts_have_timestamp_and_readable_virtual_handle():
    from sliceagent.tools import LocalToolHost
    from sliceagent.discourse import extract_addressable_anchors

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(workspace, "session-1", store_root=tempfile.mkdtemp(prefix="core-store-"))
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="one")
    assistant = "## Findings\n1. first\n2. exact archived detail"
    anchors = [item.to_dict() for item in extract_addressable_anchors(assistant)]
    store.seal(state={"task_id": "task-A"},
               record={"markdown": "exact archived detail", "assistant": assistant, "anchors": anchors},
               status="end_turn", title="Turn one")
    artifact = store.coordinator.artifacts.get(active.artifact_id)
    assert artifact.timestamp
    host = LocalToolHost(workspace)
    host._artifacts = CoreArtifactFS(store.coordinator.artifacts)
    rendered = host.run("read_file", {"path": f"artifacts/{artifact.id}.md"})
    assert "exact archived detail" in rendered and artifact.id in rendered
    assert "## User request (verbatim)\none" in rendered
    assert "Findings #2" in rendered, "artifact view exposes stable ordinal anchors"
    assert f"{artifact.id}.md" in host.run("list_files", {"path": "artifacts"})


@check
def corrupt_journal_does_not_block_later_valid_recovery():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    valid = store.begin(task_id="task-A", logical_id="z-valid", user_request="valid")
    corrupt_path = os.path.join(store_root, "journals", "a-corrupt.jsonl")
    with open(corrupt_path, "wb") as stream:
        stream.write(b"not-json\n")
    store.close()
    results = LocalTurnStore(workspace, "session-2", store_root=store_root).recover_pending()
    by_id = {result.artifact_id: result for result in results}
    assert by_id["a-corrupt"].status == "conflict"
    assert by_id[valid.artifact_id].status == "attached"
    assert not os.path.exists(valid.journal.path) and os.path.exists(corrupt_path)


@check
def malformed_semantic_journal_does_not_block_later_valid_recovery():
    from sliceagent.persistence import PendingTurnJournal

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    store = LocalTurnStore(workspace, "session-1", store_root=store_root)
    common = {
        "workspace_id": store.workspace_id, "session_id": "session-1",
        "task_id": "task-A", "base_generation": 0,
    }
    malformed = PendingTurnJournal.begin(
        store_root, artifact_id="turn-a-malformed", user_request="bad", **common,
    )
    malformed.record_invocation("call-1", name="world_set", args="not-a-map")
    malformed.record_outcome("call-1", {
        "status": "succeeded", "text": "set", "effects": [
            {"id": "effect-1", "kind": "tool_outcome",
             "payload": {"name": "world_set", "status": "succeeded"}},
        ],
    })
    malformed.record_transition("effect-1", {"kind": "tool_outcome"})
    valid = PendingTurnJournal.begin(
        store_root, artifact_id="turn-z-valid", user_request="valid", **common,
    )

    store.close()
    results = LocalTurnStore(workspace, "session-2", store_root=store_root).recover_pending()
    by_id = {result.artifact_id: result for result in results}
    assert by_id["turn-a-malformed"].status == "conflict"
    assert by_id["turn-z-valid"].status == "attached"
    assert os.path.exists(malformed.path) and not os.path.exists(valid.path)


@check
def durable_order_is_monotonic_and_shared_by_artifact_and_checkpoint():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(
        workspace, "session-order", store_root=tempfile.mkdtemp(prefix="core-store-"),
    )
    first = store.begin(task_id="task-order", logical_id="turn-z", user_request="first")
    store.seal(state={"n": 1}, record={}, status="end_turn")
    second = store.begin(task_id="task-order", logical_id="turn-a", user_request="second")
    store.seal(state={"n": 2}, record={}, status="end_turn")
    artifacts = store.coordinator.artifacts.list_all()
    assert [artifact.id for artifact in artifacts] == [first.artifact_id, second.artifact_id]
    orders = [artifact.structured_body["meta"]["order_ns"] for artifact in artifacts]
    checkpoint = store.coordinator.checkpoints.load(store.workspace_id, "task-order")
    assert orders[0] < orders[1] == checkpoint.order_ns
    store.close()


@check
def first_monotonic_order_advances_past_future_dated_legacy_artifacts():
    from sliceagent.persistence import Artifact, artifact_order_key

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(
        workspace, "session-legacy-floor", store_root=tempfile.mkdtemp(prefix="core-store-"),
    )
    legacy = Artifact(
        id="turn-future-legacy", kind="turn", workspace_id=store.workspace_id,
        session_id=store.session_id, task_id="task-floor", timestamp="2099-01-01T00:00:00Z",
    )
    store.coordinator.artifacts.put(legacy)
    active = store.begin(task_id="task-floor", logical_id="new", user_request="new")
    store.seal(state={}, record={}, status="end_turn")
    current = store.coordinator.artifacts.get(active.artifact_id)
    assert artifact_order_key(current)[0] > artifact_order_key(legacy)[0]
    assert store.coordinator.artifacts.list_all()[-1].id == current.id
    store.close()


@check
def retrying_an_existing_begin_keeps_the_fsynced_order():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    first_store = LocalTurnStore(workspace, "session-retry", store_root=store_root)
    first = first_store.begin(task_id="task-retry", logical_id="same-turn", user_request="do it")
    first_order = first.journal.snapshot().order_ns
    first_store.close()
    retry_store = LocalTurnStore(workspace, "session-retry", store_root=store_root)
    retry = retry_store.begin(task_id="task-retry", logical_id="same-turn", user_request="do it")
    assert retry.artifact_id == first.artifact_id
    assert retry.journal.snapshot().order_ns == first_order
    retry_store.close()


@check
def seal_rejects_divergent_journal_artifact_checkpoint_order():
    from sliceagent.persistence import Artifact, Checkpoint, InvalidRecordError

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(
        workspace, "session-mismatch", store_root=tempfile.mkdtemp(prefix="core-store-"),
    )
    active = store.begin(task_id="task-mismatch", logical_id="turn-1", user_request="x")
    snapshot = active.journal.snapshot()
    artifact = Artifact(
        id=active.artifact_id, kind="turn", workspace_id=store.workspace_id,
        session_id=store.session_id, task_id="task-mismatch", status="end_turn",
        structured_body={"meta": {"order_ns": snapshot.order_ns + 1}},
    )
    checkpoint = Checkpoint.create(
        workspace_id=store.workspace_id, session_id=store.session_id,
        task_id="task-mismatch", generation=1, state={}, artifact_refs=(artifact.id,),
        order_ns=snapshot.order_ns,
    )
    try:
        store.coordinator.seal(active.journal, artifact, checkpoint)
    except InvalidRecordError as exc:
        assert "publication order" in str(exc)
    else:
        raise AssertionError("a split ordering fact must not seal")
    store.close()


@check
def corrupt_artifact_listing_degrades_execution_coverage():
    from sliceagent.discourse import interpret_turn

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(
        workspace, "session-gap", store_root=tempfile.mkdtemp(prefix="core-store-"),
    )
    store.begin(task_id="task-gap", logical_id="valid", user_request="inspect")
    store.seal(state={}, record={}, status="end_turn")
    corrupt_path = store.coordinator.artifacts.path_for("turn-corrupt")
    with open(corrupt_path, "w", encoding="utf-8") as stream:
        stream.write("{not json")
    listing = store.coordinator.artifacts.list_all()
    assert len(listing) == 1 and [gap.artifact_id for gap in listing.gaps] == ["turn-corrupt"]
    preview = interpret_turn("How many commands actually ran?", listing, task_id="task-gap")
    coverage = next(
        ref for ref in preview.admission.referents
        if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_coverage"
    )
    assert coverage["coverage"] == "partial"
    assert coverage["corrupt_artifact_count"] == 1
    assert coverage["corrupt_artifact_sample"] == ["turn-corrupt"]
    virtual = CoreArtifactFS(store.coordinator.artifacts)
    index = virtual.index()
    assert "Unreadable artifact records" in index and "turn-corrupt" in index
    assert "turn-corrupt.md" not in virtual.listing(), "a corrupt record is not a readable virtual doc"
    store.close()

    from sliceagent.persistence import ArtifactStore
    corrupt_only = ArtifactStore(tempfile.mkdtemp(prefix="artifact-gap-only-"))
    with open(corrupt_only.path_for("turn-only-corrupt"), "w", encoding="utf-8") as stream:
        stream.write("not-json")
    only_index = CoreArtifactFS(corrupt_only).index()
    assert "turn-only-corrupt" in only_index and "(none yet)" not in only_index
    assert "no readable artifacts" in only_index


@check
def child_evidence_is_losslessly_recoverable_through_fixed_size_virtual_pages():
    import hashlib
    from sliceagent.persistence import Artifact, ArtifactStore

    root = tempfile.mkdtemp(prefix="child-evidence-pages-")
    store = ArtifactStore(root)
    view = ("αβγ\n" * 10_000) + "MIDDLE-SENTINEL\n" + ("tail\n" * 6_000)
    body = view.encode("utf-8")
    artifact = Artifact(
        id="subagent-page-backed", kind="subagent", workspace_id="workspace",
        session_id="session", task_id="task", status="ok", title="page-backed child",
        structured_body={
            "report": "Full child conclusion.",
            "observations": [{
                "v": 1, "tool": "read_file", "args": {
                    "path": "src/large.py", "pattern": "```adversarial```" * 500,
                },
                "status": "succeeded", "view": view,
                "raw_sha256": hashlib.sha256(body).hexdigest(),
                "view_sha256": hashlib.sha256(body).hexdigest(),
                "raw_bytes": len(body), "view_bytes": len(body),
                "redacted": False, "truncated": False,
            }],
        },
    )
    store.put(artifact)
    virtual = CoreArtifactFS(store)
    report = virtual.read_file("artifacts/subagent-page-backed.md")
    assert "Full child conclusion." in report and "MIDDLE-SENTINEL" not in report
    index = virtual.read_file("artifacts/subagent-page-backed/evidence/index.md")
    assert "args display omitted" in index and len(index) < 10_000
    assert artifact.structured_body["observations"][0]["args"]["pattern"].endswith(
        "```adversarial```"
    ), "display bounding must not modify canonical args"
    names = [
        row for row in virtual.listing("artifacts/subagent-page-backed/evidence").splitlines()
        if row.startswith("obs-")
    ]
    assert len(names) > 1 and all(name in index for name in names)
    reconstructed = "".join(
        virtual.read_file(f"artifacts/subagent-page-backed/evidence/{name}").split(
            "## Exact retained tool output\n", 1,
        )[1]
        for name in names
    )
    assert reconstructed == view
    assert hashlib.sha256(reconstructed.encode("utf-8")).hexdigest() == hashlib.sha256(body).hexdigest()


@check
def oversized_child_report_is_losslessly_recoverable_through_deterministic_pages():
    import hashlib
    from sliceagent.persistence import Artifact, ArtifactStore

    root = tempfile.mkdtemp(prefix="child-report-pages-")
    store = ArtifactStore(root)
    report = ("αβ finding with qualifiers\n" * 3000) + "REPORT-MIDDLE-SENTINEL\n" + ("tail\n" * 3000)
    encoded = report.encode("utf-8")
    store.put(Artifact(
        id="subagent-report-page-backed", kind="subagent", workspace_id="workspace",
        session_id="session", task_id="task", status="ok", title="page-backed report",
        structured_body={
            "report": report, "report_sha256": hashlib.sha256(encoded).hexdigest(),
            "report_bytes": len(encoded), "report_completion": "complete",
            "report_stop_reason": "end_turn", "observations": [],
        },
    ))
    virtual = CoreArtifactFS(store)
    root_page = virtual.read_file("artifacts/subagent-report-page-backed.md")
    assert "oversized report:" in root_page and "REPORT-MIDDLE-SENTINEL" not in root_page
    index = virtual.read_file("artifacts/subagent-report-page-backed/report/index.md")
    assert hashlib.sha256(encoded).hexdigest() in index
    names = [
        row for row in virtual.listing("artifacts/subagent-report-page-backed/report").splitlines()
        if row.startswith("page-")
    ]
    assert len(names) > 1 and all(name in index for name in names)
    reconstructed = "".join(
        virtual.read_file(f"artifacts/subagent-report-page-backed/report/{name}").split(
            "## Exact retained child report\n", 1,
        )[1]
        for name in names
    )
    assert reconstructed == report
    assert hashlib.sha256(reconstructed.encode()).hexdigest() == hashlib.sha256(encoded).hexdigest()


@check
def legacy_capsule_loss_is_never_relabelled_as_source_paging_or_full_retention():
    import hashlib
    from sliceagent.persistence import Artifact, ArtifactStore

    store = ArtifactStore(tempfile.mkdtemp(prefix="legacy-child-capsule-"))
    view = "HEAD\n…[sealed observation view truncated by capsule budget]…\nTAIL"
    encoded = view.encode()
    store.put(Artifact(
        id="subagent-legacy-capsule", kind="subagent", workspace_id="workspace",
        session_id="session", task_id="task", status="partial",
        structured_body={
            "report": "legacy child report",
            "observations": [{
                "v": 1, "tool": "read_file", "args": {"path": "old.py"},
                "status": "succeeded", "view": view,
                "raw_sha256": hashlib.sha256(encoded).hexdigest(),
                "view_sha256": hashlib.sha256(encoded).hexdigest(),
                "raw_bytes": len(encoded), "view_bytes": len(encoded),
                "redacted": False, "truncated": True,
            }],
        },
    ))
    virtual = CoreArtifactFS(store)
    report = virtual.read_file("artifacts/subagent-legacy-capsule.md")
    index = virtual.read_file("artifacts/subagent-legacy-capsule/evidence/index.md")
    page = virtual.read_file(
        "artifacts/subagent-legacy-capsule/evidence/obs-001-page-001.md"
    )
    assert "legacy archive partial" in report
    retention = next(line for line in index.splitlines() if line.startswith("- retention:"))
    assert "legacy-archive-partial" in retention and "source-partial" not in retention
    assert "legacy-archive-partial" in page


@check
def corrupt_child_evidence_is_an_integrity_error_not_a_false_missing_page():
    from sliceagent.persistence import ArtifactCorruptError, ArtifactStore

    store = ArtifactStore(tempfile.mkdtemp(prefix="corrupt-child-evidence-"))
    with open(store.path_for("subagent-corrupt-evidence"), "w", encoding="utf-8") as stream:
        stream.write("{not-json")
    virtual = CoreArtifactFS(store)
    try:
        virtual.read_file("artifacts/subagent-corrupt-evidence/evidence/index.md")
    except ArtifactCorruptError:
        pass
    else:
        raise AssertionError("corrupt canonical child evidence must not be rendered as a clean missing page")


@check
def persistence_redaction_preserves_intent_source_ranges():
    from dataclasses import asdict, replace

    from sliceagent.discourse import interpret_turn
    from sliceagent.pfc import Slice, record_user
    from sliceagent.taskstate import slice_to_task_state

    request = "Credentials: `sk-1234567890abcdefgh`. Never modify config.py."
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(
        workspace, "session-spans", store_root=tempfile.mkdtemp(prefix="core-store-"),
    )
    active = store.begin(task_id="task-spans", logical_id="turn-1", user_request=request)
    admission = replace(interpret_turn(request, ()).admission, request_source=active.artifact_id)
    store.record_admission({
        "action": "new", "task_id": "task-spans", "admission": admission.to_dict(),
        "focus": [], "consume_pending_proposal": True,
    })
    journal_admission = active.journal.snapshot().event("turn-admission")["payload"]["admission"]
    assert len(journal_admission["request_text"]) == len(request)
    assert "sk-1234567890abcdefgh" not in journal_admission["request_text"]
    state = Slice(); state.reset(request)
    record_user(state, request, source_artifact=active.artifact_id, contract=admission)
    store.seal(
        state=asdict(slice_to_task_state(
            state, "task-spans", session_id="session-spans", status="active",
        )),
        record={}, status="end_turn",
    )
    checkpoint = store.coordinator.checkpoints.load(store.workspace_id, "task-spans")
    persisted_request = checkpoint.state["current_request"]
    persisted_entry = checkpoint.state["intent_entries"][0]
    start, end = persisted_entry["source_range"]
    assert len(persisted_request) == len(request)
    assert "sk-1234567890abcdefgh" not in persisted_request
    assert persisted_request[start:end] == persisted_entry["verbatim_clause"] \
        == "Never modify config.py."
    assert tuple(tuple(span) for span in journal_admission["authority_spans"]) == ((38, 61),)
    store.close()


@check
def bound_child_reference_cannot_attach_to_a_replacement_turn():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(
        workspace, "session-bound-ref", store_root=tempfile.mkdtemp(prefix="core-store-"),
    )
    active_a = store.begin(task_id="task-A", logical_id="turn-A", user_request="inspect A")
    publish_a = store.bind_artifact_ref_sink(
        task_id="task-A", parent_id=active_a.artifact_id,
    )
    store.seal(state={}, record={}, status="aborted")
    active_b = store.begin(task_id="task-B", logical_id="turn-B", user_request="inspect B")
    try:
        publish_a("subagent-late-A")
        assert False, "a retired launch sink must reject late child publication"
    except RuntimeError as exc:
        assert "no longer active" in str(exc)
    snapshot_b = active_b.journal.snapshot()
    assert "subagent-late-A" not in snapshot_b.artifact_refs
    store.seal(state={}, record={}, status="end_turn")
    artifact_b = store.coordinator.artifacts.get(active_b.artifact_id)
    assert "subagent-late-A" not in artifact_b.refs
    store.close()


def _linearization_child(store, active, artifact_id):
    from sliceagent.persistence import Artifact

    return Artifact(
        id=artifact_id,
        kind="subagent",
        workspace_id=store.workspace_id,
        session_id=store.session_id,
        task_id=active.task_id,
        parent_id=active.artifact_id,
        status="ok",
        summary="sealed child report",
        structured_body={"report": "sealed child report"},
    )


@check
def child_publication_wins_before_seal_and_cannot_be_omitted():
    workspace = tempfile.mkdtemp(prefix="workspace-")
    store = LocalTurnStore(
        workspace, "session-child-first", store_root=tempfile.mkdtemp(prefix="core-store-"),
    )
    active = store.begin(task_id="task-A", logical_id="turn-A", user_request="inspect A")
    publish = store.bind_artifact_ref_sink(task_id=active.task_id, parent_id=active.artifact_id)
    child = _linearization_child(store, active, "subagent-child-first")
    put_entered = threading.Event()
    release_put = threading.Event()
    seal_done = threading.Event()
    errors = []

    class PutThenPause:
        root = store.coordinator.artifacts.root

        def put(self, artifact):
            result = store.coordinator.artifacts.put(artifact)
            put_entered.set()
            assert release_put.wait(2), "test did not release child publication"
            return result

    publisher = threading.Thread(
        target=lambda: _capture_thread_error(
            errors, lambda: publish.commit_artifact(PutThenPause(), child)
        ),
        daemon=True,
    )
    sealer = threading.Thread(
        target=lambda: _capture_thread_error(
            errors,
            lambda: (store.seal(state={}, record={}, status="end_turn"), seal_done.set()),
        ),
        daemon=True,
    )
    try:
        publisher.start()
        assert put_entered.wait(1), "child publication never entered its durable put"
        sealer.start()
        assert not seal_done.wait(0.05), "seal overtook a publication that already owned the turn lock"
        release_put.set()
        publisher.join(2)
        sealer.join(2)
        assert not publisher.is_alive() and not sealer.is_alive()
        assert errors == []
        parent = store.coordinator.artifacts.get(active.artifact_id)
        checkpoint = store.coordinator.checkpoints.load(store.workspace_id, active.task_id)
        assert child.id in parent.refs
        assert child.id in checkpoint.artifact_refs
    finally:
        release_put.set()
        publisher.join(2)
        sealer.join(2)
        store.close()


def _capture_thread_error(errors, action):
    try:
        return action()
    except Exception as exc:  # noqa: BLE001 — thread failures must be asserted by the caller
        errors.append(exc)
        return None


@check
def seal_wins_before_child_publication_and_rejects_the_late_child():
    from sliceagent.persistence import SealCoordinator

    workspace = tempfile.mkdtemp(prefix="workspace-")
    store_root = tempfile.mkdtemp(prefix="core-store-")
    seal_entered = threading.Event()
    release_seal = threading.Event()

    def pause_prepared_seal(stage):
        if stage == "prepared":
            seal_entered.set()
            assert release_seal.wait(2), "test did not release parent sealing"

    store = LocalTurnStore(
        workspace,
        "session-seal-first",
        store_root=store_root,
        coordinator=SealCoordinator(store_root, on_stage=pause_prepared_seal),
    )
    active = store.begin(task_id="task-A", logical_id="turn-A", user_request="inspect A")
    publish = store.bind_artifact_ref_sink(task_id=active.task_id, parent_id=active.artifact_id)
    child = _linearization_child(store, active, "subagent-seal-first")
    publication_done = threading.Event()
    seal_errors = []
    publication_errors = []

    sealer = threading.Thread(
        target=lambda: _capture_thread_error(
            seal_errors, lambda: store.seal(state={}, record={}, status="end_turn")
        ),
        daemon=True,
    )

    def publish_late():
        try:
            publish.commit_artifact(store.coordinator.artifacts, child)
        except Exception as exc:  # noqa: BLE001 — the retired launch turn is the expected result
            publication_errors.append(exc)
        finally:
            publication_done.set()

    publisher = threading.Thread(target=publish_late, daemon=True)
    try:
        sealer.start()
        assert seal_entered.wait(1), "parent seal never reached its prepared snapshot"
        publisher.start()
        assert not publication_done.wait(0.05), "late publication bypassed the seal lock"
        release_seal.set()
        sealer.join(2)
        publisher.join(2)
        assert not sealer.is_alive() and not publisher.is_alive()
        assert seal_errors == []
        assert len(publication_errors) == 1
        assert "no longer active" in str(publication_errors[0])
        assert not store.coordinator.artifacts.exists(child.id)
        parent = store.coordinator.artifacts.get(active.artifact_id)
        checkpoint = store.coordinator.checkpoints.load(store.workspace_id, active.task_id)
        assert child.id not in parent.refs
        assert child.id not in checkpoint.artifact_refs
    finally:
        release_seal.set()
        sealer.join(2)
        publisher.join(2)
        store.close()


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
