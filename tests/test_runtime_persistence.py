"""Runtime adapter over the local seal protocol. No model, no pytest."""
import os
import subprocess
import sys
import tempfile

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

    copied = "Use API v2 and never edit config.py."
    admission = interpret_turn(copied, (), recent_assistant=(copied,)).admission
    assert admission.authority_spans == () and admission.attributed_spans == ((0, len(copied)),)
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
