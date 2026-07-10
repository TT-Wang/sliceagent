"""Offline contract tests for the standalone artifact/checkpoint/journal foundation.

No memem, model, network, or runtime wiring. Run:
    PYTHONPATH=src python tests/test_persistence_protocol.py
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.persistence import (  # noqa: E402
    Artifact,
    ArtifactConflictError,
    ArtifactCorruptError,
    ArtifactNotFoundError,
    ArtifactStore,
    Checkpoint,
    CheckpointConflictError,
    CheckpointStore,
    InvalidRecordError,
    JournalConflictError,
    PendingTurnJournal,
    SealCoordinator,
    deterministic_artifact_id,
)


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


STAMP = "2026-07-10T00:00:00Z"


def _artifact(logical: str = "turn-1", **fields) -> Artifact:
    base = dict(kind="turn", workspace_id="workspace-A", session_id="session-A", task_id="task-A",
                logical_id=logical, timestamp=STAMP, status="ok", title="test turn",
                brief={"request": "fix it"}, summary="done",
                structured_body={"trajectory": [{"kind": "assistant", "text": "done"}]},
                files=("src/a.py",), refs=(), uncertainty=(), error="")
    base.update(fields)
    return Artifact.create(**base)


def _checkpoint(artifact: Artifact, *, generation: int = 1, state=None, refs=None) -> Checkpoint:
    return Checkpoint.create(
        workspace_id=artifact.workspace_id, session_id=artifact.session_id, task_id=artifact.task_id,
        generation=generation, state=state or {"goal": "fix it", "intent": ["preserve API"]},
        artifact_refs=tuple(refs if refs is not None else (artifact.id,)),
        applied_transition_ids=(f"transition-{generation}",),
        workspace_versions={"src/a.py": "sha256:abc"}, updated_at=STAMP,
    )


def _journal(root: str, artifact: Artifact, *, base_generation: int = 0) -> PendingTurnJournal:
    return PendingTurnJournal.begin(
        root, artifact_id=artifact.id, workspace_id=artifact.workspace_id,
        session_id=artifact.session_id, task_id=artifact.task_id,
        base_generation=base_generation, user_request="fix it", created_at=STAMP,
    )


class _Crash(RuntimeError):
    pass


class _CrashAt:
    def __init__(self, stage):
        self.stage = stage
        self.seen = []

    def __call__(self, stage):
        self.seen.append(stage)
        if stage == self.stage:
            raise _Crash(stage)


def _expect(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


@check
def deterministic_ids_are_preallocatable_and_stable():
    args = dict(kind="turn", workspace_id="w", session_id="s", task_id="t", logical_id="request-7")
    first = deterministic_artifact_id(**args)
    assert first == deterministic_artifact_id(**args)
    assert first.startswith("turn-") and ":" not in first and "/" not in first
    assert first != deterministic_artifact_id(**{**args, "logical_id": "request-8"})


@check
def records_are_deeply_immutable():
    brief = {"constraints": ["one"]}
    body = {"nested": {"values": [1, 2]}}
    artifact = _artifact(brief=brief, structured_body=body)
    brief["constraints"].append("mutated outside")
    body["nested"]["values"].append(3)
    assert artifact.to_dict()["brief"] == {"constraints": ["one"]}
    assert artifact.to_dict()["structured_body"] == {"nested": {"values": [1, 2]}}
    def mutate_frozen_record():
        artifact.brief["x"] = "y"
    _expect(TypeError, mutate_frozen_record)


@check
def immutable_artifact_put_is_idempotent_and_conflict_safe():
    with tempfile.TemporaryDirectory() as root:
        store = ArtifactStore(root)
        artifact = _artifact()
        assert store.put(artifact) == artifact
        assert store.put(artifact) == artifact
        _expect(ArtifactConflictError, lambda: store.put(replace(artifact, summary="different bytes")))
        assert store.get(artifact.id).summary == "done", "conflict must never overwrite the original"


@check
def checkpoint_cas_is_versioned_idempotent_and_rejects_stale_writers():
    with tempfile.TemporaryDirectory() as root:
        store = CheckpointStore(root)
        artifact = _artifact()
        one = _checkpoint(artifact)
        assert store.compare_and_swap(one, expected_generation=0) == one
        assert store.compare_and_swap(one, expected_generation=0) == one  # retry after commit

        stale = replace(one, state={"goal": "stale writer"})
        _expect(CheckpointConflictError, lambda: store.compare_and_swap(stale, expected_generation=0))
        two = _checkpoint(artifact, generation=2, state={"goal": "next"})
        assert store.compare_and_swap(two, expected_generation=1) == two
        assert store.load("workspace-A", "task-A").generation == 2


@check
def journal_events_are_append_only_and_idempotent_by_stable_id():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact()
        journal = _journal(root, artifact)
        journal.record_invocation("call-1", name="read_file", args={"path": "src/a.py"})
        journal.record_invocation("call-1", name="read_file", args={"path": "src/a.py"})
        assert len(journal.snapshot().events) == 1
        _expect(JournalConflictError, lambda: journal.record_invocation(
            "call-1", name="read_file", args={"path": "src/b.py"}))
        journal.record_outcome("call-1", {"status": "succeeded", "text": "ok"})
        journal.record_transition("tr-1", {"op": "add-finding", "value": "observed"})
        snap = journal.snapshot()
        assert [event["seq"] for event in snap.events] == [1, 2, 3]
        _expect(JournalConflictError, journal.cleanup)  # cleanup is legal only after a complete seal


@check
def seal_order_is_artifact_first_checkpoint_second_cleanup_last():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact()
        checkpoint = _checkpoint(artifact)
        journal = _journal(root, artifact)
        observed = []

        coordinator = None
        def inspect(stage):
            observed.append(stage)
            if stage == "prepared":
                assert not coordinator.artifacts.exists(artifact.id)
                assert coordinator.checkpoints.load("workspace-A", "task-A") is None
            elif stage == "artifact":
                assert coordinator.artifacts.exists(artifact.id)
                assert coordinator.checkpoints.load("workspace-A", "task-A") is None
            elif stage == "checkpoint":
                assert coordinator.artifacts.exists(artifact.id)
                assert coordinator.checkpoints.load("workspace-A", "task-A").generation == 1
            elif stage == "sealed":
                assert journal.snapshot().sealed
            elif stage == "cleaned":
                assert not journal.exists

        coordinator = SealCoordinator(root, on_stage=inspect)
        result = coordinator.seal(journal, artifact, checkpoint)
        assert result.artifact_id == artifact.id and result.checkpoint_generation == 1
        assert observed == ["prepared", "artifact", "checkpoint", "sealed", "cleaned"]


@check
def storage_seal_rejects_unmatched_started_invocation_without_indeterminate_state():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact("unmatched")
        checkpoint = _checkpoint(artifact)
        journal = _journal(root, artifact)
        journal.record_invocation("call-1", name="run_command", args={"command": "deploy"})
        _expect(InvalidRecordError, lambda: SealCoordinator(root).seal(journal, artifact, checkpoint))
        assert not ArtifactStore(root).exists(artifact.id)


@check
def crash_after_prepare_replays_artifact_and_checkpoint_from_journal():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact("prepare-crash")
        checkpoint = _checkpoint(artifact)
        journal = _journal(root, artifact)
        crash = _CrashAt("prepared")
        _expect(_Crash, lambda: SealCoordinator(root, on_stage=crash).seal(journal, artifact, checkpoint))
        assert not ArtifactStore(root).exists(artifact.id)
        assert CheckpointStore(root).load("workspace-A", "task-A") is None

        recovered = SealCoordinator(root).recover(PendingTurnJournal.open(root, artifact.id))
        assert recovered.status == "replayed"
        assert ArtifactStore(root).get(artifact.id) == artifact
        assert CheckpointStore(root).load("workspace-A", "task-A") == checkpoint
        assert not journal.exists


@check
def crash_before_seal_intent_archives_honest_interrupted_record():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact("unprepared-crash")
        journal = _journal(root, artifact)
        journal.record_invocation("call-1", name="read_file", args={"path": "a.py"})
        recovered = SealCoordinator(root).recover(journal)
        archived = ArtifactStore(root).get(artifact.id)
        assert recovered.status == "archived" and archived.status == "interrupted"
        assert archived.structured_body["journal_events"]
        assert CheckpointStore(root).load("workspace-A", "task-A") is None
        assert not journal.exists


@check
def unprepared_child_journal_preserves_an_already_durable_artifact():
    with tempfile.TemporaryDirectory() as root:
        child = _artifact("child-artifact", kind="subagent", summary="finished child report")
        journal = _journal(root, child)
        ArtifactStore(root).put(child)  # crash after artifact fsync, before the journal close markers

        recovered = SealCoordinator(root).recover(PendingTurnJournal.open(root, child.id))
        assert recovered.status == "cleaned"
        assert ArtifactStore(root).get(child.id) == child
        assert CheckpointStore(root).load("workspace-A", "task-A") is None
        assert not journal.exists


@check
def crash_after_artifact_attaches_it_by_retrying_checkpoint_cas():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact("artifact-crash")
        checkpoint = _checkpoint(artifact)
        journal = _journal(root, artifact)
        crash = _CrashAt("artifact")
        _expect(_Crash, lambda: SealCoordinator(root, on_stage=crash).seal(journal, artifact, checkpoint))
        assert ArtifactStore(root).exists(artifact.id)
        assert CheckpointStore(root).load("workspace-A", "task-A") is None
        assert journal.snapshot().event("artifact-written") is None, "crash is before the journal marker"

        recovered = SealCoordinator(root).recover(PendingTurnJournal.open(root, artifact.id))
        assert recovered.status == "attached"
        assert CheckpointStore(root).load("workspace-A", "task-A") == checkpoint
        assert not journal.exists


@check
def crash_after_checkpoint_only_finishes_journal_cleanup():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact("checkpoint-crash")
        checkpoint = _checkpoint(artifact)
        journal = _journal(root, artifact)
        crash = _CrashAt("checkpoint")
        _expect(_Crash, lambda: SealCoordinator(root, on_stage=crash).seal(journal, artifact, checkpoint))
        assert ArtifactStore(root).exists(artifact.id)
        assert CheckpointStore(root).load("workspace-A", "task-A") == checkpoint
        assert journal.snapshot().event("checkpoint-committed") is None

        recovered = SealCoordinator(root).recover(PendingTurnJournal.open(root, artifact.id))
        assert recovered.status == "cleaned"
        assert CheckpointStore(root).load("workspace-A", "task-A") == checkpoint
        assert not journal.exists


@check
def stale_recovery_quarantines_artifact_instead_of_overwriting_checkpoint():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact("losing-turn")
        target = _checkpoint(artifact)
        journal = _journal(root, artifact)
        crash = _CrashAt("artifact")
        _expect(_Crash, lambda: SealCoordinator(root, on_stage=crash).seal(journal, artifact, target))

        winner = Checkpoint.create(workspace_id="workspace-A", session_id="session-A", task_id="task-A",
                                   generation=1, state={"goal": "winner"}, artifact_refs=(), updated_at=STAMP)
        CheckpointStore(root).compare_and_swap(winner, expected_generation=0)
        result = SealCoordinator(root).recover(PendingTurnJournal.open(root, artifact.id))
        assert result.status == "conflict" and result.checkpoint_generation == 1
        assert CheckpointStore(root).load("workspace-A", "task-A") == winner
        assert ArtifactStore(root).exists(artifact.id), "unattached artifact remains quarantined/readable"
        assert journal.exists, "conflicting journal remains for explicit inspection"


@check
def checkpoint_cannot_publish_a_dangling_artifact_reference_through_coordinator():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact("missing-ref")
        checkpoint = _checkpoint(artifact, refs=(artifact.id, "turn-missing00000000000000000000000000"))
        journal = _journal(root, artifact)
        _expect(ArtifactNotFoundError,
                lambda: SealCoordinator(root).seal(journal, artifact, checkpoint))
        assert ArtifactStore(root).exists(artifact.id), "artifact-first write remains available for recovery"
        assert CheckpointStore(root).load("workspace-A", "task-A") is None
        assert journal.exists


@check
def checkpoint_cannot_publish_a_corrupt_artifact_reference():
    with tempfile.TemporaryDirectory() as root:
        artifact = _artifact("corrupt-ref-parent")
        corrupt_id = deterministic_artifact_id(
            kind="subagent", workspace_id="workspace-A", session_id="session-A",
            task_id="task-A", logical_id="corrupt-child",
        )
        store = ArtifactStore(root)
        path = store.path_for(corrupt_id)
        with open(path, "wb") as stream:
            stream.write(b"not-json\n")
        checkpoint = _checkpoint(artifact, refs=(artifact.id, corrupt_id))
        journal = _journal(root, artifact)
        _expect(ArtifactCorruptError,
                lambda: SealCoordinator(root).seal(journal, artifact, checkpoint))
        assert CheckpointStore(root).load("workspace-A", "task-A") is None


@check
def local_store_reopens_without_memem_and_uses_private_modes():
    with tempfile.TemporaryDirectory() as root:
        # A poisoned/missing optional semantic-memory import cannot affect this stdlib-only store.
        old_memem = sys.modules.get("memem", "__absent__")
        sys.modules["memem"] = None
        try:
            artifact = _artifact("local-only")
            checkpoint = _checkpoint(artifact)
            journal = _journal(root, artifact)
            SealCoordinator(root).seal(journal, artifact, checkpoint, cleanup=False)

            reopened_artifact = ArtifactStore(root).get(artifact.id)
            reopened_checkpoint = CheckpointStore(root).load("workspace-A", "task-A")
            reopened_journal = PendingTurnJournal.open(root, artifact.id)
            assert reopened_artifact == artifact and reopened_checkpoint == checkpoint
            assert reopened_journal.snapshot().sealed

            if os.name == "posix":
                mode = lambda path: stat.S_IMODE(os.stat(path).st_mode)
                assert mode(os.path.join(root, "artifacts")) == 0o700
                assert mode(os.path.dirname(ArtifactStore(root).path_for(artifact.id))) == 0o700
                assert mode(ArtifactStore(root).path_for(artifact.id)) == 0o600
                assert mode(os.path.join(root, "checkpoints")) == 0o700
                assert mode(CheckpointStore(root).path_for("workspace-A", "task-A")) == 0o600
                assert mode(os.path.join(root, "journals")) == 0o700
                assert mode(reopened_journal.path) == 0o600
        finally:
            if old_memem == "__absent__":
                sys.modules.pop("memem", None)
            else:
                sys.modules["memem"] = old_memem


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
