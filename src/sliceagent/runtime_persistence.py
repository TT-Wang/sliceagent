"""Runtime-facing adapter over the local artifact/checkpoint/journal protocol.

This is the only persistence object the agent host needs to know.  It captures a stable
task identity at turn start, journals execution, and seals an immutable artifact before
publishing the next active-state checkpoint. Typed L2 knowledge remains a downstream consumer
of the resulting artifacts rather than the durability mechanism itself.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .persistence import (
    Artifact,
    ArtifactNotFoundError,
    ArtifactStore,
    Checkpoint,
    JournalCorruptError,
    PendingTurnJournal,
    PersistenceError,
    RecoveryResult,
    SealCoordinator,
    SealResult,
    WorkspaceLease,
    artifact_order_key,
    reserve_durable_order,
)
from .recovery import root_key, state_dir
from .safety import redact_text


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        # Intent/admission records contain exact offsets into request text. Persistence redaction therefore
        # masks bytes without changing string length; every stored source span keeps naming the same clause.
        return redact_text(value, preserve_length=True)
    if isinstance(value, Mapping):
        return {redact_text(str(key)): _redact(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(child) for child in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return redact_text(str(value))


def _confirmed_transition_ids(snapshot) -> tuple[str, ...]:
    """Only complete outcome effect sets are publishable after a partial journal failure."""
    recorded = {
        str(event.get("payload", {}).get("transition_id"))
        for event in snapshot.events
        if event.get("type") == "semantic-transition"
        and event.get("payload", {}).get("transition_id")
    }
    confirmed = []
    for event in snapshot.events:
        if event.get("type") != "tool-outcome":
            continue
        effects = (event.get("payload", {}).get("outcome", {}).get("effects") or ())
        ids = [str(effect.get("id")) for effect in effects
               if isinstance(effect, Mapping) and effect.get("id")]
        if ids and set(ids).issubset(recorded):
            confirmed.extend(ids)
    return tuple(dict.fromkeys(confirmed))


def recoverable_child_report_count(artifact: Artifact) -> int:
    """Count full child tool results retained by an interrupted turn artifact.

    This recognizes only the recovery artifact written from an unprepared parent journal.  Ordinary turns
    keep child reports solely in their native tool trajectory.  Requiring both the typed ``child_outcome``
    metadata and the report envelope prevents a metadata-only or torn result from being advertised as
    readable synthesis evidence.
    """
    if not isinstance(artifact, Artifact) or artifact.kind != "turn" or artifact.status != "interrupted":
        return 0
    body = artifact.structured_body if isinstance(artifact.structured_body, Mapping) else {}
    events = body.get("journal_events") if isinstance(body, Mapping) else None
    if not isinstance(events, (list, tuple)):
        return 0
    found: set[str] = set()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or event.get("type") != "tool-outcome":
            continue
        payload = event.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        outcome = payload.get("outcome")
        outcome = outcome if isinstance(outcome, Mapping) else {}
        status = outcome.get("status")
        if not (isinstance(status, bool) or (
                isinstance(status, str)
                and status in {"succeeded", "steered", "failed", "cancelled"})):
            continue
        text = str(outcome.get("text") or "")
        if "BEGIN CHILD REPORT" not in text or "END CHILD REPORT" not in text:
            continue
        effects = outcome.get("effects")
        if not isinstance(effects, (list, tuple)):
            continue
        for effect in effects:
            if not isinstance(effect, Mapping) or effect.get("kind") != "child_outcome":
                continue
            child = effect.get("payload")
            child = child if isinstance(child, Mapping) else {}
            try:
                report_bytes = int(child.get("report_bytes") or 0)
            except (TypeError, ValueError, OverflowError):
                report_bytes = 0
            if report_bytes <= 0 or str(child.get("report_completion") or "") == "absent":
                continue
            invocation_id = str(payload.get("invocation_id") or f"journal-event-{index}")
            found.add(invocation_id)
            break
    return len(found)


@dataclass(frozen=True)
class ActiveTurn:
    task_id: str
    logical_id: str
    artifact_id: str
    journal: PendingTurnJournal
    segment_id: str = ""
    segment_index: int = 0
    workspace_epoch: int = 0


class _BoundArtifactRefSink:
    """Turn-pinned child publication boundary.

    A child artifact and its downward parent reference are one logical commit.  The
    local turn-store lock makes that commit linearizable with parent sealing: either
    publication owns the lock first and the seal snapshots the new reference, or the
    seal retires the launch turn first and publication writes nothing.

    ``__call__`` retains the legacy already-stored-reference surface.  Production
    subagents feature-detect :meth:`commit_artifact` and publish both durable facts
    while holding the same lock.
    """

    def __init__(self, owner: "LocalTurnStore", turn: ActiveTurn):
        self._owner = owner
        self._turn = turn

    def _validate_locked(self) -> None:
        self._owner._ensure_open()
        if self._owner.active is not self._turn:
            raise RuntimeError("child launch turn is no longer active")
        if self._turn.journal.snapshot().seal_intent is not None:
            raise RuntimeError("child launch turn already has a prepared seal")

    def _record_locked(self, artifact_id: str) -> None:
        value = str(artifact_id)
        self._turn.journal.record_artifact_ref(value)
        if value not in self._owner._active_refs:
            self._owner._active_refs.append(value)

    def __call__(self, artifact_id: str) -> None:
        with self._owner._state_lock:
            self._validate_locked()
            self._record_locked(artifact_id)

    def commit_artifact(self, artifact_store, artifact: Artifact) -> str:
        """Atomically publish a canonical child artifact into its launch turn.

        Filesystem crash recovery still handles the unavoidable process-death gap
        between the immutable artifact write and journal append.  In a live process,
        however, cancellation or parent sealing cannot split those two facts.
        """
        if not isinstance(artifact, Artifact):
            raise TypeError("child publication requires an Artifact")
        expected_root = os.path.realpath(self._owner.coordinator.artifacts.root)
        actual_root = os.path.realpath(str(getattr(artifact_store, "root", "") or ""))
        if actual_root != expected_root:
            raise RuntimeError("child artifact store does not match the launch turn store")
        if artifact.task_id != self._turn.task_id:
            raise RuntimeError("child artifact task identity does not match the launch turn")
        if artifact.workspace_id != self._owner.workspace_id:
            raise RuntimeError("child artifact workspace identity does not match the launch turn")
        with self._owner._state_lock:
            self._validate_locked()
            artifact_store.put(artifact)
            self._record_locked(artifact.id)
            return artifact.id


@dataclass(frozen=True)
class WorkspaceTransition:
    """Crash-visible control-plane record between two workspace-bound turn segments."""

    id: str
    session_id: str
    logical_turn_id: str
    task_id: str
    request: str
    source_root: str
    target_root: str
    source_artifact_id: str
    source_segment_index: int
    source_workspace_epoch: int
    target_segment_index: int
    target_workspace_epoch: int
    status: str = "prepared"
    target_artifact_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": 1, "id": self.id, "session_id": self.session_id,
            "logical_turn_id": self.logical_turn_id, "task_id": self.task_id,
            "request": self.request, "source_root": self.source_root, "target_root": self.target_root,
            "source_artifact_id": self.source_artifact_id,
            "source_segment_index": self.source_segment_index,
            "source_workspace_epoch": self.source_workspace_epoch,
            "target_segment_index": self.target_segment_index,
            "target_workspace_epoch": self.target_workspace_epoch,
            "status": self.status, "target_artifact_id": self.target_artifact_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkspaceTransition":
        if not isinstance(value, Mapping) or int(value.get("v", 0) or 0) != 1:
            raise PersistenceError("workspace transition has an unsupported schema")
        fields = {
            name: str(value.get(name) or "") for name in (
                "id", "session_id", "logical_turn_id", "task_id", "request", "source_root",
                "target_root", "source_artifact_id", "status", "target_artifact_id",
            )
        }
        if not all(fields[name] for name in (
                "id", "session_id", "logical_turn_id", "task_id", "source_root", "target_root",
                "source_artifact_id", "status")):
            raise PersistenceError("workspace transition is missing required identity fields")
        if fields["status"] not in {"prepared", "activated", "continuing"}:
            raise PersistenceError(f"workspace transition has unsupported status {fields['status']!r}")
        try:
            numeric = {
                name: int(value.get(name, 0)) for name in (
                    "source_segment_index", "source_workspace_epoch",
                    "target_segment_index", "target_workspace_epoch",
                )
            }
        except (TypeError, ValueError) as exc:
            raise PersistenceError("workspace transition contains an invalid segment identity") from exc
        if any(number < 0 for number in numeric.values()):
            raise PersistenceError("workspace transition segment identities must be non-negative")
        return cls(**fields, **numeric)


class WorkspaceTransitionStore:
    """Small atomic journal for the gap after source seal and before target-segment seal.

    Turn journals are workspace-local, so neither one alone can explain a crash between them.  This record is
    application-local and contains only transport identity.  It is removed after the target segment reaches a
    durable terminal boundary; startup/diagnostics can otherwise report exactly which request was stranded.
    """

    _LIVE = frozenset({"prepared", "activated", "continuing"})

    @staticmethod
    def _root_identity(path: str) -> str:
        """Canonical comparison identity while retaining the original path for display/persistence."""
        return os.path.normcase(os.path.realpath(path))

    def __init__(self, root: str | None = None):
        self.root = os.path.realpath(root or state_dir("workspace-transitions"))
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    def _path(self, transition_id: str) -> str:
        safe = hashlib.sha256(str(transition_id).encode("utf-8")).hexdigest()
        return os.path.join(self.root, safe + ".json")

    def _write(self, transition: WorkspaceTransition) -> WorkspaceTransition:
        # The live object retains the exact request; persistence follows the same redact-on-write contract as
        # turn journals. preserve_length keeps any source offsets usable during recovery diagnostics.
        body = transition.to_dict()
        body["request"] = redact_text(transition.request, preserve_length=True)
        data = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        fd, tmp = tempfile.mkstemp(prefix=".transition-", suffix=".tmp", dir=self.root)
        try:
            try:
                os.fchmod(fd, 0o600)
            except (AttributeError, OSError):
                pass
            with os.fdopen(fd, "wb") as stream:
                fd = -1
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            path = self._path(transition.id)
            os.replace(tmp, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            self._fsync_root()
            return transition
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def _fsync_root(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            fd = os.open(self.root, flags)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def prepare(
        self, *, session_id: str, logical_turn_id: str, task_id: str, request: str,
        source_root: str, target_root: str, source_artifact_id: str,
        source_segment_index: int, source_workspace_epoch: int,
    ) -> WorkspaceTransition:
        source, target = os.path.realpath(source_root), os.path.realpath(target_root)
        if self._root_identity(source) == self._root_identity(target):
            raise ValueError("workspace transition source and target must differ")
        identity = json.dumps(
            [str(session_id), str(logical_turn_id), self._root_identity(source),
             self._root_identity(target), int(source_segment_index)],
            separators=(",", ":"), ensure_ascii=False,
        )
        transition = WorkspaceTransition(
            id="workspace-transition-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            session_id=str(session_id), logical_turn_id=str(logical_turn_id), task_id=str(task_id),
            request=str(request), source_root=source, target_root=target,
            source_artifact_id=str(source_artifact_id),
            source_segment_index=int(source_segment_index),
            source_workspace_epoch=int(source_workspace_epoch),
            target_segment_index=int(source_segment_index) + 1,
            target_workspace_epoch=int(source_workspace_epoch) + 1,
        )
        path = self._path(transition.id)
        if os.path.exists(path):
            existing = self.load(transition.id)
            comparable = {**existing.to_dict(), "status": "prepared", "target_artifact_id": ""}
            expected = transition.to_dict()
            expected["request"] = redact_text(transition.request, preserve_length=True)
            # Case/separator aliases share one transition identity on case-insensitive filesystems. Compare
            # their canonical identities too while retaining the first admitted spelling for diagnostics.
            for field in ("source_root", "target_root"):
                comparable[field] = self._root_identity(comparable[field])
                expected[field] = self._root_identity(expected[field])
            if comparable != expected:
                raise PersistenceError("workspace transition identity already names different content")
            return existing
        return self._write(transition)

    def load(self, transition_id: str) -> WorkspaceTransition:
        path = self._path(transition_id)
        try:
            with open(path, encoding="utf-8") as stream:
                return WorkspaceTransition.from_dict(json.load(stream))
        except FileNotFoundError:
            raise PersistenceError(f"no workspace transition {transition_id!r}") from None
        except (OSError, TypeError, ValueError) as exc:
            raise PersistenceError(f"workspace transition {transition_id!r} is unreadable: {exc}") from exc

    def _advance(self, transition: WorkspaceTransition, status: str, **updates) -> WorkspaceTransition:
        if status not in self._LIVE:
            raise ValueError(f"unsupported live transition status {status!r}")
        current = self.load(transition.id)
        allowed = {
            "prepared": {"prepared", "activated"},
            "activated": {"activated", "continuing"},
            "continuing": {"continuing"},
        }
        if status not in allowed.get(current.status, set()):
            raise PersistenceError(f"invalid workspace transition {current.status!r} -> {status!r}")
        from dataclasses import replace
        return self._write(replace(current, status=status, **updates))

    def mark_activated(self, transition: WorkspaceTransition) -> WorkspaceTransition:
        return self._advance(transition, "activated")

    def mark_continuing(
        self, transition: WorkspaceTransition, *, target_artifact_id: str,
    ) -> WorkspaceTransition:
        if not target_artifact_id:
            raise ValueError("target_artifact_id is required")
        return self._advance(
            transition, "continuing", target_artifact_id=str(target_artifact_id),
        )

    def clear(self, transition: WorkspaceTransition) -> None:
        try:
            os.unlink(self._path(transition.id))
        except FileNotFoundError:
            return
        self._fsync_root()

    def pending(
        self, *, workspace_root: str | None = None, session_id: str | None = None,
    ) -> tuple[WorkspaceTransition, ...]:
        root = self._root_identity(workspace_root) if workspace_root else ""
        rows = []
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.root, name)
            try:
                with open(path, encoding="utf-8") as stream:
                    row = WorkspaceTransition.from_dict(json.load(stream))
            except (OSError, TypeError, ValueError, PersistenceError) as exc:
                raise PersistenceError(f"workspace transition record {name!r} is unreadable: {exc}") from exc
            if row.status not in self._LIVE:
                continue
            if session_id and row.session_id != str(session_id):
                continue
            if root and root not in {
                    self._root_identity(row.source_root), self._root_identity(row.target_root)}:
                continue
            rows.append(row)
        return tuple(rows)


class LocalTurnStore:
    """Always-on local durability with one live writer per workspace store."""

    def __init__(self, workspace_root: str, session_id: str, *, store_root: str | None = None,
                 coordinator: SealCoordinator | None = None, exclusive: bool = True):
        self.workspace_root = os.path.realpath(workspace_root)
        self.workspace_id = root_key(self.workspace_root)
        self.session_id = str(session_id)
        self.store_root = store_root or state_dir("core", self.workspace_id)
        self._lease = WorkspaceLease.acquire(self.store_root) if exclusive else None
        # Serializes active-turn identity and child-reference publication with the
        # complete seal snapshot/commit. RLock is required because the public seal
        # and event helpers call smaller store helpers while retaining ownership.
        self._state_lock = threading.RLock()
        self._closed = False
        self.coordinator = coordinator or SealCoordinator(self.store_root)
        self.active: ActiveTurn | None = None
        self._active_refs: list[str] = []

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self._lease is not None:
                self._lease.close()
                self._lease = None
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("local turn store is closed")

    def begin(
        self, *, task_id: str, logical_id: str, user_request: str,
        segment_index: int = 0, workspace_epoch: int = 0,
    ) -> ActiveTurn:
        with self._state_lock:
            return self._begin_locked(
                task_id=task_id, logical_id=logical_id, user_request=user_request,
                segment_index=segment_index, workspace_epoch=workspace_epoch,
            )

    def _begin_locked(
        self, *, task_id: str, logical_id: str, user_request: str,
        segment_index: int = 0, workspace_epoch: int = 0,
    ) -> ActiveTurn:
        self._ensure_open()
        if self.active is not None:
            raise RuntimeError(f"turn {self.active.logical_id!r} is already active")
        segment_index, workspace_epoch = int(segment_index), int(workspace_epoch)
        if segment_index < 0 or workspace_epoch < 0:
            raise ValueError("segment_index and workspace_epoch must be non-negative")
        legacy_floor = 0
        if not os.path.exists(os.path.join(self.store_root, ".turn-order")):
            legacy_floor = max(
                (artifact_order_key(artifact)[0]
                 for artifact in self.coordinator.artifacts.list_all()),
                default=0,
            )
        journal = self.coordinator.begin_turn(
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            task_id=str(task_id),
            logical_id=str(logical_id),
            user_request=redact_text(str(user_request), preserve_length=True),
            order_ns=reserve_durable_order(self.store_root, minimum=legacy_floor),
        )
        segment_id = f"{logical_id}:segment:{segment_index}"
        journal.append("logical-segment", {
            "logical_turn_id": str(logical_id), "segment_id": segment_id,
            "segment_index": segment_index, "workspace_epoch": workspace_epoch,
            "workspace_root": self.workspace_root,
        }, event_id="logical-segment")
        self.active = ActiveTurn(
            str(task_id), str(logical_id), journal.snapshot().artifact_id, journal,
            segment_id=segment_id, segment_index=segment_index, workspace_epoch=workspace_epoch,
        )
        self._active_refs = []
        return self.active

    def record_artifact_ref(self, artifact_id: str) -> None:
        """Attach one already-durable child/source artifact to the running turn checkpoint."""
        with self._state_lock:
            turn = self._turn()
            if turn.journal.snapshot().seal_intent is not None:
                raise RuntimeError("active turn already has a prepared seal")
            value = str(artifact_id)
            turn.journal.record_artifact_ref(value)
            if value not in self._active_refs:
                self._active_refs.append(value)

    def bind_artifact_ref_sink(self, *, task_id: str = "", parent_id: str = ""):
        """Pin child-ref publication to the exact active turn that launched it.

        A cancelled explorer may unwind after the UI has admitted another task. A dynamic
        ``record_artifact_ref`` callback would then attach A's child to B. This closure writes only A's captured
        journal and refuses once its identity is no longer active; it never dereferences the replacement turn.
        """
        with self._state_lock:
            turn = self._turn()
            if turn.journal.snapshot().seal_intent is not None:
                raise RuntimeError("active turn already has a prepared seal")
            if task_id and turn.task_id != str(task_id):
                raise RuntimeError("child task identity does not match the active turn")
            if parent_id and turn.artifact_id != str(parent_id):
                raise RuntimeError("child parent identity does not match the active turn")
            return _BoundArtifactRefSink(self, turn)

    def record_admission(self, admission: Mapping[str, Any]) -> None:
        """Journal the immutable admission envelope installed for this turn.

        The envelope carries route, focus/proposal deltas, and the canonical TurnAdmission. Recovery consumes
        this record rather than re-interpreting header text without its original discourse sources. A bare
        TurnAdmission mapping remains readable for journals created during the format transition.
        """
        self._turn().journal.append(
            "turn-admission", _redact(dict(admission)), event_id="turn-admission",
        )

    def _turn(self) -> ActiveTurn:
        self._ensure_open()
        if self.active is None:
            raise RuntimeError("no active turn")
        return self.active

    def record_invocation(self, invocation_id: str, *, name: str, args: Mapping[str, Any]) -> None:
        self._turn().journal.record_invocation(
            str(invocation_id), name=str(name), args=_redact(dict(args)),
        )

    def record_request(self, invocation) -> None:
        """Record a logical provider request without claiming that its handler started."""
        self._turn().journal.append("tool-requested", _redact({
            "invocation_id": str(invocation.id),
            "name": str(invocation.name),
            "args": dict(invocation.args),
            "provider_index": invocation.provider_index,
        }), event_id=f"request:{invocation.id}")

    def record_rejection(self, invocation, reason: str, *, kind: str = "rejected") -> None:
        """Record a conclusive pre-handler stop, retaining neutral lifecycle vs safety provenance."""
        self._turn().journal.append("tool-rejected", _redact({
            "invocation_id": str(invocation.id),
            "name": str(invocation.name),
            "args": dict(invocation.args),
            "provider_index": invocation.provider_index,
            "reason": str(reason),
            "rejection_kind": str(kind or "rejected"),
        }), event_id=f"reject:{invocation.id}")

    def record_execution_started(self, invocation) -> None:
        """Record the last durable boundary before a physical handler is entered.

        The legacy invocation row is written first so even a crash while appending the richer lifecycle row
        conservatively leaves an unresolved started call rather than permitting a clean checkpoint.
        """
        self.record_invocation(invocation.id, name=invocation.name, args=invocation.args)
        self._turn().journal.append("tool-execution-started", _redact({
            "invocation_id": str(invocation.id),
            "name": str(invocation.name),
            "args": dict(invocation.args),
            "provider_index": invocation.provider_index,
        }), event_id=f"start:{invocation.id}")

    def record_outcome(self, invocation_id: str, *, status: str, text: str,
                       effects: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = ()) -> None:
        self._turn().journal.record_outcome(str(invocation_id), _redact({
            "status": str(status), "text": str(text), "effects": list(effects),
        }))

    @staticmethod
    def _outcome_record(outcome) -> dict[str, Any]:
        return {
            "status": outcome.status.value,
            "text": outcome.text,
            "effects": [
                {"id": effect.id, "kind": effect.kind, "payload": dict(effect.payload)}
                for effect in outcome.effects
            ],
        }

    def record_settlement(self, outcome) -> None:
        """Record typed settlement separately from later reducer application."""
        record = self._outcome_record(outcome)
        # Preserve the old recovery protocol first; the canonical receipt row enriches the same fact.
        self.record_outcome(
            outcome.invocation.id, status=record["status"], text=record["text"], effects=record["effects"],
        )
        self._turn().journal.append("tool-settled", _redact({
            "invocation_id": str(outcome.invocation.id),
            "name": str(outcome.invocation.name),
            "args": dict(outcome.invocation.args),
            "provider_index": outcome.invocation.provider_index,
            "outcome": record,
        }), event_id=f"settle:{outcome.invocation.id}")

    def record_transition(self, transition_id: str, transition: Mapping[str, Any]) -> None:
        self._turn().journal.record_transition(str(transition_id), _redact(dict(transition)))

    def record_effect_applied(self, invocation_id: str, effect) -> None:
        """Record reducer acceptance; settlement alone never claims an effect was applied."""
        transition = {
            "kind": effect.kind, "payload": dict(effect.payload), "invocation_id": str(invocation_id),
        }
        self.record_transition(effect.id, transition)
        self._turn().journal.append("tool-effect-applied", _redact({
            "invocation_id": str(invocation_id),
            "effect_id": str(effect.id),
            "kind": str(effect.kind),
            "payload": dict(effect.payload),
        }), event_id=f"apply:{invocation_id}:{effect.id}")

    def observe_event(self, event) -> None:
        """Journal invocation/outcome truth before authoritative reduction.

        Applied semantic transitions are deliberately recorded by :meth:`observe_reduction` only after
        the required reducer returns successfully.  Keeping those two phases separate prevents a failed
        reducer from publishing effect IDs that the active checkpoint never actually applied.
        """
        if self.active is None:
            return
        # Import lazily so the durable store itself remains independent from the UI/event layer.
        from .events import (ToolEffectApplied, ToolExecutionStarted, ToolRejected, ToolRequested,
                             ToolResult, ToolSettled, ToolStarted)
        if isinstance(event, ToolRequested):
            self.record_request(event.invocation)
            return
        if isinstance(event, ToolRejected):
            self.record_rejection(event.invocation, event.reason, kind=getattr(event, "kind", "rejected"))
            return
        if isinstance(event, ToolExecutionStarted):
            self.record_execution_started(event.invocation)
            return
        if isinstance(event, ToolStarted) and getattr(event, "invocation", None) is not None:
            inv = event.invocation
            self.record_invocation(inv.id, name=inv.name, args=inv.args)
            return
        if isinstance(event, ToolSettled):
            self.record_settlement(event.outcome)
            return
        if isinstance(event, ToolEffectApplied):
            self.record_effect_applied(event.invocation_id, event.effect)
            return
        if not isinstance(event, ToolResult) or getattr(event, "outcome", None) is None:
            return
        outcome = event.outcome
        snapshot = self._turn().journal.snapshot()
        canonical_request = any(
            row.get("type") == "tool-requested"
            and row.get("payload", {}).get("invocation_id") == outcome.invocation.id
            for row in snapshot.events
        )
        # Legacy callers may still publish only ToolResult. Preserve their old started+settled projection;
        # the canonical loop always sent ToolRequested first, so rejected/deduplicated calls stay not-started.
        if not canonical_request:
            self.record_invocation(
                outcome.invocation.id, name=outcome.invocation.name, args=outcome.invocation.args,
            )
        effects = self._outcome_record(outcome)["effects"]
        self.record_outcome(
            outcome.invocation.id, status=outcome.status.value, text=outcome.text, effects=effects,
        )

    def observe_reduction(self, event) -> None:
        """Record effect IDs only after the authoritative state reducer succeeded."""
        if self.active is None:
            return
        from .events import ToolResult
        if not isinstance(event, ToolResult) or getattr(event, "outcome", None) is None:
            return
        if not getattr(event, "apply_effects", True):
            return
        outcome = event.outcome
        for effect in outcome.effects:
            self.record_effect_applied(outcome.invocation.id, effect)

    def seal(
        self,
        *,
        state: Mapping[str, Any],
        record: Mapping[str, Any],
        status: str,
        title: str = "",
        summary: str = "",
        files: tuple[str, ...] | list[str] = (),
        refs: tuple[str, ...] | list[str] = (),
        uncertainty: tuple[str, ...] | list[str] = (),
        error: str = "",
        workspace_versions: Mapping[str, Any] | None = None,
        cleanup: bool = True,
    ) -> SealResult:
        # Retain ownership through snapshot construction, immutable artifact/checkpoint
        # publication, and active-turn retirement. A bound child publisher therefore
        # cannot return success for a reference omitted from this seal.
        with self._state_lock:
            return self._seal_locked(
                state=state, record=record, status=status, title=title, summary=summary,
                files=files, refs=refs, uncertainty=uncertainty, error=error,
                workspace_versions=workspace_versions, cleanup=cleanup,
            )

    def _seal_locked(
        self,
        *,
        state: Mapping[str, Any],
        record: Mapping[str, Any],
        status: str,
        title: str = "",
        summary: str = "",
        files: tuple[str, ...] | list[str] = (),
        refs: tuple[str, ...] | list[str] = (),
        uncertainty: tuple[str, ...] | list[str] = (),
        error: str = "",
        workspace_versions: Mapping[str, Any] | None = None,
        cleanup: bool = True,
    ) -> SealResult:
        turn = self._turn()
        snapshot = turn.journal.snapshot()
        transitions = _confirmed_transition_ids(snapshot)
        safe_record = _redact(dict(record))
        safe_state = _redact(dict(state))
        effective_status = str(status)
        uncertainty = tuple(str(item) for item in uncertainty)
        unresolved = snapshot.unresolved_invocations
        if unresolved:
            from .execution import reconciliation_targets

            invocation_ids = []
            targets = []
            for invocation in unresolved:
                invocation_id = str(invocation.get("invocation_id") or "unknown")
                invocation_ids.append(invocation_id)
                args = invocation.get("args")
                targets.extend(reconciliation_targets(
                    str(invocation.get("name") or ""),
                    args if isinstance(args, Mapping) else {},
                ))
            detail = (
                "invocation(s) without conclusive outcomes: " + ", ".join(invocation_ids)
                + "; do not claim a result without fresh evidence"
            )
            # An unknown read-only explorer result is still an indeterminate receipt, but it cannot leave a
            # workspace effect behind.  Only invocations with concrete affected targets create advisory
            # execution-uncertainty state.
            if targets:
                existing_marker = str(safe_state.get("reconciliation_required") or "")
                safe_state["reconciliation_required"] = (
                    existing_marker if detail in existing_marker else
                    " | ".join(item for item in (existing_marker, detail) if item)
                )
                existing_targets = safe_state.get("reconciliation_targets")
                safe_state["reconciliation_targets"] = list(dict.fromkeys((
                    *(existing_targets if isinstance(existing_targets, (list, tuple)) else ()),
                    *targets,
                )))
                safe_state["status"] = "indeterminate"
            effective_status = "indeterminate"
            uncertainty = tuple(dict.fromkeys((*uncertainty, detail)))
            error = str(error or detail)
        refs = tuple(dict.fromkeys((
            *snapshot.artifact_refs, *self._active_refs, *tuple(str(ref) for ref in refs),
        )))
        # The receipt is a pure, immutable journal projection embedded in this SAME turn artifact. It has no
        # writer or lifecycle of its own, so execution truth cannot drift from the seal that carries it.
        from .receipts import TurnReceipt

        meta = safe_record.get("meta")
        meta = dict(meta) if isinstance(meta, Mapping) else {}
        meta.update({
            "logical_turn_id": turn.logical_id,
            "segment_id": turn.segment_id or f"{turn.logical_id}:segment:{turn.segment_index}",
            "segment_index": turn.segment_index,
            "workspace_epoch": turn.workspace_epoch,
        })
        if snapshot.order_ns:
            meta["order_ns"] = snapshot.order_ns
        safe_record["meta"] = meta
        receipt_usage = {
            "prompt_tokens": meta.get("ptok", 0),
            "completion_tokens": meta.get("ctok", 0),
        }
        safe_record["turn_receipt"] = TurnReceipt.from_events(
            snapshot.events,
            turn_id=snapshot.artifact_id,
            turn_status=effective_status,
            artifact_refs=refs,
            usage=receipt_usage,
        ).to_dict()
        artifact = Artifact(
            id=snapshot.artifact_id,
            kind="turn",
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            task_id=turn.task_id,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            status=effective_status,
            title=redact_text(str(title)),
            brief={"request": snapshot.header.get("user_request", "")},
            summary=redact_text(str(summary)),
            structured_body=safe_record,
            files=tuple(redact_text(str(path)) for path in files),
            refs=tuple(redact_text(str(ref)) for ref in refs),
            uncertainty=tuple(redact_text(str(item)) for item in uncertainty),
            error=redact_text(str(error)),
        )
        checkpoint = Checkpoint.create(
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            task_id=turn.task_id,
            generation=snapshot.base_generation + 1,
            state=safe_state,
            artifact_refs=(artifact.id, *tuple(str(ref) for ref in refs)),
            applied_transition_ids=transitions,
            workspace_versions=_redact(dict(workspace_versions or {})),
            order_ns=snapshot.order_ns,
        )
        result = self.coordinator.seal(turn.journal, artifact, checkpoint, cleanup=cleanup)
        # Ownership ends only after the artifact/checkpoint protocol succeeds. A failed seal retains the
        # active turn and blocks a newer generation from overtaking its recovery journal.
        self.active = None
        self._active_refs = []
        return result

    def checkpoints(self):
        """Return startup-safe authoritative checkpoints for task hydration.

        Checkpoint bytes and every referenced artifact are revalidated on each process start. A dangling or
        corrupt live dependency is an explicit conflict; silently dropping it could resume from invented state.
        """
        checkpoints = self.coordinator.checkpoints.list_workspace(self.workspace_id)
        for checkpoint in checkpoints:
            self.coordinator.validate_checkpoint_refs(checkpoint)
        return checkpoints

    def recover_pending(self) -> tuple[RecoveryResult, ...]:
        self._ensure_open()
        results = []
        for journal in PendingTurnJournal.pending(self.store_root):
            try:
                try:
                    snapshot = journal.snapshot()
                except JournalCorruptError:
                    snapshot = journal.salvage_torn_tail()
                checkpoint = None
                if snapshot.seal_intent is None and snapshot.artifact_id.startswith("turn-"):
                    checkpoint = self._replay_unprepared_checkpoint(snapshot)
                results.append(self.coordinator.recover(
                    journal, unprepared_checkpoint=checkpoint,
                ))
            except Exception as exc:  # noqa: BLE001 - isolate one schema-incompatible journal, not BaseException
                # One malformed or schema-incompatible journal is quarantined in place and reported, but
                # cannot hide every later valid journal in lexical order.
                artifact_id = os.path.basename(journal.path).removesuffix(".jsonl")
                results.append(RecoveryResult(
                    status="conflict", artifact_id=artifact_id, detail=f"{type(exc).__name__}: {exc}",
                ))
        return tuple(results)

    def _replay_unprepared_checkpoint(self, snapshot) -> Checkpoint:
        """Rebuild state from confirmed reducer events without re-running any external tool."""
        from dataclasses import asdict
        from .events import ToolResult
        from .execution import (ToolEffect, ToolInvocation, ToolOutcome, ToolStatus,
                                coerce_tool_status, reconciliation_targets)
        from .intent import TurnAdmission
        from .active_work import WorkGraph
        from .pfc import Slice, record_user, slice_sink
        from .session import apply_turn_continuation
        from .taskstate import (slice_to_task_state, task_state_from_checkpoint,
                                task_state_to_slice)

        header = snapshot.header
        task_id = str(header.get("task_id") or "unknown-task")
        base_generation = snapshot.base_generation
        base = self.coordinator.checkpoints.load(self.workspace_id, task_id)
        if base_generation:
            if base is None or base.generation != base_generation:
                raise PersistenceError(
                    f"cannot replay turn from base generation {base_generation}; current base is "
                    f"{getattr(base, 'generation', 0)}")
            state = task_state_to_slice(task_state_from_checkpoint(base))
        else:
            state = Slice()
            state.reset(str(header.get("user_request") or ""))

        admission_event = snapshot.event("turn-admission")
        envelope = admission_event.get("payload") if admission_event is not None else {}
        envelope = envelope if isinstance(envelope, Mapping) else {}
        raw_admission = envelope.get("admission")
        raw_admission = raw_admission if isinstance(raw_admission, Mapping) else envelope
        admission = TurnAdmission.from_dict(raw_admission) if admission_event is not None else None
        if admission_event is not None and admission is None:
            raise PersistenceError("journal contains an invalid turn admission")
        request = str(header.get("user_request") or "")
        if admission is not None and admission.request_text != request:
            raise PersistenceError("journal admission request does not match the turn header")

        # An unsealed turn is unresolved by definition.  A previously provisional objective becomes active
        # again before replay so crash recovery cannot publish it as mere background.
        state.task.activate_objective()
        action = str(envelope.get("action") or "continue")
        segment_event = snapshot.event("logical-segment")
        segment_payload = segment_event.get("payload") if segment_event is not None else {}
        segment_payload = segment_payload if isinstance(segment_payload, Mapping) else {}
        logical_id = str(
            envelope.get("logical_turn_id") or segment_payload.get("logical_turn_id")
            or snapshot.artifact_id
        )
        source_event_id = str(envelope.get("source_event_id") or "")
        workspace_epoch = int(
            envelope.get("workspace_epoch", segment_payload.get("workspace_epoch", 0)) or 0
        )
        carried_work = envelope.get("active_work") if action == "workspace_continue" else None
        if carried_work is not None:
            # The target store's base checkpoint predates the cross-workspace publication. Recreate the normal
            # conversation/intent admission below, then restore the exact source-linked graph copied into the
            # target journal before its first provider call.
            state.active_work = WorkGraph()
        record_user(
            state, request,
            source_artifact=snapshot.artifact_id,
            source_event_id=source_event_id,
            logical_id=logical_id,
            workspace_epoch=workspace_epoch,
            source_text=str(envelope.get("source_event_text") or request),
            contract=admission,
        )
        if carried_work is not None:
            state.active_work = WorkGraph.from_records(carried_work)
        if action != "new":
            apply_turn_continuation(
                state, request, resume=(action == "resume"), admission=admission,
            )
        requests = {}
        invocations = {}
        outcomes = []
        applied = set()
        for event in snapshot.events:
            payload = event.get("payload", {})
            if event.get("type") == "tool-requested":
                invocation_id = str(payload.get("invocation_id") or "")
                if invocation_id:
                    requests[invocation_id] = payload
            elif event.get("type") == "tool-invocation":
                invocation_id = str(payload.get("invocation_id") or "")
                if invocation_id:
                    invocations[invocation_id] = payload
                    requests.setdefault(invocation_id, payload)
            elif event.get("type") == "tool-outcome":
                outcomes.append(payload)
            elif event.get("type") == "semantic-transition":
                transition_id = str(payload.get("transition_id") or "")
                if transition_id:
                    applied.add(transition_id)

        outcome_ids = {
            str(payload.get("invocation_id") or "") for payload in outcomes
            if isinstance(payload, Mapping) and payload.get("invocation_id")
        }
        unresolved_ids = []
        valid_statuses = {status.value for status in ToolStatus}

        def _valid_status(value) -> bool:
            return isinstance(value, bool) or (isinstance(value, str) and value in valid_statuses)

        for payload in outcomes:
            if not isinstance(payload, Mapping):
                continue
            record = payload.get("outcome") or {}
            raw_status = record.get("status") if isinstance(record, Mapping) else None
            status_valid = _valid_status(raw_status)
            if not status_valid or coerce_tool_status(raw_status) is ToolStatus.INDETERMINATE:
                unresolved_ids.append(str(payload.get("invocation_id") or "unknown"))
        missing_ids = sorted(set(invocations) - outcome_ids)
        uncertain_ids = tuple(dict.fromkeys((*unresolved_ids, *missing_ids)))
        uncertainty_detail = ""
        uncertainty_targets = []
        if uncertain_ids:
            details = []
            if unresolved_ids:
                details.append("indeterminate or invalid outcomes: " + ", ".join(unresolved_ids))
            if missing_ids:
                details.append("invocations without conclusive outcomes: " + ", ".join(missing_ids))
            uncertainty_detail = (
                "; ".join(details) + "; do not claim a result without fresh evidence"
            )
            for invocation_id in uncertain_ids:
                invocation = invocations.get(invocation_id) or requests.get(invocation_id) or {}
                args = invocation.get("args") if isinstance(invocation, Mapping) else {}
                uncertainty_targets.extend(reconciliation_targets(
                    str(invocation.get("name") or "") if isinstance(invocation, Mapping) else "",
                    args if isinstance(args, Mapping) else {},
                ))

        reducer = slice_sink(state)
        replayed = []
        for payload in outcomes:
            invocation_id = str(payload.get("invocation_id") or "")
            invocation_record = invocations.get(invocation_id) or requests.get(invocation_id) or {}
            outcome_record = payload.get("outcome") or {}
            raw_status = outcome_record.get("status") if isinstance(outcome_record, Mapping) else None
            if (not _valid_status(raw_status)
                    or coerce_tool_status(raw_status) is ToolStatus.INDETERMINATE):
                continue
            raw_effects = outcome_record.get("effects") or ()
            effects = tuple(ToolEffect(
                str(effect.get("id") or ""), str(effect.get("kind") or "tool_outcome"),
                dict(effect.get("payload") or {}),
            ) for effect in raw_effects if isinstance(effect, Mapping) and effect.get("id"))
            effect_ids = {effect.id for effect in effects}
            # Outcome journaling precedes reduction. Only a complete transition set proves publication;
            # a partial multi-effect append is treated as unapplied rather than replayed ambiguously.
            if not effect_ids or not effect_ids.issubset(applied):
                continue
            invocation = ToolInvocation(
                invocation_id, str(invocation_record.get("name") or ""),
                dict(invocation_record.get("args") or {}), len(replayed),
            )
            outcome = ToolOutcome(
                invocation, coerce_tool_status(outcome_record.get("status")),
                str(outcome_record.get("text") or ""), effects,
            )
            reducer(ToolResult(
                invocation.name, dict(invocation.args), outcome.text, outcome.failing,
                status=outcome.status.value, invocation_id=invocation.id, outcome=outcome,
            ))
            replayed.extend(effect.id for effect in effects)

        # Replay can legitimately clear an older uncertainty marker. Crash uncertainty from THIS turn is
        # applied last so a later missing/invalid effect cannot be erased by an earlier committed
        # reconcile_execution transition in the same journal. Read-only unknowns stay receipt-only.
        if uncertainty_detail and uncertainty_targets:
            if uncertainty_detail not in state.reconciliation_required:
                state.reconciliation_required = " | ".join(
                    item for item in (state.reconciliation_required, uncertainty_detail) if item
                )
            state.reconciliation_targets = list(dict.fromkeys((
                *state.reconciliation_targets, *uncertainty_targets,
            )))

        state.seal()
        task_state = slice_to_task_state(
            state, task_id, session_id=str(header.get("session_id") or self.session_id),
            status="indeterminate" if state.reconciliation_required else "parked",
            workspace_epoch=workspace_epoch,
        )
        discovered_refs = tuple(
            child.id for child in self.coordinator.artifacts.list_all()
            if child.parent_id == snapshot.artifact_id
        )
        source_refs = tuple(dict.fromkeys(
            source for source in (
                state.task.goal_source,
                *(entry.source_artifact for entry in state.intent.entries),
            )
            if source and self.coordinator.artifacts.exists(source)
        ))
        return Checkpoint.create(
            workspace_id=str(header.get("workspace_id") or self.workspace_id),
            session_id=str(header.get("session_id") or self.session_id), task_id=task_id,
            generation=base_generation + 1, state=asdict(task_state),
            artifact_refs=tuple(dict.fromkeys((
                snapshot.artifact_id, *snapshot.artifact_refs, *source_refs, *discovered_refs,
            ))),
            applied_transition_ids=tuple(dict.fromkeys(replayed)), workspace_versions={},
            order_ns=snapshot.order_ns,
            updated_at=str(header.get("created_at") or ""),
        )

    def recover_active_seal(self) -> RecoveryResult | None:
        """Retry a prepared failed seal; never guess state for an unprepared active turn."""
        # Recovery publishes the checkpoint frozen in ``seal-intent`` and retires the active turn. Serialize
        # that operation with bound child publication exactly like the ordinary seal path; once intent exists,
        # publishers are rejected rather than appending a reference the frozen checkpoint cannot contain.
        with self._state_lock:
            self._ensure_open()
            if self.active is None or self.active.journal.snapshot().seal_intent is None:
                return None
            result = self.coordinator.recover(self.active.journal)
            if result.status in ("replayed", "attached", "cleaned"):
                self.active = None
                self._active_refs = []
            return result


class CoreArtifactFS:
    """Read-only local index with exact-ID faulting across workspace artifact stores."""

    MOUNT = "artifacts"
    _SAFE_ARTIFACT_ID = r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}"
    _EVIDENCE_INDEX = re.compile(rf"^({_SAFE_ARTIFACT_ID})/evidence/index\.md$")
    _EVIDENCE_PAGE = re.compile(
        rf"^({_SAFE_ARTIFACT_ID})/evidence/obs-(\d{{3,}})-page-(\d{{3,}})\.md$"
    )
    _REPORT_INDEX = re.compile(rf"^({_SAFE_ARTIFACT_ID})/report/index\.md$")
    _REPORT_PAGE = re.compile(rf"^({_SAFE_ARTIFACT_ID})/report/page-(\d{{3,}})\.md$")
    # Virtual reads bypass LocalToolHost's ordinary file line-windowing. Keep each evidence document small
    # enough to be a genuine page while the immutable root artifact retains the complete reconstructed text.
    EVIDENCE_PAGE_BYTES = 24 * 1024
    REPORT_PAGE_BYTES = 24 * 1024
    EVIDENCE_ARGS_DISPLAY_CHARS = 800
    _LEGACY_CAPSULE_MARKER = "sealed observation view truncated by capsule budget"

    def __init__(self, artifact_store, *, archive_root: str = ""):
        self.store = artifact_store
        self.archive_root = os.path.realpath(archive_root) if archive_root else ""

    @staticmethod
    def _leaf(path: str) -> str:
        value = str(path or "").replace("\\", "/").strip("/")
        if value == "artifacts":
            return ""
        return value[len("artifacts/"):] if value.startswith("artifacts/") else value

    def _artifacts(self):
        # Discovery stays workspace-local. Federation is an exact-handle read seam, not a way to inject or list
        # unrelated workspace history into the slice.
        return self.store.list_all()

    def _get(self, artifact_id: str):
        try:
            return self.store.get(artifact_id)
        except ArtifactNotFoundError:
            pass
        if not self.archive_root:
            raise ArtifactNotFoundError(f"no artifact {artifact_id!r}")
        shard = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()[:2]
        try:
            names = sorted(os.listdir(self.archive_root))
        except OSError as exc:
            raise ArtifactNotFoundError(f"no artifact {artifact_id!r}") from exc
        local_root = os.path.realpath(getattr(self.store, "root", ""))
        for name in names:
            store_root = os.path.realpath(os.path.join(self.archive_root, name))
            if store_root == local_root or not os.path.isfile(os.path.join(
                    store_root, "artifacts", shard, artifact_id + ".json")):
                continue
            # ArtifactStore.get revalidates ID, schema, and immutable bytes. A corrupt exact match remains a
            # visible evidence failure; it is never silently replaced by another same-ID candidate.
            return ArtifactStore(store_root).get(artifact_id)
        raise ArtifactNotFoundError(f"no artifact {artifact_id!r}")

    def get_artifact(self, artifact_id: str):
        """Resolve one exact content-verified artifact through the same federation used by read_file."""
        return self._get(artifact_id)

    @staticmethod
    def _name(artifact) -> str:
        return artifact.id + ".md"

    @staticmethod
    def _reference_handle(raw: object) -> str:
        value = str(raw or "").strip().replace("\\", "/")
        # Current records store exact artifact identities. Be defensive with legacy records that already
        # stored a rendered artifacts/<id>.md handle so the user never sees a doubled locator.
        if value.startswith("artifacts/") and value.endswith(".md"):
            return value
        return f"artifacts/{value}.md"

    @staticmethod
    def _utf8_chunks(text: object, page_bytes: int) -> tuple[str, ...]:
        """Split retained text into lossless, deterministic UTF-8-safe virtual pages."""
        value = str(text or "")
        if not value:
            return ("",)
        chunks: list[str] = []
        current: list[str] = []
        used = 0
        for char in value:
            size = len(char.encode("utf-8"))
            if current and used + size > page_bytes:
                chunks.append("".join(current))
                current, used = [], 0
            current.append(char)
            used += size
        if current:
            chunks.append("".join(current))
        return tuple(chunks)

    @classmethod
    def _evidence_chunks(cls, text: object) -> tuple[str, ...]:
        """Split one retained workspace view into fixed-size evidence pages."""
        return cls._utf8_chunks(text, cls.EVIDENCE_PAGE_BYTES)

    @classmethod
    def _report_chunks(cls, text: object) -> tuple[str, ...]:
        """Use the same UTF-8-safe deterministic paging discipline as workspace evidence."""
        return cls._utf8_chunks(text, cls.REPORT_PAGE_BYTES)

    @staticmethod
    def _subagent_report(artifact) -> str:
        body = getattr(artifact, "structured_body", {}) or {}
        return str(body.get("report") or "") if isinstance(body, Mapping) else ""

    @classmethod
    def _report_metadata(cls, artifact) -> dict[str, object]:
        body = getattr(artifact, "structured_body", {}) or {}
        body = body if isinstance(body, Mapping) else {}
        report = cls._subagent_report(artifact)
        encoded = report.encode("utf-8")
        completion = str(body.get("report_completion") or "unknown")
        if completion not in {"complete", "partial", "absent", "unknown"}:
            completion = "unknown"
        if not report:
            completion = "absent"
        return {
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "completion": completion,
            "stop_reason": str(body.get("report_stop_reason") or "unknown"),
        }

    @staticmethod
    def _report_page_name(page: int) -> str:
        return f"page-{page:03d}.md"

    @staticmethod
    def _subagent_observations(artifact) -> tuple[Mapping, ...]:
        body = getattr(artifact, "structured_body", {}) or {}
        rows = body.get("observations") if isinstance(body, Mapping) else ()
        if not isinstance(rows, (list, tuple)):
            return ()
        return tuple(row for row in rows if isinstance(row, Mapping))

    @classmethod
    def _legacy_capsule_partial(cls, row: Mapping) -> bool:
        return cls._LEGACY_CAPSULE_MARKER in str(row.get("view") or "").casefold()

    @classmethod
    def _display_args(cls, row: Mapping) -> str:
        """Bound artifact metadata without modifying the exact args retained in canonical JSON."""
        args = row.get("args") if isinstance(row.get("args"), Mapping) else {}
        value = json.dumps(dict(args), ensure_ascii=False, sort_keys=True, default=str)
        if len(value) > cls.EVIDENCE_ARGS_DISPLAY_CHARS:
            omitted = len(value) - cls.EVIDENCE_ARGS_DISPLAY_CHARS
            value = value[:cls.EVIDENCE_ARGS_DISPLAY_CHARS] + f"…[args display omitted {omitted} chars]"
        return value

    @classmethod
    def _evidence_page_name(cls, observation: int, page: int) -> str:
        return f"obs-{observation:03d}-page-{page:03d}.md"

    @staticmethod
    def _status(artifact) -> str:
        body = dict(getattr(artifact, "structured_body", {}) or {})
        receipt = body.get("turn_receipt")
        if isinstance(receipt, Mapping) and receipt.get("disposition"):
            return str(receipt.get("disposition"))
        return str(getattr(artifact, "status", "unknown") or "unknown")

    @classmethod
    def _render_subagent(cls, artifact) -> str:
        """Render the complete child report plus a locator-grade evidence manifest.

        Full observation bodies deliberately do not appear on this page. They remain in the same immutable
        artifact and are projected through fixed-size evidence pages, so opening a report never floods the
        parent context while every child-visible byte remains recoverable.
        """
        body = dict(artifact.structured_body)
        brief = body.get("brief") if isinstance(body.get("brief"), Mapping) else artifact.brief
        brief = dict(brief or {})
        report = str(body.get("report") or "")
        report_meta = cls._report_metadata(artifact)
        report_chunks = cls._report_chunks(report)
        report_is_paged = int(report_meta["bytes"]) > cls.REPORT_PAGE_BYTES
        observations = cls._subagent_observations(artifact)
        legacy_partial = sum(cls._legacy_capsule_partial(row) for row in observations)
        evidence_account = body.get("explorer_evidence")
        lines = [
            f"# SUBAGENT REPORT — {artifact.title or artifact.id}",
            f"- id: {artifact.id}", f"- task: {artifact.task_id}",
            f"- status: {artifact.status}", f"- timestamp: {artifact.timestamp or '(unknown)'}",
        ]
        objective = str(brief.get("objective") or brief.get("task") or "")
        if objective:
            lines += ["", "## Delegated objective", objective]
        scope = brief.get("scope")
        if isinstance(scope, (list, tuple)) and scope:
            lines += ["", "## Declared scope", *[f"- {item}" for item in scope]]
        exclusions = brief.get("exclusions")
        if isinstance(exclusions, (list, tuple)) and exclusions:
            lines += ["", "## Declared exclusions", *[f"- {item}" for item in exclusions]]
        lines += [
            "", "## Child report envelope",
            f"- completion: {report_meta['completion']}",
            f"- stop reason: {report_meta['stop_reason']}",
            f"- UTF-8 bytes: {report_meta['bytes']}",
            f"- sha256: {report_meta['sha256']}",
        ]
        if report_is_paged:
            lines += [
                f"- oversized report: {len(report_chunks)} deterministic page(s); testimony is not inlined here",
                f'- report index: read_file("artifacts/{artifact.id}/report/index.md")',
            ]
        else:
            lines += ["", "## Child report (verbatim testimony)",
                      report or "(no child report text was sealed)"]
        lines += [
            "", "## Page-backed workspace evidence",
            (f"- {len(observations)} determinate child-visible tool result(s) are retained after mandatory "
             "secret redaction."),
            "- These pages prove what the child observed; they do not prove the child's interpretation.",
            (f'- evidence index: read_file("artifacts/{artifact.id}/evidence/index.md")'
             if observations else "- evidence index: (no physical workspace observation was sealed)"),
        ]
        if observations and not legacy_partial:
            lines.append("- Every returned tool view in this artifact is recoverable exactly from the evidence pages.")
        if legacy_partial:
            lines.append(
                f"- legacy archive partial: {legacy_partial} observation(s) were capsule-truncated by an older "
                "SliceAgent before sealing; their omitted bytes are not recoverable."
            )
        if isinstance(evidence_account, Mapping):
            lines += ["", "## Host evidence account", "```json",
                      json.dumps(dict(evidence_account), ensure_ascii=False, indent=2, default=str), "```"]
        coverage = str(body.get("coverage") or "")
        if coverage:
            lines += ["", "## Child coverage statement", coverage]
        claims = body.get("claims")
        if isinstance(claims, (list, tuple)) and claims:
            by_hash = {
                str(row.get("view_sha256") or ""): index
                for index, row in enumerate(observations, start=1)
            }
            lines += ["", "## Legacy/explicit child claims (testimony; verify before promotion)"]
            for claim in claims:
                if not isinstance(claim, Mapping):
                    continue
                exact = str(claim.get("report_exact") or claim.get("text") or "")
                refs = [by_hash.get(str(ref)) for ref in (claim.get("observation_refs") or ())]
                refs = [ref for ref in refs if ref is not None]
                locator = (" · candidate observations: " + ", ".join(f"#{ref}" for ref in refs)) if refs else ""
                lines.append(f"- {exact or '(empty indexed claim)'}{locator}")
        for heading, key in (
            ("Coverage gaps", "gaps"), ("Source gaps", "source_gaps"),
            ("Projection gaps", "projection_gaps"),
            ("Uncertainty", "uncertainty"), ("Files touched by tools", "files"),
        ):
            values = body.get(key)
            if not isinstance(values, (list, tuple)) and key == "files":
                values = artifact.files
            if isinstance(values, (list, tuple)) and values:
                lines += ["", f"## {heading}", *[f"- {item}" for item in values]]
        if artifact.error:
            lines += ["", "## Error", artifact.error]
        if artifact.refs:
            lines += ["", "## References", *[
                f'- read_file("{cls._reference_handle(ref)}")' for ref in artifact.refs
            ]]
        return "\n".join(lines)

    @classmethod
    def _render_evidence_index(cls, artifact) -> str:
        observations = cls._subagent_observations(artifact)
        lines = [
            f"# SUBAGENT WORKSPACE EVIDENCE — {artifact.id}",
            "Exact redacted tool results delivered to the child, ordered by observation time.",
            "A source-partial marker means the tool itself returned only a page/truncated view.",
            "Parent synthesis should verify material claims against these bytes and re-open live code when needed.",
        ]
        for index, row in enumerate(observations, start=1):
            chunks = cls._evidence_chunks(row.get("view"))
            tool = str(row.get("tool") or "unknown")
            status = str(row.get("status") or "unknown")
            args = cls._display_args(row)
            flags = []
            if bool(row.get("redacted")):
                flags.append("redacted")
            if cls._legacy_capsule_partial(row):
                flags.append("legacy-archive-partial")
            elif bool(row.get("truncated")):
                flags.append("source-partial")
            flags.append(f"{int(row.get('view_bytes') or len(str(row.get('view') or '').encode('utf-8')))} bytes")
            flags.append(f"{len(chunks)} page(s)")
            lines += [
                "", f"## Observation {index} · {tool} · {status}",
                f"- args (bounded display): {args}",
                f"- retention: {', '.join(flags)}",
                f"- retained sha256: {str(row.get('view_sha256') or '(unknown)')}",
            ]
            lines.extend(
                f'- page {page}/{len(chunks)}: read_file("artifacts/{artifact.id}/evidence/'
                f'{cls._evidence_page_name(index, page)}")'
                for page in range(1, len(chunks) + 1)
            )
        if not observations:
            lines += ["", "(no determinate physical workspace observations were sealed)"]
        return "\n".join(lines)

    @classmethod
    def _render_evidence_page(cls, artifact, observation_number: int, page_number: int) -> str:
        observations = cls._subagent_observations(artifact)
        if observation_number < 1 or observation_number > len(observations):
            raise ArtifactNotFoundError("no such child evidence observation")
        row = observations[observation_number - 1]
        chunks = cls._evidence_chunks(row.get("view"))
        if page_number < 1 or page_number > len(chunks):
            raise ArtifactNotFoundError("no such child evidence page")
        args = cls._display_args(row)
        flags = []
        if bool(row.get("redacted")):
            flags.append("redacted")
        if cls._legacy_capsule_partial(row):
            flags.append("legacy-archive-partial")
        elif bool(row.get("truncated")):
            flags.append("source-partial")
        return "\n".join([
            f"# CHILD EVIDENCE {artifact.id} · observation {observation_number} · "
            f"page {page_number}/{len(chunks)}",
            f"- tool: {str(row.get('tool') or 'unknown')}",
            f"- status: {str(row.get('status') or 'unknown')}",
            f"- args (bounded display): {args}",
            f"- retained sha256: {str(row.get('view_sha256') or '(unknown)')}",
            f"- flags: {', '.join(flags) if flags else 'complete returned view'}",
            "", "## Exact retained tool output", chunks[page_number - 1],
        ])

    @classmethod
    def _render_report_index(cls, artifact) -> str:
        report = cls._subagent_report(artifact)
        chunks = cls._report_chunks(report)
        meta = cls._report_metadata(artifact)
        lines = [
            f"# CHILD REPORT PAGES — {artifact.id}",
            f"- completion: {meta['completion']}",
            f"- stop reason: {meta['stop_reason']}",
            f"- UTF-8 bytes: {meta['bytes']}",
            f"- sha256: {meta['sha256']}",
            f"- pages: {len(chunks)}",
        ]
        lines.extend(
            f'- page {page}/{len(chunks)}: read_file("artifacts/{artifact.id}/report/'
            f'{cls._report_page_name(page)}")'
            for page in range(1, len(chunks) + 1)
        )
        return "\n".join(lines)

    @classmethod
    def _render_report_page(cls, artifact, page_number: int) -> str:
        report = cls._subagent_report(artifact)
        chunks = cls._report_chunks(report)
        if page_number < 1 or page_number > len(chunks):
            raise ArtifactNotFoundError("no such child report page")
        meta = cls._report_metadata(artifact)
        return "\n".join([
            f"# CHILD REPORT {artifact.id} · page {page_number}/{len(chunks)}",
            f"- completion: {meta['completion']}",
            f"- full report sha256: {meta['sha256']}",
            "", "## Exact retained child report", chunks[page_number - 1],
        ])

    @staticmethod
    def _render(artifact) -> str:
        body = dict(artifact.structured_body)
        if str(getattr(artifact, "kind", "")) == "subagent" \
                and ("report" in body or "observations" in body):
            return CoreArtifactFS._render_subagent(artifact)
        markdown = body.get("markdown")
        request = str((dict(artifact.brief).get("request") if artifact.brief else "") or "")
        assistant = str(body.get("assistant") or "")
        lines = [
            f"# {artifact.kind.upper()} ARTIFACT — {artifact.title or artifact.id}",
            f"- id: {artifact.id}", f"- task: {artifact.task_id}",
            f"- status: {artifact.status}", f"- timestamp: {artifact.timestamp or '(unknown)'}",
        ]
        if request:
            lines += ["", "## User request (verbatim)", request]
        if assistant:
            lines += ["", "## Assistant response (verbatim)", assistant]
        if artifact.summary:
            lines += ["", "## Summary", artifact.summary]
        receipt = body.get("turn_receipt")
        if isinstance(receipt, Mapping):
            counts = receipt.get("counts")
            counts = counts if isinstance(counts, Mapping) else {}
            lines += [
                "", "## Execution receipt (canonical)",
                f"- turn disposition: {receipt.get('disposition') or '(unknown)'}",
                (f"- requested: {counts.get('requested', 0)} · execution started: "
                 f"{counts.get('execution_started', 0)} · rejected before execution: "
                 f"{counts.get('rejected_before_execution', 0)} · settled: {counts.get('settled', 0)}"),
                (f"- succeeded: {counts.get('succeeded', 0)} · steered: {counts.get('steered', 0)} · "
                 f"failed: {counts.get('failed', 0)} · "
                 f"cancelled: {counts.get('cancelled', 0)} · indeterminate: "
                 f"{counts.get('indeterminate', 0)}"),
            ]
            warnings = receipt.get("warnings")
            if isinstance(warnings, (list, tuple)) and warnings:
                lines += ["- warnings:", *[f"  - {warning}" for warning in warnings]]
            operations = receipt.get("operations")
            by_tool: dict[str, dict[str, int]] = {}
            for operation in operations if isinstance(operations, (list, tuple)) else ():
                if not isinstance(operation, Mapping):
                    continue
                name = str(operation.get("name") or "(unknown tool)")
                bucket = by_tool.setdefault(name, {
                    "requested": 0, "rejected": 0, "started": 0,
                    "succeeded": 0, "steered": 0, "failed": 0,
                    "cancelled": 0, "indeterminate": 0,
                })
                bucket["requested"] += int(bool(operation.get("requested")))
                bucket["rejected"] += int(bool(operation.get("rejected_before_execution")))
                bucket["started"] += int(bool(operation.get("execution_started")))
                disposition = str(operation.get("disposition") or "")
                if disposition in {"succeeded", "steered", "failed", "cancelled", "indeterminate"}:
                    bucket[disposition] += 1
            if by_tool:
                lines.append("- by tool:")
                for name, bucket in sorted(by_tool.items()):
                    lines.append(
                        f"  - {name} · requested {bucket['requested']} · started {bucket['started']} · "
                        f"rejected {bucket['rejected']} · succeeded {bucket['succeeded']} · "
                        f"steered {bucket['steered']} · failed {bucket['failed']} · "
                        f"cancelled {bucket['cancelled']} · "
                        f"indeterminate {bucket['indeterminate']}"
                    )
        if markdown:
            lines += ["", "## Record", str(markdown)]
        else:
            lines += ["", "## Structured record", "```json",
                      json.dumps(body, ensure_ascii=False, indent=2, default=str), "```"]
        if artifact.refs:
            lines += ["", "## References", *[
                f'- read_file("{CoreArtifactFS._reference_handle(ref)}")' for ref in artifact.refs
            ]]
        anchors = body.get("anchors")
        if isinstance(anchors, (list, tuple)) and anchors:
            lines += ["", "## Addressable output anchors"]
            for raw in anchors:
                if not isinstance(raw, Mapping):
                    continue
                collection = str(raw.get("collection") or "numbered list")
                ordinal = raw.get("ordinal")
                label = str(raw.get("label") or raw.get("excerpt") or "")
                if ordinal and label:
                    lines.append(f"- {collection} #{ordinal}: {label}")
        return "\n".join(lines)

    def index(self) -> str:
        artifacts = self._artifacts()
        lines = ["# LOCAL ARTIFACTS — immutable turn and subagent records"]
        lines += [
            f'- {self._name(item)} · {self._status(item)} · '
            f'{str((dict(item.brief).get("request") if item.brief else "") or item.title or item.task_id)[:100]} '
            f'→ read_file("artifacts/{self._name(item)}")'
            for item in artifacts
        ]
        gaps = tuple(getattr(artifacts, "gaps", ()) or ())
        if gaps:
            lines += ["", "## Unreadable artifact records (evidence gaps)"]
            for gap in gaps[:8]:
                artifact_id = " ".join(str(getattr(gap, "artifact_id", "unknown")).split())[:120]
                reason = " ".join(str(getattr(gap, "reason", "unreadable")).split())
                reason_kind = reason.split(":", 1)[0][:80] or "unreadable"
                lines.append(f"- {artifact_id} · {reason_kind} · not available as a readable document")
            if len(gaps) > 8:
                lines.append(f"- … {len(gaps) - 8} more unreadable record(s)")
        if not artifacts:
            lines.append("(no readable artifacts yet)" if gaps else "(none yet)")
        return "\n".join(lines)

    def read_file(self, path: str) -> str:
        leaf = self._leaf(path)
        if leaf in ("", "index.md"):
            return self.index()
        evidence_index = self._EVIDENCE_INDEX.fullmatch(leaf)
        evidence_page = self._EVIDENCE_PAGE.fullmatch(leaf)
        report_index = self._REPORT_INDEX.fullmatch(leaf)
        report_page = self._REPORT_PAGE.fullmatch(leaf)
        if evidence_index or evidence_page or report_index or report_page:
            artifact_id = (evidence_index or evidence_page or report_index or report_page).group(1)
            try:
                artifact = self._get(artifact_id)
                if str(getattr(artifact, "kind", "")) != "subagent":
                    raise ArtifactNotFoundError("report/evidence pages exist only for child artifacts")
                if evidence_index:
                    return self._render_evidence_index(artifact)
                if evidence_page:
                    return self._render_evidence_page(
                        artifact, int(evidence_page.group(2)), int(evidence_page.group(3)),
                    )
                if report_index:
                    return self._render_report_index(artifact)
                return self._render_report_page(artifact, int(report_page.group(2)))
            except ArtifactNotFoundError:
                return f"artifacts/{leaf}: no such retained child report/evidence page"
        if not leaf.endswith(".md"):
            return f"artifacts/{leaf}: not an artifact file; read artifacts/index.md"
        artifact_id = leaf[:-3]
        try:
            return self._render(self._get(artifact_id))
        except Exception:
            return f"artifacts/{leaf}: no such retained artifact; read artifacts/index.md"

    def listing(self, path: str = MOUNT) -> str:
        leaf = self._leaf(path)
        if leaf in ("", "index.md"):
            return "\n".join(["index.md", *[self._name(item) for item in self._artifacts()]])
        evidence_dir = re.fullmatch(rf"({self._SAFE_ARTIFACT_ID})/evidence", leaf)
        if evidence_dir:
            try:
                artifact = self._get(evidence_dir.group(1))
                if str(getattr(artifact, "kind", "")) != "subagent":
                    raise ArtifactNotFoundError("not a child artifact")
                pages = [
                    self._evidence_page_name(observation, page)
                    for observation, row in enumerate(self._subagent_observations(artifact), start=1)
                    for page in range(1, len(self._evidence_chunks(row.get("view"))) + 1)
                ]
                return "\n".join(["index.md", *pages])
            except ArtifactNotFoundError:
                return f"artifacts/{leaf}: no such retained child evidence directory"
        report_dir = re.fullmatch(rf"({self._SAFE_ARTIFACT_ID})/report", leaf)
        if report_dir:
            try:
                artifact = self._get(report_dir.group(1))
                if str(getattr(artifact, "kind", "")) != "subagent":
                    raise ArtifactNotFoundError("not a child artifact")
                pages = [
                    self._report_page_name(page)
                    for page in range(1, len(self._report_chunks(self._subagent_report(artifact))) + 1)
                ]
                return "\n".join(["index.md", *pages])
            except ArtifactNotFoundError:
                return f"artifacts/{leaf}: no such retained child report directory"
        artifact_dir = re.fullmatch(rf"({self._SAFE_ARTIFACT_ID})", leaf)
        if artifact_dir:
            try:
                artifact = self._get(artifact_dir.group(1))
                if str(getattr(artifact, "kind", "")) == "subagent":
                    return "report/\nevidence/"
            except Exception:
                pass
        return f"artifacts/{leaf}: not an artifact directory; read artifacts/index.md"

    def _docs(self, path: str):
        leaf = self._leaf(path)
        if leaf == "":
            return [("index.md", self.index()), *[
                (self._name(item), self._render(item)) for item in self._artifacts()
            ]]
        evidence_dir = re.fullmatch(rf"({self._SAFE_ARTIFACT_ID})/evidence", leaf)
        report_dir = re.fullmatch(rf"({self._SAFE_ARTIFACT_ID})/report", leaf)
        if evidence_dir or report_dir:
            names = self.listing(path).splitlines()
            return [
                (f"{leaf}/{name}", self.read_file(f"artifacts/{leaf}/{name}"))
                for name in names if name.endswith(".md")
            ]
        return [(leaf, self.read_file(path))]

    def grep(self, pattern: str, *, path: str = MOUNT, output_mode: str = "content",
             context: int = 0, offset: int = 0, limit: int = 50) -> str:
        try:
            matcher = re.compile(pattern)
        except re.error as exc:
            return f"grep: invalid regex ({exc})."
        hits, counts = [], {}
        for name, text in self._docs(path):
            for line_no, line in enumerate(text.splitlines(), 1):
                if matcher.search(line):
                    hits.append(f"artifacts/{name}:{line_no}:{line}")
                    counts[name] = counts.get(name, 0) + 1
        if output_mode == "files_with_matches":
            rows = [f"artifacts/{name}" for name in counts]
        elif output_mode == "count":
            rows = [f"artifacts/{name}:{count}" for name, count in counts.items()]
        else:
            rows = hits
        rows = rows[offset:offset + limit]
        return "\n".join(rows) if rows else "grep: no matches found."


__all__ = ["ActiveTurn", "CoreArtifactFS", "LocalTurnStore"]
