"""Application-level immutable event ledger.

The active context is a derived view, not the durable history.  This ledger sits above
workspace-local turn stores so one logical user turn can cross workspace epochs without
losing the identity or exact (redacted-on-persist) bytes of its source events.

Only facts that already happened are appended here.  Replaying the ledger never calls a
model, executes a tool, changes a workspace, or mutates task state; reducers explicitly
consume events when they need to rebuild a derived view.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from .private_state import open_private_append, private_dir, private_file
from .recovery import state_dir
from .safety import redact_text


EventKind = Literal[
    "user_utterance",
    "work_delta",
    "context_transition",
    "response_delivered",
    "child_artifact",
]

_KINDS = frozenset({
    "user_utterance", "work_delta", "context_transition",
    "response_delivered", "child_artifact",
})


class LedgerError(RuntimeError):
    """Base class for durable ledger failures."""


class LedgerCorruptError(LedgerError):
    """A complete record in the immutable prefix is malformed or conflicting."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        # Preserve byte/character positions used by source ranges.  Redaction changes
        # sensitive bytes but never their length.
        return redact_text(value, preserve_length=True)
    if isinstance(value, Mapping):
        return {redact_text(str(key)): _redact(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(child) for child in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return redact_text(str(value), preserve_length=True)


def event_id(kind: str, *identity: object) -> str:
    """Content-stable event identity for idempotent boundary retries."""
    raw = json.dumps([str(kind), *identity], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:24]
    return f"{kind}-{digest}"


def _same_event(left: "LedgerEvent", right: "LedgerEvent") -> bool:
    """Whether an idempotent retry describes the same fact.

    Wall-clock capture is metadata, not event identity.  A crash-retry may rebuild
    the same deterministic event a few milliseconds later; accepting it must not
    require the caller to persist a timestamp before the first append.
    """
    a, b = left.to_dict(), right.to_dict()
    a.pop("timestamp", None); b.pop("timestamp", None)
    return a == b


@dataclass(frozen=True)
class LedgerEvent:
    id: str
    kind: EventKind
    session_id: str
    logical_turn_id: str
    task_id: str
    segment_id: str
    workspace_epoch: int
    workspace_id: str
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    timestamp: float = field(default_factory=time.time)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("id", "session_id", "logical_turn_id", "task_id"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"ledger event {name} must be non-empty")
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported ledger event kind {self.kind!r}")
        if not isinstance(self.workspace_epoch, int) or self.workspace_epoch < 0:
            raise ValueError("workspace_epoch must be a non-negative integer")
        if self.schema_version != 1:
            raise ValueError(f"unsupported ledger event version {self.schema_version}")
        if not isinstance(self.payload, Mapping):
            raise ValueError("ledger event payload must be an object")
        object.__setattr__(self, "payload", _freeze(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.schema_version,
            "id": self.id,
            "kind": self.kind,
            "session_id": self.session_id,
            "logical_turn_id": self.logical_turn_id,
            "task_id": self.task_id,
            "segment_id": self.segment_id,
            "workspace_epoch": self.workspace_epoch,
            "workspace_id": self.workspace_id,
            "timestamp": self.timestamp,
            "payload": _thaw(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LedgerEvent":
        if not isinstance(value, Mapping):
            raise ValueError("ledger event must be an object")
        return cls(
            id=str(value.get("id") or ""),
            kind=str(value.get("kind") or ""),
            session_id=str(value.get("session_id") or ""),
            logical_turn_id=str(value.get("logical_turn_id") or ""),
            task_id=str(value.get("task_id") or ""),
            segment_id=str(value.get("segment_id") or ""),
            workspace_epoch=int(value.get("workspace_epoch") or 0),
            workspace_id=str(value.get("workspace_id") or ""),
            timestamp=float(value.get("timestamp") or 0.0),
            payload=value.get("payload") or {},
            schema_version=int(value.get("v") or 1),
        )


class EventLedger:
    """Append-only, idempotent application event store for one session."""

    def __init__(self, session_id: str, *, root: str | None = None):
        self.session_id = str(session_id or "").strip()
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        directory = private_dir(root or state_dir("event-ledger"))
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in self.session_id)
        self.path = os.path.join(directory, f"{safe}.jsonl")
        self._events: list[LedgerEvent] = []
        self._by_id: dict[str, LedgerEvent] = {}
        self._archived_events: dict[str, LedgerEvent] = {}
        self._archived_sources: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "rb") as stream:
                raw = stream.read()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise LedgerError(f"cannot read event ledger {self.path}: {exc}") from exc
        private_file(self.path)
        lines = raw.splitlines(keepends=True)
        offset = 0
        for index, line in enumerate(lines, 1):
            line_start = offset
            offset += len(line)
            complete = line.endswith((b"\n", b"\r"))
            content = line.rstrip(b"\r\n")
            if not content:
                continue
            try:
                decoded = json.loads(content.decode("utf-8"))
                event = LedgerEvent.from_dict(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                # A process may die between append bytes and fsync.  Only the final,
                # newline-less fragment is recoverable; corruption in the immutable
                # prefix must stay visible.
                if index == len(lines) and not complete:
                    self._repair_tail(line_start)
                    break
                raise LedgerCorruptError(
                    f"event ledger {self.path} has corrupt complete line {index}: {exc}"
                ) from exc
            self._accept_loaded(event, line_number=index)
            # A crash can land after a complete JSON object but before its record
            # terminator. Accept the durable fact, then restore the delimiter before
            # a later append; otherwise two individually valid records concatenate.
            if index == len(lines) and not complete:
                self._repair_tail(offset, terminate=True)

    def _repair_tail(self, valid_size: int, *, terminate: bool = False) -> None:
        """Durably discard a torn suffix or terminate one complete final record."""
        try:
            fd = os.open(self.path, os.O_RDWR)
            try:
                os.ftruncate(fd, valid_size)
                if terminate:
                    os.lseek(fd, 0, os.SEEK_END)
                    os.write(fd, b"\n")
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise LedgerError(f"cannot repair event ledger tail {self.path}: {exc}") from exc
        private_file(self.path)

    def _accept_loaded(self, event: LedgerEvent, *, line_number: int) -> None:
        if event.session_id != self.session_id:
            raise LedgerCorruptError(
                f"event ledger line {line_number} belongs to session {event.session_id!r}"
            )
        existing = self._by_id.get(event.id)
        if existing is not None:
            if not _same_event(existing, event):
                raise LedgerCorruptError(f"event id {event.id!r} has conflicting records")
            return
        self._events.append(event)
        self._by_id[event.id] = event

    def append(self, event: LedgerEvent) -> LedgerEvent:
        if event.session_id != self.session_id:
            raise ValueError("event session does not match ledger session")
        existing = self._by_id.get(event.id)
        if existing is not None:
            if not _same_event(existing, event):
                raise LedgerError(f"event id {event.id!r} already names different content")
            return existing
        line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            with open_private_append(self.path) as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise LedgerError(f"cannot append event ledger {self.path}: {exc}") from exc
        self._events.append(event)
        self._by_id[event.id] = event
        return event

    def record(
        self,
        kind: EventKind,
        *,
        logical_turn_id: str,
        task_id: str,
        segment_id: str = "",
        workspace_epoch: int = 0,
        workspace_id: str = "",
        payload: Mapping[str, Any] | None = None,
        identity: Iterable[object] = (),
        timestamp: float | None = None,
    ) -> LedgerEvent:
        stable_identity = tuple(identity) or (logical_turn_id, segment_id, workspace_epoch)
        return self.append(LedgerEvent(
            id=event_id(kind, self.session_id, *stable_identity),
            kind=kind,
            session_id=self.session_id,
            logical_turn_id=str(logical_turn_id),
            task_id=str(task_id),
            segment_id=str(segment_id),
            workspace_epoch=int(workspace_epoch),
            workspace_id=str(workspace_id),
            payload=_redact(dict(payload or {})),
            **({} if timestamp is None else {"timestamp": float(timestamp)}),
        ))

    def events(self, kind: EventKind | None = None) -> tuple[LedgerEvent, ...]:
        return tuple(event for event in self._events if kind is None or event.kind == kind)

    def get(self, identity: str) -> LedgerEvent | None:
        return self._by_id.get(str(identity))

    def logical_turn(self, logical_turn_id: str) -> tuple[LedgerEvent, ...]:
        value = str(logical_turn_id)
        return tuple(event for event in self._events if event.logical_turn_id == value)

    def latest(self, kind: EventKind | None = None) -> LedgerEvent | None:
        for event in reversed(self._events):
            if kind is None or event.kind == kind:
                return event
        return None

    def user_sources(self) -> dict[str, str]:
        """Canonical text bytes addressable by Active Work source references."""
        return {
            event.id: str(event.payload.get("text"))
            for event in self._events
            if event.kind == "user_utterance" and isinstance(event.payload.get("text"), str)
        }

    def resolve_user_sources(self, event_ids: Iterable[str]) -> dict[str, str]:
        """Resolve exact requested source IDs across durable app-session ledgers.

        A task checkpoint can outlive the process/app session that admitted its request. The WorkGraph keeps
        the globally content-stable event ID, while this bounded resolver faults in only those cited events
        from sibling ledger files. It never exposes unrelated archived utterances to the context compiler.
        """
        if isinstance(event_ids, (str, bytes)):
            raise TypeError("event_ids must be an iterable of event IDs")
        requested = tuple(dict.fromkeys(value for value in event_ids if isinstance(value, str) and value))
        if not requested:
            return {}
        events = self.resolve_events(requested)
        found: dict[str, str] = {}
        for identity in requested:
            event = events.get(identity)
            if event is None or event.kind != "user_utterance":
                continue
            text = event.payload.get("text")
            if isinstance(text, str):
                found[identity] = text
                self._archived_sources[identity] = text
        return found

    def resolve_events(self, event_ids: Iterable[str]) -> dict[str, LedgerEvent]:
        """Fault exact cited events from sibling ledgers without listing unrelated archive contents."""
        if isinstance(event_ids, (str, bytes)):
            raise TypeError("event_ids must be an iterable of event IDs")
        requested = tuple(dict.fromkeys(value for value in event_ids if isinstance(value, str) and value))
        wanted = set(requested)
        found = {
            identity: event for identity, event in {**self._archived_events, **self._by_id}.items()
            if identity in wanted
        }

        directory = os.path.dirname(self.path)
        try:
            names = sorted(os.listdir(directory))
        except OSError as exc:
            raise LedgerError(f"cannot list event ledger archive {directory}: {exc}") from exc
        for name in names:
            path = os.path.join(directory, name)
            if not name.endswith(".jsonl") or path == self.path:
                continue
            try:
                with open(path, "rb") as stream:
                    lines = stream.read().splitlines(keepends=True)
            except OSError:
                continue
            private_file(path)
            for index, line in enumerate(lines, 1):
                complete = line.endswith((b"\n", b"\r"))
                content = line.rstrip(b"\r\n")
                if not content:
                    continue
                try:
                    event = LedgerEvent.from_dict(json.loads(content.decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                    # A torn tail or an unrelated corrupt archived ledger cannot contaminate another exact
                    # source. The requested event remains visibly unavailable unless a readable ledger owns it.
                    if index == len(lines) and not complete:
                        break
                    break
                if event.id not in wanted:
                    continue
                previous = found.get(event.id)
                if previous is not None and not _same_event(previous, event):
                    raise LedgerCorruptError(f"event {event.id!r} has conflicting archived records")
                found[event.id] = event
                self._archived_events[event.id] = event
            # Deliberately scan every sibling ledger even after all requested IDs
            # have been found.  Event IDs are global content identities: returning
            # early would hide a conflicting duplicate in a later archive and make
            # lexicographic file order choose which corrupted fact becomes truth.
        return {identity: found[identity] for identity in requested if identity in found}


def backfill_delivered_responses(artifacts: Iterable[object], *, root: str) -> int:
    """Idempotently project already-sealed terminal responses after a crash gap.

    The immutable turn artifact/checkpoint protocol is the commit point. This helper repairs only the
    application-ledger projection; it never calls a model or changes task state.
    """
    ledgers: dict[str, EventLedger] = {}
    repaired = 0
    for artifact in artifacts:
        if str(getattr(artifact, "kind", "")) != "turn":
            continue
        body = getattr(artifact, "structured_body", {}) or {}
        if not isinstance(body, Mapping) or not str(body.get("assistant") or "").strip():
            continue
        meta = body.get("meta") or {}
        if not isinstance(meta, Mapping) or str(meta.get("segment_outcome") or "terminal") != "terminal":
            continue
        stop_reason = str(meta.get("stop_reason") or getattr(artifact, "status", ""))
        if stop_reason != "end_turn":
            continue
        logical_id = str(meta.get("logical_turn_id") or "")
        session_id = str(getattr(artifact, "session_id", "") or "")
        task_id = str(getattr(artifact, "task_id", "") or "")
        artifact_id = str(getattr(artifact, "id", "") or "")
        if not all((logical_id, session_id, task_id, artifact_id)):
            continue
        segment_index = int(meta.get("segment_index") or 0)
        workspace_epoch = int(meta.get("workspace_epoch") or 0)
        segment_id = str(meta.get("segment_id") or f"{logical_id}:segment:{segment_index}")
        ledger = ledgers.setdefault(session_id, EventLedger(session_id, root=root))
        before = ledger.get(event_id("response_delivered", session_id, logical_id, artifact_id))
        ledger.record(
            "response_delivered", logical_turn_id=logical_id, task_id=task_id,
            segment_id=segment_id, workspace_epoch=workspace_epoch,
            workspace_id=str(getattr(artifact, "workspace_id", "") or ""),
            payload={"artifact_id": artifact_id, "stop_reason": stop_reason},
            identity=(logical_id, artifact_id),
        )
        repaired += int(before is None)
    return repaired


__all__ = [
    "EventKind", "EventLedger", "LedgerCorruptError", "LedgerError", "LedgerEvent",
    "backfill_delivered_responses", "event_id",
]
