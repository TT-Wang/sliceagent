"""Durable local persistence primitives for SliceAgent's target seal protocol.

This module is deliberately standalone.  It does not depend on memem, the CLI, the
current episodic JSONL format, or the in-memory Slice implementation.  It establishes
the three ownership boundaries described by ``CORE-DESIGN.md``:

* :class:`ArtifactStore` owns immutable turn and subagent records.
* :class:`CheckpointStore` owns the latest versioned active state.
* :class:`PendingTurnJournal` owns only an in-flight turn until it is sealed.

``SealCoordinator`` is the only object here that couples those stores.  Its commit
order is artifact first, checkpoint compare-and-swap second, journal cleanup last.
``runtime_persistence`` wires the active runtime to these primitives without changing
their ownership boundaries.

Callers must apply their configured secret-redaction/archive policy before handing
content to this layer.  Persistence preserves the supplied record exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Literal


SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_HELD_WORKSPACE_LEASES: set[str] = set()
_WORKSPACE_LEASES_GUARD = threading.Lock()


# --------------------------------------------------------------------------- errors


class PersistenceError(RuntimeError):
    """Base class for durable-state failures that callers must not silently ignore."""


class InvalidRecordError(PersistenceError):
    """A supplied artifact/checkpoint/journal record violates the storage contract."""


class ArtifactNotFoundError(PersistenceError):
    pass


class ArtifactConflictError(PersistenceError):
    """An immutable artifact ID already names different bytes."""


class ArtifactCorruptError(PersistenceError):
    pass


class CheckpointConflictError(PersistenceError):
    """A checkpoint CAS observed a generation other than the caller's base."""


class CheckpointCorruptError(PersistenceError):
    pass


class JournalNotFoundError(PersistenceError):
    pass


class JournalConflictError(PersistenceError):
    """A stable journal event ID was reused for different content."""


class JournalCorruptError(PersistenceError):
    pass


class LeaseBusyError(PersistenceError):
    """Another live SliceAgent process owns this workspace's core state."""


# --------------------------------------------------------------------------- pure helpers


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise InvalidRecordError(f"{name} must be a {'possibly-empty ' if allow_empty else 'non-empty '}string")
    return value


def _require_id(name: str, value: object) -> str:
    value = _require_text(name, value)
    if not _SAFE_ID.fullmatch(value):
        raise InvalidRecordError(f"{name} contains unsupported characters or is too long: {value!r}")
    return value


def _require_event_id(value: object) -> str:
    value = _require_text("event_id", value)
    if not _SAFE_EVENT_ID.fullmatch(value):
        raise InvalidRecordError(f"event_id contains unsupported characters or is too long: {value!r}")
    return value


def _freeze_json(value: Any) -> Any:
    """Deep-freeze a JSON value so frozen records cannot hide mutable dict/list leaves."""
    if isinstance(value, Mapping):
        out = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise InvalidRecordError("JSON object keys must be strings")
            out[key] = _freeze_json(child)
        return MappingProxyType(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise InvalidRecordError(f"value is not JSON-serializable: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(_thaw_json(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidRecordError(f"record is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def deterministic_artifact_id(*, kind: str, workspace_id: str, session_id: str,
                              task_id: str, logical_id: str) -> str:
    """Mint the same safe artifact ID for the same logical operation.

    ``logical_id`` is supplied by the turn/tool lifecycle (for example a stable turn
    or invocation ID), so the ID can be allocated before the artifact body exists.
    """
    for name, value in (("kind", kind), ("workspace_id", workspace_id), ("session_id", session_id),
                        ("task_id", task_id), ("logical_id", logical_id)):
        _require_text(name, value)
    prefix = re.sub(r"[^a-z0-9]+", "-", kind.lower()).strip("-")[:20] or "artifact"
    body = _canonical_bytes([kind, workspace_id, session_id, task_id, logical_id])
    return f"{prefix}-{hashlib.sha256(body).hexdigest()[:32]}"


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise InvalidRecordError(f"{name} must contain strings")
    return tuple(value)


# --------------------------------------------------------------------------- durable file helpers


def _private_dir(path: str) -> None:
    os.makedirs(path, mode=0o700, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _private_file(path: str) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _fsync_dir(path: str) -> None:
    """Durably publish directory-entry changes where the platform supports it."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return  # Windows and a few filesystems do not permit opening directories.
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _temp_bytes(directory: str, data: bytes, *, prefix: str) -> str:
    _private_dir(directory)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return path
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _thread_lock(path: str) -> threading.RLock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(os.path.realpath(path), threading.RLock())


class WorkspaceLease:
    """Non-blocking lifetime ownership for one workspace's local core store.

    The open descriptor is the lease. Process death releases the OS lock automatically; there is no stale
    PID/TTL heuristic. A process-local registry closes the same-process ``flock`` ambiguity as well.
    """

    def __init__(self, path: str, fd: int):
        self.path = path
        self.fd = fd
        self.closed = False

    @classmethod
    def acquire(cls, root: str) -> "WorkspaceLease":
        root = os.path.realpath(root)
        _private_dir(root)
        path = os.path.join(root, ".active.lock")
        with _WORKSPACE_LEASES_GUARD:
            if path in _HELD_WORKSPACE_LEASES:
                raise LeaseBusyError(f"workspace state is already active in this process: {root}")
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            _private_file(path)
            try:
                os.set_inheritable(fd, False)
            except (AttributeError, OSError):
                pass
            try:
                if os.name == "nt":
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"\0")
                        os.fsync(fd)
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (ImportError, OSError) as exc:
                os.close(fd)
                raise LeaseBusyError(
                    f"workspace state ownership is busy or unavailable for {root}: {exc}") from exc
            _HELD_WORKSPACE_LEASES.add(path)
        return cls(path, fd)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        finally:
            os.close(self.fd)
            with _WORKSPACE_LEASES_GUARD:
                _HELD_WORKSPACE_LEASES.discard(self.path)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@contextmanager
def _exclusive_lock(path: str):
    """Process-local + best available cross-process lock, released by process death."""
    _private_dir(os.path.dirname(path))
    lock = _thread_lock(path)
    with lock:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        _private_file(path)
        locked = False
        try:
            if os.name == "nt":
                try:
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"\0")
                        os.fsync(fd)
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    locked = True
                except (ImportError, OSError):
                    pass
            else:
                try:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    locked = True
                except (ImportError, OSError):
                    pass
            yield
        finally:
            if locked:
                if os.name == "nt":
                    try:
                        import msvcrt
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    except (ImportError, OSError):
                        pass
                else:
                    try:
                        import fcntl
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except (ImportError, OSError):
                        pass
            os.close(fd)


def _atomic_replace(path: str, data: bytes) -> None:
    directory = os.path.dirname(path)
    tmp = _temp_bytes(directory, data, prefix=".replace-")
    try:
        os.replace(tmp, path)
        _private_file(path)
        _fsync_dir(directory)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _publish_immutable(path: str, data: bytes, *, conflict: type[PersistenceError]) -> bool:
    """Publish once; identical retries succeed, different bytes never overwrite.

    Returns ``True`` for a new object and ``False`` for an idempotent retry.
    """
    lock_path = path + ".lock"
    with _exclusive_lock(lock_path):
        if os.path.exists(path):
            try:
                with open(path, "rb") as stream:
                    old = stream.read()
            except OSError as exc:
                raise conflict(f"cannot read existing immutable record {path}: {exc}") from exc
            if old == data:
                return False
            raise conflict(f"immutable record already exists with different content: {path}")
        tmp = _temp_bytes(os.path.dirname(path), data, prefix=".publish-")
        try:
            os.replace(tmp, path)
            _private_file(path)
            _fsync_dir(os.path.dirname(path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True


# --------------------------------------------------------------------------- immutable artifacts


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: str
    workspace_id: str
    session_id: str
    task_id: str
    parent_id: str = ""
    timestamp: str = ""
    status: str = "unknown"
    title: str = ""
    brief: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""
    structured_body: Mapping[str, Any] = field(default_factory=dict)
    files: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    error: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id("artifact.id", self.id)
        for name in ("kind", "workspace_id", "session_id", "task_id"):
            _require_text(f"artifact.{name}", getattr(self, name))
        for name in ("parent_id", "timestamp", "status", "title", "summary", "error"):
            _require_text(f"artifact.{name}", getattr(self, name), allow_empty=True)
        if self.schema_version != SCHEMA_VERSION:
            raise InvalidRecordError(f"unsupported artifact schema version: {self.schema_version}")
        if not isinstance(self.brief, Mapping) or not isinstance(self.structured_body, Mapping):
            raise InvalidRecordError("artifact.brief and artifact.structured_body must be objects")
        object.__setattr__(self, "brief", _freeze_json(self.brief))
        object.__setattr__(self, "structured_body", _freeze_json(self.structured_body))
        object.__setattr__(self, "files", _string_tuple(self.files, "artifact.files"))
        object.__setattr__(self, "refs", _string_tuple(self.refs, "artifact.refs"))
        object.__setattr__(self, "uncertainty", _string_tuple(self.uncertainty, "artifact.uncertainty"))

    @classmethod
    def create(cls, *, kind: str, workspace_id: str, session_id: str, task_id: str,
               logical_id: str, timestamp: str | None = None, **fields) -> "Artifact":
        return cls(id=deterministic_artifact_id(kind=kind, workspace_id=workspace_id,
                                                session_id=session_id, task_id=task_id,
                                                logical_id=logical_id),
                   kind=kind, workspace_id=workspace_id, session_id=session_id, task_id=task_id,
                   timestamp=timestamp or _now_iso(), **fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.schema_version, "id": self.id, "kind": self.kind,
            "workspace_id": self.workspace_id, "session_id": self.session_id,
            "task_id": self.task_id, "parent_id": self.parent_id,
            "timestamp": self.timestamp, "status": self.status, "title": self.title,
            "brief": _thaw_json(self.brief), "summary": self.summary,
            "structured_body": _thaw_json(self.structured_body), "files": list(self.files),
            "refs": list(self.refs), "uncertainty": list(self.uncertainty), "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Artifact":
        if not isinstance(data, Mapping):
            raise InvalidRecordError("artifact must be an object")
        return cls(
            id=data.get("id", ""), kind=data.get("kind", ""),
            workspace_id=data.get("workspace_id", ""), session_id=data.get("session_id", ""),
            task_id=data.get("task_id", ""), parent_id=data.get("parent_id", ""),
            timestamp=data.get("timestamp", ""), status=data.get("status", "unknown"),
            title=data.get("title", ""), brief=data.get("brief") or {}, summary=data.get("summary", ""),
            structured_body=data.get("structured_body") or {}, files=data.get("files") or (),
            refs=data.get("refs") or (), uncertainty=data.get("uncertainty") or (),
            error=data.get("error", ""), schema_version=int(data.get("v", SCHEMA_VERSION)),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


class ArtifactStore:
    """Content-verified immutable local artifact store."""

    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.directory = os.path.join(self.root, "artifacts")
        _private_dir(self.root)
        _private_dir(self.directory)

    def path_for(self, artifact_id: str) -> str:
        _require_id("artifact_id", artifact_id)
        shard = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()[:2]
        directory = os.path.join(self.directory, shard)
        _private_dir(directory)
        return os.path.join(directory, artifact_id + ".json")

    def put(self, artifact: Artifact) -> Artifact:
        if not isinstance(artifact, Artifact):
            raise InvalidRecordError("ArtifactStore.put requires an Artifact")
        data = _canonical_bytes(artifact.to_dict()) + b"\n"
        _publish_immutable(self.path_for(artifact.id), data, conflict=ArtifactConflictError)
        return artifact

    def exists(self, artifact_id: str) -> bool:
        return os.path.isfile(self.path_for(artifact_id))

    def get(self, artifact_id: str) -> Artifact:
        path = self.path_for(artifact_id)
        try:
            with open(path, "rb") as stream:
                raw = stream.read()
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(f"no artifact {artifact_id!r}") from exc
        except OSError as exc:
            raise ArtifactCorruptError(f"cannot read artifact {artifact_id!r}: {exc}") from exc
        try:
            artifact = Artifact.from_dict(json.loads(raw.decode("utf-8")))
        except (UnicodeError, ValueError, TypeError, PersistenceError) as exc:
            raise ArtifactCorruptError(f"artifact {artifact_id!r} is corrupt: {exc}") from exc
        if artifact.id != artifact_id:
            raise ArtifactCorruptError(f"artifact path {artifact_id!r} contains id {artifact.id!r}")
        return artifact

    def list_all(self) -> tuple[Artifact, ...]:
        """Discover immutable artifacts without maintaining a second mutable manifest."""
        out = []
        try:
            shards = sorted(os.listdir(self.directory))
        except FileNotFoundError:
            return ()
        for shard in shards:
            directory = os.path.join(self.directory, shard)
            if not os.path.isdir(directory):
                continue
            for name in sorted(os.listdir(directory)):
                if not name.endswith(".json"):
                    continue
                try:
                    out.append(self.get(name[:-5]))
                except PersistenceError:
                    continue
        return tuple(sorted(out, key=lambda item: (item.timestamp, item.id)))


# --------------------------------------------------------------------------- versioned checkpoints


@dataclass(frozen=True)
class Checkpoint:
    workspace_id: str
    session_id: str
    task_id: str
    generation: int
    state: Mapping[str, Any]
    artifact_refs: tuple[str, ...] = ()
    applied_transition_ids: tuple[str, ...] = ()
    workspace_versions: Mapping[str, Any] = field(default_factory=dict)
    updated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("workspace_id", "session_id", "task_id"):
            _require_text(f"checkpoint.{name}", getattr(self, name))
        if not isinstance(self.generation, int) or isinstance(self.generation, bool) or self.generation < 1:
            raise InvalidRecordError("checkpoint.generation must be a positive integer")
        if self.schema_version != SCHEMA_VERSION:
            raise InvalidRecordError(f"unsupported checkpoint schema version: {self.schema_version}")
        _require_text("checkpoint.updated_at", self.updated_at, allow_empty=True)
        if not isinstance(self.state, Mapping) or not isinstance(self.workspace_versions, Mapping):
            raise InvalidRecordError("checkpoint.state and checkpoint.workspace_versions must be objects")
        object.__setattr__(self, "state", _freeze_json(self.state))
        object.__setattr__(self, "artifact_refs", _string_tuple(self.artifact_refs, "checkpoint.artifact_refs"))
        object.__setattr__(self, "applied_transition_ids",
                           _string_tuple(self.applied_transition_ids, "checkpoint.applied_transition_ids"))
        object.__setattr__(self, "workspace_versions", _freeze_json(self.workspace_versions))

    @classmethod
    def create(cls, *, workspace_id: str, session_id: str, task_id: str, generation: int,
               state: Mapping[str, Any], updated_at: str | None = None, **fields) -> "Checkpoint":
        return cls(workspace_id=workspace_id, session_id=session_id, task_id=task_id,
                   generation=generation, state=state, updated_at=updated_at or _now_iso(), **fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.schema_version, "workspace_id": self.workspace_id,
            "session_id": self.session_id, "task_id": self.task_id,
            "generation": self.generation, "state": self.thawed_state(),
            "artifact_refs": list(self.artifact_refs),
            "applied_transition_ids": list(self.applied_transition_ids),
            "workspace_versions": _thaw_json(self.workspace_versions), "updated_at": self.updated_at,
        }

    def thawed_state(self) -> dict[str, Any]:
        """Return a deep-mutable copy for application-level state reconstruction.

        Checkpoints deliberately keep their canonical payload deeply frozen. Consumers must cross this
        boundary instead of applying ``dict(checkpoint.state)``, which thaws only the outer mapping and
        leaves nested records as ``MappingProxyType`` instances.
        """
        return _thaw_json(self.state)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Checkpoint":
        if not isinstance(data, Mapping):
            raise InvalidRecordError("checkpoint must be an object")
        return cls(
            workspace_id=data.get("workspace_id", ""), session_id=data.get("session_id", ""),
            task_id=data.get("task_id", ""), generation=int(data.get("generation", 0)),
            state=data.get("state") or {}, artifact_refs=data.get("artifact_refs") or (),
            applied_transition_ids=data.get("applied_transition_ids") or (),
            workspace_versions=data.get("workspace_versions") or {}, updated_at=data.get("updated_at", ""),
            schema_version=int(data.get("v", SCHEMA_VERSION)),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


class CheckpointStore:
    """One mutable generation chain per ``(workspace_id, task_id)``."""

    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.directory = os.path.join(self.root, "checkpoints")
        _private_dir(self.root)
        _private_dir(self.directory)

    def path_for(self, workspace_id: str, task_id: str) -> str:
        _require_text("workspace_id", workspace_id)
        _require_text("task_id", task_id)
        key = hashlib.sha256(_canonical_bytes([workspace_id, task_id])).hexdigest()
        return os.path.join(self.directory, key + ".json")

    def _load_path(self, path: str) -> Checkpoint | None:
        try:
            with open(path, "rb") as stream:
                raw = stream.read()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CheckpointCorruptError(f"cannot read checkpoint {path}: {exc}") from exc
        try:
            return Checkpoint.from_dict(json.loads(raw.decode("utf-8")))
        except (UnicodeError, ValueError, TypeError, PersistenceError) as exc:
            raise CheckpointCorruptError(f"checkpoint {path} is corrupt: {exc}") from exc

    def load(self, workspace_id: str, task_id: str) -> Checkpoint | None:
        checkpoint = self._load_path(self.path_for(workspace_id, task_id))
        if checkpoint is not None and (checkpoint.workspace_id != workspace_id or checkpoint.task_id != task_id):
            raise CheckpointCorruptError("checkpoint identity does not match its storage key")
        return checkpoint

    def current_generation(self, workspace_id: str, task_id: str) -> int:
        checkpoint = self.load(workspace_id, task_id)
        return checkpoint.generation if checkpoint is not None else 0

    def list_workspace(self, workspace_id: str) -> tuple[Checkpoint, ...]:
        """Discover authoritative checkpoints without silently dropping damaged state.

        The persistence root is workspace-scoped. A corrupt record therefore represents an ambiguous live
        task in this workspace even when its JSON is too damaged to recover the embedded workspace ID. Startup
        must fail closed and report that conflict instead of treating the task as if it never existed.
        """
        _require_text("workspace_id", workspace_id)
        out = []
        try:
            names = sorted(name for name in os.listdir(self.directory) if name.endswith(".json"))
        except FileNotFoundError:
            return ()
        for name in names:
            path = os.path.join(self.directory, name)
            checkpoint = self._load_path(path)
            if checkpoint is not None and os.path.realpath(path) != self.path_for(
                    checkpoint.workspace_id, checkpoint.task_id):
                raise CheckpointCorruptError(
                    f"checkpoint {path} identity does not match its storage key")
            if checkpoint is not None and checkpoint.workspace_id == workspace_id:
                out.append(checkpoint)
        return tuple(sorted(out, key=lambda item: (item.updated_at, item.task_id)))

    def compare_and_swap(self, checkpoint: Checkpoint, *, expected_generation: int) -> Checkpoint:
        if not isinstance(checkpoint, Checkpoint):
            raise InvalidRecordError("compare_and_swap requires a Checkpoint")
        if not isinstance(expected_generation, int) or isinstance(expected_generation, bool) \
                or expected_generation < 0:
            raise InvalidRecordError("expected_generation must be a non-negative integer")
        if checkpoint.generation != expected_generation + 1:
            raise InvalidRecordError(
                f"target generation {checkpoint.generation} must equal expected+1 ({expected_generation + 1})")

        path = self.path_for(checkpoint.workspace_id, checkpoint.task_id)
        desired = _canonical_bytes(checkpoint.to_dict()) + b"\n"
        with _exclusive_lock(path + ".lock"):
            current = self._load_path(path)
            # A retry after the CAS but before journal cleanup is successful only when it names
            # the exact checkpoint bytes already committed.
            if current is not None and _canonical_bytes(current.to_dict()) + b"\n" == desired:
                return current
            generation = current.generation if current is not None else 0
            if generation != expected_generation:
                raise CheckpointConflictError(
                    f"checkpoint generation changed: expected {expected_generation}, found {generation}")
            _atomic_replace(path, desired)
        return checkpoint


# --------------------------------------------------------------------------- pending turn journal


@dataclass(frozen=True)
class JournalSnapshot:
    header: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]

    def event(self, event_id: str) -> Mapping[str, Any] | None:
        return next((event for event in self.events if event.get("event_id") == event_id), None)

    @property
    def unresolved_invocations(self) -> tuple[Mapping[str, Any], ...]:
        """Started invocations lacking one conclusive terminal outcome.

        This is intentionally a storage-level invariant: a clean seal is forbidden even if a runtime caller
        forgets to convert an interrupt into an indeterminate ToolOutcome. Legacy boolean statuses remain
        conclusive; malformed and explicit indeterminate statuses do not.
        """
        invocations: dict[str, Mapping[str, Any]] = {}
        outcomes: dict[str, object] = {}
        for event in self.events:
            payload = event.get("payload", {})
            if not isinstance(payload, Mapping):
                continue
            invocation_id = str(payload.get("invocation_id") or "")
            if not invocation_id:
                continue
            if event.get("type") == "tool-invocation":
                invocations[invocation_id] = payload
            elif event.get("type") == "tool-outcome":
                outcome = payload.get("outcome")
                outcomes[invocation_id] = (
                    outcome.get("status") if isinstance(outcome, Mapping) else None
                )
        conclusive = {"succeeded", "failed", "cancelled"}
        return tuple(
            payload for invocation_id, payload in invocations.items()
            if not (
                isinstance(outcomes.get(invocation_id), bool)
                or (isinstance(outcomes.get(invocation_id), str)
                    and outcomes.get(invocation_id) in conclusive)
            )
        )

    @property
    def seal_intent(self) -> Mapping[str, Any] | None:
        event = self.event("seal-intent")
        return event.get("payload") if event is not None else None

    @property
    def sealed(self) -> bool:
        return self.event("journal-sealed") is not None

    @property
    def artifact_id(self) -> str:
        return str(self.header.get("artifact_id") or "")

    @property
    def base_generation(self) -> int:
        return int(self.header.get("base_generation") or 0)

    @property
    def artifact_refs(self) -> tuple[str, ...]:
        """Stable dependency handoffs recorded while the turn was still in flight."""
        return tuple(dict.fromkeys(
            str(event.get("payload", {}).get("artifact_id"))
            for event in self.events
            if event.get("type") == "artifact-ref"
            and event.get("payload", {}).get("artifact_id")
        ))


class PendingTurnJournal:
    """Append-only, fsynced record for one in-flight turn.

    Stable ``event_id`` values make every append idempotent. Reusing an ID with
    different content is a hard conflict rather than an ambiguous replay.
    """

    def __init__(self, root: str, path: str):
        self.root = os.path.realpath(root)
        self.directory = os.path.join(self.root, "journals")
        self.path = path

    @classmethod
    def begin(cls, root: str, *, artifact_id: str, workspace_id: str, session_id: str,
              task_id: str, base_generation: int, user_request: str,
              created_at: str | None = None) -> "PendingTurnJournal":
        _require_id("artifact_id", artifact_id)
        for name, value in (("workspace_id", workspace_id), ("session_id", session_id),
                            ("task_id", task_id), ("user_request", user_request)):
            _require_text(name, value, allow_empty=(name == "user_request"))
        if not isinstance(base_generation, int) or isinstance(base_generation, bool) or base_generation < 0:
            raise InvalidRecordError("base_generation must be a non-negative integer")
        root = os.path.realpath(root)
        directory = os.path.join(root, "journals")
        _private_dir(root)
        _private_dir(directory)
        path = os.path.join(directory, artifact_id + ".jsonl")
        journal = cls(root, path)

        stable = {
            "v": SCHEMA_VERSION, "seq": 0, "type": "begin", "event_id": "begin",
            "artifact_id": artifact_id, "workspace_id": workspace_id, "session_id": session_id,
            "task_id": task_id, "base_generation": base_generation, "user_request": user_request,
        }
        with _exclusive_lock(path + ".lock"):
            if os.path.exists(path):
                existing = journal.snapshot().header
                for key, value in stable.items():
                    if existing.get(key) != value:
                        raise JournalConflictError(f"journal {artifact_id!r} already has different {key}")
                return journal
            header = {**stable, "created_at": created_at or _now_iso()}
            _atomic_replace(path, _canonical_bytes(header) + b"\n")
        return journal

    @classmethod
    def open(cls, root: str, artifact_id: str) -> "PendingTurnJournal":
        _require_id("artifact_id", artifact_id)
        root = os.path.realpath(root)
        path = os.path.join(root, "journals", artifact_id + ".jsonl")
        if not os.path.isfile(path):
            raise JournalNotFoundError(f"no pending journal for {artifact_id!r}")
        journal = cls(root, path)
        journal.snapshot()  # validate eagerly
        return journal

    @classmethod
    def pending(cls, root: str) -> list["PendingTurnJournal"]:
        root = os.path.realpath(root)
        directory = os.path.join(root, "journals")
        if not os.path.isdir(directory):
            return []
        return [cls(root, os.path.join(directory, name)) for name in sorted(os.listdir(directory))
                if name.endswith(".jsonl") and os.path.isfile(os.path.join(directory, name))]

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.path)

    @staticmethod
    def _decode_records(raw: bytes, path: str) -> list[dict[str, Any]]:
        lines = raw.splitlines()
        if not lines:
            raise JournalCorruptError(f"journal is empty: {path}")
        records = []
        for number, line in enumerate(lines, 1):
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise JournalCorruptError(f"journal line {number} is corrupt: {exc}") from exc
            if not isinstance(value, dict):
                raise JournalCorruptError(f"journal line {number} is not an object")
            records.append(value)
        header = records[0]
        if header.get("type") != "begin" or header.get("event_id") != "begin" or header.get("seq") != 0:
            raise JournalCorruptError("journal does not start with a valid begin record")
        seen: set[str] = set()
        previous = -1
        for record in records:
            event_id = record.get("event_id")
            seq = record.get("seq")
            if not isinstance(event_id, str) or not isinstance(seq, int) or seq <= previous or event_id in seen:
                raise JournalCorruptError("journal event IDs/sequences are not unique and monotonic")
            seen.add(event_id); previous = seq
        return records

    def _records(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, "rb") as stream:
                raw = stream.read()
        except FileNotFoundError as exc:
            raise JournalNotFoundError(f"journal no longer exists: {self.path}") from exc
        except OSError as exc:
            raise JournalCorruptError(f"cannot read journal {self.path}: {exc}") from exc
        return self._decode_records(raw, self.path)

    def salvage_torn_tail(self) -> JournalSnapshot:
        """Discard only an unambiguously torn final append and preserve its bytes for diagnosis.

        A corrupt complete line, corrupt prefix, or structurally valid unterminated record remains a hard
        conflict.  Recovery may trim only a syntactically incomplete final fragment following a fully valid,
        monotonic journal prefix; the missing outcome is then handled as indeterminate by the runtime.
        """
        with _exclusive_lock(self.path + ".lock"):
            try:
                with open(self.path, "rb") as stream:
                    raw = stream.read()
            except FileNotFoundError as exc:
                raise JournalNotFoundError(f"journal no longer exists: {self.path}") from exc
            except OSError as exc:
                raise JournalCorruptError(f"cannot read journal {self.path}: {exc}") from exc
            if not raw or raw.endswith(b"\n") or b"\n" not in raw:
                raise JournalCorruptError("journal has no salvageable torn final append")
            prefix, tail = raw.rsplit(b"\n", 1)
            if not tail:
                raise JournalCorruptError("journal has no salvageable torn final append")
            try:
                json.loads(tail.decode("utf-8"))
            except (UnicodeError, ValueError):
                pass
            else:
                raise JournalCorruptError(
                    "unterminated final journal record is syntactically complete; refusing to discard it")
            records = self._decode_records(prefix, self.path)
            _atomic_replace(self.path + ".torn", tail)
            _atomic_replace(self.path, prefix + b"\n")
            return JournalSnapshot(
                header=_freeze_json(records[0]),
                events=tuple(_freeze_json(record) for record in records[1:]),
            )

    def snapshot(self) -> JournalSnapshot:
        records = self._records()
        return JournalSnapshot(header=_freeze_json(records[0]),
                               events=tuple(_freeze_json(record) for record in records[1:]))

    def append(self, event_type: str, payload: Mapping[str, Any], *, event_id: str) -> Mapping[str, Any]:
        _require_text("event_type", event_type)
        _require_event_id(event_id)
        if not isinstance(payload, Mapping):
            raise InvalidRecordError("journal payload must be an object")
        clean_payload = _thaw_json(_freeze_json(payload))
        with _exclusive_lock(self.path + ".lock"):
            records = self._records()
            existing = next((record for record in records if record.get("event_id") == event_id), None)
            if existing is not None:
                if existing.get("type") == event_type and existing.get("payload") == clean_payload:
                    return _freeze_json(existing)
                raise JournalConflictError(f"journal event {event_id!r} already has different content")
            record = {"v": SCHEMA_VERSION, "seq": records[-1]["seq"] + 1,
                      "type": event_type, "event_id": event_id,
                      "timestamp": _now_iso(), "payload": clean_payload}
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
            try:
                data = _canonical_bytes(record) + b"\n"
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short journal write")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            _private_file(self.path)
            return _freeze_json(record)

    def record_invocation(self, invocation_id: str, *, name: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.append("tool-invocation", {"invocation_id": invocation_id, "name": name, "args": args},
                           event_id=f"invoke:{invocation_id}")

    def record_outcome(self, invocation_id: str, outcome: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.append("tool-outcome", {"invocation_id": invocation_id, "outcome": outcome},
                           event_id=f"outcome:{invocation_id}")

    def record_transition(self, transition_id: str, transition: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.append("semantic-transition", {"transition_id": transition_id, "transition": transition},
                           event_id=f"transition:{transition_id}")

    def record_artifact_ref(self, artifact_id: str) -> Mapping[str, Any]:
        """Durably hand an already-published child/source artifact into this turn's retention graph."""
        artifact_id = _require_id("artifact_id", artifact_id)
        return self.append("artifact-ref", {"artifact_id": artifact_id},
                           event_id=f"artifact-ref:{artifact_id}")

    def prepare_seal(self, artifact: Artifact, checkpoint: Checkpoint) -> Mapping[str, Any]:
        payload = {"artifact": artifact.to_dict(), "artifact_digest": artifact.digest,
                   "checkpoint": checkpoint.to_dict(), "checkpoint_digest": checkpoint.digest}
        return self.append("seal-intent", payload, event_id="seal-intent")

    def mark_artifact_written(self, artifact: Artifact) -> Mapping[str, Any]:
        return self.append("artifact-written", {"artifact_id": artifact.id, "digest": artifact.digest},
                           event_id="artifact-written")

    def mark_checkpoint_committed(self, checkpoint: Checkpoint) -> Mapping[str, Any]:
        return self.append("checkpoint-committed",
                           {"generation": checkpoint.generation, "digest": checkpoint.digest},
                           event_id="checkpoint-committed")

    def mark_sealed(self) -> Mapping[str, Any]:
        return self.append("journal-sealed", {}, event_id="journal-sealed")

    def cleanup(self) -> None:
        with _exclusive_lock(self.path + ".lock"):
            if not self.snapshot().sealed:
                raise JournalConflictError("an unsealed journal cannot be cleaned")
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                return
            _fsync_dir(self.directory)


# --------------------------------------------------------------------------- seal + recovery orchestration


SealStage = Literal["prepared", "artifact", "checkpoint", "sealed", "cleaned"]


@dataclass(frozen=True)
class SealResult:
    artifact_id: str
    checkpoint_generation: int


@dataclass(frozen=True)
class RecoveryResult:
    status: Literal["pending", "archived", "replayed", "attached", "cleaned", "conflict"]
    artifact_id: str
    checkpoint_generation: int | None = None
    detail: str = ""


class SealCoordinator:
    """Artifact-first checkpoint publication with replay-safe crash recovery."""

    def __init__(self, root: str, *, artifacts: ArtifactStore | None = None,
                 checkpoints: CheckpointStore | None = None,
                 on_stage: Callable[[SealStage], None] | None = None):
        self.root = os.path.realpath(root)
        _private_dir(self.root)
        self.artifacts = artifacts or ArtifactStore(self.root)
        self.checkpoints = checkpoints or CheckpointStore(self.root)
        self.on_stage = on_stage

    def _stage(self, stage: SealStage) -> None:
        if self.on_stage is not None:
            self.on_stage(stage)

    def begin_turn(self, *, workspace_id: str, session_id: str, task_id: str,
                   logical_id: str, user_request: str, kind: str = "turn") -> PendingTurnJournal:
        artifact_id = deterministic_artifact_id(kind=kind, workspace_id=workspace_id,
                                                session_id=session_id, task_id=task_id,
                                                logical_id=logical_id)
        base = self.checkpoints.current_generation(workspace_id, task_id)
        return PendingTurnJournal.begin(self.root, artifact_id=artifact_id,
                                        workspace_id=workspace_id, session_id=session_id,
                                        task_id=task_id, base_generation=base,
                                        user_request=user_request)

    def _validate(self, journal: PendingTurnJournal, snapshot: JournalSnapshot,
                  artifact: Artifact, checkpoint: Checkpoint) -> None:
        header = snapshot.header
        expected = {
            "artifact_id": artifact.id, "workspace_id": artifact.workspace_id,
            "session_id": artifact.session_id, "task_id": artifact.task_id,
        }
        for key, value in expected.items():
            if header.get(key) != value:
                raise JournalConflictError(f"journal {key} does not match artifact ({header.get(key)!r} != {value!r})")
        if (checkpoint.workspace_id, checkpoint.session_id, checkpoint.task_id) != \
                (artifact.workspace_id, artifact.session_id, artifact.task_id):
            raise InvalidRecordError("artifact and checkpoint identities do not match")
        if checkpoint.generation != snapshot.base_generation + 1:
            raise InvalidRecordError("checkpoint generation does not follow the journal base generation")
        if artifact.id not in checkpoint.artifact_refs:
            raise InvalidRecordError("the target checkpoint must reference the turn artifact")
        if os.path.realpath(journal.root) != self.root:
            raise InvalidRecordError("journal and seal coordinator do not share a persistence root")
        if self.artifacts.root != self.root or self.checkpoints.root != self.root:
            raise InvalidRecordError("artifact, checkpoint, and seal stores must share one persistence root")
        unresolved = snapshot.unresolved_invocations
        if unresolved:
            marker = str(checkpoint.state.get("reconciliation_required") or "")
            state_status = str(checkpoint.state.get("status") or "")
            if (not marker or artifact.status not in {"indeterminate", "interrupted"}
                    or state_status != "indeterminate"):
                ids = ", ".join(str(item.get("invocation_id") or "unknown") for item in unresolved)
                raise InvalidRecordError(
                    "cannot cleanly seal invocation(s) without conclusive outcomes: " + ids)

    def _ensure_checkpoint_refs(self, checkpoint: Checkpoint) -> None:
        missing = []
        for artifact_id in checkpoint.artifact_refs:
            if not self.artifacts.exists(artifact_id):
                missing.append(artifact_id)
                continue
            # Existence is not durability: validate canonical JSON, record identity, and workspace binding
            # before a checkpoint can make the reference authoritative.
            artifact = self.artifacts.get(artifact_id)
            if artifact.workspace_id != checkpoint.workspace_id:
                raise InvalidRecordError(
                    f"checkpoint artifact {artifact_id!r} belongs to workspace "
                    f"{artifact.workspace_id!r}, not {checkpoint.workspace_id!r}")
        if missing:
            raise ArtifactNotFoundError(
                "checkpoint references missing artifact(s): " + ", ".join(sorted(missing)))

    def validate_checkpoint_refs(self, checkpoint: Checkpoint) -> None:
        """Revalidate every dependency of an already-authoritative live checkpoint.

        Publication-time validation is not enough: users can delete or corrupt local state between sessions.
        Startup calls this before loading extensions so a dangling task graph is a visible hard conflict.
        """
        if not isinstance(checkpoint, Checkpoint):
            raise InvalidRecordError("validate_checkpoint_refs requires a Checkpoint")
        self._ensure_checkpoint_refs(checkpoint)

    def seal(self, journal: PendingTurnJournal, artifact: Artifact, checkpoint: Checkpoint,
             *, cleanup: bool = True) -> SealResult:
        snapshot = journal.snapshot()
        self._validate(journal, snapshot, artifact, checkpoint)
        journal.prepare_seal(artifact, checkpoint)
        self._stage("prepared")

        self.artifacts.put(artifact)
        self._stage("artifact")  # a crash here is recovered from the seal-intent + durable artifact
        journal.mark_artifact_written(artifact)

        self._ensure_checkpoint_refs(checkpoint)
        self.checkpoints.compare_and_swap(checkpoint, expected_generation=snapshot.base_generation)
        self._stage("checkpoint")  # a crash here sees the already-authoritative checkpoint
        journal.mark_checkpoint_committed(checkpoint)
        journal.mark_sealed()
        self._stage("sealed")
        if cleanup:
            journal.cleanup()
            self._stage("cleaned")
        return SealResult(artifact_id=artifact.id, checkpoint_generation=checkpoint.generation)

    def recover(self, journal: PendingTurnJournal, *, cleanup: bool = True,
                unprepared_checkpoint: Checkpoint | None = None) -> RecoveryResult:
        """Finish a prepared seal, or report an unprepared/conflicting journal.

        * no ``seal-intent``: preserve an already-durable child artifact, otherwise
          materialize an honest interrupted artifact from the journal;
        * prepared but no artifact: replay the immutable artifact from the journal;
        * artifact but no checkpoint: attach it with the recorded CAS;
        * checkpoint but uncleared journal: verify exact bytes and only clean up;
        * generation conflict: leave the artifact and journal quarantined for inspection.
        """
        snapshot = journal.snapshot()
        intent = snapshot.seal_intent
        if intent is None:
            # Subagents have no active-state checkpoint to publish. Their successful terminal path writes
            # the immutable artifact and then closes this journal. A process can die between those two
            # fsynced operations; in that case the artifact is already the strongest available truth and
            # must not be replaced by a conflicting synthetic "interrupted" record.
            artifact = None
            discovered_refs = tuple(
                child.id for child in self.artifacts.list_all()
                if child.parent_id == snapshot.artifact_id
            )
            if self.artifacts.exists(snapshot.artifact_id):
                try:
                    existing = self.artifacts.get(snapshot.artifact_id)
                    header = snapshot.header
                    expected = {
                        "workspace_id": str(header.get("workspace_id") or ""),
                        "session_id": str(header.get("session_id") or ""),
                        "task_id": str(header.get("task_id") or ""),
                    }
                    for field, value in expected.items():
                        if getattr(existing, field) != value:
                            raise JournalConflictError(
                                f"durable artifact {field} does not match journal "
                                f"({getattr(existing, field)!r} != {value!r})")
                    artifact = existing
                except (ArtifactCorruptError, JournalConflictError) as exc:
                    return RecoveryResult(
                        status="conflict", artifact_id=snapshot.artifact_id,
                        checkpoint_generation=snapshot.base_generation, detail=str(exc),
                    )
            header = snapshot.header
            if artifact is None:
                # Preserve the exact redacted request and every durable event. A runtime adapter may also
                # supply a checkpoint rebuilt from confirmed semantic transitions; storage itself never
                # guesses how application state should reduce.
                request = str(header.get("user_request") or "")
                kind = "subagent" if snapshot.artifact_id.startswith("subagent-") else "turn"
                recovered_refs = tuple(dict.fromkeys((*snapshot.artifact_refs, *discovered_refs)))
                recovered_body = {
                    "journal_events": [_thaw_json(event) for event in snapshot.events],
                }
                if kind == "turn":
                    # Execution lifecycle is a pure journal projection, so crash recovery can preserve it
                    # without replaying a tool or guessing application state. This gives later self-audit the
                    # same canonical source as an ordinary seal; unresolved starts remain indeterminate.
                    from .receipts import TurnReceipt
                    recovered_body["turn_receipt"] = TurnReceipt.from_events(
                        snapshot.events,
                        turn_id=snapshot.artifact_id,
                        turn_status="interrupted",
                        artifact_refs=recovered_refs,
                    ).to_dict()
                artifact = Artifact(
                    id=snapshot.artifact_id,
                    kind=kind,
                    workspace_id=str(header.get("workspace_id") or "unknown-workspace"),
                    session_id=str(header.get("session_id") or "unknown-session"),
                    task_id=str(header.get("task_id") or "unknown-task"),
                    timestamp=str(header.get("created_at") or _now_iso()), status="interrupted",
                    title=request[:120], brief={"request": request},
                    summary="Recovered after process interruption; external tools were not replayed.",
                    structured_body=recovered_body,
                    refs=recovered_refs,
                    error="process interrupted before seal",
                )
            try:
                self.artifacts.put(artifact)
                journal.mark_artifact_written(artifact)
                generation = snapshot.base_generation
                status: Literal["archived", "attached", "cleaned"] = (
                    "cleaned" if self.artifacts.exists(artifact.id) else "archived")
                if unprepared_checkpoint is not None:
                    self._validate(journal, snapshot, artifact, unprepared_checkpoint)
                    self._ensure_checkpoint_refs(unprepared_checkpoint)
                    self.checkpoints.compare_and_swap(
                        unprepared_checkpoint, expected_generation=snapshot.base_generation,
                    )
                    journal.mark_checkpoint_committed(unprepared_checkpoint)
                    generation = unprepared_checkpoint.generation
                    status = "attached"
                elif artifact.status == "interrupted":
                    status = "archived"
                journal.mark_sealed()
                if cleanup:
                    journal.cleanup()
                return RecoveryResult(status=status, artifact_id=artifact.id,
                                      checkpoint_generation=generation)
            except (ArtifactConflictError, ArtifactCorruptError, CheckpointConflictError,
                    ArtifactNotFoundError, InvalidRecordError, JournalConflictError,
                    JournalCorruptError) as exc:
                return RecoveryResult(status="conflict", artifact_id=snapshot.artifact_id,
                                      checkpoint_generation=snapshot.base_generation, detail=str(exc))
        try:
            artifact = Artifact.from_dict(intent.get("artifact") or {})
            checkpoint = Checkpoint.from_dict(intent.get("checkpoint") or {})
            if artifact.digest != intent.get("artifact_digest") \
                    or checkpoint.digest != intent.get("checkpoint_digest"):
                raise JournalCorruptError("seal-intent digest does not match its payload")
            self._validate(journal, snapshot, artifact, checkpoint)

            had_artifact = self.artifacts.exists(artifact.id)
            current = self.checkpoints.load(checkpoint.workspace_id, checkpoint.task_id)
            had_checkpoint = current is not None and current.digest == checkpoint.digest

            self.artifacts.put(artifact)
            journal.mark_artifact_written(artifact)
            self._ensure_checkpoint_refs(checkpoint)
            self.checkpoints.compare_and_swap(checkpoint, expected_generation=snapshot.base_generation)
            journal.mark_checkpoint_committed(checkpoint)
            journal.mark_sealed()
            if cleanup:
                journal.cleanup()
            status: Literal["replayed", "attached", "cleaned"] = (
                "cleaned" if had_checkpoint else "attached" if had_artifact else "replayed")
            return RecoveryResult(status=status, artifact_id=artifact.id,
                                  checkpoint_generation=checkpoint.generation)
        except (ArtifactConflictError, ArtifactCorruptError, CheckpointConflictError, ArtifactNotFoundError,
                InvalidRecordError, JournalConflictError, JournalCorruptError) as exc:
            # The immutable artifact (if already written) remains readable but unreferenced by a
            # new checkpoint: a visible quarantine, never an automatic overwrite or guessed merge.
            current_generation = self.checkpoints.current_generation(
                str(snapshot.header.get("workspace_id") or ""), str(snapshot.header.get("task_id") or ""))
            return RecoveryResult(status="conflict", artifact_id=snapshot.artifact_id,
                                  checkpoint_generation=current_generation, detail=str(exc))


__all__ = [
    "Artifact", "ArtifactStore", "Checkpoint", "CheckpointStore", "PendingTurnJournal",
    "JournalSnapshot", "SealCoordinator", "SealResult", "RecoveryResult",
    "deterministic_artifact_id", "PersistenceError", "InvalidRecordError",
    "ArtifactNotFoundError", "ArtifactConflictError", "ArtifactCorruptError",
    "CheckpointConflictError", "CheckpointCorruptError", "JournalNotFoundError",
    "JournalConflictError", "JournalCorruptError",
    "WorkspaceLease", "LeaseBusyError",
]
