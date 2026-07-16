"""Typed, provenance-linked L2 knowledge and its canonical SQLite repository.

This module is deliberately independent from ``memory.py`` and ``neocortex.py``.  It
defines SliceAgent-owned knowledge semantics and persistence; retrieval backends may
index these records, but they do not get to reinterpret their scope, lifecycle, or
provenance.

``KnowledgeSourceRef`` is intentionally not named ``SourceRef``.  Active Work already
owns an exact user-event ``SourceRef`` with different invariants.  A knowledge source
can instead point at any canonical evidence namespace and optionally select a byte
range or structured field.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

from .private_state import private_dir, private_file


KNOWLEDGE_SCHEMA_VERSION = 1
KNOWLEDGE_DB_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_MAX_QUERY_LIMIT = 100
_RECORD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,159}$")


class KnowledgeError(ValueError):
    """Base error for invalid knowledge records or repository operations."""


class KnowledgeValidationError(KnowledgeError):
    """A typed record violates a schema, provenance, or lifecycle invariant."""


class KnowledgeNotFoundError(KnowledgeError):
    """A requested canonical record does not exist."""


class KnowledgeConflictError(KnowledgeError):
    """An update conflicts with an existing record or lifecycle transition."""


class _StringEnum(str, Enum):
    """``StrEnum`` behavior without requiring the Python 3.11 enum helper at import time."""

    def __str__(self) -> str:
        return str(self.value)


class KnowledgeKind(_StringEnum):
    PREFERENCE = "preference"
    FACT = "fact"
    LESSON = "lesson"
    PROCEDURE = "procedure"


class KnowledgeStatus(_StringEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    EXPIRED = "expired"
    TOMBSTONED = "tombstoned"
    LEGACY_UNPROVENANCED = "legacy_unprovenanced"


class KnowledgeFreshness(_StringEnum):
    UNKNOWN = "unknown"
    CURRENT = "current"
    STALE = "stale"


class KnowledgeSensitivity(_StringEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    SECRET = "secret"


class FeedbackKind(_StringEnum):
    SERVED = "served"
    OPENED = "opened"
    CITED = "cited"
    APPLIED = "applied"
    VALIDATED_SUCCESS = "validated_success"
    CORRECTED = "corrected"
    CONTRADICTED = "contradicted"
    RETRACTED = "retracted"


_PROVENANCED_STATUSES = frozenset({
    KnowledgeStatus.ACTIVE,
    KnowledgeStatus.SUPERSEDED,
    KnowledgeStatus.RETRACTED,
    KnowledgeStatus.EXPIRED,
})

_STATUS_TRANSITIONS: dict[KnowledgeStatus, frozenset[KnowledgeStatus]] = {
    KnowledgeStatus.CANDIDATE: frozenset({
        KnowledgeStatus.CANDIDATE,
        KnowledgeStatus.ACTIVE,
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.TOMBSTONED,
    }),
    KnowledgeStatus.ACTIVE: frozenset({
        KnowledgeStatus.ACTIVE,
        KnowledgeStatus.SUPERSEDED,
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.EXPIRED,
        KnowledgeStatus.TOMBSTONED,
    }),
    KnowledgeStatus.EXPIRED: frozenset({
        KnowledgeStatus.EXPIRED,
        KnowledgeStatus.ACTIVE,
        KnowledgeStatus.SUPERSEDED,
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.TOMBSTONED,
    }),
    KnowledgeStatus.SUPERSEDED: frozenset({
        KnowledgeStatus.SUPERSEDED,
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.TOMBSTONED,
    }),
    KnowledgeStatus.RETRACTED: frozenset({
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.TOMBSTONED,
    }),
    KnowledgeStatus.LEGACY_UNPROVENANCED: frozenset({
        KnowledgeStatus.LEGACY_UNPROVENANCED,
        KnowledgeStatus.CANDIDATE,
        KnowledgeStatus.ACTIVE,
        KnowledgeStatus.RETRACTED,
        KnowledgeStatus.TOMBSTONED,
    }),
    KnowledgeStatus.TOMBSTONED: frozenset({KnowledgeStatus.TOMBSTONED}),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _text(value: object, name: str, *, allow_empty: bool = False, multiline: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "possibly-empty" if allow_empty else "non-empty"
        raise KnowledgeValidationError(f"{name} must be a {qualifier} string")
    if not multiline and ("\r" in value or "\n" in value):
        raise KnowledgeValidationError(f"{name} must not contain CR or LF")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _timestamp(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _text(value, name)  # type: ignore[arg-type]
    probe = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(probe)
    except ValueError as exc:
        raise KnowledgeValidationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise KnowledgeValidationError(f"{name} must include a timezone")
    return text


def _enum(value: object, enum_type: type[_StringEnum], name: str) -> _StringEnum:
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise KnowledgeValidationError(f"{name} must be one of: {allowed}") from exc


def _valid_digest(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != _SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise KnowledgeValidationError(f"{name} must be a lowercase sha256 digest")
    return text


def _string_tuple(value: Iterable[str] | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise KnowledgeValidationError(f"{name} must be a sequence of strings")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise KnowledgeValidationError(f"{name} must be a sequence of strings") from exc
    for item in result:
        _text(item, f"{name} item")
    if len(set(result)) != len(result):
        raise KnowledgeValidationError(f"{name} must not contain duplicates")
    return result


def _freeze_json(value: Any, name: str = "metadata") -> Any:
    """Validate JSON compatibility and detach/freeze caller-owned mutable structures."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        detached = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise KnowledgeValidationError(f"{name} must contain only JSON values") from exc

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise KnowledgeValidationError(f"{name} keys must be strings")
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(detached)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


@dataclass(frozen=True, order=True)
class KnowledgeScope:
    """Independent identity axes controlling where a knowledge record applies.

    Categories such as USER, PROJECT, and CRAFT are derived views rather than a
    mutually-exclusive enum.  A preference can therefore be both user- and
    project-specific, while an agent craft lesson can use ``agent_id``.
    """

    user_id: str | None = None
    project_id: str | None = None
    agent_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("user_id", "project_id", "agent_id"):
            value = getattr(self, name)
            if value is not None:
                _text(value, f"scopes.{name}")
        if self.user_id is None and self.project_id is None and self.agent_id is None:
            raise KnowledgeValidationError("at least one knowledge scope identity is required")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "user_id": self.user_id,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeScope":
        if not isinstance(value, Mapping):
            raise KnowledgeValidationError("scopes must be an object")
        return cls(
            user_id=value.get("user_id"),
            project_id=value.get("project_id"),
            agent_id=value.get("agent_id"),
        )


@dataclass(frozen=True, order=True)
class KnowledgeSourceRef:
    """Digest-bound locator into canonical L0 evidence.

    ``digest`` binds the complete canonical source record.  ``byte_start`` / ``byte_end``
    optionally identify a half-open byte range; ``field`` optionally identifies one
    structured field instead.  Resolution and proof interpretation remain the evidence
    archive's responsibility.
    """

    namespace: str
    record_id: str
    digest: str
    observer: str
    observed_at: str
    byte_start: int | None = None
    byte_end: int | None = None
    field: str | None = None
    project_id: str | None = None
    workspace_id: str | None = None
    resource_revision: str | None = None

    def __post_init__(self) -> None:
        _text(self.namespace, "source_ref.namespace")
        _text(self.record_id, "source_ref.record_id")
        _valid_digest(self.digest, "source_ref.digest")
        _text(self.observer, "source_ref.observer")
        _timestamp(self.observed_at, "source_ref.observed_at")
        _optional_text(self.field, "source_ref.field")
        _optional_text(self.project_id, "source_ref.project_id")
        _optional_text(self.workspace_id, "source_ref.workspace_id")
        _optional_text(self.resource_revision, "source_ref.resource_revision")
        if (self.byte_start is None) != (self.byte_end is None):
            raise KnowledgeValidationError("source_ref byte range requires both start and end")
        if self.byte_start is not None:
            if isinstance(self.byte_start, bool) or not isinstance(self.byte_start, int) or self.byte_start < 0:
                raise KnowledgeValidationError("source_ref.byte_start must be an integer >= 0")
            if isinstance(self.byte_end, bool) or not isinstance(self.byte_end, int) or self.byte_end <= self.byte_start:
                raise KnowledgeValidationError("source_ref.byte_end must be greater than byte_start")
        if self.field is not None and self.byte_start is not None:
            raise KnowledgeValidationError("source_ref selects either a byte range or a field, not both")

    @classmethod
    def bind_text(
        cls,
        namespace: str,
        record_id: str,
        source: str,
        *,
        observer: str,
        observed_at: str,
        byte_start: int | None = None,
        byte_end: int | None = None,
        field: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
        resource_revision: str | None = None,
    ) -> "KnowledgeSourceRef":
        if not isinstance(source, str) or not source:
            raise KnowledgeValidationError("source text must be a non-empty string")
        source_bytes = source.encode("utf-8")
        if byte_start is not None or byte_end is not None:
            if byte_start is None or byte_end is None or byte_end > len(source_bytes):
                raise KnowledgeValidationError("source_ref byte range must be within the UTF-8 source bytes")
        return cls(
            namespace=namespace,
            record_id=record_id,
            digest=hashlib.sha256(source_bytes).hexdigest(),
            observer=observer,
            observed_at=observed_at,
            byte_start=byte_start,
            byte_end=byte_end,
            field=field,
            project_id=project_id,
            workspace_id=workspace_id,
            resource_revision=resource_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "record_id": self.record_id,
            "digest": self.digest,
            "observer": self.observer,
            "observed_at": self.observed_at,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "field": self.field,
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
            "resource_revision": self.resource_revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeSourceRef":
        if not isinstance(value, Mapping):
            raise KnowledgeValidationError("knowledge source ref must be an object")
        try:
            return cls(
                namespace=value.get("namespace", ""),
                record_id=value.get("record_id", ""),
                digest=value.get("digest", ""),
                observer=value.get("observer", ""),
                observed_at=value.get("observed_at", ""),
                byte_start=value.get("byte_start"),
                byte_end=value.get("byte_end"),
                field=value.get("field"),
                project_id=value.get("project_id"),
                workspace_id=value.get("workspace_id"),
                resource_revision=value.get("resource_revision"),
            )
        except TypeError as exc:
            raise KnowledgeValidationError(f"invalid knowledge source ref: {exc}") from exc


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    kind: KnowledgeKind
    scopes: KnowledgeScope
    content: str
    applicability: str = ""
    source_refs: tuple[KnowledgeSourceRef, ...] = ()
    authority: str = "unverified"
    proof_family: str = "unknown"
    created_at: str = field(default_factory=_now_iso)
    observed_at: str | None = None
    freshness: KnowledgeFreshness = KnowledgeFreshness.UNKNOWN
    status: KnowledgeStatus = KnowledgeStatus.CANDIDATE
    supersedes: tuple[str, ...] = ()
    sensitivity: KnowledgeSensitivity = KnowledgeSensitivity.PRIVATE
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = KNOWLEDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.id, "record.id")
        if not _RECORD_ID.fullmatch(self.id):
            raise KnowledgeValidationError(
                "record.id must be a path-safe ASCII identifier (letters, digits, '_' or '-', max 160 chars)",
            )
        object.__setattr__(self, "kind", _enum(self.kind, KnowledgeKind, "record.kind"))
        if not isinstance(self.scopes, KnowledgeScope):
            raise KnowledgeValidationError("record.scopes must be a KnowledgeScope")
        _text(self.content, "record.content", multiline=True)
        _text(self.applicability, "record.applicability", allow_empty=True, multiline=True)
        if isinstance(self.source_refs, (str, bytes, Mapping)):
            raise KnowledgeValidationError("record.source_refs must be a sequence")
        refs = tuple(self.source_refs)
        if any(not isinstance(ref, KnowledgeSourceRef) for ref in refs):
            raise KnowledgeValidationError("record.source_refs must contain KnowledgeSourceRef records")
        if len(set(refs)) != len(refs):
            raise KnowledgeValidationError("record.source_refs must not contain duplicates")
        object.__setattr__(self, "source_refs", refs)
        _text(self.authority, "record.authority")
        _text(self.proof_family, "record.proof_family")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "record.created_at"))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "record.observed_at", optional=True))
        object.__setattr__(self, "freshness", _enum(self.freshness, KnowledgeFreshness, "record.freshness"))
        object.__setattr__(self, "status", _enum(self.status, KnowledgeStatus, "record.status"))
        object.__setattr__(self, "sensitivity", _enum(
            self.sensitivity, KnowledgeSensitivity, "record.sensitivity",
        ))
        supersedes = _string_tuple(self.supersedes, "record.supersedes")
        if self.id in supersedes:
            raise KnowledgeValidationError("record cannot supersede itself")
        object.__setattr__(self, "supersedes", supersedes)
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise KnowledgeValidationError("record.schema_version must be an integer")
        if self.schema_version != KNOWLEDGE_SCHEMA_VERSION:
            raise KnowledgeValidationError(f"unsupported knowledge schema version: {self.schema_version}")
        if self.status in _PROVENANCED_STATUSES and not refs:
            raise KnowledgeValidationError(f"{self.status.value} knowledge requires at least one source ref")
        if self.status == KnowledgeStatus.LEGACY_UNPROVENANCED and refs:
            raise KnowledgeValidationError("legacy_unprovenanced knowledge cannot claim source refs")
        object.__setattr__(self, "metadata", _freeze_json(dict(self.metadata), "record.metadata"))

    @classmethod
    def create(
        cls,
        *,
        kind: KnowledgeKind | str,
        scopes: KnowledgeScope,
        content: str,
        source_refs: Iterable[KnowledgeSourceRef] = (),
        record_id: str | None = None,
        **fields: Any,
    ) -> "KnowledgeRecord":
        return cls(
            id=record_id or f"knowledge-{uuid.uuid4().hex}",
            kind=kind,  # type: ignore[arg-type]
            scopes=scopes,
            content=content,
            source_refs=tuple(source_refs),
            **fields,
        )

    @property
    def digest(self) -> str:
        wire = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(wire.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "scopes": self.scopes.to_dict(),
            "applicability": self.applicability,
            "content": self.content,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "authority": self.authority,
            "proof_family": self.proof_family,
            "created_at": self.created_at,
            "observed_at": self.observed_at,
            "freshness": self.freshness.value,
            "status": self.status.value,
            "supersedes": list(self.supersedes),
            "sensitivity": self.sensitivity.value,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeRecord":
        if not isinstance(value, Mapping):
            raise KnowledgeValidationError("knowledge record must be an object")
        refs = value.get("source_refs", ())
        if isinstance(refs, (str, bytes, Mapping)):
            raise KnowledgeValidationError("record.source_refs must be a sequence")
        supersedes = value.get("supersedes", ())
        try:
            return cls(
                id=value.get("id", ""),
                schema_version=value.get("schema_version", KNOWLEDGE_SCHEMA_VERSION),
                kind=value.get("kind", ""),
                scopes=KnowledgeScope.from_dict(value.get("scopes", {})),
                applicability=value.get("applicability", ""),
                content=value.get("content", ""),
                source_refs=tuple(KnowledgeSourceRef.from_dict(ref) for ref in refs),
                authority=value.get("authority", "unverified"),
                proof_family=value.get("proof_family", "unknown"),
                created_at=value.get("created_at", ""),
                observed_at=value.get("observed_at"),
                freshness=value.get("freshness", KnowledgeFreshness.UNKNOWN.value),
                status=value.get("status", KnowledgeStatus.CANDIDATE.value),
                supersedes=tuple(supersedes),
                sensitivity=value.get("sensitivity", KnowledgeSensitivity.PRIVATE.value),
                metadata=value.get("metadata", {}),
            )
        except TypeError as exc:
            raise KnowledgeValidationError(f"invalid knowledge record: {exc}") from exc


@dataclass(frozen=True)
class FeedbackEvent:
    record_id: str
    kind: FeedbackKind
    id: str = field(default_factory=lambda: f"feedback-{uuid.uuid4().hex}")
    created_at: str = field(default_factory=_now_iso)
    source_refs: tuple[KnowledgeSourceRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.id, "feedback.id")
        _text(self.record_id, "feedback.record_id")
        object.__setattr__(self, "kind", _enum(self.kind, FeedbackKind, "feedback.kind"))
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "feedback.created_at"))
        if isinstance(self.source_refs, (str, bytes, Mapping)):
            raise KnowledgeValidationError("feedback.source_refs must be a sequence")
        refs = tuple(self.source_refs)
        if any(not isinstance(ref, KnowledgeSourceRef) for ref in refs):
            raise KnowledgeValidationError("feedback.source_refs must contain KnowledgeSourceRef records")
        object.__setattr__(self, "source_refs", refs)
        object.__setattr__(self, "metadata", _freeze_json(dict(self.metadata), "feedback.metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "record_id": self.record_id,
            "kind": self.kind.value,
            "created_at": self.created_at,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeedbackEvent":
        if not isinstance(value, Mapping):
            raise KnowledgeValidationError("feedback event must be an object")
        refs = value.get("source_refs", ())
        if isinstance(refs, (str, bytes, Mapping)):
            raise KnowledgeValidationError("feedback.source_refs must be a sequence")
        try:
            return cls(
                id=value.get("id", ""),
                record_id=value.get("record_id", ""),
                kind=value.get("kind", ""),
                created_at=value.get("created_at", ""),
                source_refs=tuple(KnowledgeSourceRef.from_dict(ref) for ref in refs),
                metadata=value.get("metadata", {}),
            )
        except TypeError as exc:
            raise KnowledgeValidationError(f"invalid feedback event: {exc}") from exc


@dataclass(frozen=True)
class KnowledgeQuery:
    text: str = ""
    user_id: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    kinds: tuple[KnowledgeKind, ...] = ()
    statuses: tuple[KnowledgeStatus, ...] = (KnowledgeStatus.ACTIVE,)
    include_secret: bool = False
    paths_context: tuple[str, ...] = ()
    limit: int = 10

    def __post_init__(self) -> None:
        _text(self.text, "query.text", allow_empty=True, multiline=True)
        for name in ("user_id", "project_id", "agent_id"):
            value = getattr(self, name)
            if value is not None:
                _text(value, f"query.{name}")
        if isinstance(self.kinds, (str, bytes, Mapping)):
            raise KnowledgeValidationError("query.kinds must be a sequence")
        if isinstance(self.statuses, (str, bytes, Mapping)):
            raise KnowledgeValidationError("query.statuses must be a sequence")
        object.__setattr__(self, "kinds", tuple(
            _enum(kind, KnowledgeKind, "query.kinds item") for kind in self.kinds
        ))
        object.__setattr__(self, "statuses", tuple(
            _enum(status, KnowledgeStatus, "query.statuses item") for status in self.statuses
        ))
        object.__setattr__(self, "paths_context", _string_tuple(
            self.paths_context, "query.paths_context",
        ))
        if not isinstance(self.include_secret, bool):
            raise KnowledgeValidationError("query.include_secret must be a boolean")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 1:
            raise KnowledgeValidationError("query.limit must be an integer >= 1")
        object.__setattr__(self, "limit", min(self.limit, _MAX_QUERY_LIMIT))


@dataclass(frozen=True)
class KnowledgeHit:
    record: KnowledgeRecord
    score: float
    snippet: str


def fts5_available() -> bool:
    try:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE VIRTUAL TABLE _knowledge_probe USING fts5(content)")
            return True
        finally:
            connection.close()
    except Exception:
        return False


def _fts_match_query(query: str) -> str:
    tokens = re.findall(r"\w+", query or "", flags=re.UNICODE)
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _lexical_tokens(text: str) -> list[str]:
    return [token.casefold() for token in re.findall(r"\w+", text or "", flags=re.UNICODE)]


def _metadata_text(record: KnowledgeRecord) -> str:
    return json.dumps(_thaw_json(record.metadata), ensure_ascii=False, sort_keys=True)


class KnowledgeRepository:
    """Canonical SQLite store with transactional native FTS and lexical fallback.

    Scope predicates are always part of the SQL candidate query.  Ranking therefore
    never sees records belonging to another user, project, or agent identity.
    """

    def __init__(self, db_path: str, *, use_fts: bool | None = None) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._closed = False
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(os.path.expanduser(db_path)))
            if parent:
                private_dir(parent)
            db_path = os.path.abspath(os.path.expanduser(db_path))
            self.db_path = db_path
        try:
            self._con = sqlite3.connect(db_path, check_same_thread=False)
            self._con.row_factory = sqlite3.Row
            self._con.execute("PRAGMA foreign_keys = ON")
            self._initialize_schema(use_fts=use_fts)
            if db_path != ":memory:":
                private_file(db_path)
        except Exception:
            try:
                self._con.close()
            except Exception:
                pass
            raise

    def _initialize_schema(self, *, use_fts: bool | None) -> None:
        version = int(self._con.execute("PRAGMA user_version").fetchone()[0])
        if version > KNOWLEDGE_DB_SCHEMA_VERSION:
            raise KnowledgeConflictError(
                f"knowledge database schema {version} is newer than supported {KNOWLEDGE_DB_SCHEMA_VERSION}",
            )
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                applicability TEXT NOT NULL,
                authority TEXT NOT NULL,
                proof_family TEXT NOT NULL,
                created_at TEXT NOT NULL,
                observed_at TEXT,
                freshness TEXT NOT NULL,
                status TEXT NOT NULL,
                supersedes_json TEXT NOT NULL,
                sensitivity TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS record_scopes (
                record_id TEXT PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
                user_id TEXT,
                project_id TEXT,
                agent_id TEXT
            );
            CREATE TABLE IF NOT EXISTS source_refs (
                record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                namespace TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                digest TEXT NOT NULL,
                observer TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                byte_start INTEGER,
                byte_end INTEGER,
                field TEXT,
                project_id TEXT,
                workspace_id TEXT,
                resource_revision TEXT,
                PRIMARY KEY (record_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS feedback_events (
                id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source_refs_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS indexed_records (
                record_id TEXT PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS supersession_edges (
                old_record_id TEXT PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
                replacement_record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS runtime_metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_records_status_kind ON records(status, kind);
            CREATE INDEX IF NOT EXISTS idx_scopes_user_project_agent
                ON record_scopes(user_id, project_id, agent_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_record_created
                ON feedback_events(record_id, created_at);
            """
        )
        requested = fts5_available() if use_fts is None else bool(use_fts)
        self.fts_enabled = False
        self.fts_error = ""
        if requested:
            try:
                self._con.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5("
                    "record_id UNINDEXED, content, applicability, kind, metadata, "
                    "tokenize='porter unicode61')",
                )
                self.fts_enabled = True
                # FTS is a rebuildable projection.  Populate it when a database created
                # under lexical fallback is later opened on a runtime with FTS5, and
                # repair any interrupted/drifted projection on ordinary startup.
                indexed = int(self._con.execute("SELECT COUNT(*) FROM indexed_records").fetchone()[0])
                projected = int(self._con.execute("SELECT COUNT(*) FROM records_fts").fetchone()[0])
                if projected != indexed:
                    self._con.execute("DELETE FROM records_fts")
                    self._con.execute(
                        """INSERT INTO records_fts(record_id, content, applicability, kind, metadata)
                           SELECT r.id, r.content, r.applicability, r.kind, r.metadata_json
                           FROM records r JOIN indexed_records i ON i.record_id = r.id""",
                    )
            except sqlite3.Error as exc:
                self.fts_error = str(exc)
                self.fts_enabled = False
        self._con.execute(f"PRAGMA user_version = {KNOWLEDGE_DB_SCHEMA_VERSION}")
        self._con.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise KnowledgeConflictError("knowledge repository is closed")

    def __enter__(self) -> "KnowledgeRepository":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._con.close()
                self._closed = True

    def _begin(self) -> None:
        self._con.execute("BEGIN IMMEDIATE")

    def _finish(self, *, success: bool) -> None:
        if success:
            self._con.commit()
            if self.db_path != ":memory:":
                private_file(self.db_path)
        else:
            self._con.rollback()

    def _row_to_record(self, row: sqlite3.Row) -> KnowledgeRecord:
        scope = self._con.execute(
            "SELECT user_id, project_id, agent_id FROM record_scopes WHERE record_id = ?",
            (row["id"],),
        ).fetchone()
        if scope is None:
            raise KnowledgeConflictError(f"record {row['id']!r} has no scope row")
        source_rows = self._con.execute(
            "SELECT * FROM source_refs WHERE record_id = ? ORDER BY ordinal",
            (row["id"],),
        ).fetchall()
        refs = tuple(KnowledgeSourceRef(
            namespace=source["namespace"],
            record_id=source["source_record_id"],
            digest=source["digest"],
            observer=source["observer"],
            observed_at=source["observed_at"],
            byte_start=source["byte_start"],
            byte_end=source["byte_end"],
            field=source["field"],
            project_id=source["project_id"],
            workspace_id=source["workspace_id"],
            resource_revision=source["resource_revision"],
        ) for source in source_rows)
        return KnowledgeRecord(
            id=row["id"],
            schema_version=row["schema_version"],
            kind=row["kind"],
            scopes=KnowledgeScope(
                user_id=scope["user_id"],
                project_id=scope["project_id"],
                agent_id=scope["agent_id"],
            ),
            content=row["content"],
            applicability=row["applicability"],
            source_refs=refs,
            authority=row["authority"],
            proof_family=row["proof_family"],
            created_at=row["created_at"],
            observed_at=row["observed_at"],
            freshness=row["freshness"],
            status=row["status"],
            supersedes=tuple(json.loads(row["supersedes_json"])),
            sensitivity=row["sensitivity"],
            metadata=json.loads(row["metadata_json"]),
        )

    def _get_unlocked(self, record_id: str) -> KnowledgeRecord | None:
        row = self._con.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        return self._row_to_record(row) if row is not None else None

    def get(self, record_id: str) -> KnowledgeRecord | None:
        _text(record_id, "record_id")
        with self._lock:
            self._ensure_open()
            return self._get_unlocked(record_id)

    @staticmethod
    def _validate_update(current: KnowledgeRecord, updated: KnowledgeRecord, *, tombstone: bool = False) -> None:
        if current.id != updated.id:
            raise KnowledgeConflictError("knowledge record identity cannot change")
        if updated.status not in _STATUS_TRANSITIONS[current.status]:
            raise KnowledgeConflictError(f"invalid knowledge lifecycle: {current.status.value} -> {updated.status.value}")
        if current.status == KnowledgeStatus.TOMBSTONED and updated != current:
            raise KnowledgeConflictError("tombstoned knowledge is immutable")
        if not tombstone and current.status not in {
            KnowledgeStatus.CANDIDATE, KnowledgeStatus.LEGACY_UNPROVENANCED,
        }:
            semantic_fields = (
                "schema_version", "kind", "scopes", "content", "applicability", "authority",
                "proof_family", "created_at", "observed_at", "freshness", "sensitivity", "metadata",
            )
            if any(getattr(current, name) != getattr(updated, name) for name in semantic_fields):
                raise KnowledgeConflictError("active knowledge meaning is immutable; supersede it with a new record")
        if not tombstone:
            if not set(current.source_refs).issubset(updated.source_refs):
                raise KnowledgeConflictError("knowledge updates cannot erase provenance")
            if not set(current.supersedes).issubset(updated.supersedes):
                raise KnowledgeConflictError("knowledge updates cannot erase supersession links")

    def _write_record_unlocked(self, record: KnowledgeRecord, *, index: bool = True) -> None:
        self._con.execute(
            """INSERT INTO records (
                   id, schema_version, kind, content, applicability, authority, proof_family,
                   created_at, observed_at, freshness, status, supersedes_json, sensitivity, metadata_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   schema_version=excluded.schema_version, kind=excluded.kind, content=excluded.content,
                   applicability=excluded.applicability, authority=excluded.authority,
                   proof_family=excluded.proof_family, created_at=excluded.created_at,
                   observed_at=excluded.observed_at, freshness=excluded.freshness, status=excluded.status,
                   supersedes_json=excluded.supersedes_json, sensitivity=excluded.sensitivity,
                   metadata_json=excluded.metadata_json""",
            (
                record.id, record.schema_version, record.kind.value, record.content, record.applicability,
                record.authority, record.proof_family, record.created_at, record.observed_at,
                record.freshness.value, record.status.value,
                json.dumps(record.supersedes, ensure_ascii=False), record.sensitivity.value,
                json.dumps(_thaw_json(record.metadata), ensure_ascii=False, sort_keys=True),
            ),
        )
        self._con.execute(
            """INSERT INTO record_scopes(record_id, user_id, project_id, agent_id)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(record_id) DO UPDATE SET user_id=excluded.user_id,
                   project_id=excluded.project_id, agent_id=excluded.agent_id""",
            (record.id, record.scopes.user_id, record.scopes.project_id, record.scopes.agent_id),
        )
        self._con.execute("DELETE FROM source_refs WHERE record_id = ?", (record.id,))
        self._con.executemany(
            """INSERT INTO source_refs(
                   record_id, ordinal, namespace, source_record_id, digest, observer, observed_at,
                   byte_start, byte_end, field, project_id, workspace_id, resource_revision
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(
                record.id, ordinal, ref.namespace, ref.record_id, ref.digest, ref.observer,
                ref.observed_at, ref.byte_start, ref.byte_end, ref.field, ref.project_id,
                ref.workspace_id, ref.resource_revision,
            ) for ordinal, ref in enumerate(record.source_refs)],
        )
        if index:
            self._index_record_unlocked(record)

    def put(self, record: KnowledgeRecord) -> KnowledgeRecord:
        if not isinstance(record, KnowledgeRecord):
            raise KnowledgeValidationError("put requires a KnowledgeRecord")
        with self._lock:
            self._ensure_open()
            current = self._get_unlocked(record.id)
            if current is not None:
                self._validate_update(current, record)
                if current.status != KnowledgeStatus.SUPERSEDED and record.status == KnowledgeStatus.SUPERSEDED:
                    raise KnowledgeConflictError("use supersede(record_id, replacement_id) to retire knowledge")
                if record.supersedes != current.supersedes:
                    raise KnowledgeConflictError("use supersede(record_id, replacement_id) to add supersession links")
            elif record.supersedes or record.status == KnowledgeStatus.SUPERSEDED:
                raise KnowledgeConflictError("new knowledge cannot claim supersession before repository linking")
            self._begin()
            try:
                self._write_record_unlocked(record)
            except Exception:
                self._finish(success=False)
                raise
            self._finish(success=True)
            return record

    def _index_record_unlocked(self, record: KnowledgeRecord) -> None:
        self._con.execute("INSERT OR IGNORE INTO indexed_records(record_id) VALUES (?)", (record.id,))
        if self.fts_enabled:
            self._con.execute("DELETE FROM records_fts WHERE record_id = ?", (record.id,))
            self._con.execute(
                "INSERT INTO records_fts(record_id, content, applicability, kind, metadata) VALUES (?, ?, ?, ?, ?)",
                (record.id, record.content, record.applicability, record.kind.value, _metadata_text(record)),
            )

    def index_records(self, records: Iterable[KnowledgeRecord]) -> None:
        records = tuple(records)
        if any(not isinstance(record, KnowledgeRecord) for record in records):
            raise KnowledgeValidationError("index_records requires KnowledgeRecord values")
        with self._lock:
            self._ensure_open()
            canonical: list[KnowledgeRecord] = []
            for supplied in records:
                stored = self._get_unlocked(supplied.id)
                if stored is None:
                    raise KnowledgeNotFoundError(f"cannot index absent record {supplied.id!r}")
                if stored.digest != supplied.digest:
                    raise KnowledgeConflictError(f"index input for {supplied.id!r} differs from canonical record")
                canonical.append(stored)
            self._begin()
            try:
                for record in canonical:
                    self._index_record_unlocked(record)
            except Exception:
                self._finish(success=False)
                raise
            self._finish(success=True)

    def remove_from_index(self, record_ids: Iterable[str]) -> None:
        ids = _string_tuple(record_ids, "record_ids")
        with self._lock:
            self._ensure_open()
            self._begin()
            try:
                self._con.executemany("DELETE FROM indexed_records WHERE record_id = ?", ((item,) for item in ids))
                if self.fts_enabled:
                    self._con.executemany("DELETE FROM records_fts WHERE record_id = ?", ((item,) for item in ids))
            except Exception:
                self._finish(success=False)
                raise
            self._finish(success=True)

    def rebuild_index(self) -> None:
        with self._lock:
            self._ensure_open()
            rows = self._con.execute("SELECT * FROM records ORDER BY id").fetchall()
            records = [self._row_to_record(row) for row in rows]
            self._begin()
            try:
                self._con.execute("DELETE FROM indexed_records")
                if self.fts_enabled:
                    self._con.execute("DELETE FROM records_fts")
                for record in records:
                    self._index_record_unlocked(record)
            except Exception:
                self._finish(success=False)
                raise
            self._finish(success=True)

    def indexed_count(self) -> int:
        with self._lock:
            self._ensure_open()
            return int(self._con.execute("SELECT COUNT(*) FROM indexed_records").fetchone()[0])

    def records_for_index_rebuild(self) -> list[KnowledgeRecord]:
        """Return canonical records for a host-owned index rebuild.

        This is deliberately a maintenance surface rather than a recall query:
        it is unpaged, unscoped, and may include inactive/secret records so an
        external projection can explicitly retire anything it must never serve.
        The repository remains the only authority for record meaning.
        """
        with self._lock:
            self._ensure_open()
            rows = self._con.execute("SELECT * FROM records ORDER BY id").fetchall()
            return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _filter_sql(query: KnowledgeQuery, *, indexed: bool) -> tuple[str, list[Any]]:
        joins = ["JOIN record_scopes s ON s.record_id = r.id"]
        if indexed:
            joins.append("JOIN indexed_records i ON i.record_id = r.id")
        clauses = [
            "(s.user_id IS NULL OR s.user_id = ?)",
            "(s.project_id IS NULL OR s.project_id = ?)",
            "(s.agent_id IS NULL OR s.agent_id = ?)",
        ]
        params: list[Any] = [query.user_id, query.project_id, query.agent_id]
        if query.kinds:
            clauses.append(f"r.kind IN ({','.join('?' for _ in query.kinds)})")
            params.extend(kind.value for kind in query.kinds)
        if query.statuses:
            clauses.append(f"r.status IN ({','.join('?' for _ in query.statuses)})")
            params.extend(status.value for status in query.statuses)
        if not query.include_secret:
            clauses.append("r.sensitivity != ?")
            params.append(KnowledgeSensitivity.SECRET.value)
        return " ".join(joins) + " WHERE " + " AND ".join(clauses), params

    def _filtered_rows_unlocked(self, query: KnowledgeQuery, *, indexed: bool) -> list[sqlite3.Row]:
        filters, params = self._filter_sql(query, indexed=indexed)
        return self._con.execute(
            "SELECT r.* FROM records r " + filters + " ORDER BY r.created_at DESC, r.id LIMIT ?",
            (*params, query.limit),
        ).fetchall()

    def query(self, query: KnowledgeQuery) -> list[KnowledgeRecord]:
        if not isinstance(query, KnowledgeQuery):
            raise KnowledgeValidationError("query requires a KnowledgeQuery")
        if query.text.strip():
            return [hit.record for hit in self.search(query)]
        with self._lock:
            self._ensure_open()
            return [self._row_to_record(row) for row in self._filtered_rows_unlocked(query, indexed=False)]

    def count_by_axis(self, query: KnowledgeQuery) -> dict[str, int]:
        """Count every in-scope record without conflating the query page limit with store size."""
        if not isinstance(query, KnowledgeQuery):
            raise KnowledgeValidationError("count_by_axis requires a KnowledgeQuery")
        if query.text.strip():
            raise KnowledgeValidationError("count_by_axis requires an empty text query")
        with self._lock:
            self._ensure_open()
            filters, params = self._filter_sql(query, indexed=False)
            row = self._con.execute(
                "SELECT "
                "COUNT(*) AS unique_count, "
                "SUM(CASE WHEN s.user_id IS NOT NULL THEN 1 ELSE 0 END) AS user_count, "
                "SUM(CASE WHEN s.project_id IS NOT NULL THEN 1 ELSE 0 END) AS project_count, "
                "SUM(CASE WHEN s.agent_id IS NOT NULL THEN 1 ELSE 0 END) AS craft_count "
                "FROM records r " + filters,
                params,
            ).fetchone()
        return {
            "unique": int(row["unique_count"] or 0),
            "user": int(row["user_count"] or 0),
            "project": int(row["project_count"] or 0),
            "craft": int(row["craft_count"] or 0),
        }

    def count_by_source_namespace(self, query: KnowledgeQuery, namespace: str) -> int:
        """Count in-scope records carrying at least one source from ``namespace``."""
        if not isinstance(query, KnowledgeQuery):
            raise KnowledgeValidationError("count_by_source_namespace requires a KnowledgeQuery")
        if query.text.strip():
            raise KnowledgeValidationError("count_by_source_namespace requires an empty text query")
        source_namespace = _text(namespace, "source namespace")
        filters, params = self._filter_sql(query, indexed=False)
        joins, clauses = filters.split(" WHERE ", 1)
        with self._lock:
            self._ensure_open()
            row = self._con.execute(
                "SELECT COUNT(DISTINCT r.id) FROM records r "
                + joins
                + " JOIN source_refs sr ON sr.record_id = r.id WHERE "
                + clauses
                + " AND sr.namespace = ?",
                (*params, source_namespace),
            ).fetchone()
        return int(row[0] or 0)

    def get_runtime_metadata(self, key: str) -> dict[str, Any] | None:
        """Read host-owned diagnostic metadata that never participates in retrieval.

        Consolidation/compatibility-transition status belongs beside the canonical repository, but
        it is not knowledge and must never enter FTS, ranking, or record authority.
        """
        identity = str(key or "").strip()
        if not identity or len(identity) > 240 or "\x00" in identity:
            raise KnowledgeValidationError("runtime metadata key must be 1-240 text characters")
        with self._lock:
            self._ensure_open()
            row = self._con.execute(
                "SELECT value_json FROM runtime_metadata WHERE key = ?", (identity,),
            ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row["value_json"])
        except (TypeError, ValueError) as exc:
            raise KnowledgeConflictError(
                f"runtime metadata {identity!r} contains invalid JSON",
            ) from exc
        if not isinstance(value, dict):
            raise KnowledgeConflictError(
                f"runtime metadata {identity!r} must contain a JSON object",
            )
        return value

    def set_runtime_metadata(
        self, key: str, value: Mapping[str, Any], *, updated_at: str,
    ) -> None:
        """Atomically publish non-authoritative host diagnostics."""
        identity = str(key or "").strip()
        if not identity or len(identity) > 240 or "\x00" in identity:
            raise KnowledgeValidationError("runtime metadata key must be 1-240 text characters")
        if not isinstance(value, Mapping):
            raise KnowledgeValidationError("runtime metadata value must be an object")
        timestamp = _text(updated_at, "runtime metadata updated_at")
        payload = json.dumps(_thaw_json(value), ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._ensure_open()
            self._con.execute(
                "INSERT INTO runtime_metadata(key, value_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
                "updated_at=excluded.updated_at",
                (identity, payload, timestamp),
            )
            self._con.commit()
            if self.db_path != ":memory:":
                private_file(self.db_path)

    def search(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        if not isinstance(query, KnowledgeQuery):
            raise KnowledgeValidationError("search requires a KnowledgeQuery")
        if not query.text.strip():
            records = self.query(query)
            return [KnowledgeHit(record=record, score=0.0, snippet=record.content[:240]) for record in records]
        with self._lock:
            self._ensure_open()
            if self.fts_enabled:
                try:
                    hits = self._search_fts_unlocked(query)
                    return hits
                except sqlite3.Error:
                    # The canonical repository remains useful even when an FTS query or extension fails.
                    pass
            return self._search_fallback_unlocked(query)

    def _search_fts_unlocked(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        match = _fts_match_query(query.text)
        if not match:
            return self._search_fallback_unlocked(query)
        filters, params = self._filter_sql(query, indexed=True)
        sql = (
            "SELECT r.*, snippet(records_fts, 1, '«', '»', ' … ', 24) AS _snippet, "
            "bm25(records_fts, 0.0, 5.0, 2.0, 0.5, 0.5) AS _rank "
            "FROM records_fts JOIN records r ON r.id = records_fts.record_id "
            + filters
            + " AND records_fts MATCH ? ORDER BY _rank ASC, r.created_at DESC LIMIT ?"
        )
        rows = self._con.execute(sql, (*params, match, query.limit)).fetchall()
        return [KnowledgeHit(
            record=self._row_to_record(row),
            score=max(0.0, -float(row["_rank"] or 0.0)),
            snippet=row["_snippet"] or row["content"][:240],
        ) for row in rows]

    @staticmethod
    def _fallback_score(query_tokens: Sequence[str], record: KnowledgeRecord) -> float:
        content_tokens = _lexical_tokens(record.content)
        auxiliary_tokens = _lexical_tokens(record.applicability + " " + _metadata_text(record))
        if not query_tokens:
            return 0.0
        content_counts = Counter(content_tokens)
        auxiliary_counts = Counter(auxiliary_tokens)
        matched = 0
        score = 0.0
        for token in set(query_tokens):
            content_tf = content_counts[token]
            auxiliary_tf = auxiliary_counts[token]
            if content_tf or auxiliary_tf:
                matched += 1
                score += 3.0 * (1.0 + math.log1p(content_tf)) + 1.0 * math.log1p(auxiliary_tf)
        score += 2.0 * matched / len(set(query_tokens))
        if query_tokens and " ".join(query_tokens) in " ".join(content_tokens):
            score += 2.0
        return score

    @staticmethod
    def _fallback_snippet(content: str, tokens: Sequence[str], width: int = 240) -> str:
        lowered = content.casefold()
        positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
        start = max(0, min(positions) - width // 3) if positions else 0
        snippet = content[start:start + width]
        for token in sorted(set(tokens), key=len, reverse=True):
            snippet = re.sub(re.escape(token), lambda match: f"«{match.group(0)}»", snippet, flags=re.IGNORECASE)
        return ("…" if start else "") + snippet + ("…" if start + width < len(content) else "")

    def _search_fallback_unlocked(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        # Fetch all scope-eligible candidates before ranking.  No out-of-scope record is
        # scored, so fallback behavior preserves the same hard boundary as FTS.
        unbounded = replace(query, limit=_MAX_QUERY_LIMIT)
        filters, params = self._filter_sql(unbounded, indexed=True)
        rows = self._con.execute("SELECT r.* FROM records r " + filters, params).fetchall()
        tokens = _lexical_tokens(query.text)
        hits: list[KnowledgeHit] = []
        for row in rows:
            record = self._row_to_record(row)
            score = self._fallback_score(tokens, record)
            if score <= 0:
                continue
            hits.append(KnowledgeHit(
                record=record,
                score=score,
                snippet=self._fallback_snippet(record.content, tokens),
            ))
        hits.sort(key=lambda hit: (hit.score, hit.record.created_at, hit.record.id), reverse=True)
        return hits[:query.limit]

    def feedback(self, event: FeedbackEvent) -> FeedbackEvent:
        if not isinstance(event, FeedbackEvent):
            raise KnowledgeValidationError("feedback requires a FeedbackEvent")
        with self._lock:
            self._ensure_open()
            if self._get_unlocked(event.record_id) is None:
                raise KnowledgeNotFoundError(f"feedback references absent record {event.record_id!r}")
            existing = self._con.execute(
                "SELECT * FROM feedback_events WHERE id = ?", (event.id,),
            ).fetchone()
            payload = (
                event.id, event.record_id, event.kind.value, event.created_at,
                json.dumps([ref.to_dict() for ref in event.source_refs], ensure_ascii=False, sort_keys=True),
                json.dumps(_thaw_json(event.metadata), ensure_ascii=False, sort_keys=True),
            )
            if existing is not None:
                current = self._feedback_from_row(existing)
                if current != event:
                    raise KnowledgeConflictError(f"feedback id {event.id!r} is already bound to another event")
                return current
            self._con.execute(
                "INSERT INTO feedback_events(id, record_id, kind, created_at, source_refs_json, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                payload,
            )
            self._con.commit()
            if self.db_path != ":memory:":
                private_file(self.db_path)
            return event

    @staticmethod
    def _feedback_from_row(row: sqlite3.Row) -> FeedbackEvent:
        return FeedbackEvent(
            id=row["id"],
            record_id=row["record_id"],
            kind=row["kind"],
            created_at=row["created_at"],
            source_refs=tuple(
                KnowledgeSourceRef.from_dict(ref) for ref in json.loads(row["source_refs_json"])
            ),
            metadata=json.loads(row["metadata_json"]),
        )

    def list_feedback(self, record_id: str) -> list[FeedbackEvent]:
        _text(record_id, "record_id")
        with self._lock:
            self._ensure_open()
            rows = self._con.execute(
                "SELECT * FROM feedback_events WHERE record_id = ? ORDER BY created_at, id", (record_id,),
            ).fetchall()
            return [self._feedback_from_row(row) for row in rows]

    def _supersession_target_unlocked(self, old_record_id: str) -> str | None:
        row = self._con.execute(
            "SELECT replacement_record_id FROM supersession_edges WHERE old_record_id = ?", (old_record_id,),
        ).fetchone()
        return row[0] if row is not None else None

    def _would_create_supersession_cycle_unlocked(self, old_id: str, replacement_id: str) -> bool:
        current = replacement_id
        seen: set[str] = set()
        while current not in seen:
            if current == old_id:
                return True
            seen.add(current)
            current = self._supersession_target_unlocked(current) or ""
            if not current:
                return False
        return True

    def supersede(self, record_id: str, replacement_id: str) -> tuple[KnowledgeRecord, KnowledgeRecord]:
        _text(record_id, "record_id")
        _text(replacement_id, "replacement_id")
        if record_id == replacement_id:
            raise KnowledgeConflictError("a knowledge record cannot supersede itself")
        with self._lock:
            self._ensure_open()
            old = self._get_unlocked(record_id)
            replacement_record = self._get_unlocked(replacement_id)
            if old is None:
                raise KnowledgeNotFoundError(f"knowledge record {record_id!r} does not exist")
            if replacement_record is None:
                raise KnowledgeNotFoundError(f"replacement record {replacement_id!r} does not exist")
            existing_target = self._supersession_target_unlocked(record_id)
            if existing_target is not None:
                if existing_target != replacement_id:
                    raise KnowledgeConflictError(
                        f"record {record_id!r} is already superseded by {existing_target!r}",
                    )
                return old, replacement_record
            if old.status not in {KnowledgeStatus.ACTIVE, KnowledgeStatus.EXPIRED}:
                raise KnowledgeConflictError(f"cannot supersede knowledge in {old.status.value} state")
            if replacement_record.status != KnowledgeStatus.ACTIVE:
                raise KnowledgeConflictError("replacement knowledge must be active")
            if self._would_create_supersession_cycle_unlocked(record_id, replacement_id):
                raise KnowledgeConflictError("knowledge supersession would create a cycle")
            replacement_links = replacement_record.supersedes
            if record_id not in replacement_links:
                replacement_record = replace(
                    replacement_record, supersedes=replacement_links + (record_id,),
                )
            superseded = replace(old, status=KnowledgeStatus.SUPERSEDED)
            self._validate_update(old, superseded)
            self._validate_update(self._get_unlocked(replacement_id), replacement_record)
            self._begin()
            try:
                self._write_record_unlocked(replacement_record)
                self._write_record_unlocked(superseded)
                self._con.execute(
                    "INSERT INTO supersession_edges(old_record_id, replacement_record_id) VALUES (?, ?)",
                    (record_id, replacement_id),
                )
            except Exception:
                self._finish(success=False)
                raise
            self._finish(success=True)
            return superseded, replacement_record

    def replacement_for(self, record_id: str) -> str | None:
        _text(record_id, "record_id")
        with self._lock:
            self._ensure_open()
            return self._supersession_target_unlocked(record_id)

    def _transition(self, record_id: str, status: KnowledgeStatus) -> KnowledgeRecord:
        _text(record_id, "record_id")
        with self._lock:
            self._ensure_open()
            current = self._get_unlocked(record_id)
            if current is None:
                raise KnowledgeNotFoundError(f"knowledge record {record_id!r} does not exist")
            updated = replace(current, status=status)
            self._validate_update(current, updated)
            self._begin()
            try:
                self._write_record_unlocked(updated)
            except Exception:
                self._finish(success=False)
                raise
            self._finish(success=True)
            return updated

    def retract(self, record_id: str) -> KnowledgeRecord:
        return self._transition(record_id, KnowledgeStatus.RETRACTED)

    def expire(self, record_id: str) -> KnowledgeRecord:
        return self._transition(record_id, KnowledgeStatus.EXPIRED)

    def tombstone(self, record_id: str) -> KnowledgeRecord:
        _text(record_id, "record_id")
        with self._lock:
            self._ensure_open()
            current = self._get_unlocked(record_id)
            if current is None:
                raise KnowledgeNotFoundError(f"knowledge record {record_id!r} does not exist")
            updated = KnowledgeRecord(
                id=current.id,
                kind=current.kind,
                scopes=current.scopes,
                content="[tombstoned]",
                status=KnowledgeStatus.TOMBSTONED,
                sensitivity=current.sensitivity,
                created_at=current.created_at,
                metadata={},
            )
            self._validate_update(current, updated, tombstone=True)
            self._begin()
            try:
                self._write_record_unlocked(updated, index=False)
                self._con.execute("DELETE FROM indexed_records WHERE record_id = ?", (record_id,))
                if self.fts_enabled:
                    self._con.execute("DELETE FROM records_fts WHERE record_id = ?", (record_id,))
            except Exception:
                self._finish(success=False)
                raise
            self._finish(success=True)
            return updated
