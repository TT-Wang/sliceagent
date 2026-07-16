"""Immutable, source-linked active-work state.

The active-work graph is deliberately a semantic *record*, not another prompt region and
not a transcript summary.  User language remains authoritative in the immutable event
ledger.  A :class:`SourceRef` identifies an exact half-open range in one such event and
binds both the complete source and selected span by digest.  The model may propose
``WorkDelta`` objects; the host applies them mechanically after checking identity,
provenance, lifecycle, dependency, and revision invariants.

This module has no dependency on ``Slice`` or the persistence stores.  Its ``to_dict`` /
``from_dict`` boundary is JSON-only, so it can be embedded in a checkpoint or artifact
without giving either layer a second interpretation of the work.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, TypeAlias


WORK_GRAPH_VERSION = 1
SOURCE_REF_VERSION = 1

WorkStatus: TypeAlias = Literal[
    "open",
    "in_progress",
    "waiting_user",
    "ready",
    "delivered",
    "verified",
    "cancelled",
    "superseded",
]
WorkKind: TypeAlias = Literal["request", "task"]

WORK_STATUSES = frozenset({
    "open", "in_progress", "waiting_user", "ready", "delivered", "verified", "cancelled", "superseded",
})
WORK_KINDS = frozenset({"request", "task"})
UNRESOLVED_STATUSES = frozenset({"open", "in_progress", "waiting_user", "ready"})
_SHA256_LENGTH = 64


class ActiveWorkError(ValueError):
    """Base class for active-work records that cannot be accepted mechanically."""


class SourceMismatchError(ActiveWorkError):
    """A source event is absent or no longer matches the exact reference."""


class GraphValidationError(ActiveWorkError):
    """A graph or delta violates an ownership, dependency, or lifecycle invariant."""


class RevisionConflictError(ActiveWorkError):
    """A delta was authored against a graph revision other than the current one."""


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "possibly-empty" if allow_empty else "non-empty"
        raise GraphValidationError(f"{name} must be a {qualifier} string")
    # Active Work metadata is rendered as one record per line.  Exact user source text lives separately in
    # SourceRef-bound ledger events and may be multiline; identifiers, descriptions, kinds, and locators may
    # not smuggle a second rendered record/header through CR/LF control characters.
    if "\r" in value or "\n" in value:
        raise GraphValidationError(f"{name} must not contain CR or LF")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise GraphValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_digest(value: object, name: str) -> str:
    value = _text(value, name)
    if len(value) != _SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in value):
        raise GraphValidationError(f"{name} must be a lowercase sha256 digest")
    return value


def _string_tuple(value: Iterable[str] | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise GraphValidationError(f"{name} must be a sequence of strings, not one string")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise GraphValidationError(f"{name} must be a sequence of strings") from exc
    for item in result:
        _text(item, f"{name} item")
    if len(set(result)) != len(result):
        raise GraphValidationError(f"{name} must not contain duplicates")
    return result


def _record_tuple(value: Iterable[Any] | None, cls: type, name: str) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise GraphValidationError(f"{name} must be a sequence")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise GraphValidationError(f"{name} must be a sequence") from exc
    if any(not isinstance(item, cls) for item in result):
        raise GraphValidationError(f"{name} must contain only {cls.__name__} records")
    return result


def _wire_sequence(value: object, name: str) -> tuple:
    """Decode one JSON array without leaking raw ``TypeError`` from hostile records."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise GraphValidationError(f"{name} must be a sequence")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise GraphValidationError(f"{name} must be a sequence") from exc


@dataclass(frozen=True, order=True)
class SourceRef:
    """Exact ``[start, end)`` Unicode-codepoint range in one immutable source event.

    ``source_sha256`` prevents an event ID from silently being rebound to different
    content.  ``span_sha256`` prevents a malformed range from being treated as the
    intended clause.  The text itself remains in the event ledger, avoiding a competing
    mutable copy in the work graph.
    """

    event_id: str
    start: int
    end: int
    source_length: int
    source_sha256: str
    span_sha256: str
    version: int = SOURCE_REF_VERSION

    def __post_init__(self) -> None:
        _text(self.event_id, "source_ref.event_id")
        _integer(self.start, "source_ref.start")
        _integer(self.end, "source_ref.end", minimum=1)
        _integer(self.source_length, "source_ref.source_length", minimum=1)
        if self.end <= self.start:
            raise GraphValidationError("source_ref range must be non-empty")
        if self.end > self.source_length:
            raise GraphValidationError("source_ref.end exceeds source_ref.source_length")
        _valid_digest(self.source_sha256, "source_ref.source_sha256")
        _valid_digest(self.span_sha256, "source_ref.span_sha256")
        _integer(self.version, "source_ref.version", minimum=1)
        if self.version != SOURCE_REF_VERSION:
            raise GraphValidationError(f"unsupported source-ref version: {self.version}")

    @classmethod
    def bind(cls, event_id: str, source: str, *, start: int = 0, end: int | None = None) -> "SourceRef":
        """Bind a range to exact source text without interpreting its meaning."""
        _text(event_id, "source_ref.event_id")
        if not isinstance(source, str) or not source:
            raise GraphValidationError("source text must be a non-empty string")
        _integer(start, "source_ref.start")
        if end is None:
            end = len(source)
        _integer(end, "source_ref.end", minimum=1)
        if end <= start or end > len(source):
            raise GraphValidationError("source range must be non-empty and within the source text")
        return cls(
            event_id=event_id,
            start=start,
            end=end,
            source_length=len(source),
            source_sha256=_sha256_text(source),
            span_sha256=_sha256_text(source[start:end]),
        )

    def extract(self, source: str) -> str:
        """Return the exact span, rejecting missing, changed, or differently-sized text."""
        if not isinstance(source, str):
            raise SourceMismatchError(f"source {self.event_id!r} is not text")
        if len(source) != self.source_length or _sha256_text(source) != self.source_sha256:
            raise SourceMismatchError(f"source {self.event_id!r} no longer matches its immutable event")
        span = source[self.start:self.end]
        if _sha256_text(span) != self.span_sha256:
            raise SourceMismatchError(f"source range for {self.event_id!r} does not match its bound span")
        return span

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.version,
            "event_id": self.event_id,
            "start": self.start,
            "end": self.end,
            "source_length": self.source_length,
            "source_sha256": self.source_sha256,
            "span_sha256": self.span_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRef":
        if not isinstance(value, Mapping):
            raise GraphValidationError("source ref must be an object")
        try:
            return cls(
                event_id=value.get("event_id", ""),
                start=value.get("start"),
                end=value.get("end"),
                source_length=value.get("source_length"),
                source_sha256=value.get("source_sha256", ""),
                span_sha256=value.get("span_sha256", ""),
                version=value.get("v", SOURCE_REF_VERSION),
            )
        except TypeError as exc:
            raise GraphValidationError(f"invalid source ref: {exc}") from exc


@dataclass(frozen=True, order=True)
class EvidenceRef:
    """Typed locator for execution/world evidence; the referenced store owns truth.

    ``qualifier`` carries a bounded mechanical condition such as ``source_partial``. It never upgrades the
    referenced testimony into a correctness verdict; exact detail remains behind ``ref``.
    """

    kind: str
    ref: str
    qualifier: str = ""

    def __post_init__(self) -> None:
        _text(self.kind, "evidence_ref.kind")
        _text(self.ref, "evidence_ref.ref")
        _text(self.qualifier, "evidence_ref.qualifier", allow_empty=True)

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref, **({"qualifier": self.qualifier} if self.qualifier else {})}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        if not isinstance(value, Mapping):
            raise GraphValidationError("evidence ref must be an object")
        return cls(
            kind=value.get("kind", ""), ref=value.get("ref", ""),
            qualifier=value.get("qualifier", ""),
        )


@dataclass(frozen=True, order=True)
class OutputRef:
    """Typed locator for a user-visible response or another delivered artifact."""

    kind: str
    ref: str

    def __post_init__(self) -> None:
        _text(self.kind, "output_ref.kind")
        _text(self.ref, "output_ref.ref")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OutputRef":
        if not isinstance(value, Mapping):
            raise GraphValidationError("output ref must be an object")
        return cls(kind=value.get("kind", ""), ref=value.get("ref", ""))


@dataclass(frozen=True, order=True)
class ResourceRef:
    """Locator for live world state consumed by one item.

    The workspace epoch prevents a file observation from workspace A being projected as
    current after a transition to workspace B.  ``revision`` is intentionally an opaque
    store-owned fingerprint rather than a host interpretation of the resource.
    """

    kind: str
    ref: str
    workspace_epoch: int = 0
    revision: str = ""

    def __post_init__(self) -> None:
        _text(self.kind, "resource_ref.kind")
        _text(self.ref, "resource_ref.ref")
        _integer(self.workspace_epoch, "resource_ref.workspace_epoch")
        _text(self.revision, "resource_ref.revision", allow_empty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ref": self.ref,
            "workspace_epoch": self.workspace_epoch,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResourceRef":
        if not isinstance(value, Mapping):
            raise GraphValidationError("resource ref must be an object")
        return cls(
            kind=value.get("kind", ""),
            ref=value.get("ref", ""),
            workspace_epoch=value.get("workspace_epoch", 0),
            revision=value.get("revision", ""),
        )


@dataclass(frozen=True)
class WorkItem:
    """One model-authored unit of work linked back to exact source language."""

    id: str
    root_id: str
    source_refs: tuple[SourceRef, ...]
    status: WorkStatus = "open"
    kind: WorkKind = "task"
    description: str = ""
    logical_id: str = ""
    workspace_epoch: int = 0
    dependencies: tuple[str, ...] = ()
    resource_refs: tuple[ResourceRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    output_refs: tuple[OutputRef, ...] = ()
    superseded_by: str = ""
    stop_reason: str = ""

    def __post_init__(self) -> None:
        _text(self.id, "work_item.id")
        _text(self.root_id, "work_item.root_id")
        _text(self.description, "work_item.description", allow_empty=True)
        if self.status not in WORK_STATUSES:
            raise GraphValidationError(f"unsupported work status: {self.status!r}")
        if self.kind not in WORK_KINDS:
            raise GraphValidationError(f"unsupported work kind: {self.kind!r}")
        if not self.logical_id and self.kind == "request":
            object.__setattr__(self, "logical_id", self.root_id)
        _text(self.logical_id, "work_item.logical_id", allow_empty=self.kind == "task")
        _integer(self.workspace_epoch, "work_item.workspace_epoch")
        _text(self.superseded_by, "work_item.superseded_by", allow_empty=True)
        _text(self.stop_reason, "work_item.stop_reason", allow_empty=True)
        object.__setattr__(self, "source_refs", _record_tuple(self.source_refs, SourceRef, "work_item.source_refs"))
        object.__setattr__(self, "dependencies", _string_tuple(self.dependencies, "work_item.dependencies"))
        object.__setattr__(self, "resource_refs", _record_tuple(
            self.resource_refs, ResourceRef, "work_item.resource_refs",
        ))
        object.__setattr__(self, "evidence_refs", _record_tuple(
            self.evidence_refs, EvidenceRef, "work_item.evidence_refs",
        ))
        object.__setattr__(self, "output_refs", _record_tuple(self.output_refs, OutputRef, "work_item.output_refs"))
        if not self.source_refs:
            raise GraphValidationError("every work item must cite at least one exact source range")
        if len(set(self.source_refs)) != len(self.source_refs):
            raise GraphValidationError("work_item.source_refs must not contain duplicates")
        if len(set(self.resource_refs)) != len(self.resource_refs):
            raise GraphValidationError("work_item.resource_refs must not contain duplicates")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise GraphValidationError("work_item.evidence_refs must not contain duplicates")
        if len(set(self.output_refs)) != len(self.output_refs):
            raise GraphValidationError("work_item.output_refs must not contain duplicates")
        if self.kind == "request" and self.root_id != self.id:
            raise GraphValidationError("a request root must name itself as root_id")
        if self.status in ("delivered", "verified") and not self.output_refs:
            raise GraphValidationError(f"{self.status} work must cite its delivered output")
        if self.status == "verified" and not self.evidence_refs:
            raise GraphValidationError("verified work must cite verification evidence")
        if self.status == "superseded":
            if not self.superseded_by or self.superseded_by == self.id:
                raise GraphValidationError("superseded work must cite a different replacement item")
        elif self.superseded_by:
            raise GraphValidationError("superseded_by is valid only when status is superseded")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "root_id": self.root_id,
            "kind": self.kind,
            "status": self.status,
            "description": self.description,
            "logical_id": self.logical_id,
            "workspace_epoch": self.workspace_epoch,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "dependencies": list(self.dependencies),
            "resource_refs": [ref.to_dict() for ref in self.resource_refs],
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "output_refs": [ref.to_dict() for ref in self.output_refs],
            "superseded_by": self.superseded_by,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkItem":
        if not isinstance(value, Mapping):
            raise GraphValidationError("work item must be an object")
        sources = _wire_sequence(value.get("source_refs") or (), "work_item.source_refs")
        resources = _wire_sequence(value.get("resource_refs") or (), "work_item.resource_refs")
        evidence = _wire_sequence(value.get("evidence_refs") or (), "work_item.evidence_refs")
        outputs = _wire_sequence(value.get("output_refs") or (), "work_item.output_refs")
        dependencies = _wire_sequence(value.get("dependencies") or (), "work_item.dependencies")
        if any(not isinstance(item, Mapping) for item in sources):
            raise GraphValidationError("work_item.source_refs must contain objects")
        if any(not isinstance(item, Mapping) for item in evidence):
            raise GraphValidationError("work_item.evidence_refs must contain objects")
        if any(not isinstance(item, Mapping) for item in resources):
            raise GraphValidationError("work_item.resource_refs must contain objects")
        if any(not isinstance(item, Mapping) for item in outputs):
            raise GraphValidationError("work_item.output_refs must contain objects")
        return cls(
            id=value.get("id", ""),
            root_id=value.get("root_id", ""),
            kind=value.get("kind", "task"),
            status=value.get("status", "open"),
            description=value.get("description", ""),
            logical_id=value.get("logical_id", ""),
            workspace_epoch=value.get("workspace_epoch", 0),
            source_refs=tuple(SourceRef.from_dict(item) for item in sources),
            dependencies=tuple(dependencies),
            resource_refs=tuple(ResourceRef.from_dict(item) for item in resources),
            evidence_refs=tuple(EvidenceRef.from_dict(item) for item in evidence),
            output_refs=tuple(OutputRef.from_dict(item) for item in outputs),
            superseded_by=value.get("superseded_by", ""),
            stop_reason=value.get("stop_reason", ""),
        )


@dataclass(frozen=True)
class WorkDelta:
    """One compare-and-swap proposal containing new and replacement item snapshots."""

    expected_revision: int
    creates: tuple[WorkItem, ...] = ()
    updates: tuple[WorkItem, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.expected_revision, "work_delta.expected_revision")
        object.__setattr__(self, "creates", _record_tuple(self.creates, WorkItem, "work_delta.creates"))
        object.__setattr__(self, "updates", _record_tuple(self.updates, WorkItem, "work_delta.updates"))
        if not self.creates and not self.updates:
            raise GraphValidationError("a work delta must create or update at least one item")
        create_ids = [item.id for item in self.creates]
        update_ids = [item.id for item in self.updates]
        if len(set(create_ids)) != len(create_ids):
            raise GraphValidationError("work_delta.creates contains duplicate item IDs")
        if len(set(update_ids)) != len(update_ids):
            raise GraphValidationError("work_delta.updates contains duplicate item IDs")
        if set(create_ids) & set(update_ids):
            raise GraphValidationError("one delta cannot both create and update the same item")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_revision": self.expected_revision,
            "creates": [item.to_dict() for item in self.creates],
            "updates": [item.to_dict() for item in self.updates],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkDelta":
        if not isinstance(value, Mapping):
            raise GraphValidationError("work delta must be an object")
        creates = _wire_sequence(value.get("creates") or (), "work_delta.creates")
        updates = _wire_sequence(value.get("updates") or (), "work_delta.updates")
        if any(not isinstance(item, Mapping) for item in creates + updates):
            raise GraphValidationError("work delta entries must be objects")
        return cls(
            expected_revision=value.get("expected_revision"),
            creates=tuple(WorkItem.from_dict(item) for item in creates),
            updates=tuple(WorkItem.from_dict(item) for item in updates),
        )


_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "open": frozenset({"open", "in_progress", "waiting_user", "ready", "delivered", "cancelled", "superseded"}),
    "in_progress": frozenset({
        "in_progress", "waiting_user", "ready", "delivered", "cancelled", "superseded",
    }),
    "waiting_user": frozenset({
        "waiting_user", "in_progress", "ready", "delivered", "cancelled", "superseded",
    }),
    # ``ready`` says a child contribution is prepared. It may be model-maintained for local work or derived
    # by the host from a bound child's successful immutable seal. Only the host can attach the real response
    # artifact and advance it to delivered.
    "ready": frozenset({"ready", "in_progress", "delivered", "cancelled", "superseded"}),
    # A delivered answer can be reopened when new evidence shows it is incomplete; verification is distinct.
    "delivered": frozenset({"delivered", "verified", "in_progress", "cancelled", "superseded"}),
    "verified": frozenset({"verified"}),
    "cancelled": frozenset({"cancelled"}),
    "superseded": frozenset({"superseded"}),
}


@dataclass(frozen=True)
class WorkGraph:
    """Immutable graph of request roots and their dependency-linked work items."""

    items: tuple[WorkItem, ...] = ()
    revision: int = 0
    version: int = WORK_GRAPH_VERSION
    _by_id: Mapping[str, WorkItem] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        _integer(self.revision, "work_graph.revision")
        _integer(self.version, "work_graph.version", minimum=1)
        if self.version != WORK_GRAPH_VERSION:
            raise GraphValidationError(f"unsupported work-graph version: {self.version}")
        items = _record_tuple(self.items, WorkItem, "work_graph.items")
        roots = {item.id: item for item in items if item.kind == "request"}
        # Child constructors may omit a redundant logical ID. Normalize it once at the
        # immutable graph boundary; serialized graph records are always explicit.
        items = tuple(
            replace(item, logical_id=roots[item.root_id].logical_id)
            if item.kind == "task" and not item.logical_id and item.root_id in roots else item
            for item in items
        )
        object.__setattr__(self, "items", items)
        by_id = {item.id: item for item in items}
        if len(by_id) != len(self.items):
            raise GraphValidationError("work graph contains duplicate item IDs")
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        self._validate_graph(by_id)

    def __deepcopy__(self, _memo) -> "WorkGraph":
        """An immutable graph is its own deep copy.

        ``_by_id`` is a read-only ``mappingproxy`` cache, which Python's generic deepcopy cannot pickle. Slice
        sealing and transactional reducer rollback deep-copy the containing Slice; returning this frozen value
        preserves those boundaries without reconstructing or sharing mutable state.
        """
        return self

    def _validate_graph(self, by_id: Mapping[str, WorkItem]) -> None:
        roots = {item.id for item in self.items if item.kind == "request"}
        logical_ids = [item.logical_id for item in self.items if item.kind == "request"]
        if len(set(logical_ids)) != len(logical_ids):
            raise GraphValidationError("request roots must have unique logical IDs")
        for root_id in roots:
            root = by_id[root_id]
            if root.status in UNRESOLVED_STATUSES:
                continue
            unresolved_children = sorted(
                item.id for item in self.items
                if item.id != root_id and item.root_id == root_id
                and item.status in UNRESOLVED_STATUSES
            )
            if unresolved_children:
                raise GraphValidationError(
                    f"terminal request root {root_id!r} has unresolved child work: "
                    + ", ".join(unresolved_children),
                )
        source_digests: dict[str, tuple[int, str]] = {}
        for item in self.items:
            if item.root_id not in roots:
                raise GraphValidationError(
                    f"work item {item.id!r} points to missing/non-request root {item.root_id!r}",
                )
            if item.logical_id != by_id[item.root_id].logical_id:
                raise GraphValidationError(
                    f"work item {item.id!r} logical_id differs from its request root",
                )
            for dependency in item.dependencies:
                if dependency not in by_id:
                    raise GraphValidationError(f"work item {item.id!r} has unknown dependency {dependency!r}")
                if dependency == item.id:
                    raise GraphValidationError(f"work item {item.id!r} depends on itself")
                if by_id[dependency].root_id != item.root_id:
                    raise GraphValidationError(
                        f"work item {item.id!r} cannot depend across request roots",
                    )
            if item.status == "superseded" and item.superseded_by not in by_id:
                raise GraphValidationError(
                    f"work item {item.id!r} has unknown replacement {item.superseded_by!r}",
                )
            if item.status == "superseded":
                replacement = by_id[item.superseded_by]
                same_request = replacement.root_id == item.root_id
                request_correction = item.kind == replacement.kind == "request"
                if not same_request and not request_correction:
                    raise GraphValidationError(
                        f"work item {item.id!r} replacement belongs to a different request root",
                    )
            for ref in item.source_refs:
                identity = (ref.source_length, ref.source_sha256)
                old = source_digests.setdefault(ref.event_id, identity)
                if old != identity:
                    raise GraphValidationError(
                        f"source event {ref.event_id!r} is bound to conflicting immutable content",
                    )
        self._reject_dependency_cycles(by_id)
        self._reject_supersession_cycles(by_id)

    @staticmethod
    def _reject_dependency_cycles(by_id: Mapping[str, WorkItem]) -> None:
        # Kahn's algorithm avoids recursion depth becoming a denial-of-service on a large valid graph.
        indegree = {item_id: 0 for item_id in by_id}
        dependants: dict[str, list[str]] = {item_id: [] for item_id in by_id}
        for item in by_id.values():
            for dependency in item.dependencies:
                indegree[item.id] += 1
                dependants[dependency].append(item.id)
        ready = [item_id for item_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            item_id = ready.pop()
            visited += 1
            for dependant in dependants[item_id]:
                indegree[dependant] -= 1
                if indegree[dependant] == 0:
                    ready.append(dependant)
        if visited != len(by_id):
            cyclic = sorted(item_id for item_id, degree in indegree.items() if degree)
            raise GraphValidationError(f"work dependency cycle detected: {', '.join(cyclic)}")

    @staticmethod
    def _reject_supersession_cycles(by_id: Mapping[str, WorkItem]) -> None:
        """A correction chain is directional history, never a way to resurrect old work."""
        for start in by_id:
            seen: set[str] = set()
            current = start
            while current:
                if current in seen:
                    raise GraphValidationError(f"work supersession cycle detected at {current!r}")
                seen.add(current)
                item = by_id[current]
                current = item.superseded_by if item.status == "superseded" else ""

    def get(self, item_id: str) -> WorkItem | None:
        return self._by_id.get(item_id)

    @property
    def request_roots(self) -> tuple[WorkItem, ...]:
        return tuple(item for item in self.items if item.kind == "request")

    @property
    def unresolved_roots(self) -> tuple[WorkItem, ...]:
        return tuple(item for item in self.request_roots if item.status in UNRESOLVED_STATUSES)

    def active_frontier(self) -> tuple[WorkItem, ...]:
        """Return every unresolved unit, including children below a delivered progress response."""
        return tuple(item for item in self.items if item.status in UNRESOLVED_STATUSES)

    def dependency_closure(self, roots: Iterable[str] | None = None) -> tuple[WorkItem, ...]:
        """Return roots plus transitive dependencies in stable graph order."""
        if isinstance(roots, (str, bytes)):
            raise GraphValidationError("dependency closure roots must be a sequence of item IDs")
        root_ids = tuple(roots) if roots is not None else tuple(item.id for item in self.active_frontier())
        if any(not isinstance(item_id, str) or item_id not in self._by_id for item_id in root_ids):
            raise GraphValidationError("dependency closure roots must name existing work items")
        selected: set[str] = set()
        pending = list(root_ids)
        while pending:
            item_id = pending.pop()
            if item_id in selected:
                continue
            selected.add(item_id)
            # ``root_id`` is an ownership edge. Include it even when a child does not redundantly list its
            # request root as an ordinary dependency.
            selected.add(self._by_id[item_id].root_id)
            pending.extend(self._by_id[item_id].dependencies)
        return tuple(item for item in self.items if item.id in selected)

    def validate_sources(self, sources: Mapping[str, str]) -> None:
        """Validate every source locator against an immutable-event lookup."""
        if not isinstance(sources, Mapping):
            raise SourceMismatchError("source lookup must be a mapping")
        for item in self.items:
            for ref in item.source_refs:
                if ref.event_id not in sources:
                    raise SourceMismatchError(f"source event {ref.event_id!r} is unavailable")
                ref.extract(sources[ref.event_id])

    def apply(self, delta: WorkDelta) -> "WorkGraph":
        """Validate and atomically apply one model-authored delta."""
        if not isinstance(delta, WorkDelta):
            raise TypeError("WorkGraph.apply requires a WorkDelta")
        if delta.expected_revision != self.revision:
            raise RevisionConflictError(
                f"work delta expected revision {delta.expected_revision}, current revision is {self.revision}",
            )
        existing = dict(self._by_id)
        for item in delta.creates:
            if item.id in existing:
                raise GraphValidationError(f"cannot create existing work item {item.id!r}")
        for item in delta.updates:
            previous = existing.get(item.id)
            if previous is None:
                raise GraphValidationError(f"cannot update missing work item {item.id!r}")
            self._validate_update(previous, item)

        updates = {item.id: item for item in delta.updates}
        next_items = tuple(updates.get(item.id, item) for item in self.items) + delta.creates
        if next_items == self.items:
            return self
        return WorkGraph(items=next_items, revision=self.revision + 1)

    def apply_delta(self, delta: WorkDelta) -> "WorkGraph":
        """Explicit integration name for :meth:`apply`."""
        return self.apply(delta)

    @staticmethod
    def _validate_update(previous: WorkItem, current: WorkItem) -> None:
        if current.kind != previous.kind:
            raise GraphValidationError(f"work item {current.id!r} cannot change kind")
        if current.root_id != previous.root_id:
            raise GraphValidationError(f"work item {current.id!r} cannot change request root")
        if current.logical_id != previous.logical_id:
            raise GraphValidationError(f"work item {current.id!r} cannot change logical request identity")
        if current.workspace_epoch != previous.workspace_epoch:
            raise GraphValidationError(f"work item {current.id!r} cannot change its admission workspace epoch")
        if current.status not in _ALLOWED_TRANSITIONS[previous.status]:
            raise GraphValidationError(
                f"invalid work status transition for {current.id!r}: {previous.status} -> {current.status}",
            )
        if not set(previous.source_refs).issubset(current.source_refs):
            raise GraphValidationError(f"work item {current.id!r} cannot erase source provenance")
        if not set(previous.dependencies).issubset(current.dependencies):
            raise GraphValidationError(f"work item {current.id!r} cannot erase dependency edges")
        if not set(previous.resource_refs).issubset(current.resource_refs):
            raise GraphValidationError(f"work item {current.id!r} cannot erase resource references")
        if not set(previous.evidence_refs).issubset(current.evidence_refs):
            raise GraphValidationError(f"work item {current.id!r} cannot erase evidence references")
        if not set(previous.output_refs).issubset(current.output_refs):
            raise GraphValidationError(f"work item {current.id!r} cannot erase output references")

    def open_request(self, source_artifact: str, text: str, *, workspace_epoch: int = 0,
                     logical_id: str | None = None, item_id: str | None = None) -> "WorkGraph":
        """Mechanically create exactly one request root for one exact user event.

        Retrying the same event is idempotent and does not advance the revision.  No NLP,
        classification, or model paraphrase participates in request admission.
        """
        candidate = request_root_item(
            source_artifact,
            text,
            workspace_epoch=workspace_epoch,
            logical_id=logical_id,
            item_id=item_id,
        )
        for root in self.request_roots:
            if root.logical_id == candidate.logical_id or any(
                    ref.event_id == source_artifact for ref in root.source_refs):
                # Lifecycle/output/evidence fields legitimately change after admission.  A retry is the same
                # admission when every immutable identity/provenance field still matches; comparing the whole
                # root to a fresh ``open`` candidate would make a crash retry fail after any progress transition.
                same_admission = (
                    root.id == candidate.id
                    and root.root_id == candidate.root_id
                    and root.logical_id == candidate.logical_id
                    and root.workspace_epoch == candidate.workspace_epoch
                    and set(candidate.source_refs).issubset(root.source_refs)
                )
                if same_admission:
                    return self
                raise GraphValidationError(
                    f"source/logical request {source_artifact!r}/{candidate.logical_id!r} "
                    f"already owns request root {root.id!r}",
                )
        if candidate.id in self._by_id:
            raise GraphValidationError(f"request-root ID {candidate.id!r} already belongs to another item")
        # Terminal roots are durable in the event/artifact stores, not resident work. Drop their complete
        # ownership subgraphs when a distinct request arrives so checkpoint size follows unresolved work rather
        # than elapsed turns. Existing-source idempotency is checked above before this compaction.
        terminal_roots = {
            root.id for root in self.request_roots if root.status not in UNRESOLVED_STATUSES
        }
        base = self
        if terminal_roots:
            kept = tuple(item for item in self.items if item.root_id not in terminal_roots)
            base = WorkGraph(items=kept, revision=self.revision, version=self.version)
        return base.apply(WorkDelta(expected_revision=self.revision, creates=(candidate,)))

    def add_request_root(self, event_id: str, utterance: str, *, item_id: str | None = None) -> "WorkGraph":
        """Backward-compatible spelling for callers that do not yet carry segment identity."""
        return self.open_request(event_id, utterance, item_id=item_id)

    def upsert(self, item: WorkItem, *, expected_revision: int | None = None) -> "WorkGraph":
        """Create or replace one item through the same validated delta boundary."""
        if not isinstance(item, WorkItem):
            raise TypeError("WorkGraph.upsert requires a WorkItem")
        revision = self.revision if expected_revision is None else expected_revision
        if item.id in self._by_id:
            if self._by_id[item.id] == item and revision == self.revision:
                return self
            return self.apply_delta(WorkDelta(expected_revision=revision, updates=(item,)))
        return self.apply_delta(WorkDelta(expected_revision=revision, creates=(item,)))

    def transition(self, item_id: str, status: WorkStatus, *,
                   evidence_refs: Iterable[EvidenceRef] = (),
                   output_refs: Iterable[OutputRef] = (), superseded_by: str = "",
                   stop_reason: str | None = None,
                   expected_revision: int | None = None) -> "WorkGraph":
        """Transition one item while append-only references remain mechanically preserved."""
        current = self.get(item_id)
        if current is None:
            raise GraphValidationError(f"cannot transition missing work item {item_id!r}")
        evidence = _record_tuple(evidence_refs, EvidenceRef, "transition.evidence_refs")
        outputs = _record_tuple(output_refs, OutputRef, "transition.output_refs")
        updated = replace(
            current,
            status=status,
            evidence_refs=tuple(dict.fromkeys(current.evidence_refs + evidence)),
            output_refs=tuple(dict.fromkeys(current.output_refs + outputs)),
            superseded_by=superseded_by,
            stop_reason=current.stop_reason if stop_reason is None else stop_reason,
        )
        return self.upsert(updated, expected_revision=expected_revision)

    def seal_current(self, stop_reason: str, response_ref: OutputRef | None = None, *,
                     transitioned: bool = False, logical_id: str | None = None,
                     expected_revision: int | None = None) -> "WorkGraph":
        """Seal one runtime segment without confusing transport with task completion.

        A context/workspace transition keeps the request ``in_progress`` even if a progress
        response was emitted.  ``waiting_user`` is the one explicit stop reason that keeps
        the request pending on dialogue.  Other stops with a response are delivered; stops
        without a response remain active for recovery.
        """
        _text(stop_reason, "seal stop_reason")
        if response_ref is not None and not isinstance(response_ref, OutputRef):
            raise TypeError("seal_current response_ref must be an OutputRef or None")
        candidates = [item for item in self.unresolved_roots
                      if logical_id is None or item.logical_id == logical_id]
        if not candidates:
            raise GraphValidationError("there is no unresolved request root to seal")
        current = candidates[-1]
        deliver_ready = bool(response_ref is not None and not transitioned and stop_reason != "waiting_user")
        ready_children = tuple(
            item for item in self.items
            if item.id != current.id and item.root_id == current.id and item.status == "ready"
        ) if deliver_ready else ()
        unresolved_children = any(
            item.id != current.id
            and item.root_id == current.id
            and item.status in UNRESOLVED_STATUSES
            and item.status != "ready"
            for item in self.items
        )
        if transitioned:
            status: WorkStatus = "in_progress"
        elif stop_reason == "waiting_user":
            status = "waiting_user"
        elif response_ref is not None and not unresolved_children:
            status = "delivered"
        else:
            status = "in_progress" if current.status == "open" else current.status
        outputs = (response_ref,) if response_ref is not None else ()
        updated_root = replace(
            current,
            status=status,
            output_refs=tuple(dict.fromkeys(current.output_refs + outputs)),
            stop_reason=stop_reason,
        )
        updated_children = tuple(
            replace(
                child,
                status="delivered",
                output_refs=tuple(dict.fromkeys(child.output_refs + outputs)),
                stop_reason=stop_reason,
            )
            for child in ready_children
        )
        revision = self.revision if expected_revision is None else expected_revision
        return self.apply_delta(WorkDelta(
            expected_revision=revision,
            updates=(*updated_children, updated_root),
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.version,
            "revision": self.revision,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkGraph":
        if not isinstance(value, Mapping):
            raise GraphValidationError("work graph must be an object")
        items = value.get("items", ())
        if items is None:
            items = ()
        if isinstance(items, (str, bytes, Mapping)):
            raise GraphValidationError("work_graph.items must be a sequence")
        try:
            parsed = tuple(WorkItem.from_dict(item) for item in items)
        except TypeError as exc:
            raise GraphValidationError("work_graph.items must contain objects") from exc
        return cls(
            items=parsed,
            revision=value.get("revision", 0),
            version=value.get("v", WORK_GRAPH_VERSION),
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return checkpoint-friendly records without losing graph revision/version."""
        return [
            {"record_type": "active_work_graph", "v": self.version, "revision": self.revision},
            *(dict(item.to_dict(), record_type="work_item") for item in self.items),
        ]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]] | None) -> "WorkGraph":
        """Rebuild from checkpoint records; absence means a pre-Active-Work checkpoint."""
        if records is None:
            return cls()
        records = _wire_sequence(records, "active-work records")
        if not records:
            return cls()
        if any(not isinstance(record, Mapping) for record in records):
            raise GraphValidationError("active-work records must contain objects")
        header = records[0]
        if header.get("record_type") != "active_work_graph":
            raise GraphValidationError("active-work records are missing the graph header")
        items = []
        for record in records[1:]:
            if record.get("record_type") != "work_item":
                raise GraphValidationError("active-work record has an unsupported record_type")
            value = dict(record)
            value.pop("record_type", None)
            items.append(WorkItem.from_dict(value))
        return cls(
            items=tuple(items),
            revision=header.get("revision", 0),
            version=header.get("v", WORK_GRAPH_VERSION),
        )

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def request_root_item(event_id: str, utterance: str, *, workspace_epoch: int = 0,
                      logical_id: str | None = None, item_id: str | None = None) -> WorkItem:
    """Create the canonical, non-semantic request root for one exact user utterance."""
    _text(event_id, "request event_id")
    if not isinstance(utterance, str) or not utterance:
        raise GraphValidationError("request utterance must be a non-empty string")
    _integer(workspace_epoch, "request workspace_epoch")
    if logical_id is None:
        logical_id = event_id
    _text(logical_id, "request logical_id")
    if item_id is None:
        item_id = f"request-{hashlib.sha256(logical_id.encode('utf-8')).hexdigest()[:24]}"
    _text(item_id, "request item_id")
    source = SourceRef.bind(event_id, utterance)
    return WorkItem(
        id=item_id,
        root_id=item_id,
        kind="request",
        status="open",
        description="",
        logical_id=logical_id,
        workspace_epoch=workspace_epoch,
        source_refs=(source,),
    )


def attach_child_artifacts(
    graph: WorkGraph,
    recent_calls: Iterable[Mapping[str, Any]],
    *,
    workspace_epoch: int,
) -> WorkGraph:
    """Promote sealed child effects into their immutable Active Work binding.

    This reducer is shared by live settlement, normal turn sealing, and crash replay.  A bound child's terminal
    *execution* state is host arithmetic: a successful seal makes that delegated work item ``ready`` for parent
    synthesis; a determinate failed/cancelled seal makes the attempt ``cancelled`` while its evidence gap remains
    visible.  Neither transition claims that the child's testimony is correct, parent-verified, or delivered.
    Indeterminate execution remains unresolved for reconciliation.
    """
    if not isinstance(graph, WorkGraph):
        raise TypeError("attach_child_artifacts requires a WorkGraph")
    _integer(workspace_epoch, "child artifact workspace_epoch")
    from .fan_in import build_fan_in_manifest

    current = graph
    # Fold only at the existing normal seal/replay chokepoint.  The live model trajectory keeps its original
    # WorkGraph revision, so adding parent-use accounting cannot manufacture a mid-turn CAS conflict.
    # Join this segment's parent-read receipts to already persisted child rows. A later complete open may have
    # no new spawn event; omitting the graph would make it live-correct yet lose the upgrade at the next reload.
    manifest = build_fan_in_manifest(recent_calls, graph=graph, max_children=None)
    for child in manifest.children:
        work_item_id = child.work_item_id
        artifact_ref = child.artifact_id
        item = current.get(work_item_id) if work_item_id else None
        if item is None or item.kind != "task" or not artifact_ref:
            continue
        source_qualifier = str(child.source_coverage_status or "").strip().casefold()
        source_qualifier = {
            "grounded": "source_complete", "partial": "source_partial",
            "unsupported": "source_unsupported",
        }.get(source_qualifier, source_qualifier)
        if source_qualifier not in {
            "source_complete", "source_partial", "source_unsupported", "not_assessed",
        }:
            source_qualifier = ""
        additions = [EvidenceRef(
            "child_artifact", artifact_ref, qualifier=source_qualifier,
        )]
        if child.operational_declared and child.operational_status:
            additions.append(EvidenceRef(
                "child_operational_status", artifact_ref,
                qualifier=child.operational_status,
            ))
        if child.digest_delivered:
            additions.append(EvidenceRef("child_digest_delivered", artifact_ref))
        if child.artifact_opened in {"partial", "complete"}:
            additions.append(EvidenceRef(
                "child_artifact_opened", artifact_ref, qualifier=child.artifact_opened,
            ))
        if child.policy_declared:
            additions.append(EvidenceRef(
                "child_integration_policy", artifact_ref,
                qualifier=child.integration_policy,
            ))
        if child.evidence_declared:
            additions.append(EvidenceRef(
                "child_evidence_status", artifact_ref,
                qualifier=child.evidence_status,
            ))
        if child.evidence_account:
            # Persist the bounded mechanical census, not merely its coarse label. This is derived truth and
            # survives restart/replay without turning the child report into prompt narration.
            additions.append(EvidenceRef(
                "child_evidence_account", artifact_ref,
                qualifier=json.dumps(
                    dict(child.evidence_account), ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"),
                ),
            ))
        operational = str(child.operational_status or "").strip().casefold()
        terminal_status = item.status
        stop_reason = item.stop_reason
        if child.operational_declared and item.status in {
            "open", "in_progress", "waiting_user", "ready",
        }:
            if operational in {"succeeded", "success", "ok", "end_turn", "ready", "sealed"}:
                terminal_status = "ready"
            elif operational in {
                "failed", "failure", "error", "cancelled", "canceled", "timeout", "max_tokens",
            }:
                terminal_status = "cancelled"
                stop_reason = f"child_{operational}"
        current = current.upsert(replace(
            item,
            status=terminal_status,
            stop_reason=stop_reason,
            evidence_refs=tuple(dict.fromkeys((*item.evidence_refs, *additions))),
            resource_refs=tuple(dict.fromkeys((*item.resource_refs, ResourceRef(
                "subagent", artifact_ref, workspace_epoch=workspace_epoch,
            )))),
        ))
    return current


__all__ = [
    "ActiveWorkError",
    "EvidenceRef",
    "GraphValidationError",
    "OutputRef",
    "ResourceRef",
    "RevisionConflictError",
    "SOURCE_REF_VERSION",
    "SourceMismatchError",
    "SourceRef",
    "UNRESOLVED_STATUSES",
    "WORK_GRAPH_VERSION",
    "WORK_KINDS",
    "WORK_STATUSES",
    "WorkDelta",
    "WorkGraph",
    "WorkItem",
    "WorkKind",
    "WorkStatus",
    "attach_child_artifacts",
    "request_root_item",
]
