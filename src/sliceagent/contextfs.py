"""Permanent, read-only virtual address space for SliceAgent's internal context.

``ContextFS`` deliberately knows nothing about the physical stores behind evidence,
history, active work, knowledge, or the roster.  Hosts inject small providers at
canonical mount points and route the ordinary read/list/grep tools here whenever a
path starts with ``@sliceagent``.  This keeps the model-facing floor plan stable while
the storage implementations evolve independently.

The root manifest is owned by this module rather than any optional backend.  It is
therefore available even when a provider (notably typed L2 knowledge) is absent or
failing, and it reports unknown/unavailable states explicitly instead of inferring
health from the presence of an object.
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Pattern, Protocol, runtime_checkable


CONTEXT_ROOT = "@sliceagent"
CONTEXTFS_SCHEMA_MARKER = "@sliceagent/index.md"
CONTEXT_REGIONS = ("evidence", "history", "work", "memory", "roster")

# The canonical floor plan remains visible even before a backend is mounted.  A
# trailing slash marks a directory in list_files output.
_CANONICAL_CHILDREN: dict[str, tuple[str, ...]] = {
    "": ("index.md", "evidence/", "history/", "work/", "memory/", "roster/"),
    "evidence": ("index.md", "events/", "turns/", "children/", "receipts/"),
    "evidence/events": (),
    "evidence/turns": (),
    "evidence/children": (),
    "evidence/receipts": (),
    "history": ("index.md", "sessions/", "search.md"),
    "history/sessions": (),
    "work": ("active.md", "plan.md", "dependencies.md", "receipts.md"),
    "memory": (
        "index.md", "status.md", "diagnostics.md", "user/", "project/", "craft/", "records/",
    ),
    "memory/user": ("index.md",),
    "memory/project": ("index.md",),
    "memory/craft": ("index.md",),
    "memory/records": (),
    "roster": ("index.md",),
}
_CANONICAL_DIRS = frozenset(_CANONICAL_CHILDREN)
_CANONICAL_FILES = frozenset(
    f"{parent}/{entry}".strip("/")
    for parent, entries in _CANONICAL_CHILDREN.items()
    for entry in entries
    if not entry.endswith("/")
)
_CONTEXTFS_OWNED_FILES = frozenset({"memory/status.md", "memory/diagnostics.md"})
_DYNAMIC_DIRS = (
    "evidence/events", "evidence/turns", "evidence/children", "evidence/receipts",
    "history/sessions", "memory/records", "roster",
)
_STATUS_STATES = frozenset({
    "available", "healthy", "empty", "degraded", "unavailable", "disabled", "unknown",
})
_MAX_PROVIDER_WALK = 2_000
_MAX_PROVIDER_GREP = 10_000


class ContextFSError(RuntimeError):
    """Base error for the virtual context namespace."""


class ContextPathError(ContextFSError, ValueError):
    """A path is malformed or attempts to traverse out of ``@sliceagent``."""


class ContextNotFoundError(ContextFSError, FileNotFoundError):
    """The requested virtual document or directory does not exist."""


class ContextReadOnlyError(ContextFSError, PermissionError):
    """A caller attempted to mutate the internal context namespace."""


@dataclass(frozen=True)
class CapabilityStatus:
    """Truthful health/availability projection for a region or retrieval backend."""

    state: str = "unknown"
    detail: str = ""
    item_count: int | None = None


@dataclass(frozen=True)
class ContextStatus:
    """Live values rendered by ``@sliceagent/index.md``.

    Unknown values stay ``None``.  In particular, zero is a real count and is never
    conflated with a missing measurement.
    """

    current_project: str | None = None
    current_workspace: str | None = None
    logical_request: str | None = None
    regions: Mapping[str, CapabilityStatus | Mapping[str, Any] | str] = field(default_factory=dict)
    open_active_work_count: int | None = None
    knowledge_counts: Mapping[str, int | None] = field(default_factory=dict)
    legacy_inventory: Mapping[str, int | None] = field(default_factory=dict)
    legacy_inventory_scope: str | None = None
    legacy_inventory_status: CapabilityStatus | Mapping[str, Any] | str = field(
        default_factory=lambda: CapabilityStatus("unknown", "not reported by host"),
    )
    compatibility_transition: CapabilityStatus | Mapping[str, Any] | str = field(
        default_factory=lambda: CapabilityStatus("unknown", "not reported by host"),
    )
    compatibility_health: CapabilityStatus | Mapping[str, Any] | str = field(
        default_factory=lambda: CapabilityStatus("unknown", "not reported by host"),
    )
    retirement_gate: CapabilityStatus | Mapping[str, Any] | str = field(
        default_factory=lambda: CapabilityStatus("unknown", "not reported by host"),
    )
    last_consolidation: CapabilityStatus | Mapping[str, Any] | str = field(
        default_factory=lambda: CapabilityStatus("unknown", "not reported by host"),
    )
    native_index: CapabilityStatus | Mapping[str, Any] | str = field(
        default_factory=lambda: CapabilityStatus("unknown", "health not reported"),
    )
    memem: CapabilityStatus | Mapping[str, Any] | str = field(
        default_factory=lambda: CapabilityStatus("unknown", "status not reported"),
    )
    cross_project_search_policy: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ContextStatus":
        """Accept a host-owned live dict without forcing it to import this dataclass."""

        memory_status = raw.get("memory_status")
        memory_status = memory_status if isinstance(memory_status, Mapping) else {}
        legacy_inventory = raw.get("legacy_inventory", raw.get("memory_inventory"))
        if not isinstance(legacy_inventory, Mapping):
            legacy_inventory = memory_status.get("legacy_inventory", memory_status.get("inventory", {}))
        if not isinstance(legacy_inventory, Mapping):
            legacy_inventory = {}

        return cls(
            current_project=_optional_text(raw.get("current_project", raw.get("project"))),
            current_workspace=_optional_text(raw.get("current_workspace", raw.get("workspace"))),
            logical_request=_optional_text(raw.get("logical_request", raw.get("current_logical_request"))),
            regions=(raw.get("regions") if isinstance(raw.get("regions"), Mapping) else {}),
            open_active_work_count=_optional_count(
                raw.get("open_active_work_count", raw.get("open_work_count")),
            ),
            knowledge_counts=(
                raw.get("knowledge_counts") if isinstance(raw.get("knowledge_counts"), Mapping) else {}
            ),
            legacy_inventory=legacy_inventory,
            legacy_inventory_scope=_optional_text(raw.get(
                "legacy_inventory_scope", memory_status.get("legacy_inventory_scope"),
            )),
            legacy_inventory_status=raw.get(
                "legacy_inventory_status",
                memory_status.get("legacy_inventory_status", "unknown"),
            ),
            compatibility_transition=raw.get(
                "compatibility_transition",
                raw.get("migration", raw.get("migration_state", memory_status.get(
                    "compatibility_transition", memory_status.get("migration", memory_status.get(
                        "migration_state", CapabilityStatus("unknown", "not reported by host"),
                    )),
                ))),
            ),
            compatibility_health=raw.get(
                "compatibility_health",
                memory_status.get("compatibility_health", CapabilityStatus(
                    "unknown", "not reported by host",
                )),
            ),
            retirement_gate=raw.get(
                "retirement_gate",
                memory_status.get("retirement_gate", CapabilityStatus(
                    "unknown", "not reported by host",
                )),
            ),
            last_consolidation=raw.get(
                "last_consolidation",
                raw.get("consolidation", memory_status.get("last_consolidation", memory_status.get(
                    "consolidation",
                    CapabilityStatus("unknown", "not reported by host"),
                ))),
            ),
            native_index=raw.get("native_index", raw.get("native_index_health", "unknown")),
            memem=raw.get("memem", raw.get("memem_status", "unknown")),
            cross_project_search_policy=_optional_text(
                raw.get("cross_project_search_policy", raw.get("cross_project_search")),
            ),
        )


@dataclass(frozen=True)
class ContextMatch:
    """One provider grep hit. ``path`` is relative to that provider's mount."""

    path: str
    line_number: int
    line: str


@runtime_checkable
class ContextProvider(Protocol):
    """Minimal provider contract; paths are relative to the injected mount point."""

    def read_file(self, path: str) -> str: ...
    def list_files(self, path: str = "") -> Sequence[str] | str: ...


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _one_line(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "(unknown)"
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def is_context_path(path: os.PathLike[str] | str) -> bool:
    """Return whether a spelling targets the reserved namespace.

    Traversal attempts still return ``True`` so a host routes them here and rejects
    them rather than accidentally treating them as ordinary workspace paths.
    """

    try:
        value = os.fspath(path)
    except TypeError:
        return False
    if not isinstance(value, str):
        return False
    value = value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value == CONTEXT_ROOT or value.startswith(CONTEXT_ROOT + "/")


def normalize_context_path(path: os.PathLike[str] | str) -> str:
    """Canonicalize a virtual path without ever resolving it through the host FS."""

    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise ContextPathError("SliceAgent context paths must be strings") from exc
    if not isinstance(value, str):
        raise ContextPathError("SliceAgent context paths must be text, not bytes")
    if "\x00" in value:
        raise ContextPathError("SliceAgent context paths cannot contain NUL bytes")
    value = value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    if value != CONTEXT_ROOT and not value.startswith(CONTEXT_ROOT + "/"):
        raise ContextPathError(f"{value or '(empty path)'} is not under {CONTEXT_ROOT}/")
    tail = value[len(CONTEXT_ROOT):].lstrip("/")
    raw_parts = tail.split("/") if tail else []
    if any(part == ".." for part in raw_parts):
        raise ContextPathError(
            f"invalid {CONTEXT_ROOT}/ path: parent traversal ('..') is not allowed",
        )
    parts = [part for part in raw_parts if part not in ("", ".")]
    return CONTEXT_ROOT + (("/" + "/".join(parts)) if parts else "")


def _relative_context_path(path: os.PathLike[str] | str) -> str:
    canonical = normalize_context_path(path)
    return canonical[len(CONTEXT_ROOT):].lstrip("/")


def _normalize_provider_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/").strip("/")
    raw = value.split("/") if value else []
    if any(part == ".." for part in raw):
        raise ContextPathError("provider paths cannot contain parent traversal ('..')")
    if any("\x00" in part for part in raw):
        raise ContextPathError("provider paths cannot contain NUL bytes")
    return "/".join(part for part in raw if part not in ("", "."))


def _coerce_capability(value: Any, *, default_detail: str = "") -> CapabilityStatus:
    if isinstance(value, CapabilityStatus):
        state = str(value.state or "unknown").lower()
        return (CapabilityStatus(state, value.detail, value.item_count) if state in _STATUS_STATES
                else CapabilityStatus("unknown", value.detail or f"unrecognized status: {state}", value.item_count))
    if isinstance(value, Mapping):
        state = str(value.get("state", value.get("status", "unknown")) or "unknown").lower()
        detail = str(value.get("detail", value.get("message", "")) or "")
        if state not in _STATUS_STATES:
            detail = detail or f"unrecognized status: {state}"
            state = "unknown"
        return CapabilityStatus(state, detail, _optional_count(value.get("item_count", value.get("count"))))
    text = str(value or "unknown").strip()
    if text.lower() in _STATUS_STATES:
        return CapabilityStatus(text.lower(), default_detail)
    return CapabilityStatus("unknown", text or default_detail)


def _coerce_lifecycle(value: Any, *, default_detail: str = "") -> CapabilityStatus:
    """Project a host-owned transition/consolidation state without inventing health.

    Lifecycle vocabularies evolve independently of capability health (for example
    ``completed_with_rejections``).  Preserve a bounded machine state instead of
    silently turning every new truthful state into ``unknown``.
    """

    if isinstance(value, CapabilityStatus):
        return value
    if isinstance(value, Mapping):
        raw_state = value.get("state", value.get("status", "unknown"))
        detail = str(value.get("detail", value.get("message", default_detail)) or "")
        count = _optional_count(value.get("item_count", value.get("count")))
    else:
        raw_state, detail, count = value, default_detail, None
    state = str(raw_state or "unknown").strip().lower().replace("_", "-")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", state):
        return CapabilityStatus("unknown", detail or "invalid lifecycle state", count)
    return CapabilityStatus(state, detail, count)


def _description_strings(schema: Mapping[str, Any]) -> Iterable[str]:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ()
    values = []
    description = function.get("description")
    if isinstance(description, str):
        values.append(description)
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    if isinstance(properties, Mapping):
        for prop in properties.values():
            if isinstance(prop, Mapping) and isinstance(prop.get("description"), str):
                values.append(prop["description"])
    return values


def schemas_advertise_contextfs(schemas: Iterable[Mapping[str, Any]] | None) -> bool:
    """Whether live file-tool schemas truthfully advertise the canonical locator.

    The marker lives in ordinary descriptions rather than a custom JSON-schema key,
    avoiding provider-specific schema rejection.  Hosts add it only while ContextFS is
    bound, and prompt compilation uses this helper instead of advertising a dead path.
    """

    for schema in schemas or ():
        function = schema.get("function") if isinstance(schema, Mapping) else None
        name = str(function.get("name") or "") if isinstance(function, Mapping) else ""
        if name not in {"read_file", "list_files", "grep"}:
            continue
        if any(CONTEXTFS_SCHEMA_MARKER in value for value in _description_strings(schema)):
            return True
    return False


class MappingContextProvider:
    """Small live provider backed by relative virtual documents.

    ``documents`` may be a mapping or a callable returning one.  Individual document
    values may also be callables, which is useful for Active Work and backend status
    pages that must render from a fresh snapshot on every read.
    """

    def __init__(
        self,
        documents: Mapping[str, str | Callable[[], str]]
        | Callable[[], Mapping[str, str | Callable[[], str]]],
    ):
        self._documents = documents

    def _snapshot(self) -> dict[str, str | Callable[[], str]]:
        raw = self._documents() if callable(self._documents) else self._documents
        if not isinstance(raw, Mapping):
            raise TypeError("context document provider must return a mapping")
        docs: dict[str, str | Callable[[], str]] = {}
        for path, value in raw.items():
            rel = _normalize_provider_path(str(path))
            if not rel:
                raise ContextPathError("context documents need a non-empty relative path")
            docs[rel] = value
        return docs

    @staticmethod
    def _value(value: str | Callable[[], str]) -> str:
        rendered = value() if callable(value) else value
        return str(rendered)

    def read_file(self, path: str) -> str:
        rel = _normalize_provider_path(path)
        docs = self._snapshot()
        target = rel
        if target not in docs and ((not rel) or any(name.startswith(rel + "/") for name in docs)):
            target = f"{rel}/index.md".strip("/")
        if target not in docs:
            raise ContextNotFoundError(f"no such context document: {rel or 'index.md'}")
        return self._value(docs[target])

    def list_files(self, path: str = "") -> tuple[str, ...]:
        rel = _normalize_provider_path(path)
        docs = self._snapshot()
        if rel in docs:
            raise ContextNotFoundError(f"{rel} is a context document, not a directory")
        prefix = rel + "/" if rel else ""
        entries: set[str] = set()
        for name in docs:
            if not name.startswith(prefix):
                continue
            tail = name[len(prefix):]
            if not tail:
                continue
            first, separator, _rest = tail.partition("/")
            entries.add(first + ("/" if separator else ""))
        if not entries and rel and not any(name.startswith(prefix) for name in docs):
            raise ContextNotFoundError(f"no such context directory: {rel}")
        return tuple(sorted(entries, key=lambda item: (not item.endswith("/"), item)))

    def grep_matches(self, pattern: Pattern[str], path: str = "") -> tuple[ContextMatch, ...]:
        rel = _normalize_provider_path(path)
        docs = self._snapshot()
        if rel in docs:
            selected = ((rel, docs[rel]),)
        else:
            prefix = rel + "/" if rel else ""
            selected = tuple((name, value) for name, value in docs.items() if name.startswith(prefix))
        hits = []
        for name, value in selected:
            for line_number, line in enumerate(self._value(value).splitlines(), 1):
                if pattern.search(line):
                    hits.append(ContextMatch(name, line_number, line))
        return tuple(hits)


class LedgerContextProvider:
    """Read-only exact event view over an application EventLedger-like object.

    Discovery lists only the current application ledger.  An exact event ID may fault from an archived sibling
    ledger through ``resolve_events``; this preserves Active Work provenance across restarts without injecting
    unrelated old sessions into context.
    """

    def __init__(self, ledger: Any | Callable[[], Any]):
        self._source = ledger

    def _ledger(self) -> Any:
        value = self._source() if callable(self._source) else self._source
        if value is None:
            # The canonical mount exists but its backing capability is unavailable. Treating this as a clean
            # missing document makes list_files report an empty ledger and erases a real evidence gap.
            raise ContextFSError("application event ledger is not bound")
        return value

    @staticmethod
    def _event_id(path: str) -> str:
        rel = _normalize_provider_path(path)
        if not rel.endswith(".md") or "/" in rel:
            raise ContextNotFoundError(f"no such event document: {rel or 'index.md'}")
        return rel[:-3]

    def _events(self) -> tuple[Any, ...]:
        events = getattr(self._ledger(), "events", None)
        if not callable(events):
            raise TypeError("event ledger provider does not expose events()")
        return tuple(events())

    def _get(self, identity: str) -> Any:
        ledger = self._ledger()
        getter = getattr(ledger, "get", None)
        event = getter(identity) if callable(getter) else None
        if event is None:
            resolver = getattr(ledger, "resolve_events", None)
            resolved = resolver((identity,)) if callable(resolver) else {}
            event = resolved.get(identity) if isinstance(resolved, Mapping) else None
        if event is None:
            raise ContextNotFoundError(f"no retained event {identity!r}")
        return event

    @staticmethod
    def _render(event: Any) -> str:
        identity = _one_line(getattr(event, "id", ""), limit=180)
        kind = _one_line(getattr(event, "kind", ""), limit=80)
        payload = getattr(event, "payload", {})
        payload = dict(payload) if isinstance(payload, Mapping) else {}
        lines = [
            f"# {kind.upper()} EVENT",
            "Historical immutable evidence. It is not a current instruction or proof of current world state.",
            f"- id: {identity}",
            f"- session: {_one_line(getattr(event, 'session_id', ''))}",
            f"- logical turn: {_one_line(getattr(event, 'logical_turn_id', ''))}",
            f"- task: {_one_line(getattr(event, 'task_id', ''))}",
            f"- segment: {_one_line(getattr(event, 'segment_id', ''))}",
            f"- workspace epoch: {getattr(event, 'workspace_epoch', '(unknown)')}",
            f"- workspace id: {_one_line(getattr(event, 'workspace_id', ''))}",
        ]
        text_value = payload.get("text") if kind == "user_utterance" else None
        if isinstance(text_value, str):
            lines += ["", "## User utterance (persisted verbatim after length-preserving secret redaction)"]
            lines.extend("> " + row for row in text_value.split("\n"))
            payload.pop("text", None)
        if payload:
            lines += ["", "## Event payload", "```json",
                      json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str), "```"]
        return "\n".join(lines)

    def read_file(self, path: str) -> str:
        rel = _normalize_provider_path(path)
        if rel in ("", "index.md"):
            events = self._events()
            rows = ["# APPLICATION EVENTS — exact source and lifecycle evidence"]
            rows.extend(
                f'- {event.id}.md · {event.kind} · logical turn {event.logical_turn_id} '
                f'→ read_file("@sliceagent/evidence/events/{event.id}.md")'
                for event in events
            )
            if not events:
                rows.append("(none yet)")
            return "\n".join(rows)
        return self._render(self._get(self._event_id(rel)))

    def list_files(self, path: str = "") -> tuple[str, ...]:
        rel = _normalize_provider_path(path)
        if rel:
            raise ContextNotFoundError(f"{rel} is not an event directory")
        return ("index.md", *(f"{event.id}.md" for event in self._events()))

    def grep_matches(self, pattern: Pattern[str], path: str = "") -> tuple[ContextMatch, ...]:
        rel = _normalize_provider_path(path)
        names = (rel,) if rel else self.list_files("")
        hits: list[ContextMatch] = []
        for name in names:
            try:
                body = self.read_file(name)
            except ContextNotFoundError:
                continue
            for line_number, line in enumerate(body.splitlines(), 1):
                if pattern.search(line):
                    hits.append(ContextMatch(name, line_number, line))
        return tuple(hits)


class ArtifactContextProvider:
    """Canonical filtered view over CoreArtifactFS without exposing its physical-era mount name."""

    def __init__(self, provider: Any, *, kinds: Iterable[str], canonical_mount: str, title: str):
        if not callable(getattr(provider, "read_file", None)):
            raise TypeError("artifact context provider requires read_file")
        self.provider = provider
        self.kinds = frozenset(str(kind) for kind in kinds)
        self.canonical_mount = normalize_context_path(canonical_mount).rstrip("/")
        self.title = str(title or "EVIDENCE")

    def _listing(self) -> Any:
        getter = getattr(self.provider, "_artifacts", None)
        if not callable(getter):
            raise TypeError("artifact context provider cannot enumerate artifacts")
        return getter()

    def _artifacts(self, listing: Any | None = None) -> tuple[Any, ...]:
        listing = self._listing() if listing is None else listing
        return tuple(item for item in listing if str(getattr(item, "kind", "")) in self.kinds)

    def _get(self, identity: str) -> Any:
        getter = getattr(self.provider, "_get", None)
        if not callable(getter):
            raise TypeError("artifact context provider cannot fault exact artifacts")
        try:
            artifact = getter(identity)
        except KeyError as exc:
            # Lightweight test/embedding providers commonly use KeyError for a clean miss.
            raise ContextNotFoundError(f"no retained artifact {identity!r}") from exc
        except Exception as exc:
            # CoreArtifactFS deliberately distinguishes a clean miss from corrupt canonical evidence. Preserve
            # that distinction: converting ArtifactCorruptError into "not found" would erase an evidence gap.
            if type(exc).__name__ == "ArtifactNotFoundError":
                raise ContextNotFoundError(f"no retained artifact {identity!r}") from exc
            raise
        if str(getattr(artifact, "kind", "")) not in self.kinds:
            raise ContextNotFoundError(f"artifact {identity!r} is not part of this evidence view")
        return artifact

    @staticmethod
    def _identity(path: str) -> str:
        rel = _normalize_provider_path(path)
        if not rel.endswith(".md") or "/" in rel:
            raise ContextNotFoundError(f"no such artifact document: {rel or 'index.md'}")
        return rel[:-3]

    @staticmethod
    def _nested_identity(path: str) -> tuple[str, str]:
        rel = _normalize_provider_path(path)
        identity, separator, suffix = rel.partition("/")
        if not separator or not identity or not suffix:
            raise ContextNotFoundError(f"no such artifact document: {rel or 'index.md'}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", identity):
            raise ContextNotFoundError(f"no such artifact document: {rel}")
        return identity, suffix

    def _read_artifact(self, artifact: Any) -> str:
        renderer = getattr(self.provider, "_render", None)
        if callable(renderer):
            return str(renderer(artifact))
        return str(self.provider.read_file(f"artifacts/{artifact.id}.md"))

    def read_file(self, path: str) -> str:
        rel = _normalize_provider_path(path)
        if rel in ("", "index.md"):
            listing = self._listing()
            artifacts = self._artifacts(listing)
            rows = [f"# {self.title}"]
            for artifact in artifacts:
                summary = _one_line(
                    getattr(artifact, "title", "") or getattr(artifact, "task_id", ""), limit=100,
                )
                rows.append(
                    f'- {artifact.id}.md · {_one_line(getattr(artifact, "status", "unknown"), limit=40)} '
                    f'· {summary} → read_file("{self.canonical_mount}/{artifact.id}.md")'
                )
            if not artifacts:
                rows.append("(none yet)")
            gaps = tuple(getattr(listing, "gaps", ()) or ())
            if gaps:
                rows += ["", "## Unreadable canonical records (evidence gaps)"]
                rows.extend(
                    f"- {_one_line(getattr(gap, 'artifact_id', 'unknown'))} · "
                    f"{_one_line(str(getattr(gap, 'reason', 'unreadable')).split(':', 1)[0])}"
                    for gap in gaps[:8]
                )
                if len(gaps) > 8:
                    rows.append(f"- … {len(gaps) - 8} more unreadable record(s)")
            return "\n".join(rows)
        if "/" in rel:
            identity, _suffix = self._nested_identity(rel)
            self._get(identity)  # exact kind check before delegating to the physical-era adapter
            reader = getattr(self.provider, "read_file", None)
            if not callable(reader):
                raise ContextNotFoundError(f"no such artifact document: {rel}")
            rendered = str(reader(f"artifacts/{rel}"))
            if rendered.startswith(f"artifacts/{rel}:") and "no such" in rendered.casefold():
                raise ContextNotFoundError(f"no such artifact document: {rel}")
            return rendered
        return self._read_artifact(self._get(self._identity(rel)))

    def list_files(self, path: str = "") -> tuple[str, ...]:
        rel = _normalize_provider_path(path)
        if not rel:
            return ("index.md", *(f"{artifact.id}.md" for artifact in self._artifacts()))
        identity = rel.split("/", 1)[0]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", identity):
            raise ContextNotFoundError(f"{rel} is not an artifact directory")
        self._get(identity)
        listing = getattr(self.provider, "listing", None)
        if not callable(listing):
            raise ContextNotFoundError(f"{rel} is not an artifact directory")
        rows = tuple(
            row.strip() for row in str(listing(f"artifacts/{rel}")).splitlines()
            if row.strip()
        )
        if rows and rows[0].startswith(f"artifacts/{rel}:"):
            raise ContextNotFoundError(f"{rel} is not an artifact directory")
        return rows

    def grep_matches(self, pattern: Pattern[str], path: str = "") -> tuple[ContextMatch, ...]:
        rel = _normalize_provider_path(path)
        names = (rel,) if rel else self.list_files("")
        hits: list[ContextMatch] = []
        for name in names:
            try:
                body = self.read_file(name)
            except ContextNotFoundError:
                continue
            for line_number, line in enumerate(body.splitlines(), 1):
                if pattern.search(line):
                    hits.append(ContextMatch(name, line_number, line))
        return tuple(hits)


class ArtifactHistoryProvider:
    """Hippocampal index over canonical turn artifacts, independent of the legacy episode mirror."""

    def __init__(self, provider: Any, *, current_session: str):
        self.provider = provider
        self.current_session = str(current_session)

    def _turns(self) -> tuple[Any, ...]:
        getter = getattr(self.provider, "_artifacts", None)
        if not callable(getter):
            raise TypeError("history provider cannot enumerate canonical artifacts")
        return tuple(item for item in getter() if str(getattr(item, "kind", "")) == "turn")

    def _gaps(self) -> tuple[Any, ...]:
        getter = getattr(self.provider, "_artifacts", None)
        listing = getter() if callable(getter) else ()
        return tuple(getattr(listing, "gaps", ()) or ())

    @staticmethod
    def _session_key(session_id: str) -> str:
        value = str(session_id or "")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
            return value
        import hashlib
        return "session-" + hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]

    def _sessions(self, *, include_current: bool = True) -> dict[str, tuple[str, tuple[Any, ...]]]:
        grouped: dict[str, list[Any]] = {}
        originals: dict[str, str] = {}
        for artifact in self._turns():
            original = str(getattr(artifact, "session_id", "") or "unknown-session")
            if not include_current and original == self.current_session:
                continue
            key = self._session_key(original)
            if key in originals and originals[key] != original:
                import hashlib
                key += "-" + hashlib.sha256(original.encode()).hexdigest()[:8]
            originals[key] = original
            grouped.setdefault(key, []).append(artifact)
        return {key: (originals[key], tuple(items)) for key, items in grouped.items()}

    def _current(self) -> tuple[Any, ...]:
        return tuple(
            artifact for artifact in self._turns()
            if str(getattr(artifact, "session_id", "")) == self.current_session
        )

    def _render_artifact(self, artifact: Any) -> str:
        renderer = getattr(self.provider, "_render", None)
        if callable(renderer):
            return str(renderer(artifact))
        return str(self.provider.read_file(f"artifacts/{artifact.id}.md"))

    def _root_index(self) -> str:
        current = self._current()
        # Current-session seals already have the compact turn-N locators above.
        # Do not advertise the same bytes a second time under sessions/; the
        # artifact-ID spelling remains directly readable as a compatibility path.
        sessions = self._sessions(include_current=False)
        lines = [
            "# HISTORY / HIPPOCAMPUS — exact canonical turn evidence",
            "History establishes what was recorded in past turns; re-observe the world for current state.",
            "", "## Current application session",
        ]
        lines.extend(
            f'- turn-{index}.md · {getattr(artifact, "status", "unknown")} · '
            f'{_one_line(getattr(artifact, "title", "") or getattr(artifact, "task_id", ""), limit=100)} '
            f'→ read_file("@sliceagent/history/turn-{index}.md")'
            for index, artifact in enumerate(current, 1)
        )
        if not current:
            lines.append("(no sealed turns in this workspace for the current session yet)")
        lines += ["", "## Prior sessions retained in this workspace"]
        lines.extend(
            f'- {key}/ · {len(items)} turn artifact(s) '
            f'→ read_file("@sliceagent/history/sessions/{key}/index.md")'
            for key, (_original, items) in sessions.items()
        )
        if not sessions:
            lines.append("(none yet)")
        gaps = self._gaps()
        if gaps:
            lines += ["", "## Unreadable canonical records (history gaps)"]
            lines.extend(
                f"- {_one_line(getattr(gap, 'artifact_id', 'unknown'))} · unavailable"
                for gap in gaps[:8]
            )
        return "\n".join(lines)

    @staticmethod
    def _search_doc() -> str:
        return (
            "# HISTORY SEARCH\n"
            "Use grep with path=@sliceagent/history for exact canonical history retained in the current workspace. "
            "The compatibility search_history tool may expose legacy cross-session previews; open a canonical "
            "turn document before relying on exact wording. Cross-project knowledge recall remains disabled by default."
        )

    def _session_index(self, key: str, original: str, artifacts: tuple[Any, ...]) -> str:
        lines = [f"# HISTORY SESSION {key}", f"- recorded session id: {_one_line(original)}"]
        lines.extend(
            f'- {artifact.id}.md · {getattr(artifact, "status", "unknown")} · '
            f'{_one_line(getattr(artifact, "title", "") or getattr(artifact, "task_id", ""), limit=100)} '
            f'→ read_file("@sliceagent/history/sessions/{key}/{artifact.id}.md")'
            for artifact in artifacts
        )
        if not artifacts:
            lines.append("(none)")
        return "\n".join(lines)

    def read_file(self, path: str) -> str:
        rel = _normalize_provider_path(path)
        if rel in ("", "index.md"):
            return self._root_index()
        if rel == "search.md":
            return self._search_doc()
        match = re.fullmatch(r"turn-(\d+)\.md", rel)
        if match:
            index = int(match.group(1))
            current = self._current()
            if 1 <= index <= len(current):
                return self._render_artifact(current[index - 1])
            raise ContextNotFoundError(f"no canonical current-session turn {index}")
        if rel in {"sessions", "sessions/index.md"}:
            lines = ["# PRIOR WORKSPACE HISTORY SESSIONS"]
            lines.extend(f"- {key}/" for key in self._sessions(include_current=False))
            return "\n".join(lines + (["(none)"] if len(lines) == 1 else []))
        parts = rel.split("/")
        if len(parts) >= 2 and parts[0] == "sessions":
            key = parts[1]
            selected = self._sessions().get(key)
            if selected is None:
                raise ContextNotFoundError(f"no retained history session {key!r}")
            original, artifacts = selected
            if len(parts) == 2 or (len(parts) == 3 and parts[2] == "index.md"):
                return self._session_index(key, original, artifacts)
            if len(parts) == 3 and parts[2].endswith(".md"):
                identity = parts[2][:-3]
                artifact = next((item for item in artifacts if item.id == identity), None)
                if artifact is not None:
                    return self._render_artifact(artifact)
        raise ContextNotFoundError(f"no such canonical history document: {rel}")

    def list_files(self, path: str = "") -> tuple[str, ...]:
        rel = _normalize_provider_path(path)
        if not rel:
            return (
                "index.md", "search.md", "sessions/",
                *(f"turn-{index}.md" for index, _artifact in enumerate(self._current(), 1)),
            )
        if rel == "sessions":
            return ("index.md", *(f"{key}/" for key in self._sessions(include_current=False)))
        if rel.startswith("sessions/") and "/" not in rel[len("sessions/"):]:
            key = rel.split("/", 1)[1]
            selected = self._sessions().get(key)
            if selected is not None:
                return ("index.md", *(f"{item.id}.md" for item in selected[1]))
        raise ContextNotFoundError(f"no such canonical history directory: {rel}")

    def grep_matches(self, pattern: Pattern[str], path: str = "") -> tuple[ContextMatch, ...]:
        return tuple(_walk_provider(self, pattern, path))


def _listing_entries(raw: Sequence[str] | str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    rows = raw.splitlines() if isinstance(raw, str) else raw
    entries = []
    for row in rows:
        value = str(row or "").strip().replace("\\", "/")
        if not value or value.startswith(("(", "[")) or value.lower().startswith(("error:", "read ")):
            continue
        if value.startswith(CONTEXT_ROOT) or value.startswith("/"):
            continue
        trailing = value.endswith("/")
        value = value.strip("/")
        try:
            value = _normalize_provider_path(value)
        except ContextPathError:
            continue
        if not value:
            continue
        first, separator, _rest = value.partition("/")
        entry = first + ("/" if trailing or separator else "")
        if entry not in entries:
            entries.append(entry)
    return tuple(entries)


class LegacyMountProvider:
    """Adapter for existing HistoryFS/CoreArtifactFS/RosterFS-style providers.

    Those objects accept paths beginning with their historical mount and expose
    ``listing`` plus ripgrep-shaped text.  The adapter strips that physical-era
    namespace so ContextFS can re-prefix every result with the canonical locator.
    """

    def __init__(self, provider: Any, source_mount: str, *, canonical_mount: str = ""):
        if not callable(getattr(provider, "read_file", None)):
            raise TypeError("legacy context provider must expose read_file")
        self.provider = provider
        self.source_mount = _normalize_provider_path(source_mount)
        self.canonical_mount = (
            normalize_context_path(canonical_mount).rstrip("/") if canonical_mount else ""
        )
        if not self.source_mount:
            raise ValueError("source_mount must be non-empty")

    def _source(self, relative: str) -> str:
        rel = _normalize_provider_path(relative)
        return self.source_mount + (("/" + rel) if rel else "")

    def read_file(self, path: str) -> str:
        return self._rewrite(str(self.provider.read_file(self._source(path))))

    def _rewrite(self, value: str) -> str:
        if not self.canonical_mount:
            return value
        # Rewrite only executable-looking read locators. A historical document may legitimately discuss a
        # project path named history/ or roster/; global text replacement would change its evidence bytes.
        value = value.replace(
            f'read_file("{self.source_mount}/', f'read_file("{self.canonical_mount}/',
        )
        return value.replace(
            f"read_file('{self.source_mount}/", f"read_file('{self.canonical_mount}/",
        )

    def list_files(self, path: str = "") -> tuple[str, ...]:
        method = getattr(self.provider, "list_files", None) or getattr(self.provider, "listing", None)
        if not callable(method):
            raise TypeError("legacy context provider does not expose list_files/listing")
        return _listing_entries(method(self._source(path)))

    def grep_matches(self, pattern: Pattern[str], path: str = "") -> tuple[ContextMatch, ...]:
        method = getattr(self.provider, "grep", None)
        if not callable(method):
            return tuple(_walk_provider(self, pattern, path))
        raw = str(method(
            pattern.pattern, path=self._source(path), output_mode="content",
            context=0, offset=0, limit=_MAX_PROVIDER_GREP,
        ))
        hits = []
        prefix = self.source_mount + "/"
        for row in raw.splitlines():
            match = re.match(r"^(.+?):(\d+):(.*)$", row)
            if not match:
                continue
            name = match.group(1).replace("\\", "/")
            if name == self.source_mount:
                name = "index.md"
            elif name.startswith(prefix):
                name = name[len(prefix):]
            else:
                continue
            try:
                name = _normalize_provider_path(name)
            except ContextPathError:
                continue
            hits.append(ContextMatch(name, int(match.group(2)), self._rewrite(match.group(3))))
        return tuple(hits)


def _walk_provider(
    provider: ContextProvider, pattern: Pattern[str], path: str,
) -> Iterable[ContextMatch]:
    """Bounded grep fallback for a provider that implements only read/list."""

    start = _normalize_provider_path(path)
    pending = [start]
    seen_dirs: set[str] = set()
    documents = 0
    while pending and documents < _MAX_PROVIDER_WALK:
        current = pending.pop()
        if current in seen_dirs:
            continue
        seen_dirs.add(current)
        try:
            list_method = getattr(provider, "list_files", None) or getattr(provider, "listing", None)
            entries = _listing_entries(list_method(current)) if callable(list_method) else ()
        except (ContextNotFoundError, FileNotFoundError, KeyError):
            entries = ()
        if entries:
            for entry in reversed(entries):
                child = f"{current}/{entry.rstrip('/')}".strip("/")
                if entry.endswith("/"):
                    pending.append(child)
                    continue
                try:
                    text = str(provider.read_file(child))
                except (ContextNotFoundError, FileNotFoundError, KeyError):
                    continue
                documents += 1
                for line_number, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        yield ContextMatch(child, line_number, line)
            continue
        try:
            text = str(provider.read_file(current))
        except (ContextNotFoundError, FileNotFoundError, KeyError):
            continue
        documents += 1
        name = current or "index.md"
        for line_number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                yield ContextMatch(name, line_number, line)


class ContextFS:
    """Always-on, read-only virtual filesystem rooted at ``@sliceagent/``."""

    def __init__(
        self,
        providers: Mapping[str, ContextProvider] | None = None,
        *,
        status: ContextStatus | Mapping[str, Any]
        | Callable[[], ContextStatus | Mapping[str, Any]] | None = None,
    ):
        self._lock = threading.RLock()
        self._providers: OrderedDict[str, ContextProvider] = OrderedDict()
        self._status_source = status
        for mount, provider in (providers or {}).items():
            self.mount(mount, provider)

    @staticmethod
    def is_path(path: os.PathLike[str] | str) -> bool:
        return is_context_path(path)

    @property
    def provider_mounts(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._providers)

    def mount(self, mount: str, provider: ContextProvider) -> None:
        """Inject or atomically replace a provider at a canonical directory mount."""

        if str(mount).strip().replace("\\", "/").startswith(CONTEXT_ROOT):
            rel = _relative_context_path(str(mount))
        else:
            rel = _normalize_provider_path(str(mount))
        if not rel or rel.split("/", 1)[0] not in CONTEXT_REGIONS:
            raise ContextPathError(
                "provider mounts must be under evidence, history, work, memory, or roster",
            )
        if rel in _CANONICAL_FILES:
            raise ContextPathError("providers must mount at a context directory, not a canonical file")
        if not callable(getattr(provider, "read_file", None)):
            raise TypeError("context providers must expose read_file(path)")
        if not (callable(getattr(provider, "list_files", None))
                or callable(getattr(provider, "listing", None))):
            raise TypeError("context providers must expose list_files(path) or listing(path)")
        with self._lock:
            self._providers[rel] = provider

    def unmount(self, mount: str) -> None:
        rel = (_relative_context_path(mount) if str(mount).startswith(CONTEXT_ROOT)
               else _normalize_provider_path(mount))
        with self._lock:
            self._providers.pop(rel, None)

    def set_status_provider(
        self,
        status: ContextStatus | Mapping[str, Any]
        | Callable[[], ContextStatus | Mapping[str, Any]] | None,
    ) -> None:
        """Replace the live manifest source without rebuilding provider mounts."""

        with self._lock:
            self._status_source = status

    def _provider_snapshot(self) -> tuple[tuple[str, ContextProvider], ...]:
        with self._lock:
            return tuple(self._providers.items())

    def _resolve_provider(self, relative: str) -> tuple[str, ContextProvider, str] | None:
        matches = []
        for mount, provider in self._provider_snapshot():
            if relative == mount:
                matches.append((mount, provider, ""))
            elif relative.startswith(mount + "/"):
                matches.append((mount, provider, relative[len(mount) + 1:]))
        return max(matches, key=lambda item: len(item[0])) if matches else None

    def _status(self) -> tuple[ContextStatus, str]:
        with self._lock:
            source = self._status_source
        if source is None:
            return ContextStatus(), ""
        try:
            raw = source() if callable(source) else source
            if isinstance(raw, ContextStatus):
                return raw, ""
            if isinstance(raw, Mapping):
                return ContextStatus.from_mapping(raw), ""
            raise TypeError("status provider returned an unsupported value")
        except Exception as exc:  # status must never take down the permanent manifest
            return ContextStatus(), type(exc).__name__

    def _region_status(self, region: str, snapshot: ContextStatus) -> CapabilityStatus:
        mounted = any(
            mount == region or mount.startswith(region + "/")
            for mount, _provider in self._provider_snapshot()
        )
        if not mounted:
            return CapabilityStatus("unavailable", "no provider mounted")
        raw = snapshot.regions.get(region) if isinstance(snapshot.regions, Mapping) else None
        if raw is None:
            return CapabilityStatus("available", "provider mounted; content health not reported")
        return _coerce_capability(raw)

    @staticmethod
    def _status_text(status: CapabilityStatus) -> str:
        state = _one_line(status.state, limit=40)
        detail = _one_line(status.detail) if status.detail else ""
        count = f"; items: {status.item_count}" if status.item_count is not None else ""
        return state + ((f" — {detail}") if detail else "") + count

    def _render_manifest(self) -> str:
        snapshot, status_error = self._status()
        lines = [
            "# SLICEAGENT INTERNAL CONTEXT",
            "Permanent read-only context. Addressable from every workspace through ordinary read/list/grep tools.",
            "",
            "## Current focus",
            f"- project: {_one_line(snapshot.current_project)}",
            f"- workspace: {_one_line(snapshot.current_workspace)}",
            f"- logical request: {_one_line(snapshot.logical_request)}",
        ]
        if status_error:
            lines.append(f"- live status: unavailable ({status_error}); unreported fields remain unknown")
        evidence = self._status_text(self._region_status("evidence", snapshot))
        history = self._status_text(self._region_status("history", snapshot))
        work = self._status_text(self._region_status("work", snapshot))
        memory = self._status_text(self._region_status("memory", snapshot))
        roster = self._status_text(self._region_status("roster", snapshot))
        open_count = snapshot.open_active_work_count
        lines += [
            "", "## Memory model — exactly three layers",
            "Indexes, retrieval backends, roster, skills, and subagents are not additional memory layers.",
            f"- L0 · HISTORY AND EVIDENCE: history: {history}; evidence: {evidence}",
            f"- L1 · ACTIVE WORK: work: {work}",
            f"- L2 · TYPED KNOWLEDGE: memory: {memory}",
            "", "## L1 Active Work",
            f"- unresolved request roots: {open_count if open_count is not None else '(unknown)'}",
            "", "## L2 typed knowledge records (current scope)",
        ]
        counts = snapshot.knowledge_counts if isinstance(snapshot.knowledge_counts, Mapping) else {}
        lowered = {str(key).lower(): value for key, value in counts.items()}
        unique_count = _optional_count(lowered.get("unique"))
        lines.append(
            f"- unique records: {unique_count if unique_count is not None else '(unknown)'}"
        )
        for scope in ("user", "project", "craft"):
            count = _optional_count(lowered.get(scope))
            lines.append(f"- {scope.upper()} scope: {count if count is not None else '(unknown)'}")
        lines.append("- scope counts may overlap; never add them to infer a record total")
        lines += [
            "", "## Adjacent capabilities (not memory layers)",
            f"- roster: {roster}",
            "", "## Retrieval backends",
            f"- native index: {self._status_text(_coerce_capability(snapshot.native_index))}",
            f"- Memem: {self._status_text(_coerce_capability(snapshot.memem))}",
            "", "## Search policy",
            f"- cross-project search: {_one_line(snapshot.cross_project_search_policy)}",
            "", "## Self-inspection status",
            "For a general memory/context status question, read this one page next and then answer:",
            f'- memory status: read_file("{CONTEXT_ROOT}/memory/status.md")',
            "", "## Specific content drill-down",
            "Use only when the request asks for that content; this is not a traversal checklist.",
            f'- detailed memory inventory/diagnostics: read_file("{CONTEXT_ROOT}/memory/diagnostics.md")',
            f'- evidence: read_file("{CONTEXT_ROOT}/evidence/index.md")',
            f'- exact history: read_file("{CONTEXT_ROOT}/history/index.md")',
            f'- active work: read_file("{CONTEXT_ROOT}/work/active.md")',
            f'- knowledge: read_file("{CONTEXT_ROOT}/memory/index.md")',
            f'- roster: read_file("{CONTEXT_ROOT}/roster/index.md")',
        ]
        return "\n".join(lines)

    def _render_memory_status(self, snapshot: ContextStatus) -> str:
        """Render the bounded general answer; detailed inventory lives on a separate page."""
        counts = snapshot.knowledge_counts if isinstance(snapshot.knowledge_counts, Mapping) else {}
        lowered = {str(key).lower(): value for key, value in counts.items()}
        unique_count = _optional_count(lowered.get("unique"))
        open_count = snapshot.open_active_work_count
        transition = _coerce_lifecycle(
            snapshot.compatibility_transition, default_detail="not reported by host",
        )
        compatibility_health = _coerce_capability(snapshot.compatibility_health)
        retirement_gate = _coerce_lifecycle(
            snapshot.retirement_gate, default_detail="not reported by host",
        )
        consolidation = _coerce_lifecycle(
            snapshot.last_consolidation, default_detail="not reported by host",
        )
        history = self._status_text(self._region_status("history", snapshot))
        evidence = self._status_text(self._region_status("evidence", snapshot))
        work = self._status_text(self._region_status("work", snapshot))
        memory = self._status_text(self._region_status("memory", snapshot))
        lines = [
            "# MEMORY STATUS — GENERAL SUMMARY",
            "This is the complete answer surface for a general memory-system check. Summarize these bullets and "
            "stop; do not repeat the detailed compatibility inventory unless the user explicitly asks for it.",
            "In this request, `what can you see?` means memory visibility. Do not append a tour of filesystem, "
            "search, shell, command-execution, or other generic capabilities. Keep the answer to at most eight bullets.",
            "Counts are a sequential live snapshot, not an atomic census.",
            "",
            "## Exactly three memory layers",
            f"- L0 HISTORY / HIPPOCAMPUS: history: {history}; evidence: {evidence}; canonical layer-size "
            "total not reported",
            f"- L1 ACTIVE WORK / PFC: work: {work}; unresolved request roots: "
            f"{open_count if open_count is not None else '(unknown)'}; total unresolved work units not reported",
            f"- L2 TYPED KNOWLEDGE / NEOCORTEX: memory: {memory}; unique active current-scope records: "
            f"{unique_count if unique_count is not None else '(unknown)'}",
        ]
        for scope in ("user", "project", "craft"):
            count = _optional_count(lowered.get(scope))
            lines.append(
                f"  - {scope.upper()} scope memberships: "
                f"{count if count is not None else '(unknown)'}"
            )
        lines += [
            "  - scope memberships overlap; never add them as a record total",
            "  - USER=0 means no visible active USER-scoped typed record in this scope only; it does not prove "
            "that no preference exists elsewhere",
            "", "## Independent lifecycle facts",
            f"- compatibility layout (global): {self._status_text(transition)}",
            f"- compatibility mirror writes (current process): {self._status_text(compatibility_health)}",
            f"- compatibility retirement gate: {self._status_text(retirement_gate)}",
            f"- selective knowledge consolidation (current project): {self._status_text(consolidation)}",
            "- compatibility telemetry is not an L2 backlog; consolidation derives selected L2 knowledge while "
            "source evidence remains L0",
            "", "## Retrieval components (not layers or whole-system health)",
            f"- native index: {self._status_text(_coerce_capability(snapshot.native_index))}",
            f"- Memem: {self._status_text(_coerce_capability(snapshot.memem))}",
            "- overall memory-system health: not assessed by these component observations",
            "- answer boundary: report the measured layer, lifecycle, and component rows, then stop; do not append "
            "an `in short` or equivalent whole-system operational/health conclusion",
            "", "## Optional drill-down",
            "Read only for an explicit request for raw counts, inventory, or backend diagnostics:",
            f'- read_file("{CONTEXT_ROOT}/memory/diagnostics.md")',
        ]
        return "\n".join(lines)

    def _render_memory_diagnostics(self, snapshot: ContextStatus) -> str:
        counts = snapshot.knowledge_counts if isinstance(snapshot.knowledge_counts, Mapping) else {}
        lowered = {str(key).lower(): value for key, value in counts.items()}
        lines = [
            "# MEMORY DIAGNOSTICS",
            "Detailed host-counted telemetry for explicit inventory/backend questions. For a general memory-system "
            f"check, use `{CONTEXT_ROOT}/memory/status.md` and do not repeat this page's raw counts.",
            "Unknown stays unknown; do not sample private directories or implementation files to replace these "
            "aggregates. Counts are a sequential live snapshot, not an atomic census.",
            "",
            "## L2 typed knowledge (current scope)",
        ]
        unique_count = _optional_count(lowered.get("unique"))
        lines.append(
            f"- unique active records: {unique_count if unique_count is not None else '(unknown)'}"
        )
        for scope in ("user", "project", "craft"):
            count = _optional_count(lowered.get(scope))
            lines.append(
                f"- records carrying {scope.upper()} scope: "
                f"{count if count is not None else '(unknown)'}"
            )
        lines += [
            "- scope-axis counts can overlap; do not add them",
            "- a low count means only that few active typed records are visible in this scope; it does not "
            "prove that context is missing or waiting to be migrated",
            "", "## Legacy compatibility telemetry (not layer sizes or an L2 backlog)",
        ]
        inventory = snapshot.legacy_inventory if isinstance(snapshot.legacy_inventory, Mapping) else {}
        lines.append(
            f"- inventory status: "
            f"{self._status_text(_coerce_capability(snapshot.legacy_inventory_status))}"
        )
        if inventory:
            for name, raw_count in sorted(inventory.items(), key=lambda item: str(item[0]).lower()):
                count = _optional_count(raw_count)
                label = _one_line(str(name).replace("_", " "), limit=80)
                lines.append(f"- {label}: {count if count is not None else '(unknown)'}")
        else:
            lines.append("- inventory: (unknown — not reported by host)")
        lines.append(f"- scope: {_one_line(snapshot.legacy_inventory_scope)}")
        transition = _coerce_lifecycle(
            snapshot.compatibility_transition, default_detail="not reported by host",
        )
        compatibility_health = _coerce_capability(snapshot.compatibility_health)
        retirement_gate = _coerce_lifecycle(
            snapshot.retirement_gate, default_detail="not reported by host",
        )
        consolidation = _coerce_lifecycle(
            snapshot.last_consolidation, default_detail="no run metadata reported by host",
        )
        lines += [
            "- classification: episodes are L0 compatibility; tasks and sessions are L1 compatibility; "
            "roster and subagent records are adjacent operational state",
            "- interpretation: these are heterogeneous, potentially overlapping file/index aggregates; never "
            "add or compare them as layer sizes, unique memories, pending migrations, or consolidation inputs",
            "- search rows may mix turn and child-discovery entries; roster profile files count stored files, "
            "not validated available specialists",
            "", "## Compatibility-layout transition (not L2 knowledge migration)",
            f"- status: {self._status_text(transition)}",
            f"- current-process mirror write health: {self._status_text(compatibility_health)}",
            "- meaning: this tracks versioned retirement or reclassification of legacy physical formats; "
            "L0 evidence stays L0 and L1 projections stay L1",
            "", "## Compatibility retirement gate",
            f"- status: {self._status_text(retirement_gate)}",
            "- deletion is never automatic; retirement requires explicit L0/L1/L2 equivalence, legacy "
            "read-fallback, and compatibility-write health proofs",
            "", "## Knowledge consolidation (selective derivation, not migration)",
            f"- last project run: {self._status_text(consolidation)}",
            "- meaning: a run may derive provenance-linked L2 knowledge from eligible L0 evidence while the "
            "source remains L0; absent run metadata does not prove a backlog, need, or eligibility",
        ]
        if isinstance(snapshot.last_consolidation, Mapping):
            attempted_at = _optional_text(snapshot.last_consolidation.get("attempted_at"))
            source_count = _optional_count(snapshot.last_consolidation.get("source_episode_count"))
            mode = _optional_text(snapshot.last_consolidation.get("mode"))
            if attempted_at:
                lines.append(f"- consolidation attempted at: {_one_line(attempted_at, limit=80)}")
            if source_count is not None:
                lines.append(f"- source episodes considered: {source_count}")
            if mode:
                lines.append(f"- consolidation mode: {_one_line(mode, limit=80)}")
            result_counts = []
            for name in ("lessons", "skills", "skills_rejected", "errors"):
                count = _optional_count(snapshot.last_consolidation.get(name))
                if count is not None:
                    result_counts.append(f"{name.replace('_', ' ')}={count}")
            if result_counts:
                lines.append("- consolidation outputs: " + ", ".join(result_counts))
        if isinstance(snapshot.compatibility_health, Mapping):
            channels = snapshot.compatibility_health.get("channels")
            if isinstance(channels, Mapping):
                lines += ["", "## Compatibility writer channels"]
                for name, raw in sorted(channels.items(), key=lambda item: str(item[0])):
                    item = raw if isinstance(raw, Mapping) else {}
                    attempts = _optional_count(item.get("attempts"))
                    failures = _optional_count(item.get("failed"))
                    lines.append(
                        f"- {_one_line(str(name), limit=60)}: attempts="
                        f"{attempts if attempts is not None else '(unknown)'}, failed="
                        f"{failures if failures is not None else '(unknown)'}, state="
                        f"{_one_line(item.get('state'), limit=40)}"
                    )
        if isinstance(snapshot.retirement_gate, Mapping):
            gates = snapshot.retirement_gate.get("gates")
            if isinstance(gates, Mapping):
                lines += ["", "## Retirement proof gates"]
                lines.extend(
                    f"- {_one_line(str(name), limit=70)}: {_one_line(value, limit=40)}"
                    for name, value in sorted(gates.items(), key=lambda item: str(item[0]))
                )
        lines += [
            "", "## Retrieval backends (not memory layers)",
            f"- native index: {self._status_text(_coerce_capability(snapshot.native_index))}",
            f"- Memem: {self._status_text(_coerce_capability(snapshot.memem))}",
            "- backend health is component-local and does not prove the whole memory system is fully functional",
            f"- cross-project search: {_one_line(snapshot.cross_project_search_policy)}",
        ]
        return "\n".join(lines)

    def _render_fallback(self, relative: str, *, provider_error: str = "") -> str:
        snapshot, status_error = self._status()
        region = relative.split("/", 1)[0]
        status = self._region_status(region, snapshot)
        if provider_error:
            status = CapabilityStatus("degraded", f"provider read failed ({provider_error})")
        title = (relative.rsplit("/", 1)[-1] or region).replace(".md", "").replace("-", " ").upper()
        if relative in _CONTEXTFS_OWNED_FILES:
            body = (
                self._render_memory_status(snapshot)
                if relative == "memory/status.md"
                else self._render_memory_diagnostics(snapshot)
            )
            if provider_error:
                body += f"\n- context provider: degraded — read failed ({provider_error})"
        else:
            has_region_provider = any(
                mount == region or mount.startswith(region + "/")
                for mount, _provider in self._provider_snapshot()
            )
            if self._resolve_provider(relative):
                availability = "- This canonical context surface is not currently exposed by its provider."
            elif has_region_provider:
                availability = "- This canonical context surface is not currently exposed by the mounted region providers."
            else:
                availability = "- This canonical context surface is unavailable because no provider is mounted."
            body = "\n".join([
                f"# {title or region.upper()}",
                f"- locator: {CONTEXT_ROOT}/{relative}",
                f"- status: {self._status_text(status)}",
                availability,
            ])
        if status_error:
            body += f"\n- live status unavailable ({status_error}); unreported fields remain unknown"
        if relative in _CANONICAL_DIRS:
            children = _CANONICAL_CHILDREN.get(relative, ())
            body += "\n\n## Entries\n" + ("\n".join(f"- {entry}" for entry in children) if children else "(empty)")
        return body

    @staticmethod
    def _known_or_dynamic(relative: str) -> bool:
        if relative in _CANONICAL_DIRS or relative in _CANONICAL_FILES:
            return True
        return any(relative.startswith(directory + "/") for directory in _DYNAMIC_DIRS)

    def read_file(self, path: os.PathLike[str] | str) -> str:
        relative = _relative_context_path(path)
        if relative in ("", "index.md"):
            return self._render_manifest()
        # These status surfaces are owned by ContextFS, not by any knowledge provider. Keeping them synthetic
        # makes the live host report canonical and prevents stale optional-backend documents from shadowing it.
        if relative in _CONTEXTFS_OWNED_FILES:
            return self._render_fallback(relative)
        region = relative.split("/", 1)[0]
        if region not in CONTEXT_REGIONS:
            raise ContextNotFoundError(
                f"{normalize_context_path(path)}: no such internal context region; read {CONTEXTFS_SCHEMA_MARKER}",
            )
        resolved = self._resolve_provider(relative)
        if resolved is not None:
            _mount, provider, provider_path = resolved
            try:
                return str(provider.read_file(provider_path))
            except (ContextNotFoundError, FileNotFoundError, KeyError):
                if not self._known_or_dynamic(relative) or relative not in _CANONICAL_FILES | _CANONICAL_DIRS:
                    raise ContextNotFoundError(
                        f"{normalize_context_path(path)}: no such internal context document",
                    ) from None
            except Exception as exc:  # a failed optional backend must not remove the canonical surface
                return self._render_fallback(relative, provider_error=type(exc).__name__)
        if self._known_or_dynamic(relative):
            return self._render_fallback(relative)
        # A parent provider may not exist, but a provider mounted below makes the parent a real directory.
        if any(mount.startswith(relative + "/") for mount, _provider in self._provider_snapshot()):
            return self._render_fallback(relative)
        raise ContextNotFoundError(
            f"{normalize_context_path(path)}: no such internal context document; read {CONTEXTFS_SCHEMA_MARKER}",
        )

    def _mount_children(self, relative: str) -> tuple[str, ...]:
        prefix = relative + "/" if relative else ""
        entries = []
        for mount, _provider in self._provider_snapshot():
            if not mount.startswith(prefix) or mount == relative:
                continue
            tail = mount[len(prefix):]
            first = tail.split("/", 1)[0] + "/"
            if first not in entries:
                entries.append(first)
        return tuple(entries)

    def list_files(self, path: os.PathLike[str] | str = CONTEXT_ROOT) -> str:
        relative = _relative_context_path(path)
        if relative in _CANONICAL_FILES or (relative.endswith(".md") and self._resolve_provider(relative)):
            raise ContextNotFoundError(f"{normalize_context_path(path)} is a document, not a directory")
        region = relative.split("/", 1)[0] if relative else ""
        if region and region not in CONTEXT_REGIONS:
            raise ContextNotFoundError(f"{normalize_context_path(path)}: no such internal context directory")
        entries = list(_CANONICAL_CHILDREN.get(relative, ()))
        entries.extend(entry for entry in self._mount_children(relative) if entry not in entries)
        resolved = self._resolve_provider(relative) if relative else None
        provider_error = ""
        if resolved is not None:
            _mount, provider, provider_path = resolved
            method = getattr(provider, "list_files", None) or getattr(provider, "listing", None)
            try:
                dynamic = _listing_entries(method(provider_path)) if callable(method) else ()
                entries.extend(entry for entry in dynamic if entry not in entries)
            except (ContextNotFoundError, FileNotFoundError, KeyError):
                if not self._known_or_dynamic(relative):
                    raise ContextNotFoundError(
                        f"{normalize_context_path(path)}: no such internal context directory",
                    ) from None
            except Exception as exc:
                provider_error = type(exc).__name__
        if not entries and not self._known_or_dynamic(relative) and not self._mount_children(relative):
            raise ContextNotFoundError(f"{normalize_context_path(path)}: no such internal context directory")
        body = "\n".join(entries) if entries else "(empty)"
        if provider_error:
            body += f"\n(provider listing unavailable: {provider_error})"
        return body

    # Existing virtual stores call this operation ``listing``.  Keeping the alias
    # makes root integration mechanical while ``list_files`` remains canonical.
    def listing(self, path: os.PathLike[str] | str = CONTEXT_ROOT) -> str:
        return self.list_files(path)

    @staticmethod
    def _coerce_match(value: Any) -> ContextMatch | None:
        if isinstance(value, ContextMatch):
            raw_path, number, line = value.path, value.line_number, value.line
        elif isinstance(value, Mapping):
            raw_path = value.get("path", "")
            number = value.get("line_number", value.get("line_no", 0))
            line = value.get("line", value.get("text", ""))
        elif isinstance(value, (tuple, list)) and len(value) >= 3:
            raw_path, number, line = value[:3]
        else:
            return None
        try:
            path = _normalize_provider_path(str(raw_path)) or "index.md"
            line_number = int(number)
        except (ContextPathError, TypeError, ValueError):
            return None
        if line_number < 1:
            return None
        return ContextMatch(path, line_number, str(line))

    def _provider_matches(
        self, provider: ContextProvider, pattern: Pattern[str], provider_path: str,
    ) -> Iterable[ContextMatch]:
        method = getattr(provider, "grep_matches", None)
        values = method(pattern, provider_path) if callable(method) else _walk_provider(
            provider, pattern, provider_path,
        )
        count = 0
        for value in values:
            match = self._coerce_match(value)
            if match is not None:
                yield match
                count += 1
                if count >= _MAX_PROVIDER_GREP:
                    return

    @staticmethod
    def _within_scope(document: str, scope: str) -> bool:
        if not scope:
            return True
        if document == scope:
            return True
        return document.startswith(scope.rstrip("/") + "/")

    def grep(
        self, pattern: str, *, path: os.PathLike[str] | str = CONTEXT_ROOT,
        output_mode: str = "content", context: int = 0, offset: int = 0, limit: int = 50,
    ) -> str:
        del context  # accepted for parity with the ordinary grep tool; providers return matching lines
        try:
            matcher = re.compile(pattern)
        except re.error as exc:
            return f"grep: invalid regex ({exc})."
        if output_mode not in {"content", "files_with_matches", "count"}:
            raise ValueError("output_mode must be content, files_with_matches, or count")
        relative = _relative_context_path(path)
        region = relative.split("/", 1)[0] if relative else ""
        if region and region not in CONTEXT_REGIONS and relative != "index.md":
            raise ContextNotFoundError(f"{normalize_context_path(path)}: no such internal context path")
        if relative and not self._known_or_dynamic(relative) and not self._resolve_provider(relative) and not any(
            mount.startswith(relative + "/") for mount, _provider in self._provider_snapshot()
        ):
            raise ContextNotFoundError(f"{normalize_context_path(path)}: no such internal context path")

        found: list[tuple[str, int, str]] = []
        errors: list[str] = []
        # The permanent manifest is synthetic and must participate in grep like every other document.
        if relative in ("", "index.md"):
            for number, line in enumerate(self._render_manifest().splitlines(), 1):
                if matcher.search(line):
                    found.append((f"{CONTEXT_ROOT}/index.md", number, line))

        mounts = self._provider_snapshot()
        for mount, provider in mounts:
            # Status/diagnostic pages are owned by ContextFS rather than the mounted
            # memory provider. Exact reads and exact searches must therefore
            # have identical isolation from a stale or unavailable backend.
            if relative in _CONTEXTFS_OWNED_FILES:
                continue
            if relative == mount or relative.startswith(mount + "/"):
                provider_path = relative[len(mount):].lstrip("/")
            elif not relative or mount.startswith(relative.rstrip("/") + "/"):
                provider_path = ""
            else:
                continue
            try:
                for match in self._provider_matches(provider, matcher, provider_path):
                    document_rel = f"{mount}/{match.path}".strip("/")
                    if document_rel in _CONTEXTFS_OWNED_FILES:
                        continue
                    # A more-specific injected provider shadows a parent provider at the same locator.
                    owner = self._resolve_provider(document_rel)
                    if owner is None or owner[0] != mount or not self._within_scope(document_rel, relative):
                        continue
                    found.append((f"{CONTEXT_ROOT}/{document_rel}", match.line_number, match.line))
            except Exception as exc:
                errors.append(f"{mount}: {type(exc).__name__}")

        # Canonical files may be provider-owned *or* truthful synthesized fallbacks. Search their actual read
        # surface in either case. Looking only for a missing parent provider was subtly wrong: a mounted
        # ``memory`` provider can omit ``status.md`` (the normal CLI arrangement), in which case read_file
        # returns the synthesized live status while grep used to claim there was no match. Provider hits above
        # are deduplicated below, so reading this small fixed surface here is safe.
        for document_rel in sorted(_CANONICAL_FILES):
            if document_rel == "index.md" or not self._within_scope(document_rel, relative):
                continue
            try:
                text = self.read_file(f"{CONTEXT_ROOT}/{document_rel}")
            except ContextNotFoundError:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if matcher.search(line):
                    found.append((f"{CONTEXT_ROOT}/{document_rel}", number, line))

        unique: list[tuple[str, int, str]] = []
        seen: set[tuple[str, int, str]] = set()
        for hit in found:
            if hit not in seen:
                unique.append(hit)
                seen.add(hit)
        counts: OrderedDict[str, int] = OrderedDict()
        for name, _number, _line in unique:
            counts[name] = counts.get(name, 0) + 1
        if output_mode == "files_with_matches":
            rows = list(counts)
        elif output_mode == "count":
            rows = [f"{name}:{count}" for name, count in counts.items()]
        else:
            rows = [f"{name}:{number}:{line}" for name, number, line in unique]
        start = max(0, _optional_count(offset) or 0)
        requested_limit = _optional_count(limit)
        size = min(requested_limit if requested_limit is not None else 50, _MAX_PROVIDER_GREP)
        window = rows[start:start + size]
        if not window:
            if errors:
                return "grep: context search incomplete; provider unavailable (" + "; ".join(errors) + ")."
            if region and not any(
                mount == region or mount.startswith(region + "/") for mount, _provider in mounts
            ):
                return f"grep: {region} context is unavailable; no provider is mounted."
            return "grep: no matches found."
        body = "\n".join(window)
        if start + size < len(rows):
            body += f"\n\n[truncated; use offset={start + size} to see more]"
        if errors:
            body += "\n\n[incomplete; provider unavailable: " + "; ".join(errors) + "]"
        return body

    def deny_write(self, path: os.PathLike[str] | str, *, operation: str = "write") -> None:
        """Reject mutation after validating that the path really targets ContextFS."""

        canonical = normalize_context_path(path)
        raise ContextReadOnlyError(
            f"{canonical} is SliceAgent's read-only internal context; {operation} is not allowed",
        )

    def read_only_message(self, path: os.PathLike[str] | str, *, operation: str = "write") -> str:
        """Plain-language form for tool hosts whose handlers return errors as text."""

        canonical = normalize_context_path(path)
        return f"{canonical} is SliceAgent's read-only internal context; {operation} is not allowed"


__all__ = [
    "CONTEXT_ROOT", "CONTEXTFS_SCHEMA_MARKER", "CONTEXT_REGIONS",
    "CapabilityStatus", "ContextStatus", "ContextMatch", "ContextProvider",
    "ContextFS", "MappingContextProvider", "LedgerContextProvider", "ArtifactContextProvider",
    "ArtifactHistoryProvider",
    "LegacyMountProvider",
    "ContextFSError", "ContextPathError", "ContextNotFoundError", "ContextReadOnlyError",
    "is_context_path", "normalize_context_path", "schemas_advertise_contextfs",
]
