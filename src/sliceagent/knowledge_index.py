"""Knowledge retrieval index contracts and dependable implementations.

The canonical :mod:`sliceagent.knowledge` repository owns meaning, lifecycle,
scope, sensitivity, and provenance.  An optional Memem backend is a first-class
semantic *projection* of those records: it may rank candidates, but every hit is
resolved back through the repository before SliceAgent can use it.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any, Callable, Protocol, runtime_checkable

from .knowledge import (
    KnowledgeHit,
    KnowledgeQuery,
    KnowledgeRecord,
    KnowledgeRepository,
    KnowledgeSensitivity,
    KnowledgeStatus,
)


@runtime_checkable
class KnowledgeIndex(Protocol):
    """An index returns stable canonical records; it never owns their meaning."""

    @property
    def is_active(self) -> bool: ...

    def index(self, records: Iterable[KnowledgeRecord]) -> None: ...

    def remove(self, record_ids: Iterable[str]) -> None: ...

    def search(self, query: KnowledgeQuery) -> list[KnowledgeHit]: ...

    def rebuild(self, repository: KnowledgeRepository | None = None) -> None: ...

    def health(self) -> dict[str, Any]: ...


class NativeKnowledgeIndex:
    """SQLite FTS5 when available, deterministic scoped lexical search otherwise."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    @property
    def is_active(self) -> bool:
        # Lexical fallback is a real native index, so FTS5 absence does not make L2 unavailable.
        return True

    def index(self, records: Iterable[KnowledgeRecord]) -> None:
        self.repository.index_records(records)

    def remove(self, record_ids: Iterable[str]) -> None:
        self.repository.remove_from_index(record_ids)

    def search(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        return self.repository.search(query)

    def rebuild(self, repository: KnowledgeRepository | None = None) -> None:
        if repository is not None and repository is not self.repository:
            raise ValueError("NativeKnowledgeIndex cannot rebuild a different repository")
        self.repository.rebuild_index()

    def health(self) -> dict[str, Any]:
        return {
            "active": True,
            "backend": "sqlite-fts5" if self.repository.fts_enabled else "sqlite-lexical",
            "fts5": self.repository.fts_enabled,
            "indexed_records": self.repository.indexed_count(),
            # FTS setup failure is not a query failure: deterministic lexical
            # search remains the active native backend.
            "error": "",
            "warning": self.repository.fts_error,
        }


class MememKnowledgeIndex:
    """Semantic L2 projection backed by Memem, with canonical resolution.

    This is intentionally not the old ``memory_save`` mirror.  Stable external
    ids preserve SliceAgent record identity; Memem indexes a concise primary
    abstraction plus multiple cue anchors while retaining the full record as a
    pull-only value.  Hard scope filtering happens inside Memem *and* is checked
    again against the canonical record here.

    The native index is a fail-open availability fallback only.  It is not
    merged into successful Memem results, because doing so would re-index the
    full value and defeat harmonic representation's noise boundary.
    """

    _EXTERNAL_PREFIX = "sliceagent:"

    def __init__(
        self,
        repository: KnowledgeRepository,
        *,
        fallback: KnowledgeIndex | None = None,
        retrieve_fn: Callable[..., list[dict[str, Any]]] | None = None,
        upsert_fn: Callable[..., dict[str, Any]] | None = None,
        remove_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.repository = repository
        self.fallback = fallback or NativeKnowledgeIndex(repository)
        self._retrieve = retrieve_fn
        self._upsert = upsert_fn
        self._remove = remove_fn
        self._load_error = ""
        if self._retrieve is None or self._upsert is None or self._remove is None:
            try:
                from memem.operations import memory_index_remove, memory_index_upsert
                from memem.retrieve import retrieve

                self._retrieve = retrieve
                self._upsert = memory_index_upsert
                self._remove = memory_index_remove
            except Exception as exc:  # optional dependency or pre-protocol Memem
                self._load_error = type(exc).__name__
                self._retrieve = None
                self._upsert = None
                self._remove = None
        self._last_error = ""
        self._last_operation = "not-observed"
        self._projection_degraded = False
        self._projected_record_ids: set[str] = set()
        self._orphan_hits_dropped = 0

    @property
    def is_active(self) -> bool:
        return self._retrieve is not None and self._upsert is not None and self._remove is not None

    @classmethod
    def _external_id(cls, record_id: str) -> str:
        return cls._EXTERNAL_PREFIX + record_id

    @classmethod
    def _canonical_id(cls, external_id: object) -> str | None:
        value = str(external_id or "")
        if not value.startswith(cls._EXTERNAL_PREFIX):
            return None
        identity = value[len(cls._EXTERNAL_PREFIX):]
        return identity or None

    @staticmethod
    def _one_line(value: object, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]

    @classmethod
    def _primary_index(cls, record: KnowledgeRecord) -> str:
        explicit = record.metadata.get("primary_index")
        if explicit:
            return cls._one_line(explicit, 500)
        title = cls._one_line(record.metadata.get("title"), 180)
        body = cls._one_line(record.content, 320)
        # One sentence/line is the primary abstraction, not a second copy of a
        # long report.  Explicit metadata can override this deterministic seam
        # when a PFC/consolidator has a better abstraction.
        sentence = re.split(r"(?<=[.!?。！？])\s+", body, maxsplit=1)[0][:260]
        parts: list[str] = []
        for part in (title, sentence):
            if part and part.casefold() not in {item.casefold() for item in parts}:
                parts.append(part)
        return cls._one_line(" — ".join(parts) or record.applicability or record.id, 500)

    @classmethod
    def _cues(cls, record: KnowledgeRecord) -> list[str]:
        raw: list[object] = []
        for key in ("cues", "tags"):
            value = record.metadata.get(key)
            if isinstance(value, str):
                raw.extend(value.split(","))
            elif isinstance(value, (list, tuple)):
                raw.extend(value)
        paths = record.metadata.get("paths")
        if isinstance(paths, str):
            paths = (paths,)
        if isinstance(paths, (list, tuple)):
            for path in paths:
                text = cls._one_line(path, 240)
                if text:
                    raw.extend((text, os.path.basename(text)))
        if record.applicability:
            raw.append(record.applicability)
        cues: list[str] = []
        seen: set[str] = set()
        for value in raw:
            cue = cls._one_line(value, 60)
            key = cue.casefold()
            if not cue or key in seen:
                continue
            seen.add(key)
            cues.append(cue)
            if len(cues) >= 12:
                break
        return cues

    @staticmethod
    def _partition(record: KnowledgeRecord) -> str:
        if record.scopes.project_id is not None:
            return "sliceagent.project:" + record.scopes.project_id
        if record.scopes.user_id is not None:
            return "sliceagent.user:" + record.scopes.user_id
        # KnowledgeScope guarantees at least one axis.
        return "sliceagent.agent:" + str(record.scopes.agent_id)

    @staticmethod
    def _query_partitions(query: KnowledgeQuery) -> tuple[str, ...]:
        values = []
        if query.project_id is not None:
            values.append("sliceagent.project:" + query.project_id)
        if query.user_id is not None:
            values.append("sliceagent.user:" + query.user_id)
        if query.agent_id is not None:
            values.append("sliceagent.agent:" + query.agent_id)
        return tuple(values)

    @staticmethod
    def _matches(record: KnowledgeRecord, query: KnowledgeQuery) -> bool:
        if query.statuses and record.status not in query.statuses:
            return False
        if query.kinds and record.kind not in query.kinds:
            return False
        if not query.include_secret and record.sensitivity is KnowledgeSensitivity.SECRET:
            return False
        for axis in ("user_id", "project_id", "agent_id"):
            bound = getattr(record.scopes, axis)
            if bound is not None and bound != getattr(query, axis):
                return False
        return True

    def _project_record(self, record: KnowledgeRecord) -> None:
        if not self.is_active:
            return
        assert self._upsert is not None and self._remove is not None
        external_id = self._external_id(record.id)
        if record.status is not KnowledgeStatus.ACTIVE or record.sensitivity is KnowledgeSensitivity.SECRET:
            self._remove(external_id)
            return
        tags = [record.kind.value]
        metadata_tags = record.metadata.get("tags")
        if isinstance(metadata_tags, str):
            tags.extend(part.strip() for part in metadata_tags.split(",") if part.strip())
        elif isinstance(metadata_tags, (list, tuple)):
            tags.extend(str(part).strip() for part in metadata_tags if str(part).strip())
        paths = record.metadata.get("paths")
        if isinstance(paths, str):
            path_values = [paths]
        elif isinstance(paths, (list, tuple)):
            path_values = [str(path) for path in paths]
        else:
            path_values = []
        self._upsert(
            external_id,
            record.content,
            primary_index=self._primary_index(record),
            cues=self._cues(record),
            scope_id=self._partition(record),
            title=self._one_line(record.metadata.get("title"), 120),
            tags=",".join(dict.fromkeys(tags)),
            paths=path_values,
        )

    def index(self, records: Iterable[KnowledgeRecord]) -> None:
        values = tuple(records)
        # Keep the deterministic fallback projection current even if Memem is
        # absent or one semantic upsert fails.
        self.fallback.index(values)
        if not self.is_active:
            return
        errors: list[Exception] = []
        for record in values:
            try:
                self._project_record(record)
                if record.status is KnowledgeStatus.ACTIVE and record.sensitivity is not KnowledgeSensitivity.SECRET:
                    self._projected_record_ids.add(record.id)
                else:
                    self._projected_record_ids.discard(record.id)
            except Exception as exc:  # attempt every independent projection
                errors.append(exc)
        self._last_operation = "index"
        if errors:
            self._projection_degraded = True
            self._last_error = type(errors[0]).__name__
            raise errors[0]
        if not self._projection_degraded:
            self._last_error = ""

    def remove(self, record_ids: Iterable[str]) -> None:
        identities = tuple(record_ids)
        self.fallback.remove(identities)
        if not self.is_active:
            return
        assert self._remove is not None
        errors: list[Exception] = []
        for record_id in identities:
            try:
                self._remove(self._external_id(record_id))
                self._projected_record_ids.discard(record_id)
            except Exception as exc:
                errors.append(exc)
        self._last_operation = "remove"
        if errors:
            self._projection_degraded = True
            self._last_error = type(errors[0]).__name__
            raise errors[0]
        if not self._projection_degraded:
            self._last_error = ""

    def search(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        # Lifecycle/secret maintenance queries require the canonical index.
        if (
            not self.is_active
            or self._projection_degraded
            or not query.text.strip()
            or query.include_secret
            or query.statuses != (KnowledgeStatus.ACTIVE,)
        ):
            return self.fallback.search(query)
        partitions = self._query_partitions(query)
        if not partitions:
            return self.fallback.search(query)
        assert self._retrieve is not None
        by_id: dict[str, KnowledgeHit] = {}
        try:
            overfetch = min(100, max(12, query.limit * 3))
            for partition in partitions:
                hits = self._retrieve(
                    query.text,
                    k=overfetch,
                    log_call_type=None,
                    writeback=False,
                    scope_id=partition,
                    scope_mode="hard",
                    paths_context=list(query.paths_context) or None,
                )
                for hit in hits:
                    record_id = self._canonical_id(hit.get("external_id"))
                    record = self.repository.get(record_id) if record_id else None
                    if record is None:
                        self._orphan_hits_dropped += 1
                        continue
                    if not self._matches(record, query):
                        continue
                    score = float(hit.get("score") or 0.0)
                    snippet = str(
                        hit.get("primary_index")
                        or hit.get("title")
                        or record.content[:240]
                    )
                    candidate = KnowledgeHit(record=record, score=score, snippet=snippet)
                    previous = by_id.get(record.id)
                    if previous is None or candidate.score > previous.score:
                        by_id[record.id] = candidate
        except Exception as exc:
            # Availability fallback is explicit and whole-query: never mix a
            # partial Memem result set with native full-body ranking.
            self._last_error = type(exc).__name__
            self._last_operation = "search-fallback"
            return self.fallback.search(query)
        self._last_error = ""
        self._last_operation = "search"
        return sorted(by_id.values(), key=lambda hit: (-hit.score, hit.record.id))[:query.limit]

    def rebuild(self, repository: KnowledgeRepository | None = None) -> None:
        if repository is not None and repository is not self.repository:
            raise ValueError("MememKnowledgeIndex cannot rebuild a different repository")
        self.fallback.rebuild(self.repository)
        if not self.is_active:
            return
        records = self.repository.records_for_index_rebuild()
        errors: list[Exception] = []
        projected_ids: set[str] = set()
        for record in records:
            try:
                self._project_record(record)
                if record.status is KnowledgeStatus.ACTIVE and record.sensitivity is not KnowledgeSensitivity.SECRET:
                    projected_ids.add(record.id)
            except Exception as exc:
                errors.append(exc)
        self._last_operation = "rebuild"
        self._projected_record_ids = projected_ids
        if errors:
            self._projection_degraded = True
            self._last_error = type(errors[0]).__name__
            raise errors[0]
        self._projection_degraded = False
        self._last_error = ""

    def health(self) -> dict[str, Any]:
        state = "unavailable" if not self.is_active else (
            "degraded" if self._last_error or self._projection_degraded else "healthy"
        )
        return {
            "active": self.is_active,
            "available": self.is_active,
            "backend": "memem-external-index" if self.is_active else "none",
            "state": state,
            "canonical_store": "sliceagent-knowledge",
            "representation": "primary-index+cues; full value pull-only",
            "indexed_records_observed": len(self._projected_record_ids),
            "orphan_hits_dropped": self._orphan_hits_dropped,
            "last_operation": self._last_operation,
            "projection_degraded": self._projection_degraded,
            "error": self._last_error or self._load_error,
            "fallback": self.fallback.health(),
        }


class NullKnowledgeIndex:
    """Explicit optional-backend absence without disabling canonical L2 persistence."""

    @property
    def is_active(self) -> bool:
        return False

    def index(self, records: Iterable[KnowledgeRecord]) -> None:
        return None

    def remove(self, record_ids: Iterable[str]) -> None:
        return None

    def search(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        return []

    def rebuild(self, repository: KnowledgeRepository | None = None) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return {
            "active": False,
            "backend": "none",
            "fts5": False,
            "indexed_records": 0,
            "error": "",
        }
