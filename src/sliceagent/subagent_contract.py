"""Typed brief-down / artifact-up contract for delegated work.

The runtime's public ``spawn_agent(task=..., grants=...)`` surface remains compact,
but delegation no longer relies on two loosely-shaped dictionaries.  A child receives
only an explicit brief plus immutable artifact handles; its result records exactly what
was covered, what remains uncertain, and the dependency-scoped workspace bytes it saw.

The legacy subagent archive still stores dictionaries. ``to_record`` deliberately emits
the old keys alongside the typed fields so HistoryFS, roster careers, and existing vaults
remain readable during migration.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from .workspace_revision import PathRevision, WorkspaceRevision, fingerprint_path


CONTRACT_VERSION = 1
OBSERVATION_VERSION = 1
CLAIM_VERSION = 1
EXPLORER_EVIDENCE_VERSION = 1
DriftPolicy = Literal["report", "fail", "ignore"]
IntegrationPolicy = Literal["digest_ok", "report_required"]
SourceCoverageStatus = Literal[
    "source_complete", "source_partial", "source_unsupported", "not_assessed",
]
ExplorerEvidenceStatus = Literal[
    "not_assessed", "none", "navigation_only", "content_partial", "content_retained",
]
ReportCompletion = Literal["complete", "partial", "absent", "unknown"]
_EXPLORER_EVIDENCE_PATH_LIMIT = 32
_EXPLORER_EVIDENCE_COUNT_LIMIT = 1_000_000
_OBSERVATION_PREVIEW_COUNT_LIMIT = 20
_OBSERVATION_PREVIEW_BYTES_LIMIT = 24 * 1024
_LEGACY_SOURCE_COVERAGE = {
    "grounded": "source_complete",
    "partial": "source_partial",
    "unsupported": "source_unsupported",
}
# Immutable fan-in inputs have two compatible addresses. ``subagents/sub-N.md`` is the
# legacy per-session archive handle; ``artifacts/<id>.md`` is the canonical local-store
# handle returned by current child tool results. Mutable identity aliases never enter a
# typed brief.
_CANONICAL_REF = re.compile(
    r"^(?:subagents/sub-\d+|artifacts/[A-Za-z0-9][A-Za-z0-9._-]{0,159})\.md$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly-empty ' if empty else 'non-empty '}string")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain strings")
    return tuple(value)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def normalize_source_coverage_status(value: object) -> SourceCoverageStatus:
    """Read the current source-coverage vocabulary plus legacy v1 artifact values.

    Source coverage is deliberately mechanical: it says whether a synthesiser completely read and path-cited
    every granted report. It is never a claim-correctness or verification verdict.
    """
    status = str(value or "not_assessed").strip().casefold()
    status = _LEGACY_SOURCE_COVERAGE.get(status, status)
    if status not in {"source_complete", "source_partial", "source_unsupported", "not_assessed"}:
        raise ValueError("subagent artifact source_coverage_status is invalid")
    return status  # type: ignore[return-value]


@dataclass(frozen=True)
class SubagentObservation:
    """One persisted view returned by a determinate read-only child tool.

    In ``SubagentArtifact.observations`` this is the complete redacted tool result delivered to the child;
    ``truncated`` then means the source tool itself returned a page/partial view. The separate
    ``observation_preview`` may contain presentation-bounded copies for model context. Immutable legacy v1
    records whose view contains the old capsule-budget marker remain archive-partial and are rendered as such;
    their flag must not be reinterpreted as source paging.

    ``raw_sha256`` binds the exact view delivered to the child before persistence redaction;
    ``view_sha256`` binds the text that is actually retained. A redacted or source-truncated view is useful
    evidence for its visible bytes, but never proof about hidden or unobserved bytes. A failed row records
    attempted scope and a gap; it never supports source claims.
    """

    tool: str
    args: Mapping[str, Any]
    status: str
    view: str
    raw_sha256: str
    view_sha256: str
    raw_bytes: int
    view_bytes: int
    redacted: bool = False
    truncated: bool = False
    version: int = OBSERVATION_VERSION

    def __post_init__(self) -> None:
        _text(self.tool, "subagent observation tool")
        _text(self.status, "subagent observation status")
        if self.status not in {"succeeded", "failed", "steered", "cancelled"}:
            raise ValueError(
                "subagent observation status must be succeeded, failed, steered, or cancelled"
            )
        _text(self.view, "subagent observation view", empty=True)
        if not isinstance(self.args, Mapping):
            raise ValueError("subagent observation args must be an object")
        selected = {}
        for key, value in self.args.items():
            if not isinstance(key, str) or not key:
                raise ValueError("subagent observation arg names must be non-empty strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("subagent observation args may contain only JSON scalar values")
            selected[key] = value
        object.__setattr__(self, "args", MappingProxyType(selected))
        for name in ("raw_sha256", "view_sha256"):
            if not isinstance(getattr(self, name), str) or not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"subagent observation {name} must be a lowercase sha256")
        for name in ("raw_bytes", "view_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"subagent observation {name} must be non-negative")
        encoded = self.view.encode("utf-8")
        if self.view_bytes != len(encoded):
            raise ValueError("subagent observation view_bytes does not match retained view")
        if self.view_sha256 != hashlib.sha256(encoded).hexdigest():
            raise ValueError("subagent observation view_sha256 does not match retained view")
        if not isinstance(self.redacted, bool) or not isinstance(self.truncated, bool):
            raise ValueError("subagent observation redacted/truncated flags must be booleans")
        if self.version != OBSERVATION_VERSION:
            raise ValueError(f"unsupported subagent observation version: {self.version}")

    def to_dict(self) -> dict:
        return {
            "v": self.version,
            "tool": self.tool,
            "args": dict(self.args),
            "status": self.status,
            "view": self.view,
            "raw_sha256": self.raw_sha256,
            "view_sha256": self.view_sha256,
            "raw_bytes": self.raw_bytes,
            "view_bytes": self.view_bytes,
            "redacted": self.redacted,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentObservation":
        if not isinstance(value, Mapping):
            raise ValueError("subagent observation must be an object")
        view = str(value.get("view") or "")
        # On reload the retained `view` IS the content; its byte-count and digest are DERIVED from it, never
        # the separately-stored fields. A JSON round-trip, a re-bounding, or a checkpoint freeze/thaw can leave
        # the stored view_bytes/view_sha256 stale — and __post_init__ then rejected the whole observation,
        # turning a COMPLETED delegation "indeterminate" over a checksum on the agent's own sealed data. The
        # view is authority; recompute its checksum so a benign round-trip re-seals cleanly. raw_* describe the
        # pre-redaction output (not retained), so they stay trusted (they are not cross-checked here).
        encoded = view.encode("utf-8")
        view_sha256 = hashlib.sha256(encoded).hexdigest()
        return cls(
            tool=str(value.get("tool") or "unknown"),
            args=value.get("args") if isinstance(value.get("args"), Mapping) else {},
            status=str(value.get("status") or "unknown"),
            view=view,
            raw_sha256=str(value.get("raw_sha256") or view_sha256),
            view_sha256=view_sha256,
            raw_bytes=int(value.get("raw_bytes") if value.get("raw_bytes") is not None else len(encoded)),
            view_bytes=len(encoded),
            redacted=bool(value.get("redacted", False)),
            truncated=bool(value.get("truncated", False)),
            version=int(value.get("v") or OBSERVATION_VERSION),
        )


@dataclass(frozen=True)
class ExplorerEvidenceAccount:
    """Host-derived account of an explorer's physical workspace evidence.

    This records durable evidence retention, not task completion or claim correctness. Navigation means only
    ``list_files``/``glob`` discovery; content means ``read_file``/``grep``/``code_review`` output. A partial
    status now means the child received a source-paged view or a determinate observation failed—not that an
    inline prompt/digest projection was bounded. Counts remain exact while path samples are deliberately capped.
    """

    status: ExplorerEvidenceStatus = "not_assessed"
    scope_path_count: int = 0
    navigation_success_count: int = 0
    content_success_count: int = 0
    gap_observation_count: int = 0
    retained_navigation_view_count: int = 0
    retained_content_view_count: int = 0
    omitted_navigation_view_count: int = 0
    omitted_content_view_count: int = 0
    truncated_content_view_count: int = 0
    scope_paths: tuple[str, ...] = ()
    navigation_paths: tuple[str, ...] = ()
    content_paths: tuple[str, ...] = ()
    gap_paths: tuple[str, ...] = ()
    version: int = EXPLORER_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.status not in {
            "not_assessed", "none", "navigation_only", "content_partial", "content_retained",
        }:
            raise ValueError("explorer evidence status is invalid")
        count_names = (
            "scope_path_count", "navigation_success_count", "content_success_count",
            "gap_observation_count", "retained_navigation_view_count",
            "retained_content_view_count", "omitted_navigation_view_count",
            "omitted_content_view_count", "truncated_content_view_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if (not isinstance(value, int) or isinstance(value, bool) or value < 0
                    or value > _EXPLORER_EVIDENCE_COUNT_LIMIT):
                raise ValueError(f"explorer evidence {name} must be a bounded non-negative integer")
        for name in ("scope_paths", "navigation_paths", "content_paths", "gap_paths"):
            paths = _unique(_strings(getattr(self, name), f"explorer evidence {name}"))
            if len(paths) > _EXPLORER_EVIDENCE_PATH_LIMIT:
                raise ValueError(f"explorer evidence {name} exceeds its bounded path limit")
            if any(len(path.encode("utf-8")) > 400 for path in paths):
                raise ValueError(f"explorer evidence {name} contains an overlong path")
            object.__setattr__(self, name, paths)
        for count_name, paths_name in (
            ("scope_path_count", "scope_paths"),
            ("navigation_success_count", "navigation_paths"),
            ("content_success_count", "content_paths"),
            ("gap_observation_count", "gap_paths"),
        ):
            if getattr(self, count_name) < len(getattr(self, paths_name)):
                raise ValueError(f"explorer evidence {count_name} cannot be smaller than retained metadata")
        if self.retained_navigation_view_count + self.omitted_navigation_view_count \
                != self.navigation_success_count:
            raise ValueError("explorer navigation retained/omitted counts must close")
        if self.retained_content_view_count + self.omitted_content_view_count != self.content_success_count:
            raise ValueError("explorer content retained/omitted counts must close")
        if self.truncated_content_view_count > self.retained_content_view_count:
            raise ValueError("explorer truncated content count exceeds retained content views")
        if self.status == "none" and (self.navigation_success_count or self.content_success_count):
            raise ValueError("explorer evidence status none cannot contain successful observations")
        if self.status == "navigation_only" and (
                not self.navigation_success_count or self.content_success_count):
            raise ValueError("explorer navigation_only status requires navigation and no content")
        if self.status == "content_retained" and (
                not self.retained_content_view_count or self.omitted_content_view_count
                or self.truncated_content_view_count):
            # Legacy v1 writers described retention only and could therefore pair this status with a non-zero
            # gap count. Continue reading those immutable records; current writers conservatively emit
            # content_partial when a determinate inspection failed.
            raise ValueError("explorer content_retained status requires every content view retained whole")
        if self.status == "content_partial" and (
                not self.content_success_count
                or not (self.omitted_content_view_count or self.truncated_content_view_count
                        or not self.retained_content_view_count or self.gap_observation_count)):
            raise ValueError("explorer content_partial status requires source partialness or an observation gap")
        if self.version != EXPLORER_EVIDENCE_VERSION:
            raise ValueError(f"unsupported explorer evidence version: {self.version}")

    def to_dict(self) -> dict:
        return {
            "v": self.version,
            "status": self.status,
            "scope_path_count": self.scope_path_count,
            "navigation_success_count": self.navigation_success_count,
            "content_success_count": self.content_success_count,
            "gap_observation_count": self.gap_observation_count,
            "retained_navigation_view_count": self.retained_navigation_view_count,
            "retained_content_view_count": self.retained_content_view_count,
            "omitted_navigation_view_count": self.omitted_navigation_view_count,
            "omitted_content_view_count": self.omitted_content_view_count,
            "truncated_content_view_count": self.truncated_content_view_count,
            "scope_paths": list(self.scope_paths),
            "navigation_paths": list(self.navigation_paths),
            "content_paths": list(self.content_paths),
            "gap_paths": list(self.gap_paths),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ExplorerEvidenceAccount":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("explorer evidence account must be an object")
        return cls(
            status=str(value.get("status") or "not_assessed"),
            scope_path_count=int(value.get("scope_path_count") or 0),
            navigation_success_count=int(value.get("navigation_success_count") or 0),
            content_success_count=int(value.get("content_success_count") or 0),
            gap_observation_count=int(value.get("gap_observation_count") or 0),
            retained_navigation_view_count=int(value.get("retained_navigation_view_count") or 0),
            retained_content_view_count=int(value.get("retained_content_view_count") or 0),
            omitted_navigation_view_count=int(value.get("omitted_navigation_view_count") or 0),
            omitted_content_view_count=int(value.get("omitted_content_view_count") or 0),
            truncated_content_view_count=int(value.get("truncated_content_view_count") or 0),
            scope_paths=_strings(value.get("scope_paths") or (), "explorer evidence scope_paths"),
            navigation_paths=_strings(
                value.get("navigation_paths") or (), "explorer evidence navigation_paths",
            ),
            content_paths=_strings(value.get("content_paths") or (), "explorer evidence content_paths"),
            gap_paths=_strings(value.get("gap_paths") or (), "explorer evidence gap_paths"),
            version=int(value.get("v") or EXPLORER_EVIDENCE_VERSION),
        )


@dataclass(frozen=True)
class IntentClause:
    """One exact still-binding clause delegated from the parent's intent ledger."""

    id: str
    verbatim_clause: str
    source_artifact: str = ""
    source_range: tuple[int, int] | None = None
    authority: str = "user"
    kind: str = "constraint"

    def __post_init__(self) -> None:
        _text(self.id, "intent clause id")
        _text(self.verbatim_clause, "intent clause")
        _text(self.source_artifact, "intent source artifact", empty=True)
        if self.authority not in ("user", "task", "legacy"):
            raise ValueError("intent authority must be user, task, or legacy")
        if self.kind not in ("constraint", "correction"):
            raise ValueError("intent kind must be constraint or correction")
        if self.source_range is not None:
            if (not isinstance(self.source_range, tuple) or len(self.source_range) != 2
                    or any(not isinstance(value, int) for value in self.source_range)
                    or self.source_range[0] < 0 or self.source_range[1] < self.source_range[0]):
                raise ValueError("intent source_range must be a non-negative (start, end) tuple")

    @classmethod
    def from_value(cls, value: object) -> "IntentClause | None":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            clause = value.get("verbatim_clause")
            if not isinstance(clause, str) or not clause.strip():
                return None
            raw_range = value.get("source_range")
            source_range = None
            if (isinstance(raw_range, (list, tuple)) and len(raw_range) == 2
                    and all(isinstance(part, int) for part in raw_range)):
                source_range = (raw_range[0], raw_range[1])
            return cls(id=str(value.get("id") or "intent"), verbatim_clause=clause,
                       source_artifact=str(value.get("source_artifact") or ""),
                       source_range=source_range, authority=str(value.get("authority") or "legacy"),
                       kind=str(value.get("kind") or "constraint"))
        clause = getattr(value, "verbatim_clause", None)
        if isinstance(clause, str) and clause.strip():
            raw_range = getattr(value, "source_range", None)
            return cls(id=str(getattr(value, "id", None) or "intent"), verbatim_clause=clause,
                       source_artifact=str(getattr(value, "source_artifact", None) or ""),
                       source_range=tuple(raw_range) if raw_range is not None else None,
                       authority=str(getattr(value, "authority", "legacy")),
                       kind=str(getattr(value, "kind", "constraint")))
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "verbatim_clause": self.verbatim_clause,
            "source_artifact": self.source_artifact,
            "source_range": list(self.source_range) if self.source_range is not None else None,
            "authority": self.authority,
            "kind": self.kind,
        }


def exact_intent_clauses(values: object) -> tuple[IntentClause, ...]:
    """Project an IntentState or explicit selection without inventing paraphrases."""
    if values is None:
        return ()
    request = str(getattr(values, "current_request", "") or "") if hasattr(values, "current_request") else ""
    source = str(getattr(values, "current_source", "") or "") if hasattr(values, "current_source") else ""
    if hasattr(values, "open_entries"):
        # Provisional completion is not user acceptance; it remains binding in a delegated brief.
        values = (values.resident_entries() if hasattr(values, "resident_entries")
                  else values.open_entries())
    elif hasattr(values, "entries"):
        values = getattr(values, "entries")
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        return ()
    values = tuple(values)
    # A captured current correction already has the authoritative kind/provenance. Do not synthesize a
    # second default-constraint copy of the same text in the child brief.
    current_is_resident = any(
        str(getattr(value, "verbatim_clause", "") if not isinstance(value, Mapping)
            else value.get("verbatim_clause") or "") == request
        for value in values
    )
    current = None
    if request.strip() and not current_is_resident:
        current = IntentClause(
            id="current-request", verbatim_clause=request,
            source_artifact=source, source_range=(0, len(request)),
        )
    out = [current] if current is not None else []
    seen = {current.id} if current is not None else set()
    for value in values:
        clause = IntentClause.from_value(value)
        if clause is None or clause.id in seen:
            continue
        status = getattr(value, "status", value.get("status") if isinstance(value, Mapping) else "active")
        if status not in (None, "active", "provisionally_satisfied"):
            continue
        seen.add(clause.id)
        out.append(clause)
    return tuple(out)


@dataclass(frozen=True)
class SubagentBrief:
    objective: str
    # Optional immutable binding to the parent's model-maintained work item.  Empty preserves old artifacts;
    # new Active Work-aware launches carry it through brief, seal, receipt, and fan-in.
    work_item_id: str = ""
    intent_clauses: tuple[IntentClause, ...] = ()
    scope: tuple[str, ...] = ()
    # Host-minted from the parent's typed DelegationRequirement, never inferred by the child. This is the
    # one-to-one fan-out identity used for deterministic reduction; `scope` may legitimately contain context.
    delegation_target: str = ""
    exclusions: tuple[str, ...] = ()
    report_shape: str = (
        "Return status, scope covered, findings with evidence, content paths actually inspected, "
        "navigation-only paths, gaps, uncertainty, and conflicts."
    )
    canonical_refs: tuple[str, ...] = ()
    drift_policy: DriftPolicy = "report"
    integration_policy: IntegrationPolicy = "digest_ok"
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _text(self.objective, "subagent objective")
        _text(self.work_item_id, "subagent work_item_id", empty=True)
        object.__setattr__(self, "intent_clauses", tuple(
            clause if isinstance(clause, IntentClause) else IntentClause.from_value(clause)
            for clause in self.intent_clauses))
        if any(clause is None for clause in self.intent_clauses):
            raise ValueError("intent_clauses contains an invalid entry")
        object.__setattr__(self, "scope", _strings(self.scope, "subagent scope"))
        _text(self.delegation_target, "subagent delegation_target", empty=True)
        object.__setattr__(self, "exclusions", _strings(self.exclusions, "subagent exclusions"))
        _text(self.report_shape, "subagent report_shape")
        refs = _unique(_strings(self.canonical_refs, "subagent canonical_refs"))
        if any(not _CANONICAL_REF.fullmatch(ref) for ref in refs):
            raise ValueError(
                "subagent refs must be immutable subagents/sub-N.md or artifacts/<id>.md handles"
            )
        object.__setattr__(self, "canonical_refs", refs)
        if self.drift_policy not in ("report", "fail", "ignore"):
            raise ValueError("drift_policy must be report, fail, or ignore")
        if self.integration_policy not in ("digest_ok", "report_required"):
            raise ValueError("integration_policy must be digest_ok or report_required")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(f"unsupported subagent brief version: {self.contract_version}")

    @classmethod
    def create(cls, objective: str, *, intent_entries: object = (), scope: object = (),
               work_item_id: str = "",
               delegation_target: str = "",
               exclusions: object = (), report_shape: str | None = None,
               canonical_refs: object = (), drift_policy: DriftPolicy = "report",
               integration_policy: IntegrationPolicy = "digest_ok") -> "SubagentBrief":
        fields = {}
        if report_shape is not None:
            fields["report_shape"] = report_shape
        return cls(objective=objective, work_item_id=str(work_item_id or ""),
                   intent_clauses=exact_intent_clauses(intent_entries),
                   scope=_strings(scope, "subagent scope"), exclusions=_strings(exclusions, "subagent exclusions"),
                   delegation_target=str(delegation_target or ""),
                   canonical_refs=_strings(canonical_refs, "subagent canonical_refs"),
                   drift_policy=drift_policy, integration_policy=integration_policy, **fields)

    def to_dict(self) -> dict:
        # task/grants are compatibility aliases consumed by existing renderers and roster tests.
        record = {
            "v": self.contract_version,
            "objective": self.objective,
            "task": self.objective,
            "work_item_id": self.work_item_id,
            "intent_clauses": [clause.to_dict() for clause in self.intent_clauses],
            "scope": list(self.scope),
            "delegation_target": self.delegation_target,
            "exclusions": list(self.exclusions),
            "report_shape": self.report_shape,
            "canonical_refs": list(self.canonical_refs),
            "grants": list(self.canonical_refs),
            "drift_policy": self.drift_policy,
        }
        # Retired from the live contract. Preserve a non-default value only when round-tripping an old artifact.
        if self.integration_policy != "digest_ok":
            record["integration_policy"] = self.integration_policy
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentBrief":
        if not isinstance(value, Mapping):
            raise ValueError("subagent brief must be an object")
        # Old v3 roster records could persist a mutable subagents/<name>.md grant. It remains readable in
        # the original record, but cannot become a typed dependency without an archive resolver. New records
        # always carry canonical_refs explicitly and are validated strictly.
        raw_refs = value.get("canonical_refs") if "canonical_refs" in value else value.get("grants") or ()
        if "canonical_refs" not in value:
            raw_refs = tuple(ref for ref in _strings(raw_refs, "subagent canonical_refs")
                             if _CANONICAL_REF.fullmatch(ref))
        return cls(
            objective=str(value.get("objective") or value.get("task") or ""),
            work_item_id=str(value.get("work_item_id") or ""),
            intent_clauses=tuple(clause for row in (value.get("intent_clauses") or ())
                                 if (clause := IntentClause.from_value(row)) is not None),
            scope=_strings(value.get("scope") or (), "subagent scope"),
            delegation_target=str(value.get("delegation_target") or ""),
            exclusions=_strings(value.get("exclusions") or (), "subagent exclusions"),
            report_shape=str(value.get("report_shape") or cls.__dataclass_fields__["report_shape"].default),
            canonical_refs=_strings(raw_refs, "subagent canonical_refs"),
            drift_policy=str(value.get("drift_policy") or "report"),
            integration_policy=str(value.get("integration_policy") or "digest_ok"),
            contract_version=int(value.get("v") or CONTRACT_VERSION),
        )

    def render(self) -> str:
        lines = ["DELEGATED OBJECTIVE (exact)", self.objective]
        if self.work_item_id:
            lines += ["", "PARENT ACTIVE WORK ITEM (immutable binding)", self.work_item_id]
        user_clauses = [clause for clause in self.intent_clauses
                        if clause.authority == "user" and clause.kind == "constraint"]
        corrections = [clause for clause in self.intent_clauses
                       if clause.authority == "user" and clause.kind == "correction"]
        task_clauses = [clause for clause in self.intent_clauses
                        if clause.authority != "user" and clause.kind == "constraint"]
        for heading, clauses in (
            ("BINDING USER CONSTRAINTS (verbatim; preserve exactly)", user_clauses),
            ("USER CORRECTIONS / CLARIFICATIONS (newer wording overrides conflicts; factual claims "
             "still require live verification)", corrections),
            ("PARENT TASK CONSTRAINTS (agent/legacy maintained; do not treat as user quotes)", task_clauses),
        ):
            if not clauses:
                continue
            lines += ["", heading]
            for clause in clauses:
                source = f" [source: {clause.source_artifact}" if clause.source_artifact else " [source: current task"
                if clause.source_range is not None:
                    source += f" bytes {clause.source_range[0]}:{clause.source_range[1]}"
                lines.append(f"- {clause.verbatim_clause}{source}]")
        if self.delegation_target:
            lines += ["", "PRIMARY DELEGATION TARGET (host-bound)", self.delegation_target]
        if self.scope:
            lines += ["", "SCOPE"] + [f"- {item}" for item in self.scope]
        if self.exclusions:
            lines += ["", "EXCLUSIONS"] + [f"- {item}" for item in self.exclusions]
        lines += [
            "", "EXPECTED REPORT", self.report_shape,
            "", "EVIDENCE STANDARD (binding)",
            "- Keep three layers distinct: OBSERVED bytes/tool results; INFERENCE from those bytes; "
            "CONDITIONAL CONSEQUENCE with every prerequisite stated.",
            "- A child report is testimony, not workspace truth. Quote the smallest load-bearing observation "
            "and do not label an interpretation as observed. Copy file:line exactly from that observation; "
            "never broaden a cited line range or guess a line number from function position.",
            "- When reporting the current workspace as fact, prefer a concrete directly observed failure over a "
            "higher-severity chain whose caller, sink, runtime path, input control, or environmental prerequisite "
            "was not observed. An explicitly requested hypothetical/threat-model analysis is still allowed when "
            "its assumptions and every material prerequisite remain explicit.",
            "- Constructing a command/query is not executing it; comparing values does not establish their "
            "storage representation or measurable exploitability; catching an exception in one try-region "
            "does not establish global process behavior.",
            "- If a definition, caller, sink, or runtime path is absent from the evidence, say 'if'/'unless' "
            "and name that gap. Never silently strengthen possible/could/may into is/does/will.",
            "- When the objective asks for one top bug/finding, end with exactly one physical line under 800 "
            "characters: 'TOP CLAIM: <the finding and every material qualifier/prerequisite>'. This line is "
            "a stable synthesis target in the sealed report, not certified as workspace fact.",
            "", f"WORKSPACE DRIFT POLICY: {self.drift_policy}",
        ]
        # Compatibility-only: new live briefs use the default and no longer teach the model an integration dial.
        if self.integration_policy != "digest_ok":
            lines.append(f"PARENT INTEGRATION POLICY (legacy): {self.integration_policy}")
        if self.canonical_refs:
            lines += ["", "INPUT REPORTS (immutable sealed inputs)"]
            lines += [f'- read_file("{ref}")' for ref in self.canonical_refs]
        return "\n".join(lines)


def capture_workspace_revision(root: str, paths: Iterable[str]) -> tuple[WorkspaceRevision, tuple[str, ...]]:
    """Capture valid in-root dependencies and report un-fingerprintable paths honestly."""
    root_real = os.path.realpath(root or ".")
    dependencies: list[PathRevision] = []
    gaps = []
    for path in _unique(str(path) for path in paths):
        try:
            dependencies.append(fingerprint_path(root_real, path))
        except (OSError, ValueError) as exc:
            gaps.append(f"could not fingerprint {path}: {exc}")
    return WorkspaceRevision(root_real, tuple(dependencies)), tuple(gaps)


@dataclass(frozen=True)
class SubagentClaim:
    """One bounded child interpretation, kept distinct from its primary observations.

    ``report_exact`` proves only that these bytes occurred in the sealed child report. ``observation_refs`` are
    candidate grounding locators, not an entailment certificate. The host therefore never promotes this object to
    workspace fact; a parent may quote it as delegated testimony or independently verify it against observations.
    """

    text: str
    report_exact: str
    modality: Literal["inference", "conditional"] = "inference"
    observation_refs: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    version: int = CLAIM_VERSION

    def __post_init__(self) -> None:
        _text(self.text, "subagent claim text")
        _text(self.report_exact, "subagent claim report_exact")
        if len(self.text.encode("utf-8")) > 600 or len(self.report_exact.encode("utf-8")) > 1200:
            raise ValueError("subagent claim exceeds its bounded projection")
        if self.modality not in ("inference", "conditional"):
            raise ValueError("subagent claim modality must be inference or conditional")
        refs = _unique(_strings(self.observation_refs, "subagent claim observation_refs"))
        if any(not _SHA256.fullmatch(ref) for ref in refs):
            raise ValueError("subagent claim observation_refs must be lowercase sha256 values")
        object.__setattr__(self, "observation_refs", refs)
        object.__setattr__(
            self, "prerequisites", _unique(_strings(self.prerequisites, "subagent claim prerequisites")),
        )
        if self.version != CLAIM_VERSION:
            raise ValueError(f"unsupported subagent claim version: {self.version}")

    def to_dict(self) -> dict:
        return {
            "v": self.version,
            "text": self.text,
            "report_exact": self.report_exact,
            "modality": self.modality,
            "observation_refs": list(self.observation_refs),
            "prerequisites": list(self.prerequisites),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentClaim":
        if not isinstance(value, Mapping):
            raise ValueError("subagent claim must be an object")
        refs = value.get("observation_refs") or ()
        prerequisites = value.get("prerequisites") or ()
        if not isinstance(refs, (list, tuple)) or not isinstance(prerequisites, (list, tuple)):
            raise ValueError("subagent claim references and prerequisites must be arrays")
        return cls(
            text=str(value.get("text") or ""),
            report_exact=str(value.get("report_exact") or ""),
            modality=str(value.get("modality") or "inference"),
            observation_refs=_strings(refs, "subagent claim observation_refs"),
            prerequisites=_strings(prerequisites, "subagent claim prerequisites"),
            version=int(value.get("v") or CLAIM_VERSION),
        )


@dataclass(frozen=True)
class SubagentArtifact:
    kind: str
    name: str
    workspace_id: str
    session_id: str
    task_id: str
    parent_id: str
    brief: SubagentBrief
    status: str
    coverage: str
    report: str
    # The report bytes are the authoritative child testimony.  Hash/size are always derived from those bytes
    # on construction/reload so stale metadata cannot make the envelope unreadable. ``report_completion`` says
    # whether the provider completed that testimony; it is deliberately independent of the child's operational
    # status (a writable worker can finish its prose while still reporting failed work).
    report_sha256: str = ""
    report_bytes: int = 0
    report_completion: ReportCompletion = "unknown"
    report_stop_reason: str = ""
    explorer_evidence: ExplorerEvidenceAccount = field(default_factory=ExplorerEvidenceAccount)
    # Operational completion and source coverage are orthogonal. ``status`` answers whether the child ran and
    # sealed. This field proves only whether a fan-in child completely read and path-cited its granted reports;
    # it does not establish that the report's claims are correct.
    source_coverage_status: SourceCoverageStatus = "not_assessed"
    consumed_refs: tuple[str, ...] = ()
    cited_refs: tuple[str, ...] = ()
    covered_refs: tuple[str, ...] = ()
    source_gaps: tuple[str, ...] = ()
    # Stable sibling identity assigned synchronously when spawn_agent is accepted. Archive handles are
    # intentionally completion-ordered for backward compatibility, so they cannot answer "the first
    # subagent" when children run concurrently. Default preserves direct-constructor compatibility.
    launch_ordinal: int = 0
    findings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    # Full redacted child-visible tool results live in ``observations`` and are exposed as page-backed evidence.
    # This independent bounded copy is the only observation payload eligible for automatic model projection.
    observation_preview: tuple[SubagentObservation, ...] = ()
    observations: tuple[SubagentObservation, ...] = ()
    claims: tuple[SubagentClaim, ...] = ()
    files: tuple[str, ...] = ()
    change_set: tuple[str, ...] = ()
    workspace_revision: WorkspaceRevision = field(
        default_factory=lambda: WorkspaceRevision(os.path.realpath("."), ()))
    gaps: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    # Optional/legacy indexes are conveniences over the raw report + observations.  Invalid entries are dropped
    # independently and recorded here; they never make the canonical evidence envelope unreadable.
    projection_gaps: tuple[str, ...] = ()
    error: str = ""
    steps: int = 0
    usage: Mapping[str, Any] = field(default_factory=dict)
    trace: str = ""
    lesson: str = ""
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("kind", "workspace_id", "session_id", "task_id", "status", "coverage"):
            _text(getattr(self, name), f"subagent artifact {name}")
        for name in ("name", "parent_id", "report", "report_stop_reason", "error", "trace", "lesson"):
            _text(getattr(self, name), f"subagent artifact {name}", empty=True)
        report_encoded = self.report.encode("utf-8")
        object.__setattr__(self, "report_sha256", hashlib.sha256(report_encoded).hexdigest())
        object.__setattr__(self, "report_bytes", len(report_encoded))
        completion = str(self.report_completion or "unknown")
        if completion not in {"complete", "partial", "absent", "unknown"}:
            completion = "unknown"
        if not self.report:
            completion = "absent"
        object.__setattr__(self, "report_completion", completion)
        if not isinstance(self.brief, SubagentBrief):
            raise ValueError("subagent artifact brief must be a SubagentBrief")
        if isinstance(self.explorer_evidence, Mapping):
            object.__setattr__(
                self, "explorer_evidence", ExplorerEvidenceAccount.from_dict(self.explorer_evidence),
            )
        elif not isinstance(self.explorer_evidence, ExplorerEvidenceAccount):
            raise ValueError("subagent artifact explorer_evidence must be an ExplorerEvidenceAccount")
        if not isinstance(self.workspace_revision, WorkspaceRevision):
            raise ValueError("subagent artifact workspace_revision must be a WorkspaceRevision")
        if not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("subagent artifact steps must be non-negative")
        if not isinstance(self.launch_ordinal, int) or self.launch_ordinal < 0:
            raise ValueError("subagent artifact launch_ordinal must be non-negative")
        if not isinstance(self.usage, Mapping):
            raise ValueError("subagent artifact usage must be an object")
        object.__setattr__(
            self, "source_coverage_status", normalize_source_coverage_status(self.source_coverage_status),
        )
        def parse_observations(values, field_name):
            parsed = []
            for observation in values:
                if isinstance(observation, SubagentObservation):
                    parsed.append(observation)
                elif isinstance(observation, Mapping):
                    parsed.append(SubagentObservation.from_dict(observation))
                else:
                    raise ValueError(
                        f"subagent artifact {field_name} must contain typed observation objects"
                    )
            return tuple(parsed)

        parsed_observations = parse_observations(self.observations, "observations")
        parsed_preview = parse_observations(self.observation_preview, "observation_preview")
        # Legacy artifacts stored only the already-bounded observation capsule. Reuse it as the preview when it
        # still satisfies today's explicit projection bound; a large third-party record is never auto-injected.
        if not parsed_preview and parsed_observations \
                and len(parsed_observations) <= _OBSERVATION_PREVIEW_COUNT_LIMIT \
                and sum(item.view_bytes for item in parsed_observations) <= _OBSERVATION_PREVIEW_BYTES_LIMIT:
            parsed_preview = parsed_observations
        if len(parsed_preview) > _OBSERVATION_PREVIEW_COUNT_LIMIT \
                or sum(item.view_bytes for item in parsed_preview) > _OBSERVATION_PREVIEW_BYTES_LIMIT:
            raise ValueError("subagent artifact observation_preview exceeds its bounded projection budget")
        object.__setattr__(self, "observations", parsed_observations)
        object.__setattr__(self, "observation_preview", parsed_preview)
        try:
            projection_gaps = list(_unique(_strings(
                self.projection_gaps, "subagent artifact projection_gaps",
            )))
        except ValueError:
            projection_gaps = ["discarded malformed projection_gaps metadata"]
        parsed_claims = []
        observation_hashes = {
            observation.view_sha256
            for observation in parsed_observations if observation.status == "succeeded"
        }
        for index, raw_claim in enumerate(self.claims, start=1):
            try:
                claim = (
                    raw_claim if isinstance(raw_claim, SubagentClaim)
                    else SubagentClaim.from_dict(raw_claim) if isinstance(raw_claim, Mapping)
                    else (_ for _ in ()).throw(ValueError("claim is not an object"))
                )
                if claim.report_exact not in self.report:
                    raise ValueError("report_exact is absent from the sealed report")
                if any(item not in self.report for item in claim.prerequisites):
                    raise ValueError("a prerequisite is absent from the sealed report")
                if not set(claim.observation_refs).issubset(observation_hashes):
                    raise ValueError("observation_refs do not resolve to successful observations")
            except (TypeError, ValueError) as exc:
                projection_gaps.append(
                    f"discarded invalid legacy claim #{index}: {type(exc).__name__}: {exc}"
                )
                continue
            parsed_claims.append(claim)
        object.__setattr__(self, "claims", tuple(parsed_claims))
        object.__setattr__(self, "projection_gaps", _unique(projection_gaps))
        for name in (
            "findings", "evidence_refs", "consumed_refs", "cited_refs", "covered_refs", "source_gaps",
            "files", "change_set", "gaps", "uncertainty", "conflicts",
        ):
            object.__setattr__(self, name, _unique(_strings(getattr(self, name), f"subagent artifact {name}")))
        required_refs = set(self.brief.canonical_refs)
        valid_consumed = tuple(ref for ref in self.consumed_refs if ref in required_refs)
        valid_cited = tuple(ref for ref in self.cited_refs if ref in required_refs)
        valid_covered = tuple(
            ref for ref in self.covered_refs if ref in set(valid_consumed) & set(valid_cited)
        )
        if valid_consumed != self.consumed_refs:
            projection_gaps.append("discarded consumed_refs outside brief canonical_refs")
        if valid_cited != self.cited_refs:
            projection_gaps.append("discarded cited_refs outside brief canonical_refs")
        if valid_covered != self.covered_refs:
            projection_gaps.append("discarded covered_refs not both consumed and cited")
        object.__setattr__(self, "consumed_refs", valid_consumed)
        object.__setattr__(self, "cited_refs", valid_cited)
        object.__setattr__(self, "covered_refs", valid_covered)
        object.__setattr__(self, "projection_gaps", _unique(projection_gaps))
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(f"unsupported subagent artifact version: {self.contract_version}")

    @classmethod
    def create(cls, *, kind: str, name: str, workspace_id: str, session_id: str,
               task_id: str, parent_id: str, brief: SubagentBrief, status: str,
               coverage: str, report: str, files: Iterable[str] = (), workspace_root: str = ".",
               gaps: Iterable[str] = (), uncertainty: Iterable[str] = (),
               launch_ordinal: int = 0, **fields) -> "SubagentArtifact":
        files = _unique(str(path) for path in files)
        revision, fingerprint_gaps = capture_workspace_revision(workspace_root, files)
        return cls(kind=kind, name=name, workspace_id=workspace_id, session_id=session_id,
                   task_id=task_id, parent_id=parent_id, launch_ordinal=launch_ordinal,
                   brief=brief, status=status,
                   coverage=coverage, report=report, files=files, workspace_revision=revision,
                   gaps=_unique((*gaps, *fingerprint_gaps)), uncertainty=_unique(uncertainty), **fields)

    def to_record(self) -> dict:
        # New records use the precise source-coverage vocabulary. ``from_record`` remains able to read the
        # old v1 epistemic/grounding keys, but we do not perpetuate that over-claiming vocabulary in new seals.
        return {
            "contract_v": self.contract_version,
            "kind": self.kind,
            "name": self.name,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "launch_ordinal": self.launch_ordinal,
            "task": self.brief.objective,
            "brief": self.brief.to_dict(),
            "status": self.status,
            "explorer_evidence": self.explorer_evidence.to_dict(),
            "source_coverage_status": self.source_coverage_status,
            "coverage": self.coverage,
            "report": self.report,
            "report_sha256": self.report_sha256,
            "report_bytes": self.report_bytes,
            "report_completion": self.report_completion,
            "report_stop_reason": self.report_stop_reason,
            "findings": list(self.findings),
            "evidence_refs": list(self.evidence_refs),
            "consumed_refs": list(self.consumed_refs),
            "cited_refs": list(self.cited_refs),
            "covered_refs": list(self.covered_refs),
            "source_gaps": list(self.source_gaps),
            "observation_preview": [
                observation.to_dict() for observation in self.observation_preview
            ],
            "observations": [observation.to_dict() for observation in self.observations],
            "claims": [claim.to_dict() for claim in self.claims],
            "refs": list(self.brief.canonical_refs),
            "files": list(self.files),
            "change_set": list(self.change_set),
            "workspace_revision": self.workspace_revision.as_dict(),
            "gaps": list(self.gaps),
            "uncertainty": list(self.uncertainty),
            "conflicts": list(self.conflicts),
            "projection_gaps": list(self.projection_gaps),
            "error": self.error,
            "steps": self.steps,
            "usage": dict(self.usage),
            "trace": self.trace,
            "lesson": self.lesson,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "SubagentArtifact":
        if not isinstance(value, Mapping):
            raise ValueError("subagent artifact must be an object")
        brief = SubagentBrief.from_dict(value.get("brief") or {"task": value.get("task", ""),
                                                                "grants": value.get("refs") or ()})
        revision_data = value.get("workspace_revision")
        revision = (WorkspaceRevision.from_dict(dict(revision_data)) if isinstance(revision_data, Mapping)
                    else WorkspaceRevision(os.path.realpath("."), ()))
        raw_claims = value.get("claims") or ()
        try:
            projection_gaps = list(_strings(
                value.get("projection_gaps") or (), "projection_gaps",
            ))
        except ValueError:
            projection_gaps = ["discarded malformed projection_gaps metadata"]
        if not isinstance(raw_claims, (list, tuple)):
            projection_gaps.append("discarded malformed legacy claims projection: expected an array")
            raw_claims = ()

        def optional_strings(key: str, *, legacy_key: str = "") -> tuple[str, ...]:
            raw = value.get(key)
            if raw is None and legacy_key:
                raw = value.get(legacy_key)
            try:
                return _strings(raw or (), key)
            except ValueError:
                projection_gaps.append(f"discarded malformed {key} projection")
                return ()

        try:
            source_coverage_status = normalize_source_coverage_status(
                value.get("source_coverage_status") or value.get("epistemic_status") or "not_assessed"
            )
        except ValueError:
            source_coverage_status = "not_assessed"
            projection_gaps.append("discarded malformed source_coverage_status projection")
        try:
            explorer_evidence = ExplorerEvidenceAccount.from_dict(
                value.get("explorer_evidence") if isinstance(value.get("explorer_evidence"), Mapping) else None
            )
        except (TypeError, ValueError):
            explorer_evidence = ExplorerEvidenceAccount()
            projection_gaps.append("discarded malformed explorer_evidence projection")
        return cls(
            kind=str(value.get("kind") or "subagent"), name=str(value.get("name") or ""),
            workspace_id=str(value.get("workspace_id") or revision.root),
            session_id=str(value.get("session_id") or "session-unknown"),
            task_id=str(value.get("task_id") or "task-unknown"),
            parent_id=str(value.get("parent_id") or ""),
            launch_ordinal=int(value.get("launch_ordinal") or 0), brief=brief,
            status=str(value.get("status") or "unknown"),
            explorer_evidence=explorer_evidence,
            source_coverage_status=source_coverage_status,
            coverage=str(value.get("coverage") or "coverage not recorded"),
            report=str(value.get("report") or ""),
            # Hash and size are derived from report bytes in __post_init__; never parse stale metadata as
            # authority or let a malformed legacy counter erase the envelope.
            report_sha256="", report_bytes=0,
            report_completion=str(value.get("report_completion") or "unknown"),
            report_stop_reason=str(value.get("report_stop_reason") or ""),
            findings=optional_strings("findings"),
            evidence_refs=optional_strings("evidence_refs"),
            consumed_refs=optional_strings("consumed_refs"),
            cited_refs=optional_strings("cited_refs"),
            covered_refs=optional_strings("covered_refs", legacy_key="grounding_refs"),
            source_gaps=optional_strings("source_gaps", legacy_key="grounding_gaps"),
            observation_preview=tuple(
                SubagentObservation.from_dict(row) for row in (value.get("observation_preview") or ())
                if isinstance(row, Mapping)
            ),
            observations=tuple(
                SubagentObservation.from_dict(row) for row in (value.get("observations") or ())
                if isinstance(row, Mapping)
            ),
            # Parse claim rows inside __post_init__, where each malformed optional row can be dropped without
            # making the report/observation envelope unreadable.
            claims=tuple(raw_claims),
            files=optional_strings("files"), change_set=optional_strings("change_set"),
            workspace_revision=revision, gaps=optional_strings("gaps"),
            uncertainty=optional_strings("uncertainty"), conflicts=optional_strings("conflicts"),
            projection_gaps=tuple(projection_gaps),
            error=str(value.get("error") or ""), steps=int(value.get("steps") or 0),
            usage=value.get("usage") or {}, trace=str(value.get("trace") or ""),
            lesson=str(value.get("lesson") or ""),
            contract_version=int(value.get("contract_v") or CONTRACT_VERSION),
        )


__all__ = [
    "IntentClause", "SubagentBrief", "SubagentArtifact", "SubagentObservation", "SubagentClaim", "DriftPolicy",
    "IntegrationPolicy", "ExplorerEvidenceAccount", "ExplorerEvidenceStatus", "ReportCompletion",
    "SourceCoverageStatus", "normalize_source_coverage_status",
    "exact_intent_clauses", "capture_workspace_revision",
]
