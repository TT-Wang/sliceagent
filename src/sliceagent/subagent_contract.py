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
DriftPolicy = Literal["report", "fail", "ignore"]
_CANONICAL_REF = re.compile(r"^subagents/sub-\d+\.md$")
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


@dataclass(frozen=True)
class SubagentObservation:
    """One bounded, persisted view returned by a successful read-only child tool.

    ``raw_sha256`` binds the exact view delivered to the child before persistence redaction/capping;
    ``view_sha256`` binds the text that is actually retained.  A redacted or truncated view is useful evidence
    for its visible bytes, but never proof about omitted bytes.
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
    intent_clauses: tuple[IntentClause, ...] = ()
    scope: tuple[str, ...] = ()
    # Host-minted from the parent's typed DelegationRequirement, never inferred by the child. This is the
    # one-to-one fan-out identity used for deterministic reduction; `scope` may legitimately contain context.
    delegation_target: str = ""
    exclusions: tuple[str, ...] = ()
    report_shape: str = (
        "Return status, scope covered, findings with evidence, files examined, gaps, uncertainty, and conflicts."
    )
    canonical_refs: tuple[str, ...] = ()
    drift_policy: DriftPolicy = "report"
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _text(self.objective, "subagent objective")
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
            raise ValueError("subagent refs must be immutable subagents/sub-N.md handles")
        object.__setattr__(self, "canonical_refs", refs)
        if self.drift_policy not in ("report", "fail", "ignore"):
            raise ValueError("drift_policy must be report, fail, or ignore")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(f"unsupported subagent brief version: {self.contract_version}")

    @classmethod
    def create(cls, objective: str, *, intent_entries: object = (), scope: object = (),
               delegation_target: str = "",
               exclusions: object = (), report_shape: str | None = None,
               canonical_refs: object = (), drift_policy: DriftPolicy = "report") -> "SubagentBrief":
        fields = {}
        if report_shape is not None:
            fields["report_shape"] = report_shape
        return cls(objective=objective, intent_clauses=exact_intent_clauses(intent_entries),
                   scope=_strings(scope, "subagent scope"), exclusions=_strings(exclusions, "subagent exclusions"),
                   delegation_target=str(delegation_target or ""),
                   canonical_refs=_strings(canonical_refs, "subagent canonical_refs"),
                   drift_policy=drift_policy, **fields)

    def to_dict(self) -> dict:
        # task/grants are compatibility aliases consumed by existing renderers and roster tests.
        return {
            "v": self.contract_version,
            "objective": self.objective,
            "task": self.objective,
            "intent_clauses": [clause.to_dict() for clause in self.intent_clauses],
            "scope": list(self.scope),
            "delegation_target": self.delegation_target,
            "exclusions": list(self.exclusions),
            "report_shape": self.report_shape,
            "canonical_refs": list(self.canonical_refs),
            "grants": list(self.canonical_refs),
            "drift_policy": self.drift_policy,
        }

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
            intent_clauses=tuple(clause for row in (value.get("intent_clauses") or ())
                                 if (clause := IntentClause.from_value(row)) is not None),
            scope=_strings(value.get("scope") or (), "subagent scope"),
            delegation_target=str(value.get("delegation_target") or ""),
            exclusions=_strings(value.get("exclusions") or (), "subagent exclusions"),
            report_shape=str(value.get("report_shape") or cls.__dataclass_fields__["report_shape"].default),
            canonical_refs=_strings(raw_refs, "subagent canonical_refs"),
            drift_policy=str(value.get("drift_policy") or "report"),
            contract_version=int(value.get("v") or CONTRACT_VERSION),
        )

    def render(self) -> str:
        lines = ["DELEGATED OBJECTIVE (exact)", self.objective]
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
            "indexed as your testimony, not certified as workspace fact.",
            "", f"WORKSPACE DRIFT POLICY: {self.drift_policy}",
        ]
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
    # Stable sibling identity assigned synchronously when spawn_agent is accepted. Archive handles are
    # intentionally completion-ordered for backward compatibility, so they cannot answer "the first
    # subagent" when children run concurrently. Default preserves direct-constructor compatibility.
    launch_ordinal: int = 0
    findings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    observations: tuple[SubagentObservation, ...] = ()
    claims: tuple[SubagentClaim, ...] = ()
    files: tuple[str, ...] = ()
    change_set: tuple[str, ...] = ()
    workspace_revision: WorkspaceRevision = field(
        default_factory=lambda: WorkspaceRevision(os.path.realpath("."), ()))
    gaps: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    error: str = ""
    steps: int = 0
    usage: Mapping[str, Any] = field(default_factory=dict)
    trace: str = ""
    lesson: str = ""
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("kind", "workspace_id", "session_id", "task_id", "status", "coverage"):
            _text(getattr(self, name), f"subagent artifact {name}")
        for name in ("name", "parent_id", "report", "error", "trace", "lesson"):
            _text(getattr(self, name), f"subagent artifact {name}", empty=True)
        if not isinstance(self.brief, SubagentBrief):
            raise ValueError("subagent artifact brief must be a SubagentBrief")
        if not isinstance(self.workspace_revision, WorkspaceRevision):
            raise ValueError("subagent artifact workspace_revision must be a WorkspaceRevision")
        if not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("subagent artifact steps must be non-negative")
        if not isinstance(self.launch_ordinal, int) or self.launch_ordinal < 0:
            raise ValueError("subagent artifact launch_ordinal must be non-negative")
        if not isinstance(self.usage, Mapping):
            raise ValueError("subagent artifact usage must be an object")
        parsed_observations = []
        for observation in self.observations:
            if isinstance(observation, SubagentObservation):
                parsed_observations.append(observation)
            elif isinstance(observation, Mapping):
                parsed_observations.append(SubagentObservation.from_dict(observation))
            else:
                raise ValueError("subagent artifact observations must contain typed observation objects")
        object.__setattr__(self, "observations", tuple(parsed_observations))
        parsed_claims = []
        for claim in self.claims:
            if isinstance(claim, SubagentClaim):
                parsed_claims.append(claim)
            elif isinstance(claim, Mapping):
                parsed_claims.append(SubagentClaim.from_dict(claim))
            else:
                raise ValueError("subagent artifact claims must contain typed claim objects")
        if any(claim.report_exact not in self.report for claim in parsed_claims):
            raise ValueError("subagent artifact claim report_exact must occur verbatim in the sealed report")
        if any(any(item not in self.report for item in claim.prerequisites) for claim in parsed_claims):
            raise ValueError("subagent artifact claim prerequisites must occur verbatim in the sealed report")
        observation_hashes = {observation.view_sha256 for observation in parsed_observations}
        if any(not set(claim.observation_refs).issubset(observation_hashes) for claim in parsed_claims):
            raise ValueError("subagent artifact claim observation_refs must resolve within the artifact")
        object.__setattr__(self, "claims", tuple(parsed_claims))
        for name in ("findings", "evidence_refs", "files", "change_set", "gaps", "uncertainty", "conflicts"):
            object.__setattr__(self, name, _unique(_strings(getattr(self, name), f"subagent artifact {name}")))
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
        # Every v1 key stays present. New readers use the explicit identity/evidence/revision fields.
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
            "coverage": self.coverage,
            "report": self.report,
            "findings": list(self.findings),
            "evidence_refs": list(self.evidence_refs),
            "observations": [observation.to_dict() for observation in self.observations],
            "claims": [claim.to_dict() for claim in self.claims],
            "refs": list(self.brief.canonical_refs),
            "files": list(self.files),
            "change_set": list(self.change_set),
            "workspace_revision": self.workspace_revision.as_dict(),
            "gaps": list(self.gaps),
            "uncertainty": list(self.uncertainty),
            "conflicts": list(self.conflicts),
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
        if not isinstance(raw_claims, (list, tuple)) or any(
                not isinstance(row, Mapping) for row in raw_claims):
            raise ValueError("subagent artifact claims must be an array of objects")
        return cls(
            kind=str(value.get("kind") or "subagent"), name=str(value.get("name") or ""),
            workspace_id=str(value.get("workspace_id") or revision.root),
            session_id=str(value.get("session_id") or "session-unknown"),
            task_id=str(value.get("task_id") or "task-unknown"),
            parent_id=str(value.get("parent_id") or ""),
            launch_ordinal=int(value.get("launch_ordinal") or 0), brief=brief,
            status=str(value.get("status") or "unknown"),
            coverage=str(value.get("coverage") or "coverage not recorded"),
            report=str(value.get("report") or ""), findings=_strings(value.get("findings") or (), "findings"),
            evidence_refs=_strings(value.get("evidence_refs") or (), "evidence_refs"),
            observations=tuple(
                SubagentObservation.from_dict(row) for row in (value.get("observations") or ())
                if isinstance(row, Mapping)
            ),
            claims=tuple(SubagentClaim.from_dict(row) for row in raw_claims),
            files=_strings(value.get("files") or (), "files"),
            change_set=_strings(value.get("change_set") or (), "change_set"),
            workspace_revision=revision, gaps=_strings(value.get("gaps") or (), "gaps"),
            uncertainty=_strings(value.get("uncertainty") or (), "uncertainty"),
            conflicts=_strings(value.get("conflicts") or (), "conflicts"),
            error=str(value.get("error") or ""), steps=int(value.get("steps") or 0),
            usage=value.get("usage") or {}, trace=str(value.get("trace") or ""),
            lesson=str(value.get("lesson") or ""),
            contract_version=int(value.get("contract_v") or CONTRACT_VERSION),
        )


__all__ = [
    "IntentClause", "SubagentBrief", "SubagentArtifact", "SubagentObservation", "SubagentClaim", "DriftPolicy",
    "exact_intent_clauses", "capture_workspace_revision",
]
