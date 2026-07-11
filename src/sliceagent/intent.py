"""Typed user-intent state.

The active intent ledger is semantic state, not a copy of the conversation.  The
current request is kept verbatim for the running turn; earlier clauses stay
resident only when they were deliberately recorded as standing obligations.

This module is deliberately pure: it owns no rendering, tools, or persistence.
Those layers serialize/project these records without becoming a second mutable
authority.
"""
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal


IntentStatus = Literal[
    "active",
    "provisionally_satisfied",
    "satisfied",
    "superseded",
    "deferred",
]
IntentAuthority = Literal["user", "task", "legacy"]
IntentKind = Literal["constraint", "correction"]
EffectAuthority = Literal["none", "explicit", "continuation", "uncertain"]
GroundingMode = Literal["sealed_past", "live_present", "both", "none"]
SourceNeed = Literal[
    "prior_user_utterance",
    "prior_assistant_utterance",
    "sealed_exchange",
    "execution_receipt",
    "historical_observation",
    "current_world",
]
SourceSpan = tuple[int, int]


@dataclass(frozen=True)
class EntityRef:
    """One resolved conversational entity, with the exact evidence for the binding.

    ``source`` distinguishes a name present in this request from a target inherited from the active task
    focus.  Inherited targets intentionally have no invented source span in the current request.
    """

    label: str
    kind: str = "entity"
    source: str = "explicit"
    source_span: SourceSpan | None = None

    def __post_init__(self) -> None:
        label = " ".join(str(self.label or "").split())
        if not label:
            raise ValueError("entity label must not be empty")
        if self.source_span is not None:
            if (not isinstance(self.source_span, tuple) or len(self.source_span) != 2
                    or not all(isinstance(value, int) for value in self.source_span)
                    or self.source_span[0] < 0 or self.source_span[1] <= self.source_span[0]):
                raise ValueError("invalid entity source span")
        object.__setattr__(self, "label", label)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "source": self.source,
            "source_span": list(self.source_span) if self.source_span is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "EntityRef | None":
        if not isinstance(value, Mapping) or not str(value.get("label") or "").strip():
            return None
        raw_span = value.get("source_span")
        span = None
        if (isinstance(raw_span, (list, tuple)) and len(raw_span) == 2
                and all(isinstance(part, int) for part in raw_span)):
            span = (raw_span[0], raw_span[1])
        try:
            return cls(
                label=str(value.get("label") or ""),
                kind=str(value.get("kind") or "entity"),
                source=str(value.get("source") or "explicit"),
                source_span=span,
            )
        except ValueError:
            return None


@dataclass(frozen=True)
class FocusRepair:
    """An explicit user repair of the active subject (for example, ``I mean Hunter``)."""

    replacement: EntityRef
    source_span: SourceSpan
    field: str = "target"

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "replacement": self.replacement.to_dict(),
            "source_span": list(self.source_span),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "FocusRepair | None":
        if not isinstance(value, Mapping):
            return None
        replacement = EntityRef.from_dict(value.get("replacement") or {})
        raw_span = value.get("source_span")
        if replacement is None or not (
            isinstance(raw_span, (list, tuple)) and len(raw_span) == 2
            and all(isinstance(part, int) for part in raw_span)
        ):
            return None
        try:
            return cls(
                replacement=replacement,
                source_span=(raw_span[0], raw_span[1]),
                field=str(value.get("field") or "target"),
            )
        except ValueError:
            return None


@dataclass(frozen=True)
class EvidenceQuery:
    """One host-routed evidence request used by selectors and renderers.

    The fields deliberately describe *what source projection is required*, not an answer.  This keeps natural
    language recognition in one deterministic place while receipt selection/rendering operate only on typed
    values.  The first vertical supports execution receipts; the vocabulary is intentionally small and closed.
    """

    source: str
    family: str = "all"
    predicate: str = "operations"
    scope: str = "task"

    def __post_init__(self) -> None:
        if self.source not in {"execution_receipt"}:
            raise ValueError(f"invalid evidence source {self.source!r}")
        if self.family not in {"all", "delegation", "command", "file", "file_read", "file_write"}:
            raise ValueError(f"invalid evidence family {self.family!r}")
        if self.predicate not in {"aggregate", "failure_detail", "operations"}:
            raise ValueError(f"invalid evidence predicate {self.predicate!r}")
        if self.scope not in {"task", "session", "latest_turn", "latest_matching_execution"}:
            raise ValueError(f"invalid evidence scope {self.scope!r}")

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "family": self.family,
            "predicate": self.predicate,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "EvidenceQuery | None":
        if not isinstance(value, Mapping):
            return None
        try:
            return cls(
                source=str(value.get("source") or ""),
                family=str(value.get("family") or "all"),
                predicate=str(value.get("predicate") or "operations"),
                scope=str(value.get("scope") or "task"),
            )
        except ValueError:
            return None


@dataclass(frozen=True)
class QualityEvidenceQuery:
    """Typed selector for claims about response quality rather than execution lifecycle.

    A response-quality judgment needs the user's request and the assistant's answer as one co-resident,
    immutable exchange.  Keeping this separate from :class:`EvidenceQuery` prevents execution receipts from
    being misused as proof that wording, scope, or helpfulness was good or bad.
    """

    scope: Literal["task", "session", "latest_response"] = "task"
    purpose: Literal["assess", "verify_assessment"] = "assess"
    prospective_requested: bool = False

    def __post_init__(self) -> None:
        if self.scope not in {"task", "session", "latest_response"}:
            raise ValueError(f"invalid quality evidence scope {self.scope!r}")
        if self.purpose not in {"assess", "verify_assessment"}:
            raise ValueError(f"invalid quality evidence purpose {self.purpose!r}")

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "purpose": self.purpose,
            "prospective_requested": self.prospective_requested,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "QualityEvidenceQuery | None":
        if not isinstance(value, Mapping) or not value or not any(
            key in value for key in ("scope", "purpose", "prospective_requested")
        ):
            return None
        try:
            return cls(
                scope=str(value.get("scope") or "task"),
                purpose=str(value.get("purpose") or "assess"),
                prospective_requested=bool(value.get("prospective_requested", False)),
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class DelegationRequirement:
    """A user-specified delegation mechanism that must be satisfied before terminal prose.

    This is not a general plan. It records only the mechanically checkable part of an explicit request: child kind,
    exact child count when supplied, named targets, and whether the calls were requested in parallel.
    """

    agent: str = "explorer"
    count: int | None = None
    targets: tuple[str, ...] = ()
    parallel: bool = False

    def __post_init__(self) -> None:
        agent = str(self.agent or "explorer").strip().casefold()
        if agent not in {"explorer"}:
            raise ValueError(f"unsupported delegation agent {agent!r}")
        if self.count is not None and (not isinstance(self.count, int) or not 1 <= self.count <= 64):
            raise ValueError("delegation count must be between 1 and 64")
        targets = tuple(dict.fromkeys(str(target).strip() for target in self.targets if str(target).strip()))
        object.__setattr__(self, "agent", agent)
        object.__setattr__(self, "targets", targets)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "count": self.count,
            "targets": list(self.targets),
            "parallel": self.parallel,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "DelegationRequirement | None":
        if not isinstance(value, Mapping) or not value:
            return None
        try:
            raw_count = value.get("count")
            return cls(
                agent=str(value.get("agent") or "explorer"),
                count=int(raw_count) if raw_count is not None else None,
                targets=tuple(value.get("targets") or ()),
                parallel=bool(value.get("parallel", False)),
            )
        except (TypeError, ValueError):
            return None


def _freeze_arg(value):
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_arg(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_arg(item) for item in value)
    return value


@dataclass(frozen=True)
class EffectGrant:
    """A turn-local operation/target capability derived from exact user-authored speech.

    ``tools`` is deliberately concrete: the execution gate never guesses that two effectful tools are
    interchangeable. ``mode=exact`` is used for a confirmed pending action and requires every captured
    argument to match. ``mode=scoped`` is used for an explicit current-turn directive and may authorize a
    small tool family, optionally restricted by ``target``/``target_arg``.
    """

    operation: str
    tools: tuple[str, ...]
    target: str = ""
    target_arg: str = ""
    exact_args: tuple[tuple[str, object], ...] = ()
    source_span: SourceSpan | None = None
    mode: Literal["scoped", "exact"] = "scoped"

    def __post_init__(self) -> None:
        operation = str(self.operation or "").strip()
        tools = tuple(dict.fromkeys(str(tool).strip() for tool in self.tools if str(tool).strip()))
        if not operation or not tools:
            raise ValueError("effect grant requires an operation and at least one tool")
        if self.mode not in ("scoped", "exact"):
            raise ValueError(f"invalid effect grant mode {self.mode!r}")
        if self.source_span is not None:
            if (not isinstance(self.source_span, tuple) or len(self.source_span) != 2
                    or not all(isinstance(value, int) for value in self.source_span)
                    or self.source_span[0] < 0 or self.source_span[1] <= self.source_span[0]):
                raise ValueError("invalid effect grant source span")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "target", str(self.target or "").strip())
        object.__setattr__(self, "target_arg", str(self.target_arg or "").strip())
        object.__setattr__(self, "exact_args", tuple(
            (str(key), _freeze_arg(value)) for key, value in self.exact_args if str(key).strip()
        ))

    @classmethod
    def exact(cls, tool: str, args: Mapping, *, source_span: SourceSpan | None = None) -> "EffectGrant":
        return cls(
            operation=f"exact:{tool}", tools=(tool,),
            exact_args=tuple((str(key), value) for key, value in (args or {}).items()),
            source_span=source_span, mode="exact",
        )

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "tools": list(self.tools),
            "target": self.target,
            "target_arg": self.target_arg,
            "exact_args": {key: value for key, value in self.exact_args},
            "source_span": list(self.source_span) if self.source_span is not None else None,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "EffectGrant | None":
        if not isinstance(value, Mapping):
            return None
        raw_span = value.get("source_span")
        span = None
        if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2 \
                and all(isinstance(part, int) for part in raw_span):
            span = (raw_span[0], raw_span[1])
        raw_args = value.get("exact_args")
        if isinstance(raw_args, Mapping):
            exact_args = tuple((str(key), item) for key, item in raw_args.items())
        elif isinstance(raw_args, (list, tuple)):
            exact_args = tuple(
                (str(item[0]), item[1]) for item in raw_args
                if isinstance(item, (list, tuple)) and len(item) == 2
            )
        else:
            exact_args = ()
        try:
            return cls(
                operation=str(value.get("operation") or ""),
                tools=tuple(value.get("tools") or ()),
                target=str(value.get("target") or ""),
                target_arg=str(value.get("target_arg") or ""),
                exact_args=exact_args,
                source_span=span,
                mode=str(value.get("mode") or "scoped"),
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class TurnAdmission:
    """The one immutable host reading of an exact user request.

    The request itself remains the semantic authority.  This record carries only the small facts the host
    must know *before* allowing side effects or promoting durable clauses: which exact source spans are the
    user's operative speech, which spans are attributed/reference data, whether effects were authorized,
    and whether an answer is about sealed past output or live present state.  It is deliberately absent from
    ``IntentState.to_records`` and is cleared at the turn seal.

    ``referents`` is the integration seam for the discourse resolver. Intent does not invent a competing
    reference model; callers may attach typed resolved-anchor objects here. This object is also the
    transactional admission seam: routing can preview it without mutation, the journal can start, and then
    the exact same value can be installed once. ``TurnContract`` below is only a compatibility alias.
    """

    request_text: str = ""
    request_source: str | None = None
    effect_authority: EffectAuthority = "none"
    grounding: GroundingMode = "none"
    authority_spans: tuple[SourceSpan, ...] = ()
    attributed_spans: tuple[SourceSpan, ...] = ()
    requested_modes: tuple[str, ...] = ()
    source_needs: tuple[SourceNeed, ...] = ()
    evidence_query: EvidenceQuery | None = None
    quality_evidence_query: QualityEvidenceQuery | None = None
    delegation_requirement: DelegationRequirement | None = None
    evidence_continuation: bool = False
    actor: EntityRef | None = None
    target: EntityRef | None = None
    focus_repairs: tuple[FocusRepair, ...] = ()
    effect_grants: tuple[EffectGrant, ...] = ()
    referents: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.effect_authority not in ("none", "explicit", "continuation", "uncertain"):
            raise ValueError(f"invalid effect authority {self.effect_authority!r}")
        if self.grounding not in ("sealed_past", "live_present", "both", "none"):
            raise ValueError(f"invalid grounding mode {self.grounding!r}")
        for label, spans in (("authority", self.authority_spans), ("attributed", self.attributed_spans)):
            if any(not isinstance(span, tuple) or len(span) != 2
                   or not all(isinstance(value, int) for value in span)
                   or span[0] < 0 or span[1] <= span[0] for span in spans):
                raise ValueError(f"invalid {label} source spans")
        object.__setattr__(self, "authority_spans", tuple(self.authority_spans))
        object.__setattr__(self, "attributed_spans", tuple(self.attributed_spans))
        object.__setattr__(self, "requested_modes", tuple(dict.fromkeys(
            str(mode).strip() for mode in self.requested_modes if str(mode).strip()
        )))
        valid_needs = {
            "prior_user_utterance", "prior_assistant_utterance", "execution_receipt",
            "sealed_exchange", "historical_observation", "current_world",
        }
        needs = tuple(dict.fromkeys(str(need).strip() for need in self.source_needs if str(need).strip()))
        if any(need not in valid_needs for need in needs):
            raise ValueError("invalid turn source need")
        if self.evidence_query is not None and not isinstance(self.evidence_query, EvidenceQuery):
            raise ValueError("evidence_query must be an EvidenceQuery")
        if (self.quality_evidence_query is not None
                and not isinstance(self.quality_evidence_query, QualityEvidenceQuery)):
            raise ValueError("quality_evidence_query must be a QualityEvidenceQuery")
        if (self.delegation_requirement is not None
                and not isinstance(self.delegation_requirement, DelegationRequirement)):
            raise ValueError("delegation_requirement must be a DelegationRequirement")
        object.__setattr__(self, "source_needs", needs)
        object.__setattr__(self, "focus_repairs", tuple(self.focus_repairs))
        object.__setattr__(self, "effect_grants", tuple(self.effect_grants))
        object.__setattr__(self, "referents", tuple(self.referents))

    def to_dict(self) -> dict:
        return {
            "request_text": self.request_text,
            "request_source": self.request_source,
            "effect_authority": self.effect_authority,
            "grounding": self.grounding,
            "authority_spans": [list(span) for span in self.authority_spans],
            "attributed_spans": [list(span) for span in self.attributed_spans],
            "requested_modes": list(self.requested_modes),
            "source_needs": list(self.source_needs),
            "evidence_query": self.evidence_query.to_dict() if self.evidence_query is not None else None,
            "quality_evidence_query": (
                self.quality_evidence_query.to_dict()
                if self.quality_evidence_query is not None else None
            ),
            "delegation_requirement": (
                self.delegation_requirement.to_dict()
                if self.delegation_requirement is not None else None
            ),
            "evidence_continuation": self.evidence_continuation,
            "actor": self.actor.to_dict() if self.actor is not None else None,
            "target": self.target.to_dict() if self.target is not None else None,
            "focus_repairs": [repair.to_dict() for repair in self.focus_repairs],
            "effect_grants": [grant.to_dict() for grant in self.effect_grants],
            # Referents may contain richer runtime objects; serialization is intentionally best-effort for
            # prompt/diagnostic compatibility rather than a second durable discourse store.
            "referents": [
                ref.to_dict() if hasattr(ref, "to_dict") else dict(ref) if isinstance(ref, Mapping) else str(ref)
                for ref in self.referents
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "TurnAdmission | None":
        """Decode the immutable journal form without re-interpreting the user's request."""
        if not isinstance(value, Mapping):
            return None

        def spans(name: str) -> tuple[SourceSpan, ...]:
            raw = value.get(name)
            if not isinstance(raw, (list, tuple)):
                return ()
            return tuple(
                (item[0], item[1]) for item in raw
                if isinstance(item, (list, tuple)) and len(item) == 2
                and all(isinstance(part, int) for part in item)
            )

        actor = EntityRef.from_dict(value.get("actor") or {})
        target = EntityRef.from_dict(value.get("target") or {})
        evidence_query = EvidenceQuery.from_dict(value.get("evidence_query") or {})
        quality_evidence_query = QualityEvidenceQuery.from_dict(
            value.get("quality_evidence_query") or {}
        )
        delegation_requirement = DelegationRequirement.from_dict(
            value.get("delegation_requirement") or {}
        )
        repairs = tuple(
            repair for raw in (value.get("focus_repairs") or ())
            if (repair := FocusRepair.from_dict(raw)) is not None
        )
        grants = tuple(
            grant for raw in (value.get("effect_grants") or ())
            if (grant := EffectGrant.from_dict(raw)) is not None
        )
        referents = tuple(
            dict(ref) if isinstance(ref, Mapping) else ref
            for ref in (value.get("referents") or ())
        )
        try:
            return cls(
                request_text=str(value.get("request_text") or ""),
                request_source=(str(value.get("request_source"))
                                if value.get("request_source") is not None else None),
                effect_authority=str(value.get("effect_authority") or "none"),
                grounding=str(value.get("grounding") or "none"),
                authority_spans=spans("authority_spans"),
                attributed_spans=spans("attributed_spans"),
                requested_modes=tuple(value.get("requested_modes") or ()),
                source_needs=tuple(value.get("source_needs") or ()),
                evidence_query=evidence_query,
                quality_evidence_query=quality_evidence_query,
                delegation_requirement=delegation_requirement,
                evidence_continuation=bool(value.get("evidence_continuation", False)),
                actor=actor,
                target=target,
                focus_repairs=repairs,
                effect_grants=grants,
                referents=referents,
            )
        except (TypeError, ValueError):
            return None


# Backwards-compatible public name. There is one object and one writer, not a TurnContract mirrored beside
# TurnAdmission.
TurnContract = TurnAdmission


# High-precision host capture for clauses whose language explicitly says they remain binding. This is not
# a general intent classifier: ordinary follow-ups stay in current_request/recent continuity, while durable
# constraints survive even if a model forgets to call require(). Exact source text is retained verbatim.
_EXPLICIT_CONSTRAINT = re.compile(
    r"(?:\b(?:must(?:n't|\s+not)?|never|do\s+not|don't|cannot|can't|preserve|required?|ensure|"
    r"make\s+sure)\b"
    r"|\bonly\s+(?:change|modify|edit|touch|use|return|output|write|call|support|include|exclude)\b"
    r"|\bwithout\s+(?:changing|modifying|editing|removing|adding|breaking)\b"
    r"|\bleave\b.{0,80}\b(?:unchanged|alone|intact)\b"
    r"|\bkeep\b.{0,80}\b(?:unchanged|stable|compatible|intact|the\s+same)\b"
    r"|^\s*(?:please\s+)?(?:use|target)\b)",
    re.IGNORECASE,
)
# Output/configuration choices are durable across the turns needed to complete a task, unlike the action
# that starts the task. Keep this deliberately narrower than _DIRECTIVE_PREFIX: ``Return exactly JSON`` and
# ``Format the response as a table`` are standing output constraints; ``Write the report`` and ``Fix app.py``
# are one-shot work already represented by current_request + the stable task objective.
_DURABLE_OUTPUT_CONSTRAINT = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"(?:return|output)\b|"
    r"respond\s+(?:in|with|as)\b|"
    r"format\b(?:[^.!?\n]{0,120}\b(?:output|response|answer|report)\b)?|"
    r"(?:make|write)\b[^.!?\n]{0,120}\b(?:as|in)\s+(?:a\s+)?"
    r"(?:json|ya?ml|xml|csv|markdown|table|list|bullets?|one\s+sentence|\d+\s+lines?)\b"
    r")",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(r"(?:\r?\n+|;+\s*|(?<=[.!?])\s+)")
_CLAUSE_PREFIX = re.compile(r"(?:[-*•]\s+|\d+[.)]\s+)")
_EXPLICIT_CORRECTION = re.compile(
    r"\b(?:correction|correcting|instead|actually|no\s+longer|change\s+of\s+plan|replace)\b",
    re.IGNORECASE,
)
_BARE_CORRECTION_HEADER = re.compile(
    r"^\s*(?:correction|correcting|change\s+of\s+plan|instead|actually|replace)\s*:?[\s-]*$",
    re.IGNORECASE,
)
_STRONG_CORRECTION = re.compile(
    r"\b(?:correction|correcting|instead|no\s+longer|change\s+of\s+plan|replace)\b",
    re.IGNORECASE,
)
_DIRECTIVE_PREFIX = re.compile(
    r"^\s*(?:(?:actually|correction|correcting)\s*:?,?\s*)?(?:please\s+)?"
    r"(?:use|target|return|switch|change|make|build|create|add|remove|drop|supersede|mark|write|"
    r"output|implement|fix(?:ing)?|test|verify|lint|format|stage|revert|solve|clean(?:\s+up)?|"
    r"refactor|modif(?:y|ying))\b(?!\s*:)",
    re.IGNORECASE,
)
_ACTUALLY_PREFIX = re.compile(r"^\s*actually\b", re.IGNORECASE)
_REVERSAL_LANGUAGE = re.compile(
    r"\b(?:allowed|permitted|forbidden|breaking\s+changes?|can\s+(?:change|modify|break)|"
    r"may\s+(?:change|modify|break))\b",
    re.IGNORECASE,
)
_ADDITIVE_LANGUAGE = re.compile(r"\b(?:also|too|in\s+addition|as\s+well)\b", re.IGNORECASE)
_FENCE_START = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\r\n]*$")
_LAZY_QUOTE_INTERRUPT = re.compile(
    r"^ {0,3}(?:[-+*]\s+|\d{1,9}[.)]\s+|#{1,6}(?:\s+|$)|(?:[*_-]\s*){3,}$|<[^>]+>)"
)
_PLAIN_QUOTED_DATA = (
    re.compile(r'"(?:\\.|[^"\\\r\n])*"'),
    re.compile(r"“[^”\r\n]*”"),
    re.compile(r"‘[^’\r\n]*’"),
)
_QUESTION_PREFIX = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|should|would|could|can|why|how|what|when|where|who)\b",
    re.IGNORECASE,
)
_COUNTERFACTUAL_QUESTION = re.compile(
    r"^\s*if\s+you\s+(?:were\s+to|could|would)\b[^?\n]{0,160}\b(?:what|how|which|would)\b",
    re.IGNORECASE,
)
_POLITE_DIRECTIVE_QUESTION = re.compile(
    r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:ensure|make\s+sure|keep|preserve|"
    r"never|do\s+not|don't|use|target|return|implement|fix(?:ing)?|test|verify|lint|format|stage|revert|"
    r"solve|clean(?:\s+up)?|add|remove|drop|supersede|mark|write|change|modify)\b",
    re.IGNORECASE,
)

# Turn-contract analysis asks only two control-plane questions: whether this exact turn authorizes an effect,
# and whether its answer is about the sealed past or live present.  It is intentionally not a general NLU
# taxonomy.  High-confidence directives pass; questions/corrections/attributed prior speech do not; anything
# else stays ``uncertain`` for the execution gate to handle conservatively.
_EFFECT_VERB_SRC = (
    r"implement|fix(?:ing)?|patch|repair|resolve|address|correct|edit|change|modify|refactor|rewrite|"
    r"replace|simplify|improve|optimize|add|remove|drop|supersede|mark|delete|create|write|build|apply|make|ensure|"
    r"test|verify|lint|format|stage|revert|solve|clean(?:\s+up)?|complete|finish|"
    r"install|configure|update|upgrade|migrate|rename|move|copy|run|execute|launch|spawn|delegate|start|stop|close|terminate|"
    r"kill|deploy|publish|commit|push|switch|resume|new|revise|target|use"
)
_DELEGATION_DIRECTIVE = re.compile(
    r"(?:^|[:;.!?]\s+|,\s+|\b(?:then|and\s+then)\s+)\s*"
    r"(?:(?:now|please|just)\s+)*(?P<verb>spawn|launch|delegate)\b"
    r"(?=[^.!?;\n]{0,180}\b(?:subagents?|child\s+agents?|explorers?|agents?)\b)",
    re.IGNORECASE,
)
_EFFECT_DIRECTIVE = re.compile(
    rf"(?:^|[.!?;]\s+|\n+|,\s+|\b(?:then|and\s+then)\s+)"
    rf"\s*(?:(?:yes|yeah|yep|sure|okay|ok|no)\s*,?\s*)?"
    rf"(?:(?:actually|correction|correcting|instead)\s*:?,?\s*)?"
    rf"(?:(?:i\s+(?:want|need)\s+you\s+to|let'?s(?:\s+go\s+ahead(?:\s+and)?)?|"
    rf"feel\s+free\s+to|you\s+can|why\s+don'?t\s+you|would\s+you\s+mind)\s+)?"
    rf"(?:(?:now|please|just|go\s+ahead(?:\s+and)?)\s+)*"
    rf"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    rf"(?:{_EFFECT_VERB_SRC})\b"
    rf"(?!\s*:|\s+(?:is|are|was|were|has|have|had|looks?|seems?|fails?|failed|"
    rf"succeeds?|succeeded|works?|worked)\b)"
    rf"(?!\s+(?:(?:me\s+)?(?:a|an|the)\s+)?(?:summary|explanation|answer|response|"
    rf"message|report|review|list)\b)",
    re.IGNORECASE,
)
_DO_CHANGE_DIRECTIVE = re.compile(
    r"^\s*(?:(?:yes|yeah|sure|okay|ok)\s*,?\s*)?"
    r"(?:(?:please\s+)?go\s+ahead(?:\s+and)?\s+|let'?s(?:\s+go\s+ahead(?:\s+and)?)?\s+)"
    r"do\s+.{0,120}\b(?:fix|patch|upgrade|implementation|change|refactor|rewrite|migration|"
    r"cleanup|simplification|improvement|optimization)\b",
    re.IGNORECASE,
)
# Natural workspace navigation is an effect directive even when the user does not happen to say the
# implementation-shaped verb "switch". Keep this separate from the generic effect-verb list: adding bare
# ``go``/``work`` there would incorrectly make phrases such as "go over the report" effectful. The target must
# visibly name a workspace/project/repository/folder or an absolute/home path.
_WORKSPACE_NAVIGATION_DIRECTIVE = re.compile(
    r"(?:^|[.!?;]\s+|\n+|,\s+|\b(?:then|and\s+then)\s+)\s*"
    r"(?:(?:yes|yeah|yep|sure|okay|ok)\s*,?\s*)?(?:please\s+)?"
    r"(?:(?:go|navigate|move)\s+(?:to|into)\s+|(?:open|enter)\s+|work\s+in\s+)"
    r"(?=[^.!?;\n]{0,180}(?:\b(?:workspace|project|repo(?:sitory)?|directory|folder)\b|"
    r"(?:~[/\\]|/|[A-Za-z]:[/\\])))",
    re.IGNORECASE,
)
_BARE_WORKSPACE_NAVIGATION_DIRECTIVE = re.compile(
    r"(?:^|[.!?;]\s+|\n+|,\s+)\s*(?:please\s+)?"
    r"(?:(?:go|navigate|move)\s+(?:to|into)\s+|work\s+in\s+)"
    r"(?:the\s+)?[A-Za-z0-9_.-]+\s*[.!?]*$",
    re.IGNORECASE,
)
_NEGATED_EFFECT = re.compile(
    rf"^\s*(?:please\s+)?(?:do\s+not|don't|never|no\s+need\s+to)\s+(?:{_EFFECT_VERB_SRC})\b"
    rf"|\b(?:did\s+not|didn't|do\s+not|don't|am\s+not|not)\s+(?:ask(?:ing|ed)?\s+you\s+to|"
    rf"authoriz(?:e|ed|ing)\s+you\s+to)\s+(?:{_EFFECT_VERB_SRC})\b",
    re.IGNORECASE,
)
_BARE_ASSENT = re.compile(
    r"^\s*(?:anyways?\s*[,;:]?\s*)?(?:"
    r"(?:yes|yeah|yep|yup|sure|okay|ok)(?:\s+please|\s*,\s*(?:go\s+ahead|do\s+it|"
    r"please(?:\s+switch\s+it)?))|"
    r"yes|yeah|yep|yup|sure|okay|ok|go\s+ahead|"
    r"go|continue|proceed|please\s+continue|keep\s+going|"
    r"do\s+it|do\s+that|please\s+do|sounds\s+good|"
    r"(?:go\s+with|choose|pick|take|let'?s\s+do)\s+(?:option\s+)?\d+|"
    r"(?:option\s+)?\d+)\s*[.!]*\s*$",
    re.IGNORECASE,
)
_OPTION_SELECTION = re.compile(
    r"(?:go\s+with|choose|pick|take|let'?s\s+do|option)\s+(?:option\s+)?(\d{1,3})\b"
    r"|^\s*(\d{1,3})\s*[.!]*\s*$",
    re.IGNORECASE,
)
_CONFIRMATION_OR_CORRECTION = re.compile(
    r"^\s*(?:no\b|yes\s*[,;:]\s*\S|correct\b|exactly\b|right\b|that(?:'s|\s+is)\b|"
    r"this(?:'s|\s+is)\b|i\s+(?:said|meant|mean)\b|to\s+confirm\b|just\s+confirm\b|"
    r"only\s+confirm\b|correction\b)",
    re.IGNORECASE,
)
_ANSWER_ONLY_DIRECTIVE = re.compile(
    r"^\s*(?:please\s+)?(?:explain|tell\s+me|give\s+me|list|summarize|confirm|show|compare|review|"
    r"inspect|check|describe|identify|locate|find|return|output|respond\b|"
    r"write\s+(?:(?:me\s+)?(?:a|an|the)\s+)?(?:summary|explanation|answer|response|"
    r"message|report|review|list)\b)",
    re.IGNORECASE,
)
_ANSWER_PRODUCTION_DIRECTIVE = re.compile(
    r"^\s*(?:please\s+)?(?:use|target|make|build|add|write|create)\b[^\n]{0,180}"
    r"(?:\b(?:table|bullets?|diagram|mental\s+model|citations?|code\s+examples?|pseudocode|"
    r"recommendation)\b|"
    r"\b(?:in|for)\s+(?:your|the)\s+(?:answer|response|explanation)\b|\bto\s+explain\b)",
    re.IGNORECASE,
)
_HISTORICAL_REFERENCE = re.compile(
    r"\b(?:you\s+(?:said|wrote|listed|reported|found|mentioned|described|told\s+me)|"
    r"what\s+(?:did|have)\s+you\s+(?:say|write|list|report|find|mention)|"
    r"(?:your|the)\s+(?:original|earlier|previous|prior)\s+"
    r"(?:findings?|report|review|list|answer|response|message)|"
    r"(?:in|from)\s+(?:your|the)\s+(?:(?:original|earlier|previous|prior)\s+)?"
    r"(?:findings?|report|review|list|answer|response|message)|"
    r"confirm\s+(?:(?:item|bug|finding|issue|number)\s*)?#?\s*\d+|"
    r"(?:confirm\s+)?(?:number\s+)?#?\s*\d+\s*(?:=|\bis\b|\bwas\b)|"
    r"what\s+was\s+(?:item|bug|finding|issue)\s*#?\s*\d+|"
    r"(?:item|bug|finding|issue)\s*#\s*\d+)\b",
    re.IGNORECASE,
)
_LIVE_REFERENCE = re.compile(
    r"\b(?:current(?:ly)?|live|workspace|working\s+tree|file|code|repo(?:sitory)?|project|"
    r"tests?|process|service|installed|running|latest\s+version)\b",
    re.IGNORECASE,
)
_EXPLICIT_LIVE_TIME = re.compile(
    r"\b(?:now|right\s+now|current(?:ly)?|still|today|live|present(?:ly)?)\b",
    re.IGNORECASE,
)
_ATTRIBUTION_INTRO = re.compile(
    r"\b(?:"
    r"for\s+reference"
    r"|here(?:'s|\s+is)\s+what\s+you\s+(?:said|wrote|listed|reported)"
    r"|(?:as\s+)?you\s+(?:said|wrote|listed|reported|found|mentioned|described|told\s+me)"
    r"|(?:in|from)\s+(?:your|the)\s+(?:(?:original|earlier|previous|prior)\s+)?"
    r"(?:findings?|report|review|list|answer|response|message)"
    r"|(?:your|the)\s+(?:(?:original|earlier|previous|prior)\s+)?"
    r"(?:findings?|report|review|list|answer|response|message)\s+"
    r"(?:says?|said|lists?|listed|reads?|read|was|is)"
    r")\b"
    r"(?:\s*,?\s*(?:it|item\s*#?\s*\d+)\s+(?:is|was|says?|said|reads?|read))?"
    r"\s*(?:that\b)?\s*(?::|—|-)?\s*",
    re.IGNORECASE,
)
_REFERENCE_EQUATION = re.compile(
    r"(?:^|[.!?;]\s+|\n+)\s*(?:(?:right|correct|exactly|yes|no)\s*,\s*)?"
    r"(?:(?:item|bug|finding|issue|number)\s*)?#?\s*\d+\s*(?:=|\bis\b|\bwas\b)\s*",
    re.IGNORECASE,
)
_REFERENCE_CONFIRMATION_TAIL = re.compile(
    r"[.!?;]\s+(?=(?:that(?:'s|\s+is)\s+(?:the\s+)?(?:gist|finding|issue|one)|"
    r"(?:is\s+)?that\s+(?:right|correct)|right\b|correct\b|yes\b|no\b))",
    re.IGNORECASE,
)
_NEW_AUTHORITY_AFTER_ATTRIBUTION = re.compile(
    rf"(?:[.!?;]\s+|\n+)"
    rf"(?=(?:(?:but|and)\s+)?(?:(?:now|then|please|just|go\s+ahead(?:\s+and)?)\s+)*"
    rf"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    rf"(?:(?:{_EFFECT_VERB_SRC})\b(?!\s*:)|"
    r"explain\b|tell\s+me\b|list\b|summarize\b|confirm\b|show\b|compare\b|review\b|"
    r"what\b|why\b|how\b|which\b|who\b|where\b))",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[._/:+#-][a-z0-9]+)*", re.IGNORECASE)
_PAST_USER_UTTERANCE = re.compile(
    r"\bwhat\s+did\s+i\s+(?:say|ask|write|tell\s+you)|\bmy\s+(?:earlier|previous|prior)\s+(?:request|message)\b",
    re.IGNORECASE,
)
_PAST_ASSISTANT_UTTERANCE = re.compile(
    r"\b(?:what\s+(?:you\s+)?(?:just\s+)?said|you\s+(?:just\s+)?said|"
    r"your\s+(?:earlier|previous|prior|last)\s+(?:answer|response|message|claim))\b",
    re.IGNORECASE,
)
_PAST_EXECUTION = re.compile(
    r"\b(?:how\s+many\b[^?\n]{0,80}\b(?:spawn(?:ed)?|launch(?:ed)?|ran|run|start(?:ed)?)|"
    r"how\s+many\b[^?\n]{0,80}\b(?:tools?|agents?|children|reports?)\b[^?\n]{0,80}"
    r"\b(?:use|used|return|returned|seal|sealed|finish|finished|came\s+back)|"
    r"how\s+many\b[^?\n]{0,80}\b(?:files?|reads?|edits?|writes?|patches?)\b[^?\n]{0,80}"
    r"\b(?:read|edit|write|patch|fail|succeed)|"
    r"did\s+(?:you|any|it|they)\s+(?:spawn|launch|run|start|fail|succeed|finish)|"
    r"did\b[^?\n]{0,80}\b(?:commands?|tools?|agents?|explorers?|children|reports?|files?|edits?|writes?)\b[^?\n]{0,80}"
    r"\b(?:run|execut(?:e|ed)|start|fail|succeed|finish|return|seal)|"
    r"were\s+any\b[^?\n]{0,80}\b(?:successful|failed|started|returned)|"
    r"why\s+did\b[^?\n]{0,80}\b(?:fail|reject|cancel)|"
    r"which\b[^?\n]{0,80}\bfailed|"
    r"what\s+(?:did\s+you\s+do|actually\s+happened|failed\s+in\s+the\s+(?:last|latest)\s+attempt)|"
    r"what\s+(?:actually\s+)?(?:ran|failed|succeeded|finished)|"
    r"(?:spawn|launch|run|tool|command|subagents?)\s+(?:count|status|result))\b",
    re.IGNORECASE,
)
_EVIDENCE_STRONG_DELEGATION_FAMILY = re.compile(
    r"\b(?:subagents?|child(?:ren|\s+agents?)?|explorers?|reports?|"
    r"delegat(?:e|ed|ion)|spawn(?:ed|ing)?|agents)\b",
    re.IGNORECASE,
)
_EVIDENCE_COMMAND_FAMILY = re.compile(
    r"\b(?:commands?|shell|terminal|process(?:es)?|execut(?:e|ed|ion))\b",
    re.IGNORECASE,
)
_EVIDENCE_BARE_RUN_FAMILY = re.compile(r"\b(?:ran|run)\b", re.IGNORECASE)
_EVIDENCE_FILE_FAMILY = re.compile(
    r"\b(?:files?|file\s+tools?|reads?|edits?|writes?|patch(?:es|ed)?|grep(?:ped)?)\b",
    re.IGNORECASE,
)
_EVIDENCE_FILE_WRITE_FAMILY = re.compile(
    r"\b(?:edit(?:ed|s|ing)?|writ(?:e|es|ten|ing)|patch(?:es|ed|ing)?|"
    r"modif(?:y|ies|ied|ying)|chang(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_EVIDENCE_FILE_READ_FAMILY = re.compile(
    r"\b(?:read|reads|reading)\b",
    re.IGNORECASE,
)
_EVIDENCE_AGGREGATE_PREDICATE = re.compile(
    r"\b(?:how\s+many|count|total|number\s+of|did\s+any|were\s+any|any\s+of\s+them|"
    r"all\s+(?:succeed|successful|failed))\b",
    re.IGNORECASE,
)
_EVIDENCE_FAILURE_DETAIL_PREDICATE = re.compile(
    r"\b(?:why\b[^?\n]{0,100}\b(?:fail|reject|cancel)|reason(?:s)?\b[^?\n]{0,80}"
    r"\b(?:fail|reject|cancel)|what\s+failed|which\b[^?\n]{0,80}\bfailed|"
    r"own\s+up\s+to\b[^?\n]{0,100}\bfailures?|(?:any|your)\s+failures?|"
    r"what\s+(?:went|has\s+gone)\s+(?:badly|wrong))\b",
    re.IGNORECASE,
)
_EVIDENCE_LATEST_TURN_SCOPE = re.compile(
    r"\b(?:last|latest|previous|most\s+recent)\s+turn\b",
    re.IGNORECASE,
)
_EVIDENCE_LATEST_EXECUTION_SCOPE = re.compile(
    r"\b(?:(?:last|latest|previous|most\s+recent)\s+(?:attempt|run|execution)|last\s+time)\b",
    re.IGNORECASE,
)
_EVIDENCE_SESSION_SCOPE = re.compile(
    r"\b(?:this|current|the)\s+session\b|\bacross\s+(?:this|the)\s+session\b",
    re.IGNORECASE,
)
_SELF_PERFORMANCE_EVIDENCE = re.compile(
    r"(?:\bown\s+up\s+to\b[^?\n]{0,100}\bfailures?|"
    r"^\s*any\s+failures?\b[^?\n]{0,80}[?.!]*\s*$|"
    r"\b(?:did\s+you\s+have|were\s+there)\s+any\s+failures?|"
    r"\bwhat\s+(?:were|are)\s+your\s+failures?|"
    r"^\s*what\s+(?:went|has\s+gone)\s+(?:badly|wrong)\s*[?.!]*\s*$|"
    r"(?:your|your\s+own|own)\s+performance\b[^?\n]{0,160}\b(?:accurate|verify|records?)|"
    r"verify\s+(?:it|that|this)\s+against\s+your\s+records?)",
    re.IGNORECASE,
)
_SELF_AGENT_CONTEXT = re.compile(r"\b(?:as\s+an?\s+agent|yourself\s+as\s+an?\s+agent)\b", re.IGNORECASE)
_SELF_AUDIT = re.compile(
    r"\b(?:improve\s+yourself|how\s+(?:could|would|should)\s+you\s+improve|"
    r"reflect\s+on\s+(?:your|your\s+own|own)\s+performance|"
    r"(?:your|your\s+own|own)\s+performance|weaknesses?\s+as\s+an?\s+agent|"
    r"own\s+up\s+to\b[^?\n]{0,100}\bfailures?|"
    r"(?:critique|audit|review|assess)\s+(?:the\s+quality\s+of\s+)?"
    r"(?:your|your\s+(?:last|previous|prior)|the\s+(?:last|previous|prior))\s+"
    r"(?:answer|response|message)|"
    r"(?:did|have)\s+you\s+(?:follow|satisfy|obey|answer)\b[^?\n]{0,100}"
    r"\b(?:my\s+)?(?:instructions?|request|constraints?|question)|"
    r"(?:your|the)\s+response\s+quality)\b",
    re.IGNORECASE,
)
_GENERIC_EVIDENCE_VERIFICATION = re.compile(
    r"(?:\b(?:verify|check|validate|confirm)\s+"
    r"(?:that|it|this|what\s+you\s+(?:just\s+)?said|your\s+(?:last|previous)\s+answer)\s+"
    r"against\s+(?:the\s+|your\s+)?(?:records?|receipts?|history)\b|"
    r"^\s*are\s+you\s+sure(?:\s+(?:about\s+)?(?:that|this|your\s+(?:last|previous)\s+answer))?"
    r"\s*[?.!]*\s*$|"
    r"^\s*is\s+(?:that|this|what\s+you\s+(?:just\s+)?said|your\s+(?:last|previous)\s+answer)\s+"
    r"(?:accurate|correct|right)\s*[?.!]*\s*$|"
    r"\b(?:double[- ]check|recheck)\s+(?:that|this|your\s+(?:last|previous)\s+(?:answer|claim))\b|"
    r"\bcheck\s+(?:your|that)\s+(?:claim|answer)\s+against\s+"
    r"(?:the\s+)?(?:records?|receipts?|history)\b|"
    r"\bwere\s+there\s+really\b[^?\n]{0,100}\?\s*check\s+"
    r"(?:the\s+)?(?:records?|receipts?|history)\b)",
    re.IGNORECASE,
)
_PROSPECTIVE_SELF_IMPROVEMENT = re.compile(
    r"\b(?:how\s+(?:could|would|should)\s+you\s+improve|"
    r"if\s+you\s+were\s+to\s+improve\b[^?\n]{0,120}\bwhat\s+would\s+you\s+do|"
    r"what\s+would\s+you\s+do\b[^?\n]{0,120}\bto\s+improve|"
    r"what\s+would\s+you\s+(?:change|do\s+differently)|"
    r"you(?:'d|\s+would)\s+(?:fix|change)\b|"
    r"ways?\s+(?:you\s+)?(?:could|would|to)\s+improve|"
    r"suggest\s+(?:future|prospective)\s+improvements?)\b",
    re.IGNORECASE,
)


def derive_evidence_query(request: str) -> EvidenceQuery | None:
    """Compile natural execution-recall language into one typed selector.

    This is the sole lexical owner for receipt family, projection predicate, and temporal scope. Downstream
    selectors/renderers must consume the returned fields and must not reinterpret the user's wording.
    """
    source = str(request or "")
    self_audit = _SELF_AUDIT.search(source) is not None
    if (_PAST_EXECUTION.search(source) is None
            and _SELF_PERFORMANCE_EVIDENCE.search(source) is None and not self_audit):
        return None
    family = "all"
    # A request may mention more than one family. In that case ``all`` is safer than silently dropping one.
    families = []
    delegation_family = (_EVIDENCE_STRONG_DELEGATION_FAMILY.search(source)
            and not (_SELF_AGENT_CONTEXT.search(source)
                     and re.search(r"\b(?:subagents?|children|explorers?|spawn|delegat)\b", source, re.I)
                     is None))
    file_family = _EVIDENCE_FILE_FAMILY.search(source) is not None
    if delegation_family:
        families.append("delegation")
    # Bare "ran/run" describes the lifecycle verb in questions such as "how many child agents ran"; it is not
    # evidence that the user also asked about shell commands. Treat it as the command family only when no stronger
    # operation-family noun is present. This keeps the projection aligned with the user's requested subject.
    if (_EVIDENCE_COMMAND_FAMILY.search(source)
            or (_EVIDENCE_BARE_RUN_FAMILY.search(source) and not delegation_family and not file_family)):
        families.append("command")
    if file_family:
        file_write = _EVIDENCE_FILE_WRITE_FAMILY.search(source) is not None
        file_read = _EVIDENCE_FILE_READ_FAMILY.search(source) is not None
        families.append("file_write" if file_write and not file_read else
                        "file_read" if file_read and not file_write else "file")
    if len(set(families)) == 1:
        family = families[0]
    predicate = (
        "failure_detail" if _EVIDENCE_FAILURE_DETAIL_PREDICATE.search(source) else
        "aggregate" if _EVIDENCE_AGGREGATE_PREDICATE.search(source) or self_audit else
        "operations"
    )
    scope = (
        "latest_turn" if _EVIDENCE_LATEST_TURN_SCOPE.search(source) else
        "latest_matching_execution" if _EVIDENCE_LATEST_EXECUTION_SCOPE.search(source) else
        "session" if _EVIDENCE_SESSION_SCOPE.search(source) else
        "task"
    )
    return EvidenceQuery(
        source="execution_receipt", family=family, predicate=predicate, scope=scope,
    )


def derive_quality_evidence_query(request: str) -> QualityEvidenceQuery | None:
    """Compile an explicit self-assessment into a paired utterance selector.

    This reducer only sees the already-unquoted, user-authored scan when called by :func:`analyze_turn`.
    Prospective advice is opt-in: a request for weaknesses or for what went wrong does not itself license a
    hypothetical preference to be presented as an observed defect.
    """
    source = str(request or "")
    if _SELF_AUDIT.search(source) is None:
        return None
    latest = re.search(
        r"\b(?:last|previous|prior|most\s+recent)\s+(?:answer|response|message)\b|"
        r"\bdid\s+you\s+(?:follow|satisfy|obey|answer)\b",
        source, re.IGNORECASE,
    ) is not None
    return QualityEvidenceQuery(
        scope=("latest_response" if latest else
               "session" if _EVIDENCE_SESSION_SCOPE.search(source) else "task"),
        prospective_requested=_PROSPECTIVE_SELF_IMPROVEMENT.search(source) is not None,
    )
_FILE_TARGET = re.compile(
    r"`([^`\r\n]+)`|"
    r"[\"“‘]([^\"”’\r\n]+)[\"”’]|"
    r"((?:[A-Za-z0-9_.-]+[/\\])*\.[A-Za-z0-9_.-]+)|"
    r"((?:~[/\\]|/|[A-Za-z]:[/\\])[^\s,;!?]+)|"
    r"\b((?:[A-Za-z0-9_.-]+[/\\])*[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+)\b|"
    r"\b((?:README|CHANGELOG|LICENSE|Makefile|Dockerfile|Procfile|Gemfile|Jenkinsfile|"
    r"Vagrantfile|Brewfile)(?:\.[A-Za-z0-9_.-]+)?)\b",
    re.IGNORECASE,
)
_WORKSPACE_TARGET = re.compile(
    r"(?:go|navigate|move|switch|change)\s+(?:to|into)\s+(?:the\s+)?([A-Za-z0-9_.-]+)\s+"
    r"(?:workspace|project|repo(?:sitory)?|directory|folder)\b|"
    r"(?:switch|change)\s+(?:the\s+)?(?:workspace|project|repo(?:sitory)?|directory|folder)\s+"
    r"(?:to\s+)?([A-Za-z0-9_.-]+)\b|"
    r"switch\s+(?:to\s+)?(?:the\s+)?([A-Za-z0-9_.-]+)\s*$|"
    r"(?:open|enter|work\s+in)\s+(?:the\s+)?([A-Za-z0-9_.-]+)\s+"
    r"(?:workspace|project|repo(?:sitory)?|directory|folder)\b",
    re.IGNORECASE,
)

_WORKSPACE_EDIT_TOOLS = ("edit_file", "append_to_file", "str_replace")
_TASK_MAINTENANCE_TOOLS = ("update_plan",)

# Linkage ignores grammatical/modal scaffolding, but deliberately does not stem words or compare embeddings.
# The fallback below therefore remains deterministic and conservative: an explicit correction must retain at
# least two content-bearing words and cover most of both the old clause and its captured replacement.
_LINKAGE_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "we", "us", "our", "you", "your", "it", "its",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "to", "of", "for", "in", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "if", "then",
    "must", "should", "shall", "may", "might", "can", "could", "would",
    "not", "no", "never", "longer",
    "actually", "correction", "correcting", "instead", "change", "plan", "replace",
    "require", "required", "requires",
    "preserve", "keep", "ensure", "make", "sure", "leave", "only", "use", "target",
})


def _phrase_tokens(text: str) -> tuple[str, ...]:
    """Punctuation-insensitive exact-word projection used only for explicit correction linkage."""
    return tuple(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def _contains_phrase(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[index:index + len(needle)] == needle
               for index in range(len(haystack) - len(needle) + 1))


def _significant_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token for token in _phrase_tokens(text)
        if len(token) >= 3 and token not in _LINKAGE_STOPWORDS
    )


def _concrete_values(text: str) -> frozenset[str]:
    """Concrete numeric/version values that make a one-subject replacement unambiguous."""
    return frozenset(re.findall(r"\d+(?:\.\d+)+|\d+", str(text or "").casefold()))


def _directive_action(text: str) -> str:
    """Normalize only an explicit leading action verb; this is not semantic/fuzzy matching."""
    cleaned = re.sub(
        r"^\s*(?:(?:actually|correction|correcting)\s*:?,?\s*)?"
        r"(?:(?:you\s+)?(?:must|should|shall)\s+)?(?:do\s+not\s+|never\s+)?",
        "", str(text or ""), flags=re.IGNORECASE,
    )
    match = re.match(
        r"(use|target|return|switch|change|make|build|create|add|remove|write|output|implement|fix|"
        r"refactor|modif(?:y|ying))\b",
        cleaned, re.IGNORECASE,
    )
    if match is None:
        return ""
    action = match.group(1).casefold()
    return "modify" if action in {"modify", "modifying"} else action


def _correction_match_score(old_text: str, replacement_text: str) -> int:
    """Rank a deterministic correction link; callers reject ties instead of guessing.

    Exact word-sequence containment remains the strongest path. The fallback is set identity after removing
    grammatical/correction scaffolding, a shared subject plus changed concrete value, or a parallel leading
    action with a shared subject. This is called only after the explicit-correction cue gate.
    """
    if _ADDITIVE_LANGUAGE.search(replacement_text):
        return 0
    old_words = _phrase_tokens(old_text)
    replacement_words = _phrase_tokens(replacement_text)
    if _contains_phrase(replacement_words, old_words):
        return 1000
    old_significant = _significant_tokens(old_text)
    replacement_significant = _significant_tokens(replacement_text)
    if len(old_significant) >= 2 and old_significant == replacement_significant:
        return 900
    old_values = _concrete_values(old_text)
    replacement_values = _concrete_values(replacement_text)
    if (
        len(old_significant) == 1
        and old_significant == replacement_significant
        and bool(old_values and replacement_values and old_values != replacement_values)
    ):
        return 850
    old_action = _directive_action(old_text)
    replacement_action = _directive_action(replacement_text)
    if old_action and old_action == replacement_action:
        old_subject = set(old_significant)
        replacement_subject = set(replacement_significant)
        for action_token in ({"modify", "modifying"} if old_action == "modify" else {old_action}):
            old_subject.discard(action_token)
            replacement_subject.discard(action_token)
        shared = old_subject.intersection(replacement_subject)
        return 600 + len(shared) if shared else 0
    return 0


def _match_key(text: str) -> str:
    """Compatibility matcher for model-supplied requirement text.

    The first spelling is retained verbatim.  Matching remains whitespace- and
    case-insensitive so the existing requirement_done/drop tool contract does
    not become brittle during migration.
    """
    return " ".join(str(text or "").split()).casefold()


def _merge_spans(spans: Iterable[SourceSpan], length: int) -> tuple[SourceSpan, ...]:
    """Clamp, sort and merge exact half-open source ranges."""
    clean = sorted(
        (max(0, start), min(length, end)) for start, end in spans
        if isinstance(start, int) and isinstance(end, int) and start < length and end > 0 and end > start
    )
    merged: list[list[int]] = []
    for start, end in clean:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged if end > start)


def _complement_spans(text: str, excluded: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    out = []
    cursor = 0
    for start, end in _merge_spans(excluded, len(text)):
        if cursor < start and text[cursor:start].strip():
            left, right = cursor, start
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
            if left < right:
                out.append((left, right))
        cursor = max(cursor, end)
    if cursor < len(text) and text[cursor:].strip():
        left, right = cursor, len(text)
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            out.append((left, right))
    return tuple(out)


def _scan_spans(text: str, spans: Iterable[SourceSpan]) -> str:
    """Same-length projection that exposes only ``spans`` while preserving line boundaries."""
    source = str(text or "")
    visible = [False] * len(source)
    for start, end in _merge_spans(spans, len(source)):
        visible[start:end] = [True] * (end - start)
    return "".join(
        char if (visible[index] or char in "\r\n") else " "
        for index, char in enumerate(source)
    )


def _spans_overlap(left: SourceSpan, right: SourceSpan) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _mask_quoted_data_with_spans(text: str) -> tuple[str, tuple[SourceSpan, ...]]:
    """Same-length instruction scan with Markdown data regions blanked.

    Source ranges still index the untouched request. Fenced code, blockquotes, and inline code are evidence
    supplied by the user, not automatically promoted into higher-authority standing directives.
    """
    source = str(text or "")
    masked = list(source)
    spans: list[SourceSpan] = []
    fence_char = ""
    fence_size = 0
    lazy_quote = False

    def blank(start: int, end: int) -> None:
        spans.append((start, end))
        for index in range(start, end):
            if masked[index] not in "\r\n":
                masked[index] = " "

    offset = 0
    for line in source.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        end = offset + len(line)
        if fence_char:
            blank(offset, end)
            # CommonMark closing fences use the same character, at least the opening length, up to three
            # leading spaces, and no info string. `````not-a-close`` therefore remains fenced data.
            if re.fullmatch(rf" {{0,3}}{re.escape(fence_char)}{{{fence_size},}}[ \t]*", body):
                fence_char = ""; fence_size = 0
            offset = end
            continue
        opened = _FENCE_START.fullmatch(body)
        if opened is not None:
            marker = opened.group(1)
            fence_char, fence_size = marker[0], len(marker)
            blank(offset, end)
            lazy_quote = False
            offset = end
            continue

        stripped = body.lstrip(" ")
        indent = len(body) - len(stripped)
        is_quote = indent <= 3 and stripped.startswith(">")
        if is_quote:
            blank(offset, end)
            lazy_quote = bool(stripped[1:].strip())
            offset = end
            continue
        if lazy_quote:
            if not body.strip():
                lazy_quote = False
            elif _LAZY_QUOTE_INTERRUPT.match(body):
                # Lists/headings/thematic/HTML block starts interrupt a CommonMark lazy paragraph. They are
                # outside the quote and may carry a real standing directive, so resume normal scanning.
                lazy_quote = False
            else:
                # A blockquote paragraph may continue lazily on following nonblank lines without another `>`.
                blank(offset, end)
                offset = end
                continue
        if body.startswith("    ") or body.startswith("\t"):
            blank(offset, end)  # indented Markdown code block
        offset = end

    # Code spans may legally cross line boundaries. Match a closing run of exactly the opening length;
    # a longer adjacent run is a different delimiter. Fenced/quoted ranges are already spaces here.
    scan = "".join(masked)
    index = 0
    while index < len(scan):
        if scan[index] != "`":
            index += 1
            continue
        opened = index
        while index < len(scan) and scan[index] == "`":
            index += 1
        width = index - opened
        cursor = index
        closed = -1
        while cursor < len(scan):
            cursor = scan.find("`", cursor)
            if cursor < 0:
                break
            finish = cursor
            while finish < len(scan) and scan[finish] == "`":
                finish += 1
            if finish - cursor == width:
                closed = finish
                break
            cursor = finish
        if closed < 0:
            continue
        blank(opened, closed)
        scan = "".join(masked)
        index = closed
    # Ordinary quoted strings are also attributed data. Mask only the payload; an enclosing directive such
    # as `Return exactly "OK"` remains capturable from the unquoted action words and retains exact source text.
    scan = "".join(masked)
    for pattern in _PLAIN_QUOTED_DATA:
        for match in pattern.finditer(scan):
            blank(match.start(), match.end())
        scan = "".join(masked)
    return "".join(masked), _merge_spans(spans, len(source))


def _operative_quoted_argument_spans(text: str, spans: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    """Return quote spans that are direct arguments of a present user directive.

    Quotes normally carry data, but ``edit \"README.md\"`` and ``run `pytest` `` are not reported speech.
    Keeping this distinction here prevents the dangerous failure mode where masking the target leaves an
    apparently broad, targetless ``edit`` capability.
    """
    source = str(text or "")
    arguments = []
    for start, end in spans:
        value = source[start:end]
        if not value or "\n" in value or value[0] not in "\"“‘`":
            continue
        boundary = max(source.rfind(mark, 0, start) for mark in (".", "!", "?", ";", "\n")) + 1
        prefix = source[boundary:start]
        suffix = source[end:min(len(source), end + 40)]
        directive = list(_EFFECT_DIRECTIVE.finditer(prefix))
        direct_argument = False
        if directive:
            gap = prefix[directive[-1].end():]
            direct_argument = re.fullmatch(
                r"[ \t]*(?:(?:the|a|an|this|that|named|called|exactly|file|path|command|process|"
                r"workspace|project|repo(?:sitory)?|directory|folder|requirement|constraint|to|into|in)"
                r"[ \t]*)*",
                gap, re.IGNORECASE,
            ) is not None
            if re.search(r"\b(?:supersede|replace)\b", prefix[:directive[-1].end()], re.I):
                direct_argument = True
        navigation_argument = bool(
            re.search(r"\b(?:go|navigate|move)\s+(?:to|into)\s+(?:the\s+)?$", prefix, re.I)
            and re.match(r"\s*(?:workspace|project|repo(?:sitory)?|directory|folder)\b", suffix, re.I)
        )
        if direct_argument or navigation_argument:
            arguments.append((start, end))
    return tuple(arguments)


def _mask_quoted_data(text: str) -> str:
    """Compatibility projection used by the existing correction/constraint reducer."""
    return _mask_quoted_data_with_spans(text)[0]


def _copied_prior_spans(text: str, prior_texts: Iterable[str]) -> tuple[SourceSpan, ...]:
    """Locate substantial verbatim-ish token runs copied from earlier assistant output.

    Whitespace and terminal wrapping may differ, so compare token sequences and project matches back to exact
    offsets in the untouched request.  The threshold is deliberately high: a common three-word phrase is not
    enough to strip authority from a new user instruction.
    """
    request_tokens = [(match.group(0).casefold(), match.start(), match.end()) for match in _TOKEN.finditer(text)]
    if len(request_tokens) < 6:
        return ()
    request_values = [token for token, _start, _end in request_tokens]
    spans = []
    for prior in prior_texts or ():
        prior_values = [match.group(0).casefold() for match in _TOKEN.finditer(str(prior or ""))]
        if len(prior_values) < 6:
            continue
        matcher = SequenceMatcher(None, request_values, prior_values, autojunk=False)
        for block in matcher.get_matching_blocks():
            if block.size < 6:
                continue
            start = request_tokens[block.a][1]
            end = request_tokens[block.a + block.size - 1][2]
            while end < len(text) and text[end] in ".,;:!?)]}":
                end += 1
            if end - start >= 32:
                spans.append((start, end))
    return _merge_spans(spans, len(text))


def _unquoted_attributed_spans(text: str) -> tuple[SourceSpan, ...]:
    """Find prose explicitly attributed to SliceAgent's earlier report/speech.

    This closes the important gap between Markdown quoting and normal conversational quotation: users commonly
    paste an earlier finding after ``in your original findings, it is ...`` without quote marks.  A clearly new
    clause (``Now fix it`` / ``Why?``) ends the attributed span; a report-internal label such as ``Fix: ...``
    deliberately does not.
    """
    spans = []
    for equation in _REFERENCE_EQUATION.finditer(text):
        start = equation.end()
        tail = _REFERENCE_CONFIRMATION_TAIL.search(text, start)
        boundary = _NEW_AUTHORITY_AFTER_ATTRIBUTION.search(text, start)
        ends = [match.start() for match in (tail, boundary) if match is not None]
        end = min(ends) if ends else len(text)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    for intro in _ATTRIBUTION_INTRO.finditer(text):
        start = intro.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            continue
        # Explicit quote syntax is already handled precisely by _mask_quoted_data_with_spans. Extending this
        # prose attribution to the end would incorrectly swallow a later instruction outside the quote.
        if text[start] in "\"'“‘`>":
            continue
        boundary = _NEW_AUTHORITY_AFTER_ATTRIBUTION.search(text, start)
        end = boundary.start() if boundary is not None else len(text)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return _merge_spans(spans, len(text))


def _validated_contract(request: str, contract: TurnContract) -> TurnContract:
    attributed = _merge_spans(contract.attributed_spans, len(request))
    authority = _merge_spans(contract.authority_spans, len(request))
    # Attributed data always wins an overlap.  This is a fail-closed control projection; it never rewrites the
    # request and the exact original remains available to the model and artifact store.
    if attributed and authority:
        pieces = []
        for start, end in authority:
            cursor = start
            for blocked_start, blocked_end in attributed:
                if blocked_end <= cursor or blocked_start >= end:
                    continue
                if cursor < blocked_start:
                    pieces.append((cursor, min(end, blocked_start)))
                cursor = max(cursor, blocked_end)
                if cursor >= end:
                    break
            if cursor < end:
                pieces.append((cursor, end))
        authority = _merge_spans(pieces, len(request))
    return replace(
        contract,
        request_text=str(request or ""),
        authority_spans=authority,
        attributed_spans=attributed,
    )


_EFFECT_LEADING_JUNK = re.compile(
    r"^[\s,.;:!?]*(?:(?:then|and\s+then)\s+)?"
    r"(?:(?:yes|yeah|yep|sure|okay|ok|no)\s*,\s*)?",
    re.IGNORECASE,
)
_EFFECT_CLAUSE_END = re.compile(
    r",\s*(?=(?:then|and\s+then)\b)|;\s*|\n+|[.!?](?=\s|$)",
    re.IGNORECASE,
)


def _effect_directive_spans(scan: str) -> tuple[SourceSpan, ...]:
    spans = []
    for match in (
        *_EFFECT_DIRECTIVE.finditer(scan),
        *_DELEGATION_DIRECTIVE.finditer(scan),
        *_WORKSPACE_NAVIGATION_DIRECTIVE.finditer(scan),
        *_BARE_WORKSPACE_NAVIGATION_DIRECTIVE.finditer(scan),
    ):
        start = match.start()
        junk = _EFFECT_LEADING_JUNK.match(scan[start:match.end()])
        if junk is not None:
            start += junk.end()
        tail = _EFFECT_CLAUSE_END.search(scan, match.end())
        if tail is None:
            end = len(scan)
        else:
            end = tail.start() + (1 if scan[tail.start():tail.start() + 1] in ".!?" else 0)
        while start < end and scan[start].isspace():
            start += 1
        while end > start and scan[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    complex_action = _DO_CHANGE_DIRECTIVE.match(scan)
    if complex_action is not None:
        tail = _EFFECT_CLAUSE_END.search(scan, complex_action.end())
        end = len(scan)
        if tail is not None:
            end = tail.start() + (1 if scan[tail.start():tail.start() + 1] in ".!?" else 0)
        spans.append((complex_action.start(), end))
    return _merge_spans(spans, len(scan))


def _looks_like_file_target(target: str, clause: str, match: re.Match) -> bool:
    normalized = target.strip("`\"'“”‘’")
    if not normalized:
        return False
    folded = normalized.casefold()
    if re.fullmatch(r"v?\d+(?:\.\d+)+", folded) or re.fullmatch(r"\d+(?:\.\d+){3}", folded):
        return False
    if folded in {"node.js"}:
        return False
    if normalized.startswith(".") or "/" in normalized or "\\" in normalized:
        return not re.match(r"^[a-z][a-z0-9+.-]*://", normalized, re.I)
    if folded in {"readme", "changelog", "license", "makefile", "dockerfile", "procfile"} \
            or folded.startswith(("readme.", "changelog.", "license.")):
        return True
    extension = os.path.splitext(normalized)[1].lstrip(".").casefold()
    if extension in {"com", "org", "net", "io", "dev", "ai", "co", "app"}:
        return False
    # Quoted/backticked arguments are deliberate. Bare dotted tokens are files only in a file-oriented
    # directive, avoiding versions, domains, and product names from narrowing a broad upgrade.
    before = clause[:match.start()]
    if match.group(1) or match.group(2):
        return bool(extension) or re.search(
            r"\b(?:edit|write|create|delete|remove|rename|move|copy)\b", before, re.IGNORECASE,
        ) is not None
    return bool(extension) and re.search(
        r"\b(?:edit|fix|patch|repair|change|update|upgrade|modify|refactor|rewrite|write|create|"
        r"delete|remove|rename|move|copy|format|lint|test)\b",
        before, re.IGNORECASE,
    ) is not None


def _file_targets(clause: str) -> tuple[str, ...]:
    targets = []
    for match in _FILE_TARGET.finditer(clause):
        target = next((part for part in match.groups() if part), "").rstrip(".,;:!)]}\"'*_~")
        if target and _looks_like_file_target(target, clause, match):
            targets.append(target)
    return tuple(dict.fromkeys(targets))


def _edit_path_scopes(clause: str) -> tuple[tuple[str, str], ...]:
    glob_match = re.search(r"(?<!\w)((?:\*\*/)?\*\.[A-Za-z0-9_.-]+)\b", clause)
    extension_match = re.search(r"\ball\s+\.([A-Za-z0-9_.-]+)\s+files?\b", clause, re.I)
    pattern = (
        glob_match.group(1) if glob_match is not None else
        f"*.{extension_match.group(1)}" if extension_match is not None else ""
    )
    directories = []
    for match in re.finditer(
        r"(?:^|\s)(?:files?\s+|everything\s+)?(?:under|within|in)\s+"
        r"([A-Za-z0-9_.-]+(?:[/\\][A-Za-z0-9_.-]+)*)[/\\]?"
        r"(?=\s|[!?](?:\s|$)|\.(?:\s|$)|$)",
        clause, re.IGNORECASE,
    ):
        directories.append(match.group(1))
    for match in re.finditer(
        r"(?:^|\s)([A-Za-z0-9_.-]+(?:[/\\][A-Za-z0-9_.-]+)*)[/\\]"
        r"(?=\s|[!?](?:\s|$)|\.(?:\s|$)|$)",
        clause, re.IGNORECASE,
    ):
        directories.append(match.group(1))
    pair = re.search(
        r"\b([A-Za-z0-9_.-]+)\s+and\s+([A-Za-z0-9_.-]+)\s+directories\b",
        clause, re.IGNORECASE,
    )
    if pair is not None:
        directories.extend(pair.groups())
    directories = list(dict.fromkeys(directory.strip("/\\") for directory in directories if directory))
    if pattern and directories:
        return tuple(("workspace.edit_scoped_glob", f"{directory} | {pattern}")
                     for directory in directories)
    if pattern:
        return (("workspace.edit_glob", pattern),)
    return tuple(("workspace.edit_prefix", directory) for directory in directories)


def _first_file_target(clause: str) -> str:
    return next(iter(_file_targets(clause)), "")


def _workspace_target(clause: str) -> str:
    target = _first_file_target(clause)
    if target:
        return target
    quoted = re.search(r"[\"“‘`]([^\"”’`\r\n]+)[\"”’`]", clause)
    if quoted is not None:
        return " ".join(quoted.group(1).split())
    match = _WORKSPACE_TARGET.search(clause)
    if match is not None:
        return next((part for part in match.groups() if part), "")
    bare = re.search(
        r"\b(?:(?:go|navigate|move|switch)\s+(?:to|into)\s+|work\s+in\s+)"
        r"(?:the\s+)?([A-Za-z0-9_.-]+)\s*[.!?]*$",
        clause, re.IGNORECASE,
    )
    return bare.group(1) if bare is not None else ""


def _execution_target(clause: str) -> str:
    """Keep the user-authored command/process hint carried by an execution grant.

    This is deliberately lexical rather than a miniature planner.  The authority hook later compares the
    hint with the concrete command; an unnamed ``run/start`` request therefore cannot become ambient shell
    authority.
    """
    quoted = re.search(r"`([^`\r\n]+)`", clause)
    if quoted is not None:
        return " ".join(quoted.group(1).split())
    match = re.search(r"\b(?:run|execute|launch|start)\b", clause, re.IGNORECASE)
    if match is None:
        return ""
    tail = re.sub(
        r"^(?:\s+(?:the|a|an|this|that|command|process|task|please|now))+\b",
        "", clause[match.end():], flags=re.IGNORECASE,
    )
    return " ".join(tail.strip(" \t.,;:!?\"'").split())


def _quoted_execution_target(clause: str) -> str:
    match = re.search(
        r"\b(?:run|execute)\b\s*(?:the\s+)?(?:command\s+)?([\"`])(.+?)\1\s*[.!?]*\s*$",
        clause, re.IGNORECASE | re.DOTALL,
    )
    return " ".join(match.group(2).split()) if match is not None else ""


def _dependency_targets(clause: str) -> tuple[str, ...]:
    quoted = [" ".join(match.group(1).split()) for match in re.finditer(
        r"[\"“‘`]([^\"”’`\r\n]+)[\"”’`]", clause,
    )]
    if quoted:
        return tuple(quoted)
    match = re.search(
        r"\binstall\b\s+(?:(?:the|a|an)\s+)?(?:(?:dependency|package|module|library)\s+)?"
        r"(.+?)\s*[.!?]*$",
        clause, re.IGNORECASE,
    )
    if match is None:
        return ()
    value = " ".join(match.group(1).split())
    if value.casefold() in {"dependency", "dependencies", "package", "packages", "requirements"}:
        return ()
    return tuple(
        part.strip() for part in re.split(r"\s*(?:,|\band\b)\s*", value, flags=re.I) if part.strip()
    )


def _push_target(clause: str) -> str:
    match = re.search(
        r"\bpush\b\s+(?:(?:the|my)\s+)?([A-Za-z0-9._/-]+)\s+branch\b",
        clause, re.IGNORECASE,
    )
    return match.group(1) if match is not None else ""


def _deploy_target(clause: str) -> str:
    match = re.search(
        r"\bdeploy\b[^.!?\n]{0,80}\bto\s+(?:(?:the|our)\s+)?([A-Za-z0-9_.-]+)\b",
        clause, re.IGNORECASE,
    )
    return match.group(1) if match is not None else ""


def _process_target(clause: str) -> str:
    match = re.search(
        r"\b(?:stop|kill|terminate|close)\b\s+(?:(?:the|this|that)\s+)?"
        r"([A-Za-z0-9_.-]+)(?:\s+(?:process|server|service|worker|job|terminal|session|daemon))?\b",
        clause, re.IGNORECASE,
    )
    return match.group(1) if match is not None else ""


def _requirement_target(clause: str) -> str:
    quoted = re.search(r"[\"“‘`]([^\"”’`\r\n]+)[\"”’`]", clause)
    if quoted is not None:
        target = " ".join(quoted.group(1).split())
        return re.sub(r"\s+(?:done|complete|completed|satisfied)\s*$", "", target, flags=re.I)
    match = re.search(
        r"\b(?:requirement|constraint)\b\s*(?::|that\b|to\b)?\s+(.+?)\s*[.!?]*$",
        clause, re.IGNORECASE,
    )
    if match is not None and match.group(1).casefold() not in {"done", "complete"}:
        target = " ".join(match.group(1).split())
        target = re.sub(r"\s+(?:done|complete|completed|satisfied)\s*$", "", target, flags=re.I)
        target = re.split(r"\s+with\s+(?:requirement\s+)?", target, maxsplit=1, flags=re.I)[0]
        return target
    match = re.search(
        r"\b(?:add|remove|drop|complete|finish|satisfy)\b\s+(.+?)\s+"
        r"(?:as\s+)?(?:a\s+|the\s+)?(?:requirement|constraint)\b",
        clause, re.IGNORECASE,
    )
    if match is not None:
        target = " ".join(match.group(1).split())
        return "" if target.casefold() in {"this", "that", "the", "a"} else target
    return ""


def _requirement_transition(clause: str) -> tuple[str, str] | None:
    quoted = [" ".join(match.group(1).split()) for match in re.finditer(
        r"[\"“‘`]([^\"”’`\r\n]+)[\"”’`]", clause,
    )]
    if len(quoted) >= 2:
        return quoted[0], quoted[1]
    match = re.search(
        r"\b(?:supersede|replace)\b\s+(?:requirement\s+|constraint\s+)?(.+?)\s+with\s+"
        r"(?:requirement\s+|constraint\s+)?(.+?)\s*[.!?]*$",
        clause, re.IGNORECASE,
    )
    if match is None:
        return None
    return " ".join(match.group(1).split()), " ".join(match.group(2).split())


def _action_spans(source: str, spans: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    """Split coordinated directives while retaining exact source evidence for each capability."""
    actions = []
    joiner = re.compile(
        rf"(?:,\s*(?:and\s+)?|\b(?:and(?:\s+then)?|then)\b\s+)"
        rf"(?=(?:(?:now|please|just)\s+)*(?:{_EFFECT_VERB_SRC})\b)",
        re.IGNORECASE,
    )

    def add_action(start: int, end: int) -> None:
        negative = re.search(
            rf"\s*,?\s*\bbut\s+(?:do\s+not|don't|never)\s+(?:{_EFFECT_VERB_SRC})\b",
            source[start:end], re.IGNORECASE,
        )
        if negative is not None:
            end = start + negative.start()
        while end > start and source[end - 1].isspace():
            end -= 1
        if start < end:
            actions.append((start, end))

    for start, end in spans:
        cursor = start
        for match in joiner.finditer(source, start, end):
            left_end = match.start()
            while left_end > cursor and source[left_end - 1].isspace():
                left_end -= 1
            add_action(cursor, left_end)
            cursor = match.end()
        while cursor < end and source[cursor].isspace():
            cursor += 1
        add_action(cursor, end)
    return tuple(actions)


def _governing_effect_verb(clause: str) -> str:
    """Return the directive's operative verb, never a later noun such as ``commit parser``."""
    delegation = _DELEGATION_DIRECTIVE.search(clause)
    if delegation is not None:
        value = delegation.group("verb").casefold()
        return "delegate" if value == "delegate" else value
    matches = list(_EFFECT_DIRECTIVE.finditer(clause))
    if not matches:
        return ""
    prefix = clause[:matches[0].end()]
    verbs = list(re.finditer(rf"\b({_EFFECT_VERB_SRC})\b", prefix, re.IGNORECASE))
    if not verbs:
        return ""
    value = verbs[-1].group(1).casefold()
    if value.startswith("fix"):
        return "fix"
    return "clean" if value.startswith("clean") else value


def _typed_pending_grant(pending_proposal: object, *, source_span: SourceSpan | None) -> EffectGrant | None:
    if not isinstance(pending_proposal, Mapping):
        return None
    action = pending_proposal.get("action")
    if not isinstance(action, Mapping):
        return None
    tool = str(action.get("tool") or "").strip()
    args = action.get("args") or {}
    if not tool or not isinstance(args, Mapping):
        return None
    return EffectGrant.exact(tool, args, source_span=source_span)


def _selected_nav_target_grant(request: str, pending_proposal: object, source: str) -> EffectGrant | None:
    """A next-turn reply naming exactly one option from the assistant's own navigation-disambiguation
    question confers scoped navigation authority to that named target. Requires a bare selection (the
    normalized reply equals one offered name) so a full sentence or a new directive never matches here."""
    if not isinstance(pending_proposal, Mapping):
        return None
    targets = pending_proposal.get("nav_targets")
    if not isinstance(targets, (list, tuple)) or not targets:
        return None
    reply = re.sub(r"[^a-z0-9]+", "", str(request or "").casefold())
    if not reply:
        return None
    for target in targets:
        name = str(target or "").strip()
        if name and re.sub(r"[^a-z0-9]+", "", name.casefold()) == reply:
            span = (0, len(source)) if source else None
            return EffectGrant("workspace.navigate", ("change_workspace",),
                               target=name, target_arg="path", source_span=span)
    return None


def _explicit_effect_grants(source: str, spans: Iterable[SourceSpan]) -> tuple[EffectGrant, ...]:
    """Compile high-confidence directive spans into a deliberately small capability set.

    This is not a general planner. It prevents one detected verb from becoming ambient authority for every
    effectful tool while retaining the ordinary coding fast path. Unknown operations get no grant and must be
    clarified rather than silently widening the turn.
    """
    grants: list[EffectGrant] = []
    for span in _action_spans(source, spans):
        start, end = span
        clause = source[start:end]
        low = clause.casefold()
        verb = _governing_effect_verb(clause)
        code_object = re.search(
            r"\b(?:button|component|parser|render(?:ing)?|handler|handling|event|modal|animation|"
            r"feature\s+flag|implementation|implementing|code|logic|behavior)\b",
            low,
        ) is not None
        if _WORKSPACE_NAVIGATION_DIRECTIVE.search(clause) \
                or _BARE_WORKSPACE_NAVIGATION_DIRECTIVE.search(clause) or (
                verb in {"switch", "change"}
                and not code_object
                and re.search(r"\b(?:workspace|project|repo(?:sitory)?|directory|folder)\b", low)):
            target = _workspace_target(clause)
            # "switch workspace" names no resolvable target. Keep the effect explicit but ungranted so the
            # model can observe/ask instead of navigating to an invented directory.
            if target:
                grants.append(EffectGrant(
                    "workspace.navigate", ("change_workspace",), target=target,
                    target_arg="path", source_span=span,
                ))
            continue
        if verb == "switch" and not code_object:
            # ``switch to Hunter`` is navigation-shaped but lacks the noun needed by the stricter detector.
            # Resolve the explicit target if possible; otherwise fail closed instead of granting code edits.
            target = _workspace_target(clause)
            if target:
                grants.append(EffectGrant(
                    "workspace.navigate", ("change_workspace",), target=target,
                    target_arg="path", source_span=span,
                ))
            continue
        if verb in {"new", "switch", "resume"} and not code_object \
                and re.search(r"\b(?:task|topic)\b\s*[.!?]*$", low):
            tools = (("new_topic",) if verb == "new" else ("switch_topic",))
            grants.append(EffectGrant("task.route", tools, source_span=span))
            continue
        if verb in {"update", "change", "revise", "replace"} and not code_object \
                and re.search(r"\bplan\b(?:\s+now)?\s*[.!?]*$", low):
            grants.append(EffectGrant("task.plan", ("update_plan",), source_span=span))
            continue
        if verb in {"add", "remove", "drop", "delete", "supersede", "replace", "mark", "complete", "finish"} \
                and re.search(r"\b(?:requirement|constraint)\b", low):
            # Constraint verbs are not interchangeable.  In particular, permission to add a requirement
            # must never imply permission to silently drop or supersede one.
            if re.search(r"\b(?:drop|remove|delete)\b", low):
                tools = ("drop_requirement",)
                operation = "task.requirement.drop"
            elif re.search(r"\b(?:supersede|replace)\b", low):
                tools = ("supersede_requirement",)
                operation = "task.requirement.supersede"
            elif re.search(r"\b(?:done|complete|satisf(?:y|ied)|finish)\b", low):
                tools = ("requirement_done",)
                operation = "task.requirement.complete"
            else:
                tools = ("require",)
                operation = "task.requirement.add"
            if tools == ("supersede_requirement",):
                transition = _requirement_transition(clause)
                if transition is not None:
                    grants.append(EffectGrant.exact(
                        "supersede_requirement",
                        {"old_text": transition[0], "new_text": transition[1]},
                        source_span=span,
                    ))
                continue
            target = _requirement_target(clause)
            if target:
                grants.append(EffectGrant(
                    operation, tools, target=target,
                    target_arg="old_text" if tools == ("supersede_requirement",) else "text",
                    source_span=span,
                ))
            continue
        if verb in {"test", "verify", "lint"}:
            grants.append(EffectGrant("workspace.verify", ("run_command",), source_span=span))
            continue
        if verb in {"spawn", "delegate", "launch"} and re.search(
            r"\b(?:subagents?|child\s+agents?|explorers?|agents?)\b", low,
        ):
            # The core product exposes read-only explorer delegation. Bind the grant to that exact profile so a
            # request for parallel review cannot silently authorize a writable/general child in advanced mode.
            grants.append(EffectGrant(
                "task.delegate", ("spawn_agent",), target="explorer", target_arg="agent",
                source_span=span,
            ))
            continue
        process_subject = re.search(
            r"\b(?:process|server|service|worker|job|terminal|session|daemon|handle)\b", low,
        ) is not None
        if verb in {"stop", "kill", "terminate", "close"} and process_subject and not code_object:
            target = _process_target(clause)
            grants.append(EffectGrant(
                # Host process handles are already scoped to processes created by this SliceAgent runtime.
                # Raw shell kill commands are intentionally not included: they cannot be bound to that set.
                "process.stop", ("proc_kill", "terminal_close"), target=target,
                target_arg="handle" if target else "", source_span=span,
            ))
            continue
        if verb in {"run", "execute", "launch", "start"} and (
                verb in {"run", "execute"} or process_subject
                or re.search(r"\b(?:tests?|pytest|lint|typecheck|check(?:s)?|build)\b", low)):
            exact_command = _quoted_execution_target(clause)
            if exact_command:
                grants.append(EffectGrant.exact("run_command", {"command": exact_command}, source_span=span))
                continue
            target = _execution_target(clause)
            if re.search(r"\b(?:tests?|pytest|lint|typecheck|check(?:s)?|build)\b", target, re.I):
                grants.append(EffectGrant("workspace.verify", ("run_command",), source_span=span))
            elif target:
                grants.append(EffectGrant(
                    "process.start" if verb in {"launch", "start"} else "workspace.run",
                    ("run_command", "proc_start", "terminal_open"),
                    target=target, target_arg="command", source_span=span,
                ))
            continue
        if verb in {"deploy", "publish", "commit", "push", "install", "stage", "revert"}:
            operation = {
                "deploy": "workspace.deploy",
                "publish": "package.publish",
                "commit": "vcs.commit",
                "push": "vcs.push",
                "install": "dependency.install",
                "stage": "vcs.stage",
                "revert": "vcs.revert",
            }[verb]
            operation_targets = (
                _file_targets(clause) if verb in {"commit", "stage", "revert"} else
                _dependency_targets(clause) if verb == "install" else
                (_push_target(clause),) if verb == "push" and _push_target(clause) else
                (_deploy_target(clause),) if verb == "deploy" and _deploy_target(clause) else ()
            )
            grants.append(EffectGrant(
                operation, ("run_command",), target=" | ".join(operation_targets),
                target_arg="command" if operation_targets else "", source_span=span,
            ))
            continue
        if verb == "write" and re.search(r"\bskill\b", low):
            grants.append(EffectGrant("skill.write", ("write_skill",), source_span=span))
            continue

        file_targets = _file_targets(clause)
        if verb in {"delete", "remove"} and file_targets:
            for target in file_targets:
                grants.append(EffectGrant(
                    "workspace.delete", ("run_command",), target=target,
                    target_arg="command", source_span=span,
                ))
            continue
        if verb in {"rename", "move", "copy"} and len(file_targets) >= 2:
            grants.append(EffectGrant(
                "workspace.copy" if verb == "copy" else "workspace.rename",
                ("run_command",), target=" → ".join(file_targets[:2]),
                target_arg="command", source_span=span,
            ))
            continue

        # A coding/change directive authorizes workspace edits, not navigation, process management, topic
        # changes, or arbitrary world-state mutation. A named file narrows every file writer to that target.
        targets = file_targets
        path_scopes = _edit_path_scopes(clause)
        if path_scopes:
            for operation, target in path_scopes:
                grants.append(EffectGrant(
                    operation, _WORKSPACE_EDIT_TOOLS,
                    target=target, target_arg="path", source_span=span,
                ))
        elif targets:
            for target in targets:
                grants.append(EffectGrant(
                    "workspace.edit", _WORKSPACE_EDIT_TOOLS,
                    target=target, target_arg="path", source_span=span,
                ))
        else:
            grants.append(EffectGrant(
                "workspace.edit", _WORKSPACE_EDIT_TOOLS, source_span=span,
            ))
            # A broad implementation directive ("build/fix this upgrade") authorizes ordinary local
            # generators, formatters, migrations, and dependency commands required to carry out that same
            # workspace change. Named-file directives stay narrow, and the authority hook independently
            # rejects release/deploy/VCS publication commands that would expand the user's objective.
            grants.append(EffectGrant(
                "workspace.implement", ("run_command",), source_span=span,
            ))
            grants.append(EffectGrant(
                # execute_code remains useful for an atomic multi-file edit, but it receives a separate
                # syntax-aware matcher instead of inheriting arbitrary Python/subprocess authority.
                "workspace.batch_edit", ("execute_code",), source_span=span,
            ))
        # Verification and plan/requirement bookkeeping are related implementation effects. The hook admits
        # only recognizable verification commands for this operation; all other shell effects remain denied.
        grants.append(EffectGrant(
            "workspace.verify", ("run_command",), source_span=span,
        ))
        grants.append(EffectGrant(
            "task.maintain", _TASK_MAINTENANCE_TOOLS, source_span=span,
        ))
    # Preserve order for explanation/rendering while removing duplicate projections of overlapping spans.
    unique = []
    seen = set()
    for grant in grants:
        key = (grant.operation, grant.tools, grant.target, grant.target_arg, grant.exact_args, grant.mode)
        if key not in seen:
            seen.add(key); unique.append(grant)
    return tuple(unique)


_DELEGATION_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def derive_delegation_requirement(request: str) -> DelegationRequirement | None:
    """Compile an explicit child-agent mechanism into a small completion invariant."""
    source = str(request or "")
    directive = _DELEGATION_DIRECTIVE.search(source)
    if directive is None:
        return None
    tail = source[directive.start():]
    count_match = re.search(
        r"\b(?:exactly\s+)?(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
        r"(?=[^.!?;\n]{0,100}\b(?:parallel\s+)?(?:explorer\s+)?"
        r"(?:subagents?|child\s+agents?|agents?)\b)",
        tail, re.IGNORECASE,
    )
    count = None
    if count_match is not None:
        token = count_match.group("count").casefold()
        count = int(token) if token.isdigit() else _DELEGATION_COUNT_WORDS[token]
    targets = []
    for match in _FILE_TARGET.finditer(tail):
        target = next((part for part in match.groups() if part), "").rstrip(".,;:!)]}\"'*_~")
        extension = os.path.splitext(target)[1].lstrip(".").casefold()
        if target and extension and extension not in {"com", "org", "net", "io", "dev", "ai", "co", "app"}:
            targets.append(target)
    targets = list(dict.fromkeys(targets))
    if count is None and targets and re.search(r"\bone\s+each\b", tail, re.IGNORECASE):
        count = len(targets)
    return DelegationRequirement(
        agent="explorer", count=count, targets=tuple(targets),
        parallel=re.search(r"\bparallel(?:ly)?\b", tail, re.IGNORECASE) is not None,
    )


def _source_needs_for(
    source: str, *, historical: bool, live: bool, evidence_query: EvidenceQuery | None = None,
    quality_evidence_query: QualityEvidenceQuery | None = None,
) -> tuple[SourceNeed, ...]:
    needs: list[SourceNeed] = []
    if quality_evidence_query is not None:
        # Quality evidence is an indivisible request/response pair. Two independent source flags did not ensure
        # co-residency and let the model judge a paged-out response from a one-line manifest preview.
        needs.append("sealed_exchange")
    if _PAST_USER_UTTERANCE.search(source):
        needs.append("prior_user_utterance")
    if _PAST_ASSISTANT_UTTERANCE.search(source):
        needs.append("prior_assistant_utterance")
    if evidence_query is not None:
        needs.append(evidence_query.source)
    if historical and not needs:
        needs.append("prior_assistant_utterance")
    if historical and re.search(r"\b(?:finding|issue|report|observ(?:e|ed|ation))\b", source, re.I):
        needs.append("historical_observation")
    if live:
        needs.append("current_world")
    return tuple(dict.fromkeys(needs))


def _intent_candidate_spans(scan: str) -> tuple[SourceSpan, ...]:
    """Exact clauses eligible for the existing durable-intent reducer.

    This is a source selection step, not a second semantic reducer: capture_explicit_constraints still owns
    the detailed correction/linkage decision. It merely prevents surrounding confirmation prose from gaining
    authority because a quoted finding happened to contain ``never`` or ``Fix:``.
    """
    spans = []
    start = 0
    raw_spans = []
    for boundary in _CLAUSE_BOUNDARY.finditer(scan):
        raw_spans.append((start, boundary.start()))
        start = boundary.end()
    raw_spans.append((start, len(scan)))
    historical = _HISTORICAL_REFERENCE.search(scan) is not None
    for raw_start, raw_end in raw_spans:
        left, right = raw_start, raw_end
        while left < right and scan[left].isspace():
            left += 1
        prefix = _CLAUSE_PREFIX.match(scan, left, right)
        if prefix is not None:
            left = prefix.end()
        while right > left and scan[right - 1].isspace():
            right -= 1
        clause = scan[left:right]
        if not clause:
            continue
        direct = bool(_DIRECTIVE_PREFIX.search(clause) or _POLITE_DIRECTIVE_QUESTION.match(clause))
        confirmation = _CONFIRMATION_OR_CORRECTION.match(clause) is not None
        if confirmation and not direct and (historical or clause.rstrip().endswith("?")):
            continue
        correction_speech = bool(
            re.match(r"^\s*(?:correction|correcting|actually|change\s+of\s+plan)\b", clause, re.I)
            or re.search(r"\b(?:instead\s+of|no\s+longer|replace\b[^.!?\n]{0,80}\bwith)\b", clause, re.I)
            or re.search(r"\bi\s+(?:mean|meant)\b", clause, re.I)
        )
        if (direct or _EXPLICIT_CONSTRAINT.search(clause) or correction_speech
                or _ACTUALLY_PREFIX.search(clause)):
            spans.append((left, right))
    return _merge_spans(spans, len(scan))


def analyze_turn(
    request: str,
    *,
    prior_texts: Iterable[str] = (),
    pending_proposal: object = None,
    referents: Iterable[object] = (),
    previous_evidence_query: EvidenceQuery | None = None,
    previous_quality_evidence_query: QualityEvidenceQuery | None = None,
) -> TurnContract:
    """Build the conservative, exact-span contract for one current request.

    This deliberately does not decide the user's full semantic intent.  It identifies attributed data, then
    asks only whether the remaining user-authored surface clearly authorizes effects.  Ambiguity stays explicit
    instead of being upgraded into permission.
    """
    source = str(request or "")
    formal_scan, formal_spans = _mask_quoted_data_with_spans(source)
    copied_spans = _copied_prior_spans(source, prior_texts)
    prose_spans = _unquoted_attributed_spans(source)
    operative_arguments = tuple(
        span for span in _operative_quoted_argument_spans(source, formal_spans)
        if not any(_spans_overlap(span, blocked) for blocked in (*copied_spans, *prose_spans))
    )
    if operative_arguments:
        restored = list(formal_scan)
        for start, end in operative_arguments:
            restored[start:end] = source[start:end]
        formal_scan = "".join(restored)
        formal_spans = tuple(span for span in formal_spans if span not in operative_arguments)
    attributed = _merge_spans((*formal_spans, *copied_spans, *prose_spans), len(source))
    available = _complement_spans(source, attributed)
    scan = _scan_spans(formal_scan, available)
    compact = " ".join(scan.split())
    delegation_requirement = derive_delegation_requirement(scan)

    generic_verification = _GENERIC_EVIDENCE_VERIFICATION.search(compact) is not None
    evidence_query = derive_evidence_query(scan)
    if generic_verification and isinstance(previous_evidence_query, EvidenceQuery):
        # "Verify that" is discourse continuity, not a fresh all-tools query. Carry only the immediately
        # preceding typed selector; callers clear the continuity slot on every unrelated turn.
        evidence_query = previous_evidence_query
    self_audit = _SELF_AUDIT.search(scan) is not None
    quality_evidence_query = derive_quality_evidence_query(scan)
    if (quality_evidence_query is not None and quality_evidence_query.scope == "latest_response"
            and evidence_query is not None and evidence_query.scope == "task"):
        evidence_query = replace(evidence_query, scope="latest_turn")
    if generic_verification and isinstance(previous_quality_evidence_query, QualityEvidenceQuery):
        quality_evidence_query = replace(
            previous_quality_evidence_query, purpose="verify_assessment",
        )
    evidence_continuation = bool(
        generic_verification
        and (previous_evidence_query is not None or previous_quality_evidence_query is not None)
    )
    historical = bool(
        _HISTORICAL_REFERENCE.search(scan) or _PAST_USER_UTTERANCE.search(scan)
        or _PAST_ASSISTANT_UTTERANCE.search(scan)
        or evidence_query is not None or quality_evidence_query is not None
        or prose_spans or copied_spans or referents
    )
    live = bool(_LIVE_REFERENCE.search(scan))
    # A family noun such as "file" identifies receipt operations; it does not by itself ask for current file
    # truth. Mixed past+present queries remain possible only through explicit temporal language ("still now").
    if evidence_query is not None and _EXPLICIT_LIVE_TIME.search(scan) is None:
        live = False
    bare_assent = _BARE_ASSENT.fullmatch(compact) is not None
    pending_grant = None
    if bare_assent and isinstance(pending_proposal, Mapping):
        grant_source: object = pending_proposal
        options = pending_proposal.get("options")
        if isinstance(options, (list, tuple)) and len(options) > 1:
            selected = _OPTION_SELECTION.search(compact)
            ordinal = int(next((part for part in selected.groups() if part), "0")) if selected else 0
            for option in options:
                if not isinstance(option, Mapping):
                    continue
                try:
                    if int(option.get("ordinal") or 0) == ordinal:
                        grant_source = option
                        break
                except (TypeError, ValueError):
                    continue
        assent_span = (0, len(source)) if source else None
        pending_grant = _typed_pending_grant(grant_source, source_span=assent_span)
        if pending_grant is None:
            # A bare "yes" to a single-target navigation offer ("switch to loom-app?") continues that one
            # navigation. Multiple targets stay ambiguous under a bare assent (name one to disambiguate).
            nav_targets = pending_proposal.get("nav_targets")
            if isinstance(nav_targets, (list, tuple)) and len(nav_targets) == 1:
                name = str(nav_targets[0] or "").strip()
                if name:
                    pending_grant = EffectGrant("workspace.navigate", ("change_workspace",),
                                                target=name, target_arg="path", source_span=assent_span)
    proposal_accepts = pending_grant is not None
    negated = _NEGATED_EFFECT.search(compact) is not None
    # Negation is clause-local. In "don't fix #2; instead fix #3", the first clause must not lend
    # authority, but it must not erase the independent positive directive that follows it either.
    effect_spans = tuple(
        (start, end) for start, end in _effect_directive_spans(scan)
        if _NEGATED_EFFECT.search(scan[start:end]) is None
        and _ANSWER_PRODUCTION_DIRECTIVE.match(scan[start:end]) is None
        and not (
            evidence_query is not None
            and re.search(r"\bverify\b[^.!?\n]{0,100}\b(?:records?|receipts?|history)\b", scan[start:end], re.I)
        )
    )
    polite_directive = _POLITE_DIRECTIVE_QUESTION.match(compact) is not None
    explicit_effect = bool(effect_spans or polite_directive)
    counterfactual_question = _COUNTERFACTUAL_QUESTION.match(compact) is not None
    question = bool(not explicit_effect and (
        _QUESTION_PREFIX.match(compact) or counterfactual_question or compact.rstrip().endswith("?")))
    confirmation = _CONFIRMATION_OR_CORRECTION.match(compact) is not None
    answer_only = bool(
        _ANSWER_ONLY_DIRECTIVE.match(compact) or _ANSWER_PRODUCTION_DIRECTIVE.match(compact)
    )
    authority = _merge_spans((*effect_spans, *_intent_candidate_spans(scan)), len(source))

    modes = []
    if historical:
        modes.append("recall")
    if self_audit or quality_evidence_query is not None:
        modes.append("audit")
    if evidence_continuation:
        modes.append("verify")
    if question:
        modes.append("explain" if re.match(r"^\s*(?:why|how)\b", compact, re.IGNORECASE) else "answer")
        if counterfactual_question:
            modes.append("recommend")
    if confirmation or negated:
        modes.append("confirm")
    if answer_only:
        modes.append("inspect")
    if delegation_requirement is not None:
        modes.append("delegate")
    if explicit_effect:
        modes.append("change")

    if bare_assent:
        if proposal_accepts:
            effect_authority: EffectAuthority = "continuation"
            modes.append("continue")
        else:
            effect_authority = "uncertain"
    elif explicit_effect:
        effect_authority = "explicit"
    elif negated or question or confirmation or answer_only or historical or not compact:
        effect_authority = "none"
    elif (nav_grant := _selected_nav_target_grant(compact, pending_proposal, source)) is not None:
        # A bare reply selecting one option from the assistant's own navigation-disambiguation question.
        pending_grant = nav_grant
        effect_authority = "continuation"
        modes.append("continue")
    else:
        effect_authority = "uncertain"

    # An authorized effect necessarily acts on present state.  A past-record reference plus a present action
    # is the mixed case (e.g. "that was finding #2; now fix #3").
    live = live or effect_authority in ("explicit", "continuation")
    grounding: GroundingMode = (
        "both" if historical and live else
        "sealed_past" if historical else
        "live_present" if live else
        "none"
    )
    grants: tuple[EffectGrant, ...] = ()
    if effect_authority == "continuation":
        # Continuation authority is emitted only together with this exact typed grant.
        grants = (pending_grant,) if pending_grant is not None else ()
    elif effect_authority == "explicit":
        grants = _explicit_effect_grants(source, effect_spans or authority)
    source_needs = _source_needs_for(
        source, historical=historical, live=live, evidence_query=evidence_query,
        quality_evidence_query=quality_evidence_query,
    )
    return _validated_contract(source, TurnAdmission(
        request_text=source,
        effect_authority=effect_authority,
        grounding=grounding,
        authority_spans=authority,
        attributed_spans=attributed,
        requested_modes=tuple(modes),
        source_needs=source_needs,
        evidence_query=evidence_query,
        quality_evidence_query=quality_evidence_query,
        delegation_requirement=delegation_requirement,
        evidence_continuation=evidence_continuation,
        actor=EntityRef("SliceAgent", kind="agent", source="conversation_role"),
        effect_grants=grants,
        referents=tuple(referents),
    ))


@dataclass(frozen=True)
class IntentEntry:
    id: str
    verbatim_clause: str
    source_artifact: str | None = None
    source_range: tuple[int, int] | None = None
    authority: IntentAuthority = "task"
    kind: IntentKind = "constraint"
    status: IntentStatus = "active"
    superseded_by: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "verbatim_clause": self.verbatim_clause,
            "source_artifact": self.source_artifact,
            "source_range": list(self.source_range) if self.source_range is not None else None,
            "authority": self.authority,
            "kind": self.kind,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "IntentEntry | None":
        if not isinstance(value, Mapping):
            return None
        clause = value.get("verbatim_clause")
        if not isinstance(clause, str) or not clause.strip():
            return None
        raw_range = value.get("source_range")
        source_range = None
        if (isinstance(raw_range, (list, tuple)) and len(raw_range) == 2
                and all(isinstance(n, int) for n in raw_range)):
            source_range = (raw_range[0], raw_range[1])
        authority = value.get("authority")
        if authority not in ("user", "task", "legacy"):
            authority = "legacy"
        kind = value.get("kind")
        if kind not in ("constraint", "correction"):
            kind = "constraint"
        status = value.get("status")
        if status not in ("active", "provisionally_satisfied", "satisfied", "superseded", "deferred"):
            status = "active"
        refs = value.get("evidence_refs")
        return cls(
            id=str(value.get("id") or ""),
            verbatim_clause=clause,
            source_artifact=(str(value["source_artifact"])
                             if value.get("source_artifact") is not None else None),
            source_range=source_range,
            authority=authority,
            kind=kind,
            status=status,
            superseded_by=(str(value["superseded_by"])
                           if value.get("superseded_by") is not None else None),
            evidence_refs=tuple(str(r) for r in refs) if isinstance(refs, (list, tuple)) else (),
        )


@dataclass
class IntentState:
    current_request: str = ""
    current_source: str | None = None
    entries: list[IntentEntry] = field(default_factory=list)
    next_id: int = 1
    # Current-turn control state only. It is neither an intent-entry authority nor serialized task memory.
    turn_admission: TurnAdmission = field(default_factory=TurnAdmission, repr=False, compare=False)

    @property
    def turn_contract(self) -> TurnAdmission:
        """Compatibility projection; ``turn_admission`` is the sole mutable owner."""
        return self.turn_admission

    @turn_contract.setter
    def turn_contract(self, value: TurnAdmission) -> None:
        self.turn_admission = value

    def reset(self, request: str = "", *, source_artifact: str | None = None) -> None:
        self.current_request = str(request or "")
        self.current_source = source_artifact
        self.entries = []
        self.next_id = 1
        self.turn_admission = TurnAdmission()

    def begin_turn(
        self,
        request: str,
        *,
        source_artifact: str | None = None,
        admission: TurnAdmission | None = None,
        contract: TurnContract | None = None,
        prior_texts: Iterable[str] = (),
        pending_proposal: object = None,
        referents: Iterable[object] = (),
        previous_evidence_query: EvidenceQuery | None = None,
        previous_quality_evidence_query: QualityEvidenceQuery | None = None,
    ) -> None:
        """Install the verbatim request and its ephemeral authority contract for this turn.

        Existing callers need pass nothing new.  A host with richer discourse state may supply prior assistant
        text, a pending proposal, typed referents, or a pre-resolved contract; all paths feed the same exact-span
        reducer below.
        """
        if admission is not None and contract is not None and admission != contract:
            raise ValueError("pass one TurnAdmission, not competing admission/contract values")
        selected_admission = admission or contract
        self.current_request = str(request or "")
        self.current_source = source_artifact
        self.turn_admission = _validated_contract(
            self.current_request,
            selected_admission or analyze_turn(
                self.current_request, prior_texts=prior_texts,
                pending_proposal=pending_proposal, referents=referents,
                previous_evidence_query=previous_evidence_query,
                previous_quality_evidence_query=previous_quality_evidence_query,
            ),
        )
        if source_artifact is not None and self.turn_admission.request_source != source_artifact:
            self.turn_admission = replace(self.turn_admission, request_source=source_artifact)
        prior = tuple(self.resident_entries())
        captured = self.capture_explicit_constraints(
            prior, authority_spans=self.turn_admission.authority_spans,
        )
        self.reconcile_explicit_corrections(prior, captured)

    def reconcile_explicit_corrections(
        self, prior: Iterable[IntentEntry], captured: Iterable[IntentEntry],
    ) -> tuple[IntentEntry, ...]:
        """Link a high-confidence user correction to the exact older clause it names.

        The host does not guess semantic similarity. Supersession fires only when the current request uses
        explicit correction language and a newly captured exact constraint either repeats the older clause's
        word sequence or conservatively overlaps its significant words. More implicit rewrites remain
        available through ``supersede_requirement``.
        """
        scan = _scan_spans(self.current_request, self.turn_admission.authority_spans)
        if _EXPLICIT_CORRECTION.search(scan) is None:
            return ()
        # A correction cue elsewhere in the message (especially inside quoted evidence) cannot lend
        # supersession authority to an unrelated directive. Only the exact captured clause whose own masked
        # text was classified as a correction may retire an older user obligation.
        replacements = tuple(entry for entry in captured if entry.kind == "correction")
        candidates = [entry for entry in prior
                      if entry.status in ("active", "provisionally_satisfied")]
        changed = []
        used: set[str] = set()
        for replacement in replacements:
            scored = [
                (_correction_match_score(old.verbatim_clause, replacement.verbatim_clause), old)
                for old in candidates if old.id != replacement.id and old.id not in used
            ]
            best = max((score for score, _old in scored), default=0)
            matches = [old for score, old in scored if score == best and score > 0]
            # One replacement may retire at most one uniquely best older clause. A shared action verb alone
            # cannot delete several independent requirements; ambiguity keeps both until an explicit tool/user
            # transition names the target.
            if len(matches) != 1:
                continue
            old = matches[0]
            # Once linked, the replacement owns the superseded obligation and becomes a standing constraint.
            # Unlinked correction text stays a distinct clarification rather than being guessed into one.
            replacement = self._replace(replacement, kind="constraint")
            self._replace(old, status="superseded", superseded_by=replacement.id)
            used.add(old.id)
            changed.append(replacement)
        return tuple(changed)

    def capture_explicit_constraints(
        self, prior: Iterable[IntentEntry] = (), *,
        authority_spans: Iterable[SourceSpan] | None = None,
    ) -> tuple[IntentEntry, ...]:
        """Promote explicit durable clauses without copying the whole historical message."""
        text = self.current_request
        # ``None`` preserves the standalone/back-compat helper behavior. begin_turn always passes its exact
        # contract spans, including an intentionally empty set when the whole request is attributed data.
        scan = (_mask_quoted_data(text) if authority_spans is None
                else _scan_spans(text, authority_spans))
        prior_significant = tuple(
            _significant_tokens(entry.verbatim_clause) for entry in prior
            if entry.status in ("active", "provisionally_satisfied")
        )
        captured = []
        pending_correction_header = False
        start = 0
        spans = []
        for boundary in _CLAUSE_BOUNDARY.finditer(scan):
            spans.append((start, boundary.start()))
            start = boundary.end()
        spans.append((start, len(scan)))
        for raw_start, raw_end in spans:
            left = raw_start
            while left < raw_end and scan[left].isspace():
                left += 1
            prefix = _CLAUSE_PREFIX.match(scan, left, raw_end)
            if prefix is not None:
                left = prefix.end()
            right = raw_end
            while right > left and scan[right - 1].isspace():
                right -= 1
            clause = text[left:right]
            scan_clause = scan[left:right]
            if _BARE_CORRECTION_HEADER.fullmatch(scan_clause):
                # A syntactic header scopes exactly the immediately following clause; it is not itself a
                # durable obligation. Quoted headers were blanked before this point and cannot lend authority.
                pending_correction_header = True
                continue
            scoped_correction = pending_correction_header
            if scan_clause.strip():
                pending_correction_header = False
            if (scan_clause.rstrip().endswith("?") and _QUESTION_PREFIX.match(scan_clause)
                    and not _POLITE_DIRECTIVE_QUESTION.match(scan_clause)):
                # Modal words inside a question describe uncertainty, not a durable instruction. Preserve
                # polite `could you ensure ...?` requests, but do not turn `Do we need to never ...?` into law.
                continue
            clause_significant = _significant_tokens(scan_clause)
            related_actually = (
                _ACTUALLY_PREFIX.search(scan_clause) is not None
                and (
                    bool(_concrete_values(scan_clause))
                    or any(clause_significant.intersection(old) for old in prior_significant)
                    or _REVERSAL_LANGUAGE.search(scan_clause) is not None
                )
            )
            # The standing ledger is not a shadow task transcript. Generic action directives are already
            # represented by the exact CURRENT REQUEST and stable objective; promoting them here leaves a
            # stale unchecked ``Fix/Review/Spawn...`` obligation after a clean turn. Only clauses with actual
            # durable language, output/configuration choices, or explicit correction provenance cross turns.
            if not clause or not (
                _EXPLICIT_CONSTRAINT.search(scan_clause)
                or _DURABLE_OUTPUT_CONSTRAINT.search(scan_clause)
                or _STRONG_CORRECTION.search(scan_clause)
                or scoped_correction
                or (_ACTUALLY_PREFIX.search(scan_clause) and _DIRECTIVE_PREFIX.search(scan_clause))
                or related_actually
            ):
                continue
            entry = self.add_exact(
                clause, source_artifact=self.current_source,
                source_range=(left, right), authority="user",
                kind=("correction" if scoped_correction or _EXPLICIT_CORRECTION.search(scan_clause)
                      else "constraint"),
            )
            if entry is not None:
                captured.append(entry)
        return tuple(captured)

    def seal(self) -> None:
        """Standing entries are task-scoped; the effect/grounding contract is turn-scoped."""
        self.turn_admission = TurnAdmission()

    def _mint_id(self) -> str:
        live = {entry.id for entry in self.entries}
        while True:
            value = f"intent-{self.next_id}"
            self.next_id += 1
            if value not in live:
                return value

    def find(self, text: str, *, statuses: Iterable[str] | None = None) -> IntentEntry | None:
        key = _match_key(text)
        allowed = set(statuses) if statuses is not None else None
        for entry in self.entries:
            if _match_key(entry.verbatim_clause) == key and (allowed is None or entry.status in allowed):
                return entry
        return None

    def add_exact(
        self,
        clause: str,
        *,
        source_artifact: str | None = None,
        source_range: tuple[int, int] | None = None,
        authority: IntentAuthority = "task",
        kind: IntentKind = "constraint",
    ) -> IntentEntry | None:
        """Add a standing clause without a semantic count or character cap."""
        text = str(clause or "").strip()
        if not text:
            return None
        hit = self.find(text, statuses=("active", "provisionally_satisfied", "satisfied"))
        if hit is not None:
            # Session routing may see the request before its pending artifact id is allocated. When the
            # journal-backed record arrives, enrich the same authoritative entry rather than duplicating it.
            if (source_artifact is not None and hit.source_artifact is None
                    and hit.authority == authority):
                return self._replace(hit, source_artifact=source_artifact,
                                     source_range=source_range or hit.source_range,
                                     kind=(kind if hit.kind == "correction" else hit.kind))
            return hit
        entry = IntentEntry(
            id=self._mint_id(),
            verbatim_clause=text,
            source_artifact=source_artifact,
            source_range=source_range,
            authority=authority,
            kind=kind,
        )
        self.entries.append(entry)
        return entry

    def add_from_current_request(self, clause: str) -> IntentEntry | None:
        """Record a tool-supplied clause, preserving whether it is an exact user quote.

        A model-authored paraphrase remains useful task state, but it must not be
        mislabeled as user-authored intent.  Exact substring membership supplies
        the source range without an extra model call.
        """
        text = str(clause or "").strip()
        if not text:
            return None
        start = self.current_request.find(text)
        return self.add_exact(
            text,
            source_artifact=self.current_source,
            source_range=(start, start + len(text)) if start >= 0 else None,
            authority="user" if start >= 0 else "task",
        )

    def _replace(self, old: IntentEntry, **changes) -> IntentEntry:
        new = replace(old, **changes)
        self.entries[self.entries.index(old)] = new
        return new

    def mark_provisional(self, text: str, *, evidence_refs: Iterable[str] = ()) -> IntentEntry | None:
        hit = self.find(text, statuses=("active", "provisionally_satisfied"))
        if hit is None:
            return None
        refs = tuple(str(r) for r in evidence_refs if r)
        if not refs:
            return None  # model assertion alone is not verification evidence
        return self._replace(hit, status="provisionally_satisfied", evidence_refs=refs)

    def satisfy(self, text: str, *, evidence_refs: Iterable[str] = ()) -> IntentEntry | None:
        """Finalize satisfaction; callers must enforce the user/task-boundary policy."""
        hit = self.find(text, statuses=("active", "provisionally_satisfied"))
        if hit is None:
            return None
        refs = tuple(str(r) for r in evidence_refs if r) or hit.evidence_refs
        return self._replace(hit, status="satisfied", evidence_refs=refs)

    def defer_model_entry(self, text: str) -> IntentEntry | None:
        """Allow model-owned task state to leave the active set.

        User-authored entries deliberately cannot take this transition.  A model
        calling drop_requirement is not evidence that the user retracted it.
        """
        hit = self.find(text, statuses=("active", "provisionally_satisfied"))
        if hit is None or hit.authority == "user":
            return None
        return self._replace(hit, status="deferred")

    def supersede_from_user(
        self,
        old_text: str,
        new_clause: str,
        *,
        source_artifact: str | None = None,
        source_range: tuple[int, int] | None = None,
    ) -> IntentEntry | None:
        old = self.find(old_text, statuses=("active", "provisionally_satisfied", "superseded"))
        if old is None:
            return None
        # begin_turn may already have retained a wrapper such as ``Correction: use API v2 instead`` so the
        # correction cannot disappear before the model calls this transition. Canonicalize that same entry
        # to the exact replacement substring instead of creating a redundant second authority record.
        wrapper = next((entry for entry in self.resident_entries()
                        if entry.authority == "user" and entry.id != old.id
                        and new_clause in entry.verbatim_clause
                        and (source_artifact is None or entry.source_artifact in (None, source_artifact))), None)
        if wrapper is None and old.status == "superseded" and old.superseded_by:
            wrapper = next((entry for entry in self.entries
                            if entry.id == old.superseded_by and new_clause in entry.verbatim_clause), None)
        if wrapper is not None:
            new = self._replace(
                wrapper, verbatim_clause=new_clause,
                source_artifact=source_artifact or wrapper.source_artifact,
                source_range=source_range or wrapper.source_range,
                kind="constraint",
            )
        else:
            new = self.add_exact(
                new_clause,
                source_artifact=source_artifact,
                source_range=source_range,
                authority="user",
            )
        if new is None:
            return None
        if old.status != "superseded" or old.superseded_by != new.id:
            self._replace(old, status="superseded", superseded_by=new.id)
        return new

    def resident_entries(self) -> list[IntentEntry]:
        return [entry for entry in self.entries
                if entry.status in ("active", "provisionally_satisfied")]

    def open_entries(self) -> list[IntentEntry]:
        return [entry for entry in self.entries if entry.status == "active"]

    def as_legacy_requirements(self) -> list[dict]:
        """Read-only compatibility projection for old call sites/checkpoints."""
        return [
            {
                "text": entry.verbatim_clause,
                "done": entry.status in ("provisionally_satisfied", "satisfied"),
            }
            for entry in self.resident_entries()
            if entry.kind == "constraint"
        ]

    def load_legacy_requirements(self, requirements: Iterable[Mapping]) -> None:
        """Replace entries from a v1 requirement list without upgrading authority."""
        self.entries = []
        self.next_id = 1
        for item in requirements or ():
            if not isinstance(item, Mapping):
                continue
            entry = self.add_exact(str(item.get("text") or ""), authority="legacy")
            if entry is not None and item.get("done"):
                self._replace(entry, status="provisionally_satisfied")

    def to_records(self) -> list[dict]:
        return [entry.to_dict() for entry in self.entries]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping],
        *,
        current_request: str = "",
        current_source: str | None = None,
        next_id: int = 1,
    ) -> "IntentState":
        state = cls(current_request=str(current_request or ""), current_source=current_source)
        for index, raw in enumerate(records or ()):
            entry = IntentEntry.from_dict(raw)
            if entry is None:
                raise ValueError(f"invalid intent record at index {index}")
            if not entry.id:
                raise ValueError(f"intent record at index {index} has no id")
            if any(old.id == entry.id for old in state.entries):
                # A duplicate source id makes supersession references inherently ambiguous.  Guessing a new
                # id here silently rewires authority; authoritative v2 state must fail visibly instead.
                raise ValueError(f"duplicate intent id {entry.id!r}")
            state.entries.append(entry)

        by_id = {entry.id: entry for entry in state.entries}
        for entry in state.entries:
            target = entry.superseded_by
            if entry.status == "superseded":
                if not target:
                    raise ValueError(f"superseded intent {entry.id!r} has no replacement link")
                if target == entry.id:
                    raise ValueError(f"intent {entry.id!r} supersedes itself")
                if target not in by_id:
                    raise ValueError(
                        f"intent {entry.id!r} points to missing replacement {target!r}"
                    )
            elif target is not None:
                raise ValueError(
                    f"non-superseded intent {entry.id!r} carries replacement link {target!r}"
                )

        # Chained corrections are valid; cycles are not.  Validate after every id is known so record order
        # cannot change the result.
        for entry in state.entries:
            seen = {entry.id}
            cursor = entry
            while cursor.superseded_by:
                target = cursor.superseded_by
                if target in seen:
                    raise ValueError(f"cyclic intent supersession involving {target!r}")
                seen.add(target)
                cursor = by_id[target]
        state.next_id = max(int(next_id or 1), state.next_id)
        # Avoid reusing numeric ids from a restored state even if next_id was absent.
        for entry in state.entries:
            if entry.id.startswith("intent-") and entry.id[7:].isdigit():
                state.next_id = max(state.next_id, int(entry.id[7:]) + 1)
        return state
