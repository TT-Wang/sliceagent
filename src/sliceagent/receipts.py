"""Canonical, immutable projections of one turn's durable execution journal.

The journal remains the write authority.  A :class:`TurnReceipt` is rebuilt from its
events at seal time and embedded in the existing turn artifact; it is not another
mutable store.  The projection deliberately separates logical requests, pre-handler
rejections, physical execution, typed settlement, and reducer-applied effects.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal


TurnDisposition = Literal[
    "completed", "completed_with_warnings", "paused", "blocked", "interrupted", "indeterminate",
]


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


def _one_line(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:max(0, limit - 1)].rstrip() + "…"


def _status(value: object) -> str:
    if isinstance(value, bool):
        return "succeeded" if value else "failed"
    text = str(value or "").strip().lower()
    return text if text in {"succeeded", "failed", "cancelled", "indeterminate"} else ""


@dataclass(frozen=True)
class OperationReceipt:
    """Lifecycle of one provider invocation, in provider-request order."""

    invocation_id: str
    name: str
    provider_index: int | None
    args: Mapping[str, Any]
    requested: bool
    rejected_before_execution: bool
    rejection_reason: str
    execution_started: bool
    settled: bool
    outcome_status: str
    outcome_text: str
    effect_ids: tuple[str, ...]
    applied_effect_ids: tuple[str, ...]
    artifact_refs: tuple[str, ...] = ()

    @property
    def disposition(self) -> str:
        if self.rejected_before_execution:
            return "rejected"
        if self.settled:
            return self.outcome_status or "settled"
        if self.execution_started:
            return "indeterminate"
        return "not_started"

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "name": self.name,
            "provider_index": self.provider_index,
            "args": _thaw(self.args),
            "requested": self.requested,
            "rejected_before_execution": self.rejected_before_execution,
            "rejection_reason": self.rejection_reason,
            "execution_started": self.execution_started,
            "settled": self.settled,
            "disposition": self.disposition,
            "outcome_status": self.outcome_status,
            "outcome_text": self.outcome_text,
            "effect_ids": list(self.effect_ids),
            "applied_effect_ids": list(self.applied_effect_ids),
            "artifact_refs": list(self.artifact_refs),
        }


@dataclass(frozen=True)
class TurnReceipt:
    """Read-only execution truth sealed inside one existing turn artifact."""

    turn_id: str
    turn_status: str
    disposition: TurnDisposition
    operations: tuple[OperationReceipt, ...]
    warnings: tuple[str, ...]
    artifact_refs: tuple[str, ...] = ()
    usage: Mapping[str, int | float] = field(default_factory=lambda: MappingProxyType({}))
    schema_version: int = 1

    @property
    def counts(self) -> Mapping[str, int]:
        operations = self.operations
        return MappingProxyType({
            "requested": sum(operation.requested for operation in operations),
            "rejected_before_execution": sum(operation.rejected_before_execution for operation in operations),
            "execution_started": sum(operation.execution_started for operation in operations),
            "settled": sum(operation.settled for operation in operations),
            "succeeded": sum(operation.disposition == "succeeded" for operation in operations),
            "failed": sum(operation.disposition == "failed" for operation in operations),
            "cancelled": sum(operation.disposition == "cancelled" for operation in operations),
            "indeterminate": sum(operation.disposition == "indeterminate" for operation in operations),
            "not_started": sum(operation.disposition == "not_started" for operation in operations),
            "effects_declared": sum(len(operation.effect_ids) for operation in operations),
            "effects_applied": sum(len(operation.applied_effect_ids) for operation in operations),
            "child_artifacts": len(set(
                ref for operation in operations for ref in operation.artifact_refs
            )),
            "artifact_refs": len(self.artifact_refs),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "turn_id": self.turn_id,
            "turn_status": self.turn_status,
            "disposition": self.disposition,
            "counts": dict(self.counts),
            "warnings": list(self.warnings),
            "artifact_refs": list(self.artifact_refs),
            "usage": _thaw(self.usage),
            "operations": [operation.to_dict() for operation in self.operations],
        }

    @classmethod
    def from_events(
        cls,
        events: Iterable[Mapping[str, Any]],
        *,
        turn_id: str = "",
        turn_status: str = "end_turn",
        artifact_refs: Iterable[str] = (),
        usage: Mapping[str, int | float] | None = None,
    ) -> "TurnReceipt":
        return derive_turn_receipt(
            events,
            turn_id=turn_id,
            turn_status=turn_status,
            artifact_refs=artifact_refs,
            usage=usage,
        )


def _turn_disposition(turn_status: str, operations: tuple[OperationReceipt, ...], warnings: tuple[str, ...]) \
        -> TurnDisposition:
    status = str(turn_status or "").strip().lower()
    if status == "indeterminate" or any(operation.disposition == "indeterminate" for operation in operations):
        return "indeterminate"
    if status in {"blocked", "stuck"}:
        return "blocked"
    if status in {"max_steps", "token_budget", "max_tokens", "overflow"}:
        return "paused"
    if status in {"aborted", "error", "filtered", "failed", "interrupted"}:
        return "interrupted"
    if warnings:
        return "completed_with_warnings"
    return "completed"


def derive_turn_receipt(
    events: Iterable[Mapping[str, Any]],
    *,
    turn_id: str = "",
    turn_status: str = "end_turn",
    artifact_refs: Iterable[str] = (),
    usage: Mapping[str, int | float] | None = None,
) -> TurnReceipt:
    """Purely reduce a journal event stream into one immutable receipt.

    New lifecycle records are preferred. Legacy ``tool-invocation`` / ``tool-outcome`` /
    ``semantic-transition`` records remain readable so old pending journals and direct
    persistence callers retain their recovery semantics.
    """
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    unapportioned_effects: list[str] = []

    def row(invocation_id: object) -> dict[str, Any]:
        identity = str(invocation_id or "unknown")
        if identity not in rows:
            rows[identity] = {
                "invocation_id": identity, "name": "", "provider_index": None, "args": {},
                "requested": False, "rejected": False, "rejection_reason": "", "started": False,
                "settled": False, "status": "", "text": "", "effects": [], "applied": [],
                "artifacts": [],
                "canonical_requested": False, "canonical_started": False, "canonical_settled": False,
            }
            order.append(identity)
        return rows[identity]

    def identify(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
        if payload.get("name") and not target["name"]:
            target["name"] = str(payload.get("name"))
        args = payload.get("args")
        if isinstance(args, Mapping) and not target["args"]:
            target["args"] = dict(args)
        index = payload.get("provider_index")
        if isinstance(index, int) and not isinstance(index, bool) and target["provider_index"] is None:
            target["provider_index"] = index

    def capture_effects(target: dict[str, Any], effects: object) -> None:
        """Index declared effect identities and typed child-report relationships."""
        if not isinstance(effects, (list, tuple)):
            return
        for effect in effects:
            if not isinstance(effect, Mapping):
                continue
            effect_id = str(effect.get("id") or "")
            if effect_id and effect_id not in target["effects"]:
                target["effects"].append(effect_id)
            if str(effect.get("kind") or "") != "child_artifact":
                continue
            effect_payload = effect.get("payload")
            if not isinstance(effect_payload, Mapping):
                continue
            artifact_id = str(effect_payload.get("artifact_id") or "")
            if artifact_id and artifact_id not in target["artifacts"]:
                target["artifacts"].append(artifact_id)

    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_type = str(event.get("type") or "")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        invocation_id = payload.get("invocation_id")
        if event_type == "tool-requested":
            target = row(invocation_id)
            identify(target, payload)
            target["requested"] = target["canonical_requested"] = True
        elif event_type == "tool-rejected":
            target = row(invocation_id)
            identify(target, payload)
            target["requested"] = True
            target["rejected"] = True
            target["rejection_reason"] = _one_line(payload.get("reason"))
        elif event_type == "tool-execution-started":
            target = row(invocation_id)
            identify(target, payload)
            target["requested"] = True
            target["started"] = target["canonical_started"] = True
        elif event_type == "tool-settled":
            target = row(invocation_id)
            identify(target, payload)
            outcome = payload.get("outcome")
            outcome = outcome if isinstance(outcome, Mapping) else payload
            target["requested"] = True
            target["settled"] = target["canonical_settled"] = True
            target["status"] = _status(outcome.get("status"))
            target["text"] = str(outcome.get("text") or "")
            capture_effects(target, outcome.get("effects"))
        elif event_type == "tool-effect-applied":
            target = row(invocation_id)
            effect_id = str(payload.get("effect_id") or "")
            if effect_id and effect_id not in target["applied"]:
                target["applied"].append(effect_id)
            if str(payload.get("kind") or "") == "child_artifact":
                effect_payload = payload.get("payload")
                if isinstance(effect_payload, Mapping):
                    artifact_id = str(effect_payload.get("artifact_id") or "")
                    if artifact_id and artifact_id not in target["artifacts"]:
                        target["artifacts"].append(artifact_id)
        elif event_type == "tool-invocation":
            target = row(invocation_id)
            identify(target, payload)
            target["requested"] = True
            # Legacy journals used tool-invocation to mean ToolStarted. New journals emit this
            # compatibility row alongside the canonical start, never for a pre-handler rejection.
            target["started"] = True
        elif event_type == "tool-outcome":
            target = row(invocation_id)
            outcome = payload.get("outcome")
            if not isinstance(outcome, Mapping):
                outcome = {}
            target["requested"] = True
            if not target["canonical_settled"]:
                target["settled"] = True
                target["status"] = _status(outcome.get("status"))
                target["text"] = str(outcome.get("text") or "")
                capture_effects(target, outcome.get("effects"))
                # Some migrated legacy journals carry an explicit policy/pre-handler marker alongside the
                # old compatibility ToolStarted row. Only that positive provenance may override the physical
                # start claim. Output prose is untrusted: a plugin can legitimately return the same prefix.
                denial_prefix = "Error: blocked by policy:"
                provenance = str(
                    payload.get("rejection_provenance") or outcome.get("rejection_provenance")
                    or payload.get("rejection_source") or outcome.get("rejection_source") or ""
                ).strip().lower()
                policy_denied = bool(
                    payload.get("policy_denied") is True or outcome.get("policy_denied") is True
                    or provenance in {"policy", "policy_denial", "policy_gate", "legacy_policy_gate"}
                )
                rejected_before_execution = bool(
                    payload.get("rejected_before_execution") is True
                    or outcome.get("rejected_before_execution") is True
                    or str(payload.get("rejection_phase") or outcome.get("rejection_phase") or "").lower()
                    in {"pre_execution", "before_execution"}
                )
                if (not target["canonical_requested"] and not target["canonical_started"]
                        and target["status"] == "failed"
                        and policy_denied and rejected_before_execution):
                    target["rejected"] = True
                    target["started"] = False
                    reason = str(
                        payload.get("rejection_reason") or outcome.get("rejection_reason") or ""
                    ).strip()
                    if not reason:
                        text = target["text"].lstrip()
                        reason = text[len(denial_prefix):].strip() if text.startswith(denial_prefix) else text
                    target["rejection_reason"] = _one_line(reason or "denied")
        elif event_type == "semantic-transition":
            transition_id = str(payload.get("transition_id") or "")
            transition = payload.get("transition")
            transition = transition if isinstance(transition, Mapping) else {}
            target_id = transition.get("invocation_id")
            if target_id:
                target = row(target_id)
                if transition_id and transition_id not in target["applied"]:
                    target["applied"].append(transition_id)
            elif transition_id:
                unapportioned_effects.append(transition_id)

    # Old direct journal callers did not attach invocation_id to transitions. Match an
    # effect ID only when exactly one operation declared it; otherwise leave it unclaimed.
    for effect_id in unapportioned_effects:
        candidates = [target for target in rows.values() if effect_id in target["effects"]]
        if len(candidates) == 1 and effect_id not in candidates[0]["applied"]:
            candidates[0]["applied"].append(effect_id)

    operations = tuple(OperationReceipt(
        invocation_id=target["invocation_id"],
        name=target["name"],
        provider_index=target["provider_index"],
        args=_freeze(target["args"]),
        requested=bool(target["requested"]),
        rejected_before_execution=bool(target["rejected"]),
        rejection_reason=target["rejection_reason"],
        execution_started=bool(target["started"] and not target["rejected"]),
        settled=bool(target["settled"] or target["rejected"]),
        outcome_status=target["status"],
        outcome_text=target["text"],
        effect_ids=tuple(target["effects"]),
        applied_effect_ids=tuple(target["applied"]),
        artifact_refs=tuple(target["artifacts"]),
    ) for identity in order for target in (rows[identity],))

    warnings = []
    for operation in operations:
        label = operation.name or operation.invocation_id
        if operation.rejected_before_execution:
            warnings.append(f"{label} rejected before execution: {operation.rejection_reason or 'denied'}")
        elif operation.disposition in {"failed", "cancelled", "indeterminate", "not_started"}:
            detail = _one_line(operation.outcome_text)
            warnings.append(f"{label} {operation.disposition}" + (f": {detail}" if detail else ""))
        unapplied = tuple(effect_id for effect_id in operation.effect_ids
                          if effect_id not in operation.applied_effect_ids)
        if unapplied:
            warnings.append(
                f"{label} settled but {len(unapplied)} declared effect(s) were not accepted by state reduction"
            )
    warning_tuple = tuple(warnings)
    return TurnReceipt(
        turn_id=str(turn_id),
        turn_status=str(turn_status),
        disposition=_turn_disposition(str(turn_status), operations, warning_tuple),
        operations=operations,
        warnings=warning_tuple,
        artifact_refs=tuple(dict.fromkeys(str(ref) for ref in artifact_refs if ref)),
        usage=_freeze(dict(usage or {})),
    )


_COMPACT_COUNT_KEYS = (
    "requested", "rejected_before_execution", "execution_started", "settled",
    "succeeded", "failed", "cancelled", "indeterminate", "not_started",
)
_SPAWN_TOOLS = frozenset({"spawn_agent", "spawn_explore", "spawn_subagent"})


def _count(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _operation_count_projection(operations: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in _COMPACT_COUNT_KEYS}
    for operation in operations:
        requested = bool(operation.get("requested"))
        rejected = bool(operation.get("rejected_before_execution"))
        started = bool(operation.get("execution_started")) and not rejected
        settled = bool(operation.get("settled")) or rejected
        disposition = str(operation.get("disposition") or "").strip().lower()
        if not disposition:
            if rejected:
                disposition = "rejected"
            elif settled:
                disposition = _status(operation.get("outcome_status")) or "settled"
            elif started:
                disposition = "indeterminate"
            else:
                disposition = "not_started"
        counts["requested"] += int(requested)
        counts["rejected_before_execution"] += int(rejected)
        counts["execution_started"] += int(started)
        counts["settled"] += int(settled)
        if disposition in counts:
            counts[disposition] += 1
    return counts


def compact_receipt_projection(receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a constant-size, presentation-safe projection of one sealed receipt.

    The commit event must not carry tool arguments, outputs, or an operation list back into the
    interactive frontend. It carries only lifecycle totals plus a delegation subset, enough for every
    renderer to make the same truthful completion claim.
    """
    if not isinstance(receipt, Mapping):
        return None
    raw_operations = receipt.get("operations")
    operations = tuple(
        operation for operation in (raw_operations if isinstance(raw_operations, (list, tuple)) else ())
        if isinstance(operation, Mapping)
    )
    derived = _operation_count_projection(operations)
    raw_counts = receipt.get("counts")
    raw_counts = raw_counts if isinstance(raw_counts, Mapping) else {}
    counts = {
        key: _count(raw_counts.get(key)) if key in raw_counts else derived[key]
        for key in _COMPACT_COUNT_KEYS
    }
    agents = _operation_count_projection(
        operation for operation in operations
        if str(operation.get("name") or "") in _SPAWN_TOOLS
    )
    child_refs = {
        str(ref)
        for operation in operations
        if str(operation.get("name") or "") in _SPAWN_TOOLS
        for ref in (operation.get("artifact_refs")
                    if isinstance(operation.get("artifact_refs"), (list, tuple)) else ())
        if ref
    }
    agents["child_artifacts"] = len(child_refs)
    warnings = receipt.get("warnings")
    warning_count = len(warnings) if isinstance(warnings, (list, tuple)) else 0
    return {
        "schema_version": 1,
        "turn_status": str(receipt.get("turn_status") or ""),
        "disposition": str(receipt.get("disposition") or ""),
        "counts": counts,
        "agents": agents,
        "warning_count": warning_count,
    }


def _subtract_counts(total: Mapping[str, Any], subset: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: max(0, _count(total.get(key)) - _count(subset.get(key)))
        for key in _COMPACT_COUNT_KEYS
    }


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or singular + "s")


def _lifecycle_parts(counts: Mapping[str, Any], *, agent: bool) -> tuple[str, ...]:
    requested = _count(counts.get("requested"))
    if not requested:
        return ()
    started = _count(counts.get("execution_started"))
    rejected = _count(counts.get("rejected_before_execution"))
    succeeded = _count(counts.get("succeeded"))
    failed = _count(counts.get("failed"))
    cancelled = _count(counts.get("cancelled"))
    indeterminate = _count(counts.get("indeterminate"))
    not_started = _count(counts.get("not_started"))

    noun = _plural(requested, "agent", "agents") if agent else _plural(
        requested, "operation", "operations",
    )
    if succeeded == requested and started == requested:
        return (f"{requested} {noun} succeeded",)
    if rejected == requested:
        return (f"{requested} {noun} rejected before start",)

    parts = [f"{'agents' if agent else 'operations'}: {requested} requested"]

    def status_part(value: int, status: str) -> None:
        if not value:
            return
        parts.append(f"{value} {status}")

    # Adverse facts lead so terminal truncation cannot hide them behind successful lifecycle detail.
    status_part(rejected, "rejected before start")
    status_part(failed, "failed")
    status_part(cancelled, "cancelled")
    status_part(indeterminate, "indeterminate")
    status_part(not_started, "not started")
    status_part(started, "started")
    status_part(succeeded, "succeeded")
    return tuple(parts)


def receipt_summary_parts(receipt: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Format canonical lifecycle totals without collapsing rejection into execution failure."""
    if not isinstance(receipt, Mapping):
        return ()
    counts = receipt.get("counts")
    agents = receipt.get("agents")
    counts = counts if isinstance(counts, Mapping) else {}
    agents = agents if isinstance(agents, Mapping) else {}
    return (
        *_lifecycle_parts(agents, agent=True),
        *_lifecycle_parts(_subtract_counts(counts, agents), agent=False),
    )


def receipt_has_adverse_lifecycle(receipt: Mapping[str, Any] | None) -> bool:
    """Whether the sealed projection contains facts that must outrank cosmetic completion detail."""
    if not isinstance(receipt, Mapping):
        return False
    counts = receipt.get("counts")
    if not isinstance(counts, Mapping):
        return False
    return any(_count(counts.get(key)) for key in (
        "rejected_before_execution", "failed", "cancelled", "indeterminate", "not_started",
    ))


def receipt_completion_label(receipt: Mapping[str, Any] | None, stop_reason: str) -> str:
    """Shared durable-boundary wording for plain, Rich, and live terminal modes."""
    reason = str(stop_reason or "turn")
    disposition = str(receipt.get("disposition") or "") if isinstance(receipt, Mapping) else ""
    if disposition == "indeterminate":
        return "indeterminate state saved"
    if disposition == "completed_with_warnings":
        return "turn saved with warnings" if reason == "end_turn" else f"{reason} state saved with warnings"
    return "turn saved" if reason == "end_turn" else f"{reason} state saved"


__all__ = [
    "OperationReceipt", "TurnDisposition", "TurnReceipt", "compact_receipt_projection",
    "derive_turn_receipt", "receipt_completion_label", "receipt_has_adverse_lifecycle",
    "receipt_summary_parts",
]
