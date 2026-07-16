"""Typed reducer for the active Slice.

The event bus carries presentation, persistence, and semantic events together.  This
module is the one semantic fold: typed event dispatch owns lifecycle reduction, typed
``ToolEffect`` families own durable effects, and the small name-based compatibility
table is isolated to legacy model tools that do not yet emit typed effects.

Keeping these boundaries explicit matters.  A tool name is an invocation label, not a
proof that an effect occurred; typed effects and receipts remain the authority for work,
resources, child artifacts, and exactly-once replay.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum, auto
from functools import singledispatchmethod
from typing import Any

from .active_work import WorkDelta
from .events import (
    AssistantText,
    Event,
    StepBegin,
    StepEnd,
    ToolResult,
    ToolStarted,
    TurnEnd,
    TurnInterrupted,
)
from .execution import ToolStatus, reconciliation_targets
from .regions import MAX_PLAN_CHARS, MAX_PLAN_ITEMS, record_action, record_note
from .subagent_contract import SubagentClaim
from .text_utils import one_line


class _CompatibilityFamily(Enum):
    """Legacy semantic families for tools that predate typed ``ToolEffect`` values."""

    WORLD = auto()
    INTENT = auto()
    PLAN = auto()
    WORK = auto()


_COMPATIBILITY_FAMILY: dict[str, _CompatibilityFamily] = {
    "reconcile_execution": _CompatibilityFamily.WORLD,
    "world_set": _CompatibilityFamily.WORLD,
    "world_clear": _CompatibilityFamily.WORLD,
    "require": _CompatibilityFamily.INTENT,
    "requirement_done": _CompatibilityFamily.INTENT,
    "drop_requirement": _CompatibilityFamily.INTENT,
    "supersede_requirement": _CompatibilityFamily.INTENT,
    "update_plan": _CompatibilityFamily.PLAN,
    "update_work": _CompatibilityFamily.WORK,
}

_DELEGATION_TOOLS = frozenset({"spawn_agent", "spawn_explore", "spawn_subagent"})
_DIRECT_EDIT_TOOLS = frozenset({"edit_file", "append_to_file", "str_replace"})
_DIRECTORY_SCOPE_TOOLS = frozenset({"list_files", "grep", "glob"})
_NON_PROGRESS_TOOLS = frozenset({
    "require", "requirement_done", "drop_requirement", "supersede_requirement",
    "update_plan", "update_work", "world_set", "world_clear",
})


def _event_args(event: ToolResult | ToolStarted) -> dict[str, Any]:
    """Return the dispatch contract's mapping without trusting third-party event producers."""

    return dict(event.args) if isinstance(event.args, Mapping) else {}


def _effects(event: ToolResult) -> tuple[Any, ...]:
    return tuple(getattr(getattr(event, "outcome", None), "effects", ()) or ())


def _canonical_tool_result(event: ToolResult) -> ToolResult:
    """Validate a typed outcome before any Slice mutation, then project its canonical fields.

    Legacy events without ``outcome`` remain readable.  Once a typed outcome is present, however, its
    invocation and status are semantic authority: contradictory compatibility fields are corruption, not a
    second vote.  Missing optional compatibility identity/status fields are filled from the outcome.
    """
    outcome = event.outcome
    if outcome is None:
        return event
    invocation = getattr(outcome, "invocation", None)
    if invocation is None or not isinstance(getattr(invocation, "args", None), Mapping):
        raise RuntimeError("typed tool outcome has an invalid invocation")
    canonical_args = dict(invocation.args)
    event_args = _event_args(event)
    canonical_status = getattr(getattr(outcome, "status", None), "value", "")
    if canonical_status not in {status.value for status in ToolStatus}:
        raise RuntimeError("typed tool outcome has an invalid status")
    mismatches = []
    if event.name != invocation.name:
        mismatches.append("name")
    if not isinstance(event.args, Mapping) or event_args != canonical_args:
        mismatches.append("args")
    if event.invocation_id and event.invocation_id != invocation.id:
        mismatches.append("invocation_id")
    if event.status and str(event.status).casefold() != canonical_status:
        mismatches.append("status")
    if bool(event.failing) != bool(outcome.failing):
        mismatches.append("failing")
    if str(event.output) != str(outcome.text):
        mismatches.append("output")
    if mismatches:
        raise RuntimeError(
            "typed tool outcome disagrees with compatibility fields: " + ", ".join(mismatches)
        )
    return replace(
        event,
        name=str(invocation.name),
        args=canonical_args,
        output=str(outcome.text),
        failing=bool(outcome.failing),
        status=canonical_status,
        invocation_id=str(invocation.id),
    )


@dataclass
class _ToolFrame:
    """Facts established while reducing one logical tool outcome."""

    slice: Any
    event: ToolResult
    args: dict[str, Any]
    effects: tuple[Any, ...]
    effect_ids: tuple[str, ...]
    call: dict[str, Any]
    newly_settled: bool
    effects_replayed: bool
    neutral_cancel: bool
    neutral_steer: bool
    new_finding: bool = False
    did_edit: bool = False
    repair_proven: bool = False
    repair_observed_before_call: bool = False


class SliceReducer:
    """Fold typed runtime events into the currently active :class:`Slice`.

    ``state`` may be a Slice or Session.  Session routing is resolved for every
    event so a workspace/topic transition redirects subsequent events atomically.
    Unknown presentation events intentionally reduce to a no-op.
    """

    def __init__(self, state: Any) -> None:
        self._state = state

    def __call__(self, event: Event) -> None:
        self.reduce(event)

    def _slice(self):
        # Local import avoids making pfc import this module while pfc's Slice and
        # compatibility helpers are still being defined.
        from .pfc import _active

        return _active(self._state)

    @singledispatchmethod
    def reduce(self, event: Event) -> None:
        """Ignore events that have no semantic Slice projection."""

    @reduce.register
    def _(self, event: StepBegin) -> None:
        self._slice().runtime.step = event.step

    @reduce.register
    def _(self, event: StepEnd) -> None:
        s = self._slice()
        for key, value in (event.usage or {}).items():
            if isinstance(value, (int, float)):
                s.runtime.usage[key] = s.runtime.usage.get(key, 0) + value

    @reduce.register
    def _(self, event: TurnEnd) -> None:
        s = self._slice()
        open_plan = any(
            not isinstance(item, dict) or item.get("status") != "done"
            for item in (s.plan or ())
        )
        unresolved = bool(
            s.last_error or s.open_report or s.reconciliation_required
            or s.intent.open_entries() or open_plan
        )
        if event.stop_reason == "end_turn" and not unresolved:
            s.task.mark_objective_provisional()
        else:
            s.task.activate_objective()

    @reduce.register
    def _(self, event: TurnInterrupted) -> None:
        s = self._slice()
        s.task.activate_objective()
        if event.message and event.reason != "aborted":
            s.last_error = one_line(f"turn {event.reason}: {event.message}", 400)

    @reduce.register
    def _(self, event: ToolStarted) -> None:
        s = self._slice()
        call_id = getattr(getattr(event, "invocation", None), "id", "")
        call = next((
            item for item in reversed(s.runtime.recent_calls)
            if call_id and item.get("id") == call_id
        ), None)
        if call is None:
            s.runtime.recent_calls.append({
                "id": call_id,
                "name": event.name,
                "args": _event_args(event),
                "status": "running",
                "step": s.runtime.step,
            })
        elif str(call.get("status") or "") == "running":
            call.update(name=event.name, args=_event_args(event), step=s.runtime.step)

    @reduce.register
    def _(self, event: AssistantText) -> None:
        s = self._slice()
        # Tool-bearing/progress prose is presentation state, not conversation truth.  Only the one response
        # actually delivered to the user becomes cross-turn continuity.
        if not event.final or not s.conversation or not (event.content or "").strip():
            return
        s.conversation[-1]["assistant"] = str(event.content)
        from .discourse import extract_pending_proposal

        s.continuity.pending_proposal = extract_pending_proposal(event.content)

    @reduce.register
    def _(self, event: ToolResult) -> None:
        event = _canonical_tool_result(event)
        frame = self._begin_tool_result(event)
        s = frame.slice

        self._record_uncertainty(frame)
        if frame.newly_settled and event.failing and not frame.neutral_cancel:
            s.runtime.blocked_calls += 1
        if frame.effects_replayed or not event.apply_effects:
            return
        if frame.neutral_cancel or frame.neutral_steer:
            s.since_edit += 1
            return

        frame.new_finding = record_note(
            s,
            frame.args.get("note", ""),
            source="tool-note" if not event.failing else "claim",
        )
        self._reduce_compatibility_family(frame)
        self._reduce_delegated_testimony(frame)
        frame.repair_observed_before_call = s.runtime.report_repair_observed
        self._reduce_resource_family(frame)
        self._finish_tool_result(frame)

    # ---- Tool outcome lifecycle -------------------------------------------------

    def _begin_tool_result(self, event: ToolResult) -> _ToolFrame:
        s = self._slice()
        effects = _effects(event)
        effect_ids = tuple(
            str(effect.id) for effect in effects if getattr(effect, "id", "")
        )
        seen_effect_ids = s.runtime.applied_effect_ids.intersection(effect_ids)
        if seen_effect_ids and len(seen_effect_ids) != len(set(effect_ids)):
            raise RuntimeError("partially replayed tool outcome cannot be reduced safely")
        effects_replayed = bool(
            effect_ids and len(seen_effect_ids) == len(set(effect_ids))
        )

        outcome_invocation = getattr(getattr(event, "outcome", None), "invocation", None)
        call_id = event.invocation_id or getattr(outcome_invocation, "id", "")
        call = next((
            item for item in reversed(s.runtime.recent_calls)
            if call_id and item.get("id") == call_id
        ), None)
        prior_status = str(call.get("status") or "") if call is not None else ""
        accept_terminal_metadata = not prior_status or prior_status == "running"
        args = _event_args(event)
        if call is None:
            call = {
                "id": call_id,
                "name": event.name,
                "args": args,
                "step": s.runtime.step,
            }
            s.runtime.recent_calls.append(call)
        call["status"] = event.status or ("failed" if event.failing else "succeeded")
        if accept_terminal_metadata:
            self._capture_child_artifact(call, event.name, effects)
            if (event.name in _DELEGATION_TOOLS and call.get("child_operational_status")
                    and not event.failing and str(event.output or "").strip()):
                # The complete child outcome is already in the parent trajectory. Record that delivery
                # independently from whether an optional artifact was persisted.
                call["child_digest_delivered"] = True

        return _ToolFrame(
            slice=s,
            event=event,
            args=args,
            effects=effects,
            effect_ids=effect_ids,
            call=call,
            newly_settled=not prior_status or prior_status == "running",
            effects_replayed=effects_replayed,
            neutral_cancel=str(event.status or "").casefold() == "cancelled",
            neutral_steer=str(event.status or "").casefold() == "steered",
        )

    @staticmethod
    def _capture_child_artifact(
        call: dict[str, Any], name: str, effects: tuple[Any, ...],
    ) -> None:
        from .fan_in import (
            normalize_evidence_account,
            normalize_evidence_status,
            normalize_integration_policy,
        )

        if name not in _DELEGATION_TOOLS:
            return
        # Read the direct outcome first, then merge optional legacy/artifact metadata such as locators.
        relevant = sorted(
            (effect for effect in effects
             if getattr(effect, "kind", "") in {"child_outcome", "child_artifact"}),
            key=lambda effect: 0 if getattr(effect, "kind", "") == "child_outcome" else 1,
        )
        for effect in relevant:
            payload = getattr(effect, "payload", {}) or {}
            artifact_id = str(payload.get("artifact_id") or "")
            if artifact_id:
                call["child_artifact_id"] = artifact_id[:200]
            target = payload.get("delegation_target")
            if isinstance(target, str) and target.strip() and len(target.encode("utf-8")) <= 300:
                call["child_target"] = target.strip()
            work_item_id = payload.get("work_item_id")
            if (isinstance(work_item_id, str) and work_item_id.strip()
                    and len(work_item_id.encode("utf-8")) <= 200):
                call["child_work_item_id"] = work_item_id.strip()
            source_status = str(
                payload.get("source_coverage_status") or payload.get("epistemic_status") or ""
            ).strip().casefold()
            source_status = {
                "grounded": "source_complete", "partial": "source_partial",
                "unsupported": "source_unsupported",
            }.get(source_status, source_status)
            if source_status in {
                "source_complete", "source_partial", "source_unsupported", "not_assessed",
            }:
                call["child_source_coverage_status"] = source_status
            # Evidence retention and parent integration policy are deliberately independent from the
            # synthesiser's source_coverage_status. Missing legacy fields degrade in the live projection, but
            # are not persisted as if that producer explicitly declared a v1 evidence account or policy.
            evidence_account_declared = (
                "explorer_evidence" in payload or "evidence_account" in payload
            )
            account = normalize_evidence_account(
                payload.get("explorer_evidence")
                if "explorer_evidence" in payload else payload.get("evidence_account")
            )
            evidence_status_declared = (
                "explorer_evidence_status" in payload or "evidence_status" in payload
                or evidence_account_declared
            )
            if evidence_status_declared:
                call["child_evidence_status"] = normalize_evidence_status(
                    payload.get("explorer_evidence_status")
                    if "explorer_evidence_status" in payload else (
                        payload.get("evidence_status") if "evidence_status" in payload
                        else account.get("status")
                    )
                )
            if account:
                call["child_evidence_account"] = account
            policy_value = payload.get("integration_policy")
            if not policy_value and (payload.get("report_required") is True
                                     or account.get("report_required") is True):
                policy_value = "report_required"
            policy_declared = (
                "integration_policy" in payload or isinstance(payload.get("report_required"), bool)
                or isinstance(account.get("report_required"), bool)
            )
            if policy_declared:
                call["child_integration_policy"] = normalize_integration_policy(policy_value)
            operational = str(payload.get("operational_status") or payload.get("status") or "").strip()
            if operational:
                call["child_operational_status"] = operational[:40]
            for target, source in (
                ("child_required_ref_count", "required_ref_count"),
                ("child_consumed_ref_count", "consumed_refs"),
                ("child_cited_ref_count", "cited_refs"),
                ("child_covered_ref_count", "covered_refs"),
                ("child_source_gap_count", "source_gaps"),
            ):
                legacy_source = {
                    "covered_refs": "grounding_refs", "source_gaps": "grounding_gaps",
                }.get(source, "")
                value = payload.get(source) or (payload.get(legacy_source) if legacy_source else ()) or ()
                if source == "required_ref_count":
                    try:
                        count = max(0, min(int(value or 0), 10_000))
                    except (TypeError, ValueError, OverflowError):
                        count = 0
                else:
                    count = min(len(value), 10_000) if isinstance(value, (list, tuple)) else 0
                call[target] = count
            raw_scope = payload.get("scope") or ()
            if (isinstance(raw_scope, (list, tuple)) and len(raw_scope) <= 16
                    and all(
                        isinstance(item, str) and item.strip()
                        and len(item.encode("utf-8")) <= 300
                        for item in raw_scope
                    )):
                call["child_scope"] = list(dict.fromkeys(
                    item.strip() for item in raw_scope
                ))
            normalized_claims: list[dict] = []
            raw_claims = payload.get("claims") or ()
            valid_claims = isinstance(raw_claims, (list, tuple)) and len(raw_claims) <= 3
            if valid_claims:
                for row in raw_claims:
                    if (not isinstance(row, Mapping)
                            or not isinstance(row.get("text"), str)
                            or not isinstance(row.get("report_exact"), str)):
                        valid_claims = False
                        break
                    try:
                        normalized_claims.append(SubagentClaim.from_dict(row).to_dict())
                    except (TypeError, ValueError):
                        valid_claims = False
                        break
            if valid_claims and normalized_claims:
                call["child_claims"] = normalized_claims

    @staticmethod
    def _record_uncertainty(frame: _ToolFrame) -> None:
        if not frame.newly_settled or frame.call["status"] != "indeterminate":
            return
        targets = reconciliation_targets(frame.event.name, frame.args)
        if not targets:
            return
        call_id = str(frame.call.get("id") or "unknown invocation")
        detail = (
            f"{frame.event.name} ({call_id}) returned an indeterminate outcome; "
            "its effect remains unknown and should be re-observed if relevant"
        )
        s = frame.slice
        if detail not in s.reconciliation_required:
            s.reconciliation_required = " | ".join(
                item for item in (s.reconciliation_required, detail) if item
            )
        s.reconciliation_targets = list(dict.fromkeys((
            *s.reconciliation_targets, *targets,
        )))

    # ---- Semantic effect families ---------------------------------------------

    def _reduce_compatibility_family(self, frame: _ToolFrame) -> None:
        if frame.event.failing:
            return
        family = _COMPATIBILITY_FAMILY.get(frame.event.name)
        if family is None:
            return
        handlers = {
            _CompatibilityFamily.WORLD: self._reduce_world_effect,
            _CompatibilityFamily.INTENT: self._reduce_intent_effect,
            _CompatibilityFamily.PLAN: self._reduce_plan_effect,
            _CompatibilityFamily.WORK: self._reduce_work_effect,
        }
        handlers[family](frame)

    @staticmethod
    def _reduce_world_effect(frame: _ToolFrame) -> None:
        s, event, args = frame.slice, frame.event, frame.args
        if event.name == "reconcile_execution":
            resolution = one_line(str(args.get("resolution") or "state re-observed"), 300)
            s.reconciliation_required = ""
            s.reconciliation_targets = []
            s.task.add_progress("reconciliation", resolution)
            s.last_error = ""
        elif event.name == "world_set":
            key = str(args.get("key", "")).strip()
            if key:
                s.world[key] = str(args.get("value", ""))
        else:  # world_clear
            key = str(args.get("key", "")).strip()
            if key:
                s.world.pop(key, None)
            else:
                s.world.clear()

    @staticmethod
    def _reduce_intent_effect(frame: _ToolFrame) -> None:
        s, event, args = frame.slice, frame.event, frame.args
        text = str(args.get("text", "")).strip()
        if event.name == "supersede_requirement":
            old = str(args.get("old_text", "")).strip()
            new = str(args.get("new_text", "")).strip()
            start = s.intent.current_request.find(new) if new else -1
            if old and new and start >= 0:
                s.intent.supersede_from_user(
                    old,
                    new,
                    source_artifact=s.intent.current_source,
                    source_range=(start, start + len(new)),
                )
            return
        if not text:
            return
        if event.name == "require":
            s.intent.add_from_current_request(text)
        elif event.name == "requirement_done":
            prior = next((
                call for call in reversed(s.runtime.recent_calls[:-1])
                if call.get("status") == "succeeded"
                and call.get("name") not in _NON_PROGRESS_TOOLS
                and call.get("id")
            ), None)
            if prior is not None:
                s.intent.mark_provisional(
                    text, evidence_refs=(f"invocation:{prior['id']}",),
                )
        else:  # drop_requirement
            s.intent.defer_model_entry(text)

    @staticmethod
    def _reduce_plan_effect(frame: _ToolFrame) -> None:
        new_plan = []
        for item in (frame.args.get("steps") or [])[:MAX_PLAN_ITEMS]:
            if not isinstance(item, dict):
                continue
            step = " ".join(str(item.get("step", "")).split())[:MAX_PLAN_CHARS]
            status = str(item.get("status", "pending")).strip().lower()
            if status not in ("pending", "in_progress", "done"):
                status = "pending"
            if step:
                new_plan.append({"step": step, "status": status})
        frame.slice.plan = new_plan

    @staticmethod
    def _reduce_work_effect(frame: _ToolFrame) -> None:
        effect = next((
            effect for effect in frame.effects
            if getattr(effect, "kind", "") == "work_delta"
        ), None)
        if effect is None:
            raise RuntimeError("successful update_work outcome is missing its typed WorkDelta")
        payload = getattr(effect, "payload", {}) or {}
        frame.slice.active_work = frame.slice.active_work.apply_delta(
            WorkDelta.from_dict(payload.get("delta") or {})
        )

    @staticmethod
    def _reduce_delegated_testimony(frame: _ToolFrame) -> None:
        # Delegation results already live in the current tool trajectory and in the turn ledger.  Promoting a
        # full report into PFC findings duplicates it, while auto-advancing Active Work couples scheduler
        # lifecycle to user commitments and creates stale-revision races.  Observers consume typed effects.
        return

    @staticmethod
    def _reduce_resource_family(frame: _ToolFrame) -> None:
        from .pfc import add_skill, edited_paths_in_code, paths_in_code, touch_file

        s, event, args = frame.slice, frame.event, frame.args
        if event.name == "skill" and not event.failing:
            add_skill(s, args.get("name", ""), event.output)
            activation = next((
                effect for effect in frame.effects
                if getattr(effect, "kind", "") == "skill_activated"
            ), None)
            payload = getattr(activation, "payload", {}) or {}
            roots = tuple(getattr(s.active_work, "unresolved_roots", ()) or ())
            if activation is not None and roots:
                from .deliverables import requirement_for_contract

                name = str(payload.get("name") or args.get("name") or "").strip()
                requirement = requirement_for_contract(
                    payload.get("completion_contract"),
                    logical_id=str(getattr(roots[-1], "logical_id", "") or ""),
                    source=f"skill:{name}",
                )
                if requirement is not None:
                    s.task.bind_deliverable(requirement)
                    s.task.add_progress("deliverable", f"{requirement.kind} required")

        path = args.get("path")
        if path and event.name not in _DIRECTORY_SCOPE_TOOLS:
            frame.did_edit = event.name in _DIRECT_EDIT_TOOLS and not event.failing
            frame.repair_proven = frame.did_edit
            resource_effect = next((
                effect for effect in frame.effects
                if getattr(effect, "kind", "") == "resource_observed"
            ), None)
            resource_kind = str(
                getattr(resource_effect, "payload", {}).get("resource_kind", "")
                if resource_effect is not None else ""
            )
            if resource_effect is not None and not event.failing:
                payload = getattr(resource_effect, "payload", {}) or {}
                artifact_id = str(payload.get("artifact_id") or "").strip()
                read_coverage = str(payload.get("read_coverage") or "").strip().casefold()
                if artifact_id:
                    frame.call["observed_artifact_id"] = artifact_id[:200]
                if read_coverage in {"partial", "complete"}:
                    frame.call["observed_read_coverage"] = read_coverage
                artifact_view = str(payload.get("artifact_view") or "").strip().casefold()
                if artifact_view in {"report", "evidence"}:
                    frame.call["observed_artifact_view"] = artifact_view
                handle = str(payload.get("handle") or "").strip()
                if handle:
                    frame.call["observed_resource_handle"] = handle[:500]
                if resource_kind:
                    frame.call["observed_resource_kind"] = resource_kind[:40]
            virtual_read = event.name == "read_file" and resource_kind in {
                "artifact", "history", "subagent", "roster", "internal_context",
            }
            if not event.failing and not virtual_read:
                touch_file(s, path, edited=frame.did_edit)
            if event.name == "str_replace" and args.get("old_string"):
                anchor = next((
                    line.strip() for line in str(args["old_string"]).splitlines()
                    if line.strip()
                ), "")
                if anchor:
                    s.edit_anchor[path] = anchor[:80]
        elif event.name == "execute_code":
            code = args.get("code", "")
            mutated = set(edited_paths_in_code(code))
            frame.did_edit = bool(mutated) and not event.failing
            if not event.failing:
                for touched_path in paths_in_code(code):
                    touch_file(s, touched_path, edited=touched_path in mutated)

    @staticmethod
    def _finish_tool_result(frame: _ToolFrame) -> None:
        from .pfc import _verification_families_call, _verification_matches_report

        s, event, args = frame.slice, frame.event, frame.args
        record_action(
            s,
            event.name,
            args,
            event.output,
            failing=bool(event.failing and not frame.neutral_cancel),
        )
        if event.failing and not frame.neutral_cancel:
            s.task.add_progress("blocked", f"{event.name} failed")
        if frame.did_edit:
            s.task.add_progress("edit", str(args.get("path") or event.name))
        if frame.repair_proven and s.open_report:
            s.runtime.report_repair_observed = True
            s.runtime.report_verification_families.clear()
        verification_families = _verification_families_call(event.name, args)
        if s.open_report and frame.repair_observed_before_call and verification_families:
            if event.failing:
                s.runtime.report_verification_families.difference_update(verification_families)
            else:
                s.runtime.report_verification_families.update(verification_families)
        if (s.open_report and frame.repair_observed_before_call
                and _verification_matches_report(
                    s.open_report, s.runtime.report_verification_families,
                )):
            s.open_report = ""
            s.runtime.report_repair_observed = False
            s.runtime.report_verification_families.clear()
            s.task.add_progress("verification", "user-reported defect verified after repair")
        if frame.new_finding:
            s.task.add_progress("evidence", f"new evidence from {event.name}")
        s.since_edit = 0 if (frame.did_edit or frame.new_finding) else s.since_edit + 1
        s.runtime.applied_effect_ids.update(frame.effect_ids)
