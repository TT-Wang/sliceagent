"""Small runtime extension seams used by the execution loop.

The autonomous kernel has no permission mode or general action gate. A
host may observe lifecycle events, verify completion, apply a token budget, or add
the deliberately narrow catastrophic-command safeguard.

Hook return conventions (all optional, return None to no-op):
  before_step(step)                     -> {"stop_turn": bool, "reason": str} | None
  record_step_usage(usage)              -> {"stop_turn": bool} | None
  should_continue_after_stop(stop)      -> {"continue"|"park": bool, "exclusive"?: bool} | None
  assess_terminal_candidate(stop, text) -> {"continue": bool} | None
  preflight_tool(name, args)            -> ToolPreflight
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .safeguards import catastrophic_reason


@dataclass(frozen=True)
class ToolPreflight:
    stop: bool = False
    reason: str = ""
    kind: str = ""      # catastrophic | lifecycle; presentation never infers this from prose


PROCEED = ToolPreflight()


class Hooks:
    def before_step(self, step: int):
        return None

    def record_step_usage(self, usage: dict):
        return None

    def remaining_token_budget(self) -> int | None:
        """Return the remaining task-token allowance, or None when uncapped."""
        return None

    def should_continue_after_stop(self, stop_reason: str):
        return None

    def assess_terminal_candidate(self, stop_reason: str, candidate: str):
        """Optionally request one advisory rewrite before terminal publication.

        This is intentionally not a completion gate: implementations may nudge an obvious non-answer once,
        but must not grade quality, enforce response structure, or prevent a later candidate from publishing.
        """
        return None

    def preflight_tool(self, name: str, args: dict) -> ToolPreflight:
        return PROCEED

    def reset_for_turn(self):
        """Reset per-turn extension state once at the start of a user task."""
        return None

    # --- mutating seams (events can't mutate; these can) ---
    def prepare_messages(self, messages: list[dict]):
        """Last chance to transform the model-visible messages before the LLM call
        (e.g. inject context). Return new messages, or None to leave unchanged."""
        return None

    def prepare_tool_schemas(self, schemas: list[dict]):
        """Optionally narrow the tool surface for this turn before the first model call."""
        return None

    def transform_tool_result(self, name: str, args: dict, output: str):
        """Rewrite a tool result before it enters the slice (e.g. redaction, formatting).
        Return new output, or None to leave unchanged."""
        return None


class CompositeHooks(Hooks):
    """Fan a single hook surface out over several independent runtime extensions."""

    def __init__(self, *hooks: Hooks):
        self.hooks = hooks

    def before_step(self, step):
        for h in self.hooks:
            try:
                r = h.before_step(step)
                if isinstance(r, Mapping) and r.get("stop_turn"):
                    return r
            except Exception:
                continue
        return None

    def record_step_usage(self, usage):
        # materialize ALL results first — these callbacks have side effects (e.g. BudgetHook.spent +=), so a
        # generator-fed any() that short-circuits on the first stop_turn would skip trailing hooks' observation.
        flags = []
        for hook in self.hooks:
            try:
                result = hook.record_step_usage(usage)
                flags.append(bool(result.get("stop_turn")) if isinstance(result, Mapping) else False)
            except Exception:
                flags.append(False)
        return {"stop_turn": True} if any(flags) else None

    def remaining_token_budget(self):
        remaining = []
        for hook in self.hooks:
            try:
                value = hook.remaining_token_budget()
                if value is not None:
                    remaining.append(max(0, int(value)))
            except Exception:
                continue
        return min(remaining) if remaining else None

    def should_continue_after_stop(self, stop_reason):
        continuation = None
        for h in self.hooks:
            try:
                r = h.should_continue_after_stop(stop_reason)
                if not isinstance(r, Mapping):
                    continue
                # An exclusive lifecycle hook owns this completion edge. The signal is deliberately limited to
                # completion composition; ordinary tool execution remains autonomous except for catastrophic.
                if r.get("exclusive"):
                    return r
                if r.get("park"):
                    return r
                if continuation is None and r.get("continue"):
                    continuation = r
            except Exception:
                continue
        return continuation

    def assess_terminal_candidate(self, stop_reason, candidate):
        continuation = None
        for hook in self.hooks:
            callback = getattr(hook, "assess_terminal_candidate", None)
            if callback is None:
                continue
            try:
                result = callback(stop_reason, candidate)
            except Exception:
                continue  # an optional presentation nudge can never block an otherwise valid completion edge
            if not isinstance(result, Mapping):
                continue
            if continuation is None and result.get("continue"):
                continuation = result
        return continuation

    def preflight_tool(self, name, args):
        for h in self.hooks:
            try:
                result = h.preflight_tool(name, args)
                if result is not None and result.stop:
                    return result
            except Exception:
                # Preflight extensions are independent and fail open for their own opinion. In particular,
                # one optional lifecycle hook returning None or crashing must not skip the later catastrophic
                # floor in the same composite. Process interrupts still propagate.
                continue
        return PROCEED

    def prepare_messages(self, messages):
        changed = False
        for h in self.hooks:
            try:
                r = h.prepare_messages(messages)
                if r is not None:
                    messages, changed = r, True
            except Exception:
                continue
        return messages if changed else None

    def prepare_tool_schemas(self, schemas):
        changed = False
        current = list(schemas or ())
        for hook in self.hooks:
            callback = getattr(hook, "prepare_tool_schemas", None)
            if callback is None:
                continue
            try:
                result = callback(current)
                if result is not None:
                    current, changed = list(result), True
            except Exception:
                continue
        return current if changed else None

    def transform_tool_result(self, name, args, output):
        changed = False
        for h in self.hooks:
            try:
                r = h.transform_tool_result(name, args, output)
                if r is not None:
                    output, changed = r, True
            except Exception:
                continue
        return output if changed else None

    def reset_for_turn(self):
        for h in self.hooks:
            try:
                h.reset_for_turn()
            except Exception:
                continue


# --- concrete hooks ---

class OracleHook(Hooks):
    """Optional verification hook: when the model declares done, run an oracle (tests/lint).
    If it fails, record the failure into the slice and force another turn."""

    def __init__(self, oracle, on_feedback):
        self.oracle = oracle
        self.on_feedback = on_feedback  # callable(output:str) -> records into the slice

    def should_continue_after_stop(self, stop_reason):
        if stop_reason != "end_turn":
            return None
        try:
            result = self.oracle.verify()
            ok, output = result
        except Exception as e:  # noqa: BLE001 — a verify error must force another turn, never silently pass
            ok, output = False, f"verification could not run: {type(e).__name__}: {e}"
            result = None
        from .execution import ToolStatus, coerce_tool_status
        if result is not None and getattr(result, "status", None) is not None:
            if coerce_tool_status(result.status) is ToolStatus.INDETERMINATE:
                return {
                    "park": True,
                    "reason": "verification outcome is indeterminate; do not continue or seal cleanly"
                              + (f"\n{output}" if output else ""),
                }
        if ok:
            return None
        self.on_feedback(output)   # also record into the slice (for the NEXT turn's seed / durable cache)
        # CRITICAL: the failure detail must ride the MESSAGE channel — under the accumulate loop the seed
        # is built once and never re-rendered mid-turn, so a slice mutation (last_error) is invisible to
        # THIS turn's retry. Put `output` in `feedback` so the loop appends it as the model's next input.
        return {"continue": True, "feedback": f"Verification failed — fix this, then finish:\n{output}"}


class ActiveWorkContinuationHook(Hooks):
    """Give typed unfinished Active Work one bounded reconciliation pass before terminal prose.

    This is deliberately narrower than a semantic completion or quality gate.  The model chose to create the
    child records; the host only performs arithmetic over their typed lifecycle.  One unchanged frontier is
    nudged once, so a stale record cannot spin the turn forever. Metadata-only graph revisions do not re-arm the
    hook; adding/removing a child or changing its lifecycle does. ``ready`` is intentionally deliverable, while
    ``waiting_user`` is a legitimate dialogue boundary. This compatibility hook is not installed by the 0.3
    runtime: Active Work tracks user commitments, not child execution lifecycle.
    """

    def __init__(self, provider):
        self.provider = provider
        self._seen: set[tuple] = set()

    def reset_for_turn(self):
        self._seen.clear()

    def should_continue_after_stop(self, stop_reason):
        if stop_reason != "end_turn":
            return None
        snapshot = self.provider() if callable(self.provider) else self.provider
        if not isinstance(snapshot, tuple) or len(snapshot) not in {2, 3}:
            return None
        graph, logical_id = snapshot[:2]
        roots = tuple(
            root for root in getattr(graph, "unresolved_roots", ())
            if not logical_id or getattr(root, "logical_id", "") == logical_id
        )
        if not roots:
            return None
        root = roots[-1]

        pending = tuple(
            item for item in getattr(graph, "items", ())
            if getattr(item, "id", "") != root.id
            and getattr(item, "root_id", "") == root.id
            and getattr(item, "status", "") in {"open", "in_progress"}
        )
        if not pending:
            return None
        fingerprint = (root.id, tuple((item.id, item.status) for item in pending))
        if fingerprint in self._seen:
            return None
        self._seen.add(fingerprint)
        shown = pending[:12]
        rows = [
            f"- {item.id} [{item.status}]: {str(getattr(item, 'description', '') or '').strip()}"
            for item in shown
        ]
        if len(pending) > len(shown):
            rows.append(f"- (+{len(pending) - len(shown)} more unfinished child items)")
        return {
            "continue": True,
            "exclusive": True,
            "feedback": (
                "# HOST ACTIVE WORK FRONTIER (typed control state; not a new user request)\n"
                "A terminal response was deferred once because the current request still owns unfinished "
                "model-maintained child work:\n"
                + "\n".join(rows)
                + "\nContinue the remaining work, or reconcile it with update_work. Mark an item ready only "
                  "after its result is available; cancel or supersede it only when it is genuinely no longer "
                  "part of the exact request. A settled delegation batch is not proof that the parent scope is "
                  "complete."
            ),
        }


class DeliverableCompletionHook(Hooks):
    """Give an obvious non-delivery one bounded, response-only rewrite opportunity.

    This does not check report format, findings, source citations, or completeness. After one reminder the model's
    next terminal candidate publishes normally, even if imperfect; model judgment remains the completion owner.
    """

    def __init__(self, provider):
        self.provider = provider
        self._nudged: set[tuple[str, str]] = set()

    def reset_for_turn(self):
        self._nudged.clear()

    def assess_terminal_candidate(self, stop_reason, candidate):
        if stop_reason != "end_turn":
            return None
        snapshot = self.provider() if callable(self.provider) else self.provider
        if not isinstance(snapshot, tuple) or len(snapshot) != 2:
            return None
        requirement, current_logical_id = snapshot
        from .deliverables import DeliverableRequirement, assess_deliverable

        if requirement is not None and not isinstance(requirement, DeliverableRequirement):
            return None
        current_logical_id = str(current_logical_id or "")
        if requirement is not None and (
                not current_logical_id or requirement.logical_id != current_logical_id):
            return None
        assessment = assess_deliverable(requirement, candidate)
        if assessment.complete:
            return None
        key = (
            requirement.logical_id if requirement is not None else (current_logical_id or "current-turn"),
            requirement.kind if requirement is not None else "self-contained-response",
        )
        if key in self._nudged:
            return None
        self._nudged.add(key)
        detail = assessment.reason or "the response does not yet answer the request"
        return {
            "continue": True,
            "response_only": True,
            "feedback": (
                "# HOST RESPONSE NUDGE (not a new user request)\n"
                f"{detail}. The user cannot see private tool or child-report text. Answer the user's request now "
                "from the evidence already gathered, using whatever clear structure fits. Do not perform more "
                "searches merely to satisfy this reminder; mention a concrete evidence gap in the answer if needed."
            ),
        }


class CatastrophicSafeguardHook(Hooks):
    """Stop only a directly recognized catastrophic shell action.

    Parser uncertainty and every ordinary action abstain to the autonomous kernel. This is deliberately not a
    permission mode, confirmation system, or general risk classifier.
    """

    def preflight_tool(self, name, args):
        reason = catastrophic_reason(name, args)
        if reason is None:
            return PROCEED
        return ToolPreflight(
            True,
            f"Safety stop: refused a potentially catastrophic command ({reason}).",
            kind="catastrophic",
        )


class BudgetHook(Hooks):
    """Stop the turn once cumulative tokens cross a ceiling."""

    def __init__(self, max_total_tokens: int):
        self.max = max_total_tokens
        self.spent = 0

    def reset_for_turn(self):
        # PER-TURN budget: reset the tally at the start of each user task (run_turn calls this). Without
        # this, the cap silently became a whole-SESSION budget across the REPL. A true
        # session-wide cap, if ever wanted, should be a separate named hook — not this one.
        self.spent = 0

    def before_step(self, step):
        if self.spent >= self.max:
            return {"stop_turn": True, "reason": "token budget exhausted"}
        return None

    def record_step_usage(self, usage):
        self.spent += int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        return {"stop_turn": True} if self.spent >= self.max else None

    def remaining_token_budget(self):
        return max(0, self.max - self.spent)
