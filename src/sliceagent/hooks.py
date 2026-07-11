"""Hooks: the policy seam. The loop calls these; the host supplies them.

This is how policy stays OUT of the moat: the Oracle, permission gate, and token
budget are all hooks, not hardcoded loop logic.

Hook return conventions (all optional, return None to no-op):
  before_step(step)                     -> {"block": bool, "reason": str} | None
  record_step_usage(usage)              -> {"stop_turn": bool} | None
  after_step(step, usage, stop_reason)  -> {"stop_turn": bool} | None
  validate_completion(text, stop)       -> {"continue"|"park": bool, "replacement"?: str} | None
  should_continue_after_stop(stop)      -> {"continue"|"park": bool, "exclusive"?: bool} | None
  authorize_tool(name, args)            -> ToolDecision
"""
from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass

from .context import ResourceKind
from .guardrails import ToolCallGuardrail
from .guidance import DENIAL_NO_PROMPT, DENIAL_USER
from .registry import ToolIntentEffect, coerce_intent_effect

# Commands a wide AGENT_AUTO_APPROVE glob (e.g. "git *") must NEVER silently approve: destructive ops that
# are not catastrophic (so the policy floor lets them through to ASK) yet discard work/data. These always
# fall through to a confirmation even when a glob matches. The catastrophic floor is screened too (below).
_DESTRUCTIVE_AUTO = [
    re.compile(r"\bgit\b[^\n]*\b(reset|clean|checkout|restore|rebase|filter-branch)\b", re.I),
    re.compile(r"\bgit\b[^\n]*\bbranch\b[^\n]*\s-D\b", re.I),         # re.I → also catches -d (deletes a branch ref)
    re.compile(r"\bgit\b[^\n]*\bstash\b[^\n]*\b(drop|clear)\b", re.I),
    re.compile(r"\bgit\b[^\n]*\bpush\b", re.I),                      # H8: ANY push, not just --force — OUTWARD and
    #                                                                 publishes history (a plain push is hard to undo too)
    re.compile(r"\b(npm|yarn|pnpm)\b[^\n]*\bpublish\b|\btwine\b[^\n]*\bupload\b"
               r"|\b(cargo|poetry)\b[^\n]*\bpublish\b|\bgem\b[^\n]*\bpush\b", re.I),  # publish a package — outward, irreversible
    re.compile(r"\brm\b(?=[^|;&\n]*\s-[a-z]*r)", re.I),              # any recursive rm
    re.compile(r"\brmdir\b", re.I),                                 # removes a directory
    re.compile(r"\b(shred|mkfs|wipefs)\b", re.I),
]


def _is_destructive_command(name: str, cmd: str) -> bool:
    """True if `cmd` must never be silently auto-approved — catastrophic OR work-discarding."""
    from . import policy   # deferred: policy imports hooks, so import here to avoid a cycle at load
    if policy.no_dangerous_commands(name, {"command": cmd}) is not None:
        return True
    return any(p.search(cmd) for p in _DESTRUCTIVE_AUTO)


@dataclass
class ToolDecision:
    allow: bool
    reason: str = ""
    ask: bool = False   # policy abstains to an interactive prompt (resolved by PermissionHook)
    # Does this block count toward the per-turn STUCK floor (loop.py STUCK_BLOCK_BUDGET)? True for a genuine
    # SPIN (a repeated failing call, a policy denial the model keeps retrying); FALSE for a harmless dedup
    # (re-reading the same file → the guard just skips it). So a long, legit exploration that re-reads a file
    # a few times is NOT killed as "stuck" — only real spinning is.
    counts_as_stuck: bool = True


ALLOW = ToolDecision(True)


class Hooks:
    def before_step(self, step: int):
        return None

    def record_step_usage(self, usage: dict):
        return None

    def remaining_token_budget(self) -> int | None:
        """Return the remaining task-token allowance, or None when uncapped."""
        return None

    def after_step(self, step: int, usage: dict, stop_reason: str):
        return None

    def validate_completion(self, candidate: str, stop_reason: str):
        """Inspect unpublished terminal prose before ordinary completion hooks run.

        This is separate from ``should_continue_after_stop`` so existing hooks and plugins keep their stable
        one-argument ABI. A validator may request a retry, park, or replace unsafe prose with a conservative
        answer. The loop never dispatches the rejected candidate as final assistant truth.
        """
        return None

    def should_continue_after_stop(self, stop_reason: str):
        return None

    def authorize_tool(self, name: str, args: dict) -> ToolDecision:
        return ALLOW

    def reset_for_turn(self):
        """Reset any per-turn state at the start of a user task (fires ONCE per turn,
        not per step). Used by the guardrail to clear cross-step loop counters so they
        do not bleed across tasks. No-op by default."""
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
    """Fan a single hook surface out over several hooks (first deny / any stop / any continue)."""

    def __init__(self, *hooks: Hooks):
        self.hooks = hooks

    def before_step(self, step):
        for h in self.hooks:
            r = h.before_step(step)
            if r and r.get("block"):
                return r
        return None

    def record_step_usage(self, usage):
        # materialize ALL results first — these callbacks have side effects (e.g. BudgetHook.spent +=), so a
        # generator-fed any() that short-circuits on the first stop_turn would skip trailing hooks' observation.
        flags = [(h.record_step_usage(usage) or {}).get("stop_turn") for h in self.hooks]
        return {"stop_turn": True} if any(flags) else None

    def remaining_token_budget(self):
        remaining = [value for h in self.hooks
                     if (value := h.remaining_token_budget()) is not None]
        return min(remaining) if remaining else None

    def after_step(self, step, usage, stop_reason):
        flags = [(h.after_step(step, usage, stop_reason) or {}).get("stop_turn") for h in self.hooks]
        return {"stop_turn": True} if any(flags) else None

    def validate_completion(self, candidate, stop_reason):
        decision = None
        for h in self.hooks:
            # Third-party hooks written against the older surface may be duck-typed rather than subclasses.
            # The new seam is optional for them; do not turn an additive hook into an ABI break.
            callback = getattr(h, "validate_completion", None)
            if callback is None:
                continue
            r = callback(candidate, stop_reason)
            if r and r.get("exclusive"):
                return r
            if r and r.get("park"):
                return r
            if decision is None and r and (r.get("continue") or "replacement" in r):
                decision = r
        return decision

    def should_continue_after_stop(self, stop_reason):
        continuation = None
        for h in self.hooks:
            r = h.should_continue_after_stop(stop_reason)
            # Some completion gates own the turn while active.  In particular, reconciliation must run
            # before verification/oracle hooks because those hooks can execute commands.  ``exclusive`` is
            # deliberately a completion-only composition signal: it prevents later callbacks from observing
            # (or changing) a terminal decision while leaving ordinary continue/park aggregation intact.
            if r and r.get("exclusive"):
                return r
            if r and r.get("park"):
                return r
            if continuation is None and r and r.get("continue"):
                continuation = r
        return continuation

    def authorize_tool(self, name, args):
        for h in self.hooks:
            d = h.authorize_tool(name, args)
            if not d.allow:
                return d
        return ALLOW

    def prepare_messages(self, messages):
        changed = False
        for h in self.hooks:
            r = h.prepare_messages(messages)
            if r is not None:
                messages, changed = r, True
        return messages if changed else None

    def prepare_tool_schemas(self, schemas):
        changed = False
        current = list(schemas or ())
        for hook in self.hooks:
            callback = getattr(hook, "prepare_tool_schemas", None)
            if callback is None:
                continue
            result = callback(current)
            if result is not None:
                current, changed = list(result), True
        return current if changed else None

    def transform_tool_result(self, name, args, output):
        changed = False
        for h in self.hooks:
            r = h.transform_tool_result(name, args, output)
            if r is not None:
                output, changed = r, True
        return output if changed else None

    def reset_for_turn(self):
        for h in self.hooks:
            h.reset_for_turn()


# --- concrete hooks ---

_QUALITY_NO_ISSUE = "No supported response-quality issue is evidenced"
_QUALITY_UNCERTAIN = (
    "The sealed response-quality evidence is incomplete, so no observed-quality verdict is asserted"
)
_QUALITY_COMPLETE_AUDIT_FALLBACK = (
    "The draft did not produce a protocol-valid source audit, so no observed-quality verdict is asserted"
)
_QUALITY_PROSPECTIVE_HEADING = "Prospective (not observed)"
_QUALITY_MISMATCH_CATEGORIES = (
    "omitted explicit requirement",
    "contradicted explicit requirement",
    "unsupported factual claim",
    "violated explicit format or constraint",
)
_QUALITY_ISSUE_BLOCK = re.compile(
    r"(?im)^\s{0,3}(?:#{1,6}\s*)?Observed issue\s*$\n"
    r"^\s*Source:\s*(?P<source>artifacts/[^\s]+\.md)\s*$\n"
    r"^\s*Requested exact:\s*(?P<requested>\"(?:\\.|[^\"\\])*\")\s*$\n"
    r"^\s*Produced exact:\s*(?P<produced>\"(?:\\.|[^\"\\])*\")\s*$\n"
    r"(?:^\s*Grounding source:\s*(?P<grounding_source>artifacts/[^\s]+\.md)\s*$\n"
    r"^\s*Grounding exact:\s*(?P<grounding>\"(?:\\.|[^\"\\])*\")\s*$\n)?"
    r"^\s*Mismatch:\s*(?P<category>"
    + "|".join(re.escape(item) for item in _QUALITY_MISMATCH_CATEGORIES)
    + r")\s*(?:—|--|-|:)\s*(?P<explanation>[^\n]+)\s*$",
)
_QUALITY_CHECK_BLOCK = re.compile(
    r"(?im)^\s{0,3}(?:#{1,6}\s*)?Quality check\s*$\n"
    r"^\s*Source:\s*(?P<source>artifacts/[^\s]+\.md)\s*$\n"
    r"^\s*Basis:\s*exact request/response and any sealed grounding\s*$\n"
    r"^\s*Verdict:\s*no admitted mismatch\s*$",
)
_QUALITY_CHECK_LINE = re.compile(
    r"(?im)^\s{0,3}(?:(?:[-*+]|\d+[.)])\s*)?(?:Quality check\s*(?:—|--|-|:)?\s*)?"
    r"(?P<source>artifacts/[^\s]+\.md)\s*(?:—|--|-|:|=)\s*"
    r"(?:no admitted mismatch|no supported mismatch|clean)\s*[.!]?\s*$",
)
_QUALITY_CHECK_ORDINAL = re.compile(
    r"(?im)^\s*\|?\s*(?:pair\s+)?turn-(?P<ordinal>\d+)(?=\s|\()[^\n|]*"
    r"(?:\|[^\n|]*)?\bno admitted mismatch\b[^\n]*$",
)
_QUALITY_AUDIT_ATTESTATION = re.compile(
    r"\b(?:(?:I|we)(?:'ve|\s+have)?\s+)?(?:after\s+)?"
    r"(?:auditing|audited|checking|checked|reviewing|reviewed|rechecking|rechecked)\s+"
    r"(?:all\s+)?(?P<count>\d+)\s+exact\s+(?:sealed\s+)?request/response\s+pairs?\b",
    re.IGNORECASE,
)
_QUALITY_CLEAN_ATTESTATION_TAIL = re.compile(
    r"\bno\s+admitted\s+mismatch(?:es)?\s+(?:was|were)\s+found\b",
    re.IGNORECASE,
)
_QUALITY_VERIFICATION_BLOCK = re.compile(
    r"(?im)^\s{0,3}(?:#{1,6}\s*)?Verification item\s*$\n"
    r"^\s*Prior claim exact:\s*(?P<claim>\"(?:\\.|[^\"\\])*\")\s*$\n"
    r"^\s*Verdict:\s*(?P<verdict>supported|contradicted|not verifiable)\s*$\n"
    r"^\s*Evidence:\s*(?P<evidence>\"(?:\\.|[^\"\\])*\")\s*$",
)
_QUALITY_UNIVERSAL_OVERCLAIMS = (
    re.compile(
        r"\b(?:all|every|each)\b[^.\n]{0,120}\b(?:questions?|responses?|answers?|requests?|instructions?|claims?)\b"
        r"[^.\n]{0,120}\b(?:answered|correct|accurate|consistent|complied|fulfilled|satisfied|followed|matched)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:questions?|responses?|answers?|requests?|instructions?|claims?)\b[^.\n]{0,120}\b(?:all|every|each)\b"
        r"[^.\n]{0,120}\b(?:answered|correct|accurate|consistent|complied|fulfilled|satisfied|followed|matched)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:fully|completely|entirely)\s+(?:correct|accurate)\b", re.IGNORECASE),
    re.compile(r"\beverything\b[^.\n]{0,80}\b(?:correct|accurate|fulfilled|satisfied)\b", re.IGNORECASE),
    re.compile(r"\bnothing\b[^.\n]{0,50}\b(?:wrong|inaccurate)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:the|this|that)\s+(?:core\s+)?(?:task|review|request|analysis|answer|response)\s+"
        r"(?:was|were|has\s+been)\s+(?:executed|completed|performed|handled|answered|delivered)\s+"
        r"(?:correctly|accurately)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:bugs?|findings?|results?|answers?|responses?)\s+(?:were\s+)?"
        r"(?:reported|answered|handled|delivered)\s+(?:correctly|accurately)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\b[^.\n]{0,100}\b(?:incorrect\s+answers?|violations?)\b[^.\n]{0,60}"
        r"\b(?:occurred|were\s+(?:found|observed))\b",
        re.IGNORECASE,
    ),
)
_QUALITY_RETROSPECTIVE_PROSPECTIVE = re.compile(
    r"\b(?:I\s+failed|I\s+should\s+have|the\s+(?:failure|weakness)\s+was|"
    r"I\s+could\s+have|I\s+(?:did|didn't|did\s+not|had|hadn't|had\s+not|haven't|have\s+not|"
    r"already|never)\b|observed\s+(?:failure|weakness)|what\s+went\s+wrong|"
    r"turn\s+\d+|after\s+having|"
    r"(?:the|this|that)\s+(?:core\s+)?(?:task|review|request|analysis|answer|response)\s+"
    r"(?:was|were|has\s+been)\s+(?:executed|completed|performed|handled|answered|delivered)\s+"
    r"(?:correctly|accurately)|"
    r"(?:bugs?|findings?|results?|answers?|responses?)\s+(?:were\s+)?"
    r"(?:reported|answered|handled|delivered)\s+(?:correctly|accurately)|"
    r"based\s+on\s+(?:this|the)\s+session(?:\s+evidence)?|"
    r"(?:all|every)\s+(?:recorded\s+)?(?:operations?|turns?)\b[^.\n]{0,80}\b"
    r"(?:succeeded|completed|were\s+clean)|"
    r"nothing\b[^.\n]{0,50}\bfailed|"
    r"no\b[^.\n]{0,100}\b(?:incorrect\s+answers?|violations?)\b[^.\n]{0,60}"
    r"\b(?:occurred|were\s+(?:found|observed)))\b",
    re.IGNORECASE,
)
_QUALITY_INTERNAL_FEEDBACK_LEAK = re.compile(
    r"(?:^\s*(?:you(?:'re|\s+are)\s+right|understood[,.]?\s+I'll\s+confine)|"
    r"\b(?:the\s+host\s+rejected|unpublished\s+(?:draft|answer|assessment)|"
    r"I\s+converted\b[^.\n]{0,100}\bverdict|the\s+gate\s+(?:doesn't|does\s+not)\s+support)\b)",
    re.IGNORECASE,
)
_QUALITY_NO_ISSUE_EQUIVALENT = re.compile(
    r"(?:\bno\b[^.\n]{0,90}(?:response[- ]quality\s+issues?|four[- ]field\s+mismatch(?:es)?|"
    r"observed\s+(?:issues?|flaws?))[^.\n]{0,60}(?:evidenced|supported|found|documented|admitted)\b|"
    r"\b(?:found|shows?|reports?)\s+zero\s+(?:supported\s+)?(?:four[- ]field\s+)?mismatch(?:es)?\b|"
    r"\b(?:finds?|found|shows?|reports?)\s+no\s+supported\s+response[- ]quality\s+issues?\b|"
    r"\bno\s+admitted\s+mismatch(?:es)?\s+(?:was|were)\s+found\b)",
    re.IGNORECASE,
)
_QUALITY_SPECULATIVE_CRITIQUE = re.compile(
    r"\b(?:that\s+said|observed\s+weakness|my\s+weakness|I\s+should\s+have|I\s+could\s+have|"
    r"I\s+failed\s+to|I\s+didn't|I\s+did\s+not|too\s+terse|over[- ]?explained|"
    r"proactive|would\s+have\s+been\s+better|better\s+practice)\b",
    re.IGNORECASE,
)


class ExecutionEvidenceCompletionHook(Hooks):
    """Publish a host-derived answer core for pure canonical-receipt recall.

    A language model is useful for recognizing the user's question; it is not a second receipt database. Once the
    turn contract has selected only canonical execution receipts, copying and describing those lifecycle facts is
    deterministic. Owning the terminal answer here prevents a truthful count from acquiring a plausible but
    invented gloss about what an extra operation "probably" represented.

    Mixed-source verification remains model-owned: if the turn also needs a prior utterance, a quality judgment, or
    any other source, this hook deliberately abstains instead of flattening a semantic comparison into a count dump.
    """

    def __init__(self, state_provider):
        self.state_provider = state_provider

    def _state(self):
        try:
            return self.state_provider()
        except Exception:  # optional completion normalization must never break an otherwise valid turn
            return None

    @staticmethod
    def _contract(state):
        return getattr(getattr(state, "intent", None), "turn_contract", None)

    @staticmethod
    def _referents(contract, kind: str) -> tuple[Mapping, ...]:
        return tuple(
            item for item in (getattr(contract, "referents", ()) or ())
            if isinstance(item, Mapping) and item.get("kind") == kind
        )

    @staticmethod
    def _count(counts: Mapping, field: str) -> int:
        try:
            return max(0, int(counts.get(field, 0) or 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _canonical_answer(cls, state) -> str | None:
        contract = cls._contract(state)
        query = getattr(contract, "evidence_query", None)
        if query is None or getattr(contract, "quality_evidence_query", None) is not None:
            return None
        needs = set(getattr(contract, "source_needs", ()) or ())
        if needs != {"execution_receipt"}:
            return None
        aggregates = cls._referents(contract, "execution_receipt_aggregate")
        coverage_rows = cls._referents(contract, "execution_receipt_coverage")
        absence = cls._referents(contract, "execution_receipt_absence")
        coverage = coverage_rows[0] if coverage_rows else {}
        coverage_status = str(coverage.get("coverage") or "unavailable").casefold()
        scope = str(getattr(query, "scope", "task") or "task")
        family = str(getattr(query, "family", "all") or "all")
        predicate = str(getattr(query, "predicate", "operations") or "operations")
        if not aggregates:
            if absence:
                reason = str(absence[0].get("reason") or "no canonical receipt source was available").strip()
            else:
                reason = "no canonical receipt aggregate was available"
            return (
                f"Execution evidence is unavailable for scope={scope}, family={family}: {reason}. "
                "That gap establishes neither success nor failure."
            )

        aggregate = aggregates[0]
        counts = aggregate.get("counts") if isinstance(aggregate.get("counts"), Mapping) else {}
        partial = coverage_status != "complete"
        qualifier = "available canonical receipts (partial lower bound)" if partial else "canonical receipts"
        requested = cls._count(counts, "requested")
        started = cls._count(counts, "execution_started")
        settled = cls._count(counts, "settled")
        succeeded = cls._count(counts, "succeeded")
        rejected = cls._count(counts, "rejected_before_execution")
        failed = cls._count(counts, "failed")
        cancelled = cls._count(counts, "cancelled")
        indeterminate = cls._count(counts, "indeterminate")
        not_started = cls._count(counts, "not_started")
        unknown = cls._count(counts, "unknown")
        operation_count = cls._count(aggregate, "operation_count")
        lines = [f"Execution evidence ({qualifier}; scope={scope}, family={family}):"]
        if family == "delegation":
            lines.append(
                f"{started} child agents started execution; {settled} settled; {succeeded} succeeded; "
                f"{rejected} were rejected before execution; {failed} failed; {cancelled} were cancelled; "
                f"{indeterminate} were indeterminate; {not_started} did not start; {unknown} were unknown."
            )
            lines.append(
                f"Requested child-agent launches={requested}; relevant delegation operations={operation_count}; "
                f"distinct sealed child artifacts={cls._count(aggregate, 'child_artifact_count')}."
            )
        else:
            lines.append(
                f"Relevant operations={operation_count}; requested={requested}; started={started}; "
                f"settled={settled}; succeeded={succeeded}; rejected-before-execution={rejected}; failed={failed}; "
                f"cancelled={cancelled}; indeterminate={indeterminate}; not-started={not_started}; unknown={unknown}."
            )

        if predicate in {"operations", "failure_detail"}:
            details = cls._referents(contract, "execution_receipt")
            by_tool: dict[str, int] = {}
            adverse = []
            for receipt in details:
                for operation in receipt.get("operations") or ():
                    if not isinstance(operation, Mapping):
                        continue
                    name = str(operation.get("name") or "unknown tool")
                    disposition = str(operation.get("disposition") or "unknown")
                    by_tool[name] = by_tool.get(name, 0) + 1
                    if disposition != "succeeded":
                        reason = str(operation.get("reason") or "").strip()
                        adverse.append(
                            f"{name}: {disposition}" + (f" — {reason}" if reason else "")
                        )
            if predicate == "operations" and by_tool:
                lines.append("Operation types: " + "; ".join(
                    f"{name}={by_tool[name]}" for name in sorted(by_tool)
                ) + ".")
            if predicate == "failure_detail" and adverse:
                lines.append("Adverse operation detail: " + "; ".join(adverse) + ".")

        if partial:
            lines.append("Coverage is partial; totals outside the available receipts and the overall outcome are unknown.")
        return "\n".join(lines)

    def validate_completion(self, candidate: str, stop_reason: str):
        if stop_reason != "end_turn":
            return None
        replacement = self._canonical_answer(self._state())
        if replacement is None or str(candidate or "").strip() == replacement:
            return None
        return {
            "replacement": replacement,
            "exclusive": True,
            "reason": "pure execution recall is normalized to its canonical receipt projection",
        }


class DelegationCompletionHook(Hooks):
    """Keep an explicit child-agent mechanism binding through terminal completion."""

    _TOOLS = frozenset({"spawn_agent", "spawn_explore", "spawn_subagent"})

    def __init__(self, state_provider, max_revisions: int = 3):
        self.state_provider = state_provider
        self.max_revisions = max(1, int(max_revisions))
        self._revisions = 0

    def reset_for_turn(self):
        self._revisions = 0

    def _state(self):
        try:
            return self.state_provider()
        except Exception:
            return None

    @staticmethod
    def _field(value, name, default=None):
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    @classmethod
    def _requirement(cls, state):
        intent = cls._field(state, "intent")
        contract = cls._field(intent, "turn_contract")
        return cls._field(contract, "delegation_requirement")

    @classmethod
    def _calls(cls, state, agent: str) -> tuple[Mapping, ...]:
        runtime = cls._field(state, "runtime")
        calls = []
        for call in cls._field(runtime, "recent_calls", ()) or ():
            if not isinstance(call, Mapping) or str(call.get("name") or "") not in cls._TOOLS:
                continue
            args = call.get("args") if isinstance(call.get("args"), Mapping) else {}
            actual_agent = "explorer" if call.get("name") == "spawn_explore" else str(
                args.get("agent") or ""
            ).casefold()
            if actual_agent == agent:
                calls.append(call)
        return tuple(calls)

    @staticmethod
    def _call_text(call: Mapping) -> str:
        args = call.get("args") if isinstance(call.get("args"), Mapping) else {}
        return json.dumps(args, sort_keys=True, default=str).casefold()

    @classmethod
    def _failure_answer(cls, requirement, calls: tuple[Mapping, ...], reason: str) -> str:
        expected = cls._field(requirement, "count")
        agent = str(cls._field(requirement, "agent", "explorer") or "explorer")
        succeeded = sum(str(call.get("status") or "") == "succeeded" for call in calls)
        target_text = ", ".join(cls._field(requirement, "targets", ()) or ()) or "the requested scope"
        return (
            "I couldn't satisfy the requested delegation protocol, so I won't present direct parent analysis as "
            f"if it came from the requested children. Required: {expected if expected is not None else 'at least 1'} "
            f"{agent} child agent(s) covering {target_text}; completed successfully: {succeeded}. "
            f"Recorded gap: {reason}."
        )

    def validate_completion(self, candidate: str, stop_reason: str):
        if stop_reason != "end_turn":
            return None
        state = self._state()
        requirement = self._requirement(state)
        if requirement is None:
            return None
        agent = str(self._field(requirement, "agent", "explorer") or "explorer")
        expected = self._field(requirement, "count")
        targets = tuple(self._field(requirement, "targets", ()) or ())
        parallel = bool(self._field(requirement, "parallel", False))
        calls = self._calls(state, agent)
        successful = tuple(call for call in calls if str(call.get("status") or "") == "succeeded")
        missing_targets = tuple(
            target for target in targets
            if not any(target.casefold() in self._call_text(call) for call in successful)
        )
        count_ok = len(successful) == expected if expected is not None else bool(successful)
        steps = {call.get("step") for call in successful if call.get("step") is not None}
        parallel_ok = not parallel or len(successful) < 2 or len(steps) == 1
        if count_ok and not missing_targets and parallel_ok:
            return None

        reasons = []
        if not count_ok:
            reasons.append(
                f"successful {agent} children={len(successful)}, expected="
                f"{expected if expected is not None else 'at least 1'}"
            )
        if missing_targets:
            reasons.append("missing target coverage=" + ", ".join(missing_targets))
        if not parallel_ok:
            reasons.append("the child calls were issued across multiple model steps, not one parallel wave")
        reason = "; ".join(reasons)
        # More successful children than requested, wrong completed targets, or an already-sequential completed
        # wave cannot be repaired by adding another call. Report the invariant failure without laundering it.
        irreparable = (
            (expected is not None and len(successful) >= expected)
            or (parallel and len(successful) >= 2 and not parallel_ok)
        )
        if not irreparable and self._revisions < self.max_revisions:
            self._revisions += 1
            target_rule = (
                " Use one spawn_agent(agent='explorer', ...) call for each still-missing target: "
                + ", ".join(missing_targets or targets)
                + "."
                if (missing_targets or targets) else " Use spawn_agent(agent='explorer', ...)."
            )
            parallel_rule = " Emit those calls together in one response." if parallel else ""
            return {
                "continue": True,
                "exclusive": True,
                "feedback_role": "system",
                "feedback": (
                    "The host rejected the unpublished terminal answer because the current request's explicit "
                    "delegation mechanism is incomplete: " + reason + ". The user has not seen the draft; do not "
                    "mention this correction or claim the direct parent review satisfied the mechanism."
                    + target_rule + parallel_rule
                    + " After the requested children return, synthesize only their sealed results."
                ),
            }
        return {
            "replacement": self._failure_answer(requirement, calls, reason),
            "exclusive": True,
            "reason": reason,
        }


class _QualityEvidenceCompletionBase(Hooks):
    """Fail closed on unsupported autobiographical quality claims.

    The model still interprets the evidence and writes the answer. The host only enforces the claim-admission
    protocol already rendered beside the exact sealed pairs: either a source-backed four-field mismatch, or an
    evidence-sufficiency verdict that does not grow a speculative tail. This gate activates only for an original
    self-assessment; adjacent challenges use their separately frozen evidence projection.
    """

    def __init__(self, state_provider, max_revisions: int = 3):
        self.state_provider = state_provider
        self.max_revisions = max(1, int(max_revisions))
        self._revisions = 0

    def reset_for_turn(self):
        self._revisions = 0

    def _state(self):
        try:
            return self.state_provider()
        except Exception:  # unavailable state means this optional semantic gate cannot establish activation
            return None

    @staticmethod
    def _contract(state):
        return getattr(getattr(state, "intent", None), "turn_contract", None)

    @staticmethod
    def _source_rows(state) -> dict[str, Mapping]:
        rows = {}
        for item in getattr(getattr(state, "runtime", None), "source_projections", ()) or ():
            if not isinstance(item, Mapping) or item.get("kind") != "quality_exchange":
                continue
            artifact_id = str(item.get("artifact_id") or "").strip()
            if artifact_id:
                rows[f"artifacts/{artifact_id}.md"] = item
        return rows

    @classmethod
    def _grounding_sources(cls, state) -> dict[str, str]:
        """Exact sources admissible for proving an unsupported factual claim."""
        sources = {}
        for source, row in cls._source_rows(state).items():
            sources[source] = str(row.get("request") or "") + "\n" + str(row.get("assistant") or "")
            for grounding in row.get("grounding_artifacts") or ():
                if not isinstance(grounding, Mapping):
                    continue
                artifact_id = str(grounding.get("artifact_id") or "").strip()
                source_text = grounding.get("source_text")
                if artifact_id and isinstance(source_text, str):
                    sources[f"artifacts/{artifact_id}.md"] = source_text
        return sources

    @staticmethod
    def _coverage_complete(state) -> bool:
        for item in getattr(getattr(state, "runtime", None), "source_projections", ()) or ():
            if isinstance(item, Mapping) and item.get("kind") == "quality_exchange_coverage":
                return str(item.get("coverage") or "").strip().casefold() == "complete"
        return False

    @staticmethod
    def _execution_aggregate(state) -> Mapping | None:
        contract = QualityEvidenceCompletionHook._contract(state)
        for item in getattr(contract, "referents", ()) or ():
            if isinstance(item, Mapping) and item.get("kind") == "execution_receipt_aggregate":
                return item
        return None


_DELEGATED_HIGH_CONSEQUENCE = re.compile(
    r"\b(?:"
    r"(?:sql|command|code|prompt|shell|path|template|ldap|xml|html)\s+injection|"
    r"remote\s+code\s+execution|\brce\b|"
    r"(?:timing|side[- ]channel)\s+(?:attack|vulnerabilit|leak|exploit)|"
    r"leaks?\s+(?:password|secret|token|credential|data|timing|length|content)|"
    r"(?:plain[- ]?text|unhashed)\s+password|no\s+(?:password\s+)?hashing|"
    r"unkillable|uninterruptible|arbitrary\s+(?:code|command)\s+execution|"
    r"exfiltrat(?:e|es|ion)|account\s+takeover|data\s+(?:loss|destruction)|"
    r"(?:trivially|directly|fully)\s+exploitable|every\s+call\b[^.\n]{0,40}\bexploitable"
    r")\b",
    re.IGNORECASE,
)
_DELEGATED_MODALITY = re.compile(
    r"\b(?:if|unless|only\s+when|provided\s+that|depends?\s+on|conditional(?:ly)?|"
    r"could|may|might|possible|potential|risk\s+of|would\s+permit|would\s+allow|"
    r"not\s+(?:shown|observed|established|proven)|does\s+not\s+(?:show|establish|prove)|"
    r"requires?\s+(?:a|an|the|independent|runtime|caller|sink|measurement))\b",
    re.IGNORECASE,
)
_DELEGATED_REVISION_LEAK = re.compile(
    r"\b(?:unpublished\s+(?:draft|answer)|host\s+(?:rejected|gate)|"
    r"current\s+turn\s+contract|the\s+complaint\s+is|(?:my|the)\s+original\s+(?:line|claim)s?|"
    r"revision\s+you(?:'re|\s+are)\s+asking|revis(?:e|ed|ing)\s+(?:the\s+)?(?:line|summary)s?|"
    r"corrected\s+summary|per\s+your\s+instruction|let\s+me\s+revise|"
    r"the\s+user\s+asked\s+me|exact\s+requested\s+shape|no\s+tool\s+calls|"
    r"i\s+need\s+to\s+(?:lead|revise|qualify)|first\s+line\s+now\s+carries)\b",
    re.IGNORECASE,
)
_DELEGATED_NEGATED_CONSEQUENCE = re.compile(
    r"\b(?:no|not|never|without)\b[^.?!;\n]{0,40}$", re.IGNORECASE,
)
_QUALITY_CAUTION_CRITIQUE = re.compile(
    r"\b(?:conditional|qualifier|qualified|uncertain|uncertainty|cautious|caveat|"
    r"casts?\s+doubt|undercut|understate|mislabel|unnecessary)\b",
    re.IGNORECASE,
)
_QUALITY_EXPLICIT_CONFIDENCE_REQUEST = re.compile(
    r"\b(?:unconditional|without\s+(?:a\s+)?qualifier|state\s+as\s+(?:certain|fact)|"
    r"definitive|certainty|confidence\s+level|do\s+not\s+qualify)\b",
    re.IGNORECASE,
)


class DelegatedClaimCompletionHook(Hooks):
    """Preserve epistemic strength when delegated testimony becomes parent prose.

    A sealed child report proves what the child *said*. It does not turn a possible exploit chain into a live
    workspace fact. The prompt carries that rule, while this narrow publication boundary catches the costly
    class of qualifier loss: high-consequence security/liveness/data-impact claims stated categorically after a
    child fan-out. It does not judge ordinary bug descriptions and is inactive on turns without delegation.

    The first attempts are model-owned revisions with exact offending lines. A bounded final fallback labels
    any remaining line as conditional rather than publishing it as fact; this is a strength downgrade, never a
    host-authored claim that the consequence is true.
    """

    def __init__(self, state_provider, *, max_revisions: int = 0, source_review: bool = True):
        self.state_provider = state_provider
        self.max_revisions = max(0, int(max_revisions))
        self.source_review = bool(source_review)
        self.revisions = 0
        self._source_reviewed = False

    def reset_for_turn(self):
        self.revisions = 0
        self._source_reviewed = False

    def _state(self):
        try:
            return self.state_provider()
        except Exception:
            return None

    @staticmethod
    def _has_delegation(state) -> bool:
        calls = getattr(getattr(state, "runtime", None), "recent_calls", ()) or ()
        current = any(
            isinstance(call, Mapping)
            and str(call.get("name") or "") in {"spawn_agent", "spawn_explore", "spawn_subagent"}
            and str(call.get("status") or "").casefold() == "succeeded"
            for call in calls
        )
        sources = getattr(getattr(state, "evidence", None), "finding_source", {}) or {}
        return current or any(str(value).casefold() == "delegated" for value in sources.values())

    @staticmethod
    def _has_current_delegation(state) -> bool:
        calls = getattr(getattr(state, "runtime", None), "recent_calls", ()) or ()
        return any(
            isinstance(call, Mapping)
            and str(call.get("name") or "") in {"spawn_agent", "spawn_explore", "spawn_subagent"}
            and str(call.get("status") or "").casefold() == "succeeded"
            for call in calls
        )

    @staticmethod
    def _units(text: str) -> tuple[str, ...]:
        # Keep Markdown headings/list items intact. Sentence splitting catches prose paragraphs that put several
        # independent findings on one line without turning dotted identifiers/paths into fragments.
        units = []
        for line in str(text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"(?<=[!?;])\s+|(?<=\.)\s+(?=[A-Za-z*#])", line)
            units.extend(part.strip() for part in parts if part.strip())
        return tuple(units)

    @classmethod
    def _violations(cls, text: str) -> tuple[str, ...]:
        violations = []
        for unit in cls._units(text):
            match = _DELEGATED_HIGH_CONSEQUENCE.search(unit)
            if not match or _DELEGATED_MODALITY.search(unit):
                continue
            prefix = unit[:match.start()]
            if _DELEGATED_NEGATED_CONSEQUENCE.search(prefix):
                continue
            if unit.lstrip().startswith(">") or re.search(
                    r"\b(?:child|report|reviewer|source)\s+(?:said|says|reported|called|labeled)\b",
                    prefix, re.IGNORECASE):
                continue
            violations.append(unit)
        return tuple(violations)

    @staticmethod
    def _exact_shape(state) -> tuple[str, ...]:
        contract = getattr(getattr(state, "intent", None), "turn_contract", None)
        requirement = getattr(contract, "delegation_requirement", None)
        targets = tuple(str(item) for item in (getattr(requirement, "targets", ()) or ()) if str(item))
        try:
            value = int(getattr(requirement, "count", 0) or 0)
        except (TypeError, ValueError):
            value = 0
        request = str(getattr(getattr(state, "intent", None), "current_request", "") or "")
        number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        match = re.search(
            r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)[ -]?line\s+summary\b",
            request, re.IGNORECASE,
        )
        if not match:
            return ()
        token = match.group("count").casefold()
        requested = int(token) if token.isdigit() else number_words[token]
        return targets if value and requested == value and len(targets) == value else ()

    @staticmethod
    def _shape_result(text: str, targets: tuple[str, ...]) -> str:
        if not targets:
            return str(text or "").strip()
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        selected = []
        for target in targets:
            target_folded = target.casefold()
            match = next((line for line in reversed(lines) if target_folded in line.casefold()), None)
            if match is None:
                return str(text or "").strip()
            selected.append(match)
        return "\n".join(selected)

    @staticmethod
    def _missing_shape_targets(text: str, targets: tuple[str, ...]) -> tuple[str, ...]:
        lines = [line.casefold() for line in str(text or "").splitlines() if line.strip()]
        return tuple(target for target in targets
                     if not any(target.casefold() in line for line in lines))

    @staticmethod
    def _one_line(value: object, limit: int = 560) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"

    @staticmethod
    def _scope_key(value: object) -> str:
        text = str(value or "").strip().replace("\\", "/")
        absolute = text.startswith("/")
        pieces = []
        for piece in text.split("/"):
            if not piece or piece == ".":
                continue
            if piece == "..":
                if pieces:
                    pieces.pop()
                elif not absolute:
                    pieces.append(piece)
                continue
            pieces.append(piece)
        return ("/" if absolute else "") + "/".join(pieces)

    @classmethod
    def _canonical_exact_testimony(cls, state, targets: tuple[str, ...]) -> str | None:
        """Reduce exact fan-out to sealed child testimony instead of asking the parent to paraphrase it.

        This is intentionally a statement *about the child report*, not a workspace-fact synthesis. The ledger
        entry is bound to a verbatim report span by ``SubagentArtifact``; observation hashes remain candidate
        locators and are labeled that way rather than treated as entailment certificates.
        """
        if not targets:
            return None
        requirement = DelegationCompletionHook._requirement(state)
        agent = str(DelegationCompletionHook._field(requirement, "agent", "explorer") or "explorer")
        calls = [call for call in DelegationCompletionHook._calls(state, agent)
                 if str(call.get("status") or "").casefold() == "succeeded"]
        used: set[int] = set()
        lines = []
        for ordinal, target in enumerate(targets, 1):
            target_key = cls._scope_key(target)
            matching = [
                index for index, call in enumerate(calls)
                if index not in used and target_key
                and target_key == cls._scope_key(call.get("child_target"))
            ]
            if len(matching) != 1:
                return None
            match_index = matching[0]
            call = calls[match_index]
            used.add(match_index)
            claims = call.get("child_claims") or ()
            claim = next((row for row in claims if isinstance(row, Mapping)
                          and str(row.get("text") or "").strip()
                          and str(row.get("report_exact") or "").strip()), None)
            if claim is None:
                return None
            artifact_id = cls._one_line(call.get("child_artifact_id") or "sealed-child-report", 80)
            artifact_handle = (
                f"artifacts/{artifact_id}.md" if artifact_id != "sealed-child-report"
                else artifact_id
            )
            label = cls._one_line(target, 120)
            # JSON encoding makes the verbatim report span one physical output line without normalizing or
            # truncating any byte. This proves report membership only; workspace truth remains a separate check.
            testimony = json.dumps(str(claim.get("report_exact") or ""), ensure_ascii=False)
            lines.append(
                f"{ordinal}. **{label}** — {agent.capitalize()} report states "
                f"(unverified delegated testimony; {artifact_handle}): {testimony}"
            )
        return "\n".join(lines)

    @classmethod
    def _downgrade(cls, text: str, *, exact_targets: tuple[str, ...] = ()) -> str:
        out = []
        changed = False
        removed_revision = False
        for line in str(text or "").splitlines():
            parts = re.split(r"(?<=[!?;])\s+|(?<=\.)\s+(?=[A-Za-z*#])", line)
            rewritten = []
            for part in parts:
                if not part:
                    continue
                # A private retry may have taught the model control-plane vocabulary. Never publish those units
                # from the exclusive fallback; retain the surrounding user-facing result and its physical lines.
                if _DELEGATED_REVISION_LEAK.search(part):
                    removed_revision = True
                    continue
                if cls._violations(part):
                    part = "Conditional (not established by the retained observation alone): " + part.strip()
                    changed = True
                rewritten.append(part)
            out.append(" ".join(rewritten))
        result = cls._shape_result("\n".join(out), exact_targets)
        if _DELEGATED_REVISION_LEAK.search(result) or (removed_revision and not result.strip()):
            return (
                "I couldn't safely publish the delegated synthesis because the draft contained private retry "
                "narration and no clean user-facing result remained."
            )
        # Fail closed on the backstop itself: an exclusive replacement may not retain a claim the detector
        # just rejected. Prefixing the whole surviving unit is the final model-independent strength downgrade.
        if cls._violations(result):
            result = "\n".join(
                "Conditional (not established by the retained observation alone): " + line
                if cls._violations(line) else line
                for line in result.splitlines()
            )
        if changed and not exact_targets:
            result += "\n\n" + "\n".join([
                "",
                "Prerequisite: verify the caller, execution sink, runtime path, input control, and measurable "
                "effect before treating any conditional consequence above as an observed fact.",
            ]).strip()
        return result.strip()

    def validate_completion(self, candidate: str, stop_reason: str):
        state = self._state()
        if stop_reason != "end_turn" or not self._has_delegation(state):
            return None
        contract = getattr(getattr(state, "intent", None), "turn_contract", None)
        if getattr(contract, "quality_evidence_query", None) is not None:
            # Self-assessment has its own source-exact protocol. Rewriting quoted historical claims here would
            # corrupt those bytes and make the two gates recursively critique each other's safety labels.
            return None
        exact_targets = self._exact_shape(state)
        canonical_testimony = self._canonical_exact_testimony(state, exact_targets)
        if canonical_testimony is not None:
            if str(candidate or "").strip() == canonical_testimony:
                return None
            return {
                "replacement": canonical_testimony,
                "exclusive": True,
                "reason": "reduced exact delegated fan-out from sealed typed child testimony",
            }
        if exact_targets and self._has_current_delegation(state):
            # Exact fan-out has a typed reduction contract. Falling back to unconstrained parent prose after the
            # children have settled can silently swap targets, lose TOP CLAIM qualifiers, or emit provider-control
            # syntax that merely happens to occupy N lines. Fail visibly instead; the next self-audit can then
            # measure the unmet exact-line requirement from sealed request/response bytes.
            return {
                "replacement": (
                    "I couldn't produce the requested delegated summary because the completed child artifacts "
                    "did not provide one unambiguous host-bound target and exact TOP CLAIM for every requested "
                    "target. I will not substitute free parent synthesis for those missing typed bindings."
                ),
                "exclusive": True,
                "reason": "exact delegated reduction lacked complete typed target/claim bindings",
            }
        if self.source_review and self._has_current_delegation(state) and not self._source_reviewed:
            self._source_reviewed = True
            target_rule = (
                " Return exactly one result line for each target, in this order: "
                + ", ".join(exact_targets) + "."
                if exact_targets else " Return only the synthesis requested by the user."
            )
            return {
                "continue": True,
                "exclusive": True,
                "prose_only": True,
                "feedback_role": "system",
                "feedback": (
                    "Private source-reconciliation pass: the user has not seen the candidate. Make no tool call "
                    "and do not mention this check, a draft, or a correction. Re-read the child tool results "
                    "already in the trajectory. Treat each child report as testimony and its PRIMARY OBSERVATION "
                    "as the workspace source. For every output claim: follow local control flow (including catches "
                    "and earlier failures), prefer the most certain directly observed defect, copy file:line from "
                    "the observation, and preserve every missing caller, execution sink, type/storage assumption, "
                    "attacker path, or other prerequisite in the same line. Construction is not execution; an "
                    "unknown value representation is not plaintext/hash evidence; a caught exception does not "
                    "crash its caller. Remove any claim the bytes do not support."
                    + target_rule + " No preamble or postscript."
                ),
            }
        missing_shape = self._missing_shape_targets(candidate, exact_targets)
        violations = self._violations(candidate)
        leaked_revision = bool(self.revisions and _DELEGATED_REVISION_LEAK.search(candidate or ""))
        shaped = self._shape_result(candidate, exact_targets)
        shape_changed = bool(exact_targets and shaped != str(candidate or "").strip())
        if not violations and not leaked_revision:
            if missing_shape:
                return {
                    "replacement": (
                        "I couldn't produce the requested exact-line delegated synthesis from the completed "
                        "child results. Missing result line(s): " + ", ".join(missing_shape) + "."
                    ),
                    "exclusive": True,
                    "reason": "explicit delegated response shape was incomplete",
                }
            if shape_changed:
                return {
                    "replacement": shaped,
                    "exclusive": True,
                    "reason": "normalized the explicit exact-line delegated response shape",
                }
            return None
        if self.revisions < self.max_revisions:
            self.revisions += 1
            examples = "\n".join(f"- {line[:300]}" for line in violations[:5]) or "- private retry narration"
            return {
                "continue": True,
                "feedback_role": "system",
                "feedback": (
                    "This is a private publication check, not a new user request. The user has not seen the "
                    "candidate. Do not call any tool, edit any file, mention this check, narrate a correction, or "
                    "add a preamble. Return only the user's requested final response in the exact requested shape. "
                    "The candidate strengthened an inference/conditional consequence into a fact. Revise the "
                    "exact lines below. Lead with the directly observed defect. For any "
                    "security, liveness, or data-impact consequence, put its if/unless/may/could qualifier and "
                    "missing caller/sink/runtime prerequisite in that same line. Do not merely add a generic "
                    "disclaimer elsewhere. Offending lines:\n" + examples
                ),
            }
        return {
            "replacement": self._downgrade(
                candidate, exact_targets=exact_targets,
            ),
            "exclusive": True,
            "reason": "downgraded unqualified delegated consequences at the publication boundary",
        }


class QualityEvidenceCompletionHook(_QualityEvidenceCompletionBase):
    """Complete the response-quality gate after the independent delegated-claim publication gate."""

    @classmethod
    def _deterministic_quality_mismatches(cls, state) -> tuple[tuple[str, Mapping], ...]:
        out = []
        for source, row in cls._source_rows(state).items():
            raw = row.get("deterministic_mismatches") or ()
            if not isinstance(raw, (list, tuple)):
                continue
            out.extend((source, item) for item in raw if isinstance(item, Mapping))
        return tuple(out)

    @classmethod
    def _canonical_deterministic_issues(cls, state) -> str:
        """Render host-measured explicit-constraint failures as ordinary source-exact issue blocks."""
        grouped: dict[str, list[Mapping]] = {}
        for source, mismatch in cls._deterministic_quality_mismatches(state):
            grouped.setdefault(source, []).append(mismatch)
        blocks = []
        for source, mismatches in grouped.items():
            first = mismatches[0]
            requested = str(first.get("requested_exact") or "")
            produced = str(first.get("produced_exact") or "")
            row = cls._source_rows(state).get(source, {})
            if not requested or requested not in str(row.get("request") or "") \
                    or not produced or produced not in str(row.get("assistant") or ""):
                continue
            explanations = []
            for mismatch in mismatches:
                explanation = re.sub(r"\s+", " ", str(mismatch.get("explanation") or "")).strip()
                if explanation:
                    if mismatch.get("produced_exact_is_bounded_prefix"):
                        explanation += (
                            "; Produced exact is a bounded verbatim prefix, while the measurements cover the full "
                            "sealed response"
                        )
                    explanations.append(explanation)
            if not explanations:
                continue
            blocks.append("\n".join((
                "Observed issue",
                f"Source: {source}",
                "Requested exact: " + json.dumps(requested, ensure_ascii=False),
                "Produced exact: " + json.dumps(produced, ensure_ascii=False),
                "Mismatch: violated explicit format or constraint — " + "; ".join(explanations),
            )))
        return "\n\n".join(blocks)

    @classmethod
    def _deterministic_prospective(cls, state) -> str:
        constraints = {str(item.get("constraint") or "")
                       for _source, item in cls._deterministic_quality_mismatches(state)}
        policies = []
        if "brief_response" in constraints:
            policies.append(
                "honor explicit brevity requests by staying within the stated brief ceiling and omitting "
                "unnecessary recap or follow-up prompts"
            )
        if "exact_nonempty_lines" in constraints:
            policies.append("count the final physical lines before publishing an exact-line response")
        if "valid_json" in constraints:
            policies.append("parse-check the final payload before publishing a response required to be valid JSON")
        if not policies:
            return cls._safe_prospective()
        return "In future, I will " + "; and ".join(policies) + "."

    def prepare_tool_schemas(self, schemas):
        """Frozen verification is closed-book over an already materialized snapshot.

        Removing tools from the provider schema prevents a model from reopening moving history, accruing a rejected
        operation, or confusing "not in the live tail" with "not sealed". The completion hook still checks the
        final prose against the same frozen typed projection.
        """
        state = self._state()
        contract = self._contract(state)
        query = getattr(contract, "quality_evidence_query", None)
        if str(getattr(query, "purpose", "") or "") != "verify_assessment":
            return None
        snapshot = next((item for item in getattr(contract, "referents", ()) or ()
                         if isinstance(item, Mapping) and item.get("kind") == "evidence_snapshot"), {})
        return [] if str(snapshot.get("status") or "").casefold() == "frozen" else None

    @classmethod
    def _execution_verification_sentence(cls, state, prior: str) -> str:
        execution = cls._canonical_execution_summary(state)
        if not execution or execution not in str(prior or ""):
            return ""
        return (
            "The prior execution statement " + json.dumps(execution, ensure_ascii=False)
            + " is also supported by the frozen canonical receipt aggregate."
        )

    @classmethod
    def _canonical_execution_summary(cls, state) -> str:
        """Render a tiny host-owned execution preamble without borrowing quality-gate vocabulary."""
        contract = cls._contract(state)
        if getattr(contract, "evidence_query", None) is None:
            return ""
        aggregate = cls._execution_aggregate(state)
        if not isinstance(aggregate, Mapping):
            return ""
        counts = aggregate.get("counts") if isinstance(aggregate.get("counts"), Mapping) else {}
        turn_counts = (aggregate.get("turn_counts")
                       if isinstance(aggregate.get("turn_counts"), Mapping) else {})
        coverage = next((item for item in getattr(contract, "referents", ()) or ()
                         if isinstance(item, Mapping)
                         and item.get("kind") == "execution_receipt_coverage"), {})
        partial = str(coverage.get("coverage") or "complete").casefold() != "complete"
        operation_labels = (
            ("rejected-before-execution operations", "rejected_before_execution"),
            ("failed operations", "failed"), ("cancelled operations", "cancelled"),
            ("indeterminate operations", "indeterminate"),
            ("not-started operations", "not_started"), ("unknown operations", "unknown"),
        )
        turn_labels = (
            ("completed-with-warnings turns", "completed_with_warnings"),
            ("paused turns", "paused"), ("blocked turns", "blocked"),
            ("interrupted turns", "interrupted"),
            ("indeterminate turns", "indeterminate"), ("unknown turns", "unknown"),
        )
        adverse = [f"{label}={int(counts.get(field, 0) or 0)}"
                   for label, field in operation_labels if int(counts.get(field, 0) or 0)]
        adverse.extend(f"{label}={int(turn_counts.get(field, 0) or 0)}"
                       for label, field in turn_labels if int(turn_counts.get(field, 0) or 0))
        prefix = "Available execution evidence" if partial else "Execution evidence"
        if adverse:
            result = f"{prefix} (canonical receipts): " + "; ".join(adverse) + "."
        else:
            result = f"{prefix} (canonical receipts): no adverse operation or non-clean turn is recorded."
        if partial:
            result += " Coverage is partial, so this is not a whole-scope success claim."
        return result

    @classmethod
    def _scoped_no_issue_answer(cls, state, prospective_tail: str = "") -> str:
        parts = []
        execution = cls._canonical_execution_summary(state)
        if execution:
            parts.append(execution)
        parts.append(_QUALITY_NO_ISSUE + ".")
        if prospective_tail:
            tail = prospective_tail.strip()
            lines = tail.splitlines()
            while lines and re.fullmatch(r"[*_`~#>|-]{1,12}", lines[0].strip()):
                lines.pop(0)
                while lines and not lines[0].strip():
                    lines.pop(0)
            tail = "\n".join(lines).strip()
            if tail:
                parts.append(_QUALITY_PROSPECTIVE_HEADING + "\n\n" + tail)
        return "\n\n".join(parts)

    @staticmethod
    def _number(value: str) -> int:
        return 0 if str(value).casefold() == "zero" else int(value)

    @classmethod
    def _validate_numeric_evidence(cls, candidate: str, state) -> str:
        """Check only explicit high-confidence aggregate copies; prose semantics remain the model's job."""
        aggregate = cls._execution_aggregate(state)
        if not isinstance(aggregate, Mapping):
            return ""
        counts = aggregate.get("counts") if isinstance(aggregate.get("counts"), Mapping) else {}
        turn_counts = (aggregate.get("turn_counts")
                       if isinstance(aggregate.get("turn_counts"), Mapping) else {})
        operation_count = int(aggregate.get("operation_count", 0) or 0)
        lifecycle_fields = {
            "requested": "requested",
            "started": "execution_started", "execution-started": "execution_started",
            "settled": "settled",
            "succeeded": "succeeded", "successful": "succeeded",
            "failed": "failed", "failure": "failed", "failures": "failed",
            "cancelled": "cancelled", "indeterminate": "indeterminate",
            "not-started": "not_started", "unknown": "unknown",
        }
        number = r"(?:\d+|zero)"
        status = "|".join(re.escape(item) for item in lifecycle_fields)
        for segment in re.split(r"[\n.;]+", candidate):
            lowered = segment.casefold()
            if "operation" not in lowered:
                continue
            pattern = re.compile(
                rf"\b(?P<num>{number})(?:\s*/\s*(?P<den>\d+))?\s+"
                rf"(?:(?:relevant\s+)?operations?\s+)?(?P<status>{status})\b",
                re.IGNORECASE,
            )
            for match in pattern.finditer(segment):
                label = match.group("status").casefold()
                field = lifecycle_fields[label]
                actual = cls._number(match.group("num"))
                expected = int(counts.get(field, 0) or 0)
                if actual != expected:
                    return (f"the execution claim '{match.group(0).strip()}' copies {actual} for {field}, "
                            f"but the canonical aggregate says {expected}")
                if match.group("den") is not None and int(match.group("den")) != operation_count:
                    return (f"the execution claim '{match.group(0).strip()}' uses denominator "
                            f"{match.group('den')}, but the canonical relevant-operation total is "
                            f"{operation_count}")

        turn_labels = {
            "completed": "completed", "completed-with-warnings": "completed_with_warnings",
            "completed with warnings": "completed_with_warnings", "paused": "paused",
            "blocked": "blocked", "interrupted": "interrupted",
            "indeterminate": "indeterminate", "unknown": "unknown",
        }
        turn_status = "|".join(re.escape(item) for item in turn_labels)
        for segment in re.split(r"[\n.;]+", candidate):
            lowered = segment.casefold()
            if "turn" not in lowered:
                continue
            pattern = re.compile(
                rf"\b(?P<num>{number})\s+(?:turns?\s+)?(?P<status>{turn_status})\b",
                re.IGNORECASE,
            )
            for match in pattern.finditer(segment):
                # "turn 1 completed ..." identifies an ordinal turn; it is not a claim that one turn completed.
                if re.search(r"\bturn\s*$", segment[:match.start()], re.IGNORECASE):
                    continue
                field = turn_labels[match.group("status").casefold()]
                actual = cls._number(match.group("num"))
                expected = int(turn_counts.get(field, 0) or 0)
                if actual != expected:
                    return (f"the turn-lifecycle claim '{match.group(0).strip()}' copies {actual} for {field}, "
                            f"but the canonical aggregate says {expected}")

        scalar_claims = (
            (re.compile(rf"\b(?P<num>{number})\s+(?:distinct\s+)?child\s+(?:agents?|artifacts?)\b", re.I),
             int(aggregate.get("child_artifact_count", 0) or 0), "distinct child artifacts"),
            (re.compile(rf"\b(?P<num>{number})\s+warnings?\b", re.I),
             int(aggregate.get("turn_warning_count", 0) or 0), "turn warnings"),
            (re.compile(rf"\b(?P<num>{number})\s+non[- ]clean\s+turns?\b", re.I),
             int(aggregate.get("nonclean_turn_count", 0) or 0), "non-clean turns"),
        )
        for pattern, expected, label in scalar_claims:
            for match in pattern.finditer(candidate):
                actual = cls._number(match.group("num"))
                if actual != expected:
                    return (f"the claim '{match.group(0).strip()}' copies {actual} for {label}, but the "
                            f"canonical aggregate says {expected}")

        pair_expected = next((
            int(item.get("complete_exchange_pairs", 0) or 0)
            for item in getattr(getattr(state, "runtime", None), "source_projections", ()) or ()
            if isinstance(item, Mapping) and item.get("kind") == "quality_exchange_coverage"
        ), None)
        if pair_expected is not None:
            pair_pattern = re.compile(
                rf"\b(?P<num>{number})\s+(?:exact\s+)?(?:sealed\s+)?(?:request/response\s+)?pairs?\b",
                re.IGNORECASE,
            )
            for match in pair_pattern.finditer(candidate):
                actual = cls._number(match.group("num"))
                if actual != pair_expected:
                    return (f"the quality-evidence claim '{match.group(0).strip()}' copies {actual} pairs, "
                            f"but the sealed projection says {pair_expected}")
        return ""

    @staticmethod
    def _has_universal_overclaim(candidate: str) -> bool:
        return any(pattern.search(candidate) for pattern in _QUALITY_UNIVERSAL_OVERCLAIMS)

    @staticmethod
    def _prospective_tail(after_verdict: str) -> str | None:
        # Markdown heading/list adornment is presentation, not part of the required literal label.
        normalized = re.sub(r"^[\s#*_`>~.,;:!?\-—]+", "", after_verdict)
        if not normalized.startswith(_QUALITY_PROSPECTIVE_HEADING):
            return None
        return normalized[len(_QUALITY_PROSPECTIVE_HEADING):].lstrip(" \t:-—\r\n")

    @staticmethod
    def _terminal_replacement(candidate: str, verdict_at: int, verdict: str) -> str:
        """Replace the verdict's whole paragraph so quoted/italic copies cannot leave broken markup."""
        paragraph_at = candidate.rfind("\n\n", 0, verdict_at)
        prefix = candidate[:paragraph_at].rstrip() if paragraph_at >= 0 else ""
        return (prefix + "\n\n" if prefix else "") + verdict.rstrip(" .") + "."

    def _validate_issue_blocks(self, candidate: str, state) -> tuple[bool, str]:
        matches = tuple(_QUALITY_ISSUE_BLOCK.finditer(candidate))
        if not matches:
            return False, "no protocol-valid observed-issue block was supplied"
        if len(matches) != len(re.findall(r"(?im)^\s{0,3}(?:#{1,6}\s*)?Observed issue\s*$", candidate)):
            return False, "at least one observed-issue block is incomplete or malformed"
        rows = self._source_rows(state)
        if not rows:
            return False, "the exact sealed pair projection is unavailable"
        for match in matches:
            source = match.group("source")
            row = rows.get(source)
            if row is None:
                return False, f"{source} is outside the admitted quality-evidence source set"
            try:
                requested = json.loads(match.group("requested"))
                produced = json.loads(match.group("produced"))
            except (TypeError, ValueError, json.JSONDecodeError):
                return False, "Requested exact and Produced exact must be valid JSON strings"
            if not isinstance(requested, str) or not requested.strip():
                return False, "Requested exact must quote non-empty bytes from the sealed user request"
            if not isinstance(produced, str) or not produced.strip():
                return False, "Produced exact must quote non-empty bytes from the sealed assistant response"
            if requested not in str(row.get("request") or ""):
                return False, f"Requested exact is not present in {source}"
            if produced not in str(row.get("assistant") or ""):
                return False, f"Produced exact is not present in {source}"
            grounding_source = str(match.group("grounding_source") or "").strip()
            grounding_literal = match.group("grounding")
            category = str(match.group("category") or "").casefold()
            explanation = str(match.group("explanation") or "")
            # Adding an epistemic-strength downgrade is not itself a contradiction or unsupported factual claim.
            # It can be a style preference, but the quality gate admits only incompatibility with exact user
            # behavior. Unless the request explicitly demanded a certainty/confidence treatment, do not let the
            # model manufacture a past failure merely because a response was more cautious than a child report.
            if (_QUALITY_CAUTION_CRITIQUE.search(explanation + "\n" + produced)
                    and not _QUALITY_EXPLICIT_CONFIDENCE_REQUEST.search(requested)):
                return False, (
                    "an epistemic qualifier/caution is being treated as a response failure even though the exact "
                    "request did not specify a confidence level or forbid qualification"
                )
            if category == "unsupported factual claim" and not grounding_source:
                return False, (
                    "an unsupported factual claim requires Grounding source and Grounding exact from the "
                    "admitted sealed source set"
                )
            if grounding_source:
                source_text = self._grounding_sources(state).get(grounding_source)
                if source_text is None:
                    return False, f"{grounding_source} is outside the admitted quality-grounding source set"
                try:
                    grounding = json.loads(grounding_literal)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return False, "Grounding exact must be a valid JSON string"
                if not isinstance(grounding, str) or not grounding.strip():
                    return False, "Grounding exact must quote non-empty bytes from a sealed grounding source"
                if grounding not in source_text:
                    return False, f"Grounding exact is not present in {grounding_source}"
        return True, ""

    def _validate_audit_coverage(self, candidate: str, state) -> str:
        """Require private source-by-source proof work before any complete quality verdict."""
        expected = set(self._source_rows(state))
        block_matches = tuple(_QUALITY_CHECK_BLOCK.finditer(candidate))
        without_blocks = _QUALITY_CHECK_BLOCK.sub("", candidate)
        line_matches = tuple(_QUALITY_CHECK_LINE.finditer(without_blocks))
        ordered_sources = tuple(self._source_rows(state))
        ordinal_sources = []
        for match in _QUALITY_CHECK_ORDINAL.finditer(without_blocks):
            ordinal = int(match.group("ordinal"))
            if 1 <= ordinal <= len(ordered_sources):
                ordinal_sources.append(ordered_sources[ordinal - 1])
        headings = len(re.findall(r"(?im)^\s{0,3}(?:#{1,6}\s*)?Quality check\s*$", candidate))
        if len(block_matches) != headings:
            return "at least one source-audit Quality check block is incomplete or malformed"
        checked = [match.group("source") for match in (*block_matches, *line_matches)] + ordinal_sources
        attestation = _QUALITY_AUDIT_ATTESTATION.search(candidate)
        attested_all = False
        if attestation is not None:
            claimed = int(attestation.group("count"))
            if claimed != len(ordered_sources):
                return (
                    f"the source-audit attestation says {claimed} exact pair(s), but the projection contains "
                    f"{len(ordered_sources)}"
                )
            attested_all = True
        if len(checked) != len(set(checked)):
            return "a source appears more than once as a no-mismatch Quality check"
        issue_sources = [match.group("source") for match in _QUALITY_ISSUE_BLOCK.finditer(candidate)]
        if len(issue_sources) != len(set(issue_sources)):
            return "a source appears more than once as an observed issue"
        overlap = set(checked).intersection(issue_sources)
        if overlap:
            return f"{sorted(overlap)[0]} is marked both no-mismatch and observed-issue"
        if attested_all:
            # A neutral exact-count attestation proves the model considered the whole source set; source-exact
            # issue blocks still carry every admitted mismatch. A *clean* attestation cannot coexist with an
            # issue, but requiring six repeated clean lines is unnecessary protocol ceremony.
            tail = candidate[attestation.start():attestation.end() + 100]
            if issue_sources and _QUALITY_CLEAN_ATTESTATION_TAIL.search(tail):
                return "the audit declares every pair clean while also reporting an observed issue"
            outside = set(issue_sources).difference(expected)
            if outside:
                return f"{sorted(outside)[0]} is outside the exact sealed pair projection"
            return "" if expected else "the exact sealed pair projection contains no auditable source"
        accounted = set(checked).union(issue_sources)
        outside = accounted - expected
        if outside:
            return f"{sorted(outside)[0]} is outside the exact sealed pair projection"
        missing = expected - accounted
        if missing:
            return (
                f"the source-complete audit omitted {len(missing)} exact pair(s), beginning with "
                f"{sorted(missing)[0]}"
            )
        if not expected:
            return "the exact sealed pair projection contains no auditable source"
        return ""

    @staticmethod
    def _strip_quality_checks(candidate: str) -> str:
        text = _QUALITY_CHECK_BLOCK.sub("", candidate)
        text = _QUALITY_CHECK_LINE.sub("", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def _safe_prospective() -> str:
        return (
            "In future, I would keep self-assessments source-scoped: separate execution receipts from "
            "request/response evidence, quote load-bearing facts exactly, and label suggestions as prospective "
            "instead of presenting them as observed history."
        )

    def _retry_or_replace(self, reason: str, *, prospective: bool):
        if self._revisions < self.max_revisions:
            self._revisions += 1
            state = self._state()
            pair_count = len(self._source_rows(state))
            prospective_rule = (
                f" Future advice is optional; if included, put it under the literal heading "
                f"'{_QUALITY_PROSPECTIVE_HEADING}' and describe only a future policy."
                if prospective else " Do not add prospective advice; it was not requested."
            )
            categories = ", ".join(_QUALITY_MISMATCH_CATEGORIES)
            return {
                "continue": True,
                "exclusive": True,
                "feedback_role": "system",
                "feedback": (
                    "The host rejected the unpublished self-assessment because " + reason + ". Revise from "
                    "the already-projected sealed evidence; do not run tools. Revise silently: the user never saw "
                    "the draft or this feedback. First write the private coverage line "
                    f"'I audited all {pair_count} exact request/response pairs.' Then choose one path only: CLEAN — "
                    f"write the exact terminal sentence '{_QUALITY_NO_ISSUE}.'; or ISSUE — for each supported flaw "
                    "write 'Observed issue' followed by 'Source:', JSON-string 'Requested exact:', JSON-string "
                    "'Produced exact:', and 'Mismatch: <category> — <why>', where category is one of: "
                    f"{categories}. Only an unsupported factual claim also needs 'Grounding source:' with an "
                    "admitted artifact handle and JSON-string 'Grounding exact:' from that source. The host strips "
                    "the coverage line and "
                    "checks every source and quote. The clean verdict is not universal correctness."
                    + prospective_rule
                ),
            }
        state = self._state()
        if self._coverage_complete(state):
            # This describes the observable protocol outcome, not an uncheckable capability claim. In
            # particular, complete projected evidence means "couldn't" would itself be a false explanation.
            fallback = _QUALITY_COMPLETE_AUDIT_FALLBACK + "."
        else:
            fallback = (
                "I couldn't produce an evidence-admissible self-assessment from the available sealed exchange "
                "pairs, so I won't assert either an observed flaw or universal correctness."
            )
        if prospective:
            future = (
                "In future, I would keep self-assessments source-scoped: separate execution receipts from "
                "request/response evidence, quote load-bearing facts exactly, and label suggestions as "
                "prospective instead of presenting them as observed history."
            )
            fallback += "\n\n" + _QUALITY_PROSPECTIVE_HEADING + "\n\n" + future
        return {"replacement": fallback, "exclusive": True, "reason": reason}

    def _retry_verification_or_replace(self, reason: str):
        if self._revisions < self.max_revisions:
            self._revisions += 1
            state = self._state()
            pair_count = len(self._source_rows(state))
            return {
                "continue": True,
                "exclusive": True,
                "feedback_role": "system",
                "feedback": (
                    "The host rejected the unpublished verification because " + reason + ". Revise silently: "
                    "the user never saw the draft or this feedback. Use only the exact prior response and FROZEN "
                    "projection; do not run tools. First write the private coverage line "
                    f"'I rechecked all {pair_count} exact request/response pairs.' If all remain clean, include the "
                    f"exact prior sentence '{_QUALITY_NO_ISSUE}.' If a mismatch exists, use the source-exact "
                    "Observed issue block. Quote any attributed prior claim verbatim; never paraphrase it."
                ),
            }
        return {
            "replacement": (
                "I couldn't produce a source-exact verification of the prior response from the frozen baseline, "
                "so I won't claim it was accurate or inaccurate."
            ),
            "exclusive": True,
            "reason": reason,
        }

    @staticmethod
    def _prior_assessment(state) -> tuple[str, str, str]:
        contract = QualityEvidenceCompletionHook._contract(state)
        snapshot = next((item for item in getattr(contract, "referents", ()) or ()
                         if isinstance(item, Mapping) and item.get("kind") == "evidence_snapshot"), {})
        source_id = str(snapshot.get("source_turn_id") or "")
        status = str(snapshot.get("status") or "")
        for row in reversed(getattr(state, "conversation", ()) or ()):
            if not isinstance(row, Mapping) or str(row.get("artifact_id") or "") != source_id:
                continue
            return source_id, str(row.get("assistant") or ""), status
        return source_id, "", status

    def _validate_verification(self, candidate: str, state):
        source_id, prior, snapshot_status = self._prior_assessment(state)
        if not source_id or not prior:
            return self._retry_verification_or_replace(
                "the exact immediately preceding assessment is unavailable from its frozen source handle"
            )
        if _QUALITY_COMPLETE_AUDIT_FALLBACK in prior:
            coverage = next((
                item for item in getattr(getattr(state, "runtime", None), "source_projections", ()) or ()
                if isinstance(item, Mapping) and item.get("kind") == "quality_exchange_coverage"
            ), {})
            complete = str(coverage.get("coverage") or "").casefold() == "complete"
            pairs = int(coverage.get("complete_exchange_pairs", 0) or 0)
            exact = f'"{_QUALITY_COMPLETE_AUDIT_FALLBACK}."'
            if snapshot_status == "frozen" and complete:
                pair_text = f"{pairs} exact sealed request/response pair" + ("" if pairs == 1 else "s")
                prospective_note = (
                    " Its prospective section was advice, not an observed-history claim."
                    if _QUALITY_PROSPECTIVE_HEADING in prior else ""
                )
                return {
                    "replacement": (
                        f"The prior response's exact disposition was {exact} That publication-state statement "
                        "is accurate: no response-quality verdict was admitted. The frozen projection is "
                        f"complete and contains {pair_text}, so the abstention does not mean evidence was "
                        "unavailable or an audit was impossible; it also does not verify the underlying response "
                        "quality." + prospective_note
                    ),
                    "exclusive": True,
                    "reason": "normalized verification of the host-owned no-verdict disposition",
                }
            return {
                "replacement": (
                    f"The prior response's exact disposition was {exact} That publication-state statement is "
                    "accurate: no response-quality verdict was admitted. The frozen baseline is unavailable or "
                    "incomplete, so the underlying response quality cannot be rechecked from this snapshot."
                ),
                "exclusive": True,
                "reason": "the prior host-owned no-verdict disposition has no complete frozen baseline",
            }
        if _QUALITY_NO_ISSUE in prior:
            coverage = next((
                item for item in getattr(getattr(state, "runtime", None), "source_projections", ()) or ()
                if isinstance(item, Mapping) and item.get("kind") == "quality_exchange_coverage"
            ), {})
            complete = str(coverage.get("coverage") or "").casefold() == "complete"
            pairs = int(coverage.get("complete_exchange_pairs", 0) or 0)
            exact = f'"{_QUALITY_NO_ISSUE}."'
            if snapshot_status != "frozen" or not complete:
                return {
                    "replacement": (
                        f"The prior response's exact evidence-scoped claim was {exact} The frozen baseline is "
                        "unavailable or incomplete, so I cannot verify that claim now."
                    ),
                    "exclusive": True,
                    "reason": "the prior no-issue verdict has no complete frozen verification baseline",
                }
            has_issue_heading = re.search(
                r"(?im)^\s{0,3}(?:#{1,6}\s*)?Observed issue\s*$", candidate,
            ) is not None
            if has_issue_heading:
                valid_blocks, block_error = self._validate_issue_blocks(candidate, state)
                if not valid_blocks:
                    return self._retry_verification_or_replace(block_error)
            coverage_error = self._validate_audit_coverage(candidate, state)
            if coverage_error:
                return self._retry_verification_or_replace(coverage_error)
            pair_text = f"{pairs} exact sealed request/response pair" + ("" if pairs == 1 else "s")
            if has_issue_heading:
                issues = "\n\n".join(
                    match.group(0).strip() for match in _QUALITY_ISSUE_BLOCK.finditer(candidate)
                )
                return {
                    "replacement": (
                        f"The prior response's exact evidence-scoped claim was {exact} A source-complete recheck "
                        f"of the same frozen projection of {pair_text} contradicts that claim:\n\n{issues}"
                    ),
                    "exclusive": True,
                    "reason": "the frozen source recheck found an admitted mismatch",
                }
            if _QUALITY_NO_ISSUE not in candidate:
                return self._retry_verification_or_replace(
                    "it did not retain the prior no-issue sentence after completing the frozen source audit"
                )
            return {
                "replacement": (
                    f"The prior response's exact evidence-scoped claim was {exact} Rechecking it against the "
                    f"same frozen projection of {pair_text} produces the same evidence-sufficiency verdict. "
                    + (self._execution_verification_sentence(state, prior) + " "
                       if self._execution_verification_sentence(state, prior) else "")
                    + "This verifies only those source-scoped claims; it does not "
                    "establish that every underlying answer was universally correct."
                ),
                "exclusive": True,
                "reason": "normalized verification of the canonical no-issue verdict",
            }
        matches = tuple(_QUALITY_VERIFICATION_BLOCK.finditer(candidate))
        heading_count = len(re.findall(
            r"(?im)^\s{0,3}(?:#{1,6}\s*)?Verification item\s*$", candidate,
        ))
        if len(matches) != heading_count:
            return self._retry_verification_or_replace(
                "at least one optional source-exact verification block is malformed"
            )
        copied_claims = []
        for match in matches:
            try:
                claim = json.loads(match.group("claim"))
                evidence = json.loads(match.group("evidence"))
            except (TypeError, ValueError, json.JSONDecodeError):
                return self._retry_verification_or_replace(
                    "Prior claim exact and Evidence must be valid JSON strings"
                )
            if not isinstance(claim, str) or not claim.strip() or claim not in prior:
                return self._retry_verification_or_replace(
                    f"an attributed prior claim is not verbatim text from artifacts/{source_id}.md"
                )
            if not isinstance(evidence, str) or not evidence.strip():
                return self._retry_verification_or_replace("a verification item has no frozen evidence")
            verdict = match.group("verdict").casefold()
            if snapshot_status == "unavailable" and verdict != "not verifiable":
                return self._retry_verification_or_replace(
                    "the frozen baseline is unavailable, so no item may be marked supported or contradicted"
                )
            copied_claims.append(claim)

        # At least one material claim must be copied exactly. The canonical no-issue verdict is mandatory when
        # present; otherwise an exact substantive sentence/line from the prior answer is enough to bind the check.
        if _QUALITY_NO_ISSUE in prior:
            if _QUALITY_NO_ISSUE not in candidate:
                return self._retry_verification_or_replace(
                    "it omitted the prior assessment's exact evidence-sufficiency verdict"
                )
        else:
            anchors = []
            for part in re.split(r"\n+|(?<=[.!?])\s+", prior):
                plain = re.sub(r"^[\s>*_`#-]+|[\s*_`]+$", "", part).strip()
                if len(plain) >= 24:
                    anchors.append(plain)
            if not any(anchor in candidate for anchor in anchors):
                return self._retry_verification_or_replace(
                    f"it did not copy any material prior claim verbatim from artifacts/{source_id}.md"
                )

        # Natural prose is allowed, but an attribution sentence may not add a quote, lifecycle field, or number
        # absent from the source answer. This catches plausible reconstructions such as "I referenced 4 counts"
        # when the prior answer never did, without forcing a machine-shaped response on the user.
        attribution = re.compile(
            r"\b(?:my\s+(?:prior|last|previous)\s+(?:answer|response|claim)|"
            r"what\s+I\s+(?:said|wrote|claimed)|I\s+(?:said|stated|wrote|claimed|referenced|reported))\b",
            re.IGNORECASE,
        )
        prior_lower = prior.casefold()
        lifecycle_words = (
            "requested", "started", "execution-started", "settled", "succeeded", "failed",
            "cancelled", "indeterminate", "lifecycle", "warnings", "child agents", "operations",
        )
        for sentence in re.split(r"\n+|(?<=[.!?])\s+", candidate):
            if not attribution.search(sentence):
                continue
            for quote in re.findall(r"[\"“]([^\"”]{6,})[\"”]", sentence):
                if quote not in prior:
                    return self._retry_verification_or_replace(
                        "a quoted attribution is not verbatim text from the prior response"
                    )
            for word in lifecycle_words:
                if word in sentence.casefold() and word not in prior_lower:
                    return self._retry_verification_or_replace(
                        f"the attribution adds '{word}', which the prior response never stated"
                    )
            prior_numbers = set(re.findall(r"\b\d+\b", prior))
            number_surface = re.sub(r"\bartifacts/turn-[A-Za-z0-9_-]+", "", sentence, flags=re.I)
            number_surface = re.sub(r"\bturn\s+\d+\b", "turn", number_surface, flags=re.I)
            number_surface = re.sub(r"^\s*\d+[.)]\s*", "", number_surface)
            for number in re.findall(r"\b\d+\b", number_surface):
                if number not in prior_numbers:
                    return self._retry_verification_or_replace(
                        f"the attribution adds the number {number}, which the prior response never stated"
                    )

        conclusion = re.compile(
            r"\b(?:accurate|inaccurate|supported|contradicted|not\s+verifiable|cannot\s+verify|"
            r"can't\s+verify|won't\s+claim|will\s+not\s+claim)\b",
            re.IGNORECASE,
        )
        if conclusion.search(candidate) is None:
            return self._retry_verification_or_replace(
                "it never gives an accuracy/support verdict for the exact prior claim"
            )
        if snapshot_status == "unavailable":
            unavailable = re.compile(
                r"\b(?:not\s+verifiable|cannot\s+verify|can't\s+verify|won't\s+claim|will\s+not\s+claim)\b",
                re.IGNORECASE,
            )
            if unavailable.search(candidate) is None:
                return self._retry_verification_or_replace(
                    "the frozen baseline is unavailable, so the answer must remain not verifiable"
                )
        if self._has_universal_overclaim(candidate):
            return self._retry_verification_or_replace(
                "it expands a source-scoped verification into universal correctness"
            )
        return None

    def validate_completion(self, candidate: str, stop_reason: str):
        if stop_reason != "end_turn":
            return None
        state = self._state()
        contract = self._contract(state)
        query = getattr(contract, "quality_evidence_query", None)
        purpose = str(getattr(query, "purpose", "assess") or "assess") if query is not None else ""
        if purpose not in {"assess", "verify_assessment"}:
            return None
        prospective = bool(getattr(query, "prospective_requested", False))
        text = str(candidate or "")
        deterministic_issues = self._canonical_deterministic_issues(state)
        if deterministic_issues:
            if purpose == "verify_assessment":
                _source_id, prior, snapshot_status = self._prior_assessment(state)
                if snapshot_status != "frozen" or not self._coverage_complete(state):
                    return self._validate_verification(text, state)
                relation = (
                    "supports the prior assessment" if "Observed issue" in prior
                    else "contradicts the prior assessment's clean/no-verdict disposition"
                )
                execution = self._execution_verification_sentence(state, prior)
                replacement = (
                    "A source-complete recheck of the same frozen request/response projection "
                    + relation + ":\n\n" + deterministic_issues
                    + (("\n\n" + execution) if execution else "")
                )
                if text.strip() == replacement:
                    return None
                return {
                    "replacement": replacement, "exclusive": True,
                    "reason": "verified host-measured explicit response constraints against the frozen baseline",
                }
            replacement = deterministic_issues
            if prospective:
                replacement += (
                    "\n\n" + _QUALITY_PROSPECTIVE_HEADING + "\n\n"
                    + self._deterministic_prospective(state)
                )
            if text.strip() == replacement:
                return None
            return {
                "replacement": replacement, "exclusive": True,
                "reason": "admitted host-measured explicit response-constraint mismatch",
            }
        if self._revisions and _QUALITY_INTERNAL_FEEDBACK_LEAK.search(text):
            return self._retry_or_replace(
                "it exposed internal correction feedback that the user never saw",
                prospective=prospective,
            )
        numeric_error = self._validate_numeric_evidence(text, state)
        # Assessment outputs are normalized to host-owned execution prose or source-exact issue blocks, so a
        # wrong model-written lifecycle preamble is removed rather than consuming retries. Generic verification
        # prose remains model-owned and still needs numeric checking inside its validator.
        if purpose == "verify_assessment":
            prior = self._prior_assessment(state)[1]
            if _QUALITY_COMPLETE_AUDIT_FALLBACK in prior:
                return self._validate_verification(text, state)
            if numeric_error and _QUALITY_NO_ISSUE not in prior:
                return self._retry_verification_or_replace(numeric_error)
            return self._validate_verification(text, state)
        universal_overclaim = self._has_universal_overclaim(text)

        uncertainty_at = text.rfind(_QUALITY_UNCERTAIN)
        if uncertainty_at >= 0:
            uncertainty_end = uncertainty_at + len(_QUALITY_UNCERTAIN)
            after = text[uncertainty_end:]
            if not prospective:
                if re.search(r"[A-Za-z0-9]", after):
                    return {"replacement": self._terminal_replacement(
                                text, uncertainty_at, _QUALITY_UNCERTAIN), "exclusive": True,
                            "reason": "unsupported prose followed the terminal uncertainty verdict"}
                return None
            tail = self._prospective_tail(after)
            if tail is None:
                return self._retry_or_replace(
                    f"requested future advice was not separated under '{_QUALITY_PROSPECTIVE_HEADING}'",
                    prospective=True,
                )
            if _QUALITY_RETROSPECTIVE_PROSPECTIVE.search(tail):
                return self._retry_or_replace(
                    "the prospective section relabeled advice as an observed past failure",
                    prospective=True,
                )
            return None

        verdict_at = text.rfind(_QUALITY_NO_ISSUE)
        if verdict_at >= 0:
            if not self._coverage_complete(state):
                return self._retry_or_replace(
                    f"the evidence coverage is incomplete; use the exact uncertainty sentence '{_QUALITY_UNCERTAIN}.'",
                    prospective=prospective,
                )
            verdict_end = verdict_at + len(_QUALITY_NO_ISSUE)
            after = text[verdict_end:]
            valid_blocks, _ = self._validate_issue_blocks(text, state)
            if "Observed issue" in text or valid_blocks:
                return self._retry_or_replace(
                    "it both alleged an observed flaw and declared that no flaw was supported",
                    prospective=prospective,
                )
            # Private coverage is position-insensitive and stripped before publication. Models commonly place
            # the attestation immediately after the scoped verdict; rejecting that order adds no truth value.
            coverage_error = self._validate_audit_coverage(text, state)
            if coverage_error:
                if universal_overclaim:
                    coverage_error = (
                        "it converted an evidence-sufficiency result into a universal correctness claim and "
                        "did not supply a source-complete private audit"
                    )
                return self._retry_or_replace(coverage_error, prospective=prospective)
            if not prospective:
                replacement = self._scoped_no_issue_answer(state)
                if text.strip() == replacement:
                    return None
                return {"replacement": replacement, "exclusive": True,
                        "reason": "normalized execution and quality claims into separate host-owned scopes"}
            tail = self._prospective_tail(after)
            if tail is None:
                # The source audit and scoped verdict are already host-valid. A missing optional prose tail is
                # presentation, not evidence failure; finish the user's future-facing request with the safe,
                # source-neutral policy instead of spending another model retry that can reintroduce history.
                return {
                    "replacement": self._scoped_no_issue_answer(state, self._safe_prospective()),
                    "exclusive": True,
                    "reason": "added safe prospective guidance after a complete clean audit",
                }
            if _QUALITY_RETROSPECTIVE_PROSPECTIVE.search(tail):
                return {
                    "replacement": self._scoped_no_issue_answer(state, self._safe_prospective()),
                    "exclusive": True,
                    "reason": "replaced retrospective claims inside the prospective section",
                }
            if self._has_universal_overclaim(tail):
                return {
                    "replacement": self._scoped_no_issue_answer(state, self._safe_prospective()),
                    "exclusive": True,
                    "reason": "replaced a universal correctness claim inside the prospective section",
                }
            replacement = self._scoped_no_issue_answer(state, tail)
            if text.strip() == replacement:
                return None
            return {"replacement": replacement, "exclusive": True,
                    "reason": "normalized execution and quality claims into separate host-owned scopes"}

        equivalent = _QUALITY_NO_ISSUE_EQUIVALENT.search(text)
        if equivalent is not None:
            if not self._coverage_complete(state):
                return self._retry_or_replace(
                    f"the evidence coverage is incomplete; use the exact uncertainty sentence '{_QUALITY_UNCERTAIN}.'",
                    prospective=prospective,
                )
            if _QUALITY_SPECULATIVE_CRITIQUE.search(text):
                return self._retry_or_replace(
                    "a no-issue paraphrase is mixed with an unadmitted retrospective critique",
                    prospective=prospective,
                )
            coverage_error = self._validate_audit_coverage(text, state)
            if coverage_error:
                if universal_overclaim:
                    coverage_error = (
                        "it converted an evidence-sufficiency result into a universal correctness claim and "
                        "did not supply a source-complete private audit"
                    )
                return self._retry_or_replace(coverage_error, prospective=prospective)
            if not prospective:
                return {"replacement": self._scoped_no_issue_answer(state), "exclusive": True,
                        "reason": "normalized a supported no-issue paraphrase to the exact terminal verdict"}
            tail = self._prospective_tail(text[equivalent.end():])
            if tail is None:
                return {
                    "replacement": self._scoped_no_issue_answer(state, self._safe_prospective()),
                    "exclusive": True,
                    "reason": "added safe prospective guidance after a complete clean audit",
                }
            if _QUALITY_RETROSPECTIVE_PROSPECTIVE.search(tail):
                return {
                    "replacement": self._scoped_no_issue_answer(state, self._safe_prospective()),
                    "exclusive": True,
                    "reason": "replaced retrospective claims inside the prospective section",
                }
            if self._has_universal_overclaim(tail):
                return {
                    "replacement": self._scoped_no_issue_answer(state, self._safe_prospective()),
                    "exclusive": True,
                    "reason": "replaced a universal correctness claim inside the prospective section",
                }
            return {"replacement": self._scoped_no_issue_answer(state, tail),
                    "exclusive": True,
                    "reason": "normalized a supported no-issue paraphrase to the exact verdict"}

        valid_blocks, block_error = self._validate_issue_blocks(text, state)
        if not valid_blocks:
            return self._retry_or_replace(block_error, prospective=prospective)
        coverage_error = self._validate_audit_coverage(text, state)
        if coverage_error:
            return self._retry_or_replace(coverage_error, prospective=prospective)
        prospective_tail = ""
        if prospective:
            marker = text.find(_QUALITY_PROSPECTIVE_HEADING)
            if marker < 0:
                prospective_tail = self._safe_prospective()
            elif _QUALITY_RETROSPECTIVE_PROSPECTIVE.search(
                    text[marker + len(_QUALITY_PROSPECTIVE_HEADING):]):
                return self._retry_or_replace(
                    "the prospective section relabeled advice as an observed past failure",
                    prospective=True,
                )
            else:
                prospective_tail = text[marker + len(_QUALITY_PROSPECTIVE_HEADING):].strip(" \t:-—\r\n")
            if self._has_universal_overclaim(prospective_tail):
                return self._retry_or_replace(
                    "the prospective section added a retrospective universal correctness claim",
                    prospective=True,
                )
        elif _QUALITY_PROSPECTIVE_HEADING in text:
            return self._retry_or_replace("prospective advice was not requested", prospective=False)
        issues = "\n\n".join(match.group(0).strip() for match in _QUALITY_ISSUE_BLOCK.finditer(text))
        replacement = issues
        if prospective_tail:
            replacement += "\n\n" + _QUALITY_PROSPECTIVE_HEADING + "\n\n" + prospective_tail
        if text.strip() == replacement:
            return None
        return {
            "replacement": replacement,
            "exclusive": True,
            "reason": "removed private audit checks and normalized the admitted observed-issue report",
        }


class FrozenEvidenceCutoffHook(Hooks):
    """Keep an adjacent evidence challenge on its immutable as-of projection.

    This is deliberately narrower than an authority policy. Dialogue remains available, and ordinary/mixed-live
    turns are untouched. During a pure frozen verification, observation may only reopen an exact artifact already
    named by that snapshot; moving indexes, live workspace reads, shell probes, network reads, and fresh observers
    would silently change the cutoff being verified.
    """

    def __init__(self, state_provider, effect_resolver, resource_resolver):
        self.state_provider = state_provider
        self.effect_resolver = effect_resolver
        self.resource_resolver = resource_resolver

    @staticmethod
    def _field(value, name, default=None):
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def _context(self):
        try:
            state = self.state_provider()
            contract = self._field(self._field(state, "intent"), "turn_contract")
        except Exception:  # an unavailable optional cutoff does not replace the ordinary fail-closed authority gate
            return None
        if not contract or not bool(self._field(contract, "evidence_continuation", False)):
            return None
        if str(self._field(contract, "grounding", "") or "") != "sealed_past":
            return None
        if "current_world" in set(self._field(contract, "source_needs", ()) or ()):
            return None
        if str(self._field(state, "reconciliation_required", "") or ""):
            return None
        referents = tuple(self._field(contract, "referents", ()) or ())
        snapshot = next((item for item in referents if isinstance(item, Mapping)
                         and item.get("kind") == "evidence_snapshot"), None)
        status = str((snapshot or {}).get("status") or "")
        if status not in {"frozen", "unavailable"}:
            return None
        allowed: set[str] = set()
        if status == "frozen":
            source_turn_id = str((snapshot or {}).get("source_turn_id") or "").strip()
            if source_turn_id:
                allowed.add(f"artifacts/{source_turn_id}.md")
            for item in referents:
                if not isinstance(item, Mapping) or item.get("kind") != "execution_receipt":
                    continue
                artifact_id = str(item.get("artifact_id") or "").strip()
                if artifact_id:
                    allowed.add(f"artifacts/{artifact_id}.md")
            runtime = self._field(state, "runtime")
            for item in self._field(runtime, "source_projections", ()) or ():
                if not isinstance(item, Mapping) or item.get("kind") != "quality_exchange":
                    continue
                artifact_id = str(item.get("artifact_id") or "").strip()
                if artifact_id:
                    allowed.add(f"artifacts/{artifact_id}.md")
                for grounding in item.get("grounding_artifacts") or ():
                    if not isinstance(grounding, Mapping):
                        continue
                    grounding_id = str(grounding.get("artifact_id") or "").strip()
                    if grounding_id:
                        allowed.add(f"artifacts/{grounding_id}.md")
        return status, frozenset(allowed)

    @staticmethod
    def _normalized_path(value) -> str:
        path = str(value or "").strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        return path.rstrip("/")

    def authorize_tool(self, name, args):
        context = self._context()
        if context is None:
            return ALLOW
        try:
            effect = coerce_intent_effect(self.effect_resolver(name, args or {}))
        except Exception:  # TurnAuthorityHook remains the owner of unknown/errored effect metadata
            return ALLOW
        if effect is not ToolIntentEffect.OBSERVE:
            return ALLOW
        status, allowed = context
        path = self._normalized_path((args or {}).get("path"))
        if name == "read_file" and status == "frozen" and path in allowed:
            try:
                ref = self.resource_resolver(path)
            except Exception:
                ref = None
            if (getattr(ref, "kind", None) is ResourceKind.ARTIFACT
                    and self._normalized_path(getattr(ref, "handle", "")) == path):
                return ALLOW
        detail = (
            "no artifact reads are available because the frozen baseline is unavailable"
            if status == "unavailable" else
            "allowed immutable reads: " + (", ".join(sorted(allowed)) if allowed else "(none)")
        )
        return ToolDecision(
            False,
            "frozen_evidence_cutoff: this verification must use the prior response's as-of projection; "
            "live or moving observation would change the claim being checked; " + detail,
            counts_as_stuck=False,
        )


class TurnAuthorityHook(Hooks):
    """Enforce the current turn's semantic authority before ordinary permission policy.

    ``contract_provider`` supplies either a contract with ``effect_authority``, a mapping containing that
    key, or the authority string itself. ``effect_resolver(name, args)`` supplies the registry-owned
    :class:`ToolIntentEffect` for the exact call. This hook deliberately does not remember a "first block":
    every retry and every sibling call is independently gated for the whole turn.

    Turn authority and permission policy are orthogonal. Explicit/continuation authority merely abstains
    here (ALLOW); a later PermissionHook may still confirm or deny the call. With none/uncertain authority,
    observation and dialogue remain available while task-state, external, and unknown effects fail closed.
    """

    _AUTHORIZED = frozenset({"explicit", "continuation"})
    _RESTRICTED = frozenset({"none", "uncertain"})
    _NON_EFFECTFUL = frozenset({ToolIntentEffect.OBSERVE, ToolIntentEffect.DIALOGUE})
    # Reconciliation is not user-requested work. It is the kernel's recovery handshake after an
    # indeterminate effect, and ReconciliationHook (which runs before this hook) independently proves that
    # every affected target was re-observed before allowing it. Blocking that handshake on a conversational
    # follow-up would leave the runtime permanently unable to return to a known state.
    _RECOVERY_TOOLS = frozenset({"reconcile_execution"})
    # NAVIGATION tier: change_workspace only re-points the working directory. It is REVERSIBLE and destroys
    # nothing, so it is authorized by navigation INTENT (a workspace.navigate grant) rather than by an exact
    # target match — the user directs where to go and can correct it in one more turn. Kept a set so a future
    # navigation tool joins the tier by name; mirrors policy.NAVIGATION_TOOLS (not imported: policy→hooks cycle).
    _NAVIGATION_TOOLS = frozenset({"change_workspace"})

    def __init__(self, contract_provider, effect_resolver):
        self.contract_provider = contract_provider
        self.effect_resolver = effect_resolver

    def _contract(self):
        try:
            return self.contract_provider()
        except Exception:  # unavailable intent state must not silently grant effects
            return None

    def _authority(self, contract=None) -> str:
        contract = self._contract() if contract is None else contract
        if isinstance(contract, Mapping):
            value = contract.get("effect_authority")
        elif hasattr(contract, "effect_authority"):
            value = getattr(contract, "effect_authority")
        else:
            value = contract
        if hasattr(value, "value"):
            value = value.value
        normalized = str(value or "").strip().casefold()
        return normalized if normalized in self._AUTHORIZED | self._RESTRICTED else "uncertain"

    @staticmethod
    def _effect_grants(contract) -> tuple[object, ...]:
        if isinstance(contract, Mapping):
            grants = contract.get("effect_grants") or ()
        else:
            grants = getattr(contract, "effect_grants", ()) or ()
        if grants:
            return tuple(grants)
        # Compatibility for a typed proposal produced by an older discourse projection. Untyped proposal
        # prose intentionally produces no grant and therefore fails closed for every effectful call.
        if isinstance(contract, Mapping):
            referents = contract.get("referents") or ()
        else:
            referents = getattr(contract, "referents", ()) or ()
        actions: list[dict] = []
        for referent in referents:
            if not isinstance(referent, Mapping) or referent.get("kind") != "pending_proposal":
                continue
            action = referent.get("action")
            if isinstance(action, Mapping) and str(action.get("tool") or "").strip():
                actions.append({
                    "operation": f"exact:{action.get('tool')}",
                    "tools": (str(action.get("tool")),),
                    "exact_args": dict(action.get("args") or {}),
                    "mode": "exact",
                })
        return tuple(actions)

    @staticmethod
    def _field(grant, name: str, default=None):
        return grant.get(name, default) if isinstance(grant, Mapping) else getattr(grant, name, default)

    @staticmethod
    def _freeze(value):
        if isinstance(value, Mapping):
            return tuple(sorted((str(key), TurnAuthorityHook._freeze(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(TurnAuthorityHook._freeze(item) for item in value)
        return value

    @staticmethod
    def _same_path(expected: str, actual: str, *, basename_anywhere: bool = False) -> bool:
        expected = str(expected or "").strip().strip("`\"'")
        actual = str(actual or "").strip().strip("`\"'")
        if not expected or not actual:
            return False
        expected_expanded = os.path.expanduser(expected)
        actual_expanded = os.path.expanduser(actual)
        try:
            if os.path.isabs(expected_expanded) or "/" in expected or "\\" in expected:
                return os.path.normcase(os.path.normpath(expected_expanded)) == \
                    os.path.normcase(os.path.normpath(actual_expanded))
        except (OSError, ValueError):
            pass
        # A bare filename is root-relative authority, not permission for every nested file with the same
        # basename. Workspace navigation is the one explicit exception: users naturally say "Hunter" while
        # the tool resolves it to /Users/.../Hunter, and callers opt into that behavior below.
        actual_dir = os.path.dirname(os.path.normpath(actual_expanded))
        if actual_dir not in ("", ".") and not basename_anywhere:
            return False
        wanted = os.path.basename(expected_expanded).casefold()
        got = os.path.basename(actual_expanded).casefold()
        if wanted == got:
            return True
        # Spoken navigation shorthand: the user says "loom" and disambiguates (via the ask_user menu or a
        # follow-up) to the real directory "loom-app"/"loom_v2" whose name begins with that token at a
        # component boundary. Navigation only (basename_anywhere) and reversible; a bare edit target still
        # needs an exact basename. ponytail: covers stem→dir disambiguation, not arbitrary ask_user renames.
        if basename_anywhere and wanted and any(got.startswith(f"{wanted}{sep}") for sep in ("-", "_", ".", " ")):
            return True
        # Keep only the conversational shorthand we actually promise.  A generic stem equivalence made
        # "README" authorize README.py and "Makefile" authorize Makefile.py.
        return wanted in {"readme", "changelog", "license"} and got == f"{wanted}.md"

    @staticmethod
    def _same_text(expected: str, actual: str) -> bool:
        def normalize(value: str) -> str:
            return " ".join(str(value or "").strip(" \t\r\n`\"'“”‘’.!?;:").split()).casefold()
        return bool(normalize(expected) and normalize(expected) == normalize(actual))

    @staticmethod
    def _path_in_prefix(prefix: str, actual: str) -> bool:
        prefix = os.path.normpath(os.path.expanduser(str(prefix or "").strip().strip("`\"'")))
        actual = os.path.normpath(os.path.expanduser(str(actual or "").strip().strip("`\"'")))
        if not prefix or not actual or actual == ".." or actual.startswith(f"..{os.sep}"):
            return False
        try:
            return os.path.commonpath((prefix, actual)) == prefix and actual != prefix
        except (OSError, ValueError):
            return False

    @staticmethod
    def _path_matches_glob(pattern: str, actual: str) -> bool:
        pattern = str(pattern or "").strip().strip("`\"'")
        actual = os.path.normpath(str(actual or "").strip().strip("`\"'"))
        if not pattern or not actual or os.path.isabs(actual) or actual == ".." \
                or actual.startswith(f"..{os.sep}"):
            return False
        if "/" not in pattern and "\\" not in pattern:
            return fnmatch.fnmatchcase(os.path.basename(actual), pattern)
        normalized_actual = actual.replace("\\", "/")
        normalized_pattern = pattern.replace("\\", "/")
        return fnmatch.fnmatchcase(normalized_actual, normalized_pattern) or (
            normalized_pattern.startswith("**/")
            and fnmatch.fnmatchcase(normalized_actual, normalized_pattern[3:])
        )

    @staticmethod
    def _has_shell_control(raw: str) -> bool:
        """Detect shell composition/expansion outside literal single quotes.

        ``shlex.split`` alone is not enough: it happily tokenizes ``git$IFS push`` and grouped programs.
        This small scanner still permits ordinary punctuation inside quoted test expressions while rejecting
        every construct that can turn one inspected argv into a different or second program.
        """
        quote = ""
        escaped = False
        for char in raw:
            if quote == "'":
                if char == "'":
                    quote = ""
                continue
            if escaped:
                if char == "\n":
                    return True
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if quote == '"':
                if char == '"':
                    quote = ""
                elif char in "$`":
                    return True
                continue
            if char in "'\"":
                quote = char
            elif char in ";&|<>`()$\n\r":
                return True
        return False

    @classmethod
    def _simple_command(cls, command: str) -> tuple[str, ...]:
        raw = str(command or "").strip()
        if not raw or cls._has_shell_control(raw):
            return ()
        try:
            tokens = shlex.split(raw)
        except ValueError:
            return ()
        while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0], re.DOTALL):
            name, value = tokens.pop(0).split("=", 1)
            safe_names = {
                "CI", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "NODE_ENV",
                "RUST_BACKTRACE", "RUST_LOG", "CARGO_TERM_COLOR", "FORCE_COLOR", "NO_COLOR",
                "TERM", "TZ", "LANG", "LC_ALL", "LC_CTYPE",
            }
            if name == "PYTHONPATH":
                parts = value.split(os.pathsep)
                if not parts or any(not cls._local_executable(part or ".") for part in parts):
                    return ()
            elif name not in safe_names:
                return ()
        return tuple(tokens)

    @staticmethod
    def _local_executable(token: str) -> bool:
        """Accept PATH names and explicitly workspace-relative executables, never absolute/parent paths."""
        token = str(token or "")
        if not token or token.startswith("~") or os.path.isabs(token):
            return False
        normalized = os.path.normpath(token)
        return normalized not in {"..", "~"} and not normalized.startswith(f"..{os.sep}")

    @classmethod
    def _verification_command(cls, command: str) -> bool:
        tokens = cls._simple_command(command)
        if not tokens or not cls._local_executable(tokens[0]):
            return False
        verb = os.path.basename(tokens[0]).casefold()
        rest = [token.casefold() for token in tokens[1:]]
        for token in tokens[1:]:
            candidate = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
            if candidate.startswith(("/", "~", "..")) or os.path.isabs(candidate):
                return False
        unsafe_option_markers = (
            "--fix", "--write", "--output", "--junitxml", "--html", "--json-report",
            "--generate-trace", "--generatetrace", "--cache-dir", "--basetemp", "--result-log",
            "--deploy", "--publish", "--release", "--install",
        )
        if any(
            token in {"deploy", "publish", "release", "install"}
            or any(token == marker or token.startswith(marker + "=") or token.startswith(marker + "-")
                   for marker in unsafe_option_markers)
            or token.startswith("--cov-report=") and token.split("=", 1)[1] not in {"term", "term-missing"}
            for token in rest
        ):
            return False
        if verb in {"python", "python3"}:
            if len(rest) < 2 or rest[0] != "-m" or rest[1] not in {"pytest", "unittest", "ruff", "mypy"}:
                return False
            nested = " ".join(shlex.quote(token) for token in tokens[2:])
            return cls._verification_command(nested)
        if verb in {"bash", "sh"}:
            return bool(rest) and not rest[0].startswith("-") and cls._local_executable(tokens[1]) \
                and bool(re.search(r"(?:test|check|lint|verify)", os.path.basename(rest[0]))) \
                and not re.search(r"(?:deploy|publish|release|install)", os.path.basename(rest[0]))
        if verb == "ruff":
            return "--fix" not in rest and not (rest[:1] == ["format"] and "--check" not in rest)
        if verb in {"eslint", "biome"}:
            return not any(token in {"--fix", "--write"} for token in rest)
        if verb == "tsc":
            return "--noemit" in rest
        if verb in {"pytest", "mypy", "pyright"}:
            return True
        if verb in {"npm", "pnpm", "yarn"}:
            return bool(rest) and (
                rest[0] in {"test", "lint", "check", "typecheck", "build"}
                or (len(rest) >= 2 and rest[0] == "run"
                    and rest[1] in {"test", "lint", "check", "typecheck", "build"})
            )
        if verb == "cargo":
            return bool(rest) and rest[0] in {"test", "check", "build", "clippy"}
        if verb == "go":
            return bool(rest) and rest[0] in {"test", "vet", "build"}
        if verb == "dotnet":
            return bool(rest) and rest[0] in {"test", "build"}
        if verb in {"mvn", "mvnw", "gradle", "gradlew"}:
            return any(token in {"test", "check", "build", "verify"} for token in rest[:3])
        if verb == "make":
            return bool(rest) and rest[0] in {"test", "check", "lint", "verify", "build"}
        if verb == "ninja":
            return not any(token in {"install", "deploy", "publish", "release"} for token in rest[:2])
        if verb == "cmake":
            return bool(rest) and rest[0] == "--build" \
                and not any(token in {"install", "deploy", "publish", "release"} for token in rest)
        if verb in {"tox", "nox"}:
            return True
        if verb in {"uv", "poetry"} and rest[:1] == ["run"]:
            nested = " ".join(shlex.quote(token) for token in tokens[2:])
            return cls._verification_command(nested)
        return False

    @classmethod
    def _implementation_command(cls, command: str) -> bool:
        """Admit ordinary local construction through a positive, one-program vocabulary."""
        if cls._verification_command(command):
            return True
        tokens = cls._simple_command(command)
        if not tokens or not cls._local_executable(tokens[0]):
            return False
        verb = os.path.basename(tokens[0]).casefold()
        rest = [token.casefold() for token in tokens[1:]]
        for token in tokens[1:]:
            candidate = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
            if candidate.startswith(("/", "~", "..")) or os.path.isabs(candidate):
                return False

        if verb in {"python", "python3"}:
            if not rest or any(token in {"-c", "--command"} for token in rest):
                return False
            if rest[0] == "-m":
                return len(rest) >= 2 and rest[1] in {
                    "build", "compileall", "pip", "pytest", "unittest", "ruff", "mypy", "coverage",
                }
            script = next((token for token in tokens[1:] if not token.startswith("-")), "")
            return bool(script and script.casefold().endswith(".py") and cls._local_executable(script)
                        and not re.search(r"(?:deploy|publish|release)", os.path.basename(script), re.I))
        if verb in {"node", "deno", "bun", "ruby", "perl", "php"}:
            if not rest or any(token in {"-e", "--eval", "eval", "-r"} for token in rest):
                return False
            script = tokens[1]
            return cls._local_executable(script) and not re.search(
                r"(?:deploy|publish|release)", os.path.basename(script), re.I,
            )
        if verb in {"bash", "sh"}:
            return bool(rest) and not rest[0].startswith("-") and cls._local_executable(tokens[1]) \
                and not re.search(r"(?:deploy|publish|release)", os.path.basename(tokens[1]), re.I)
        if verb in {"npm", "pnpm", "yarn"}:
            if not rest or rest[0] in {"exec", "dlx", "publish", "deploy", "release"}:
                return False
            return rest[0] in {
                "install", "ci", "add", "remove", "update", "upgrade", "test", "run", "build",
                "lint", "check", "typecheck",
            } and not any(token in {"publish", "deploy", "release"} for token in rest[1:3])
        if verb in {"pip", "pip3"}:
            return bool(rest) and rest[0] in {"install", "uninstall", "wheel", "check", "compile"}
        if verb == "uv":
            if rest[:1] == ["run"]:
                nested = " ".join(shlex.quote(token) for token in tokens[2:])
                return cls._implementation_command(nested)
            return bool(rest) and (
                rest[0] in {"sync", "lock", "add", "remove"}
                or (len(rest) >= 2 and rest[0] == "pip" and rest[1] in {"install", "uninstall", "compile"})
            )
        if verb in {"poetry", "pdm"}:
            if rest[:1] == ["run"]:
                nested = " ".join(shlex.quote(token) for token in tokens[2:])
                return cls._implementation_command(nested)
            return bool(rest) and rest[0] in {"install", "add", "remove", "update", "lock", "build"}
        if verb in {"cargo", "go", "dotnet", "mvn", "mvnw", "gradle", "gradlew", "make", "ninja", "cmake"}:
            if any(token in {"deploy", "publish", "release", "install"} for token in rest[:3]):
                return False
            if verb == "make" and any(token == "-f" or token.startswith("-f") for token in rest):
                return False
            return True
        if verb in {"pre-commit", "black", "isort", "ruff", "mypy", "pyright", "eslint", "tsc", "biome"}:
            return True
        if verb in {"rm", "mkdir", "touch", "cp", "mv"}:
            operands = [token for token in tokens[1:] if not token.startswith("-")]
            minimum = 2 if verb in {"cp", "mv"} else 1
            if verb == "rm" and any(token in {".", "*", "./*", ".git", "./.git"} for token in operands):
                return False
            return len(operands) >= minimum and all(cls._local_executable(token) for token in operands)
        if verb == "docker":
            return rest[:1] == ["build"] and not any(
                token == "--push" or token.startswith("--output") and "registry" in token
                for token in rest
            ) and all(
                cls._local_executable(token) for token in tokens[2:] if not token.startswith("-")
            )
        # A checked-in relative generator is part of the trusted workspace; absolute and parent paths were
        # rejected above.  It still cannot be a release-named script without a separately explicit request.
        if "/" in tokens[0] or "\\" in tokens[0]:
            return not re.search(r"(?:deploy|publish|release)", os.path.basename(tokens[0]), re.I)
        return False

    @classmethod
    def _operation_command(cls, operation: str, command: str, hint: str = "") -> bool:
        tokens = cls._simple_command(command)
        if not tokens or not cls._local_executable(tokens[0]):
            return False
        verb = os.path.basename(tokens[0]).casefold()
        rest = [token.casefold() for token in tokens[1:]]
        for token in tokens[1:]:
            candidate = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
            if candidate.startswith(("/", "~", "..")) or os.path.isabs(candidate):
                return False

        def git_parts():
            if verb != "git":
                return "", []
            index = 0
            while index < len(tokens) - 1:
                token = tokens[index + 1]
                if token == "-C" and index + 2 < len(tokens) \
                        and cls._local_executable(tokens[index + 2]):
                    index += 2
                    continue
                if token.startswith("-"):
                    return "", []
                return token.casefold(), list(tokens[index + 2:])
            return "", []

        git_sub, git_args = git_parts()
        git_folded = [token.casefold() for token in git_args]
        scoped_paths = [path for path in hint.split(" | ") if path]

        def exact_scoped_paths(paths) -> bool:
            return len(paths) == len(scoped_paths) and all(
                any(cls._same_path(expected, actual) for actual in paths) for expected in scoped_paths
            )

        if operation in {"vcs.commit", "vcs.stage"} and git_sub == "add":
            paths = [token for token in git_args if not token.startswith("-")]
            allowed_flags = {"-a", "--all", "-u", "--update", "-p", "--patch", "--intent-to-add"}
            if not all(token.casefold() in allowed_flags for token in git_args if token.startswith("-")):
                return False
            if scoped_paths:
                return exact_scoped_paths(paths)
            return (bool(paths) and all(cls._local_executable(path) for path in paths)
                    or any(token in {"-a", "--all", "-u", "--update"} for token in git_folded))
        if operation == "vcs.commit":
            dangerous = {"--amend", "--fixup", "--squash", "--reuse-message", "-c", "--reedit-message"}
            if git_sub != "commit" or any(
                token in dangerous or token.startswith(("--fixup=", "--squash=", "--reuse-message="))
                for token in git_folded
            ):
                return False
            if scoped_paths:
                paths = []
                skip_value = False
                value_flags = {"-m", "--message", "-f", "--file", "--author", "--date", "--cleanup", "--trailer"}
                for token in git_args:
                    if skip_value:
                        skip_value = False
                    elif token.casefold() in value_flags:
                        skip_value = True
                    elif token == "--":
                        continue
                    elif not token.startswith("-"):
                        paths.append(token)
                return "-a" not in git_folded and "--all" not in git_folded and exact_scoped_paths(paths)
            return True
        if operation == "vcs.push":
            dangerous = {"--delete", "-d", "--force", "-f", "--mirror", "--prune", "--all", "--tags"}
            allowed = git_sub == "push" and not any(
                token in dangerous or token.startswith("--force-") for token in git_folded
            ) and not any(token.startswith("+") or token.startswith(":") for token in git_args)
            if not allowed or not scoped_paths:
                return allowed
            target = scoped_paths[0].casefold()
            refs = [token.casefold() for token in git_args if not token.startswith("-")]
            return any(
                ref == target or ref.endswith("/" + target) or ref.rsplit(":", 1)[-1] == target
                for ref in refs[1:]
            )
        if operation == "vcs.revert":
            if git_sub == "restore":
                paths = [token for token in git_args if not token.startswith("-")]
                return bool(paths) and not any(path in {".", "*", "./*", ".git", "./.git"} for path in paths) \
                    and (exact_scoped_paths(paths) if scoped_paths else
                                        all(cls._local_executable(path) for path in paths)) \
                    and not any(token in {"--staged", "-s", "--source"} or token.startswith("--source=")
                                for token in git_folded)
            return not scoped_paths and git_sub == "revert" and bool(git_args) \
                and not any(token in {"--no-commit", "--abort", "--continue", "--quit", "--skip"}
                            for token in git_folded)
        if operation == "dependency.install":
            unsafe = {"-g", "--global", "--user", "--target", "--prefix", "--root", "--install-option"}
            if any(token in unsafe or any(token.startswith(flag + "=") for flag in unsafe if flag.startswith("--"))
                   for token in rest):
                return False
            package_args = None
            if verb in {"npm", "pnpm", "yarn"} and rest[:1] in (["install"], ["ci"], ["add"]):
                package_args = list(tokens[2:])
            elif verb in {"pip", "pip3"} and rest[:1] == ["install"]:
                package_args = list(tokens[2:])
            elif verb in {"python", "python3"} and rest[:3] == ["-m", "pip", "install"]:
                package_args = list(tokens[4:])
            elif verb == "uv" and rest[:1] in (["sync"], ["add"]):
                package_args = list(tokens[2:])
            elif verb in {"poetry", "pdm"} and rest[:1] in (["install"], ["add"]):
                package_args = list(tokens[2:])
            if package_args is None:
                return False
            values = []
            skip = False
            value_flags = {"-r", "--requirement", "-c", "--constraint", "-i", "--index-url", "--extra-index-url"}
            for token in package_args:
                folded_token = token.casefold()
                if skip:
                    skip = False
                elif folded_token in value_flags:
                    skip = True
                elif not token.startswith("-"):
                    values.append(token)

            def package_name(value: str) -> str:
                folded_value = value.casefold().strip()
                if folded_value.startswith("@") and "@" in folded_value[1:]:
                    folded_value = folded_value.rsplit("@", 1)[0]
                elif not folded_value.startswith("@"):
                    folded_value = re.split(r"[@<>=!~\[]", folded_value, maxsplit=1)[0]
                return folded_value

            if scoped_paths:
                return len(values) == len(scoped_paths) and {
                    package_name(value) for value in values
                } == {package_name(value) for value in scoped_paths}
            return not values
        if operation == "package.publish":
            return (
                verb in {"npm", "pnpm", "yarn", "cargo", "poetry"} and rest[:1] == ["publish"]
            ) or (verb == "twine" and rest[:1] == ["upload"]) \
                or (verb == "gem" and rest[:1] == ["push"])
        if operation == "workspace.deploy":
            allowed = False
            if verb == "deploy":
                allowed = True
            elif verb in {"vercel", "netlify", "fly", "render"}:
                allowed = not rest or rest[0] == "deploy" or (verb == "vercel" and rest[0].startswith("--"))
            elif verb == "kubectl":
                allowed = rest[:1] == ["apply"]
            elif verb == "terraform":
                allowed = rest[:1] == ["apply"]
            elif verb == "pulumi":
                allowed = rest[:1] == ["up"]
            if not allowed or not scoped_paths:
                return allowed
            environment = scoped_paths[0].casefold()
            if environment in {"staging", "stage", "preview", "development", "dev", "test"}:
                return not any(token in {"--prod", "--production", "production", "prod"} for token in rest)
            if environment in {"production", "prod"}:
                return any(token in {"--prod", "--production", "production", "prod"} for token in rest)
            return any(environment == token.strip("-=") for token in rest)
        if operation == "workspace.delete":
            operands = [token for token in tokens[1:] if not token.startswith("-")]
            return verb in {"rm", "unlink"} and len(operands) == 1 \
                and cls._same_path(hint, operands[0]) \
                and all(token in {"-f", "--force", "--"} for token in tokens[1:] if token.startswith("-"))
        if operation in {"workspace.rename", "workspace.copy"}:
            expected = hint.split(" → ")
            operands = [token for token in tokens[1:] if not token.startswith("-")]
            allowed_flags = {"-f", "--force", "-n", "--no-clobber", "-p", "--preserve", "--"}
            return len(expected) == 2 and len(operands) == 2 \
                and verb == ("cp" if operation == "workspace.copy" else "mv") \
                and all(cls._same_path(wanted, actual) for wanted, actual in zip(expected, operands)) \
                and all(token in allowed_flags for token in tokens[1:] if token.startswith("-"))
        if operation in {"workspace.run", "process.start"}:
            if verb in {
                "bash", "sh", "zsh", "fish", "dash", "ksh", "ash", "csh", "tcsh", "cmd",
                "powershell", "pwsh", "env", "sudo", "doas", "eval", "exec", "command", "builtin",
                "xargs", "nohup", "setsid", "nice", "time", "busybox", "npx",
            }:
                return False
            if verb in {"npm", "pnpm", "yarn"} and rest[:1] in (["exec"], ["dlx"]):
                return False
            if verb in {"python", "python3", "node", "deno", "bun", "ruby", "perl", "php"} \
                    and any(token in {"-c", "-e", "--eval", "eval", "-r"} for token in rest):
                return False
            hint_words = {
                word for word in re.findall(r"[a-z0-9_.+/-]+", hint.casefold())
                if word not in {"the", "a", "an", "command", "process", "task", "please", "now", "it"}
            }
            if operation == "process.start":
                if hint_words.intersection({"server", "app", "service"}):
                    return verb in {"uvicorn", "gunicorn", "flask", "django-admin", "rails"} or (
                        verb in {"npm", "pnpm", "yarn"} and any(
                            token in {"start", "dev", "serve"} for token in rest[:3]
                        )
                    ) or (
                        verb in {"python", "python3"}
                        and ("http.server" in rest[:3] or "runserver" in rest[:4])
                    ) or (
                        verb == "node" and bool(rest)
                        and "server" in os.path.basename(rest[0])
                    ) or (
                        verb == "go" and rest[:1] == ["run"]
                        and any("server" in token for token in rest[1:3])
                    ) or (verb == "docker" and rest[:2] == ["compose", "up"])
                if "worker" in hint_words and (
                    (verb == "celery" and "worker" in rest)
                    or (verb in {"python", "python3", "node"} and rest
                        and "worker" in os.path.basename(rest[0]))
                ):
                    return True
                if verb in {"git", "ssh", "scp", "sftp", "rsync", "curl", "wget"}:
                    return False
                primary = {verb}
                if verb in {"python", "python3", "node", "deno", "bun", "ruby", "perl", "php"}:
                    script = os.path.basename(rest[0] if rest else "")
                    primary.update({script, os.path.splitext(script)[0]})
                elif verb in {"npm", "pnpm", "yarn"} and rest[:1] == ["run"] and len(rest) >= 2:
                    primary.update(re.findall(r"[a-z0-9_.+/-]+", rest[1]))
                return bool(hint_words and hint_words.intersection(primary))
            outward = {"git", "curl", "wget", "ssh", "scp", "sftp", "rsync"}
            if verb in outward and verb not in hint_words:
                return False
            primary = {verb}
            if verb in {"python", "python3", "node", "deno", "bun", "ruby", "perl", "php"} and rest:
                script = os.path.basename(rest[0])
                primary.update({script, os.path.splitext(script)[0]})
            elif verb in {"npm", "pnpm", "yarn"} and rest[:1] == ["run"] and len(rest) >= 2:
                primary.add(rest[1])
            if hint_words.intersection({"migration", "migrate"}) and (
                any("migrat" in word for word in primary)
                or (verb == "alembic" and rest[:1] in (["upgrade"], ["downgrade"]))
                or (verb in {"python", "python3"} and "migrate" in rest[:5])
                or (verb == "rails" and any("migrate" in token for token in rest[:3]))
            ):
                return True
            if hint_words and hint_words.intersection(primary):
                return True
        return False

    @classmethod
    def _batch_edit_code(cls, code: str, target: str = "") -> bool:
        """Allow code-as-action only as a confined helper program, never as arbitrary Python."""
        try:
            tree = ast.parse(str(code or ""), mode="exec")
        except (SyntaxError, ValueError, TypeError):
            return False
        allowed_calls = {
            "read_file", "write_file", "append_file", "str_replace", "list_files", "print",
            "len", "range", "enumerate", "zip", "min", "max", "sum", "sorted", "str", "int", "bool",
        }
        write_calls = {"write_file", "append_file", "str_replace"}
        protected_names = allowed_calls | {"run"}
        unsafe_names = {
            "open", "exec", "eval", "compile", "__import__", "breakpoint", "input", "globals", "locals",
            "getattr", "setattr", "delattr", "vars", "memoryview",
        }
        blocked_nodes = (ast.Import, ast.ImportFrom, ast.Attribute, ast.Global, ast.Nonlocal)
        wrote = False
        for node in ast.walk(tree):
            if isinstance(node, blocked_nodes):
                return False
            if isinstance(node, ast.Name):
                if node.id in {"_os", "_sys", "_sp"} or "__" in node.id or node.id in unsafe_names:
                    return False
                if node.id in protected_names and not isinstance(node.ctx, ast.Load):
                    return False
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and node.name in protected_names:
                return False
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:
                return False
            if node.func.id in write_calls:
                wrote = True
                if target:
                    if not node.args or not isinstance(node.args[0], ast.Constant) \
                            or not isinstance(node.args[0].value, str) \
                            or not cls._same_path(target, node.args[0].value):
                        return False
        return wrote

    @classmethod
    def _matches_grant(cls, grant, name: str, args: Mapping) -> bool:
        tools = cls._field(grant, "tools", ()) or ()
        if isinstance(tools, str):
            tools = (tools,)
        if name not in tuple(str(tool) for tool in tools):
            return False
        mode = str(cls._field(grant, "mode", "scoped") or "scoped")
        expected = cls._field(grant, "exact_args", ()) or ()
        if isinstance(expected, Mapping):
            expected_items = tuple(expected.items())
        else:
            expected_items = tuple(expected)
        if mode == "exact":
            for key, expected_value in expected_items:
                if key not in args:
                    return False
                actual_value = args.get(key)
                if key == "path" and isinstance(expected_value, str) and isinstance(actual_value, str):
                    if not cls._same_path(expected_value, actual_value):
                        return False
                elif cls._freeze(actual_value) != cls._freeze(expected_value):
                    return False
            return True

        operation = str(cls._field(grant, "operation", "") or "")
        if operation == "workspace.verify" and name == "run_command":
            return cls._verification_command(str(args.get("command") or ""))
        if operation == "workspace.implement" and name == "run_command":
            return cls._implementation_command(str(args.get("command") or ""))
        if operation == "workspace.batch_edit" and name == "execute_code":
            return cls._batch_edit_code(str(args.get("code") or ""))
        if operation == "process.stop" and name in {"proc_kill", "terminal_close"}:
            target = str(cls._field(grant, "target", "") or "").casefold()
            handle = str(args.get("handle") or "").casefold()
            return bool(target and handle and target in re.findall(r"[a-z0-9_.-]+", handle)) \
                or bool(target and handle and target in handle.split("-"))
        if operation in {
            "vcs.commit", "vcs.push", "dependency.install", "package.publish", "workspace.deploy",
            "vcs.stage", "vcs.revert", "workspace.delete", "workspace.rename", "workspace.copy",
            "workspace.run", "process.start",
        } and name in {"run_command", "proc_start", "terminal_open"}:
            return cls._operation_command(
                operation, str(args.get("command") or ""), str(cls._field(grant, "target", "") or ""),
            )
        # Shell-like tools never inherit an unknown/forgotten operation by tool name alone. Every admitted
        # execution family must have an argument predicate above.
        if name in {"run_command", "proc_start", "terminal_open", "execute_code"}:
            return False
        target = str(cls._field(grant, "target", "") or "")
        target_arg = str(cls._field(grant, "target_arg", "") or "")
        if operation.startswith("task.requirement."):
            return bool(target and target_arg in args and cls._same_text(target, str(args.get(target_arg) or "")))
        if operation == "workspace.edit_prefix":
            return bool(target_arg in args and cls._path_in_prefix(target, str(args.get(target_arg) or "")))
        if operation == "workspace.edit_glob":
            return bool(target_arg in args and cls._path_matches_glob(target, str(args.get(target_arg) or "")))
        if operation == "workspace.edit_scoped_glob":
            prefix, separator, pattern = target.partition(" | ")
            actual = str(args.get(target_arg) or "")
            return bool(separator and target_arg in args and cls._path_in_prefix(prefix, actual)
                        and cls._path_matches_glob(pattern, actual))
        if not target:
            return True
        if target_arg:
            return target_arg in args and cls._same_path(
                target, str(args.get(target_arg) or ""),
                basename_anywhere=(operation == "workspace.navigate"),
            )
        return True

    @staticmethod
    def _grant_summary(grant) -> str:
        field = TurnAuthorityHook._field
        operation = str(field(grant, "operation", "effect") or "effect")
        target = str(field(grant, "target", "") or "")
        return f"{operation}:{target}" if target else operation

    @staticmethod
    def _matches_action(action: Mapping, name: str, args: Mapping) -> bool:
        """Legacy helper retained for third-party callers; delegates to an exact grant."""
        if str(action.get("tool") or "") != name:
            return False
        expected = action.get("args") or {}
        if not isinstance(expected, Mapping):
            return False
        for key, expected_value in expected.items():
            if key not in args:
                return False
            actual_value = args.get(key)
            if key == "path" and isinstance(expected_value, str) and isinstance(actual_value, str):
                try:
                    expected_path = os.path.normcase(os.path.realpath(os.path.expanduser(expected_value)))
                    actual_path = os.path.normcase(os.path.realpath(os.path.expanduser(actual_value)))
                    if expected_path != actual_path:
                        return False
                    continue
                except (OSError, ValueError):
                    pass
            if actual_value != expected_value:
                return False
        return True

    def authorize_tool(self, name, args):
        if name in self._RECOVERY_TOOLS:
            return ALLOW
        contract = self._contract()
        authority = self._authority(contract)
        try:
            effect = coerce_intent_effect(self.effect_resolver(name, args or {}))
        except Exception:  # resolver failure is semantically unknown and therefore effectful for this gate
            effect = ToolIntentEffect.UNKNOWN
        if effect in self._NON_EFFECTFUL:
            return ALLOW
        # NAVIGATION tier: a reversible workspace switch is authorized by navigation INTENT — any
        # workspace.navigate grant this turn — regardless of the grant's target OR the turn's authority level.
        # The grant comes from an explicit "go to X" directive, an accepted switch proposal, OR the user
        # answering the agent's own in-turn question (stamped in cli._workspace_ask_user): a turn contract
        # frozen as `uncertain` from an ambiguous opening request must not block a switch the user just
        # disambiguated. A turn with NO such grant still falls through and asks (an injected switch buried in a
        # non-actionable turn stays blocked); destructive effects below still require an AUTHORIZED, matching grant.
        if name in self._NAVIGATION_TOOLS and any(
                str(self._field(grant, "operation", "") or "") == "workspace.navigate"
                for grant in self._effect_grants(contract)):
            return ALLOW
        grants = self._effect_grants(contract) if authority in self._AUTHORIZED else ()
        if grants and any(self._matches_grant(grant, name, args or {}) for grant in grants):
            return ALLOW
        if authority in self._AUTHORIZED:
            expected = ", ".join(self._grant_summary(grant) for grant in grants) or "no typed effect grant"
            return ToolDecision(
                False,
                "turn_authority_scope_mismatch("
                f"tool={name!r}, expected={expected!r}): the user's confirmation authorizes only the "
                "recorded operation and target; observation and ask_user remain available",
                counts_as_stuck=False,
            )
        reason = (
            "turn_authority_missing("
            f"tool={name!r}, effect={effect.value}, authority={authority}): "
            "the current request does not authorize task-state or external effects; "
            "observation and ask_user remain available, and an explicit instruction or continuation "
            "approval is required before this call can run"
        )
        return ToolDecision(False, reason, counts_as_stuck=False)


class OracleHook(Hooks):
    """Verification gate: when the model declares done, run an oracle (tests/lint).
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
        except Exception as e:  # noqa: BLE001 — a verify ERROR must FORCE another turn, never silently pass the done-gate
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


_SELF_CHECK = (
    "STOP — definition-of-done check (required). Before you finish, verify your work against "
    "the task's REAL acceptance criteria:\n"
    "1) List EVERY concrete requirement: start from your STANDING REQUIREMENTS contract if you have one "
    "(each open '[ ]' item is binding), and ALSO re-read the task for anything not yet recorded — the exact "
    "output file path(s), required fields/values/format, each distinct sub-task, and any 'do not change X'.\n"
    "2) For EACH requirement, CONFIRM it against the ACTUAL end-state right now — run a command or read "
    "the real file (do NOT trust your memory, a note, or a schema-shape check): the required output "
    "exists at the EXACT path, its contents/values are correct, every sub-task is done, and you changed "
    "nothing you were told to leave alone. Call requirement_done(...) on each contract item you confirm.\n"
    "3) If anything is unmet or unverified, fix it and re-check. When a value must match something that "
    "already exists (a file, a git object, expected output), COPY it exactly — do not retype it.\n"
    "Finish only when ALL requirements are confirmed against the real end-state. If everything already "
    "checks out, just say so and finish — do not make changes for their own sake."
)


class SelfCheckHook(Hooks):
    """GROUNDED definition-of-done gate for AUTONOMOUS runs (no human to catch a premature 'done'). When the
    model declares done, force a verification round: re-derive the task's real acceptance criteria and
    CONFIRM each against the actual end-state by RUNNING tools (not asserting). Crucially it accepts 'done'
    only once the model has actually done verification WORK (a tool step) since the gate fired — a bare
    re-assertion of 'done' re-fires the gate. Bounded to `max_fires` rounds (env AGENT_SELFCHECK_MAX) so it
    can never loop. Moat-safe: appends a message (the proven feedback channel) + observes tool activity; the
    agent does the real work. The no-oracle cousin of OracleHook — the agent self-sources its acceptance
    check instead of declaring done blind. (Targets the measured premature-stop losses: produced-no-output,
    incomplete sweeps, symptom-not-root fixes — make it verify before it is allowed to finish.)"""

    def __init__(self, max_fires: int = 3):
        import os
        try:
            self._max = max(1, int(os.environ.get("AGENT_SELFCHECK_MAX") or max_fires))
        except (TypeError, ValueError):
            self._max = max(1, max_fires)   # a non-numeric env value must not crash hook construction
        self._fires = 0
        self._acted = False   # did the model run a tool since the gate last fired?

    def reset_for_turn(self):
        self._fires = 0
        self._acted = False

    def after_step(self, step: int, usage: dict, stop_reason: str):
        if stop_reason == "tool_use":   # the model actually ran verification/fix tools this round
            self._acted = True
        return None

    def should_continue_after_stop(self, stop_reason):
        if stop_reason != "end_turn":
            return None
        if self._fires > 0 and self._acted:
            return None                  # verified-by-doing after a nudge → honest done, accept
        if self._fires >= self._max:
            return None                  # bounded → never loop
        self._fires += 1
        self._acted = False
        return {"continue": True, "feedback": _SELF_CHECK}


class PermissionHook(Hooks):
    """Gate tool execution. `policy(name, args) -> ToolDecision`.

    When a policy returns `ask`, resolve it interactively via `on_ask(name, args, reason)
    -> 'yes'|'no'|'always'` (the host supplies a TTY prompt). Non-interactive hosts
    (on_ask=None) deny an `ask` — safe by default.

    'always' memorizes a session approval — but keyed by the CALL, not the bare tool name
    (rule patterns). Approving one shell command must NOT bless every shell command:
    run_command/execute_code are remembered by their exact command/code; other tools (already
    gated by policy) are remembered by name. `auto_approve` pre-seeds fnmatch rules matched
    against the command (e.g. ["git status*", "ls *"]) so safe read-only commands never prompt."""

    _CMD_TOOLS = ("run_command", "execute_code", "proc_start", "terminal_open", "terminal_send")
    # Tool-suite verbs whose SUBCOMMAND is part of the operation identity: approving `git commit` should
    # stick for `git commit -m ...` but NOT for `git push`. A bare verb (pytest, ls, make) sticks whole.
    _SUBCMD_TOOLS = frozenset({
        "git", "npm", "yarn", "pnpm", "cargo", "pip", "pip3", "poetry", "uv", "docker", "kubectl", "go",
        "gh", "conda", "mvn", "gradle", "dotnet", "deno", "bun", "bundle", "gem", "brew", "apt", "apt-get",
        "systemctl", "terraform", "make",
    })

    def __init__(self, policy, on_ask=None, auto_approve=None):
        self.policy = policy
        self.on_ask = on_ask
        self._approved: set[str] = set()        # exact approval keys (call patterns, not bare tool names)
        self._rules: list[str] = list(auto_approve or [])   # pre-seeded fnmatch globs over the command

    @classmethod
    def _key(cls, name: str, args: dict) -> str:
        # command-SPECIFIC for the dangerous tools — approving `npm test` must not auto-allow `rm -rf`.
        if name in cls._CMD_TOOLS:
            return f"{name}:{(args.get('command') or args.get('code') or args.get('input') or '').strip()}"
        return name                             # name-level for the rest (policy already gates them)

    @classmethod
    def _command_prefix(cls, name: str, cmd: str) -> str:
        """The sticky-approval PREFIX for a command: its verb, plus the subcommand for tool-suite verbs
        (git commit, npm test). A later variant of the SAME operation matches; a different subcommand
        (git push) or a destructive form (guarded separately) does not. Empty for a non-command tool."""
        if name not in cls._CMD_TOOLS or not cmd.strip():
            return ""
        try:
            toks = shlex.split(cmd)
        except ValueError:
            toks = cmd.split()
        if not toks:
            return ""
        verb = os.path.basename(toks[0])
        if verb in cls._SUBCMD_TOOLS:
            sub = next((t for t in toks[1:] if not t.startswith("-")), "")
            return f"{verb} {sub}".strip()
        return verb

    def _pre_allowed(self, name: str, args: dict, key: str) -> bool:
        if key in self._approved:
            return True
        cmd = (args.get("command") or args.get("code") or args.get("input") or "").strip()
        # Sticky per-PREFIX 'always': `npm test` sticks for `npm test --coverage`, never for a destructive
        # form (rm -rf, git reset/push, …) which always falls back to a fresh confirm.
        prefix = self._command_prefix(name, cmd)
        if prefix and f"pfx:{name}:{prefix}" in self._approved and not _is_destructive_command(name, cmd):
            return True
        if cmd and self._rules:
            import fnmatch
            if any(fnmatch.fnmatch(cmd, rule) for rule in self._rules):
                # A broad glob must NOT silently green-light a destructive command — fall through to ask.
                return not _is_destructive_command(name, cmd)
        return False

    def authorize_tool(self, name, args):
        d = self.policy(name, args)
        if not d.ask:
            return d
        key = self._key(name, args)
        if self._pre_allowed(name, args, key):
            return ALLOW
        if self.on_ask is None:
            return ToolDecision(False, DENIAL_NO_PROMPT)
        verdict = (self.on_ask(name, args, d.reason) or "no").lower()
        if verdict == "always":
            cmd = (args.get("command") or args.get("code") or args.get("input") or "").strip()
            prefix = self._command_prefix(name, cmd)
            # remember the OPERATION prefix for a command (so the same op won't re-prompt), else the tool name
            self._approved.add(f"pfx:{name}:{prefix}" if prefix else key)
            return ALLOW
        return ALLOW if verdict == "yes" else ToolDecision(False, DENIAL_USER)


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
            return {"block": True, "reason": "token budget exhausted"}
        return None

    def record_step_usage(self, usage):
        self.spent += int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        return {"stop_turn": True} if self.spent >= self.max else None

    def remaining_token_budget(self):
        return max(0, self.max - self.spent)


class ReconciliationHook(Hooks):
    """Permit observation, then require an explicit resolution before any further effect."""

    _OBSERVATION_TOOLS = frozenset({
        "read_file", "list_files", "grep", "glob", "code_review",
        "proc_poll", "proc_tail", "proc_wait", "terminal_read", "terminal_wait", "ask_user",
    })

    def __init__(self, state_provider):
        self.state_provider = state_provider
        self._key: tuple[str, tuple[str, ...]] = ("", ())
        self._observed: set[str] = set()

    def _required(self) -> tuple[str, tuple[str, ...]]:
        try:
            state = self.state_provider()
            marker = str(getattr(state, "reconciliation_required", "") or "")
            targets = tuple(str(target) for target in
                            (getattr(state, "reconciliation_targets", ()) or ("workspace:*",)))
            return marker, targets
        except Exception as exc:  # noqa: BLE001 - unavailable gate state must fail closed
            return f"reconciliation state unavailable ({type(exc).__name__})", ("workspace:*",)

    def _sync(self) -> tuple[str, tuple[str, ...]]:
        key = self._required()
        if key != self._key:
            self._key = key
            self._observed = set()
        return key

    def reset_for_turn(self):
        self._key = self._required()
        self._observed = set()

    def authorize_tool(self, name, args):
        marker, targets = self._sync()
        if not marker:
            return ALLOW
        if name in self._OBSERVATION_TOOLS:
            return ALLOW
        if name == "reconcile_execution":
            from .execution import reconciliation_covered
            if reconciliation_covered(targets, self._observed):
                return ALLOW
            missing = ", ".join(target for target in targets
                                if not reconciliation_covered((target,), self._observed))
            hint = ("; for workspace:* use code_review(include_ignored=true)"
                    if any(target == "workspace:*" for target in targets) else "")
            return ToolDecision(False, "reconciliation still needs matching live observation for: "
                                + missing + hint)
        return ToolDecision(
            False,
            "an earlier operation is indeterminate; use read-only observation tools, then "
            "reconcile_execution before any further side effect",
        )

    def transform_tool_result(self, name, args, output):
        marker, _targets = self._sync()
        if marker and name in self._OBSERVATION_TOOLS:
            from .execution import (ToolStatus, coerce_tool_status,
                                    observation_targets)
            if coerce_tool_status(getattr(output, "status", None), legacy_text=str(output)) \
                    is ToolStatus.SUCCEEDED:
                if name != "ask_user" or "no interactive user is available" not in str(output).lower():
                    self._observed.update(observation_targets(name, args, output))
        return None

    def should_continue_after_stop(self, stop_reason):
        marker, targets = self._sync()
        if marker and stop_reason == "end_turn":
            return {
                "continue": True,
                "exclusive": True,
                "feedback": "Execution is still indeterminate. Re-observe every affected target with "
                            "matching read-only tools, then call reconcile_execution before finishing: "
                            + ", ".join(targets)
                            + (". For workspace:* use code_review(include_ignored=true)."
                               if "workspace:*" in targets else ""),
            }
        return None

class GuardrailHook(Hooks):
    """Cross-step loop guard: block a tool call that repeats an identical failing call,
    or an idempotent call that keeps making no progress. State is per-turn (cleared by
    `reset_for_turn`), so counters never bleed across user tasks."""

    def __init__(self, config=None):
        self.guard = ToolCallGuardrail(config)

    def reset_for_turn(self):
        self.guard.reset_for_turn()

    def authorize_tool(self, name, args):
        d = self.guard.before_call(name, args)
        if not d.block:
            return ALLOW
        # Only a HARD spin counts toward STUCK: a repeated FAILING call, or no-edit-progress (failing edits).
        # A deduped idempotent/result no-progress read is harmless — block (skip) it but DON'T kill the turn,
        # so a long exploration that re-reads a file isn't falsely flagged as stuck.
        hard = d.code in ("repeated_exact_failure", "no_edit_progress")
        return ToolDecision(False, d.message, counts_as_stuck=hard)

    def transform_tool_result(self, name, args, output):
        # NEVER feed a guardrail/policy BLOCK back into the counters: a blocked call never ran, so counting
        # its synthetic "Error: blocked by policy:" result as a real failure would advance the failing /
        # no-edit-progress axes and falsely escalate a harmless soft-block into a hard 'stuck' turn-kill.
        if isinstance(output, str) and output.startswith("Error: blocked by policy:"):
            return None
        self.guard.after_call(name, args, output)
        return None
