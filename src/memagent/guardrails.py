"""Per-turn tool-call loop guardrail — the slice's anti-loop defense.

Ported from /tmp/hermes-agent/agent/tool_guardrails.py (the ToolCallSignature /
ToolCallGuardrailController design), adapted to memagent's no-transcript invariant.

WHY THIS IS MOAT-CRITICAL
-------------------------
The active memory slice ERASES the model's memory of prior identical failed calls:
each turn is reconstructed fresh, so the model cannot "remember" that it already ran
the exact same failing command three steps ago. The slice's REPEATED/FAILING tier
mitigates this softly (it tells the model in prose), but a model can still ignore it.
This controller is the HARD floor: it counts, per turn, every (tool, canonical-args)
signature and BLOCKS a call once it has failed N times unchanged, or once a read-only
call has returned the same result N times with no progress. Because the model has no
transcript memory of the failure, the blocked-call message is ACTION-ORIENTED: it tells
the model what to do INSTEAD (the failure context the slice can't carry for it).

NO-TRANSCRIPT INVARIANT
-----------------------
State lives ONLY in this controller for the duration of ONE turn (reset_for_turn at the
top of run_turn). It feeds NO durable store and assumes NO growing message history. The
block decision becomes a synthetic tool RESULT (which the slice folds into its tiers like
any other result) — never a message appended to a transcript.

CONVENTIONS BORROWED FROM MEMAGENT (not from Hermes)
----------------------------------------------------
- "failing" is memagent's existing convention (loop.py / slice.record_action /
  mining.py): out.startswith("Error") or out.startswith("Exit code"). We do NOT port
  Hermes's JSON exit-code classifier (memagent has no safe_json_loads and a different
  result shape). Callers may pass `failed=` explicitly; otherwise we classify here.
- The idempotent / mutating tool sets are memagent's actual builtins.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

# memagent's read-only (idempotent) builtins. Repeating one of these with the SAME args
# and getting the SAME result is "no progress" — a soft loop the slice can't see through.
IDEMPOTENT_TOOL_NAMES = frozenset({"read_file", "list_files", "recall_history"})

# memagent's mutating builtins (+ topic/skill routing). A mutating tool is never treated
# as idempotent (its repeated identical RESULT is not a no-progress signal — only its
# repeated identical FAILURE is).
MUTATING_TOOL_NAMES = frozenset({
    "edit_file", "append_to_file", "str_replace",
    "run_command", "execute_code",
    "new_topic", "switch_topic", "skill",
})

# memagent's failing convention — kept in one place so guardrail counting agrees with
# slice.record_action / loop.run_tool_batch / mining (all use this exact prefix test).
_FAIL_PREFIXES = ("Error", "Exit code")


def is_failing_output(output: str | None) -> bool:
    """memagent's standard failing-result test (loop.py:68, slice.py:228, mining.py)."""
    return bool(output) and (output.startswith("Error") or output.startswith("Exit code"))


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Per-turn loop-detection thresholds.

    Defaults: HARD-BLOCK is ON (unlike Hermes, whose interactive CLI defaults to warn-only).
    memagent is uniquely loop-prone because the slice erases failure memory, so the block
    floor must be active by default. Thresholds are intentionally low — by the time the same
    exact call has failed `exact_failure_block_after` times, the model is in a loop the slice
    cannot break on its own.
    """

    exact_failure_block_after: int = 3      # same (tool,args) FAILED this many times → block
    no_progress_block_after: int = 3        # idempotent call returned SAME result this many times → block
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name + canonical args (no raw values)."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        return cls(tool_name=tool_name, args_hash=_sha256(canonical_tool_args(args or {})))


@dataclass(frozen=True)
class GuardrailDecision:
    """What the controller decided for one call. `block` is the only actionable field for
    the hook: when True, authorize_tool denies and surfaces `message` to the model."""

    block: bool = False
    code: str = "allow"          # allow | repeated_exact_failure | idempotent_no_progress
    message: str = ""
    tool_name: str = ""
    count: int = 0


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Sorted compact JSON of the args, with the 'note' findings-arg STRIPPED.

    The 'note' arg is the model's per-turn distilled conclusion (tools.with_note); it rides
    on every call and changes turn-to-turn, so including it would make every signature unique
    and HIDE loops. Stripping it is the memagent-specific fix — the canonical identity is the
    real action (path/command/code), not the commentary attached to it.
    """
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    filtered = {k: v for k, v in args.items() if k != "note"}
    return json.dumps(filtered, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class ToolCallGuardrail:
    """Per-turn controller. `reset_for_turn()` at the top of each turn; `before_call()` in
    authorize_tool (returns a block decision); `after_call()` in transform_tool_result
    (counts the result). Side-effect free except for its own per-turn counters."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        """Drop all per-turn counters. MUST be called at the start of every turn (run_turn),
        so a fresh user task never inherits the prior task's loop counts."""
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        # signature -> (result_hash, repeat_count) for idempotent no-progress detection
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> GuardrailDecision:
        """Decide whether to BLOCK this call, based on counts from prior calls THIS turn.
        Pure read of the counters — does not mutate them (after_call does the counting)."""
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))

        exact_count = self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            return GuardrailDecision(
                block=True,
                code="repeated_exact_failure",
                message=(
                    f"Loop blocked: '{tool_name}' has already failed {exact_count} times this "
                    f"turn with these EXACT arguments. You have no record of those failures "
                    f"(your context is rebuilt each turn) — they happened. Do NOT retry it "
                    f"unchanged. Read CURRENT ERROR and OPEN FILES, then either fix the root "
                    f"cause with a DIFFERENT call (different args/path/command, or a different "
                    f"tool), or, if the work is already complete, write the final summary and "
                    f"make NO tool call."
                ),
                tool_name=tool_name,
                count=exact_count,
            )

        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None and record[1] >= self.config.no_progress_block_after:
                return GuardrailDecision(
                    block=True,
                    code="idempotent_no_progress",
                    message=(
                        f"Loop blocked: this read-only '{tool_name}' call has returned the SAME "
                        f"result {record[1]} times this turn. Repeating it cannot reveal anything "
                        f"new. Use the result already shown in OPEN FILES / RECENT, or change the "
                        f"query/path. If you have what you need, act on it or write the final "
                        f"summary."
                    ),
                    tool_name=tool_name,
                    count=record[1],
                )

        return GuardrailDecision(tool_name=tool_name)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: bool | None = None,
    ) -> None:
        """Record one observed result into the per-turn counters. Called AFTER the tool ran.
        No return value — counting only; blocking is before_call's job next time the signature
        recurs."""
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed = is_failing_output(result)

        if failed:
            self._exact_failure_counts[signature] = self._exact_failure_counts.get(signature, 0) + 1
            self._no_progress.pop(signature, None)
            return

        # success clears the exact-failure streak for this signature
        self._exact_failure_counts.pop(signature, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return

        result_hash = _sha256(result or "")
        previous = self._no_progress.get(signature)
        repeat = previous[1] + 1 if (previous is not None and previous[0] == result_hash) else 1
        self._no_progress[signature] = (result_hash, repeat)


def guardrail_blocked_result(decision: GuardrailDecision) -> str:
    """The synthetic tool-result string surfaced when a call is blocked. Starts with 'Error'
    so the slice's existing failing-detection (slice.record_action) tallies it as a failure —
    keeping the block visible in the REPEATED/FAILING tier too."""
    return f"Error: {decision.message}"


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ToolCallGuardrailConfig",
    "ToolCallSignature",
    "GuardrailDecision",
    "ToolCallGuardrail",
    "canonical_tool_args",
    "is_failing_output",
    "guardrail_blocked_result",
    "IDEMPOTENT_TOOL_NAMES",
    "MUTATING_TOOL_NAMES",
]
