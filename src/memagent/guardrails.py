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

# Known NON-mutating (read/search) tools. The no-progress streak treats a tool as a potential MUTATOR
# unless it is in here — so unknown plugin/MCP tools AND mutating builtins missing from the static set
# above (world_set, terminal_*, proc_*, update_plan, …) still drive loop detection (pessimistic).
_NON_MUTATORS = IDEMPOTENT_TOOL_NAMES | frozenset({"grep", "glob", "ask_user"})

# memagent's failing convention — kept in one place so guardrail counting agrees with
# slice.record_action / loop.run_tool_batch / mining (all use this exact prefix test).
_FAIL_PREFIXES = ("Error", "Exit code")


def is_failing_output(output: str | None) -> bool:
    """memagent's standard failing-result test (loop.py:68, slice.py:228, mining.py)."""
    return bool(output) and (output.startswith("Error") or output.startswith("Exit code"))


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Per-turn (= per-episode) loop-detection thresholds.

    Defaults: HARD-BLOCK is ON (unlike Hermes, whose interactive CLI defaults to warn-only).
    memagent is uniquely loop-prone because the slice erases failure memory, so the block
    floor must be active by default. Thresholds are intentionally low — by the time the same
    exact call has failed `exact_failure_block_after` times, the model is in a loop the slice
    cannot break on its own.

    I3 — RESULT axis. The exact-(tool,args) axis above misses the live failure mode: the agent
    looped ~13× re-inspecting the same directory via DISTINCT command text (`ls X`, `ls -la X`,
    an `execute_code` listing) — every call a unique arg signature at count 1, so nothing ever
    blocked across a 411k-token spin (GR1/2/3). The RESULT axis is tool-AGNOSTIC: it keys on the
    OUTPUT hash across ALL tools (incl. run_command/execute_code), so semantically-redundant calls
    with different text collapse to one progress signature. A repeated identical RESULT — even from
    a 'mutating' tool — means the action is not changing observable state: a no-progress loop.
    """

    exact_failure_block_after: int = 3      # same (tool,args) FAILED this many times → block
    no_progress_block_after: int = 3        # idempotent call returned SAME result this many times → block
    # I3 RESULT axis — tool-agnostic, keyed on the OUTPUT not the args.
    result_repeat_block_after: int = 4      # SAME result_hash recurring this many times across ANY tools → soft block
    no_edit_mutations_before_warn: int = 6  # this many mutating attempts with NO successful edit → soft warn
    call_budget_warn_after: int = 18        # this many tool calls with NO successful change landing → soft stop (floor)
    trajectory_ring_cap: int = 20           # bounded per-episode ring of (op_kind, result_hash) progress signatures
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
    # allow | repeated_exact_failure | idempotent_no_progress | result_no_progress | no_edit_progress
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0


# I3 — COARSE op_kind. The trajectory ring stores a tool-AGNOSTIC operation class per step (not the
# raw tool name), so different tools doing the same kind of thing share a signature. Task-agnostic:
# only the builtin tool TAXONOMY, never command/argument parsing.
def op_kind(tool_name: str) -> str:
    """Map a tool name to a coarse operation class for the trajectory ring."""
    if tool_name in ("edit_file", "append_to_file", "str_replace"):
        return "edit"
    if tool_name in ("read_file",):
        return "read"
    if tool_name in ("list_files",):
        return "list"
    if tool_name in ("run_command", "execute_code"):
        return "exec"
    if tool_name in ("new_topic", "switch_topic", "skill", "recall_history"):
        return "meta"
    return "other"


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
        """Drop all per-turn (= per-EPISODE) counters. MUST be called at the start of every turn
        (run_turn), so a fresh user task never inherits the prior task's loop counts. Every field
        here is BOUNDED — the result-axis ring is capped at trajectory_ring_cap, never a transcript."""
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        # signature -> (result_hash, repeat_count) for idempotent no-progress detection
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        # I3 RESULT axis — a BOUNDED per-episode ring of progress signatures (op_kind, result_hash),
        # last `trajectory_ring_cap` steps. NOT the transcript: fixed-length, op-class + hash only (no
        # args, no output text). Drives result-repeat detection across ANY tool (incl. shell).
        self._trajectory: list[tuple[str, str]] = []
        # result_hash -> count, derived from the ring (also bounded by the ring's distinct entries).
        self._result_counts: dict[str, int] = {}
        # mutating-attempt streak with no successful edit (the "act or stop" no-progress floor for edits).
        self._mutations_since_edit: int = 0
        # total tool calls since the last successful change landed — the coarse per-turn budget floor.
        self._calls_since_edit: int = 0

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
                    f"turn with these EXACT arguments — they are in the transcript above. Do NOT "
                    f"retry it unchanged. Read CURRENT ERROR and OPEN FILES, then either fix the "
                    f"root cause with a DIFFERENT call (different args/path/command, or a different "
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
                        f"new. Use the result already shown in OPEN FILES or the transcript above, "
                        f"or change the query/path. If you have what you need, act on it or write "
                        f"the final summary."
                    ),
                    tool_name=tool_name,
                    count=record[1],
                )

        # I3 — no-edit-progress axis. The agent has made `no_edit_mutations_before_warn` consecutive
        # FAILING mutating attempts this episode with nothing landing — it keeps trying to change the
        # world and nothing sticks (e.g. a str_replace whose old_string never matches, an edit/run that
        # keeps erroring). A SUCCESSFUL mutation resets the streak (so productive non-edit shell work is
        # never penalized). Tool-agnostic over the mutating set; only fires on the next mutating attempt
        # so a read/answer is never blocked. Soft "act or stop": stop hammering, read OPEN FILES, change.
        if (self._mutations_since_edit >= self.config.no_edit_mutations_before_warn
                and tool_name in self.config.mutating_tools):
            return GuardrailDecision(
                block=True,
                code="no_edit_progress",
                message=(
                    f"Loop blocked: your last {self._mutations_since_edit} mutating attempts this turn "
                    f"all FAILED with nothing landing — you are spinning. Re-read OPEN FILES to base "
                    f"the next change on the ACTUAL current contents (a str_replace must match the file "
                    f"verbatim), then make ONE precise edit with DIFFERENT arguments. If the work is "
                    f"already complete, write the final summary and make NO tool call."
                ),
                tool_name=tool_name,
                count=self._mutations_since_edit,
            )

        # I3 RESULT axis (tool-AGNOSTIC) — the same OUTPUT has recurred ≥K times this episode across
        # ANY tools (incl. run_command/execute_code). The agent is re-observing state that isn't
        # changing — a no-progress loop the exact-(tool,args) axis cannot see (different command text,
        # same result). Soft "you've seen this output N times — act or stop". Pure read of the ring.
        top_hash, top_count = self._hottest_result()
        if top_count >= self.config.result_repeat_block_after:
            return GuardrailDecision(
                block=True,
                code="result_no_progress",
                message=(
                    f"Loop blocked: you have already seen this EXACT output {top_count} times this "
                    f"turn (across possibly different commands/tools) — the repeats are in the "
                    f"transcript above. Re-observing it cannot reveal anything new. Act on the "
                    f"result already in OPEN FILES with a DIFFERENT step, or — if the work is "
                    f"complete — write the final summary and make NO tool call."
                ),
                tool_name=tool_name,
                count=top_count,
            )

        # Coarse per-turn BUDGET floor (the plan's backstop): this many tool calls this episode with NO
        # successful change landing means the agent is exploring in circles — the slice rebuilds each turn
        # so it cannot see its own spin. A productive mutating call resets the budget, so real multi-step
        # work is never throttled; only a pure read/failed-mutation spree trips it. Soft "act or answer".
        if self._calls_since_edit >= self.config.call_budget_warn_after:
            return GuardrailDecision(
                block=True,
                code="call_budget",
                message=(
                    f"Loop blocked: {self._calls_since_edit} tool calls this turn with NO successful "
                    f"change landing — you are exploring in circles (the whole sequence is in the "
                    f"transcript above). Stop calling tools: act on what OPEN FILES already shows, "
                    f"or write your final summary/answer and make NO tool call."
                ),
                tool_name=tool_name,
                count=self._calls_since_edit,
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

        # I3 RESULT axis — record EVERY observed result into the bounded per-episode ring, regardless
        # of tool or success. Tool-agnostic (keyed on op_kind + result_hash), so a loop of distinct
        # commands returning the same output is visible. Bounded: the ring is capped, counts derived
        # from it. We hash even failing results — a tool that fails the SAME way via different calls is
        # still a no-progress loop.
        kind = op_kind(tool_name)
        result_hash = _sha256(result or "")
        self._push_trajectory(kind, result_hash)
        # Budget floor counts NON-PROGRESS calls only. Progress = a change that lands (the mutator branch
        # below) OR a successful call that returns NEW information (a result not already in the recent ring).
        # A distinct, successful read IS progress: analysis / review / debugging-by-reading legitimately
        # never edit, and must not be strangled at call_budget_warn_after. Only a re-read of the SAME output
        # or a FAILED call advances the floor — and genuine re-reads/repeats are already caught by the
        # result/idempotent axes. This makes the floor task-AGNOSTIC: it fires on flailing, not on reading.
        if (not failed) and self._result_counts.get(result_hash, 0) <= 1:
            self._calls_since_edit = 0          # new information landed → reset the floor
        else:
            self._calls_since_edit += 1

        # I3 no-edit axis — track mutating attempts that make NO observable progress. A mutating call
        # that SUCCEEDS (an edit that lands, or a clean run/script that produced a non-error result) is
        # progress → resets the streak; only a FAILING mutating attempt (a str_replace whose old_string
        # never matches, an edit/run that errors) advances it. This targets the "trying to change the
        # world and nothing sticks" loop WITHOUT penalizing productive non-edit shell work (running a
        # build/test that passes is progress, not spinning). Tool-agnostic over the mutating set.
        if tool_name not in _NON_MUTATORS:   # pessimistic: unknown/plugin tools count as mutators too
            if not failed:
                self._mutations_since_edit = 0
                self._calls_since_edit = 0  # a change landed → the budget floor resets
            else:
                self._mutations_since_edit += 1

        if failed:
            self._exact_failure_counts[signature] = self._exact_failure_counts.get(signature, 0) + 1
            self._no_progress.pop(signature, None)
            return

        # success clears the exact-failure streak for this signature
        self._exact_failure_counts.pop(signature, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return

        previous = self._no_progress.get(signature)
        repeat = previous[1] + 1 if (previous is not None and previous[0] == result_hash) else 1
        self._no_progress[signature] = (result_hash, repeat)

    def _push_trajectory(self, kind: str, result_hash: str) -> None:
        """Append one (op_kind, result_hash) progress signature to the BOUNDED per-episode ring and
        recompute the result-hash counts from it. The ring is capped at trajectory_ring_cap so it can
        never grow into a transcript; counts are derived from the live ring (so a result that scrolls
        out of the window stops counting — bounded memory, recent-window semantics)."""
        cap = self.config.trajectory_ring_cap
        self._trajectory.append((kind, result_hash))
        if len(self._trajectory) > cap:
            del self._trajectory[:-cap]
        counts: dict[str, int] = {}
        for _, h in self._trajectory:
            counts[h] = counts.get(h, 0) + 1
        self._result_counts = counts

    def _hottest_result(self) -> tuple[str, int]:
        """The most-repeated result_hash in the current ring and its count (('', 0) when empty)."""
        if not self._result_counts:
            return ("", 0)
        h = max(self._result_counts, key=self._result_counts.get)
        return (h, self._result_counts[h])


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
    "op_kind",
    "IDEMPOTENT_TOOL_NAMES",
    "MUTATING_TOOL_NAMES",
]
