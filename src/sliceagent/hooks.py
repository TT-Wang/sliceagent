"""Hooks: the policy seam. The loop calls these; the host supplies them.

This is how policy stays OUT of the moat: the Oracle, permission gate, and token
budget are all hooks, not hardcoded loop logic.

Hook return conventions (all optional, return None to no-op):
  before_step(step)                     -> {"block": bool, "reason": str} | None
  record_step_usage(usage)              -> {"stop_turn": bool} | None
  after_step(step, usage, stop_reason)  -> {"stop_turn": bool} | None
  should_continue_after_stop(stop)      -> {"continue"|"park": bool, "exclusive"?: bool} | None
  authorize_tool(name, args)            -> ToolDecision
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .guardrails import ToolCallGuardrail
from .guidance import DENIAL_NO_PROMPT, DENIAL_USER

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

    def _pre_allowed(self, name: str, args: dict, key: str) -> bool:
        if key in self._approved:
            return True
        cmd = (args.get("command") or args.get("code") or args.get("input") or "").strip()
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
            self._approved.add(key)             # remember THIS call pattern, not the whole tool
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
