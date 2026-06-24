"""Hooks: the policy seam (from Kimi). The loop calls these; the host supplies them.

This is how policy stays OUT of the moat: the Oracle, permission gate, and token
budget are all hooks, not hardcoded loop logic.

Hook return conventions (all optional, return None to no-op):
  before_step(step)                     -> {"block": bool, "reason": str} | None
  record_step_usage(usage)              -> {"stop_turn": bool} | None
  after_step(step, usage, stop_reason)  -> {"stop_turn": bool} | None
  should_continue_after_stop(stop)      -> {"continue": bool} | None
  authorize_tool(name, args)            -> ToolDecision
"""
from __future__ import annotations

from dataclasses import dataclass

from .guardrails import ToolCallGuardrail
from .guidance import DENIAL_NO_PROMPT, DENIAL_USER


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
        stop = any((h.record_step_usage(usage) or {}).get("stop_turn") for h in self.hooks)
        return {"stop_turn": True} if stop else None

    def after_step(self, step, usage, stop_reason):
        stop = any((h.after_step(step, usage, stop_reason) or {}).get("stop_turn") for h in self.hooks)
        return {"stop_turn": True} if stop else None

    def should_continue_after_stop(self, stop_reason):
        for h in self.hooks:
            r = h.should_continue_after_stop(stop_reason)
            if r and r.get("continue"):
                return r
        return None

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
        ok, output = self.oracle.verify()
        if ok:
            return None
        self.on_feedback(output)   # also record into the slice (for the NEXT turn's seed / durable cache)
        # CRITICAL: the failure detail must ride the MESSAGE channel — under the accumulate loop the seed
        # is built once and never re-rendered mid-turn, so a slice mutation (last_error) is invisible to
        # THIS turn's retry. Put `output` in `feedback` so the loop appends it as the model's next input.
        return {"continue": True, "feedback": f"Verification failed — fix this, then finish:\n{output}"}


_SELF_CHECK = (
    "STOP — definition-of-done check (required, ONE time). Before you finish, verify your work against "
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
    """Structural definition-of-done gate for AUTONOMOUS runs (no human to catch a premature 'done').
    On the FIRST time the model declares done, force exactly ONE verification pass: re-derive the task's
    requirements and confirm each against the REAL end-state. Fires once per turn then accepts the next
    'done', so it can never loop. Moat-safe: it only appends a message (the proven feedback channel) — the
    agent does the real tool calls to verify. The no-oracle cousin of OracleHook: with no external
    AGENT_VERIFY_CMD, the agent self-sources its acceptance check instead of declaring done blind."""

    def __init__(self):
        self._fired = False

    def reset_for_turn(self):
        self._fired = False

    def should_continue_after_stop(self, stop_reason):
        if stop_reason != "end_turn" or self._fired:
            return None
        self._fired = True
        return {"continue": True, "feedback": _SELF_CHECK}


class PermissionHook(Hooks):
    """Gate tool execution. `policy(name, args) -> ToolDecision`.

    When a policy returns `ask`, resolve it interactively via `on_ask(name, args, reason)
    -> 'yes'|'no'|'always'` (the host supplies a TTY prompt). Non-interactive hosts
    (on_ask=None) deny an `ask` — safe by default.

    'always' memorizes a session approval — but keyed by the CALL, not the bare tool name
    (Kimi-style rule patterns). Approving one shell command must NOT bless every shell command:
    run_command/execute_code are remembered by their exact command/code; other tools (already
    gated by policy) are remembered by name. `auto_approve` pre-seeds fnmatch rules matched
    against the command (e.g. ["git status*", "ls *"]) so safe read-only commands never prompt."""

    _CMD_TOOLS = ("run_command", "execute_code")

    def __init__(self, policy, on_ask=None, auto_approve=None):
        self.policy = policy
        self.on_ask = on_ask
        self._approved: set[str] = set()        # exact approval keys (call patterns, not bare tool names)
        self._rules: list[str] = list(auto_approve or [])   # pre-seeded fnmatch globs over the command

    @classmethod
    def _key(cls, name: str, args: dict) -> str:
        # command-SPECIFIC for the dangerous tools — approving `npm test` must not auto-allow `rm -rf`.
        if name in cls._CMD_TOOLS:
            return f"{name}:{(args.get('command') or args.get('code') or '').strip()}"
        return name                             # name-level for the rest (policy already gates them)

    def _pre_allowed(self, name: str, args: dict, key: str) -> bool:
        if key in self._approved:
            return True
        cmd = (args.get("command") or args.get("code") or "").strip()
        if cmd and self._rules:
            import fnmatch
            return any(fnmatch.fnmatch(cmd, rule) for rule in self._rules)
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
        # this, the cap silently became a whole-SESSION budget across the REPL (Kimi-review #2). A true
        # session-wide cap, if ever wanted, should be a separate named hook — not this one.
        self.spent = 0

    def record_step_usage(self, usage):
        self.spent += int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        return {"stop_turn": True} if self.spent >= self.max else None


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
        self.guard.after_call(name, args, output)
        return None
