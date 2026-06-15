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


@dataclass
class ToolDecision:
    allow: bool
    reason: str = ""
    ask: bool = False   # policy abstains to an interactive prompt (resolved by PermissionHook)


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
        self.on_feedback(output)
        return {"continue": True}


class PermissionHook(Hooks):
    """Gate tool execution. `policy(name, args) -> ToolDecision`.

    When a policy returns `ask`, resolve it interactively via `on_ask(name, args, reason)
    -> 'yes'|'no'|'always'` (the host supplies a TTY prompt). 'always' memorizes the tool
    for the session so it isn't re-asked (Kimi-style session-approval). Non-interactive
    hosts (on_ask=None) deny an `ask` — safe by default."""

    def __init__(self, policy, on_ask=None):
        self.policy = policy
        self.on_ask = on_ask
        self._approved: set[str] = set()  # session-approved tool names

    def authorize_tool(self, name, args):
        d = self.policy(name, args)
        if not d.ask:
            return d
        if name in self._approved:
            return ALLOW
        if self.on_ask is None:
            return ToolDecision(False, "permission required but no prompt available")
        verdict = (self.on_ask(name, args, d.reason) or "no").lower()
        if verdict == "always":
            self._approved.add(name)
            return ALLOW
        return ALLOW if verdict == "yes" else ToolDecision(False, "denied by user")


class BudgetHook(Hooks):
    """Stop the turn once cumulative tokens cross a ceiling."""

    def __init__(self, max_total_tokens: int):
        self.max = max_total_tokens
        self.spent = 0

    def record_step_usage(self, usage):
        self.spent += int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        return {"stop_turn": True} if self.spent >= self.max else None
