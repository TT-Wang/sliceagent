"""The agent loop — the moat. Stateless core over contracts; ONE model call per turn,
no accumulated history. Mirrors Kimi Code's run-turn/turn-step split.

The core depends ONLY on: build_slice (the reconstruction seam), an LLMClient, a
ToolHost, a dispatch_event callable, and hooks. It never imports implementations and
never touches slice internals (tool results flow back via the slice_sink on events).
"""
from __future__ import annotations

from dataclasses import dataclass

from .context_overflow import ContextOverflow
from .errors import with_retry
from .events import (
    AssistantText,
    Dispatcher,
    SliceBuilt,
    SliceTightened,
    StepBegin,
    StepEnd,
    ToolResult,
    ToolStarted,
    TurnEnd,
    TurnInterrupted,
)
from .guidance import BUDGET_EXHAUSTED, STUCK
from .hooks import Hooks
from .scheduler import run_scheduled


# Anti-spin floor: after this many guardrail BLOCKS in one turn the loop stops and hands control back
# to the user (TurnInterrupted "stuck") instead of letting a weak model keep generating variants
# against the guard. The proactive path is the ask_user tool; this is the harness backstop. Each block
# already represents a 3-4x repeat caught by the guardrail, so a few blocks = genuinely stuck.
STUCK_BLOCK_BUDGET = 3


@dataclass
class StepOutcome:
    usage: dict
    stop_reason: str
    blocked: int = 0


@dataclass
class TurnResult:
    stop_reason: str
    steps: int
    usage: dict


def _normalize_stop(resp) -> str:
    fr = (resp.finish_reason or "").lower()
    if fr in ("length", "max_tokens"):
        return "max_tokens"
    if fr in ("content_filter", "filtered"):
        return "filtered"
    return "tool_use" if resp.tool_calls else "end_turn"


def run_tool_batch(tool_calls, tools, dispatch: Dispatcher, hooks: Hooks) -> int:
    """Authorize, schedule (safe-parallel by resource access), and report results in provider order.
    Returns the number of calls BLOCKED by a hook (the anti-spin floor counts these per turn)."""
    tasks = []
    metas = []
    blocked = 0
    for tc in tool_calls:
        dispatch(ToolStarted(tc.name, tc.args))
        decision = hooks.authorize_tool(tc.name, tc.args)
        if not decision.allow:
            blocked += 1
            tasks.append(([], (lambda d=decision: f"Error: blocked by policy: {d.reason or 'denied'}")))
        else:
            tasks.append((tools.accesses(tc.name, tc.args), (lambda tc=tc: str(tools.run(tc.name, tc.args)))))
        metas.append((tc.name, tc.args))

    outputs = run_scheduled(tasks)
    for (name, args), out in zip(metas, outputs):
        transformed = hooks.transform_tool_result(name, args, out)  # mutating seam (plugins/redaction)
        if transformed is not None:
            out = transformed
        failing = out.startswith("Error") or out.startswith("Exit code")
        dispatch(ToolResult(name, args, out, failing))
    return blocked


def run_step(*, step_num: int, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks) -> StepOutcome:
    before = hooks.before_step(step_num)
    if before and before.get("block"):
        raise RuntimeError(before.get("reason") or f"step {step_num} blocked")

    def _prepare(msgs):
        prepared = hooks.prepare_messages(msgs)  # pre-LLM-call seam (inject context, prompt-cache-safe)
        return prepared if prepared is not None else msgs

    messages = _prepare(build_slice())
    dispatch(SliceBuilt(messages[-1]["content"], messages))
    dispatch(StepBegin(step_num))

    # ITEM 3: context-overflow rebuild loop. On overflow, ask the slice builder to
    # tighten its tier caps and rebuild — NEVER grow a transcript. Bounded so an
    # unfixable overflow re-raises at the tier floor instead of spinning forever.
    # step_num is unchanged across tightenings; messages stay [system, user] (length 2).
    overflow_tries = 0
    while True:
        try:
            resp = with_retry(
                lambda: llm.complete(messages, tools.schemas()),
                is_retryable=getattr(llm, "is_retryable", None),
                dispatch=dispatch,
            )
            break
        except ContextOverflow:
            tightened = getattr(build_slice, "rebuild_tighter", lambda: False)()
            if not tightened or overflow_tries >= 3:
                raise
            overflow_tries += 1
            dispatch(SliceTightened(level=overflow_tries))
            messages = _prepare(build_slice())
            dispatch(SliceBuilt(messages[-1]["content"], messages))

    usage = resp.usage or {}
    usage_result = hooks.record_step_usage(usage)  # recorded BEFORE tools, so aborts don't lose it
    stop_turn = bool(usage_result and usage_result.get("stop_turn"))

    stop_reason = _normalize_stop(resp)
    if resp.content:
        dispatch(AssistantText(resp.content))

    effective = "end_turn" if (stop_turn and stop_reason == "tool_use") else stop_reason
    blocked = 0
    if effective == "tool_use":
        blocked = run_tool_batch(resp.tool_calls, tools, dispatch, hooks)

    dispatch(StepEnd(step_num, usage, effective))

    after = hooks.after_step(step_num, usage, effective)
    if after and after.get("stop_turn") and effective == "tool_use":
        effective = "end_turn"
    return StepOutcome(usage=usage, stop_reason=effective, blocked=blocked)


def run_turn(*, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks | None = None,
             max_steps: int = 40, signal=None) -> TurnResult:
    hooks = hooks or Hooks()
    hooks.reset_for_turn()  # ITEM 1: clear per-turn guards ONCE per user task (not per step)
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    steps = 0
    total_blocked = 0
    stop_reason = "end_turn"

    while True:
        if signal is not None and signal.is_set():
            dispatch(TurnInterrupted("aborted"))
            return TurnResult("aborted", steps, total)
        if steps >= max_steps:
            dispatch(TurnInterrupted("max_steps", message=BUDGET_EXHAUSTED("max_steps")))
            stop_reason = "max_steps"
            break

        steps += 1
        try:
            outcome = run_step(step_num=steps, build_slice=build_slice, llm=llm, tools=tools,
                               dispatch=dispatch, hooks=hooks)
        except KeyboardInterrupt:
            # ctrl-c MID-STEP (incl. while the LLM is "thinking" — the blocking llm.complete call):
            # the only way to interrupt a blocking request is to let SIGINT raise, then abort cleanly
            # here. (signal= covers programmatic/between-step aborts; this covers the interactive one.)
            dispatch(TurnInterrupted("aborted"))
            return TurnResult("aborted", steps, total)
        total["prompt_tokens"] += outcome.usage.get("prompt_tokens", 0)
        total["completion_tokens"] += outcome.usage.get("completion_tokens", 0)

        # anti-spin floor: repeated guardrail blocks this turn → stop and hand control back to the user
        # (the model should have called ask_user; this is the backstop so a weak model can't spin forever).
        total_blocked += outcome.blocked
        if total_blocked >= STUCK_BLOCK_BUDGET:
            dispatch(TurnInterrupted("stuck", message=STUCK))
            stop_reason = "stuck"
            break

        if outcome.stop_reason == "tool_use":
            continue  # ran tools → keep going

        stop_reason = outcome.stop_reason
        cont = hooks.should_continue_after_stop(stop_reason)  # Oracle/verification plugs in here
        if not (cont and cont.get("continue")):
            break
        # else: a hook forced another turn (e.g. tests failed); the slice now carries the feedback

    dispatch(TurnEnd(stop_reason, steps, total))
    return TurnResult(stop_reason, steps, total)
