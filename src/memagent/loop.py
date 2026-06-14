"""The agent loop — the moat. Stateless core over contracts; ONE model call per turn,
no accumulated history. Mirrors Kimi Code's run-turn/turn-step split.

The core depends ONLY on: build_slice (the reconstruction seam), an LLMClient, a
ToolHost, a dispatch_event callable, and hooks. It never imports implementations and
never touches slice internals (tool results flow back via the slice_sink on events).
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import with_retry
from .events import (
    AssistantText,
    Dispatcher,
    SliceBuilt,
    StepBegin,
    StepEnd,
    ToolResult,
    ToolStarted,
    TurnEnd,
    TurnInterrupted,
)
from .hooks import Hooks
from .scheduler import run_scheduled


@dataclass
class StepOutcome:
    usage: dict
    stop_reason: str


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


def run_tool_batch(tool_calls, tools, dispatch: Dispatcher, hooks: Hooks) -> None:
    """Authorize, schedule (safe-parallel by resource access), and report results in provider order."""
    tasks = []
    metas = []
    for tc in tool_calls:
        dispatch(ToolStarted(tc.name, tc.args))
        decision = hooks.authorize_tool(tc.name, tc.args)
        if not decision.allow:
            tasks.append(([], (lambda d=decision: f"Error: blocked by policy: {d.reason or 'denied'}")))
        else:
            tasks.append((tools.accesses(tc.name, tc.args), (lambda tc=tc: str(tools.run(tc.name, tc.args)))))
        metas.append((tc.name, tc.args))

    outputs = run_scheduled(tasks)
    for (name, args), out in zip(metas, outputs):
        failing = out.startswith("Error") or out.startswith("Exit code")
        dispatch(ToolResult(name, args, out, failing))


def run_step(*, step_num: int, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks) -> StepOutcome:
    before = hooks.before_step(step_num)
    if before and before.get("block"):
        raise RuntimeError(before.get("reason") or f"step {step_num} blocked")

    messages = build_slice()
    dispatch(SliceBuilt(messages[-1]["content"]))
    dispatch(StepBegin(step_num))

    resp = with_retry(
        lambda: llm.complete(messages, tools.schemas()),
        is_retryable=getattr(llm, "is_retryable", None),
        dispatch=dispatch,
    )

    usage = resp.usage or {}
    usage_result = hooks.record_step_usage(usage)  # recorded BEFORE tools, so aborts don't lose it
    stop_turn = bool(usage_result and usage_result.get("stop_turn"))

    stop_reason = _normalize_stop(resp)
    if resp.content:
        dispatch(AssistantText(resp.content))

    effective = "end_turn" if (stop_turn and stop_reason == "tool_use") else stop_reason
    if effective == "tool_use":
        run_tool_batch(resp.tool_calls, tools, dispatch, hooks)

    dispatch(StepEnd(step_num, usage, effective))

    after = hooks.after_step(step_num, usage, effective)
    if after and after.get("stop_turn") and effective == "tool_use":
        effective = "end_turn"
    return StepOutcome(usage=usage, stop_reason=effective)


def run_turn(*, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks | None = None,
             max_steps: int = 40, signal=None) -> TurnResult:
    hooks = hooks or Hooks()
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    steps = 0
    stop_reason = "end_turn"

    while True:
        if signal is not None and signal.is_set():
            dispatch(TurnInterrupted("aborted"))
            return TurnResult("aborted", steps, total)
        if steps >= max_steps:
            dispatch(TurnInterrupted("max_steps"))
            stop_reason = "max_steps"
            break

        steps += 1
        outcome = run_step(step_num=steps, build_slice=build_slice, llm=llm, tools=tools,
                           dispatch=dispatch, hooks=hooks)
        total["prompt_tokens"] += outcome.usage.get("prompt_tokens", 0)
        total["completion_tokens"] += outcome.usage.get("completion_tokens", 0)

        if outcome.stop_reason == "tool_use":
            continue  # ran tools → keep going

        stop_reason = outcome.stop_reason
        cont = hooks.should_continue_after_stop(stop_reason)  # Oracle/verification plugs in here
        if not (cont and cont.get("continue")):
            break
        # else: a hook forced another turn (e.g. tests failed); the slice now carries the feedback

    dispatch(TurnEnd(stop_reason, steps, total))
    return TurnResult(stop_reason, steps, total)
