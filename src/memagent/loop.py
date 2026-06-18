"""The agent loop — the moat. Stateless core over contracts; mirrors Kimi Code's run-turn/turn-step split.

Two loop modes (AGENT_LOOP_MODE; the mode= arg overrides):
  - "accumulate" (DEFAULT): one while(true) = one "thought" = one memory slice. The slice is the SEED
    (built once); working memory ACCUMULATES within the loop as native assistant/tool messages and is
    folded to the durable cache at the boundary. Markov ACROSS loops (no transcript), continuous WITHIN —
    validated to hold accuracy + multi-turn continuity while lifting cache% / cutting cost and removing
    the per-step eviction churn. (run_turn_accumulate)
  - "rebuild" (fallback): the slice is reconstructed every step from durable state — strictly [system,
    user], no within-loop accumulation (the prior validated core). (run_turn body)

The core depends ONLY on: build_slice (the reconstruction seam), an LLMClient, a ToolHost, a
dispatch_event callable, and hooks. It never imports implementations and never touches slice internals
(tool results flow back via the slice_sink on events).
"""
from __future__ import annotations

import json
import os
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


def _final_answer(llm, msgs: list, tools, dispatch, guidance: str) -> None:
    """Closeout helper (Fix 5b): a turn must NEVER end silently or with a bare stub. Offer ONLY ask_user
    (all other tools stay banned) so the model can ASK instead of guessing when blocked/ambiguous; if it
    asks, surface the question as the final message. Otherwise emit its summary — and if that is empty,
    a deterministic, honest fallback so the user always gets a useful, non-empty closing. No new per-step
    cost: runs only on the rare max_steps/stuck path."""
    ask = None
    try:
        for sc in (tools.schemas() if hasattr(tools, "schemas") else []):
            if sc.get("function", {}).get("name") == "ask_user":
                ask = sc
                break
    except Exception:  # noqa: BLE001
        ask = None
    resp = None
    try:
        resp = llm.complete(msgs, [ask] if ask else [])
    except Exception:  # noqa: BLE001
        resp = None
    for tc in (getattr(resp, "tool_calls", None) or []):     # the model chose to ASK → surface the question
        if getattr(tc, "name", "") == "ask_user":
            q = (getattr(tc, "args", None) or {}).get("question")
            if q:
                dispatch(AssistantText(str(q)))
                return
    content = (getattr(resp, "content", "") or "").strip()
    if len(content) >= 40:                                   # a real summary
        dispatch(AssistantText(content))
        return
    if content:                                              # short but non-empty — keep it, don't lose info
        dispatch(AssistantText(content))
        return
    dispatch(AssistantText(                                  # deterministic, never-empty, honest fallback
        "I had to stop here (" + guidance.strip().rstrip(".") + "). I could not confirm the task is fully "
        "complete — please review the changes so far, or re-run with more steps, and tell me if you'd like "
        "me to continue."))


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


def run_tool_batch(tool_calls, tools, dispatch: Dispatcher, hooks: Hooks):
    """Authorize, schedule (safe-parallel by resource access), and report results in provider order.
    Returns (blocked_count, results): `blocked` feeds the anti-spin floor (rebuild path uses only this);
    `results` carries each call's {id,name,args,output,failing} in provider order so the ACCUMULATE path
    can build native tool messages. The rebuild path discards `results` (the slice_sink folds outputs via
    the dispatched ToolResult events)."""
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
        # real OpenAI tool_calls carry an id; synthesize a stable index-based one if absent (e.g. test
        # fakes / a provider that omits it) so accumulate's assistant.tool_calls ↔ tool messages still match.
        metas.append((getattr(tc, "id", None) or f"call_{len(metas)}", tc.name, tc.args))

    outputs = run_scheduled(tasks)
    results = []
    for (tcid, name, args), out in zip(metas, outputs):
        transformed = hooks.transform_tool_result(name, args, out)  # mutating seam (plugins/redaction)
        if transformed is not None:
            out = transformed
        failing = out.startswith("Error") or out.startswith("Exit code")
        dispatch(ToolResult(name, args, out, failing))
        results.append({"id": tcid, "name": name, "args": args, "output": out, "failing": failing})
    return blocked, results


def _assistant_message(resp) -> dict:
    """Reconstruct the OpenAI assistant message (with native tool_calls) for the ACCUMULATE transcript.
    ids are synthesized index-based when absent (matching run_tool_batch's scheme) so the assistant's
    tool_calls and the following tool messages reference the SAME ids."""
    msg: dict = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": getattr(tc, "id", None) or f"call_{i}", "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.args, ensure_ascii=False)}}
            for i, tc in enumerate(resp.tool_calls)
        ]
    return msg


def _prepared(hooks, msgs: list) -> list:
    """Pre-LLM-call hook seam (context injection, prompt-cache-safe): return the hook's rewrite, or
    `msgs` unchanged when it returns None. ONE definition for the three call sites that need identical
    semantics — note `is not None` (an empty-list rewrite is honored), not truthiness."""
    prepared = hooks.prepare_messages(msgs)
    return prepared if prepared is not None else msgs


def run_step(*, step_num: int, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks) -> StepOutcome:
    before = hooks.before_step(step_num)
    if before and before.get("block"):
        raise RuntimeError(before.get("reason") or f"step {step_num} blocked")

    messages = _prepared(hooks, build_slice())
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
            messages = _prepared(hooks, build_slice())
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
        blocked, _ = run_tool_batch(resp.tool_calls, tools, dispatch, hooks)

    dispatch(StepEnd(step_num, usage, effective))

    after = hooks.after_step(step_num, usage, effective)
    if after and after.get("stop_turn") and effective == "tool_use":
        effective = "end_turn"
    return StepOutcome(usage=usage, stop_reason=effective, blocked=blocked)


def run_turn(*, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks | None = None,
             max_steps: int = 40, signal=None, mode: str | None = None) -> TurnResult:
    hooks = hooks or Hooks()
    hooks.reset_for_turn()  # ITEM 1: clear per-turn guards ONCE per user task (not per step)
    # Loop mode. DEFAULT is now "accumulate" — the per-LOOP working-memory loop (3-layer redesign),
    # validated to hold coding accuracy + multi-turn continuity while lifting cache% / cutting cost and
    # dissolving the per-step eviction churn. "rebuild" (the prior per-step reconstruction) stays as the
    # AGENT_LOOP_MODE=rebuild fallback. Env-driven so it's overridable without code changes; mode arg wins.
    if (mode or os.environ.get("AGENT_LOOP_MODE", "accumulate")) == "accumulate":
        return run_turn_accumulate(build_slice=build_slice, llm=llm, tools=tools, dispatch=dispatch,
                                   hooks=hooks, max_steps=max_steps, signal=signal)
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    steps = 0
    total_blocked = 0
    stop_reason = "end_turn"

    def _closeout(guidance: str) -> None:
        # NEVER end a turn silently. On a hard stop (max_steps / stuck) make ONE tool-less call so the
        # model delivers its best answer/summary (or a clarifying question) from what it already has —
        # instead of leaving the user a bare status line. tools=[] structurally forbids further tool
        # calls; fully guarded so a close-out failure can't crash the turn.
        try:
            msgs = _prepared(hooks, build_slice())
            if not msgs:
                return
            msgs = msgs[:-1] + [{"role": msgs[-1]["role"], "content": msgs[-1]["content"]
                                 + "\n\n# TURN IS ENDING — " + guidance + " Give the user your best answer or "
                                 "summary NOW (what you did, what you verified, what remains) from what you "
                                 "already have; make NO edit/run tool call. If the request was ambiguous or you "
                                 "are blocked, call ask_user with ONE concise clarifying question instead."}]
            _final_answer(llm, msgs, tools, dispatch, guidance)
        except Exception:
            pass

    while True:
        if signal is not None and signal.is_set():
            dispatch(TurnInterrupted("aborted"))
            return TurnResult("aborted", steps, total)
        if steps >= max_steps:
            _closeout(BUDGET_EXHAUSTED("max_steps"))
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
            _closeout(STUCK)
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


def run_turn_accumulate(*, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks,
                        max_steps: int = 40, signal=None) -> TurnResult:
    """PROTOTYPE loop_mode='accumulate' — per-LOOP working-memory accumulation (the 3-layer redesign).

    The slice is the SEED, built ONCE (layer-2 distilled carry-forward + memem recall + world-state +
    task). Within the while(true) the working memory ACCUMULATES as native assistant/tool messages —
    NO per-step rebuild, NO eviction. The LLM ends the loop by not calling tools (Markov at the loop
    boundary; continuous within). Same event/contract surface as run_turn; the dispatched ToolResult
    events still drive slice_sink + the episode cache (layer 2 raw). On overflow it threshold-compacts
    the oldest post-seed exchange (the minimal disaster-guardrail; full distillation = Phase B/D).

    hooks.reset_for_turn() is the CALLER's responsibility (run_turn does it before delegating)."""
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    steps = 0
    total_blocked = 0
    stop_reason = "end_turn"

    messages = list(_prepared(hooks, build_slice()))   # SEED — built ONCE, then we only append
    seed_len = len(messages)                # never compact below the seed
    dispatch(SliceBuilt(messages[-1]["content"], messages))

    def _closeout(guidance: str) -> None:
        # NEVER end silently: one tool-less call for a final answer from the accumulated working memory.
        try:
            msgs = messages + [{"role": "user", "content": "# TURN IS ENDING — " + guidance
                                + " Give your best answer/summary NOW (what you did, what you verified, what "
                                "remains) from what you already have; make NO edit/run tool call. If the request "
                                "was ambiguous or you are blocked, call ask_user with ONE concise question instead."}]
            _final_answer(llm, msgs, tools, dispatch, guidance)
        except Exception:  # noqa: BLE001
            pass

    while True:
        if signal is not None and signal.is_set():
            dispatch(TurnInterrupted("aborted"))
            return TurnResult("aborted", steps, total)
        if steps >= max_steps:
            _closeout(BUDGET_EXHAUSTED("max_steps"))
            dispatch(TurnInterrupted("max_steps", message=BUDGET_EXHAUSTED("max_steps")))
            stop_reason = "max_steps"
            break

        steps += 1
        before = hooks.before_step(steps)
        if before and before.get("block"):
            raise RuntimeError(before.get("reason") or f"step {steps} blocked")
        dispatch(StepBegin(steps))

        # overflow → threshold-compact the oldest post-seed exchange and retry (bounded)
        overflow_tries = 0
        while True:
            try:
                resp = with_retry(
                    lambda: llm.complete(messages, tools.schemas()),
                    is_retryable=getattr(llm, "is_retryable", None), dispatch=dispatch,
                )
                break
            except ContextOverflow:
                if len(messages) <= seed_len + 2 or overflow_tries >= 4:
                    raise
                overflow_tries += 1
                del messages[seed_len:seed_len + 2]   # drop the oldest accumulated exchange
                dispatch(SliceTightened(level=overflow_tries))
            except KeyboardInterrupt:
                dispatch(TurnInterrupted("aborted"))
                return TurnResult("aborted", steps, total)

        usage = resp.usage or {}
        usage_result = hooks.record_step_usage(usage)
        stop_turn = bool(usage_result and usage_result.get("stop_turn"))
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)
        if resp.content:
            dispatch(AssistantText(resp.content))

        stop = _normalize_stop(resp)
        if stop != "tool_use" or stop_turn:
            # the LLM ended the while(true) (or budget forced it): append its FINAL text only — never a
            # dangling assistant.tool_calls (that would make the transcript / close-out invalid).
            if resp.content:
                messages.append({"role": "assistant", "content": resp.content})
            stop_reason = "end_turn" if stop_turn else stop
            dispatch(StepEnd(steps, usage, stop_reason))
            cont = hooks.should_continue_after_stop(stop_reason)  # Oracle/verification plugs in here
            if cont and cont.get("continue"):
                messages.append({"role": "user", "content": cont.get("feedback") or "Continue."})
                continue
            break

        # tool_use: accumulate the assistant turn (with tool_calls), run, accumulate the tool results
        messages.append(_assistant_message(resp))
        blocked, results = run_tool_batch(resp.tool_calls, tools, dispatch, hooks)
        total_blocked += blocked
        for r in results:
            messages.append({"role": "tool", "tool_call_id": r["id"], "content": r["output"]})
        dispatch(StepEnd(steps, usage, "tool_use"))

        after = hooks.after_step(steps, usage, "tool_use")
        if after and after.get("stop_turn"):
            stop_reason = "end_turn"
            break
        if total_blocked >= STUCK_BLOCK_BUDGET:
            _closeout(STUCK)
            dispatch(TurnInterrupted("stuck", message=STUCK))
            stop_reason = "stuck"
            break

    dispatch(TurnEnd(stop_reason, steps, total))
    return TurnResult(stop_reason, steps, total)
