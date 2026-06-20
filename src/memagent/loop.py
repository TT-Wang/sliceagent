"""The agent loop — the moat. Stateless core over contracts.

One while(true) = one "thought" = one memory slice. The slice is the SEED (built ONCE); working memory
ACCUMULATES within the loop as native assistant/tool messages and is folded to the durable cache at the
turn boundary (the seal). Markov ACROSS loops (no transcript), continuous WITHIN — validated to hold
coding accuracy + multi-turn continuity while lifting cache% / cutting cost and dissolving per-step
eviction churn. On context overflow it drops the oldest accumulated exchange (never grows a transcript).

The core depends ONLY on: build_slice (the reconstruction seam), an LLMClient, a ToolHost, a
dispatch_event callable, and hooks. It never imports implementations and never touches slice internals
(tool results flow back via the slice_sink on events).
"""
from __future__ import annotations

import json
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

# Shown when the working context overflows and can't be compacted further (the seed itself is too big).
# With one loop mode there's no tighten-ladder fallback, so we fail SOFT here instead of crashing.
OVERFLOW_MSG = ("The working context overflowed and could not be compacted further. Stopping this turn — "
                "try a narrower request, or reduce the number of files in play, and continue.")


def _final_answer(llm, msgs: list, tools, dispatch, guidance: str) -> dict:
    """Closeout helper: a turn must NEVER end silently or with a bare stub. Offer ONLY ask_user (all other
    tools stay banned) so the model can ASK instead of guessing when blocked/ambiguous; if it asks, surface
    the question as the final message. Otherwise emit its summary — and if that is empty, a deterministic,
    honest fallback. RETURNS the closeout completion's usage so the caller accounts it (it's a real model
    call, and the budget must see its own closeout — no silent overspend)."""
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
    usage = getattr(resp, "usage", None) or {}
    for tc in (getattr(resp, "tool_calls", None) or []):     # the model chose to ASK → surface the question
        if getattr(tc, "name", "") == "ask_user":
            q = (getattr(tc, "args", None) or {}).get("question")
            if q:
                dispatch(AssistantText(str(q)))
                return usage
    content = (getattr(resp, "content", "") or "").strip()
    if content:                                              # a real (or short) summary — keep it
        dispatch(AssistantText(content))
        return usage
    dispatch(AssistantText(                                  # deterministic, never-empty, honest fallback
        "I had to stop here (" + guidance.strip().rstrip(".") + "). I could not confirm the task is fully "
        "complete — please review the changes so far, or re-run with more steps, and tell me if you'd like "
        "me to continue."))
    return usage


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


def _tool_call_id(tc, i: int) -> str:
    """The ONE id-assigner: a real provider id, else a stable index fallback. run_tool_batch and
    _assistant_message MUST agree on this or the `tool` messages orphan their `tool_calls`."""
    return getattr(tc, "id", None) or f"call_{i}"


def run_tool_batch(tool_calls, tools, dispatch: Dispatcher, hooks: Hooks):
    """Authorize, schedule (safe-parallel by resource access), and report results in provider order.
    Returns (blocked_count, results): `blocked` feeds the anti-spin floor; `results` carries each call's
    {id,name,args,output,failing} in provider order so the loop can build native tool messages. The
    dispatched ToolResult events drive slice_sink + the episode cache independently of the return."""
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
        metas.append((_tool_call_id(tc, len(metas)), tc.name, tc.args))

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
    """Reconstruct the OpenAI assistant message (with native tool_calls) for the accumulated transcript.
    ids are synthesized index-based when absent (matching run_tool_batch's scheme) so the assistant's
    tool_calls and the following tool messages reference the SAME ids."""
    msg: dict = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": _tool_call_id(tc, i), "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.args, ensure_ascii=False)}}
            for i, tc in enumerate(resp.tool_calls)
        ]
    return msg


def _prepared(hooks, msgs: list) -> list:
    """Pre-LLM-call hook seam (context injection, prompt-cache-safe): return the hook's rewrite, or
    `msgs` unchanged when it returns None. Note `is not None` (an empty-list rewrite is honored)."""
    prepared = hooks.prepare_messages(msgs)
    return prepared if prepared is not None else msgs


def run_turn(*, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks | None = None,
             max_steps: int = 40, signal=None) -> TurnResult:
    """One per-LOOP working-memory turn. The slice is the SEED, built ONCE; within the while(true) working
    memory ACCUMULATES as native assistant/tool messages — NO per-step rebuild, NO eviction. The LLM ends
    by not calling tools (Markov at the loop boundary; continuous within).

    Because the seed is built once, mid-turn hook→model communication rides the MESSAGE channel, never a
    slice mutation (which would never re-render): prepare_messages is applied per llm.complete, and a
    continue-hook's `feedback` (e.g. the Oracle's test failure) is appended as the model's next input.

    Every NON-clean exit (max_steps, stuck, token budget, hook block, overflow, abort) routes through ONE
    helper, _park: honest reason + exactly one TurnInterrupted (+ an ACCOUNTED closeout where another model
    call is affordable). A budget/hook stop PARKS — never `end_turn` (the caller checkpoints end_turn⇒done).
    Overflow compacts the oldest WHOLE exchange; a seed that alone overflows parks soft."""
    hooks = hooks or Hooks()
    hooks.reset_for_turn()  # clear per-turn guards ONCE per user task (not per step)
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    steps = 0
    total_blocked = 0
    said_anything = False    # did the turn emit ANY assistant text? (never end truly silent)

    schemas = tools.schemas() if hasattr(tools, "schemas") else []  # stable per session → hoist once
    messages = list(build_slice())   # SEED — built ONCE, stored RAW (prepare_messages applies per-call)
    seed_len = len(messages)         # never compact below the seed
    dispatch(SliceBuilt(messages[-1]["content"], messages))

    def _account(usage: dict) -> None:
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)

    def _park(reason: str, msg: str | None, *, closeout: bool = True) -> TurnResult:
        """The ONE non-clean exit: an optional ACCOUNTED closeout, then exactly one TurnInterrupted."""
        if closeout and msg is not None:
            try:
                cmsgs = _prepared(hooks, messages + [{"role": "user", "content": "# TURN IS ENDING — " + msg
                    + " Give your best answer/summary NOW (what you did, what you verified, what remains) from "
                    "what you already have; make NO edit/run tool call. If the request was ambiguous or you are "
                    "blocked, call ask_user with ONE concise question instead."}])
                _account(_final_answer(llm, cmsgs, tools, dispatch, msg))
            except Exception:  # noqa: BLE001
                pass
        dispatch(TurnInterrupted(reason, message=msg))
        return TurnResult(reason, steps, total)

    while True:
        if signal is not None and signal.is_set():
            return _park("aborted", None, closeout=False)
        if steps >= max_steps:
            return _park("max_steps", BUDGET_EXHAUSTED("max_steps"))

        steps += 1
        before = hooks.before_step(steps)
        if before and before.get("block"):
            # a hook signalled "block this step" — PARK gracefully (closeout would re-block), never crash.
            return _park("blocked", before.get("reason") or "step blocked by a hook", closeout=False)
        dispatch(StepBegin(steps))

        # The whole step is interrupt-guarded: ctrl-C anywhere — the blocking llm.complete OR a slow tool
        # in run_tool_batch (a hung run_command) — aborts the turn cleanly instead of crashing it.
        try:
            # overflow → compact the OLDEST WHOLE exchange (assistant + ALL its tool replies; a fixed
            # 2-window would orphan tool messages on parallel calls → invalid sequence → provider 400).
            # If the SEED itself overflows (nothing left to compact), fail SOFT — no tighten ladder exists.
            overflow_tries = 0
            while True:
                try:
                    resp = with_retry(
                        lambda: llm.complete(_prepared(hooks, messages), schemas),
                        is_retryable=getattr(llm, "is_retryable", None), dispatch=dispatch,
                    )
                    break
                except ContextOverflow:
                    if len(messages) <= seed_len or overflow_tries >= 4:
                        return _park("overflow", OVERFLOW_MSG, closeout=False)
                    end = seed_len + 1
                    while end < len(messages) and messages[end].get("role") == "tool":
                        end += 1
                    overflow_tries += 1
                    del messages[seed_len:end]
                    dispatch(SliceTightened(level=overflow_tries))

            usage = resp.usage or {}
            _account(usage)
            budget_stop = bool((hooks.record_step_usage(usage) or {}).get("stop_turn"))
            if resp.content:
                dispatch(AssistantText(resp.content))
                said_anything = True
            stop = _normalize_stop(resp)

            if budget_stop:
                # F: a token-budget stop is a PARK, never end_turn/done. Append the final content (never a
                # dangling tool_calls); no closeout — we're already at the ceiling.
                if resp.content:
                    messages.append({"role": "assistant", "content": resp.content})
                dispatch(StepEnd(steps, usage, "token_budget"))
                return _park("token_budget", BUDGET_EXHAUSTED("token_budget"), closeout=False)

            if stop != "tool_use":
                # the LLM ended the while(true): append its FINAL text only — never a dangling tool_calls.
                if resp.content:
                    messages.append({"role": "assistant", "content": resp.content})
                dispatch(StepEnd(steps, usage, stop))
                cont = hooks.should_continue_after_stop(stop)  # Oracle/verify: feedback rides the message channel
                if cont and cont.get("continue"):
                    messages.append({"role": "user", "content": cont.get("feedback") or "Continue."})
                    continue
                if not said_anything:   # never end the turn truly silent (e.g. empty end_turn after tools)
                    dispatch(AssistantText("Done — no summary to add."))
                dispatch(TurnEnd(stop, steps, total))   # the ONE clean-exit event
                return TurnResult(stop, steps, total)

            # tool_use: accumulate the assistant turn (with tool_calls), run, accumulate the tool results
            messages.append(_assistant_message(resp))
            blocked, results = run_tool_batch(resp.tool_calls, tools, dispatch, hooks)
            total_blocked += blocked
            for r in results:
                messages.append({"role": "tool", "tool_call_id": r["id"], "content": r["output"]})
            dispatch(StepEnd(steps, usage, "tool_use"))
        except KeyboardInterrupt:
            return _park("aborted", None, closeout=False)

        after = hooks.after_step(steps, usage, "tool_use")
        if after and after.get("stop_turn"):
            return _park("token_budget", BUDGET_EXHAUSTED("token_budget"), closeout=False)
        if total_blocked >= STUCK_BLOCK_BUDGET:
            return _park("stuck", STUCK)
