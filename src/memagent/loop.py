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
from .registry import ToolText
from .scheduler import run_scheduled


def _as_text(out):
    """Preserve a ToolText (str subclass carrying .ok); coerce anything non-str defensively. Used so the
    scheduler step does not strip the registry's structured success flag back to a plain string."""
    return out if isinstance(out, str) else str(out)


def _tool_timeout() -> float | None:
    """Opt-in per-tool wall-clock deadline (seconds) from AGENT_TOOL_TIMEOUT; None/0/invalid → off (the
    default), preserving the original wait-for-every-tool behaviour. A last-resort net above each tool's
    own subprocess/SIGALRM timeout, for a custom/MCP tool that blocks with no internal limit."""
    import os
    raw = os.environ.get("AGENT_TOOL_TIMEOUT", "").strip()
    try:
        v = float(raw)
        return v if v > 0 else None
    except ValueError:
        return None


# Anti-spin floor: after this many guardrail BLOCKS in one turn the loop stops and hands control back
# to the user (TurnInterrupted "stuck") instead of letting a weak model keep generating variants
# against the guard. The proactive path is the ask_user tool; this is the harness backstop. Each block
# already represents a 3-4x repeat caught by the guardrail, so a few blocks = genuinely stuck.
STUCK_BLOCK_BUDGET = 3

# Shown when the working context overflows and can't be compacted further (the seed itself is too big).
# With one loop mode there's no tighten-ladder fallback, so we fail SOFT here instead of crashing.
OVERFLOW_MSG = ("The working context overflowed and could not be compacted further. Stopping this turn — "
                "try a narrower request, or reduce the number of files in play, and continue.")

# #11: a 'length'/'content_filter' finish is NOT a clean turn — the reply was truncated/blocked, not
# completed, so we PARK (interrupted) instead of sealing it as done.
MAX_TOKENS_MSG = ("The response hit the output token limit and was cut off mid-answer — it is INCOMPLETE. "
                  "Continue, or ask a narrower question.")
FILTERED_MSG = "The response was stopped by the provider's content filter; the turn is incomplete."

# Breadcrumb inserted ONCE when overflow compaction drops the oldest exchange, so the loss is never
# silent: the model is told it happened and how to recover (the episode sink archived it losslessly).
OVERFLOW_COMPACTED = ("[context note: the oldest step(s) of this turn were compacted out to fit the window. "
                      "If you need details from an early step, re-derive them or call recall_history if it is "
                      "available — do not assume that work is undone.]")
_CRUMB_PREFIX = "[context note: the oldest"   # stable prefix to detect the breadcrumb (with or without the checkpoint)


def _overflow_breadcrumb(consolidate) -> dict:
    """F2 — REBUILD-FROM-CHECKPOINT: the overflow breadcrumb carries the DISTILLED state (the deterministic
    checkpoint), not just a generic 'oldest steps compacted' note — so when overflow sheds the oldest raw
    exchanges, the turn's intent/decisions/change-set survive in front of the model. Best-effort: a failing
    or empty checkpoint degrades to the plain note."""
    snap = ""
    if consolidate is not None:
        try:
            snap = (consolidate() or "").strip()
        except Exception:  # noqa: BLE001 — a checkpoint hiccup must never break overflow handling
            snap = ""
    content = (OVERFLOW_COMPACTED + "\n\n# CHECKPOINT — state of play (the distilled state of the compacted "
               "steps; recall_history for raw detail):\n" + snap) if snap else OVERFLOW_COMPACTED
    return {"role": "user", "content": content}

# Micro-compaction (borrowed from Kimi's micro.ts): on overflow, the FIRST move is to clear the BODIES of
# OLD tool-result messages — the bulky, stale part — while keeping the assistant reasoning skeleton and the
# recent window. Strictly better than dropping whole exchanges (which loses the reasoning too), and it keeps
# every tool_call↔reply pairing intact so the message sequence stays valid. The full output is archived in
# the episode cache, so this is lossless-by-default (recall_history pages it back).
MICRO_KEEP_RECENT = 10
MICRO_MARKER = "[old tool result cleared to fit the window — recall_history can page it back]"


def _micro_compact(messages: list, *, floor: int, keep_recent: int = MICRO_KEEP_RECENT) -> bool:
    """Clear the bodies of OLD tool-result messages between `floor` and the recent window (last
    `keep_recent` messages). Returns True if it cleared at least one (the caller retries the LLM call
    before resorting to dropping whole exchanges)."""
    cleared = False
    for i in range(floor, max(floor, len(messages) - keep_recent)):
        m = messages[i]
        if m.get("role") == "tool" and m.get("content") and m["content"] != MICRO_MARKER:
            m["content"] = MICRO_MARKER
            cleared = True
    return cleared


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


def _try_model_fallback(llm) -> bool:
    """On exhausted-compaction overflow, swap to AGENT_MODEL_FALLBACK ONCE (a larger-context model) and
    return True so the loop retries; False if no fallback is configured / already used / same model. Sticky
    for the session — once you've overflowed the primary, the bigger model is the right place to stay."""
    import os
    fb = os.environ.get("AGENT_MODEL_FALLBACK", "").strip()
    if not fb or getattr(llm, "_fellback", False) or fb == getattr(llm, "model", None):
        return False
    llm._fellback = True
    try:
        llm.model = fb
    except Exception:  # noqa: BLE001
        return False
    return True


def _hook_debug(where: str, e: Exception) -> None:
    import os as _os
    if _os.environ.get("MEMAGENT_DEBUG_TRACE"):
        import sys as _sys
        import traceback as _tb
        print(f"[hook error in {where}: {type(e).__name__}: {e}]", file=_sys.stderr)
        _tb.print_exc(file=_sys.stderr)


def _safe_advisory(where: str, fn, default=None):
    """Run an ADVISORY hook (budget/oracle/plugin: before/after_step, record_step_usage, prepare_messages,
    transform_tool_result, should_continue_after_stop). A misbehaving plugin hook must DEGRADE the turn, not
    end it — on any exception we log (opt-in) and return `default` (= 'no opinion'), so the loop continues."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        _hook_debug(where, e)
        return default


def _safe_authorize(hooks, name, args):
    """authorize_tool is SECURITY, so it fails CLOSED: any exception in the permission hook DENIES the call,
    never silently allows it. Returns the hook's ToolDecision, or a synthetic deny on error."""
    try:
        return hooks.authorize_tool(name, args)
    except Exception as e:  # noqa: BLE001
        _hook_debug("authorize_tool", e)
        from types import SimpleNamespace
        return SimpleNamespace(allow=False, reason=f"permission hook errored ({type(e).__name__}) — denied")


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
        decision = _safe_authorize(hooks, tc.name, tc.args)   # SECURITY: fails CLOSED on hook error
        # #44: `note` is the universal slice-capture seam injected by with_note onto EVERY tool — it is
        # NOT a handler parameter. Strip it before invoking the tool so strict MCP/plugin handlers don't
        # reject an unexpected property; it still rides the dispatched ToolResult.args (metas, below),
        # where slice_sink folds it into FINDINGS.
        call_args = {k: v for k, v in (tc.args or {}).items() if k != "note"} if tc.args else (tc.args or {})
        if not decision.allow:
            if getattr(decision, "counts_as_stuck", True):
                blocked += 1   # only a HARD spin counts toward STUCK; a deduped read is skipped but free
            tasks.append(([], (lambda d=decision: ToolText(f"Error: blocked by policy: {d.reason or 'denied'}", ok=False))))
        else:
            # preserve ToolText (a str subclass carrying .ok) — coercing with str() here would strip the
            # success flag and force the failing check back onto the prose-match it is meant to replace.
            tasks.append((tools.accesses(tc.name, call_args),
                          (lambda name=tc.name, ca=call_args: _as_text(tools.run(name, ca)))))
        # real OpenAI tool_calls carry an id; synthesize a stable index-based one if absent (e.g. test
        # fakes / a provider that omits it) so accumulate's assistant.tool_calls ↔ tool messages still match.
        metas.append((_tool_call_id(tc, len(metas)), tc.name, tc.args))

    outputs = run_scheduled(tasks, timeout=_tool_timeout())
    results = []
    for (tcid, name, args), out in zip(metas, outputs):
        transformed = _safe_advisory("transform_tool_result", lambda: hooks.transform_tool_result(name, args, out))
        if transformed is not None:
            out = transformed
        # M: trust the STRUCTURED success flag the registry attached (ToolText.ok) — a tool that returned
        # normally succeeded even if its output begins with "Error"/"Exit code" (a grep hit, a log line).
        # Fall back to the old prose-match ONLY for plain strings with no flag (a plugin transform that
        # returned a bare str, an MCP tool, or a test fake).
        ok = getattr(out, "ok", None)
        failing = (out.startswith("Error") or out.startswith("Exit code")) if ok is None else (not ok)
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
             # #13: tc.args may be None (a tool call with no args) → emit "{}", not "null" (which some
             # providers reject as an invalid arguments payload).
             "function": {"name": tc.name, "arguments": json.dumps(tc.args or {}, ensure_ascii=False)}}
            for i, tc in enumerate(resp.tool_calls)
        ]
    return msg


def _prepared(hooks, msgs: list) -> list:
    """Pre-LLM-call hook seam (context injection, prompt-cache-safe): return the hook's rewrite, or
    `msgs` unchanged when it returns None. Note `is not None` (an empty-list rewrite is honored)."""
    prepared = _safe_advisory("prepare_messages", lambda: hooks.prepare_messages(msgs))
    return prepared if prepared is not None else msgs


def run_turn(*, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks | None = None,
             max_steps: int = 40, signal=None, checkpoint=None, consolidate=None) -> TurnResult:
    """One per-LOOP working-memory turn. The slice is the SEED, built ONCE; within the while(true) working
    memory ACCUMULATES as native assistant/tool messages — NO per-step rebuild, NO eviction. The LLM ends
    by not calling tools (Markov at the loop boundary; continuous within).

    Because the seed is built once, mid-turn hook→model communication rides the MESSAGE channel, never a
    slice mutation (which would never re-render): prepare_messages is applied per llm.complete, and a
    continue-hook's `feedback` (e.g. the Oracle's test failure) is appended as the model's next input.

    Every NON-clean exit — max_steps, stuck, token budget, hook block, overflow, abort, AND any
    UNEXPECTED internal error (a non-retryable llm failure, a throwing build_slice) — routes through ONE
    helper, _park: honest reason + exactly one TurnInterrupted (+ an ACCOUNTED closeout where another model
    call is affordable). A budget/hook stop PARKS — never `end_turn` (the caller checkpoints end_turn⇒done).
    Overflow compacts the oldest WHOLE exchange; a seed that alone overflows parks soft.

    (Known deferred — M: failing-tool detection in run_tool_batch is still a prose match, not a structured
    ToolHost.run ok-flag, so a legit tool output beginning with "Error"/"Exit code" can false-flag.)"""
    hooks = hooks or Hooks()
    hooks.reset_for_turn()  # clear per-turn guards ONCE per user task (not per step)
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    steps = 0
    total_blocked = 0
    said_anything = False    # did the turn emit ANY assistant text? (never end truly silent)
    messages: list = []      # defined BEFORE the seed build so _park's closure is safe even if it throws
    seed_len = 0

    def _account(usage: dict) -> None:
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)

    def _park(reason: str, msg: str | None, *, closeout: bool = True) -> TurnResult:
        """The ONE non-clean exit: an optional ACCOUNTED closeout, then exactly one TurnInterrupted."""
        if closeout and msg is not None and messages:
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

    # The ENTIRE turn (seed build + loop) is wrapped so EVERY non-clean exit routes through _park — even
    # ones we did not anticipate: a non-retryable llm error past with_retry, or a throwing build_slice /
    # retriever / probe. The session must NEVER die uncaught with no TurnInterrupted (Q + R).
    try:
        schemas = tools.schemas() if hasattr(tools, "schemas") else []  # stable per session → hoist once
        messages = list(build_slice())   # SEED — built ONCE, stored RAW (prepare_messages applies per-call)
        seed_len = len(messages)         # never compact below the seed
        seed_view = _prepared(hooks, messages)  # dispatch what the model will SEE (== messages unless a hook injects)
        # #14: guard an empty seed (a build_slice that returns []) — index it directly and we'd IndexError
        # into the generic 'error' park, masking the real cause.
        _rendered = seed_view[-1]["content"] if seed_view else ""
        if isinstance(_rendered, list):   # multimodal user content (image parts) → use the TEXT part for the
            _rendered = next((p.get("text", "") for p in _rendered                # rendered/inspection view
                              if isinstance(p, dict) and p.get("type") == "text"), "")
        dispatch(SliceBuilt(_rendered, seed_view))

        while True:
            if signal is not None and signal.is_set():
                return _park("aborted", None, closeout=False)
            if steps >= max_steps:
                return _park("max_steps", BUDGET_EXHAUSTED("max_steps"))

            steps += 1
            before = _safe_advisory("before_step", lambda: hooks.before_step(steps))
            if before and before.get("block"):
                # a hook signalled "block this step" — PARK gracefully (closeout would re-block), never crash.
                return _park("blocked", before.get("reason") or "step blocked by a hook", closeout=False)
            dispatch(StepBegin(steps))
            if checkpoint is not None:   # crash-recovery WAL: persist the in-flight turn BEFORE the LLM
                _safe_advisory("checkpoint", lambda: checkpoint(messages, steps))   # call (best-effort)

            # The step is interrupt-guarded: ctrl-C anywhere — the blocking llm.complete OR a slow tool in
            # run_tool_batch (a hung run_command) — aborts the turn cleanly instead of crashing it.
            try:
                # overflow → compact the OLDEST WHOLE exchange (assistant + ALL its tool replies; a fixed
                # 2-window would orphan tool messages on parallel calls → invalid sequence → provider 400).
                # If the SEED itself overflows (nothing left to compact), fail SOFT — no tighten ladder.
                overflow_tries = 0
                while True:
                    try:
                        resp = with_retry(
                            lambda: llm.complete(_prepared(hooks, messages), schemas),
                            is_retryable=getattr(llm, "is_retryable", None), dispatch=dispatch,
                        )
                        break
                    except ContextOverflow:
                        # The breadcrumb (if present) is pinned at seed_len; derive its presence from the
                        # transcript so it is inserted exactly ONCE PER TURN even across multiple overflow
                        # steps (a per-step flag would stack duplicates). floor keeps it below the seed.
                        has_crumb = bool(messages[seed_len:]) and str(messages[seed_len].get("content", "")).startswith(_CRUMB_PREFIX)
                        floor = seed_len + (1 if has_crumb else 0)
                        # MICRO-COMPACTION FIRST (Kimi-style): clear OLD tool-result BODIES — keeping the
                        # assistant reasoning, the recent window, and valid tool pairings — before resorting
                        # to dropping a whole exchange. Lossless-by-default (full content in the episode cache).
                        micro = _micro_compact(messages, floor=floor)
                        if not micro and (len(messages) <= floor or overflow_tries >= 4):
                            # micro-clear exhausted AND nothing left to drop (even the seed overflows).
                            # SECONDARY net: if a bigger-context model is configured (AGENT_MODEL_FALLBACK),
                            # swap to it ONCE and retry rather than parking — the moat's compaction stays the
                            # primary, cheaper path.
                            if _try_model_fallback(llm):
                                dispatch(AssistantText(f"[context overflow — switching to {llm.model} "
                                                       "for a larger window]"))
                                overflow_tries = 0
                                continue
                            return _park("overflow", OVERFLOW_MSG, closeout=False)
                        if not micro:   # micro-clear exhausted → drop the oldest WHOLE exchange (assistant + replies)
                            end = floor + 1
                            while end < len(messages) and messages[end].get("role") == "tool":
                                end += 1
                            del messages[floor:end]
                        overflow_tries += 1
                        if not has_crumb:   # breadcrumb ONCE PER TURN, carrying the distilled CHECKPOINT (F2)
                            messages.insert(seed_len, _overflow_breadcrumb(consolidate))
                        dispatch(SliceTightened(level=overflow_tries))

                usage = resp.usage or {}
                _account(usage)
                # record_step_usage is the BUDGET path → fail CLOSED: on a hook exception, default to
                # stop_turn=True (a crashing accountant must STOP, never silently overspend). No downside in
                # normal use — the built-in BudgetHook is plain arithmetic and never raises, so the default
                # only fires on a genuinely broken (e.g. 3rd-party) usage hook, where stopping is correct.
                budget_stop = bool((_safe_advisory("record_step_usage",
                                                   lambda: hooks.record_step_usage(usage),
                                                   default={"stop_turn": True}) or {}).get("stop_turn"))
                if resp.content:
                    dispatch(AssistantText(resp.content))
                    said_anything = True
                stop = _normalize_stop(resp)

                if budget_stop:
                    # F: a token-budget stop is a PARK, never end_turn/done. Append the final content (never
                    # a dangling tool_calls); no closeout — we're already at the ceiling.
                    if resp.content:
                        messages.append({"role": "assistant", "content": resp.content})
                    dispatch(StepEnd(steps, usage, "token_budget"))
                    return _park("token_budget", BUDGET_EXHAUSTED("token_budget"), closeout=False)

                if stop != "tool_use":
                    # the LLM ended the while(true): append its FINAL text only — never a dangling tool_calls.
                    if resp.content:
                        messages.append({"role": "assistant", "content": resp.content})
                    dispatch(StepEnd(steps, usage, stop))
                    cont = _safe_advisory("should_continue_after_stop", lambda: hooks.should_continue_after_stop(stop))
                    if cont and cont.get("continue"):
                        messages.append({"role": "user", "content": cont.get("feedback") or "Continue."})
                        continue
                    if stop in ("max_tokens", "filtered"):
                        # #11: a truncated (length) or content-filtered response is INCOMPLETE — park it as
                        # interrupted instead of sealing a partial answer as a clean turn. Content (if any)
                        # was already emitted + appended above; no closeout (it would truncate again too).
                        return _park(stop, MAX_TOKENS_MSG if stop == "max_tokens" else FILTERED_MSG,
                                     closeout=False)
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

            after = _safe_advisory("after_step", lambda: hooks.after_step(steps, usage, "tool_use"))
            if after and after.get("stop_turn"):
                return _park("token_budget", BUDGET_EXHAUSTED("token_budget"), closeout=False)
            if total_blocked >= STUCK_BLOCK_BUDGET:
                return _park("stuck", STUCK)
    except KeyboardInterrupt:
        # ctrl-C during SETUP (build_slice/schemas), before the step's own interrupt guard is in scope.
        return _park("aborted", None, closeout=False)
    except Exception as e:  # noqa: BLE001 — Q + R: any unexpected error PARKS, never crashes the session.
        import os as _os
        if _os.environ.get("MEMAGENT_DEBUG_TRACE"):  # opt-in traceback so a parked 'error' is diagnosable
            import sys as _sys
            import traceback as _tb
            _tb.print_exc(file=_sys.stderr)
        return _park("error", f"an internal error ended the turn ({type(e).__name__})", closeout=False)
