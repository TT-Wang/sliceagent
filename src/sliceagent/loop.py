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

import copy
import json

from .access import AllAccess, FileAccess, ReadAllAccess
from .context_overflow import ContextOverflow
from .context import ContextUnfitError, SeedPlan
from .events import (
    AssistantText,
    Dispatcher,
    ModelCallPrepared,
    SliceBuilt,
    SliceTightened,
    StepBegin,
    StepEnd,
    ToolResult,
    ToolStarted,
    TurnEnd,
    TurnInterrupted,
    TurnPhaseChanged,
)
from .guardrails import DEDUP_SAFE_TOOL_NAMES, canonical_tool_args
from .guidance import BUDGET_EXHAUSTED, STUCK
from .hooks import Hooks
from .model_runner import complete_model_call
from .execution import (CHILD_TOKEN_BUDGET_ARG, ToolInvocation, ToolOutcome, ToolPurity,
                        PreflightOverflow, ToolStatus, TurnOutcome, Usage,
                        available_content_capacity, estimate_model_call)
from .registry import ToolText, finalize_tool_outcome, tool_result_text
from .scheduler import ScheduledTool, run_ordered


def _as_text(out):
    """Backward-compatible alias for the registry's canonical presentation coercion."""
    return tool_result_text(out)


# Path-targeted file mutators — a read of a path written by one of these IN THE SAME BATCH must not be
# served from a cached earlier read (it would be stale). Focused on tools that carry a `path` arg.
_FILE_MUTATING_TOOLS = frozenset({"write_file", "edit_file", "str_replace", "append_to_file"})


def _dedup_key(name: str, args):
    """Same-step exact-call dedup key: (name, canonical args). Reuses the guardrail's canonicalizer
    (sorted JSON, `note` stripped) so the dedup identity matches loop-detection's. None ⇒ never dedup
    (odd/unserializable args) — paging that case to the normal execute path, never failing the batch."""
    try:
        return name + "\x00" + canonical_tool_args(args or {})
    except Exception:  # noqa: BLE001
        return None


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
                      "If you need details from an early step, re-derive them or read this session's history/ "
                      "files if available — do not assume that work is undone.]")
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
               "steps; read the history/ files for raw detail):\n" + snap) if snap else OVERFLOW_COMPACTED
    return {"role": "user", "content": content}

# Micro-compaction: on overflow, the FIRST move is to clear the BODIES of
# OLD tool-result messages — the bulky, stale part — while keeping the assistant reasoning skeleton and the
# recent window. Strictly better than dropping whole exchanges (which loses the reasoning too), and it keeps
# every tool_call↔reply pairing intact so the message sequence stays valid. The full output is archived in
# the episode cache, so this is lossless-by-default (once the turn seals, it's in this session's history/ files).
MICRO_KEEP_RECENT = 10
MICRO_MARKER = "[old tool result cleared to fit the window — it's archived losslessly; re-derive it or read the history/ files]"


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


def _complete_preflighted(llm, messages: list[dict], schemas: list[dict], *, on_attempt=None):
    """The one model-call seam used by normal steps and closeout.

    Unknown windows are an explicitly named migration compatibility mode. Setting
    ``llm.require_known_context = True`` (or configuring a positive window) makes it strict.
    """
    return complete_model_call(llm, messages, schemas, retry=False, on_attempt=on_attempt)


def _project_request_seed(plan: SeedPlan, trajectory: list[dict], llm, schemas: list[dict],
                          *, capacity_hint: int | None = None) -> list[dict]:
    """Render one provider-fit seed from a turn-stable logical plan.

    Capacity is recalculated for every call after accounting for the current native trajectory, schemas,
    and output reserve. Exact strict preflight then corrects JSON escaping/Unicode overhead by tightening
    the controller budget until one graded representation fits.
    """
    empty_content: str | list[dict]
    if plan.media_parts:
        empty_content = [{"type": "text", "text": ""}, *[dict(part) for part in plan.media_parts]]
    else:
        empty_content = ""
    fixed = [
        {"role": "system", "content": plan.system},
        {"role": "user", "content": empty_content},
        *trajectory,
    ]
    capacity = available_content_capacity(llm, fixed, schemas)
    if capacity is None:
        try:
            return plan.project(capacity_hint) if capacity_hint is not None else plan.project()
        except ContextUnfitError as error:
            raise ContextOverflow(error) from error
    if capacity_hint is not None:
        capacity = min(capacity, capacity_hint)

    # Each failed iteration either selects a smaller alternative or reduces the exact byte/character gap.
    # The bounded attempt count is defensive; normal plans converge in one or two passes.
    attempts = max(4, len(plan.blocks) + 2)
    for _ in range(attempts):
        try:
            projected = plan.project(capacity)
        except ContextUnfitError as error:
            raise ContextOverflow(error) from error
        candidate = [*projected, *trajectory]
        report = estimate_model_call(llm, candidate, schemas)
        if report.required_tokens <= report.context_window:
            return projected
        capacity = max(0, capacity - max(1, report.required_tokens - report.context_window))
    raise ContextOverflow(ValueError("elastic seed could not converge on a provider-fit representation"))


def _merge_tighter_user(hooked: dict, original: dict, replacement: dict) -> dict | None:
    """Replace the original seed text inside one hook-transformed user message without losing injection."""
    if hooked == original:
        return copy.deepcopy(replacement)
    if not all(isinstance(item, dict) for item in (hooked, original, replacement)):
        return None
    if hooked.get("role") != original.get("role") or replacement.get("role") != original.get("role"):
        return None
    old_content = original.get("content")
    new_content = replacement.get("content")
    live_content = hooked.get("content")
    merged = copy.deepcopy(hooked)
    if isinstance(old_content, str) and isinstance(new_content, str) and isinstance(live_content, str):
        if old_content not in live_content:
            return None
        # Preserve a hook's prefix/suffix (policy or live context), changing only the exact seed projection.
        merged["content"] = live_content.replace(old_content, new_content, 1)
        return merged
    if isinstance(old_content, list) and isinstance(new_content, list) and isinstance(live_content, list):
        old_text = next((part.get("text") for part in old_content
                         if isinstance(part, dict) and part.get("type") == "text"), None)
        new_text = next((part.get("text") for part in new_content
                         if isinstance(part, dict) and part.get("type") == "text"), None)
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            return None
        live_parts = copy.deepcopy(live_content)
        for part in live_parts:
            if isinstance(part, dict) and part.get("type") == "text" \
                    and isinstance(part.get("text"), str) and old_text in part["text"]:
                part["text"] = part["text"].replace(old_text, new_text, 1)
                merged["content"] = live_parts
                return merged
    return None


def _replace_prepared_base(
    prepared: list[dict], hooked_base: list[dict], original_base: list[dict], replacement: list[dict],
) -> tuple[list[dict], list[dict]] | None:
    """Tighten only the SeedPlan user message inside one opaque, already-executed hook result.

    System/policy mutations, appended/prepended messages, and trajectory objects remain byte-for-byte as the
    hook produced them. If an opaque rewrite makes the seed unidentifiable, fail honestly rather than replaying
    a stateful hook or silently dropping its injection.
    """
    if len(hooked_base) < 2 or len(original_base) != len(hooked_base) or len(replacement) != len(hooked_base):
        return None
    limit = len(prepared) - len(hooked_base) + 1
    starts = [
        start for start in range(max(0, limit))
        if all(left is right for left, right in zip(
            prepared[start:start + len(hooked_base)], hooked_base,
        ))
    ]
    if not starts:
        starts = [
            start for start in range(max(0, limit))
            if prepared[start:start + len(hooked_base)] == hooked_base
        ]
    if len(starts) != 1:
        return None
    updated_base = copy.deepcopy(hooked_base)
    merged_user = _merge_tighter_user(hooked_base[1], original_base[1], replacement[1])
    if merged_user is None:
        return None
    updated_base[1] = merged_user
    start = starts[0]
    return ([*prepared[:start], *updated_base, *prepared[start + len(hooked_base):]], updated_base)


def _prepare_model_messages(
    *, seed_plan: SeedPlan | None, trajectory: list[dict], messages: list[dict], llm,
    schemas: list[dict], prepare=None, capacity_hint: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Prepare exactly once, tighten that exact value if needed, and return ``(seed, provider_messages)``.

    The previous path invoked a stateful ``prepare_messages`` hook during fixed-size measurement, candidate
    measurement, inspection, and again for dispatch. Here the hook sees one candidate per real provider
    attempt; strict preflight and ``llm.complete`` consume the same prepared value.
    """
    if seed_plan is None:
        base = copy.deepcopy(messages)
        prepared = prepare(base) if prepare is not None else base
        prepared = base if prepared is None else prepared
        if not isinstance(prepared, list):
            raise TypeError("prepare_messages must return a message list or None")
        return messages[:], prepared

    projected = _project_request_seed(
        seed_plan, trajectory, llm, schemas, capacity_hint=capacity_hint,
    )
    base = [*projected, *trajectory]
    original_base = copy.deepcopy(base)
    hook_base = copy.deepcopy(base)
    prepared = prepare(hook_base) if prepare is not None else hook_base
    prepared = hook_base if prepared is None else prepared
    if not isinstance(prepared, list):
        raise TypeError("prepare_messages must return a message list or None")

    report = estimate_model_call(llm, prepared, schemas)
    if not report.context_window or report.required_tokens <= report.context_window:
        return projected, prepared

    current_hook_base = hook_base
    current_original_base = original_base
    attempts = max(4, len(seed_plan.blocks) + 4)
    for _ in range(attempts):
        selected = seed_plan.last_selection
        used = int(getattr(selected, "used_chars", 0) or 0)
        current_capacity = seed_plan._fixed_user_chars(seed_plan.last_request_copies) + used
        deficit = max(1, report.required_tokens - report.context_window)
        tighter_capacity = max(0, current_capacity - deficit)
        try:
            tighter_seed = _project_request_seed(
                seed_plan, trajectory, llm, schemas, capacity_hint=tighter_capacity,
            )
        except ContextOverflow:
            raise
        replacement = [*tighter_seed, *trajectory]
        tightened = _replace_prepared_base(
            prepared, current_hook_base, current_original_base, replacement,
        )
        if tightened is None:
            raise ContextOverflow(ValueError(
                "prepare_messages rewrote an overflowing seed opaquely; cannot tighten it without replaying the hook"
            ))
        prepared, current_hook_base = tightened
        current_original_base = copy.deepcopy(replacement)
        report = estimate_model_call(llm, prepared, schemas)
        if report.required_tokens <= report.context_window:
            return tighter_seed, prepared
    raise ContextOverflow(ValueError("prepared elastic seed could not converge on a provider-fit representation"))


def _final_answer(llm, msgs: list, tools, dispatch, guidance: str, *, seed_plan=None,
                  seed_len: int = 0, prepare=None, on_attempt=None) -> dict:
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
    call_schemas = [ask] if ask else []
    if seed_plan is not None:
        trajectory = msgs[seed_len:]
        _, msgs = _prepare_model_messages(
            seed_plan=seed_plan, trajectory=trajectory, messages=msgs, llm=llm,
            schemas=call_schemas, prepare=prepare,
        )
    else:
        _, msgs = _prepare_model_messages(
            seed_plan=None, trajectory=[], messages=msgs, llm=llm,
            schemas=call_schemas, prepare=prepare,
        )
    resp = None
    try:
        resp = _complete_preflighted(llm, msgs, call_schemas, on_attempt=on_attempt)
    except Exception:  # noqa: BLE001
        resp = None
    usage = getattr(resp, "usage", None) or {}
    for tc in (getattr(resp, "tool_calls", None) or []):     # the model chose to ASK → surface the question
        if getattr(tc, "name", "") == "ask_user":
            q = (getattr(tc, "args", None) or {}).get("question")
            if q:
                dispatch(AssistantText(str(q), final=False))
                return usage
    content = (getattr(resp, "content", "") or "").strip()
    if content:                                              # a real (or short) summary — keep it
        dispatch(AssistantText(content, final=False))
        return usage
    dispatch(AssistantText(                                  # deterministic, never-empty, honest fallback
        "I had to stop here (" + guidance.strip().rstrip(".") + "). I could not confirm the task is fully "
        "complete — please review the changes so far, or re-run with more steps, and tell me if you'd like "
        "me to continue.", final=False))
    return usage


# Backward-compatible public name; the canonical result is typed and still exposes
# ``stop_reason`` plus a mapping-shaped ``usage`` for existing hosts/tests.
TurnResult = TurnOutcome


def _normalize_stop(resp) -> str:
    fr = (resp.finish_reason or "").lower()
    if fr in ("length", "max_tokens"):
        return "max_tokens"
    if fr in ("content_filter", "filtered"):
        return "filtered"
    return "tool_use" if resp.tool_calls else "end_turn"


def _tool_call_id(tc, i: int, step: int = 0) -> str:
    """The ONE id-assigner: a real provider id, else a stable index fallback. run_tool_batch and
    _assistant_message MUST agree on this or the `tool` messages orphan their `tool_calls`."""
    return getattr(tc, "id", None) or (f"call_{step}_{i}" if step else f"call_{i}")


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
    if _os.environ.get("SLICEAGENT_DEBUG_TRACE"):
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
        return hooks.authorize_tool(name, args or {})   # None args → {} so an argless call isn't an opaque fail-closed deny
    except Exception as e:  # noqa: BLE001
        _hook_debug("authorize_tool", e)
        from types import SimpleNamespace
        return SimpleNamespace(allow=False, reason=f"permission hook errored ({type(e).__name__}) — denied")


def _entry_for(tools, name: str):
    try:
        registry = getattr(tools, "registry", None)
        return registry.entry(name) if registry is not None and hasattr(registry, "entry") else None
    except Exception:  # noqa: BLE001 - metadata failure means conservative UNKNOWN
        return None


def _purity_for(tools, name: str, args: dict, entry) -> ToolPurity:
    if entry is not None:
        return entry.purity
    if name in DEDUP_SAFE_TOOL_NAMES:          # legacy built-in/fake host compatibility
        return ToolPurity.PURE_READ
    try:
        accesses = tools.accesses(name, args)
    except Exception:  # noqa: BLE001
        return ToolPurity.UNKNOWN
    # A dynamic read-only subagent advertises ReadAllAccess but is not in the base registry.
    if accesses and all(isinstance(a, (ReadAllAccess,)) for a in accesses):
        return ToolPurity.PURE_READ
    if any(isinstance(a, FileAccess) and a.operation in ("write", "readwrite") for a in accesses):
        return ToolPurity.EFFECTFUL
    if any(isinstance(a, AllAccess) for a in accesses):
        return ToolPurity.UNKNOWN
    return ToolPurity.UNKNOWN


def run_tool_batch(tool_calls, tools, dispatch: Dispatcher, hooks: Hooks, *, step: int = 0,
                   turn_id: str = "", signal=None):
    """Authorize and execute one provider batch through canonical typed outcomes.

    The return value remains ``(blocked_count, legacy_dicts)`` for callers, while each dict also carries
    its canonical ``outcome`` and status. Only consecutive pure reads overlap; mutations and unknowns are
    ordered barriers. An unconfirmed timeout is INDETERMINATE and cancels every later wave.
    """
    tool_calls = list(tool_calls or ())
    descriptors: list[dict] = []
    scheduled: list[ScheduledTool] = []
    blocked = 0
    dup_of: dict[int, int] = {}
    wave_seen: dict[str, int] = {}
    started_ids: set[str] = set()

    # A provider batch may fan out several children concurrently. Reserve an equal slice of the
    # owning turn's *remaining* budget for each child so parallel delegation cannot multiply the cap.
    # The metadata is host-private: authorization, events, journals, and provider-visible args retain
    # only the model's original call. Nested child loops apply the same rule recursively.
    spawn_names = frozenset({"spawn_agent", "spawn_explore", "spawn_subagent"})
    decisions = []
    for tc in tool_calls:
        raw_args = tc.args if isinstance(getattr(tc, "args", None), dict) else {}
        decisions.append(_safe_authorize(hooks, getattr(tc, "name", "") or "", raw_args))
    spawn_indices = [index for index, tc in enumerate(tool_calls)
                     if (getattr(tc, "name", "") or "") in spawn_names
                     and decisions[index].allow]
    spawn_count = len(spawn_indices)
    remaining = (_safe_advisory(
        "remaining_token_budget", hooks.remaining_token_budget, default=0,
    ) if spawn_count else None)
    child_shares = {}
    if remaining is not None and spawn_count:
        quotient, remainder = divmod(max(0, int(remaining)), spawn_count)
        child_shares = {
            provider_index: quotient + (position < remainder)
            for position, provider_index in enumerate(spawn_indices)
        }

    for provider_index, tc in enumerate(tool_calls):
        name = getattr(tc, "name", "") or ""
        raw_args = tc.args if isinstance(getattr(tc, "args", None), dict) else {}
        call_args = {k: v for k, v in raw_args.items()
                     if k not in ("note", CHILD_TOKEN_BUDGET_ARG)}
        if provider_index in child_shares:
            call_args[CHILD_TOKEN_BUDGET_ARG] = child_shares[provider_index]
        invocation = ToolInvocation(
            _tool_call_id(tc, provider_index, step), name, raw_args, provider_index)
        decision = decisions[provider_index]
        entry = _entry_for(tools, name)
        purity = _purity_for(tools, name, call_args, entry)
        if purity is not ToolPurity.PURE_READ:
            wave_seen.clear()                  # dedup never crosses a mutation/unknown barrier

        can_dedup = bool(entry.deduplicable) if entry is not None else name in DEDUP_SAFE_TOOL_NAMES
        key = _dedup_key(name, call_args) if decision.allow and can_dedup and purity is ToolPurity.PURE_READ else None
        desc = {"invocation": invocation, "args": raw_args, "call_args": call_args,
                "decision": decision, "entry": entry, "purity": purity}
        descriptors.append(desc)
        if key is not None and key in wave_seen:
            dup_of[provider_index] = wave_seen[key]
            continue
        if key is not None:
            wave_seen[key] = provider_index
        if purity is not ToolPurity.PURE_READ:
            wave_seen.clear()

        if not decision.allow and getattr(decision, "counts_as_stuck", True):
            blocked += 1

        def execute(d=desc):
            inv = d["invocation"]
            if not d["decision"].allow:
                raw = ToolText(
                    f"Error: blocked by policy: {d['decision'].reason or 'denied'}", ok=False)
            else:
                try:
                    raw = tools.run(inv.name, d["call_args"])
                except Exception as error:
                    # Dynamic/wrapper hosts may not own a registry boundary. Convert their exception to typed
                    # result data here so it still passes through the same effect factory/default-effect path.
                    uncertain = d["purity"] is not ToolPurity.PURE_READ
                    suffix = (" (the operation may have applied side effects before raising)"
                              if uncertain else "")
                    raw = ToolText(
                        f"Error: {error}{suffix}",
                        status=(ToolStatus.INDETERMINATE if uncertain else ToolStatus.FAILED),
                    )
            return finalize_tool_outcome(
                inv, raw, entry=d["entry"],
                default_effect_id=f"{turn_id or 'turn'}:{step}:{inv.provider_index}:{inv.id}:0",
            )

        def announce(inv=invocation, a=raw_args):
            # Record durable execution truth before admitting the handler into the started set. If required
            # journaling itself fails, the handler never runs and must not be described as indeterminate.
            dispatch(ToolStarted(inv.name, a, inv))
            started_ids.add(inv.id)

        scheduled.append(ScheduledTool(
            invocation, purity, execute, on_start=announce,
            # Read-only children may overlap, but they finish by sealing artifacts and handing references to
            # the parent. A generic thread deadline must not abandon those lifecycle callbacks into a later
            # turn; the parent waits for settlement while still allowing sibling explorers to run in parallel.
            timeout_safe=invocation.name not in ("spawn_agent", "spawn_explore", "spawn_subagent"),
        ))

    outcomes: list[ToolOutcome | None] = [None] * len(descriptors)
    def publish(wave: list[ToolOutcome]) -> None:
        # Transform and reduce this completed pure-read wave / mutation barrier before the scheduler may
        # start the next barrier. Status/effects cannot be rewritten by presentation hooks.
        for raw in wave:
            index = raw.invocation.provider_index
            desc = descriptors[index]
            view = ToolText(raw.text, status=raw.status, effects=raw.effects)
            transformed = _safe_advisory(
                "transform_tool_result",
                lambda d=desc, v=view: hooks.transform_tool_result(
                    d["invocation"].name, d["args"], v),
            )
            out = raw.with_text(transformed) if transformed is not None else raw
            outcomes[index] = out
            dispatch(ToolResult(
                out.invocation.name, dict(out.invocation.args), out.text, out.failing,
                status=out.status.value, invocation_id=out.invocation.id, outcome=out,
            ))

    try:
        run_ordered(
            scheduled, timeout=_tool_timeout(), on_outcomes=publish,
            should_cancel=(signal.is_set if signal is not None else None),
        )
    except KeyboardInterrupt:
        # A handler can be interrupted after ToolStarted but before returning a typed outcome. Publish an
        # explicit indeterminate result for every actually-started call still lacking one, so the live Slice
        # gets exact reconciliation targets immediately. LocalTurnStore's seal invariant independently treats
        # this status as unresolved, preserving the same gate through crash recovery.
        interrupted = []
        for desc in descriptors:
            inv = desc["invocation"]
            if inv.id in started_ids and outcomes[inv.provider_index] is None:
                interrupted.append(ToolOutcome(
                    inv, ToolStatus.INDETERMINATE,
                    "Error: tool execution was interrupted; final side effects are indeterminate",
                ))
        if interrupted:
            publish(interrupted)
        raise

    for index, source in dup_of.items():
        src = outcomes[source]
        inv = descriptors[index]["invocation"]
        outcomes[index] = ToolOutcome(inv, src.status, src.text, ())
        # Every provider invocation gets one durable logical outcome. The source call already applied the
        # semantic effects, so this compatibility reply is explicitly non-reducing.
        dispatch(ToolResult(
            inv.name, dict(inv.args), src.text, src.failing,
            status=src.status.value, invocation_id=inv.id, outcome=outcomes[index], apply_effects=False,
        ))

    return blocked, [out.as_legacy() for out in outcomes]


def _assistant_message(resp, *, step: int = 0) -> dict:
    """Reconstruct the OpenAI assistant message (with native tool_calls) for the accumulated transcript.
    ids are synthesized index-based when absent (matching run_tool_batch's scheme) so the assistant's
    tool_calls and the following tool messages reference the SAME ids."""
    msg: dict = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": _tool_call_id(tc, i, step), "type": "function",
             # #13: tc.args may be None (a tool call with no args) → emit "{}", not "null" (which some
             # providers reject as an invalid arguments payload).
             "function": {"name": tc.name, "arguments": json.dumps(tc.args or {}, ensure_ascii=False)}}
            for i, tc in enumerate(resp.tool_calls)
        ]
    return msg


def _model_usage_from_tool_results(results: list[dict]) -> Usage:
    """Fold child/nested model usage carried by typed tool effects into the owning turn exactly once."""
    total = Usage()
    for result in results:
        outcome = result.get("outcome") if isinstance(result, dict) else None
        for effect in (getattr(outcome, "effects", ()) or ()):
            if effect.kind == "model_usage":
                total = total + Usage.from_value(effect.payload)
    return total


def _prepared(hooks, msgs: list) -> list:
    """Pre-LLM-call hook seam (context injection, prompt-cache-safe): return the hook's rewrite, or
    `msgs` unchanged when it returns None. Note `is not None` (an empty-list rewrite is honored)."""
    prepared = _safe_advisory("prepare_messages", lambda: hooks.prepare_messages(msgs))
    return prepared if prepared is not None else msgs


def run_turn(*, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks | None = None,
             max_steps: int = 40, signal=None, checkpoint=None, consolidate=None,
             turn_id: str = "") -> TurnResult:
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
    total = Usage()
    steps = 0
    total_blocked = 0
    messages: list = []      # defined BEFORE the seed build so _park's closure is safe even if it throws
    seed_len = 0
    seed_plan = None
    slice_built_dispatched = False
    model_attempts: dict[int, int] = {}

    def _model_attempt_observer(step: int):
        """Build one observer shared by retries/re-projections for this semantic step."""
        def observe(_runner_attempt, prepared_messages, report):
            attempt = model_attempts.get(step, 0) + 1
            model_attempts[step] = attempt
            selection = getattr(seed_plan, "last_selection", None)
            pressure = getattr(getattr(selection, "pressure", None), "value", None) or "unknown"
            dispatch(ModelCallPrepared(
                step=step, attempt=attempt, messages=copy.deepcopy(prepared_messages),
                pressure=pressure, preflight_mode=str(getattr(report, "mode", "") or ""),
            ))
        return observe

    def _account(usage: dict) -> None:
        nonlocal total
        total = total + Usage.from_value(usage)

    def _park(reason: str, msg: str | None, *, closeout: bool = True) -> TurnResult:
        """The ONE non-clean exit: an optional ACCOUNTED closeout, then exactly one TurnInterrupted."""
        if closeout and msg is not None and messages:
            try:
                cmsgs = messages + [{"role": "user", "content": "# TURN IS ENDING — " + msg
                    + " Give your best answer/summary NOW (what you did, what you verified, what remains) from "
                    "what you already have; make NO edit/run tool call. If the request was ambiguous or you are "
                    "blocked, call ask_user with ONE concise question instead."}]
                close_usage = _final_answer(
                    llm, cmsgs, tools, dispatch, msg, seed_plan=seed_plan, seed_len=seed_len,
                    prepare=lambda candidate: _prepared(hooks, candidate),
                    on_attempt=_model_attempt_observer(max(1, steps)),
                )
                _account(close_usage)
                _safe_advisory("record_step_usage.closeout", lambda: hooks.record_step_usage(close_usage))
                # Closeout is a real model call. Emit its typed usage before TurnInterrupted so metrics,
                # the episodic collector, and the active runtime persist the same total as TurnOutcome.
                dispatch(StepEnd(steps, Usage.from_value(close_usage).as_dict(), "closeout"))
            except Exception:  # noqa: BLE001
                pass
        dispatch(TurnInterrupted(reason, message=msg))
        return TurnResult(reason, steps, total)

    # The ENTIRE turn (seed build + loop) is wrapped so EVERY non-clean exit routes through _park — even
    # ones we did not anticipate: a non-retryable llm error past with_retry, or a throwing build_slice /
    # retriever / probe. The session must NEVER die uncaught with no TurnInterrupted (Q + R).
    try:
        hooks.reset_for_turn()  # clear per-turn guards ONCE; failures route through the honest park below
        schemas = tools.schemas() if hasattr(tools, "schemas") else []  # stable per session → hoist once
        built_seed = build_slice()       # logical SEED PLAN — built ONCE
        seed_plan = built_seed if isinstance(built_seed, SeedPlan) else None
        messages = list(built_seed)
        seed_len = len(messages)         # never compact below the seed

        reactive_seed_capacity = None  # unknown-window pressure learned from a real provider overflow
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
            tool_phase = False
            try:
                # overflow → compact the OLDEST WHOLE exchange (assistant + ALL its tool replies; a fixed
                # 2-window would orphan tool messages on parallel calls → invalid sequence → provider 400).
                # If the SEED itself overflows (nothing left to compact), fail SOFT — no tighten ladder.
                overflow_tries = 0
                while True:
                    provider_call_started = False
                    try:
                        if seed_plan is not None:
                            projected, provider_messages = _prepare_model_messages(
                                seed_plan=seed_plan, trajectory=messages[seed_len:], messages=messages,
                                llm=llm, schemas=schemas,
                                prepare=lambda candidate: _prepared(hooks, candidate),
                                capacity_hint=reactive_seed_capacity,
                            )
                            messages[:seed_len] = projected
                        else:
                            _, provider_messages = _prepare_model_messages(
                                seed_plan=None, trajectory=[], messages=messages, llm=llm,
                                schemas=schemas, prepare=lambda candidate: _prepared(hooks, candidate),
                            )
                        if not slice_built_dispatched:
                            # Once-per-turn lifecycle/initial-slice event. ModelCallPrepared separately records
                            # every exact physical request, including retries and reactive re-projections.
                            seed_view = provider_messages
                            _rendered = seed_view[-1]["content"] if seed_view else ""
                            if isinstance(_rendered, list):
                                _rendered = next((part.get("text", "") for part in _rendered
                                                  if isinstance(part, dict)
                                                  and part.get("type") == "text"), "")
                            dispatch(SliceBuilt(_rendered, seed_view))
                            slice_built_dispatched = True
                        provider_call_started = True
                        resp = complete_model_call(
                            llm, provider_messages, schemas, dispatch=dispatch,
                            on_attempt=_model_attempt_observer(steps),
                        )
                        break
                    except ContextOverflow as overflow:
                        # A real provider rejection is stronger evidence than a configured/catalog estimate:
                        # stale metadata and multimodal accounting can overflow even a known window. Tighten
                        # one graded seed representation before deleting trajectory. Local preflight failures
                        # have already exhausted the projector and do not replay this reactive path.
                        provider_pressure = provider_call_started and not isinstance(overflow, PreflightOverflow)
                        if seed_plan is not None and provider_pressure:
                            tighter = seed_plan.next_tighter_capacity()
                            if tighter is not None and (reactive_seed_capacity is None
                                                       or tighter < reactive_seed_capacity):
                                before = (
                                    seed_plan.last_request_copies,
                                    tuple(
                                    block.block_id for block in (seed_plan.last_selection.blocks
                                                                 if seed_plan.last_selection else ())
                                    ),
                                )
                                try:
                                    projected = _project_request_seed(
                                        seed_plan, messages[seed_len:], llm, schemas,
                                        capacity_hint=tighter,
                                    )
                                except ContextOverflow:
                                    projected = None
                                after = (
                                    seed_plan.last_request_copies,
                                    tuple(
                                        block.block_id for block in (seed_plan.last_selection.blocks
                                                                     if seed_plan.last_selection else ())
                                    ),
                                )
                                if projected is not None and after != before:
                                    messages[:seed_len] = projected
                                    reactive_seed_capacity = tighter
                                    overflow_tries += 1
                                    dispatch(SliceTightened(level=overflow_tries,
                                                            reason="provider_overflow_seed"))
                                    continue
                        # The breadcrumb (if present) is pinned at seed_len; derive its presence from the
                        # transcript so it is inserted exactly ONCE PER TURN even across multiple overflow
                        # steps (a per-step flag would stack duplicates). floor keeps it below the seed.
                        has_crumb = bool(messages[seed_len:]) and str(messages[seed_len].get("content", "")).startswith(_CRUMB_PREFIX)
                        floor = seed_len + (1 if has_crumb else 0)
                        # MICRO-COMPACTION FIRST: clear OLD tool-result BODIES — keeping the
                        # assistant reasoning, the recent window, and valid tool pairings — before resorting
                        # to dropping a whole exchange. Lossless-by-default (full content in the episode cache).
                        micro = _micro_compact(messages, floor=floor)
                        if not micro and len(messages) <= floor:
                            # micro-clear exhausted AND nothing left to drop (even the seed overflows).
                            # SECONDARY net: if a bigger-context model is configured (AGENT_MODEL_FALLBACK),
                            # swap to it ONCE and retry rather than parking — the moat's compaction stays the
                            # primary, cheaper path.
                            if _try_model_fallback(llm):
                                # A different model owns a different capacity. Do not carry the primary's
                                # learned physical hint across the routing boundary; project afresh.
                                reactive_seed_capacity = None
                                dispatch(SliceTightened(
                                    level=overflow_tries,
                                    reason="model_fallback",
                                    detail=f"switching to {llm.model} for a larger context window",
                                ))
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
                step_usage = Usage.from_value(usage)
                _account(step_usage)
                # record_step_usage is the BUDGET path → fail CLOSED: on a hook exception, default to
                # stop_turn=True (a crashing accountant must STOP, never silently overspend). No downside in
                # normal use — the built-in BudgetHook is plain arithmetic and never raises, so the default
                # only fires on a genuinely broken (e.g. 3rd-party) usage hook, where stopping is correct.
                budget_stop = bool((_safe_advisory("record_step_usage",
                                                   lambda: hooks.record_step_usage(step_usage.as_dict()),
                                                   default={"stop_turn": True}) or {}).get("stop_turn"))
                # A cancellation requested while provider I/O was blocked must stop before any returned tool
                # call can start. The call's real usage is still accounted and made visible.
                if signal is not None and signal.is_set():
                    dispatch(StepEnd(steps, step_usage.as_dict(), "aborted"))
                    return _park("aborted", None, closeout=False)
                stop = _normalize_stop(resp)
                candidate = resp.content or ""

                if budget_stop:
                    # F: a token-budget stop is a PARK, never end_turn/done. Append the final content (never
                    # a dangling tool_calls); no closeout — we're already at the ceiling.
                    if candidate:
                        messages.append({"role": "assistant", "content": candidate})
                        dispatch(AssistantText(candidate, final=False))
                    dispatch(StepEnd(steps, step_usage.as_dict(), "token_budget"))
                    return _park("token_budget", BUDGET_EXHAUSTED("token_budget"), closeout=False)

                if stop != "tool_use":
                    # Keep the candidate in the model trajectory so a failed completion gate can explain what
                    # to revise, but do not publish it as assistant truth until the gate accepts this attempt.
                    if candidate:
                        messages.append({"role": "assistant", "content": candidate})
                    dispatch(StepEnd(steps, step_usage.as_dict(), stop))
                    dispatch(TurnPhaseChanged("checking_completion", "checking whether the turn can finish"))
                    cont = _safe_advisory("should_continue_after_stop", lambda: hooks.should_continue_after_stop(stop))
                    if cont and cont.get("park"):
                        return _park("indeterminate", cont.get("reason") or
                                     "completion verification was indeterminate", closeout=False)
                    if cont and cont.get("continue"):
                        messages.append({"role": "user", "content": cont.get("feedback") or "Continue."})
                        continue
                    if stop in ("max_tokens", "filtered"):
                        # #11: a truncated (length) or content-filtered response is INCOMPLETE — park it as
                        # interrupted instead of sealing a partial answer as a clean turn. Surface any partial
                        # content explicitly as an update, never as the accepted terminal response.
                        if candidate:
                            dispatch(AssistantText(candidate, final=False))
                        return _park(stop, MAX_TOKENS_MSG if stop == "max_tokens" else FILTERED_MSG,
                                     closeout=False)
                    dispatch(AssistantText(candidate or "Done — no summary to add.", final=True))
                    dispatch(TurnEnd(stop, steps, total.as_dict()))   # the ONE clean-exit event
                    return TurnResult(stop, steps, total)

                # tool_use: accumulate the assistant turn (with tool_calls), run, accumulate the tool results
                if candidate:
                    dispatch(AssistantText(candidate, final=False))
                messages.append(_assistant_message(resp, step=steps))
                tool_phase = True
                blocked, results = run_tool_batch(
                    resp.tool_calls, tools, dispatch, hooks, step=steps, turn_id=turn_id, signal=signal,
                )
                tool_phase = False
                total_blocked += blocked
                for r in results:
                    messages.append({"role": "tool", "tool_call_id": r["id"], "content": r["output"]})
                child_usage = _model_usage_from_tool_results(results)
                combined_usage = step_usage + child_usage
                child_budget_stop = False
                if child_usage.prompt_tokens or child_usage.completion_tokens or child_usage.cost_usd is not None:
                    _account(child_usage)
                    child_budget_stop = bool((_safe_advisory(
                        "record_step_usage.child",
                        lambda: hooks.record_step_usage(child_usage.as_dict()),
                        default={"stop_turn": True},
                    ) or {}).get("stop_turn"))
                if any(r.get("status") == ToolStatus.INDETERMINATE.value for r in results):
                    dispatch(StepEnd(steps, combined_usage.as_dict(), "indeterminate"))
                    return _park(
                        "indeterminate",
                        "a tool may still be running after its deadline; dependent actions are blocked until "
                        "the workspace/process state is reconciled",
                        closeout=False,
                    )
                if child_budget_stop:
                    dispatch(StepEnd(steps, combined_usage.as_dict(), "token_budget"))
                    return _park("token_budget", BUDGET_EXHAUSTED("token_budget"), closeout=False)
                usage = combined_usage.as_dict()
                dispatch(StepEnd(steps, usage, "tool_use"))
            except KeyboardInterrupt:
                if tool_phase:
                    # ToolStarted is durably emitted before a handler runs, but Ctrl-C can arrive before a
                    # ToolResult exists. The runtime cannot infer whether effects landed. Preserve that fact
                    # in the Slice immediately; the journal completeness check independently enforces it at
                    # seal/recovery even if a custom host omitted the state reducer.
                    return _park(
                        "indeterminate",
                        "a tool was interrupted after it started; its final effects must be re-observed before "
                        "any further side effect",
                        closeout=False,
                    )
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
        if _os.environ.get("SLICEAGENT_DEBUG_TRACE"):  # opt-in traceback so a parked 'error' is diagnosable
            import sys as _sys
            import traceback as _tb
            _tb.print_exc(file=_sys.stderr)
        return _park("error", f"an internal error ended the turn ({type(e).__name__})", closeout=False)
