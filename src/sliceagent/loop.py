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
import hashlib
import json
import math
import threading
import time
from collections import Counter, deque

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
    ToolExecutionStarted,
    ToolRejected,
    ToolQueued,
    ToolRequested,
    ToolResult,
    ToolSettled,
    ToolStarted,
    TurnEnd,
    TurnInterrupted,
    TurnPhaseChanged,
)
from .tool_identity import DEDUP_SAFE_TOOL_NAMES, canonical_tool_args
from .guidance import BUDGET_EXHAUSTED
from .hooks import Hooks, ToolPreflight
from .errors import IndeterminateModelCallError, RetryCancelledError
from .model_runner import complete_model_call
from .execution import (CHILD_CANCEL_SIGNAL_ARG, CHILD_INVOCATION_ID_ARG,
                        CHILD_REQUEST_ORDINAL_ARG, CHILD_TOKEN_BUDGET_ARG,
                        ToolInvocation, ToolOutcome, ToolPurity,
                        PreflightOverflow, ToolStatus, TurnOutcome, Usage,
                        available_content_capacity, estimate_model_call)
from .registry import ToolAdmission, ToolText, finalize_tool_outcome, tool_result_text
from .scheduler import ScheduledTool, run_ordered


def _as_text(out):
    """Backward-compatible alias for the registry's canonical presentation coercion."""
    return tool_result_text(out)


# Path-targeted file mutators — a read of a path written by one of these IN THE SAME BATCH must not be
# served from a cached earlier read (it would be stale). Focused on tools that carry a `path` arg.
_FILE_MUTATING_TOOLS = frozenset({"write_file", "edit_file", "str_replace", "append_to_file"})

# A deliberately advisory liveness signal, not an execution gate. These are the built-in observation
# surfaces whose successful text can be compared meaningfully across calls. Eight *distinct* calls returning
# the same non-empty observation is strong evidence that the model is re-inspecting rather than learning.
# The bounded ring catches that live failure without refusing a ninth call or exposing a policy error.
_OBSERVATION_TOOLS = DEDUP_SAFE_TOOL_NAMES | frozenset({"code_review"})
_OBSERVATION_REPEAT_THRESHOLD = 8
_OBSERVATION_REPEAT_WINDOW = 24
_OBSERVATION_REPEAT_NUDGE = (
    "# INTERNAL RECOVERY NUDGE (liveness advisory; not a new user request)\n"
    "Eight distinct observation calls returned the same non-empty result. Stop re-inspecting that state. "
    "Use the evidence already present: synthesize the answer, take a concrete next action, or ask the user "
    "one concise question if a genuinely missing choice prevents progress. Ordinary tools remain available."
)


class _ObservationRepeatAdvisory:
    """One-shot, per-turn detector for varying reads that yield one repeated observation.

    Only fixed-size hashes are retained in a bounded ring. Exact call repeats do not count as distinct,
    failures/empty results do not count, and effectful or open-ended tools are outside the observation set.
    """

    def __init__(self, *, threshold: int = _OBSERVATION_REPEAT_THRESHOLD,
                 window: int = _OBSERVATION_REPEAT_WINDOW):
        self.threshold = max(2, int(threshold))
        self._recent: deque[tuple[bytes, bytes]] = deque(maxlen=max(self.threshold, int(window)))
        self._emitted = False

    @staticmethod
    def _signature(row: dict) -> tuple[bytes, bytes] | None:
        name = str(row.get("name") or "")
        if name not in _OBSERVATION_TOOLS or row.get("status") != ToolStatus.SUCCEEDED.value:
            return None
        normalized = " ".join(str(row.get("output") or "").split())
        if not normalized:
            return None
        try:
            call = name + "\x00" + canonical_tool_args(row.get("args") or {})
        except Exception:  # malformed extension metadata is not a reason to invent a repetition signal
            return None
        return (
            hashlib.sha256(normalized.encode("utf-8", "replace")).digest(),
            hashlib.sha256(call.encode("utf-8", "replace")).digest(),
        )

    def observe(self, rows: list[dict]) -> bool:
        """Return True exactly once when the advisory should be appended to the live trajectory."""
        if self._emitted:
            return False
        for row in rows:
            signature = self._signature(row)
            if signature is None or signature in self._recent:
                continue
            self._recent.append(signature)
            result_digest = signature[0]
            if sum(seen_result == result_digest for seen_result, _call in self._recent) >= self.threshold:
                self._emitted = True
                return True
        return False


def _dedup_key(name: str, args):
    """Same-step exact-call dedup key: ``(name, canonical args)``.

    Canonicalization uses sorted JSON with ``note`` stripped. ``None`` means never deduplicate
    (odd/unserializable args), so that call follows the normal execution path.
    """
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


def _delegation_timeout() -> float:
    """Wall-clock CEILING (seconds) for a delegation/lifecycle read-wave, from AGENT_DELEGATION_TIMEOUT.
    Defaults NON-None (900s) and, unlike _tool_timeout, cannot be turned off: a spawned child is exempt
    from the SHORT per-tool reader deadline (it must be allowed to SEAL its report rather than be abandoned
    mid-write), but a child whose loop never terminates would otherwise freeze the parent turn forever —
    the wave's own deadline machinery marks a still-running child INDETERMINATE at this ceiling and the turn
    continues. Default is generous (a real child, bounded by max_steps and its per-call watchdog, seals in
    well under it); 0/invalid → the default. To tolerate a slow proxy, RAISE it — there is deliberately no
    disable, since disabling reinstates the freeze."""
    import os
    raw = os.environ.get("AGENT_DELEGATION_TIMEOUT", "").strip()
    try:
        v = float(raw)
        return v if math.isfinite(v) and v > 0 else 900.0
    except ValueError:
        return 900.0


def _delegation_cancel_grace() -> float:
    """Scheduler wait for a child to unwind its cancellable transport/tool stack after wave cutoff.

    The transport owns ``LLM_STREAM_CLOSE_GRACE_SEC``. Give the child a small host-side margin to fold the
    cancellation into a typed result and release its lifecycle slot; invalid/non-finite values use the same
    two-second transport default. This is not extra execution time—the cancellation lease is already set.
    """
    import os
    raw = os.environ.get("LLM_STREAM_CLOSE_GRACE_SEC", "").strip()
    try:
        value = float(raw) if raw else 2.0
    except ValueError:
        value = 2.0
    if not math.isfinite(value) or value <= 0:
        value = 2.0
    return max(0.15, value + 0.10)


class _ChildCancellationLease:
    """Per-invocation Event-like cancellation composed with the owning parent turn.

    The scheduler sets the local edge on delegation deadline/cutoff. Parent cancellation remains live through
    composition, while ``wait`` lets retry backoff wake promptly through the existing Event-owner feature test.
    """

    def __init__(self, parent=None):
        self._parent = parent
        self._local = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""

    @staticmethod
    def _set(source) -> bool:
        try:
            return bool(source is not None and source.is_set())
        except Exception:
            return False

    def request(self, reason: str = "cancel") -> None:
        with self._lock:
            if not self._reason:
                self._reason = str(reason or "cancel")
        self._local.set()

    def is_set(self) -> bool:
        return self._local.is_set() or self._set(self._parent)

    def wait(self, timeout: float | None = None) -> bool:
        if self.is_set():
            return True
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return self.is_set()
            self._local.wait(0.05 if remaining is None else min(0.05, remaining))
            if self.is_set():
                return True

    @property
    def reason(self) -> str:
        with self._lock:
            local = self._reason
        if local:
            return local
        return "parent" if self._set(self._parent) else ""


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
# every tool_call↔reply pairing intact so the message sequence stays valid. Cleared bytes are not silently
# claimed recoverable: use an emitted artifact/blob locator when one exists, or re-observe the source.
MICRO_KEEP_RECENT = 10
MICRO_MARKER = ("[old tool result cleared to fit the window — use its artifact/blob locator if one was "
                "emitted, otherwise re-observe the source]")


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


def _complete_preflighted(
    llm,
    messages: list[dict],
    schemas: list[dict],
    *,
    on_attempt=None,
    should_cancel=None,
    transport_activity=None,
):
    """The one model-call seam used by normal steps and closeout.

    Unknown windows are an explicitly named migration compatibility mode. Setting
    ``llm.require_known_context = True`` (or configuring a positive window) makes it strict.
    """
    return complete_model_call(
        llm, messages, schemas, retry=False, on_attempt=on_attempt,
        should_cancel=should_cancel, transport_activity=transport_activity,
    )


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
        # Preserve a hook's prefix/suffix (for example live context), changing only the exact seed projection.
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

    System/hook mutations, appended/prepended messages, and trajectory objects remain byte-for-byte as the
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
                  seed_len: int = 0, prepare=None, on_attempt=None,
                  should_cancel=None, transport_activity=None) -> dict:
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
        resp = _complete_preflighted(
            llm, msgs, call_schemas, on_attempt=on_attempt,
            should_cancel=should_cancel, transport_activity=transport_activity,
        )
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
        "me to continue.", final=False, synthetic=True))
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


def _tool_call_id(tc, i: int, step: int = 0, namespace: str = "") -> str:
    """The ONE id-assigner: a real provider id, else a stable index fallback. run_tool_batch and
    _assistant_message MUST agree on this or the `tool` messages orphan their `tool_calls`."""
    if getattr(tc, "id", None):
        # A rebuilt lifecycle (currently timeout recovery) may receive the same provider-issued ID as its
        # first attempt. Prefix both real and synthesized IDs inside that private namespace; the reconstructed
        # assistant call and its tool reply still share this exact value.
        base = f"{namespace}_{tc.id}" if namespace else str(tc.id)
        if step:
            # Provider IDs need pair calls/replies only inside one assistant exchange; some compatible
            # endpoints reuse them later. Scope physical identity to this model pass while keeping both
            # reconstructed assistant calls and replies on the same normalized value.
            digest = hashlib.sha256(base.encode("utf-8", errors="replace")).hexdigest()[:8]
            return f"{base[:44]}__s{step}_{digest}"
        return base
    prefix = f"call_{namespace}_" if namespace else "call_"
    return f"{prefix}{step}_{i}" if step else f"{prefix}{i}"


def _batch_tool_call_ids(tool_calls, step: int = 0, namespace: str = "") -> list[str]:
    """Return provider-pairing IDs that are unique inside one assistant tool-call batch."""
    calls = list(tool_calls or ())
    bases = [_tool_call_id(call, index, step, namespace) for index, call in enumerate(calls)]
    counts = Counter(bases)
    used: set[str] = set()
    result = []
    for index, base in enumerate(bases):
        candidate = base
        if counts[base] > 1 or candidate in used:
            candidate = f"{base}__slice_{step}_{index}"
        suffix = 1
        while candidate in used:
            candidate = f"{base}__slice_{step}_{index}_{suffix}"
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


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
    """Run an advisory hook (budget/oracle/observation callbacks).

    A callback defect degrades the turn instead of ending it: log only in debug mode and return ``default``
    (no opinion), so ordinary work continues.
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        _hook_debug(where, e)
        return default


def _safe_preflight(hooks, name, args):
    """Run the narrow tool preflight without turning hook bugs into user-facing blockers.

    The catastrophic safeguard is deliberately small and deterministic. Lifecycle hooks cannot strand ordinary
    work merely because they raised: log the defect in debug mode and proceed.
    """
    try:
        result = hooks.preflight_tool(name, args or {})
        return result if result is not None else ToolPreflight()
    except Exception as e:  # noqa: BLE001
        _hook_debug("preflight_tool", e)
        return ToolPreflight()


def _stop_parts(preflight) -> tuple[str, str, str]:
    """Return ``(kind, reason, model_text)`` for a pre-execution stop.

    The kind is data, never inferred from prose. Catastrophic refusals keep an explicit safety-stop message;
    the only other stop is a neutral lifecycle cancellation rather than an error or permission accusation.
    """
    raw_kind = str(getattr(preflight, "kind", "") or "lifecycle").strip().lower()
    kind = "catastrophic" if raw_kind == "catastrophic" else "lifecycle"
    reason = str(
        getattr(preflight, "reason", "") or "the tool was cancelled before execution"
    ).strip()
    if kind == "catastrophic":
        if not reason.startswith("Safety stop"):
            reason = f"Safety stop: {reason}"
        return kind, reason, reason
    return kind, reason, f"Not run: {reason}"


def _entry_for(tools, name: str):
    try:
        registry = getattr(tools, "registry", None)
        return registry.entry(name) if registry is not None and hasattr(registry, "entry") else None
    except (Exception, SystemExit):  # metadata failure means conservative UNKNOWN; extension exit is contained
        return None


def _purity_for(tools, name: str, args: dict, entry) -> ToolPurity:
    if entry is not None:
        return entry.purity
    if name in DEDUP_SAFE_TOOL_NAMES:          # legacy built-in/fake host compatibility
        return ToolPurity.PURE_READ
    try:
        accesses = tools.accesses(name, args)
    except (Exception, SystemExit):
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
                   turn_id: str = "", signal=None, call_namespace: str = ""):
    """Preflight and execute one provider batch through canonical typed outcomes.

    The return value retains its legacy ``(0, legacy_dicts)`` shape for callers; pre-handler rejections are
    represented by typed terminal outcomes plus ``rejection_kind``/``rejection_reason`` metadata, never a
    synthetic error count. Only consecutive pure reads overlap; mutations and unknowns are ordered barriers. Generic
    deadlines apply only to declared pure reads: a reader settling during bounded grace is a normal failure,
    while one still running after grace is indeterminate and cancels every later wave.
    """
    # Freeze dynamic workspace/session routers for this physical batch. A daemon read whose start journal
    # crosses a deadline may finish its callback after the caller has sealed or switched workspace; bound
    # sinks either keep that edge on the original active epoch or ignore it once that epoch is no longer live.
    bind_dispatch = getattr(dispatch, "bind_dispatch", None)
    if callable(bind_dispatch):
        dispatch = bind_dispatch()
    tool_calls = list(tool_calls or ())
    physical_ids = _batch_tool_call_ids(tool_calls, step, call_namespace)
    raw_provider_ids = [
        str(getattr(call, "id", "") or "") for call in tool_calls
    ]
    raw_provider_id_counts = Counter(raw_provider_ids)
    # Provider call IDs are normalized/model-step-scoped for lifecycle correlation, but default semantic
    # effect IDs retain the provider's canonical raw identity. This keeps retry/replay idempotence and the
    # established durable effect contract. A missing (or malformed duplicate) provider ID falls back to the
    # unique physical invocation identity so two calls can never share one effect ID.
    effect_call_ids = [
        raw_id if raw_id and raw_provider_id_counts[raw_id] == 1 else physical_ids[index]
        for index, raw_id in enumerate(raw_provider_ids)
    ]

    def default_effect_id(invocation: ToolInvocation) -> str:
        return (
            f"{turn_id or 'turn'}:{step}:{invocation.provider_index}:"
            f"{effect_call_ids[invocation.provider_index]}:0"
        )

    descriptors: list[dict] = []
    scheduled: list[ScheduledTool] = []
    dup_of: dict[int, int] = {}
    wave_seen: dict[str, int] = {}
    start_publication_attempt_ids: set[str] = set()
    started_ids: set[str] = set()
    handoff_index: int | None = None

    # A provider batch may fan out several children concurrently. Reserve an equal slice of the
    # owning turn's *remaining* budget for each child so parallel delegation cannot multiply the cap.
    # The metadata is host-private: preflight, events, journals, and provider-visible args retain
    # only the model's original call. Nested child loops apply the same rule recursively.
    spawn_names = frozenset({"spawn_agent", "spawn_explore", "spawn_subagent"})
    invocations = []
    for provider_index, tc in enumerate(tool_calls):
        raw_args = tc.args if isinstance(getattr(tc, "args", None), dict) else {}
        invocation = ToolInvocation(
            physical_ids[provider_index], getattr(tc, "name", "") or "",
            raw_args, provider_index,
        )
        invocations.append(invocation)
        # The logical request exists whether it proceeds, is deduplicated, is cancelled, or physically runs.
        # Required journal sinks see it before preflight or scheduling can start any handler.
        dispatch(ToolRequested(invocation))
    for provider_index, tc in enumerate(tool_calls):
        name = getattr(tc, "name", "") or ""
        raw_args = tc.args if isinstance(getattr(tc, "args", None), dict) else {}
        invocation = invocations[provider_index]
        call_args = {k: v for k, v in raw_args.items()
                     if k not in ("note", CHILD_TOKEN_BUDGET_ARG, CHILD_CANCEL_SIGNAL_ARG,
                                  CHILD_INVOCATION_ID_ARG, CHILD_REQUEST_ORDINAL_ARG)}
        child_cancel = None
        if name in spawn_names:
            # Every physical child gets its own cancellation edge even when the parent has no signal. The
            # scheduler owns the delegation deadline; composition keeps parent Esc/Ctrl-C live as well.
            child_cancel = _ChildCancellationLease(signal)
            call_args[CHILD_INVOCATION_ID_ARG] = invocation.id
            call_args[CHILD_REQUEST_ORDINAL_ARG] = provider_index + 1
            call_args[CHILD_CANCEL_SIGNAL_ARG] = child_cancel
        entry = _entry_for(tools, name)
        purity = _purity_for(tools, name, call_args, entry)
        if purity is not ToolPurity.PURE_READ:
            wave_seen.clear()                  # dedup never crosses a mutation/unknown barrier

        can_dedup = bool(entry.deduplicable) if entry is not None else name in DEDUP_SAFE_TOOL_NAMES
        key = _dedup_key(name, call_args) if can_dedup and purity is ToolPurity.PURE_READ else None
        desc = {"invocation": invocation, "args": raw_args, "call_args": call_args,
                "preflight": ToolPreflight(), "entry": entry, "purity": purity,
                "deduplicable": can_dedup,
                "admission": None, "run_preflighted": None, "prepared_not_started": False,
                "child_cancel": child_cancel}
        descriptors.append(desc)
        if key is not None and key in wave_seen:
            dup_of[provider_index] = wave_seen[key]
            continue
        if key is not None:
            wave_seen[key] = provider_index
        if purity is not ToolPurity.PURE_READ:
            wave_seen.clear()

        def execute(d=desc):
            inv = d["invocation"]
            if d["preflight"].stop:
                _, _, text = _stop_parts(d["preflight"])
                raw = ToolText(text, status=ToolStatus.CANCELLED)
            else:
                try:
                    run_preflighted = d["run_preflighted"]
                    if d["admission"] is not None and callable(run_preflighted):
                        raw = run_preflighted(inv.name, d["call_args"], d["admission"])
                    else:
                        raw = tools.run(inv.name, d["call_args"])
                except (Exception, SystemExit) as error:
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
                # A proven preflight cancellation never entered the handler. Semantic effect factories
                # describe executed tool outcomes and may themselves fail; invoking one here could turn a
                # truthful CANCELLED into INDETERMINATE (or invent domain effects) for work that never ran.
                inv, raw, entry=(None if d["preflight"].stop else d["entry"]),
                default_effect_id=default_effect_id(inv),
            )

        def prepare(d=desc):
            """Resolve narrow safety/lifecycle preflight against every prior barrier's settled state."""
            nonlocal handoff_index
            inv = d["invocation"]
            if handoff_index is not None and inv.provider_index > handoff_index:
                preflight = ToolPreflight(
                    True,
                    "an earlier tool in this batch scheduled a workspace switch",
                    kind="lifecycle",
                )
            else:
                preflight = _safe_preflight(hooks, inv.name, d["args"])
            d["preflight"] = preflight
            if not preflight.stop:
                try:
                    host_preflight = getattr(tools, "preflight_run", None)
                    host_run_preflighted = getattr(tools, "run_preflighted", None)
                except (Exception, SystemExit):
                    host_preflight = None
                    host_run_preflighted = None
                supports_preflight = callable(host_preflight)
                supports_admitted_run = callable(host_run_preflighted)
                if supports_preflight != supports_admitted_run:
                    d["prepared_not_started"] = True
                    return finalize_tool_outcome(
                        inv,
                        ToolText(
                            "Error: tool host exposes an incomplete one-shot preflight protocol", ok=False,
                        ),
                        entry=None,
                        default_effect_id=default_effect_id(inv),
                    )
                if supports_preflight:
                    try:
                        admission, validation = host_preflight(inv.name, d["call_args"])
                    except (Exception, SystemExit) as error:
                        admission = None
                        validation = ToolText(f"Error: tool preflight failed ({error})", ok=False)
                    if validation is not None:
                        d["prepared_not_started"] = True
                        return finalize_tool_outcome(
                            inv, validation, entry=None,
                            default_effect_id=default_effect_id(inv),
                        )
                    d["admission"] = admission
                    d["run_preflighted"] = host_run_preflighted
                    # Registry replacement can occur at an earlier ordered barrier, after wave partitioning,
                    # deduplication, capabilities, and timeout semantics were frozen. Never combine that stale
                    # descriptor metadata with a different handler/effect factory: settle before start and let
                    # the next model step schedule against the current registry as one coherent snapshot.
                    if isinstance(admission, ToolAdmission):
                        if (d["entry"] is not None and admission.entry is not d["entry"]):
                            d["prepared_not_started"] = True
                            return finalize_tool_outcome(
                                inv,
                                ToolText(
                                    "Error: tool registration changed before execution started; "
                                    "retry the call against the current registry",
                                    ok=False,
                                ),
                                entry=None,
                                default_effect_id=default_effect_id(inv),
                            )
                        if ((admission.entry.purity is ToolPurity.PURE_READ)
                                != (d["purity"] is ToolPurity.PURE_READ)):
                            # A host without a descriptor-time registry entry can still return a registry
                            # admission. Ensure its inferred read-vs-barrier class agrees before using it.
                            d["prepared_not_started"] = True
                            return finalize_tool_outcome(
                                inv,
                                ToolText(
                                    "Error: tool admission changed execution class before start; retry",
                                    ok=False,
                                ),
                                entry=None,
                                default_effect_id=default_effect_id(inv),
                            )
                        if bool(admission.entry.deduplicable) != d["deduplicable"]:
                            d["prepared_not_started"] = True
                            return finalize_tool_outcome(
                                inv,
                                ToolText(
                                    "Error: tool admission changed deduplication metadata before start; retry",
                                    ok=False,
                                ),
                                entry=None,
                                default_effect_id=default_effect_id(inv),
                            )
                        d["entry"] = admission.entry
                if "workspace_handoff" in (
                    getattr(d["entry"], "capabilities", frozenset()) or frozenset()
                ):
                    handoff_index = inv.provider_index
                return None
            d["prepared_not_started"] = True
            _, _, text = _stop_parts(preflight)
            raw = ToolText(text, status=ToolStatus.CANCELLED)
            return finalize_tool_outcome(
                d["invocation"], raw, entry=None,
                default_effect_id=default_effect_id(d["invocation"]),
            )

        def announce(is_abandoned, inv=invocation, a=raw_args):
            # Record durable execution truth immediately before starting a handler. Preflight stops have no
            # start callback and therefore can never masquerade as physical starts. The scheduler-owned lease
            # is checked between lifecycle edges so a blocked journal crossing a deadline cannot later publish
            # ToolStarted or enter the handler after this batch has already settled.
            if is_abandoned():
                return
            # Record the attempt before crossing the opaque dispatcher boundary. If SIGINT lands inside a
            # required start journal, the handler is still provably uncalled, but one sink may already contain
            # a start row. The outer recovery path must therefore close that partial edge explicitly instead
            # of forgetting the invocation or pretending ordinary execution began.
            start_publication_attempt_ids.add(inv.id)
            dispatch(ToolExecutionStarted(inv))
            if is_abandoned():
                return
            dispatch(ToolStarted(inv.name, a, inv))
            started_ids.add(inv.id)

        child_cancel = desc.get("child_cancel")
        scheduled.append(ScheduledTool(
            invocation, purity, execute, on_start_guarded=announce,
            # Read-only children may overlap, but they finish by sealing artifacts and handing references to
            # the parent. A generic thread deadline must not abandon those lifecycle callbacks into a later
            # turn; the parent waits for settlement while still allowing sibling explorers to run in parallel.
            timeout_safe=invocation.name not in ("spawn_agent", "spawn_explore", "spawn_subagent"),
            prepare=prepare,
            on_queued=(
                (lambda reason, inv=invocation: dispatch(ToolQueued(
                    inv, reason, invocation_id=inv.id, request_ordinal=inv.provider_index + 1,
                )))
                if invocation.name in spawn_names else None
            ),
            request_cancel=(child_cancel.request if child_cancel is not None else None),
            cancel_grace=(_delegation_cancel_grace() if child_cancel is not None else 0.0),
        ))

    child_budget_reserved = 0
    outstanding_children = {
        task.invocation.provider_index
        for task in scheduled
        if task.invocation.name in spawn_names
    }

    def allocate_child_budgets(ready: list[ScheduledTool]) -> None:
        """Fairly reserve one batch-wide budget across current and future child waves."""
        nonlocal child_budget_reserved
        # Every task in the scheduler's current wave has completed preflight before this callback. Remove every
        # child conclusively settled there (hook stop *or* host/registry validation failure): none receives
        # ToolStarted, so retaining it would underallocate valid siblings against work proven not to start.
        for index in tuple(outstanding_children):
            if descriptors[index]["prepared_not_started"]:
                outstanding_children.discard(index)
        children = sorted(
            (task for task in ready if task.invocation.name in spawn_names
             and task.invocation.provider_index in outstanding_children),
            key=lambda task: task.invocation.provider_index,
        )
        if not children:
            return
        remaining = _safe_advisory(
            "remaining_token_budget", hooks.remaining_token_budget, default=None,
        )
        if remaining is None:
            for task in children:
                outstanding_children.discard(task.invocation.provider_index)
            return
        try:
            remaining = max(0, int(remaining))
        except (TypeError, ValueError, OverflowError):
            # Budget reporting is advisory. A malformed custom hook must not crash or block an otherwise
            # valid delegation batch; omit the child cap just as for a hook with no budget opinion.
            for task in children:
                outstanding_children.discard(task.invocation.provider_index)
            return
        # Divide against every still-outstanding child, not merely this wave. Otherwise a serialized writer or
        # an effect barrier gives the first child the entire allowance and forces every later child to cap=0.
        # Earlier shares get at most one remainder token; the running reservation still proves sum(caps) <= R.
        for task in children:
            available = max(0, remaining - child_budget_reserved)
            slots = max(1, len(outstanding_children))
            quotient, remainder = divmod(available, slots)
            share = quotient + bool(remainder)
            index = task.invocation.provider_index
            descriptors[index]["call_args"][CHILD_TOKEN_BUDGET_ARG] = share
            child_budget_reserved += share
            outstanding_children.discard(index)

    outcomes: list[ToolOutcome | None] = [None] * len(descriptors)
    rejection_published_ids: set[str] = set()
    settlement_published_ids: set[str] = set()
    result_published_ids: set[str] = set()

    def publish_rejection(out: ToolOutcome) -> None:
        invocation_id = out.invocation.id
        desc = descriptors[out.invocation.provider_index]
        if invocation_id in rejection_published_ids:
            return
        if desc["preflight"].stop:
            reason = str(desc["preflight"].reason or "cancelled")
            kind = str(getattr(desc["preflight"], "kind", "") or "lifecycle")
        elif desc["prepared_not_started"]:
            reason = str(out.text or "tool validation rejected the call before execution")
            kind = "steered" if out.status is ToolStatus.STEERED else "validation"
        else:
            return
        dispatch(ToolRejected(out.invocation, reason, out, kind=kind))
        rejection_published_ids.add(invocation_id)

    def publish_settlement(out: ToolOutcome) -> None:
        invocation_id = out.invocation.id
        if invocation_id in settlement_published_ids:
            return
        dispatch(ToolSettled(out))
        settlement_published_ids.add(invocation_id)

    def publish_result(out: ToolOutcome) -> None:
        invocation_id = out.invocation.id
        if invocation_id in result_published_ids:
            return
        dispatch(ToolResult(
            out.invocation.name, dict(out.invocation.args), out.text, out.failing,
            status=out.status.value, invocation_id=invocation_id, outcome=out,
        ))
        result_published_ids.add(invocation_id)

    def publish_edges(out: ToolOutcome) -> None:
        """Publish one terminal lifecycle in order; acknowledged edges are replay-safe by invocation ID."""
        publish_rejection(out)
        publish_settlement(out)
        publish_result(out)

    def recover_edges(out: ToolOutcome) -> None:
        """Best-effort completion that preserves the original user interrupt.

        A required sink can receive an edge and then be interrupted before returning. Retrying is therefore
        intentionally at-least-once; durable journals, reducers, and presentation projections all key these
        lifecycle facts by invocation ID. One repeatedly failing edge must not prevent settled siblings from
        receiving their remaining terminal facts.
        """
        for publisher in (publish_rejection, publish_settlement, publish_result):
            try:
                publisher(out)
            except BaseException:
                pass

    def publish(wave: list[ToolOutcome]) -> None:
        # Materialize EVERY physical result before running even the first transform/reducer callback. If
        # SIGINT lands while child 1 is being published, already-finished siblings 2..N remain recoverable as
        # their real outcomes instead of being fabricated as indeterminate by the interrupt synthesizer.
        for raw in wave:
            outcomes[raw.invocation.provider_index] = raw

        # Transform the complete wave before publishing any terminal edge. Status/effects cannot be rewritten
        # by presentation hooks; if a user interrupt crosses an advisory transform, recovery uses the known
        # canonical raw outcome for that call.
        transformed_wave = []
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
            transformed_wave.append(out)

        for out in transformed_wave:
            publish_edges(out)

    try:
        run_ordered(
            scheduled, timeout=_tool_timeout(), lifecycle_timeout=_delegation_timeout(),
            on_outcomes=publish,
            on_wave_ready=allocate_child_budgets,
            should_cancel=(signal.is_set if signal is not None else None),
        )
    except KeyboardInterrupt:
        # Finish every missing rejection/settlement/result edge for all known physical outcomes. The scheduler
        # may already have retried the completed wave; per-edge acknowledgements make this second recovery pass
        # a no-op in that case and exact-ID replay remains safe if a required sink was interrupted mid-call.
        for known in tuple(outcomes):
            if known is not None:
                recover_edges(known)

        # A signal inside ToolExecutionStarted/ToolStarted aborts _announce before the handler is entered, yet
        # an earlier required sink may already contain the start edge. Close that partial journal explicitly.
        # Once both start publications returned, the exact handler boundary is no longer observable here, so
        # retain the stronger execution uncertainty used for an interrupt raised from inside the handler.
        for desc in descriptors:
            inv = desc["invocation"]
            if inv.id not in start_publication_attempt_ids or outcomes[inv.provider_index] is not None:
                continue
            if inv.id in started_ids:
                text = "Error: tool execution was interrupted; final side effects are indeterminate"
            else:
                text = (
                    "Error: tool start publication was interrupted; the handler did not run, "
                    "but the durable start record may be partial"
                )
            interrupted = ToolOutcome(inv, ToolStatus.INDETERMINATE, text)
            outcomes[inv.provider_index] = interrupted
            recover_edges(interrupted)
        raise

    for index, source in dup_of.items():
        src = outcomes[source]
        if src is None:
            raise RuntimeError("deduplicated source call did not settle")
        inv = descriptors[index]["invocation"]
        descriptors[index]["preflight"] = descriptors[source]["preflight"]
        descriptors[index]["prepared_not_started"] = descriptors[source]["prepared_not_started"]
        outcomes[index] = ToolOutcome(inv, src.status, src.text, ())
        # Every provider invocation gets one durable logical outcome. The source call already applied the
        # semantic effects, so this compatibility reply is explicitly non-reducing.
        if descriptors[index]["preflight"].stop or descriptors[index]["prepared_not_started"]:
            if descriptors[index]["preflight"].stop:
                reason = str(descriptors[index]["preflight"].reason or "cancelled")
                kind = str(getattr(descriptors[index]["preflight"], "kind", "") or "lifecycle")
            else:
                reason = str(src.text or "tool validation rejected the call before execution")
                kind = "steered" if src.status is ToolStatus.STEERED else "validation"
            dispatch(ToolRejected(
                inv, reason, outcomes[index], kind=kind,
            ))
        dispatch(ToolSettled(outcomes[index], apply_effects=False))
        dispatch(ToolResult(
            inv.name, dict(inv.args), src.text, src.failing,
            status=src.status.value, invocation_id=inv.id, outcome=outcomes[index], apply_effects=False,
        ))

    legacy = []
    for desc, out in zip(descriptors, outcomes):
        row = out.as_legacy()
        preflight = desc["preflight"]
        if preflight.stop:
            kind, reason, _ = _stop_parts(preflight)
            row.update({
                "rejected_before_execution": kind == "catastrophic",
                "not_run_before_execution": kind == "lifecycle",
                "rejection_kind": kind,
                "rejection_reason": reason,
            })
        elif desc["prepared_not_started"]:
            kind = "steered" if out.status is ToolStatus.STEERED else "validation"
            row.update({
                "rejected_before_execution": True,
                "not_run_before_execution": True,
                "rejection_kind": kind,
                "rejection_reason": str(out.text or "tool validation rejected the call before execution"),
            })
        legacy.append(row)
    return 0, legacy


def _assistant_message(resp, *, step: int = 0, call_namespace: str = "") -> dict:
    """Reconstruct the OpenAI assistant message (with native tool_calls) for the accumulated transcript.
    ids are synthesized index-based when absent (matching run_tool_batch's scheme) so the assistant's
    tool_calls and the following tool messages reference the SAME ids."""
    msg: dict = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        # DeepSeek V4 thinking mode requires the exact assistant reasoning_content to accompany every
        # accumulated tool-call message. Omitting it makes the following tool-result request fail with 400.
        # Keep it provider-agnostic and optional: other adapters/fakes need not expose the field, and hidden
        # reasoning is replay-only data rather than user-facing transcript text.
        reasoning_content = getattr(resp, "reasoning_content", None)
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        physical_ids = _batch_tool_call_ids(resp.tool_calls, step, call_namespace)
        msg["tool_calls"] = [
            {"id": physical_ids[i], "type": "function",
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


_DELEGATION_TOOLS = frozenset({"spawn_agent", "spawn_explore", "spawn_subagent"})


def _delegation_batch_receipt(results: list[dict]) -> str:
    """Project one provider batch into a small model-visible child lifecycle receipt.

    The sealed receipt is the durable authority after the turn, but synthesis happens *before* that seal. A
    parent previously had to count several long child prose results itself and turned an observed 3-success /
    4-failure batch into "4 ok / 3 errors". Keep the result bodies for evidence, then append this deterministic
    control projection so status arithmetic and truncation/timeout labels never depend on model narration.
    """
    def bounded_field(value: object, limit: int = 120) -> str:
        return " ".join(str(value or "").split())[:limit]

    children = [row for row in results if str(row.get("name") or "") in _DELEGATION_TOOLS]
    if len(children) < 2:
        return ""

    counts = {status.value: 0 for status in ToolStatus}
    facts = []
    for row in children:
        status = str(row.get("status") or "").casefold()
        if status not in counts:
            status = ToolStatus.INDETERMINATE.value
        counts[status] += 1
        child_stop = ""
        child_cause = ""
        recovered_from: tuple[str, ...] = ()
        artifact_id = ""
        source_coverage_status = ""
        outcome = row.get("outcome")
        for effect in (getattr(outcome, "effects", ()) or ()):
            if getattr(effect, "kind", "") != "child_artifact":
                continue
            payload = getattr(effect, "payload", {}) or {}
            child_stop = str(payload.get("stop_reason") or payload.get("status") or "")
            child_cause = str(payload.get("stop_cause") or "")
            raw_recovery = payload.get("recovered_from") or ()
            if isinstance(raw_recovery, (list, tuple)):
                recovered_from = tuple(str(item) for item in raw_recovery if item)
            artifact_id = str(payload.get("artifact_id") or "")
            source_coverage_status = str(payload.get("source_coverage_status") or "")
            break
        parts = [f"tool_status={status}"]
        if child_stop:
            parts.append(f"child_stop={child_stop}")
        if child_cause:
            parts.append(f"cause={child_cause}")
        if recovered_from:
            parts.append("recovered_from=" + ",".join(recovered_from))
        if artifact_id:
            parts.append(f"sealed={artifact_id}")
        if source_coverage_status and source_coverage_status != "not_assessed":
            parts.append(f"source_coverage={source_coverage_status}")
        args = row.get("args") if isinstance(row.get("args"), dict) else {}
        work_item_id = bounded_field(args.get("work_item_id"))
        if work_item_id:
            parts.append(f"work_item={work_item_id}")
        raw_scope = args.get("scope") or ()
        if isinstance(raw_scope, (list, tuple)):
            scope = tuple(bounded_field(value) for value in raw_scope if bounded_field(value))
            if scope:
                shown = scope[:6]
                parts.append("declared_scope=" + ",".join(shown)
                             + (f",+{len(scope) - len(shown)}" if len(scope) > len(shown) else ""))
        facts.append(f"- {row.get('id') or '(unknown call)'}: " + "; ".join(parts))

    return (
        "# HOST DELEGATION RECEIPT (authoritative lifecycle facts; not a new user request)\n"
        f"requested={len(children)}; succeeded={counts['succeeded']}; failed={counts['failed']}; "
        f"steered={counts['steered']}; cancelled={counts['cancelled']}; "
        f"indeterminate={counts['indeterminate']}\n"
        + "\n".join(facts)
        + "\nCount only tool_status=succeeded as complete accepted child results. A failed child's sealed "
          "report may contain partial leads, but label them incomplete and never include them in the success "
          "count. Use the typed cause/recovered_from fields; do not infer a different cause from report prose. "
          "For a synthesiser, source_coverage=source_complete means only that every granted report was "
          "completely read and path-cited. It does not establish claim correctness, entailment, agreement, or "
          "independent verification. source_partial/source_unsupported remains orthogonal to operational success. "
          "This receipt establishes no parent-task coverage outside each child's declared_scope. The host has "
          "already projected each bound terminal child lifecycle into ACTIVE WORK: succeeded becomes ready for "
          "parent synthesis; a determinate failure becomes cancelled with its gap preserved. Do not call "
          "update_work merely to mirror those lifecycle facts. Continue only genuinely open partitions before "
          "claiming broad coverage."
    )


def _delegation_fan_in_bundle(results: list[dict], tools) -> str:
    """Reconstruct complete sealed child reports for the next parent synthesis call.

    The provider transcript is deliberately not the authority here: child tool results contain presentation
    excerpts and may later be compacted.  Join the typed child-artifact effects to the canonical ContextFS reports
    instead.  Loading is per-child fail-soft; the bundle always retains exact locators and lifecycle metadata.
    """
    from .fan_in import build_fan_in_bundle

    calls = []
    for result in results:
        if str(result.get("name") or "") not in _DELEGATION_TOOLS:
            continue
        row = {
            "id": str(result.get("id") or ""),
            "status": str(result.get("status") or "unknown"),
        }
        outcome = result.get("outcome")
        for effect in (getattr(outcome, "effects", ()) or ()):
            if getattr(effect, "kind", "") != "child_artifact":
                continue
            payload = getattr(effect, "payload", {}) or {}
            artifact_id = str(payload.get("artifact_id") or "")
            if artifact_id:
                row["child_artifact_id"] = artifact_id
            row["child_work_item_id"] = str(payload.get("work_item_id") or "")
            row["child_operational_status"] = str(
                payload.get("operational_status") or payload.get("status") or row["status"]
            )
            row["child_source_coverage_status"] = str(
                payload.get("source_coverage_status") or ""
            )
            if "explorer_evidence_status" in payload or "evidence_status" in payload:
                row["child_evidence_declared"] = True
                row["child_evidence_status"] = str(
                    payload.get("explorer_evidence_status") or payload.get("evidence_status") or ""
                )
            account = payload.get("explorer_evidence") or payload.get("evidence_account")
            if isinstance(account, dict):
                row["child_evidence_account"] = account
            break
        calls.append(row)

    if not any(str(row.get("child_artifact_id") or "") for row in calls):
        return ""

    def load(handle: str) -> str:
        router = getattr(tools, "_history_route", None)
        if not callable(router):
            raise FileNotFoundError("host exposes no ContextFS route")
        provider = router(handle)
        if provider is None:
            raise FileNotFoundError(f"no canonical artifact route for {handle}")
        canonicalize = getattr(tools, "_archive_handle", None)
        canonical = canonicalize(handle) if callable(canonicalize) else handle
        return str(provider.read_file(canonical))

    return build_fan_in_bundle(calls, report_loader=load, max_children=None).render()


def _prepared(hooks, msgs: list) -> list:
    """Pre-LLM-call hook seam (context injection, prompt-cache-safe): return the hook's rewrite, or
    `msgs` unchanged when it returns None. Note `is not None` (an empty-list rewrite is honored)."""
    prepared = _safe_advisory("prepare_messages", lambda: hooks.prepare_messages(msgs))
    return prepared if prepared is not None else msgs


def run_turn(*, build_slice, llm, tools, dispatch: Dispatcher, hooks: Hooks | None = None,
             max_steps: int = 40, signal=None, checkpoint=None, consolidate=None,
             turn_id: str = "", call_namespace: str = "", transport_activity=None,
             allow_park_closeout: bool = True) -> TurnResult:
    """One per-LOOP working-memory turn. The slice is the SEED, built ONCE; within the while(true) working
    memory ACCUMULATES as native assistant/tool messages — NO per-step rebuild, NO eviction. The LLM ends
    by not calling tools (Markov at the loop boundary; continuous within).

    Because the seed is built once, mid-turn hook→model communication rides the MESSAGE channel, never a
    slice mutation (which would never re-render): prepare_messages is applied per llm.complete, and a
    continue-hook's `feedback` (e.g. the Oracle's test failure) is appended as the model's next input.

    Every NON-clean exit — max_steps, token budget, catastrophic safety stop, overflow, abort, AND any
    UNEXPECTED internal error (a non-retryable llm failure, a throwing build_slice) — routes through ONE
    helper, _park: honest reason + exactly one TurnInterrupted (+ an ACCOUNTED closeout where another model
    call is affordable). A budget/safety stop PARKS — never `end_turn` (the caller checkpoints end_turn⇒done).
    ``allow_park_closeout=False`` delegates every such closeout to an outer lifecycle owner (used by the staged
    explorer navigator, whose separately budgeted full synthesis is the sole allowed follow-up model call).
    Overflow compacts the oldest WHOLE exchange; a seed that alone overflows parks soft."""
    hooks = hooks or Hooks()
    total = Usage()
    steps = 0
    messages: list = []      # defined BEFORE the seed build so _park's closure is safe even if it throws
    seed_len = 0
    seed_plan = None
    slice_built_dispatched = False
    model_attempts: dict[int, int] = {}
    repeated_observation = _ObservationRepeatAdvisory()
    failure_origin = ""
    response_only_next = False
    should_cancel = signal.is_set if signal is not None else None

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

    def _park(reason: str, msg: str | None, *, closeout: bool = True,
              error_origin: str = "", error_kind: str = "") -> TurnResult:
        """The ONE non-clean exit: an optional ACCOUNTED closeout, then exactly one TurnInterrupted."""
        if allow_park_closeout and closeout and msg is not None and messages:
            try:
                cmsgs = messages + [{"role": "user", "content": "# TURN IS ENDING — " + msg
                    + " Give your best answer/summary NOW (what you did, what you verified, what remains) from "
                    "what you already have; make NO edit/run tool call. If the request was ambiguous or you are "
                    "blocked, call ask_user with ONE concise question instead."}]
                close_usage = _final_answer(
                    llm, cmsgs, tools, dispatch, msg, seed_plan=seed_plan, seed_len=seed_len,
                    prepare=lambda candidate: _prepared(hooks, candidate),
                    on_attempt=_model_attempt_observer(max(1, steps)),
                    should_cancel=should_cancel, transport_activity=transport_activity,
                )
                _account(close_usage)
                _safe_advisory("record_step_usage.closeout", lambda: hooks.record_step_usage(close_usage))
                # Closeout is a real model call. Emit its typed usage before TurnInterrupted so metrics,
                # the episodic collector, and the active runtime persist the same total as TurnOutcome.
                dispatch(StepEnd(steps, Usage.from_value(close_usage).as_dict(), "closeout"))
            except Exception:  # noqa: BLE001
                pass
        dispatch(TurnInterrupted(reason, message=msg))
        # Preserve the typed stop detail for callers that own a higher-level lifecycle (notably one-shot
        # subagents). The event remains the durable/UI boundary; this field prevents wrappers from scraping
        # mutable Slice prose to distinguish a provider timeout from an ordinary stop.
        return TurnResult(
            reason, steps, total, message=msg, error_origin=error_origin, error_kind=error_kind,
        )

    # The ENTIRE turn (seed build + loop) is wrapped so EVERY non-clean exit routes through _park — even
    # ones we did not anticipate: a non-retryable llm error past with_retry, or a throwing build_slice /
    # retriever / probe. The session must NEVER die uncaught with no TurnInterrupted (Q + R).
    try:
        _safe_advisory("reset_for_turn", hooks.reset_for_turn)
        schemas = tools.schemas() if hasattr(tools, "schemas") else []  # stable per session → hoist once
        built_seed = build_slice()       # logical SEED PLAN — built ONCE
        prepared_schemas = _safe_advisory(
            "prepare_tool_schemas", lambda: hooks.prepare_tool_schemas(list(schemas)),
        )
        if prepared_schemas is not None:
            schemas = list(prepared_schemas)
        seed_plan = built_seed if isinstance(built_seed, SeedPlan) else None
        messages = list(built_seed)
        seed_len = len(messages)         # never compact below the seed
        reactive_seed_capacity = None  # unknown-window pressure learned from a real provider overflow
        while True:
            if signal is not None and signal.is_set():
                return _park("aborted", None, closeout=False)
            if steps >= max_steps:
                # Parent turns keep the generic best-effort closeout by default. A staged explorer has a
                # separately reserved, full-reasoning synthesis owner; its fast navigator opts out here so
                # budget exhaustion cannot mint a redundant hidden model call before that planned handoff.
                return _park("max_steps", BUDGET_EXHAUSTED("max_steps"))

            steps += 1
            call_schemas = [] if response_only_next else schemas
            response_only_next = False
            before = _safe_advisory("before_step", lambda: hooks.before_step(steps))
            if before and before.get("stop_turn"):
                # The built-in producer is the explicit token ceiling. Tool preflight stops belong to typed
                # outcomes; this resource-limit seam is not an execution-permission decision.
                return _park(
                    "token_budget", before.get("reason") or BUDGET_EXHAUSTED("token_budget"),
                    closeout=False,
                )
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
                                llm=llm, schemas=call_schemas,
                                prepare=lambda candidate: _prepared(hooks, candidate),
                                capacity_hint=reactive_seed_capacity,
                            )
                            messages[:seed_len] = projected
                        else:
                            _, provider_messages = _prepare_model_messages(
                                seed_plan=None, trajectory=[], messages=messages, llm=llm,
                                schemas=call_schemas, prepare=lambda candidate: _prepared(hooks, candidate),
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
                        failure_origin = "model_call"
                        resp = complete_model_call(
                            llm, provider_messages, call_schemas, dispatch=dispatch,
                            on_attempt=_model_attempt_observer(steps),
                            should_cancel=should_cancel, transport_activity=transport_activity,
                        )
                        failure_origin = ""
                        break
                    except ContextOverflow as overflow:
                        # A real provider rejection is stronger evidence than a configured/catalog estimate:
                        # stale metadata and multimodal accounting can overflow even a known window. Tighten
                        # one graded seed representation before deleting trajectory. Local preflight failures
                        # have already exhausted the projector and do not replay this reactive path.
                        provider_pressure = provider_call_started and not isinstance(overflow, PreflightOverflow)
                        failure_origin = ""  # handled pressure is no longer an outstanding provider failure
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
                                        seed_plan, messages[seed_len:], llm, call_schemas,
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
                # Usage observers are advisory extensions. The built-in BudgetHook is plain arithmetic; if a
                # third-party observer crashes, log in debug mode and keep ordinary work moving.
                budget_stop = bool((_safe_advisory("record_step_usage",
                                                   lambda: hooks.record_step_usage(step_usage.as_dict()),
                                                   default=None) or {}).get("stop_turn"))
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
                    # Lifecycle completion and response delivery are distinct.  Only procedures that explicitly
                    # declared a typed output envelope participate here; ordinary turns remain untouched. An
                    # exclusive lifecycle edge (notably workspace transport) owns this segment and defers the
                    # logical request's deliverable to the resumed target workspace.
                    candidate_check = None
                    if stop == "end_turn" and not (cont and cont.get("exclusive")):
                        candidate_check = _safe_advisory(
                            "assess_terminal_candidate",
                            lambda: hooks.assess_terminal_candidate(stop, candidate),
                        )
                    if candidate_check and candidate_check.get("continue"):
                        # A response nudge is optional presentation help, never a reason to replace the ordinary
                        # max-step boundary with an interruption. If no pass remains, publish the model's candidate.
                        if steps < max_steps:
                            response_only_next = bool(candidate_check.get("response_only"))
                            messages.append({
                                "role": "user",
                                "content": candidate_check.get("feedback") or "Answer the user's request now.",
                            })
                            continue
                    if stop in ("max_tokens", "filtered"):
                        # #11: a truncated (length) or content-filtered response is INCOMPLETE — park it as
                        # interrupted instead of sealing a partial answer as a clean turn. Surface any partial
                        # content explicitly as an update, never as the accepted terminal response.
                        if candidate:
                            dispatch(AssistantText(candidate, final=False))
                        return _park(stop, MAX_TOKENS_MSG if stop == "max_tokens" else FILTERED_MSG,
                                     closeout=False)
                    dispatch(AssistantText(
                        candidate or "Done — no summary to add.", final=True,
                        synthetic=not bool(candidate),
                    ))
                    dispatch(TurnEnd(stop, steps, total.as_dict()))   # the ONE clean-exit event
                    return TurnResult(stop, steps, total)

                # tool_use: accumulate the assistant turn (with tool_calls), run, accumulate the tool results
                if candidate:
                    dispatch(AssistantText(candidate, final=False))
                messages.append(_assistant_message(
                    resp, step=steps, call_namespace=call_namespace,
                ))
                tool_phase = True
                _, results = run_tool_batch(
                    resp.tool_calls, tools, dispatch, hooks, step=steps, turn_id=turn_id,
                    signal=signal, call_namespace=call_namespace,
                )
                tool_phase = False
                catastrophic_stop: str | None = None
                for r in results:
                    messages.append({"role": "tool", "tool_call_id": r["id"], "content": r["output"]})
                    if r.get("rejection_kind") == "catastrophic" and catastrophic_stop is None:
                        catastrophic_stop = str(r.get("rejection_reason") or r.get("output") or
                                                "Safety stop: potentially catastrophic command refused")
                delegation_receipt = _delegation_batch_receipt(results)
                fan_in_bundle = _delegation_fan_in_bundle(results, tools)
                if fan_in_bundle:
                    synthesis_packet = (
                        "# HOST FAN-IN SYNTHESIS SLICE (typed host state; not a new user request)\n"
                        "The delegation batch has settled. Use every complete child report below as attributed "
                        "testimony, preserve failed/partial coverage gaps, verify load-bearing claims against live "
                        "source when needed, and answer the CURRENT REQUEST directly. Bound child lifecycle has "
                        "already been projected into ACTIVE WORK; do not call update_work merely to mark these "
                        "children ready. If verification still needs tools, keep pre-tool prose to a brief factual "
                        "update. Only the later tool-free terminal response is delivered as the answer, so it must "
                        "contain the requested synthesis itself and must never point to a report 'above'.\n\n"
                        + (delegation_receipt + "\n\n" if delegation_receipt else "")
                        + fan_in_bundle
                    )
                    delegation_count = sum(
                        str(row.get("name") or "") in _DELEGATION_TOOLS for row in results
                    )
                    delegation_only = delegation_count == len(results)
                    if delegation_only and delegation_count >= 2:
                        # Clean map→reduce boundary: the durable ledger already owns the noisy exploration
                        # trajectory.  The parent synthesis call needs the current request plus complete reports,
                        # not several pages of spawn progress, excerpts, retries, and update chatter.
                        messages[seed_len:] = [{"role": "user", "content": synthesis_packet}]
                    else:
                        messages.append({"role": "user", "content": synthesis_packet})
                elif delegation_receipt:
                    # Tool messages preserve each report/excerpt. This host-derived control projection gives
                    # the next model step exact batch arithmetic before it synthesizes or reconciles anything;
                    # the canonical durable receipt will independently project the same typed outcomes at seal.
                    messages.append({"role": "user", "content": delegation_receipt})
                if repeated_observation.observe(results):
                    # Model-only liveness advice after every real result has been delivered. It is neither a
                    # rejection nor a stop condition, and deliberately emits no presentation event to the user.
                    messages.append({"role": "user", "content": _OBSERVATION_REPEAT_NUDGE})
                child_usage = _model_usage_from_tool_results(results)
                combined_usage = step_usage + child_usage
                child_budget_stop = False
                if child_usage.prompt_tokens or child_usage.completion_tokens or child_usage.cost_usd is not None:
                    _account(child_usage)
                    child_budget_stop = bool((_safe_advisory(
                        "record_step_usage.child",
                        lambda: hooks.record_step_usage(child_usage.as_dict()),
                        default=None,
                    ) or {}).get("stop_turn"))
                if any(r.get("status") == ToolStatus.INDETERMINATE.value for r in results):
                    dispatch(StepEnd(steps, combined_usage.as_dict(), "indeterminate"))
                    return _park(
                        "indeterminate",
                        "a tool outcome is indeterminate; this turn paused so later operations do not overtake "
                        "unknown effects. Re-observe the relevant state if it matters before relying on it",
                        closeout=False,
                    )
                if catastrophic_stop is not None:
                    dispatch(StepEnd(steps, combined_usage.as_dict(), "blocked"))
                    return _park("blocked", catastrophic_stop, closeout=False)
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
                        "a tool was interrupted after it started, so this turn paused rather than letting later "
                        "operations overtake an unknown outcome. Re-observe the relevant state if it matters",
                        closeout=False,
                    )
                return _park("aborted", None, closeout=False)
    except KeyboardInterrupt:
        # ctrl-C during SETUP (build_slice/schemas), before the step's own interrupt guard is in scope.
        return _park("aborted", None, closeout=False)
    except RetryCancelledError:
        return _park("aborted", None, closeout=False)
    except Exception as e:  # noqa: BLE001 — Q + R: any unexpected error PARKS, never crashes the session.
        import os as _os
        if _os.environ.get("SLICEAGENT_DEBUG_TRACE"):  # opt-in traceback so a parked 'error' is diagnosable
            import sys as _sys
            import traceback as _tb
            _tb.print_exc(file=_sys.stderr)
        # Carry the actual cause (type AND message, bounded) — the TurnInterrupted sink records it into
        # last_error so a parked/child 'error' is diagnosable from the seal (was: bare type name only, and
        # the child artifact then degraded even that to the literal string "error").
        cause = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        return _park(
            "error", f"an internal error ended the turn ({cause[:300]})", closeout=False,
            error_origin=failure_origin,
            error_kind=("indeterminate_model_call"
                        if isinstance(e, IndeterminateModelCallError) else ""),
        )
