"""Event system: the loop's ONLY output path.

The core never prints or writes files — it dispatches events. The host composes a
dispatcher from sinks (slice-updater, durable log, CLI/TUI, SDK). Sink failures are
contained so a frontend can't break the loop.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .execution import ToolEffect, ToolInvocation, ToolOutcome


@dataclass
class Event:
    pass


@dataclass
class TurnStarted(Event):
    """Host boundary emitted after task routing and before slice construction."""

    request: str
    task_title: str = ""
    task_id: str = ""
    plan: list | None = None
    # Physical turn/segment identity.  Presentation callbacks from concurrent child
    # workers use it to prove ownership; task_id alone is intentionally longer-lived.
    turn_id: str = ""


@dataclass
class TurnPhaseChanged(Event):
    """Long host/core phase that cannot be derived from a model or tool event."""

    phase: str
    detail: str = ""


@dataclass
class StepBegin(Event):
    step: int


@dataclass
class StepEnd(Event):
    step: int
    usage: dict
    stop_reason: str


@dataclass
class SliceBuilt(Event):
    rendered: str  # the initial volatile user message; also the once-per-turn lifecycle boundary
    messages: list | None = None  # initial prepared request; later physical calls use ModelCallPrepared


@dataclass
class ModelCallPrepared(Event):
    """Exact prepared request immediately before one physical provider attempt.

    ``attempt`` is 1-based within ``step``. SliceAgent is the sole retry owner (the provider SDK is
    configured one-shot), so every transport attempt passes this boundary; reactive re-projections do too.
    Unlike :class:`SliceBuilt`, this is an observation event: consumers must not interpret it as a new turn
    or semantic-step boundary.
    """

    step: int
    attempt: int
    messages: list
    pressure: str = "unknown"
    preflight_mode: str = ""


@dataclass
class AssistantText(Event):
    content: str
    # Tool-using responses may contain explanatory text before execution.  Only a terminal
    # response is held behind completion verification and durable commit by interactive UIs.
    final: bool = True
    # True only for deterministic host wording emitted when the provider supplied no assistant text.  Sinks
    # may render it, but evidence/report collectors must not mistake it for model-authored testimony.
    synthetic: bool = False


@dataclass
class ToolRequested(Event):
    """One provider-requested logical invocation, before preflight or scheduling.

    This is deliberately separate from :class:`ToolExecutionStarted`: a rejected or
    deduplicated call still needs an auditable logical identity, but it never entered a
    tool handler.
    """

    invocation: "ToolInvocation"


@dataclass
class ToolRejected(Event):
    """A requested invocation was conclusively stopped before its handler ran.

    ``lifecycle`` is a neutral not-run transition; ``catastrophic`` remains an explicit safety refusal.
    """

    invocation: "ToolInvocation"
    reason: str
    outcome: "ToolOutcome | None" = None
    kind: str = "rejected"


@dataclass
class ToolExecutionStarted(Event):
    """The pre-handler durability boundary for a physical execution."""

    invocation: "ToolInvocation"


@dataclass
class ToolStarted(Event):
    """Compatibility projection of :class:`ToolExecutionStarted` for existing sinks."""

    name: str
    args: dict
    invocation: "ToolInvocation | None" = None


@dataclass
class ToolSettled(Event):
    """A logical invocation obtained one typed terminal outcome.

    Settlement does not mean semantic effects were applied. The authoritative reducer
    records those only after it succeeds.
    """

    outcome: "ToolOutcome"
    apply_effects: bool = True


@dataclass
class ToolEffectApplied(Event):
    """One outcome effect was accepted by the authoritative state reducer."""

    invocation_id: str
    effect: "ToolEffect"


@dataclass
class ToolResult(Event):
    name: str
    args: dict
    output: str
    failing: bool
    status: str | None = None
    invocation_id: str = ""
    outcome: "ToolOutcome | None" = None
    apply_effects: bool = True  # false for a logical dedup reply whose source outcome was already reduced


@dataclass
class ToolQueued(Event):
    """A requested invocation is admitted but waiting for a scheduler slot.

    This is presentation truth, not an execution edge: the handler has not started and may still settle as
    cancelled.  ``ToolExecutionStarted``/``ToolStarted`` remain the only authoritative physical-start facts.
    """

    invocation: "ToolInvocation"
    reason: str = "waiting for scheduler slot"
    invocation_id: str = ""
    request_ordinal: int = 0


@dataclass
class SubagentProgress(Event):
    """Turn-scoped child activity sent to presentation adapters.

    This is not parent context and is never model-visible.  Stable identities and a
    monotonic sequence prevent concurrent children (or a late callback from an older
    turn) from overwriting one another's terminal state.
    """

    agent_id: str
    parent_turn_id: str
    launch_ordinal: int = 0
    kind: str = ""
    name: str = ""
    depth: int = 1
    # Live phases are execution facts, not prose labels. ``running`` remains a compatibility fallback for older
    # emitters; core children use the more specific awaiting_model/model_active/reasoning/writing/running_tool/
    # retry_wait/settling states so the matrix never claims "waiting for model" after a tool, delta, or final
    # child-loop interruption has already arrived. ``settling`` is nonterminal until the outer ToolResult.
    phase: str = "running"
    detail: str = ""
    tool_count: int = 0
    sequence: int = 0
    session_id: str = ""
    # Immediate child-artifact parent for nested delegation.  parent_turn_id remains
    # the root physical turn owner used for stale-callback rejection.
    parent_agent_id: str = ""
    # Exact physical spawn identity.  Artifact/agent ids are durable result identities and may be minted only
    # after execution begins; invocation identity exists at request time and disambiguates identical siblings.
    invocation_id: str = ""
    request_ordinal: int = 0
    # Stable objective is separate from ``detail``, which is intentionally overwritten by live activity.
    objective: str = ""
    # Typed activity metadata. These fields are optional so third-party/older structured callbacks continue to
    # work; renderers must never parse ``detail`` to invent attempt, tool, timeout, or partial-result facts.
    attempt: int = 0
    max_attempts: int = 0
    retry_delay_s: float = 0.0
    tool_name: str = ""
    terminal_reason: str = ""
    partial: bool = False


@dataclass
class ApiRetry(Event):
    attempt: int
    error: str
    delay_s: float = 0.0
    max_attempts: int = 3


@dataclass
class SliceTightened(Event):
    level: int
    reason: str = "context_overflow"
    detail: str = ""


@dataclass
class TurnEnd(Event):
    stop_reason: str
    steps: int
    usage: dict


@dataclass
class TurnCommitted(Event):
    """Host durability boundary; ``TurnEnd`` alone is never a saved/completed claim."""

    ok: bool
    stop_reason: str
    artifact_id: str = ""
    detail: str = ""
    # Constant-size lifecycle totals projected from the receipt inside ``artifact_id``. Frontends use this
    # sealed truth instead of reconstructing completion semantics from lossy live counters.
    receipt: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Make the sealed status authoritative over the loop's pre-seal stop reason.

        A journal-completeness check may strengthen ``end_turn`` to ``indeterminate`` while sealing. Every
        consumer, including ones that do not understand receipt counts, must see that effective stop.
        """
        if not isinstance(self.receipt, Mapping):
            return
        original = str(self.stop_reason or "")
        disposition = str(self.receipt.get("disposition") or "")
        sealed = str(self.receipt.get("turn_status") or "")
        effective = "indeterminate" if disposition == "indeterminate" else sealed
        if effective:
            self.stop_reason = effective
        if self.ok and effective and effective != original and (not self.detail or self.detail == "checkpoint saved"):
            self.detail = "checkpoint saved" if effective == "end_turn" else f"{effective} state saved"


@dataclass
class TurnInterrupted(Event):
    reason: str  # aborted | max_steps | token_budget | blocked(catastrophic only) | overflow | error | indeterminate
    message: str | None = None


@dataclass
class LessonSaved(Event):
    title: str
    content: str  # the lesson mined into memem (write side of the memory loop)


Dispatcher = Callable[[Event], None]


def make_dispatcher(*sinks: Callable[[Event], None],
                    required: tuple[Callable[[Event], None], ...] = ()) -> Dispatcher:
    """Compose required reducers/journals with best-effort observers.

    Required sinks run first and propagate failure into the turn. UI, metrics and logging observers remain
    isolated. This prevents authoritative state reduction or crash journaling from being silently skipped by
    the same blanket exception policy used for presentation.
    """
    def detached(value):
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{
                field.name: detached(getattr(value, field.name)) for field in fields(value)
            })
        if isinstance(value, Mapping):
            return {str(key): detached(child) for key, child in value.items()}
        if isinstance(value, list):
            return [detached(child) for child in value]
        if isinstance(value, tuple):
            return tuple(detached(child) for child in value)
        return copy.deepcopy(value)

    def sink_view(event: Event) -> Event:
        # Events, invocation args, outcome payloads, usage, plans, and prepared messages all carry mutable
        # containers. Give every required sink and observer a detached dataclass graph so one consumer cannot
        # rewrite a later consumer's evidence—or the caller's original event. This is intentionally recursive
        # only over the event payload; callable/runtime owners are never event fields.
        return detached(event)

    def dispatch(event: Event) -> None:
        for sink in required:
            sink(sink_view(event))
        for sink in sinks:
            try:
                sink(sink_view(event))
            except Exception:
                pass  # a sink/listener failure must not affect the loop

    def bind_dispatch() -> Dispatcher:
        """Freeze any dynamic sink routers to the active turn they currently address.

        Tool workers may outlive a cancellation/deadline. A router that simply dereferences the application's
        current workspace would let a late lifecycle callback journal into the next turn. Required/runtime
        routers expose ``bind_dispatch`` to return an epoch-pinned sink; ordinary observers are already stable.
        """
        def bind_one(sink):
            binder = getattr(sink, "bind_dispatch", None)
            return binder() if callable(binder) else sink

        return make_dispatcher(
            *(bind_one(sink) for sink in sinks),
            required=tuple(bind_one(sink) for sink in required),
        )

    dispatch.bind_dispatch = bind_dispatch  # type: ignore[attr-defined]
    return dispatch
