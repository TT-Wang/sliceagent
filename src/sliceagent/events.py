"""Event system: the loop's ONLY output path.

The core never prints or writes files — it dispatches events. The host composes a
dispatcher from sinks (slice-updater, durable log, CLI/TUI, SDK). Sink failures are
contained so a frontend can't break the loop.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
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

    ``attempt`` is 1-based within ``step`` and includes SDK-level retries as well as reactive
    re-projections. Unlike :class:`SliceBuilt`, this is an observation event: consumers must not
    interpret it as a new turn or semantic-step boundary.
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


@dataclass
class ToolRequested(Event):
    """One provider-requested logical invocation, before authorization or scheduling.

    This is deliberately separate from :class:`ToolExecutionStarted`: a rejected or
    deduplicated call still needs an auditable logical identity, but it never entered a
    tool handler.
    """

    invocation: "ToolInvocation"


@dataclass
class ToolRejected(Event):
    """A requested invocation was conclusively rejected before its handler ran."""

    invocation: "ToolInvocation"
    reason: str
    outcome: "ToolOutcome | None" = None


@dataclass
class ToolExecutionStarted(Event):
    """The pre-handler durability boundary for an authorized physical execution."""

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
    reason: str  # "aborted" | "max_steps" | "error"
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
        if isinstance(value, Mapping):
            return {str(key): detached(child) for key, child in value.items()}
        if isinstance(value, list):
            return [detached(child) for child in value]
        if isinstance(value, tuple):
            return tuple(detached(child) for child in value)
        return copy.deepcopy(value)

    def sink_view(event: Event) -> Event:
        # Completion truth is a compact mapping, but mappings and dataclass events are otherwise mutable.
        # Give every sink an independent view so an observer cannot rewrite a later observer's terminal claim.
        if isinstance(event, TurnCommitted):
            return replace(event, receipt=detached(event.receipt))
        return event

    def dispatch(event: Event) -> None:
        for sink in required:
            sink(sink_view(event))
        for sink in sinks:
            try:
                sink(sink_view(event))
            except Exception:
                pass  # a sink/listener failure must not affect the loop
    return dispatch
