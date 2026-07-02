"""Event system: the loop's ONLY output path.

The core never prints or writes files — it dispatches events. The host composes a
dispatcher from sinks (slice-updater, durable log, CLI/TUI, SDK). Sink failures are
contained so a frontend can't break the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Event:
    pass


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
    rendered: str  # the volatile user message this turn (for debugging/inspection)
    messages: list | None = None  # the FULL model-visible messages [system, user] (monitor/inspection)


@dataclass
class AssistantText(Event):
    content: str


@dataclass
class ToolStarted(Event):
    name: str
    args: dict


@dataclass
class ToolResult(Event):
    name: str
    args: dict
    output: str
    failing: bool


@dataclass
class ApiRetry(Event):
    attempt: int
    error: str


@dataclass
class SliceTightened(Event):
    level: int
    reason: str = "context_overflow"


@dataclass
class TurnEnd(Event):
    stop_reason: str
    steps: int
    usage: dict


@dataclass
class TurnInterrupted(Event):
    reason: str  # "aborted" | "max_steps" | "error"
    message: str | None = None


@dataclass
class LessonSaved(Event):
    title: str
    content: str  # the lesson mined into memem (write side of the memory loop)


Dispatcher = Callable[[Event], None]


def make_dispatcher(*sinks: Callable[[Event], None]) -> Dispatcher:
    def dispatch(event: Event) -> None:
        for sink in sinks:
            try:
                sink(event)
            except Exception:
                pass  # a sink/listener failure must not affect the loop
    return dispatch
