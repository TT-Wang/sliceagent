"""UI-neutral turn progress state machine.

The runtime event stream is execution truth; this module reduces that stream into one
small presentation state that any terminal/UI can render.  It deliberately distinguishes
the model loop finishing from the host durably committing the turn.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Mapping

from .events import (
    ApiRetry,
    AssistantText,
    Event,
    ModelCallPrepared,
    SliceBuilt,
    SliceTightened,
    StepBegin,
    StepEnd,
    ToolResult,
    ToolStarted,
    TurnCommitted,
    TurnEnd,
    TurnInterrupted,
    TurnPhaseChanged,
    TurnStarted,
)
from .regions import MAX_PLAN_CHARS, MAX_PLAN_ITEMS


class ProgressPhase(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    THINKING = "thinking"
    WRITING = "writing"
    INSPECTING = "inspecting"
    EDITING = "editing"
    RUNNING = "running"
    DELEGATING = "delegating"
    WAITING = "waiting"
    RETRYING = "retrying"
    COMPACTING = "compacting"
    INTEGRATING = "integrating"
    VERIFYING = "verifying"
    FINALIZING = "finalizing"
    SAVING = "saving"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


_PHASE_LABEL = {
    ProgressPhase.IDLE: "Idle",
    ProgressPhase.PREPARING: "Preparing",
    ProgressPhase.THINKING: "Thinking",
    ProgressPhase.WRITING: "Writing",
    ProgressPhase.INSPECTING: "Inspecting",
    ProgressPhase.EDITING: "Editing",
    ProgressPhase.RUNNING: "Running",
    ProgressPhase.DELEGATING: "Delegating",
    ProgressPhase.WAITING: "Waiting",
    ProgressPhase.RETRYING: "Retrying",
    ProgressPhase.COMPACTING: "Fitting context",
    ProgressPhase.INTEGRATING: "Integrating",
    ProgressPhase.VERIFYING: "Checking completion",
    ProgressPhase.FINALIZING: "Finalizing",
    ProgressPhase.SAVING: "Saving",
    ProgressPhase.COMPLETE: "Turn finished",
    ProgressPhase.INTERRUPTED: "Interrupted",
    ProgressPhase.FAILED: "Save failed",
}

_READ_TOOLS = {
    "read_file", "list_files", "grep", "glob", "search_history", "code_review",
    "proc_poll", "proc_tail", "proc_wait", "terminal_read", "terminal_wait",
}
_EDIT_TOOLS = {"edit_file", "append_to_file", "str_replace"}
_COMMAND_TOOLS = {
    "run_command", "execute_code", "proc_start", "proc_kill",
    "terminal_start", "terminal_send", "terminal_close",
}


def _one_line(value: object, limit: int = 100) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _primary_arg(args: Mapping | None) -> str:
    args = args if isinstance(args, Mapping) else {}
    for key in ("path", "command", "pattern", "task", "goal", "question", "name", "ref"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _one_line(value, 64)
    value = next(
        (item for key, item in args.items()
         if key != "note" and isinstance(item, str) and item.strip()),
        "",
    )
    return _one_line(value, 64)


def tool_bucket(name: str) -> str:
    """Return the stable activity bucket used for phase selection and compact counts."""
    if name in _READ_TOOLS or name.startswith(("fetch_", "web_search")):
        return "read"
    if name in _EDIT_TOOLS:
        return "edit"
    if name in _COMMAND_TOOLS:
        return "cmd"
    if name.startswith(("spawn_", "delegate_")):
        return "agent"
    if name == "ask_user":
        return "wait"
    if name == "update_plan":
        return "plan"
    return "tool"


def _tool_detail(name: str, args: Mapping | None) -> tuple[ProgressPhase, str]:
    primary = _primary_arg(args)
    bucket = tool_bucket(name)
    if bucket == "read":
        phase, verb = ProgressPhase.INSPECTING, "reading"
    elif bucket == "edit":
        phase, verb = ProgressPhase.EDITING, "editing"
    elif bucket == "cmd":
        phase, verb = ProgressPhase.RUNNING, "running"
    elif bucket == "agent":
        phase, verb = ProgressPhase.DELEGATING, "delegating"
    elif bucket == "wait":
        phase, verb = ProgressPhase.WAITING, "waiting for input"
    elif bucket == "plan":
        phase, verb = ProgressPhase.INTEGRATING, "updating plan"
    else:
        phase, verb = ProgressPhase.RUNNING, name.replace("_", " ")
    return phase, f"{verb} {primary}".rstrip()


@dataclass(frozen=True)
class PlanProgress:
    total: int = 0
    done: int = 0
    current: str = ""
    current_index: int = 0


@dataclass(frozen=True)
class ActiveTool:
    invocation_id: str
    name: str
    detail: str
    bucket: str


@dataclass(frozen=True)
class ProgressSnapshot:
    phase: ProgressPhase = ProgressPhase.IDLE
    detail: str = ""
    task_title: str = ""
    task_id: str = ""
    model_pass: int = 0
    provider_attempt: int = 0
    plan: PlanProgress = field(default_factory=PlanProgress)
    active_tools: tuple[ActiveTool, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)
    started_at: float | None = None
    phase_started_at: float | None = None
    stop_reason: str = ""
    loop_finished: bool = False
    committed: bool = False
    turn_complete: bool = False
    last_milestone: str = ""

    @property
    def active(self) -> bool:
        return self.started_at is not None and self.phase not in {
            ProgressPhase.COMPLETE, ProgressPhase.INTERRUPTED, ProgressPhase.FAILED,
        }

    def status_text(self) -> str:
        """One semantic status string shared by every renderer."""
        phase = _PHASE_LABEL[self.phase]
        activity = phase + (f" — {self.detail}" if self.detail else "")
        if self.plan.total:
            if self.plan.current:
                pos = self.plan.current_index or min(self.plan.done + 1, self.plan.total)
                return f"{pos}/{self.plan.total} · {self.plan.current} — {activity}"
            return f"{self.plan.done}/{self.plan.total} · plan complete — {activity}"
        return activity


class TurnProgress:
    """Thread-safe reducer from runtime events to :class:`ProgressSnapshot`.

    ``await_commit`` is true for the CLI host, where ``TurnEnd`` means only that the
    model loop stopped.  A standalone loop adapter without a durable host may set it
    false and treat ``TurnEnd`` as its terminal boundary.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, await_commit: bool = True):
        self._clock = clock
        self._await_commit = await_commit
        self._lock = threading.RLock()
        self._legacy_tool_seq = 0
        self._active_tools: dict[str, ActiveTool] = {}
        self._counts: dict[str, int] = {}
        self._plan = PlanProgress()
        self._state = ProgressSnapshot()

    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> ProgressSnapshot:
        s = self._state
        return ProgressSnapshot(
            phase=s.phase, detail=s.detail, task_title=s.task_title, task_id=s.task_id,
            model_pass=s.model_pass, provider_attempt=s.provider_attempt, plan=self._plan,
            active_tools=tuple(self._active_tools.values()), counts=dict(self._counts),
            started_at=s.started_at, phase_started_at=s.phase_started_at,
            stop_reason=s.stop_reason, loop_finished=s.loop_finished, committed=s.committed,
            turn_complete=s.turn_complete, last_milestone=s.last_milestone,
        )

    def _replace(self, **changes) -> None:
        self._state = replace(
            self._state,
            **changes,
            plan=self._plan,
            active_tools=tuple(self._active_tools.values()),
            counts=dict(self._counts),
        )

    def _ensure_started(self, now: float) -> None:
        if self._state.started_at is None:
            self._replace(
                phase=ProgressPhase.PREPARING, detail="building task context",
                started_at=now, phase_started_at=now,
            )

    def _transition(self, phase: ProgressPhase, detail: str = "", *, now: float) -> None:
        phase_started = self._state.phase_started_at
        if phase != self._state.phase or detail != self._state.detail:
            phase_started = now
        self._replace(phase=phase, detail=_one_line(detail, 120), phase_started_at=phase_started)

    def _reset(self, event: TurnStarted, now: float) -> None:
        self._legacy_tool_seq = 0
        self._active_tools = {}
        self._counts = {}
        self._plan = PlanProgress()
        title = _one_line(event.task_title or event.request, 80)
        self._state = ProgressSnapshot(
            phase=ProgressPhase.PREPARING,
            detail="building task context",
            task_title=title,
            task_id=str(event.task_id or ""),
            started_at=now,
            phase_started_at=now,
        )
        if event.plan is not None:
            self._update_plan(event.plan)
            self._replace()

    def _update_plan(self, steps: object) -> None:
        if not isinstance(steps, list):
            return
        valid = []
        for item in steps[:MAX_PLAN_ITEMS]:
            if not isinstance(item, dict):
                continue
            step = " ".join(str(item.get("step", "")).split())[:MAX_PLAN_CHARS]
            status = str(item.get("status", "pending")).strip().lower()
            if status not in ("pending", "in_progress", "done"):
                status = "pending"
            if step:
                valid.append({"step": step, "status": status})
        done = sum(1 for item in valid if item.get("status") == "done")
        current_index, current = 0, ""
        for wanted in ("in_progress", "pending"):
            for index, item in enumerate(valid, 1):
                if item.get("status") == wanted:
                    current_index = index
                    current = _one_line(item.get("step", ""), 90)
                    break
            if current:
                break
        self._plan = PlanProgress(len(valid), done, current, current_index)

    def _remove_tool(self, event: ToolResult) -> None:
        invocation_id = str(event.invocation_id or "")
        if not invocation_id and event.outcome is not None:
            invocation_id = str(getattr(getattr(event.outcome, "invocation", None), "id", "") or "")
        if invocation_id and invocation_id in self._active_tools:
            self._active_tools.pop(invocation_id, None)
            return
        key = next((key for key, tool in reversed(tuple(self._active_tools.items()))
                    if tool.name == event.name), None)
        if key is not None:
            self._active_tools.pop(key, None)

    def reduce(self, event: Event) -> ProgressSnapshot:
        now = self._clock()
        with self._lock:
            if isinstance(event, TurnStarted):
                self._reset(event, now)
                return self._snapshot()

            # SliceBuilt is an observation, never a lifecycle reset.  The fallback start keeps
            # standalone run_turn consumers useful when they do not have a host TurnStarted event.
            if isinstance(event, SliceBuilt):
                self._ensure_started(now)
            elif isinstance(event, StepBegin):
                self._ensure_started(now)
                self._active_tools.clear()
                self._replace(model_pass=event.step, provider_attempt=0, loop_finished=False)
                self._transition(ProgressPhase.THINKING, "preparing model call", now=now)
            elif isinstance(event, ModelCallPrepared):
                self._ensure_started(now)
                self._replace(model_pass=event.step, provider_attempt=event.attempt)
                detail = "waiting for model"
                if event.attempt > 1:
                    detail += f" · attempt {event.attempt}"
                self._transition(ProgressPhase.THINKING, detail, now=now)
            elif isinstance(event, ApiRetry):
                self._ensure_started(now)
                next_attempt = max(1, event.attempt + 1)
                delay = f" in {event.delay_s:.1f}s" if event.delay_s > 0 else ""
                error = _one_line(event.error, 60)
                detail = f"model attempt {next_attempt}/{event.max_attempts}{delay}"
                if error:
                    detail += f" · {error}"
                self._transition(ProgressPhase.RETRYING, detail, now=now)
            elif isinstance(event, SliceTightened):
                self._ensure_started(now)
                detail = event.detail or event.reason.replace("_", " ")
                self._transition(ProgressPhase.COMPACTING, detail, now=now)
            elif isinstance(event, ToolStarted):
                self._ensure_started(now)
                invocation_id = str(getattr(event.invocation, "id", "") or "")
                if not invocation_id:
                    self._legacy_tool_seq += 1
                    invocation_id = f"legacy-{self._legacy_tool_seq}"
                phase, detail = _tool_detail(event.name, event.args)
                self._active_tools[invocation_id] = ActiveTool(
                    invocation_id, event.name, detail, tool_bucket(event.name),
                )
                if len(self._active_tools) > 1 and phase is ProgressPhase.INSPECTING:
                    detail = f"{len(self._active_tools)} reads in parallel"
                self._transition(phase, detail, now=now)
            elif isinstance(event, ToolResult):
                self._ensure_started(now)
                self._remove_tool(event)
                bucket = tool_bucket(event.name)
                if bucket in ("read", "edit", "cmd", "agent"):
                    self._counts[bucket] = self._counts.get(bucket, 0) + 1
                if event.failing:
                    self._counts["fail"] = self._counts.get("fail", 0) + 1
                event_args = event.args if isinstance(event.args, dict) else {}
                if event.name == "update_plan" and not event.failing:
                    self._update_plan(event_args.get("steps"))
                note = event_args.get("note")
                if not event.failing and isinstance(note, str) and note.strip():
                    self._replace(last_milestone=_one_line(note, 160))
                if self._active_tools:
                    latest = next(reversed(self._active_tools.values()))
                    phase, detail = _tool_detail(latest.name, {})
                    if len(self._active_tools) > 1 and phase is ProgressPhase.INSPECTING:
                        detail = f"{len(self._active_tools)} reads in parallel"
                    else:
                        detail = latest.detail or detail
                    self._transition(phase, detail, now=now)
                else:
                    self._transition(ProgressPhase.INTEGRATING, "integrating results", now=now)
            elif isinstance(event, StepEnd):
                self._ensure_started(now)
                if event.stop_reason == "tool_use":
                    self._transition(ProgressPhase.INTEGRATING, "integrating observations", now=now)
                else:
                    self._transition(ProgressPhase.FINALIZING, "model response ready", now=now)
            elif isinstance(event, AssistantText):
                # AssistantText alone is also used by cheap chitchat and host error paths; do not
                # fabricate or resurrect a task lifecycle when no turn is active.
                if self._snapshot().active and not self._state.loop_finished:
                    detail = "drafting response" if not event.final else "response ready"
                    self._transition(
                        ProgressPhase.WRITING if not event.final else ProgressPhase.FINALIZING,
                        detail,
                        now=now,
                    )
            elif isinstance(event, TurnPhaseChanged):
                self._ensure_started(now)
                phase = {
                    "preparing": ProgressPhase.PREPARING,
                    "checking_completion": ProgressPhase.VERIFYING,
                    "verifying": ProgressPhase.VERIFYING,
                    "saving": ProgressPhase.SAVING,
                    "finalizing": ProgressPhase.FINALIZING,
                }.get(event.phase, ProgressPhase.RUNNING)
                self._transition(phase, event.detail, now=now)
            elif isinstance(event, TurnEnd):
                self._ensure_started(now)
                self._active_tools.clear()
                self._replace(stop_reason=event.stop_reason, loop_finished=True)
                if self._await_commit:
                    self._transition(ProgressPhase.FINALIZING, "model loop finished", now=now)
                else:
                    self._replace(turn_complete=True)
                    self._transition(ProgressPhase.COMPLETE, "", now=now)
            elif isinstance(event, TurnInterrupted):
                self._ensure_started(now)
                self._active_tools.clear()
                self._replace(stop_reason=event.reason, loop_finished=True, turn_complete=False)
                self._transition(ProgressPhase.INTERRUPTED, event.message or event.reason, now=now)
            elif isinstance(event, TurnCommitted):
                self._ensure_started(now)
                self._active_tools.clear()
                receipt = event.receipt if isinstance(event.receipt, Mapping) else {}
                disposition = str(receipt.get("disposition") or "")
                sealed_stop = str(receipt.get("turn_status") or "")
                effective_stop = (
                    "indeterminate" if disposition == "indeterminate" else
                    sealed_stop or event.stop_reason or self._state.stop_reason
                )
                self._replace(
                    stop_reason=effective_stop,
                    committed=bool(event.ok),
                    turn_complete=bool(event.ok and effective_stop == "end_turn"),
                )
                if not event.ok:
                    self._transition(ProgressPhase.FAILED, event.detail or "checkpoint was not saved", now=now)
                elif effective_stop == "end_turn":
                    self._transition(ProgressPhase.COMPLETE, event.detail, now=now)
                else:
                    detail = event.detail or f"{effective_stop} state saved"
                    self._transition(ProgressPhase.INTERRUPTED, detail, now=now)

            return self._snapshot()

    def on_delta(self, kind: str, text: str) -> ProgressSnapshot:
        now = self._clock()
        with self._lock:
            if not text or not self._snapshot().active or self._state.loop_finished:
                return self._snapshot()
            if kind == "reasoning":
                self._transition(ProgressPhase.THINKING, "reasoning", now=now)
            elif kind == "content":
                self._transition(ProgressPhase.WRITING, "drafting response", now=now)
            return self._snapshot()

    def subagent_activity(self, text: str) -> ProgressSnapshot:
        now = self._clock()
        with self._lock:
            if not self._snapshot().active or self._state.loop_finished:
                return self._snapshot()
            self._transition(ProgressPhase.DELEGATING, _one_line(text, 110), now=now)
            return self._snapshot()


__all__ = [
    "ActiveTool", "PlanProgress", "ProgressPhase", "ProgressSnapshot", "TurnProgress", "tool_bucket",
]
