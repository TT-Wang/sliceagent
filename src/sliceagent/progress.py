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

from .execution import coerce_tool_status
from .events import (
    ApiRetry,
    AssistantText,
    Event,
    ModelCallPrepared,
    SliceBuilt,
    SliceTightened,
    StepBegin,
    StepEnd,
    SubagentProgress,
    ToolQueued,
    ToolResult,
    ToolStarted,
    TurnCommitted,
    TurnEnd,
    TurnInterrupted,
    TurnPhaseChanged,
    TurnStarted,
)
from .regions import MAX_PLAN_CHARS, MAX_PLAN_ITEMS
from .tui_projection import child_incompleteness_label, normalized_report_completion


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
    "proc_poll", "proc_tail", "terminal_read",
}
_EDIT_TOOLS = {"edit_file", "append_to_file", "str_replace"}
_COMMAND_TOOLS = {
    "run_command", "execute_code", "proc_start", "proc_wait", "proc_kill",
    "terminal_open", "terminal_send", "terminal_wait", "terminal_close",
}


def _one_line(value: object, limit: int = 100) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return max(0, default)


def _nonnegative_float(value: object, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError, OverflowError):
        return max(0.0, default)


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
    if name in {"proc_wait", "terminal_wait"}:
        phase, verb = ProgressPhase.WAITING, "waiting for"
    elif bucket == "read":
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
class ActiveSubagent:
    agent_id: str
    parent_turn_id: str
    parent_agent_id: str
    invocation_id: str
    request_ordinal: int
    launch_ordinal: int
    kind: str
    name: str
    objective: str
    depth: int
    phase: str
    detail: str
    tool_count: int
    sequence: int
    updated_order: int
    started_at: float | None
    updated_at: float
    queued_at: float | None = None
    finished_at: float | None = None
    source_coverage_status: str = "not_assessed"
    evidence_status: str = "not_assessed"
    evidence_account: tuple[tuple[str, int], ...] = ()
    attempt: int = 0
    max_attempts: int = 0
    retry_delay_s: float = 0.0
    tool_name: str = ""
    terminal_reason: str = ""
    partial: bool = False
    report_completion: str = "unknown"


_ACTIVE_AGENT_PHASES = frozenset({
    "queued", "starting", "awaiting_model", "model_active", "reasoning", "writing", "running_tool",
    "retry_wait", "settling",
    "running",  # compatibility for older structured callbacks
})
_TERMINAL_AGENT_PHASES = frozenset({
    "report_ready", "completed", "steered", "failed", "timed_out", "cancelled", "indeterminate",
})
_AGENT_PHASES = _ACTIVE_AGENT_PHASES | _TERMINAL_AGENT_PHASES
# Within one physical model response these are refinements of the same fact, so a
# later callback may only move forward.  Some provider SDKs can deliver a buffered
# reasoning/activity callback after visible content; accepting it would make the
# matrix lie by rewinding ``writing`` back to ``reasoning``.  A new pass/attempt is
# still free to restart the sequence because StepBegin/ApiRetry first moves the row
# through a non-model phase (starting/retry_wait).
_MODEL_AGENT_PHASE_ORDER = {
    "awaiting_model": 0,
    "model_active": 1,
    "reasoning": 2,
    "writing": 3,
}
_TIMEOUT_CAUSES = frozenset({
    "provider_timeout", "model_timeout", "tool_timeout", "read_timeout", "delegation_timeout",
})
_EVIDENCE_STATUSES = frozenset({
    "not_assessed", "none", "navigation_only", "content_partial", "content_retained",
})
_EVIDENCE_COUNT_FIELDS = (
    "scope_path_count", "navigation_success_count", "content_success_count",
    "gap_observation_count", "retained_navigation_view_count",
    "retained_content_view_count", "omitted_navigation_view_count",
    "omitted_content_view_count", "truncated_content_view_count",
)


def _timeout_cause(value: object) -> bool:
    cause = str(value or "").strip().casefold()
    return cause in _TIMEOUT_CAUSES or cause.endswith("_timeout")


def _evidence_status(value: object) -> str:
    status = str(value or "not_assessed").strip().casefold().replace("-", "_").replace(" ", "_")
    return status if status in _EVIDENCE_STATUSES else "not_assessed"


def _evidence_account(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        return ()
    counts = []
    for key in _EVIDENCE_COUNT_FIELDS:
        raw = value.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool):
            counts.append((key, max(0, min(raw, 1_000_000))))
    return tuple(counts)


def _agent_phase_detail(
    phase: str,
    *,
    detail: object = "",
    attempt: int = 0,
    max_attempts: int = 0,
    retry_delay_s: float = 0.0,
    tool_name: object = "",
    terminal_reason: object = "",
    partial: bool = False,
    report_completion: object = "unknown",
) -> str:
    """Build bounded display text from typed child activity without inferring state from prose."""
    if phase == "report_ready":
        return "report ready"
    if phase == "completed":
        completion = normalized_report_completion(report_completion)
        return "completed · no report" if completion == "absent" else "completed · report status unknown"
    if phase == "settling":
        return "finalizing outcome"
    explicit = _one_line(detail, 80)
    if explicit:
        return explicit
    attempt_text = ""
    if attempt:
        attempt_text = f"attempt {attempt}/{max_attempts}" if max_attempts else f"attempt {attempt}"
    if phase == "awaiting_model":
        return "awaiting model" + (f" · {attempt_text}" if attempt_text else "")
    if phase == "model_active":
        return "model responding" + (f" · {attempt_text}" if attempt_text else "")
    if phase == "reasoning":
        return "reasoning"
    if phase == "writing":
        return "writing report"
    if phase == "running_tool":
        return "running " + (_one_line(tool_name, 50) or "tool")
    if phase == "retry_wait":
        parts = ["retry wait"]
        if attempt_text:
            parts.append(attempt_text)
        if retry_delay_s > 0:
            parts.append(f"{retry_delay_s:.1f}s")
        return " · ".join(parts)
    if phase in _TERMINAL_AGENT_PHASES:
        reason = _one_line(terminal_reason, 60) or phase.replace("_", " ")
        qualifier = child_incompleteness_label(report_completion, partial)
        return reason + (f" · {qualifier}" if qualifier else "")
    return phase.replace("_", " ")


@dataclass(frozen=True)
class ProgressSnapshot:
    phase: ProgressPhase = ProgressPhase.IDLE
    detail: str = ""
    task_title: str = ""
    task_id: str = ""
    turn_id: str = ""
    model_pass: int = 0
    provider_attempt: int = 0
    plan: PlanProgress = field(default_factory=PlanProgress)
    active_tools: tuple[ActiveTool, ...] = ()
    subagents: tuple[ActiveSubagent, ...] = ()
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
        self._subagents: dict[str, ActiveSubagent] = {}
        self._retired_subagent_ids: set[str] = set()
        self._retired_subagent_invocation_ids: set[tuple[str, str]] = set()
        self._settled_invocation_ids: set[str] = set()
        self._agent_outcomes: dict[str, int] = {}
        self._agent_unlinked_outcomes: dict[str, int] = {}
        self._subagent_update_seq = 0
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
            turn_id=s.turn_id,
            model_pass=s.model_pass, provider_attempt=s.provider_attempt, plan=self._plan,
            active_tools=tuple(self._active_tools.values()), subagents=tuple(self._subagents.values()),
            counts=dict(self._counts),
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
            subagents=tuple(self._subagents.values()),
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
        self._subagents = {}
        self._retired_subagent_ids = set()
        self._retired_subagent_invocation_ids = set()
        self._settled_invocation_ids = set()
        self._agent_outcomes = {}
        self._agent_unlinked_outcomes = {}
        self._subagent_update_seq = 0
        self._counts = {}
        self._plan = PlanProgress()
        title = _one_line(event.task_title or event.request, 80)
        self._state = ProgressSnapshot(
            phase=ProgressPhase.PREPARING,
            detail="building task context",
            task_title=title,
            task_id=str(event.task_id or ""),
            turn_id=str(event.turn_id or ""),
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
        if invocation_id:
            # A typed ID is authoritative.  Falling back by tool name when that ID
            # never started can retire an unrelated concurrent sibling.
            self._active_tools.pop(invocation_id, None)
            return
        key = next((key for key, tool in reversed(tuple(self._active_tools.items()))
                    if tool.name == event.name), None)
        if key is not None:
            self._active_tools.pop(key, None)

    def _agent_activity_detail(self) -> str:
        """Describe concurrent child work from identities, never last-writer-wins prose."""
        values = tuple(self._subagents.values())

        def identity(item: ActiveSubagent) -> str:
            return item.name or (f"#{item.launch_ordinal}" if item.launch_ordinal else item.kind or "agent")

        def breadcrumb(item: ActiveSubagent) -> str:
            path = [identity(item)]
            seen = {item.agent_id}
            parent_id = item.parent_agent_id
            while parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent = self._subagents.get(parent_id)
                if parent is None:
                    break
                path.append(identity(parent))
                parent_id = parent.parent_agent_id
            return " → ".join(reversed(path))

        queued = [item for item in values if item.phase == "queued"]
        running = [item for item in values if item.phase in _ACTIVE_AGENT_PHASES - {"queued"}]
        # Structured terminal updates precede the parent ToolResult.  ``max`` keeps
        # those two observations from double-counting the same child when the
        # scheduler publishes a completed wave.
        def settled_count(status: str, phase: str) -> int:
            # Exact child outcomes have retired their terminal projection.  An
            # effectless but physically-started result may overlap a retained
            # terminal projection, hence max for only that unlinked subset.
            terminal = sum(item.phase == phase for item in values)
            return self._agent_outcomes.get(status, 0) + max(
                self._agent_unlinked_outcomes.get(status, 0), terminal,
            )

        ready = settled_count("succeeded", "report_ready")
        completed = settled_count("completed", "completed")
        steered = settled_count("steered", "steered")
        failed = settled_count("failed", "failed")
        timed_out = settled_count("timed_out", "timed_out")
        cancelled = settled_count("cancelled", "cancelled")
        indeterminate = settled_count("indeterminate", "indeterminate")
        parent_agents = sum(tool.bucket == "agent" for tool in self._active_tools.values())
        top_level_known = sum(not item.parent_agent_id for item in values)
        # Known top-level children account for their corresponding parent spawn;
        # nested children add real running work but no extra parent invocation.
        active = len(running) + max(0, parent_agents - top_level_known)
        parts = []
        if active:
            parts.append(f"{active} agent{'s' if active != 1 else ''} running")
        if queued:
            parts.append(f"{len(queued)} agent{'s' if len(queued) != 1 else ''} queued")
        if ready:
            parts.append(f"{ready} report{'s' if ready != 1 else ''} ready")
        if completed:
            parts.append(f"{completed} completed without report")
        if steered:
            parts.append(f"{steered} steered")
        if failed:
            parts.append(f"{failed} failed")
        if timed_out:
            parts.append(f"{timed_out} timed out")
        if cancelled:
            parts.append(f"{cancelled} cancelled")
        if indeterminate:
            parts.append(f"{indeterminate} state unknown")
        adverse = max(
            (item for item in values if item.phase in {"failed", "timed_out", "cancelled", "indeterminate"}),
            key=lambda item: item.updated_order, default=None,
        )
        latest = max((*running, *queued), key=lambda item: item.updated_order, default=None)
        highlighted = adverse or latest
        if highlighted is not None:
            action = highlighted.detail or highlighted.phase.replace("_", " ")
            if action:
                parts.append(f"{breadcrumb(highlighted)} · {action}")
        return " · ".join(parts) or "integrating delegated reports"

    @staticmethod
    def _child_payload(event: ToolResult) -> Mapping[str, object]:
        merged: dict[str, object] = {}
        for effect in (getattr(getattr(event, "outcome", None), "effects", ()) or ()):
            if str(getattr(effect, "kind", "") or "") not in {"child_outcome", "child_artifact"}:
                continue
            payload = getattr(effect, "payload", None)
            if isinstance(payload, Mapping):
                merged.update(payload)
        return merged

    @classmethod
    def _child_artifact_id(cls, event: ToolResult) -> str:
        return str(cls._child_payload(event).get("artifact_id") or "")

    @classmethod
    def _child_source_coverage(cls, event: ToolResult) -> str:
        payload = cls._child_payload(event)
        status = str(
            payload.get("source_coverage_status") or payload.get("epistemic_status") or "not_assessed"
        ).strip().casefold()
        status = {
            "grounded": "source_complete", "partial": "source_partial",
            "unsupported": "source_unsupported",
        }.get(status, status)
        if status in {"source_complete", "source_partial", "source_unsupported", "not_assessed"}:
            return status
        return "not_assessed"

    def _settle_subagent(self, event: ToolResult, status: str, now: float) -> bool:
        """Bind an authoritative result to one matrix row and tombstone late callbacks.

        Terminal rows remain visible through ``StepEnd``.  Deleting them as soon as the
        scheduler published a ToolResult made a successful child disappear before the
        settled batch could replace the transient live surface.
        """
        payload = self._child_payload(event)
        stop_cause = str(payload.get("stop_cause") or "").strip().casefold()
        stop_reason = str(payload.get("stop_reason") or "").strip().casefold()
        report_completion = normalized_report_completion(payload.get("report_completion"))
        terminal_phase = {
            "succeeded": (
                "report_ready" if report_completion in {"complete", "partial"} else "completed"
            ), "steered": "steered",
            "failed": "failed", "cancelled": "cancelled",
            "indeterminate": "indeterminate",
        }.get(status, "indeterminate")
        # Timeout is a child terminal cause, not a sixth tool status. Only typed outcome fields may refine a
        # FAILED row into timed_out; prose such as "timeout handling looks correct" must never change state.
        if status == "failed" and (_timeout_cause(stop_cause) or _timeout_cause(stop_reason)):
            terminal_phase = "timed_out"
        raw_detail = _one_line(getattr(event, "output", ""), 100)
        terminal_reason = _one_line(stop_cause or stop_reason, 80).replace("_", " ")
        child_id = self._child_artifact_id(event)
        explicit_partial = payload.get("partial")
        partial = bool(explicit_partial) if isinstance(explicit_partial, bool) else False
        terminal_detail = (
            "report ready" if terminal_phase == "report_ready"
            else _agent_phase_detail(
                terminal_phase, detail="", terminal_reason=terminal_reason or raw_detail, partial=partial,
                report_completion=report_completion,
            )
        )
        source_coverage_status = self._child_source_coverage(event)
        evidence_status = _evidence_status(
            payload.get("explorer_evidence_status")
            if "explorer_evidence_status" in payload else payload.get("evidence_status")
        )
        evidence_account = _evidence_account(
            payload.get("explorer_evidence")
            if "explorer_evidence" in payload else payload.get("evidence_account")
        )

        def settle_tree(root_id: str) -> bool:
            matched = False
            pending = [root_id]
            visited: set[str] = set()
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(
                    child.agent_id for child in tuple(self._subagents.values())
                    if child.parent_agent_id == current
                )
                item = self._subagents.get(current)
                if item is not None:
                    matched = True
                    self._subagent_update_seq += 1
                    row_partial = partial or item.partial
                    row_report_completion = (
                        report_completion if report_completion != "unknown" else item.report_completion
                    )
                    if current == root_id:
                        phase = terminal_phase
                        detail = (
                            terminal_detail if terminal_phase == "report_ready" else
                            _agent_phase_detail(
                                terminal_phase, terminal_reason=terminal_reason or raw_detail,
                                partial=row_partial,
                                report_completion=row_report_completion,
                            )
                        ) or item.detail or phase.replace("_", " ")
                    elif item.phase in _ACTIVE_AGENT_PHASES:
                        phase = "indeterminate"
                        detail = "parent settled before this child outcome was observed"
                    else:
                        phase, detail = item.phase, item.detail
                    self._subagents[current] = replace(
                        item, phase=phase, detail=detail,
                        sequence=max(item.sequence, 2_147_483_647),
                        updated_order=self._subagent_update_seq, updated_at=now,
                        finished_at=item.finished_at or now,
                        source_coverage_status=(
                            source_coverage_status if current == root_id
                            else item.source_coverage_status
                        ),
                        evidence_status=(
                            evidence_status if current == root_id else item.evidence_status
                        ),
                        evidence_account=(
                            evidence_account if current == root_id else item.evidence_account
                        ),
                        terminal_reason=(
                            terminal_reason if current == root_id else item.terminal_reason
                        ),
                        partial=(row_partial if current == root_id else item.partial),
                        report_completion=(
                            row_report_completion if current == root_id else item.report_completion
                        ),
                    )
                    if item.invocation_id:
                        self._retired_subagent_invocation_ids.add((item.parent_agent_id, item.invocation_id))
                self._retired_subagent_ids.add(current)
            return matched

        invocation_id = str(event.invocation_id or "")
        if not invocation_id and event.outcome is not None:
            invocation_id = str(getattr(getattr(event.outcome, "invocation", None), "id", "") or "")
        if invocation_id:
            exact_item = next((item for item in self._subagents.values()
                               if item.invocation_id == invocation_id and not item.parent_agent_id), None)
            if exact_item is not None and child_id and exact_item.agent_id != child_id:
                child_item = self._subagents.get(child_id)
                compatible = child_item is None or not child_item.invocation_id \
                    or child_item.invocation_id == invocation_id
                if compatible:
                    # Merge the request-time placeholder into the authoritative artifact root. If a nested
                    # callback already materialized that root, preserve its topology/activity instead of
                    # replacing it. Never steal a sibling carrying a contradictory invocation identity.
                    self._subagents.pop(exact_item.agent_id, None)
                    self._retired_subagent_ids.add(exact_item.agent_id)
                    self._retired_subagent_invocation_ids.add(("", invocation_id))
                    if child_item is None:
                        child_item = replace(exact_item, agent_id=child_id)
                    elif not child_item.invocation_id:
                        child_item = replace(child_item, invocation_id=invocation_id)
                    self._subagents[child_id] = child_item
                    exact_item = child_item
            exact = exact_item.agent_id if exact_item is not None else ""
            # Effectless rejected-before-start calls have no row because ToolStarted was never emitted.
            if exact:
                return settle_tree(exact)
            if child_id:
                child = self._subagents.get(child_id)
                if child is not None and child.invocation_id and child.invocation_id != invocation_id:
                    return False
        if child_id and settle_tree(child_id):
            return True
        # Legacy/non-persistent launches have no artifact handle.  The launch ordinal is
        # presentation-only, so settle the oldest matching task kind deterministically.
        kind = "explorer" if event.name == "spawn_explore" else ""
        args = event.args if isinstance(event.args, Mapping) else {}
        kind = str(args.get("agent") or kind or "general")
        key = next((key for key, item in sorted(
            self._subagents.items(), key=lambda pair: (pair[1].launch_ordinal, pair[0]),
        ) if item.kind == kind), None)
        if key is not None:
            return settle_tree(key)
        return False

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
                self._subagents.clear()
                self._agent_outcomes.clear()
                self._agent_unlinked_outcomes.clear()
                self._retired_subagent_invocation_ids.clear()
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
            elif isinstance(event, ToolQueued):
                self._ensure_started(now)
                invocation = event.invocation
                invocation_id = str(getattr(invocation, "id", "") or "")
                name = str(getattr(invocation, "name", "") or "")
                if not invocation_id or tool_bucket(name) != "agent" \
                        or invocation_id in self._settled_invocation_ids:
                    return self._snapshot()
                args = getattr(invocation, "args", {})
                args = args if isinstance(args, Mapping) else {}
                provider_index = getattr(invocation, "provider_index", -1)
                request_ordinal = provider_index + 1 if isinstance(provider_index, int) else 0
                placeholder_id = f"invocation:{invocation_id}"
                previous = self._subagents.get(placeholder_id)
                if previous is None or previous.phase == "queued":
                    kind = str(args.get("agent") or (
                        "explorer" if name == "spawn_explore" else "general"
                    ))
                    self._subagent_update_seq += 1
                    queued_at = previous.queued_at if previous is not None else now
                    self._subagents[placeholder_id] = ActiveSubagent(
                        agent_id=placeholder_id, parent_turn_id=self._state.turn_id,
                        parent_agent_id="", invocation_id=invocation_id,
                        request_ordinal=max(0, request_ordinal), launch_ordinal=max(0, request_ordinal),
                        kind=_one_line(kind, 32), name=_one_line(args.get("name", ""), 40),
                        objective=_one_line(args.get("task", ""), 100), depth=1, phase="queued",
                        detail=_one_line(event.reason, 80) or "waiting for scheduler slot",
                        tool_count=0, sequence=-1, updated_order=self._subagent_update_seq,
                        started_at=None, updated_at=now, queued_at=queued_at,
                    )
                # Queue admission is presentation-only: it deliberately creates no ActiveTool
                # and increments no operation count until ToolStarted establishes physical start.
                self._transition(ProgressPhase.DELEGATING, self._agent_activity_detail(), now=now)
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
                if tool_bucket(event.name) == "agent":
                    args = event.args if isinstance(event.args, Mapping) else {}
                    invocation = getattr(event, "invocation", None)
                    provider_index = getattr(invocation, "provider_index", -1)
                    request_ordinal = provider_index + 1 if isinstance(provider_index, int) else 0
                    placeholder_id = f"invocation:{invocation_id}"
                    kind = str(args.get("agent") or (
                        "explorer" if event.name == "spawn_explore" else "general"
                    ))
                    objective = _one_line(args.get("task", ""), 100)
                    previous = self._subagents.get(placeholder_id)
                    self._subagent_update_seq += 1
                    self._subagents[placeholder_id] = ActiveSubagent(
                        agent_id=placeholder_id, parent_turn_id=self._state.turn_id,
                        parent_agent_id="", invocation_id=invocation_id,
                        request_ordinal=max(0, request_ordinal), launch_ordinal=max(0, request_ordinal),
                        kind=_one_line(kind, 32), name=_one_line(args.get("name", ""), 40),
                        objective=objective, depth=1, phase="starting",
                        detail="starting", tool_count=0, sequence=-1,
                        updated_order=self._subagent_update_seq,
                        started_at=now, updated_at=now,
                        queued_at=previous.queued_at if previous is not None else None,
                    )
                active_agents = sum(tool.bucket == "agent" for tool in self._active_tools.values())
                if active_agents > 1:
                    detail = self._agent_activity_detail()
                elif len(self._active_tools) > 1 and phase is ProgressPhase.INSPECTING:
                    detail = f"{len(self._active_tools)} reads in parallel"
                self._transition(phase, detail, now=now)
            elif isinstance(event, ToolResult):
                self._ensure_started(now)
                result_id = str(event.invocation_id or "")
                if not result_id and event.outcome is not None:
                    result_id = str(getattr(getattr(event.outcome, "invocation", None), "id", "") or "")
                if result_id and result_id in self._settled_invocation_ids:
                    return self._snapshot()
                if result_id:
                    self._settled_invocation_ids.add(result_id)
                was_active = (
                    result_id in self._active_tools if result_id else
                    any(tool.name == event.name for tool in self._active_tools.values())
                )
                queued_unstarted = bool(result_id) and any(
                    item.invocation_id == result_id and item.phase == "queued"
                    and item.started_at is None
                    for item in self._subagents.values()
                )
                self._remove_tool(event)
                raw_status = getattr(getattr(event, "outcome", None), "status", None)
                raw_status = getattr(raw_status, "value", raw_status)
                if raw_status is None or raw_status == "":
                    raw_status = event.status
                status = coerce_tool_status(
                    raw_status if raw_status not in (None, "") else not event.failing,
                ).value
                cancelled = status == "cancelled"
                steered = status == "steered"
                bucket = tool_bucket(event.name)
                if bucket == "agent":
                    matched_row = self._settle_subagent(event, status, now)
                    outcome_status = status
                    payload = self._child_payload(event)
                    report_completion = normalized_report_completion(payload.get("report_completion"))
                    if status == "succeeded" and report_completion not in {"complete", "partial"}:
                        outcome_status = "completed"
                    if status == "failed" and (
                        _timeout_cause(payload.get("stop_cause"))
                        or _timeout_cause(payload.get("stop_reason"))
                    ):
                        outcome_status = "timed_out"
                    if not matched_row and (self._child_artifact_id(event) or not was_active):
                        self._agent_outcomes[outcome_status] = \
                            self._agent_outcomes.get(outcome_status, 0) + 1
                    elif not matched_row:
                        self._agent_unlinked_outcomes[outcome_status] = \
                            self._agent_unlinked_outcomes.get(outcome_status, 0) + 1
                if bucket in ("read", "edit", "cmd", "agent") and not queued_unstarted:
                    self._counts[bucket] = self._counts.get(bucket, 0) + 1
                if steered and not queued_unstarted:
                    self._counts["steer"] = self._counts.get("steer", 0) + 1
                if event.failing and not cancelled and not queued_unstarted:
                    self._counts["fail"] = self._counts.get("fail", 0) + 1
                event_args = event.args if isinstance(event.args, dict) else {}
                if event.name == "update_plan" and status == "succeeded":
                    self._update_plan(event_args.get("steps"))
                note = event_args.get("note")
                if status == "succeeded" and isinstance(note, str) and note.strip():
                    self._replace(last_milestone=_one_line(note, 160))
                if self._active_tools:
                    latest = next(reversed(self._active_tools.values()))
                    phase, detail = _tool_detail(latest.name, {})
                    active_agents = sum(tool.bucket == "agent" for tool in self._active_tools.values())
                    if active_agents:
                        phase, detail = ProgressPhase.DELEGATING, self._agent_activity_detail()
                    elif len(self._active_tools) > 1 and phase is ProgressPhase.INSPECTING:
                        detail = f"{len(self._active_tools)} reads in parallel"
                    else:
                        detail = latest.detail or detail
                    self._transition(phase, detail, now=now)
                elif any(item.phase in _ACTIVE_AGENT_PHASES
                         for item in self._subagents.values()):
                    self._transition(ProgressPhase.DELEGATING, self._agent_activity_detail(), now=now)
                else:
                    self._transition(ProgressPhase.INTEGRATING, "integrating results", now=now)
            elif isinstance(event, StepEnd):
                self._ensure_started(now)
                # The sink replaces the transient matrix with one durable settled group at this boundary.
                # Keep tombstones for the rest of the physical turn so late worker callbacks cannot resurrect it.
                self._subagents.clear()
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
                self._subagents.clear()
                self._replace(stop_reason=event.stop_reason, loop_finished=True)
                if self._await_commit:
                    self._transition(ProgressPhase.FINALIZING, "model loop finished", now=now)
                else:
                    self._replace(turn_complete=True)
                    self._transition(ProgressPhase.COMPLETE, "", now=now)
            elif isinstance(event, TurnInterrupted):
                self._ensure_started(now)
                self._active_tools.clear()
                self._subagents.clear()
                self._replace(stop_reason=event.reason, loop_finished=True, turn_complete=False)
                self._transition(ProgressPhase.INTERRUPTED, event.message or event.reason, now=now)
            elif isinstance(event, TurnCommitted):
                self._ensure_started(now)
                self._active_tools.clear()
                self._subagents.clear()
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

    def subagent_activity(self, update: SubagentProgress | str) -> ProgressSnapshot:
        now = self._clock()
        with self._lock:
            if not self._snapshot().active or self._state.loop_finished:
                return self._snapshot()
            if isinstance(update, str):
                # Compatibility for third-party presentation callbacks.  Core children
                # use the structured path below.
                self._transition(ProgressPhase.DELEGATING, _one_line(update, 110), now=now)
                return self._snapshot()
            if not isinstance(update, SubagentProgress):
                return self._snapshot()
            if update.parent_turn_id and self._state.turn_id \
                    and update.parent_turn_id != self._state.turn_id:
                return self._snapshot()
            agent_id = str(update.agent_id or "")
            if not agent_id:
                return self._snapshot()
            if agent_id in self._retired_subagent_ids:
                return self._snapshot()
            invocation_id = str(update.invocation_id or "")
            invocation_scope = (str(update.parent_agent_id or ""), invocation_id)
            if invocation_id and invocation_scope in self._retired_subagent_invocation_ids:
                return self._snapshot()
            placeholder_key = next((key for key, item in self._subagents.items()
                                    if key.startswith("invocation:") and invocation_id
                                    and item.invocation_id == invocation_id
                                    and item.parent_agent_id == str(update.parent_agent_id or "")), "")
            if not placeholder_key and not invocation_id and not update.parent_agent_id:
                # Compatibility for third-party/older structured callbacks that predate exact invocation
                # identity.  Match only one unambiguous physical placeholder; never guess between siblings.
                candidates = [
                    key for key, item in self._subagents.items()
                    if key.startswith("invocation:")
                    and item.kind == _one_line(update.kind, 32)
                    and (not update.launch_ordinal
                         or item.request_ordinal == int(update.launch_ordinal)
                         or item.launch_ordinal == int(update.launch_ordinal))
                ]
                if len(candidates) == 1:
                    placeholder_key = candidates[0]
            if agent_id not in self._subagents and not placeholder_key \
                    and not any(tool.bucket == "agent" for tool in self._active_tools.values()):
                return self._snapshot()
            phase = str(update.phase or "running").strip().casefold()
            if phase not in _AGENT_PHASES:
                phase = "running"
            previous_key = agent_id if agent_id in self._subagents else placeholder_key
            previous = self._subagents.get(previous_key) if previous_key else None
            if previous is not None:
                if previous.invocation_id and invocation_id and previous.invocation_id != invocation_id:
                    return self._snapshot()  # stable row identity cannot be reassigned to a sibling invocation
                if previous.phase in _TERMINAL_AGENT_PHASES:
                    return self._snapshot()  # only authoritative ToolResult may rewrite a terminal hint
                update_sequence = _nonnegative_int(update.sequence)
                if update_sequence < previous.sequence:
                    return self._snapshot()
                if update_sequence == previous.sequence and previous.sequence >= 0:
                    # Same-sequence delivery is either an idempotent retry or a conflicting stale writer.
                    # Neither may reorder or rewrite the row.
                    return self._snapshot()
                previous_model_order = _MODEL_AGENT_PHASE_ORDER.get(previous.phase)
                update_model_order = _MODEL_AGENT_PHASE_ORDER.get(phase)
                if previous_model_order is not None and update_model_order is not None \
                        and update_model_order < previous_model_order:
                    # Sequence numbers order delivery, not semantic model progress. A delayed SDK callback
                    # with a newer sequence must not rewind a more-specific state already observed.
                    return self._snapshot()
            self._subagent_update_seq += 1
            started_at = (
                previous.started_at
                if previous is not None and previous.started_at is not None
                else now
            )
            request_ordinal = _nonnegative_int(update.request_ordinal)
            if not request_ordinal and previous is not None:
                request_ordinal = previous.request_ordinal
            objective = _one_line(update.objective, 100)
            if not objective and previous is not None:
                objective = previous.objective
            kind = _one_line(update.kind, 32) or (previous.kind if previous is not None else "")
            name = _one_line(update.name, 40) or (previous.name if previous is not None else "")
            launch_ordinal = _nonnegative_int(update.launch_ordinal)
            if not launch_ordinal and previous is not None:
                launch_ordinal = previous.launch_ordinal
            attempt = _nonnegative_int(update.attempt)
            max_attempts = _nonnegative_int(update.max_attempts)
            retry_delay_s = _nonnegative_float(update.retry_delay_s)
            tool_name = _one_line(update.tool_name, 50)
            terminal_reason = _one_line(update.terminal_reason, 80)
            partial = bool(update.partial)
            tool_count = _nonnegative_int(update.tool_count)
            if previous is not None:
                attempt = attempt or previous.attempt
                max_attempts = max_attempts or previous.max_attempts
                retry_delay_s = retry_delay_s or previous.retry_delay_s
                tool_name = tool_name or previous.tool_name
                terminal_reason = terminal_reason or previous.terminal_reason
                partial = partial or previous.partial
                # Tool count is cumulative physical work. Wrapper terminal updates intentionally carry no
                # duplicate counter, so zero means "no newer count", never "the child used no tools".
                tool_count = max(previous.tool_count, tool_count)
            detail = _agent_phase_detail(
                phase,
                detail=update.detail,
                attempt=attempt,
                max_attempts=max_attempts,
                retry_delay_s=retry_delay_s,
                tool_name=tool_name,
                terminal_reason=terminal_reason,
                partial=partial,
                report_completion=(previous.report_completion if previous is not None else "unknown"),
            )
            finished_at = now if phase in _TERMINAL_AGENT_PHASES else None
            if previous_key and previous_key != agent_id:
                self._subagents.pop(previous_key, None)
            self._subagents[agent_id] = ActiveSubagent(
                agent_id=agent_id,
                parent_turn_id=str(update.parent_turn_id or ""),
                parent_agent_id=str(update.parent_agent_id or ""),
                invocation_id=invocation_id or (previous.invocation_id if previous is not None else ""),
                request_ordinal=request_ordinal,
                launch_ordinal=launch_ordinal,
                kind=kind, name=name,
                objective=objective,
                depth=max(1, _nonnegative_int(update.depth, 1)), phase=phase,
                detail=detail,
                tool_count=tool_count,
                sequence=_nonnegative_int(update.sequence),
                updated_order=self._subagent_update_seq,
                started_at=started_at, updated_at=now,
                queued_at=previous.queued_at if previous is not None else None,
                finished_at=(previous.finished_at if previous is not None and previous.finished_at is not None
                             else finished_at),
                source_coverage_status=(
                    previous.source_coverage_status if previous is not None else "not_assessed"
                ),
                evidence_status=(
                    previous.evidence_status if previous is not None else "not_assessed"
                ),
                evidence_account=(previous.evidence_account if previous is not None else ()),
                attempt=attempt, max_attempts=max_attempts, retry_delay_s=retry_delay_s,
                tool_name=tool_name, terminal_reason=terminal_reason, partial=partial,
                report_completion=(previous.report_completion if previous is not None else "unknown"),
            )
            self._replace()
            self._transition(ProgressPhase.DELEGATING, self._agent_activity_detail(), now=now)
            return self._snapshot()


__all__ = [
    "ActiveSubagent", "ActiveTool", "PlanProgress", "ProgressPhase", "ProgressSnapshot",
    "TurnProgress", "tool_bucket",
]
