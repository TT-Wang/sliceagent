"""Authoritative semantic regions for the active Slice.

Each region owns both its data and its task/turn lifecycle.  ``pfc.Slice`` only
composes the regions and exposes temporary flat attribute aliases for older call
sites; it no longer duplicates lifecycle policy in a parallel table.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace


MAX_PROGRESS_SIGNALS = 8


@dataclass(frozen=True)
class ProgressSignal:
    """Small task-scoped evidence of progress, never a raw call/output record."""

    kind: str
    detail: str
    count: int = 1

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "count": self.count}

    @classmethod
    def from_dict(cls, value: Mapping) -> "ProgressSignal | None":
        if not isinstance(value, Mapping):
            return None
        kind = str(value.get("kind") or "").strip()
        detail = str(value.get("detail") or "").strip()
        if not kind or not detail:
            return None
        try:
            count = max(1, int(value.get("count") or 1))
        except (TypeError, ValueError):
            count = 1
        return cls(kind=kind, detail=detail, count=count)


@dataclass
class TaskProgress:
    """Task objective, deliberate plan/state, and compact cross-turn progress."""

    goal: str = ""
    goal_source: str = ""
    # The immutable goal names the topic; this lifecycle says whether it is still an outstanding
    # instruction.  A clean turn can only make it provisional.  Explicit continuation/failure reactivates
    # it, while the exact original remains recoverable through ``goal_source``.
    objective_status: str = "active"
    plan: list[dict] = field(default_factory=list)
    action_log: dict[str, dict] = field(default_factory=dict)
    world: dict = field(default_factory=dict)
    progress_signals: list[ProgressSignal] = field(default_factory=list)

    def reset(self, goal: str = "") -> None:
        self.goal = str(goal or "")
        self.goal_source = ""
        self.objective_status = "active" if self.goal else "provisionally_satisfied"
        self.plan = []
        self.action_log = {}
        self.world = {}
        self.progress_signals = []

    def seal(self) -> None:
        # Detailed/repeated invocation state belongs to TurnRuntime. Cross-turn anti-loop information is
        # represented only by bounded/coalesced progress signals.
        self.action_log = {}

    def set_objective_status(self, status: str) -> None:
        """Apply the small explicit lifecycle without accepting invented persisted values."""
        value = str(status or "active")
        if value not in ("active", "provisionally_satisfied"):
            value = "active"
        self.objective_status = value

    def activate_objective(self) -> None:
        if self.goal:
            self.objective_status = "active"

    def mark_objective_provisional(self) -> None:
        if self.goal:
            self.objective_status = "provisionally_satisfied"

    def add_progress(self, kind: str, detail: str) -> None:
        """Coalesce one bounded semantic signal and refresh its recency."""
        k = " ".join(str(kind or "").split())[:40]
        d = " ".join(str(detail or "").split())[:200]
        if not k or not d:
            return
        hit = next((s for s in self.progress_signals if s.kind == k and s.detail == d), None)
        if hit is not None:
            self.progress_signals.remove(hit)
            self.progress_signals.append(replace(hit, count=hit.count + 1))
        else:
            self.progress_signals.append(ProgressSignal(k, d))
        del self.progress_signals[:-MAX_PROGRESS_SIGNALS]

    def progress_records(self) -> list[dict]:
        return [signal.to_dict() for signal in self.progress_signals]

    def load_progress_records(self, records) -> None:
        self.progress_signals = []
        for raw in records or ():
            signal = ProgressSignal.from_dict(raw)
            if signal is not None:
                self.progress_signals.append(signal)
        self.progress_signals = self.progress_signals[-MAX_PROGRESS_SIGNALS:]


@dataclass
class EvidenceState:
    """Claims and blockers with their provenance tier."""

    findings: list[str] = field(default_factory=list)
    finding_source: dict = field(default_factory=dict)
    last_error: str = ""
    open_report: str = ""
    reconciliation_required: str = ""
    reconciliation_targets: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.findings = []
        self.finding_source = {}
        self.last_error = ""
        self.open_report = ""
        self.reconciliation_required = ""
        self.reconciliation_targets = []

    def seal(self) -> None:
        # Logical evidence is task-elastic. Physical projection handles pressure; an arbitrary turn-boundary
        # count must not silently delete still-load-bearing findings.
        live = set(self.findings)
        self.finding_source = {k: v for k, v in self.finding_source.items() if k in live}


@dataclass
class WorkingSet:
    """Elastic file/skill residency and its derived cache-control state."""

    active_files: list[str] = field(default_factory=list)
    active_skills: list[dict] = field(default_factory=list)
    edit_anchor: dict[str, str] = field(default_factory=dict)
    edited_files: set = field(default_factory=set)
    ghosts: list[dict] = field(default_factory=list)
    protected_deps: set = field(default_factory=set)
    pre_defs: dict = field(default_factory=dict)
    stale_deps: set = field(default_factory=set)
    io: dict = field(default_factory=lambda: {"hit": 0, "miss": 0, "refault": 0, "evict": 0})
    hot: dict = field(default_factory=dict)
    # Compatibility defaults; Slice passes swap.py's canonical values on reset/factory construction.
    read_budget: int = 4
    read_ceiling: int = 16

    def reset(self, *, read_budget: int, read_ceiling: int) -> None:
        self.active_files = []
        self.active_skills = []
        self.edit_anchor = {}
        self.edited_files = set()
        self.ghosts = []
        self.protected_deps = set()
        self.pre_defs = {}
        self.stale_deps = set()
        self.io = {"hit": 0, "miss": 0, "refault": 0, "evict": 0}
        self.hot = {}
        self.read_budget = read_budget
        self.read_ceiling = read_ceiling

    def seal(self) -> None:
        # SwapManager already owns bounded eviction, ghosts, hot/refault promotion and the adaptive budget.
        # Preserve that task-scoped working set across turns instead of applying a second, contradictory
        # "edited files only" policy here. Live bytes are still re-read by seed reconstruction.
        self.edited_files = type(self.edited_files)(p for p in self.edited_files if p in self.active_files)
        self.edit_anchor = {p: a for p, a in self.edit_anchor.items() if p in self.active_files}
        self.pre_defs = {p: value for p, value in self.pre_defs.items() if p in self.active_files}
        self.protected_deps = set()
        self.stale_deps = set()


@dataclass
class ContinuityState:
    """Short-range language continuity."""

    conversation: list[dict] = field(default_factory=list)
    turns: int = 0

    def reset(self) -> None:
        self.conversation = []
        self.turns = 0

    def seal(self) -> None:
        # The conversation ring is bounded when written; raw requests belong to immutable turn artifacts.
        return None


@dataclass
class TurnRuntime:
    """Detailed execution state that never survives a turn seal."""

    step: int = 0
    usage: dict = field(default_factory=dict)
    recent_calls: list[dict] = field(default_factory=list)
    applied_effect_ids: set[str] = field(default_factory=set)
    blocked_calls: int = 0
    since_edit: int = 0
    turn_actions: int = 0
    explore_mode: bool = False

    def reset(self) -> None:
        self.step = 0
        self.usage = {}
        self.recent_calls = []
        self.applied_effect_ids = set()
        self.blocked_calls = 0
        self.since_edit = 0
        self.turn_actions = 0
        self.explore_mode = False

    def seal(self) -> None:
        self.reset()
