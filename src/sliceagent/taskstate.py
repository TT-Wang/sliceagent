"""Pure Slice <-> TaskState mappers (MEMORY-SPEC step 2).

No I/O, no memem — keeps pfc.py (the moat) byte-identical. TaskState stores REFS (paths +
anchors), never file contents; resume re-reads files live (ground truth = disk). Transient tiers
(recent, action_log, active_skills) are intentionally NOT serialized — they're per-turn residue
re-derived from ground truth. NOTE on resume: active_skills is dropped (skill bodies live on disk
and re-load via the `skill` tool); since_edit is zeroed (a fresh action epoch — otherwise the
restored counter could fire render_convergence's STOP nudge on turn 1). The `conversation` ring +
`turns` counter are also intentionally NOT carried CROSS-session (they survive within a session via
seal()): the durable distilled state (goal, findings, world, requirements, edited set) is what a
resume rebuilds on, and short-range chat is re-grounded from the goal — carrying it would persist
verbatim user text into the vault for marginal continuity gain.
"""
from __future__ import annotations

from .intent import IntentState
from .interfaces import TaskState
from .persistence import Checkpoint
from .pfc import Slice
from .text_utils import one_line


def task_state_from_checkpoint(checkpoint: Checkpoint) -> TaskState:
    """Decode one immutable checkpoint through its public deep-thaw boundary."""
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("task_state_from_checkpoint requires a Checkpoint")
    return TaskState(**checkpoint.thawed_state())


def slice_to_task_state(s: Slice, task_id: str, *, session_id: str = "", title: str = "",
                        status: str = "active", tags: str = "", resolution: str = "",
                        links: list[str] | None = None) -> TaskState:
    return TaskState(
        task_id=task_id, session_id=session_id,
        title=title or one_line(s.goal, 60), status=status, goal=s.goal,
        goal_source=s.task.goal_source,
        objective_status=s.task.objective_status,
        findings=list(s.findings),
        finding_source={k: v for k, v in s.finding_source.items() if k in set(s.findings)},  # provenance, bounded to live findings
        current_request=s.intent.current_request,
        intent_entries=s.intent.to_records(),
        intent_next_id=s.intent.next_id,
        requirements=[dict(r) for r in s.intent.as_legacy_requirements()],  # derived v1 compatibility view
        plan=[dict(p) for p in s.plan],                   # carry the PLAN (TodoWrite) across resume
        progress_signals=s.task.progress_records(),       # compact task-scoped progress, never raw calls
        open_report=getattr(s, "open_report", ""),        # carry the OPEN USER REPORT blocker (was silently lost)
        reconciliation_required=getattr(s, "reconciliation_required", ""),
        reconciliation_targets=list(getattr(s, "reconciliation_targets", ())),
        world=dict(s.world),                              # carry the agent WORLD MODEL (was silently lost)
        active_files=list(s.active_files),
        edited_files=sorted(s.edited_files),                              # byte-stable across checkpoints
        edit_anchor={p: a for p, a in s.edit_anchor.items() if p in s.active_files},  # consistency
        last_error=s.last_error, since_edit=s.since_edit,                # serialized faithfully
        links=list(links or []), tags=tags, resolution=resolution,
    )


def task_state_to_slice(ts: TaskState, s: Slice | None = None) -> Slice:
    s = s or Slice()
    s.reset(ts.goal)                 # zeroes transient tiers (recent/action_log/active_skills/...)
    s.task.goal_source = getattr(ts, "goal_source", "")
    s.task.set_objective_status(getattr(ts, "objective_status", "active"))
    s.findings = list(ts.findings)
    s.finding_source = dict(getattr(ts, "finding_source", {}))            # restore provenance (getattr: back-compat with old checkpoints)
    # v2 typed intent wins. A v1 checkpoint has only Requirements; import those as legacy-authority
    # provisional/active entries without pretending we know their original user provenance.
    _records = list(getattr(ts, "intent_entries", []) or [])
    s.intent = IntentState.from_records(
        _records,
        current_request=getattr(ts, "current_request", "") or ts.goal,
        next_id=getattr(ts, "intent_next_id", 1),
    )
    version = int(getattr(ts, "schema_version", 1) or 1)
    if version >= 2 and not _records and getattr(ts, "requirements", None):
        raise ValueError("v2 task state has legacy requirements but no valid typed intent records")
    if version < 2 and not _records and getattr(ts, "requirements", None):
        s.intent.load_legacy_requirements(getattr(ts, "requirements", []))
    s.plan = [dict(p) for p in getattr(ts, "plan", [])]                   # restore the PLAN (TodoWrite)
    s.task.load_progress_records(getattr(ts, "progress_signals", []))
    s.open_report = getattr(ts, "open_report", "")                        # restore the OPEN USER REPORT blocker
    s.reconciliation_required = getattr(ts, "reconciliation_required", "")
    s.reconciliation_targets = list(getattr(ts, "reconciliation_targets", ()))
    s.world = dict(getattr(ts, "world", {}))                              # restore the WORLD MODEL
    s.active_files = list(ts.active_files)
    s.edited_files = set(ts.edited_files)
    s.edit_anchor = dict(ts.edit_anchor)
    s.last_error = ts.last_error
    s.since_edit = 0                 # resume = fresh action epoch (don't fire render_convergence)
    return s
