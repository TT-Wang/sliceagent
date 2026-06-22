"""Pure Slice <-> TaskState mappers (MEMORY-SPEC step 2).

No I/O, no memem — keeps slice.py (the moat) byte-identical. TaskState stores REFS (paths +
anchors), never file contents; resume re-reads files live (ground truth = disk). Transient tiers
(recent, action_log, active_skills) are intentionally NOT serialized — they're per-turn residue
re-derived from ground truth. NOTE on resume: active_skills is dropped (skill bodies live on disk
and re-load via the `skill` tool); since_edit is zeroed (a fresh action epoch — otherwise the
restored counter could fire render_convergence's STOP nudge on turn 1).
"""
from __future__ import annotations

from .interfaces import TaskState
from .slice import Slice, one_line


def slice_to_task_state(s: Slice, task_id: str, *, session_id: str = "", title: str = "",
                        status: str = "active", tags: str = "", resolution: str = "",
                        links: list[str] | None = None) -> TaskState:
    return TaskState(
        task_id=task_id, session_id=session_id,
        title=title or one_line(s.goal, 60), status=status, goal=s.goal,
        findings=list(s.findings),
        requirements=[dict(r) for r in s.requirements],   # carry the standing contract across resume
        plan=[dict(p) for p in s.plan],                   # carry the PLAN (TodoWrite) across resume
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
    s.findings = list(ts.findings)
    s.requirements = [dict(r) for r in getattr(ts, "requirements", [])]   # restore the standing contract
    s.plan = [dict(p) for p in getattr(ts, "plan", [])]                   # restore the PLAN (TodoWrite)
    s.world = dict(getattr(ts, "world", {}))                              # restore the WORLD MODEL
    s.active_files = list(ts.active_files)
    s.edited_files = set(ts.edited_files)
    s.edit_anchor = dict(ts.edit_anchor)
    s.last_error = ts.last_error
    s.since_edit = 0                 # resume = fresh action epoch (don't fire render_convergence)
    return s
