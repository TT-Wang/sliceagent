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

from .interfaces import TaskState
from .pfc import Slice
from .text_utils import one_line


def slice_to_task_state(s: Slice, task_id: str, *, session_id: str = "", title: str = "",
                        status: str = "active", tags: str = "", resolution: str = "",
                        links: list[str] | None = None) -> TaskState:
    return TaskState(
        task_id=task_id, session_id=session_id,
        title=title or one_line(s.goal, 60), status=status, goal=s.goal,
        findings=list(s.findings),
        finding_source={k: v for k, v in s.finding_source.items() if k in set(s.findings)},  # provenance, bounded to live findings
        requirements=[dict(r) for r in s.requirements],   # carry the standing contract across resume
        plan=[dict(p) for p in s.plan],                   # carry the PLAN (TodoWrite) across resume
        open_report=getattr(s, "open_report", ""),        # carry the OPEN USER REPORT blocker (was silently lost)
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
    s.finding_source = dict(getattr(ts, "finding_source", {}))            # restore provenance (getattr: back-compat with old checkpoints)
    s.requirements = [dict(r) for r in getattr(ts, "requirements", [])]   # restore the standing contract
    s.plan = [dict(p) for p in getattr(ts, "plan", [])]                   # restore the PLAN (TodoWrite)
    s.open_report = getattr(ts, "open_report", "")                        # restore the OPEN USER REPORT blocker
    s.world = dict(getattr(ts, "world", {}))                              # restore the WORLD MODEL
    s.active_files = list(ts.active_files)
    s.edited_files = set(ts.edited_files)
    s.edit_anchor = dict(ts.edit_anchor)
    s.last_error = ts.last_error
    s.since_edit = 0                 # resume = fresh action epoch (don't fire render_convergence)
    return s
