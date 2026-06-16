"""Skill write-origin provenance (item 13) — distinguishes consolidation-AUTHORED skills
from user-authored ones, so a future curator prunes ONLY auto skills, never the user's.

PORTED from /tmp/hermes-agent/tools/skill_provenance.py (ContextVar shape) + the frontmatter
provenance idea from skill_usage.py. memagent has no foreground tool that writes skills today —
consolidate.render_skill is the ONLY writer — so we mark provenance two ways, belt-and-braces:

  1. A `provenance:` frontmatter field on auto-written SKILL.md (durable, survives a sidecar
     wipe; readable by the loader and any future curator without a side channel).
  2. A ContextVar (`set_authoring_origin`) the writer can set, so a future foreground
     skill-write tool inherits the right default without each call passing a flag.

NO-TRANSCRIPT INVARIANT: provenance is a property of the durable SKILL.md store, never of any
message history. The ContextVar is process-local control state, not context.

PUBLIC SIGNATURES (pinned):
    AUTO = "consolidation"
    USER = "user"
    set_authoring_origin(origin: str) -> Token
    reset_authoring_origin(token: Token) -> None
    current_authoring_origin() -> str          # default USER (a bare write is the user's)
    frontmatter_line(origin: str) -> str        # "provenance: <origin>"
    provenance_of(meta: dict) -> str            # read it back from parsed frontmatter
    is_auto(meta: dict) -> bool                 # only auto skills are curator-prunable
"""
from __future__ import annotations

import contextvars

AUTO = "consolidation"
USER = "user"

_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_authoring_origin", default=USER
)


def set_authoring_origin(origin: str) -> contextvars.Token:
    """Bind the active skill-authoring origin to the current context. Returns a Token the
    caller MUST pass to reset_authoring_origin in a finally block."""
    return _origin.set(origin or USER)


def reset_authoring_origin(token: contextvars.Token) -> None:
    _origin.reset(token)


def current_authoring_origin() -> str:
    """The active authoring origin. Default USER — a bare skill write belongs to the user
    and must never be auto-pruned. consolidate.py sets AUTO around its render_skill writes."""
    return _origin.get()


def frontmatter_line(origin: str) -> str:
    """The frontmatter line to embed in an auto-written SKILL.md."""
    return f"provenance: {origin or USER}"


def provenance_of(meta: dict) -> str:
    """Read provenance back from parsed frontmatter (skills.parse_frontmatter output).
    Missing field → USER (legacy/user-authored skills predate the field; never prune them)."""
    if not isinstance(meta, dict):
        return USER
    v = (meta.get("provenance") or "").strip().lower()
    return v or USER


def is_auto(meta: dict) -> bool:
    """True iff this skill was authored by consolidation (curator-prunable). User skills,
    and any skill missing the field, return False — safe by default."""
    return provenance_of(meta) == AUTO
