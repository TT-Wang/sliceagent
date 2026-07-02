"""Typed findings schema (item 14a) — deterministic classification of a promoted note into
one of a small fixed vocabulary, so recall is sharper (a 'decision' reads differently from a
'ruled-out' dead end). Pure + deterministic → testable offline, no LLM.

Vocabulary (fixed, small — the point is sharp typed recall, not a taxonomy):
    DECISION         — a choice was made / an approach was adopted
    RESOLVED         — a question got answered / a bug was fixed / it now works
    RULED_OUT        — an approach was tried and abandoned / didn't work (a dead end to avoid)
    FILE_TOUCHED     — a concrete edit landed (the change set)
    NOTE             — fallback: an observation with no stronger signal

The classifier reads cheap lexical signals from the note text and the episode meta (was a
file edited? did an error clear?). It is intentionally conservative: an ambiguous note stays
NOTE rather than mis-typed. neocortex.py tags promoted lessons with the type; hippocampus.py
renders it as a leading [TYPE] badge.

NO-TRANSCRIPT INVARIANT: classification reads already-stored episode records; it produces a
tag on a durable note, never new context.

PUBLIC SIGNATURES (pinned):
    DECISION, RESOLVED, RULED_OUT, FILE_TOUCHED, NOTE   # str constants
    classify_finding(note: str, *, edited: bool = False, had_error: bool = False,
                     resolved: bool = False) -> str
    badge(kind: str) -> str                              # "[decision] " etc. (or "" for NOTE)
"""
from __future__ import annotations

import re

DECISION = "decision"
RESOLVED = "resolved-question"
RULED_OUT = "ruled-out"
FILE_TOUCHED = "file-touched"
NOTE = "note"

_RULED_OUT_RE = re.compile(
    r"\b(rule[ds]?\s+out|ruled out|doesn'?t work|didn'?t work|won'?t work|not\s+the\s+"
    r"(cause|issue|problem)|dead\s*end|abandon(ed)?|gave up|turned out not|no longer "
    r"(works|needed)|reverted|backed out)\b", re.I)
# NOTE: "instead of" is deliberately NOT a ruled-out signal — it reads as a DECISION
# ("use a queue instead of a list"). Ruled-out needs an explicit negative-outcome phrase.
_DECISION_RE = re.compile(
    r"\b(decided|decision|chose|choosing|will use|going with|approach|opt(ed)?\s+for|"
    r"settled on|plan to|strategy|prefer)\b", re.I)
_RESOLVED_RE = re.compile(
    r"\b(fixed|resolved|works now|passing|solved|answer(ed)?|root cause|the (bug|issue|"
    r"problem) was|turned out (to be|that)|because)\b", re.I)


def classify_finding(note: str, *, edited: bool = False, had_error: bool = False,
                     resolved: bool = False) -> str:
    """Classify `note` into the fixed vocabulary. Precedence (strongest signal wins):
    RULED_OUT (a dead end is the most valuable-to-flag and easiest to mis-file as RESOLVED) >
    DECISION > RESOLVED > FILE_TOUCHED > NOTE. `edited`/`had_error`/`resolved` come from the
    episode meta and reinforce the structural signal when the text is ambiguous."""
    text = note or ""
    if _RULED_OUT_RE.search(text):
        return RULED_OUT
    if _DECISION_RE.search(text):
        return DECISION
    if _RESOLVED_RE.search(text) or (had_error and resolved):
        return RESOLVED
    if edited:
        return FILE_TOUCHED
    return NOTE


def badge(kind: str) -> str:
    """Leading badge for rendering. NOTE → '' (the unmarked default keeps the index clean)."""
    return "" if (not kind or kind == NOTE) else f"[{kind}] "
