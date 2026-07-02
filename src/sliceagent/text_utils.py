"""Tiny shared text helpers — ONE home for the whitespace/timestamp normalizers that were copy-pasted
across regions.py, skills.py, pagetable.py and hippocampus.py. Leaf module (no intra-package imports) so
any module can use it without an import cycle. Behavior is byte-identical to the expressions it replaces.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

_WS = re.compile(r"\s+")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_ws(s) -> str:
    """Collapse all runs of whitespace to a single space and strip. `None`/non-str -> ''."""
    return _WS.sub(" ", str(s or "")).strip()


def one_line(s, n: int = 80) -> str:
    """`normalize_ws` truncated to `n` chars — a one-line, bounded rendering of arbitrary text."""
    return normalize_ws(s)[:n]


def format_ts(ts) -> str:
    """ISO timestamp -> compact 'MM-DD HH:MM' (e.g. '06-16 12:30'); '' for empty/None."""
    return (ts or "")[5:16].replace("T", " ")


# Pure social/greeting messages that never imply work. The host answers these on a CHEAP path — no slice,
# no tool schemas, a tiny prompt — so a "hi" or "thanks" doesn't pay the full per-turn token cost.
_CHITCHAT = frozenset({
    "hi", "hii", "hiya", "hello", "hey", "heya", "hey there", "hi there", "hello there", "yo", "sup",
    "howdy", "good morning", "good afternoon", "good evening", "morning", "evening", "gm",
    "how are you", "how are you?", "how's it going", "hows it going", "what's up", "whats up", "how's things",
    "thanks", "thank you", "thanks!", "thank you!", "thx", "ty", "tysm", "cheers", "much appreciated",
    "appreciate it", "thanks a lot", "thank you so much",
    "ok", "okay", "k", "kk", "cool", "nice", "great", "awesome", "perfect", "got it", "sounds good",
    "great thanks", "ok thanks", "okay thanks", "nice thanks", "perfect thanks",
    "bye", "goodbye", "see you", "see ya", "later", "good night", "gn", "night",
    "lol", "haha", "nice one", "well done", "good job", "good bot", "gg",
})

# Minimal system prompt for the chitchat fast-path — keep it tiny; the whole point is to spend few tokens.
CHITCHAT_PROMPT = ("You are sliceagent, a coding agent. The user sent a brief greeting or social message. "
                   "Reply in ONE short, warm line. Do not call tools, summarize, or start any work.")


def is_chitchat(text) -> bool:
    """True ONLY for a pure greeting/social message (high precision — never fires on a real request). The
    whole message must match a known social phrase after trimming punctuation; a length cap guards against
    anything substantive ('hi, can you fix X' is not chitchat)."""
    t = str(text or "").strip().lower().rstrip("!.?,~ ")
    return 0 < len(t) <= 40 and t in _CHITCHAT
