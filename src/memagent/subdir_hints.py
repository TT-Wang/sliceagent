"""Progressive subdirectory hint discovery — pure per-turn lookup, no transcript.

Ported (shape) from /tmp/hermes-agent/agent/subdirectory_hints.py:
  - the directory tracker            (:57  SubdirectoryHintTracker)
  - the ancestor walk                (:120 _add_path_candidate)
  - valid-subdir confinement to root (:169 _is_valid_subdir)
  - first-match-per-dir hint load    (:198 _load_hints_for_directory)

ADAPTED TO THE NO-TRANSCRIPT MOAT
---------------------------------
Hermes appends discovered hints onto a tool result that lives in a growing conversation.
memagent has no transcript. Instead, ``hints_for(active_files)`` is a PURE per-turn lookup:
the slice calls it every turn with the CURRENT working set and gets back the hint text for
any subtree it has not surfaced yet THIS TASK. The model sees each new subtree's conventions
exactly ONCE — re-surfacing it every turn would bloat the slice and waste the cache.

The instance's ``_loaded_dirs`` set is a DURABLE STORE, not a transcript: it records which
subtrees have already been surfaced so the lookup is idempotent within a task. It holds only
directory paths (bounded, no accumulating content) and is wiped by ``reset()`` at task
boundaries so a new task re-surfaces the conventions it actually touches.

BOUNDED + CONFINED
------------------
At most ``MAX_DIRS`` subtrees surface per turn; each hint is capped at ``MAX_HINT_CHARS``;
the ancestor walk stops after ``MAX_ANCESTOR_WALK`` levels; and ``_is_valid_subdir`` rejects
anything outside the workspace root (no cross-agent ~/.claude/CLAUDE.md contamination).

INJECTION GUARD (lightweight, inline)
-------------------------------------
Hint files come from a (possibly cloned) repo and re-inject into the VOLATILE user tier each
turn, ranked below OPEN FILES. We neutralize obvious instruction markers inline so the model
treats hint text as data, not directives. The full threat-pattern scan (safety.scan_for_threats)
is DEFERRED — this tier is volatile and ranked low, so the lightweight guard suffices for now.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


class SubdirHints:
    """Track visited subtrees and surface each one's convention files ONCE per task.

    ``hints_for`` is pure per-turn: same active_files + same already-surfaced state -> same
    output (and '' once a subtree has been surfaced). ``reset`` clears the surfaced state so a
    new task starts fresh.
    """

    # Convention filenames to look for, in priority order. First match per directory wins
    # (different subtrees may use different conventions, but one file per dir is enough).
    HINT_FILENAMES = (
        "AGENTS.md", "agents.md",
        "CLAUDE.md", "claude.md",
        ".cursorrules",
    )

    # Per-hint-file cap (chars) — prevents a giant convention file from blowing the slice.
    MAX_HINT_CHARS = 4000
    # At most this many NEW subtrees surface in a single turn.
    MAX_DIRS = 4
    # How many parent directories to walk up from an active file before giving up.
    MAX_ANCESTOR_WALK = 5

    def __init__(self, root: str):
        # Resolve the workspace root once; everything is confined to this tree.
        self._root = Path(os.path.realpath(root or os.getcwd()))
        # DURABLE STORE (not a transcript): directories already surfaced this task.
        # Pre-mark the root — its conventions are loaded at session start elsewhere.
        self._loaded_dirs: set[Path] = {self._root}

    # ------------------------------------------------------------------ public
    def hints_for(self, active_files: list[str]) -> str:
        """Return convention-file text for any subtree in *active_files* not yet surfaced.

        Pure per-turn lookup: collects the new directories implied by the working set (each
        file's dir plus ancestors up to the root), loads the first convention file found in
        each, marks them surfaced, and returns a bounded, injection-guarded block. Returns ''
        when there is nothing new (or nothing valid) to surface.
        """
        if not active_files:
            return ""

        new_dirs = self._collect_new_dirs(active_files)
        if not new_dirs:
            return ""

        sections: list[str] = []
        for directory in new_dirs:
            if len(sections) >= self.MAX_DIRS:
                break
            # Mark surfaced BEFORE load so an empty/unreadable dir still won't be retried
            # every turn (idempotent within the task).
            self._loaded_dirs.add(directory)
            hint = self._load_hint(directory)
            if hint:
                sections.append(hint)

        if not sections:
            return ""
        return "\n\n".join(sections)

    def reset(self) -> None:
        """Clear the surfaced-subtree store so the next task re-surfaces what it touches."""
        self._loaded_dirs = {self._root}

    # ----------------------------------------------------------------- internal
    def _collect_new_dirs(self, active_files: list[str]) -> list[Path]:
        """Map active file paths to NEW, valid, in-root directories (with ancestor walk).

        Deterministic order (first-seen) so the per-turn output is stable for caching.
        """
        seen: set[Path] = set()
        ordered: list[Path] = []
        for raw in active_files:
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                p = Path(raw).expanduser()
                if not p.is_absolute():
                    p = self._root / p
                p = Path(os.path.realpath(p))
            except (OSError, ValueError):
                continue
            # Treat as a file path -> use its directory.
            if p.suffix or (p.exists() and p.is_file()):
                p = p.parent
            # Walk up ancestors, stopping at an already-surfaced dir or the root.
            for _ in range(self.MAX_ANCESTOR_WALK):
                if p in self._loaded_dirs:
                    break
                if p not in seen and self._is_valid_subdir(p):
                    seen.add(p)
                    ordered.append(p)
                parent = p.parent
                if parent == p:
                    break  # filesystem root
                p = parent
        return ordered

    def _is_valid_subdir(self, path: Path) -> bool:
        """True iff *path* is a real directory inside the workspace root and not yet surfaced.

        Confinement to the root is the security boundary: it blocks loading conventions from
        outside the workspace (e.g. ~/.claude/CLAUDE.md), which would cross-contaminate context.
        """
        if path in self._loaded_dirs:
            return False
        try:
            if not path.is_dir():
                return False
        except OSError:
            return False
        # Confine to the root tree (root itself is pre-marked, so this means a strict subdir).
        return self._within_root(path)

    def _within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self._root)
            return True
        except ValueError:
            return False

    def _load_hint(self, directory: Path) -> str:
        """First convention file found in *directory*, capped + injection-guarded. '' if none."""
        if not self._within_root(directory):
            return ""
        for filename in self.HINT_FILENAMES:
            hint_path = directory / filename
            try:
                if not hint_path.is_file():
                    continue
                content = hint_path.read_text(encoding="utf-8", errors="replace").strip()
            except (OSError, ValueError):
                continue
            if not content:
                continue
            content = _neutralize_injection(content)
            if len(content) > self.MAX_HINT_CHARS:
                content = content[: self.MAX_HINT_CHARS] + "\n[...truncated]"
            rel = self._display_path(hint_path)
            # First match per directory wins (like startup context loading).
            return f"[Subdirectory context: {rel}]\n{content}"
        return ""

    def _display_path(self, hint_path: Path) -> str:
        try:
            return str(hint_path.relative_to(self._root))
        except ValueError:
            return str(hint_path)


# ----------------------------------------------------------------------------
# Lightweight inline injection guard (full threat-scan DEFERRED — see module docstring)
# ----------------------------------------------------------------------------

# Obvious instruction-override markers a poisoned convention file might carry. We do not block
# the file (it is volatile, low-ranked reference) — we neutralize the marker so the model reads
# it as inert data. Conservative set: classic overrides, role hijack, HTML-comment injection.
_INJECTION_MARKERS = [
    re.compile(r'ignore\s+(?:all\s+|any\s+|the\s+)*(?:previous|prior|above)\s+instructions', re.I),
    re.compile(r'disregard\s+(?:all\s+|any\s+|your\s+|the\s+)*(?:instructions|rules|guidelines)', re.I),
    re.compile(r'system\s+prompt\s+override', re.I),
    re.compile(r'you\s+are\s+now\s+(?:a|an|the)\b', re.I),
    re.compile(r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->', re.I),
]

# Zero-width / invisible characters sometimes used to smuggle hidden instructions.
_INVISIBLE = re.compile(r'[​‌‍⁠﻿]')


def _neutralize_injection(text: str) -> str:
    """Defang obvious instruction markers inline; strip invisible chars. Never raises."""
    if not text:
        return text
    out = _INVISIBLE.sub("", text)
    for pat in _INJECTION_MARKERS:
        out = pat.sub("[neutralized-instruction]", out)
    return out
