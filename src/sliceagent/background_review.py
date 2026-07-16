"""Async background-review fork (item 16) — OPT-IN, OFF by default, behind an env flag.

After a turn, fork a daemon thread that reviews the just-finished compatibility episode and proposes typed
L2 knowledge plus adjacent skills. SliceAgent has no growing transcript to replay, so the fork reuses the
existing promotion helpers over the legacy episodic JSONL mirror. Production writes bind native knowledge to
the project identity captured when the source work ran; Memem is only an optional downstream bridge.

WHY THIS CAN'T DESTABILIZE THE DEFAULT PATH (the hard requirement):
  - OFF by default. Only runs when AGENT_BACKGROUND_REVIEW is truthy.
  - The fork reads ONLY the episodic compatibility mirror (already flushed by ``EpisodeSink``) and writes
    ONLY typed L2 candidates plus adjacent ``SKILL.md`` assets. It NEVER
    touches the Slice, the Session, the loop, the dispatcher, or the prompt cache.
  - It is a daemon thread: it can't block process exit and an exception in it is swallowed.
  - Re-entrancy guard: at most one review in flight; a new turn while one runs is skipped.
  - The main turn has already returned before this is even spawned (cli wires it AFTER
    run_turn), so latency is never on the user's critical path.

NO-TRANSCRIPT INVARIANT: upheld structurally — the worker's only inputs are persisted records and its only
outputs are durable stores. Nothing it does enters a slice directly; a knowledge record is later admitted by
the same hard scope and relevance rules as any other L2 lead.

PUBLIC SIGNATURES (pinned):
    background_review_enabled() -> bool
    class BackgroundReviewer:
        def __init__(self, memory, *, scope: str, project_id: str = "", on_log=None) -> None
        def review(self, session_id: str) -> None        # spawns the daemon (no-op if disabled/busy)
        def join(self, timeout: float | None = None) -> None   # tests/shutdown: wait for the worker
    make_background_reviewer(memory, *, scope: str, project_id: str = "", on_log=None) -> BackgroundReviewer | None
"""
from __future__ import annotations

import os
import threading

_ENV_FLAG = "AGENT_BACKGROUND_REVIEW"


def background_review_enabled() -> bool:
    """True iff AGENT_BACKGROUND_REVIEW is set truthy. OFF by default — the whole feature is
    inert unless explicitly opted in, so the synchronous default path is byte-for-byte unchanged."""
    return str(os.environ.get(_ENV_FLAG, "")).strip().lower() in ("1", "true", "yes", "on")


class BackgroundReviewer:
    """Fork a daemon thread that incrementally derives knowledge and skill candidates.

    This module adds only the fork and safety scaffolding around the existing pure promotion helpers.
    """

    def __init__(self, memory, *, scope: str, project_id: str = "", on_log=None) -> None:
        self.memory = memory
        self.scope = scope
        self.project_id = str(project_id or "")
        self._on_log = on_log
        self._lock = threading.Lock()
        self._busy = False
        self._thread: threading.Thread | None = None

    def review(self, session_id: str) -> None:
        """Spawn the review daemon for `session_id`. No-op when disabled, when memory isn't
        durable, or when a prior review is still running (re-entrancy guard). Returns
        immediately — the work happens off-thread."""
        if not background_review_enabled():
            return
        if not getattr(self.memory, "is_durable", False):
            return
        with self._lock:
            if self._busy:
                return                      # at most one review in flight
            self._busy = True
        t = threading.Thread(
            target=self._run, args=(session_id,),
            name="sliceagent-bg-review", daemon=True)
        # publish self._thread only AFTER a successful start: otherwise a start() failure both latches
        # _busy=True forever (no more reviews) AND leaves join() calling .join() on a never-started thread.
        try:
            t.start()
        except Exception:  # noqa: BLE001 — a thread-spawn failure must not escape into the foreground caller
            with self._lock:
                self._busy = False
            self._log("background review: thread start failed")
            return
        self._thread = t

    def join(self, timeout: float | None = None) -> None:
        """Wait for the in-flight review (for deterministic tests / clean shutdown)."""
        t = self._thread
        if t is not None:
            t.join(timeout)

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(msg)
            except Exception:
                pass

    def _run(self, session_id: str) -> None:
        """Worker body — durable-in, durable-out. Every failure is swallowed: a background
        critique must NEVER affect the foreground session."""
        try:
            from .neocortex import promote_episodes, promote_procedures, render_skill
            read_bound = getattr(self.memory, "read_project_episodes", None)
            if self.project_id and callable(read_bound):
                records = read_bound(session_id, project_id=self.project_id)
            else:
                records = self.memory.read_episodes(session_id)
            if not records:
                return
            # critique the LATEST turn in the context of the session: promote_* are pure and
            # frequency-weight across the whole session, so passing all records gives the newest
            # corrective/procedural signal its proper recurrence count. Dedup against what's
            # Native project-bound writes own durable identity; an optional Memem mirror is downstream.
            lessons = promote_episodes(records)
            for lesson in lessons:
                try:
                    remember_bound = getattr(self.memory, "remember_for_project", None)
                    if self.project_id and callable(remember_bound):
                        remember_bound(
                            lesson["content"], project_id=self.project_id,
                            title=lesson["title"], tags=lesson["tags"], paths=lesson.get("files"),
                        )
                    else:
                        self.memory.remember(
                            lesson["content"], title=lesson["title"], scope=self.scope,
                            tags=lesson["tags"], paths=lesson.get("files"),
                        )  # compatibility memory without an immutable project-binding seam
                except Exception:
                    pass
            procs = promote_procedures(records)
            if procs:
                from .memory import _skills_dir, write_skill_file   # SAME guarded writer as session-end
                from .skill_provenance import AUTO, reset_authoring_origin, set_authoring_origin
                skills_dir = _skills_dir()
                token = set_authoring_origin(AUTO)   # mark fork-authored skills curator-prunable
                try:
                    for proc in procs:
                        # guarded writer: validate frontmatter + strict threat-scan + redact + ATOMIC replace
                        # (parity with memory.consolidate; closes the bypass + non-atomic-write bugs)
                        write_skill_file(proc["name"], render_skill(proc), skills_dir=skills_dir)
                finally:
                    reset_authoring_origin(token)
            self._log(f"background review: {len(lessons)} lesson(s), {len(procs)} skill(s)")
        except Exception:
            pass
        finally:
            with self._lock:
                self._busy = False


def make_background_reviewer(
    memory, *, scope: str, project_id: str = "", on_log=None,
) -> BackgroundReviewer | None:
    """Factory. Returns None when the feature is disabled OR memory isn't durable — so the
    host can skip wiring entirely and the default path adds zero objects."""
    if not background_review_enabled():
        return None
    if not getattr(memory, "is_durable", False):
        return None
    return BackgroundReviewer(memory, scope=scope, project_id=project_id, on_log=on_log)
