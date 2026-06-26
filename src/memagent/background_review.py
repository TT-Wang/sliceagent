"""Async background-review fork (item 16) — OPT-IN, OFF by default, behind an env flag.

PORTED INTENT from /tmp/hermes-agent/agent/background_review.py: after a turn, fork a daemon
thread that critiques the just-finished episode and writes lessons to the memory store. Hermes
forks a whole AIAgent and replays the conversation; memagent has NO transcript to replay, so
the fork instead re-runs the EXISTING consolidation code (consolidate.promote_episodes /
promote_procedures) over the durable episodic JSONL — same write surface as session-end
consolidate, just incremental and off the critical path.

WHY THIS CAN'T DESTABILIZE THE DEFAULT PATH (the hard requirement):
  - OFF by default. Only runs when AGENT_BACKGROUND_REVIEW is truthy.
  - The fork reads ONLY the durable episodic cache (already flushed to disk by episode.py)
    and writes ONLY to durable stores (memem remember / SKILL.md / FTS5 index). It NEVER
    touches the Slice, the Session, the loop, the dispatcher, or the prompt cache.
  - It is a daemon thread: it can't block process exit and an exception in it is swallowed.
  - Re-entrancy guard: at most one review in flight; a new turn while one runs is skipped.
  - The main turn has already returned before this is even spawned (cli wires it AFTER
    run_turn), so latency is never on the user's critical path.

NO-TRANSCRIPT INVARIANT: upheld structurally — the worker's only inputs are durable records
and its only outputs are durable stores. Nothing it does can enter a slice tier directly; a
lesson it writes is recalled later through the SAME relevance-gated memory tier as any other.

PUBLIC SIGNATURES (pinned):
    background_review_enabled() -> bool
    class BackgroundReviewer:
        def __init__(self, memory, *, scope: str, on_log=None) -> None
        def review(self, session_id: str) -> None        # spawns the daemon (no-op if disabled/busy)
        def join(self, timeout: float | None = None) -> None   # tests/shutdown: wait for the worker
    make_background_reviewer(memory, *, scope: str, on_log=None) -> BackgroundReviewer | None
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
    """Forks a daemon thread per review that consolidates the latest episode incrementally.

    Reuses consolidate.promote_episodes / promote_procedures verbatim (the moat's existing
    write logic) — this module adds ONLY the fork + safety scaffolding, no new mining logic.
    """

    def __init__(self, memory, *, scope: str, on_log=None) -> None:
        self.memory = memory
        self.scope = scope
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
            name="memagent-bg-review", daemon=True)
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
            from .consolidate import promote_episodes, promote_procedures, render_skill
            records = self.memory.read_episodes(session_id)
            if not records:
                return
            # critique the LATEST turn in the context of the session: promote_* are pure and
            # frequency-weight across the whole session, so passing all records gives the newest
            # corrective/procedural signal its proper recurrence count. Dedup against what's
            # already stored is memem's job (remember is idempotent-ish on identical content).
            lessons = promote_episodes(records)
            for lesson in lessons:
                try:
                    self.memory.remember(lesson["content"], title=lesson["title"], scope=self.scope,
                                         tags=lesson["tags"], paths=lesson.get("files"))  # file-context parity
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


def make_background_reviewer(memory, *, scope: str, on_log=None) -> BackgroundReviewer | None:
    """Factory. Returns None when the feature is disabled OR memory isn't durable — so the
    host can skip wiring entirely and the default path adds zero objects."""
    if not background_review_enabled():
        return None
    if not getattr(memory, "is_durable", False):
        return None
    return BackgroundReviewer(memory, scope=scope, on_log=on_log)
