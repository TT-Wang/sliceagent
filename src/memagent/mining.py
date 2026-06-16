"""Lesson mining — the WRITE side of the memory loop.

The read side recalls lessons into the RELEVANT MEMORY tier. This closes the loop:
after a task SUCCEEDS, distill a durable lesson from what happened (the pitfall that
was hit, and that it was resolved) and remember() it into memem — so a future similar
task recalls it. That's what makes memagent memory-NATIVE rather than memory-using.

It's an event SINK (like slice_sink / log_sink) holding a ref to the slice for task
and outcome context — so the loop and the moat never change, and memem stays behind
the Memory interface. Signal-dense by construction: it mines ONLY a validated episode
— a successful turn (end_turn) in which an error was encountered and then cleared. No
error, no success, no lesson. An optional one-shot LLM pass distills a crisper lesson.
"""
from __future__ import annotations

from .events import Event, LessonSaved, ToolResult, TurnEnd
from .slice import _active, one_line

# touched-file extensions → coarse tags (helps recall group lessons by stack)
_EXT_TAG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby", ".c": "c", ".cpp": "cpp",
    ".sh": "shell",
}


def _err_key(err: str) -> str:
    """A stable-ish signature for an error, for in-session dedup."""
    return one_line(err, 100).lower()


class LessonMiner:
    """Event sink that mines a lesson per successful, error-resolving turn.

    Pass it as a sink. Late-bind `.dispatch` after the dispatcher is built so the
    LessonSaved event flows through the same log/terminal sinks as everything else.
    """

    def __init__(self, memory, state, *, llm=None, mode: str = "deterministic",
                 scope: str = "default"):
        self.memory = memory
        self.state = state
        self.llm = llm
        self.mode = mode            # "deterministic" | "llm"
        self.scope = scope
        self.dispatch = None        # late-bound; emits LessonSaved
        self._errors: list[str] = []   # errors seen this turn (in order)
        self._saved: set[str] = set()  # error signatures saved this session (dedup)

    def __call__(self, event: Event) -> None:
        if isinstance(event, ToolResult):
            if event.failing and event.output:
                self._errors.append(event.output)
        elif isinstance(event, TurnEnd):
            try:
                self._on_turn_end(event)
            finally:
                self._errors = []  # reset per-turn buffer regardless

    # --- mining ----------------------------------------------------------
    def _on_turn_end(self, event: TurnEnd) -> None:
        # validated episode only: success + an error that was hit AND finally cleared
        if event.stop_reason != "end_turn":
            return
        if not self._errors or _active(self.state).last_error:
            return
        pitfall = self._errors[-1]
        key = _err_key(pitfall)
        if key in self._saved:
            return

        title, content, tags = self._build(pitfall)
        if not content:
            return
        try:
            self.memory.remember(content, title=title, scope=self.scope, tags=tags)
        except Exception:
            return  # mining must never break the session
        self._saved.add(key)
        if self.dispatch is not None:
            self.dispatch(LessonSaved(title, content))

    def _build(self, pitfall: str):
        s = _active(self.state)
        task = (s.goal or "").strip()
        files = list(s.active_files)
        tags = self._tags(files)
        if self.mode == "llm" and self.llm is not None:
            content = self._distill(task, pitfall, files)
            if content:
                return ("Lesson: " + one_line(task, 60), content, tags)
        # deterministic lesson (default): honest — records the pitfall + that it was
        # resolved + what was touched, without claiming to know the exact one-line fix.
        content = (
            f"Task: {task}\n"
            f"Pitfall encountered: {one_line(pitfall, 200)}\n"
            f"Resolution: edited {', '.join(files) or '(files)'} and verified the task passing.\n"
        )
        return ("Lesson: " + one_line(task, 60), content, tags)

    def _distill(self, task: str, pitfall: str, files: list[str]) -> str:
        sys_msg = (
            "You distill ONE durable, generalizable engineering lesson from a coding "
            "episode, for a future agent. Output 1-3 sentences. Phrase it as a declarative "
            "FACT about the code/problem (what was wrong, why, and what the correct approach "
            "is) — NOT as an imperative to your future self: e.g. 'str_replace no-ops unless "
            "its snippet is unique' ✓, 'Always add context to str_replace' ✗. Imperatives get "
            "re-read as directives in later sessions and cause wrong or repeated work. "
            "No preamble, no markdown."
        )
        user = (
            f"Task: {task}\nError that was hit and then resolved:\n{one_line(pitfall, 400)}\n"
            f"Files changed: {', '.join(files) or '(unknown)'}\nWrite the lesson:"
        )
        try:
            resp = self.llm.complete(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}], []
            )
        except Exception:
            return ""
        text = (resp.content or "").strip()
        if not text:
            return ""
        return f"Task context: {one_line(task, 120)}\nLesson: {text}\n"

    @staticmethod
    def _tags(files: list[str]) -> str:
        import os
        tags = {"memagent"}
        for p in files:
            t = _EXT_TAG.get(os.path.splitext(p)[1])
            if t:
                tags.add(t)
        return ",".join(sorted(tags))


def make_miner(memory, state, *, llm=None, mode: str = "deterministic", scope: str = "default"):
    """Factory. Returns None when there's nothing to write to (NullMemory) — so the
    sink list stays clean and mining is a true no-op without memem."""
    if type(memory).__name__ == "NullMemory":
        return None
    return LessonMiner(memory, state, llm=llm, mode=mode, scope=scope)
