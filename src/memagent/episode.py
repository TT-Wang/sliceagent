"""Episodic cache — the lossless WRITE side (MEMORY-SPEC step 1).

An output-only event sink (sibling of LessonMiner): it buffers one turn's events and flushes ONE
record via `memory.append_episode` when the turn closes. It NEVER touches the Slice, so the cache
can never enter the LLM context — Markov by construction. Record shape:
`{steps: [{slice, action:[{name,args,failing}], observation:[...]}], note, meta}` — per-step units
so a multi-step turn keeps coherent (state, action) pairs.
"""
from __future__ import annotations

from .events import AssistantText, Event, SliceBuilt, ToolResult, TurnEnd, TurnInterrupted
from .slice import paths_in_code


def _files_of(event: ToolResult) -> list[str]:
    out = []
    p = event.args.get("path")
    if p and event.name != "list_files":   # list_files' path is a dir to browse, not a working file
        out.append(p)
    out += paths_in_code(event.args.get("code", ""))
    return out


class EpisodeSink:
    """Buffers a turn's events; flushes one lossless record on TurnEnd OR TurnInterrupted."""

    def __init__(self, memory, *, session_id: str, task_id_fn, title_fn=lambda: ""):
        self.memory = memory
        self.session_id = session_id
        self.task_id_fn = task_id_fn   # () -> current task_id (host supplies; Step 3 seam)
        self.title_fn = title_fn       # () -> human title (goal one-liner) for cheap trace-back
        self._turn = 0
        self._reset()

    def _reset(self) -> None:
        self._steps: list[dict] = []
        self._note = ""
        self._meta = {"failing": False, "files": []}

    def _cur(self) -> dict:
        if not self._steps:
            self._steps.append({"slice": "", "action": [], "observation": []})
        return self._steps[-1]

    def __call__(self, event: Event) -> None:
        if isinstance(event, SliceBuilt):
            # run_step dispatches one SliceBuilt first each step → opens a new step
            self._steps.append({"slice": event.rendered, "action": [], "observation": []})
        elif isinstance(event, AssistantText):
            if event.content and event.content.strip():   # content-emitting models' note
                self._note = event.content.strip()
        elif isinstance(event, ToolResult):
            st = self._cur()
            st["action"].append({"name": event.name, "args": event.args, "failing": event.failing})
            st["observation"].append(event.output)        # VERBATIM — lossless (not observe()'d)
            note = event.args.get("note", "")             # reasoning models' note (empty content)
            if note:
                self._note = note
            if event.failing:
                self._meta["failing"] = True
            self._meta["files"] += _files_of(event)
        elif isinstance(event, TurnEnd):
            self._flush(event.stop_reason, event.usage)   # usage = per-turn TOTAL
        elif isinstance(event, TurnInterrupted):
            self._flush(event.reason, {})                 # abort path: loop returns WITHOUT TurnEnd

    def _flush(self, stop_reason: str, usage: dict) -> None:
        if not self._steps and not self._note and not self._meta["files"]:
            return  # nothing buffered (e.g. the empty TurnEnd right after a TurnInterrupted)
        self._turn += 1
        try:
            try:
                title = self.title_fn() or ""
            except Exception:   # noqa: BLE001 — a title hiccup must not lose the record
                title = ""
            record = {
                "title": title,            # human breadcrumb for cheap trace-back (topic is task_id)
                "steps": self._steps,
                "note": self._note,
                "meta": {**self._meta, "stop_reason": stop_reason,
                         "ptok": usage.get("prompt_tokens", 0),
                         "ctok": usage.get("completion_tokens", 0),
                         "files": sorted(set(self._meta["files"]))},
            }
            self.memory.append_episode(self.session_id, self.task_id_fn(), self._turn, record)
        finally:
            self._reset()  # reset regardless, so a turn can never bleed into the next


def make_episode_sink(memory, *, session_id: str, task_id_fn, title_fn=lambda: ""):
    """None for non-durable memory (NullMemory) → host skips it → evals untouched."""
    if not getattr(memory, "is_durable", False):
        return None
    return EpisodeSink(memory, session_id=session_id, task_id_fn=task_id_fn, title_fn=title_fn)
