"""Episodic cache — the lossless WRITE side (MEMORY-SPEC step 1).

An output-only event sink (sibling of LessonMiner): it buffers one turn's events and flushes ONE
record via `memory.append_episode` when the turn closes. It NEVER touches the Slice, so the cache
can never enter the LLM context — Markov by construction. Record shape:
`{steps: [{slice, action:[{name,args,failing}], observation:[...]}], note, meta}` — the SEED slice is
captured once (step 1) plus the turn's accumulated (action, observation) units; lossless for turn recall.
"""
from __future__ import annotations

from .events import AssistantText, Event, SliceBuilt, ToolResult, TurnEnd, TurnInterrupted
from .slice import edited_paths_in_code, paths_in_code  # noqa: F401  (paths_in_code kept for back-compat callers)


_EDIT_TOOL_NAMES = ("edit_file", "append_to_file", "str_replace", "write_file")


def _files_of(event: ToolResult) -> list[str]:
    """CHANGED files for meta['files'] — mirror slice_sink: only SUCCESSFUL edit tools (and the mutated
    paths of a successful execute_code). A read, a dir-scope grep, or a FAILED edit changed nothing, so
    labeling those 'changed/edited' misled recall + mis-classified consolidated lessons (FILE_TOUCHED)."""
    if event.failing:
        return []
    out = []
    args = event.args if isinstance(event.args, dict) else {}   # raw model args may be a non-dict (list/str/number)
    if event.name in _EDIT_TOOL_NAMES:
        p = args.get("path")
        if isinstance(p, str) and p:
            out.append(p)
    if event.name == "execute_code":
        code = args.get("code", "")
        out += edited_paths_in_code(code if isinstance(code, str) else "")   # mutate-only (write/open-w), not reads
    return out


_OBS_KEEP_WHOLE = 2000   # keep an observation WHOLE up to this (covers a ~120-line config / a page of
_OBS_HEAD = 1200         # output), so a value in the MIDDLE survives; beyond it, a generous head+tail.
_OBS_TAIL = 600          # Bounded — the archive is L2 (on disk, not the slice), and recall caps what it serves.


def _obs_excerpt(obs: str) -> str:
    """A BOUNDED excerpt of a tool observation, indented under its trace line, so the actual DATA a turn
    SAW (a value, a grep match, an error) survives into the recallable markdown. The one-line summary
    ('read_file x -> 250 chars, 8 lines') cannot answer 'what was the value?' later — this is the precision
    the cross-slice recall channel needs. Keep the WHOLE observation up to _OBS_KEEP_WHOLE (so a value in
    the MIDDLE of a normal file/output is not lost — the measured gap); only a truly large observation is
    reduced to head+tail. Bounded: the archive is L2 (on disk, not in context until recalled) and
    recall_history caps the total it serves; page-out (#74) already bounds huge tool outputs upstream."""
    o = (obs or "").strip()
    if not o:
        return ""
    body = o if len(o) <= _OBS_KEEP_WHOLE else (o[:_OBS_HEAD] + "\n…⋯…\n" + o[-_OBS_TAIL:])
    return "  " + body.replace("\n", "\n  ")   # indent the excerpt under its "- " trace bullet


def turn_markdown(title: str, steps: list[dict], note: str, meta: dict) -> str:
    """Render a SEALED turn as a clean, self-contained MARKDOWN snapshot — the readable artifact the
    cache holds and the next loop pages back via recall_history (the slice saved into the cache as
    markdown). Distilled, not a raw dump: heading, changed files, outcome, the action→result trace WITH
    a bounded excerpt of each observation (so the data the turn saw is recallable), and the conclusion.
    Built from the buffered turn data alone (no Slice coupling — Markov)."""
    from .tool_summary import summarize_tool_result
    files = meta.get("files") or []
    out = [f"# {title or '(turn)'}"]
    if files:
        out.append(f"**changed files:** {', '.join(files)}")
    if meta.get("stop_reason"):
        out.append(f"**outcome:** {meta['stop_reason']}")
    trace = []
    for st in steps:
        for a, o in zip(st.get("action", []), st.get("observation", [])):
            trace.append("- " + summarize_tool_result(a.get("name", ""), a.get("args", {}), o,
                                                       failing=bool(a.get("failing"))))
            ex = _obs_excerpt(o)        # keep the actual observed DATA, bounded, so recall is USEFUL
            if ex:
                trace.append(ex)
    if trace:
        out.append("\n## what happened\n" + "\n".join(trace))
    if note:
        out.append(f"\n## conclusion\n{note}")
    return "\n".join(out)


class EpisodeSink:
    """Buffers a turn's events; flushes one lossless record on TurnEnd OR TurnInterrupted."""

    def __init__(self, memory, *, session_id: str, task_id_fn, title_fn=lambda: "", outcome_fn=lambda: {}):
        self.memory = memory
        self.session_id = session_id
        self.task_id_fn = task_id_fn   # () -> current task_id (host supplies; Step 3 seam)
        self.title_fn = title_fn       # () -> human title (goal one-liner) for cheap trace-back
        self.outcome_fn = outcome_fn   # () -> {} of task-OUTCOME signals (e.g. requirements_open) for meta
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
            # the loop dispatches SliceBuilt for the seed → opens a new step segment
            self._steps.append({"slice": event.rendered, "action": [], "observation": []})
        elif isinstance(event, AssistantText):
            if event.content and event.content.strip():   # content-emitting models' note
                self._note = event.content.strip()
        elif isinstance(event, ToolResult):
            st = self._cur()
            args = event.args if isinstance(event.args, dict) else {}   # coerce: persist a dict so downstream
            st["action"].append({"name": event.name, "args": args, "failing": event.failing})   # (search_index/consolidate) never read a non-dict from the episode
            st["observation"].append(event.output)        # VERBATIM — lossless (not observe()'d)
            note = args.get("note", "")                   # reasoning models' note (empty content)
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
            try:
                outcome = self.outcome_fn() or {}   # task-OUTCOME signals (requirements_open) — what
            except Exception:   # noqa: BLE001       # consolidation gates promotion on; never lose a record
                outcome = {}
            meta = {**self._meta, "stop_reason": stop_reason,
                    "ptok": usage.get("prompt_tokens", 0),
                    "ctok": usage.get("completion_tokens", 0),
                    "files": sorted(set(self._meta["files"])),
                    **outcome}
            record = {
                "title": title,            # human breadcrumb for cheap trace-back (topic is task_id)
                "steps": self._steps,      # lossless raw events (full=true / step recall)
                "note": self._note,
                # the SEAL artifact: the turn's slice as a clean MARKDOWN snapshot — what recall_history
                # returns by default, so paging a past turn back reads like opening a readable doc.
                "markdown": turn_markdown(title, self._steps, self._note, meta),
                "meta": meta,
            }
            self.memory.append_episode(self.session_id, self.task_id_fn(), self._turn, record)
        finally:
            self._reset()  # reset regardless, so a turn can never bleed into the next


def make_episode_sink(memory, *, session_id: str, task_id_fn, title_fn=lambda: "", outcome_fn=lambda: {}):
    """None for non-durable memory (NullMemory) → host skips it → evals untouched."""
    if not getattr(memory, "is_durable", False):
        return None
    return EpisodeSink(memory, session_id=session_id, task_id_fn=task_id_fn, title_fn=title_fn,
                       outcome_fn=outcome_fn)
