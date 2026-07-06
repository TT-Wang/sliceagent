"""HIPPOCAMPUS — the episodic memory: explicit recall of this session's own past turns. Two
complementary sides live in this one module: the WRITE side (EpisodeSink, buffers one turn's events
and flushes a lossless record via memory.append_episode when the turn closes) and the READ side (the
`history/` virtual files + search_history). Neither ever touches the Slice directly — the cache can
never enter the LLM context except through an explicit read (Markov by construction); this is what
distinguishes HIPPOCAMPUS from PFC (pfc.py, the carried working memory) and from NEOCORTEX
(neocortex.py, the auto-surfaced, distilled lessons vault) — episodic recall is precise, verbatim, and
only happens on request.

WRITE SIDE — episodic cache, the lossless side (MEMORY-SPEC step 1).
An output-only event sink (sibling of LessonMiner): it buffers one turn's events and flushes ONE
record via `memory.append_episode` when the turn closes. It NEVER touches the Slice, so the cache
can never enter the LLM context — Markov by construction. Record shape:
`{steps: [{slice, action:[{name,args,failing}], observation:[...]}], note, meta}` — the SEED slice is
captured once (step 1) plus the turn's accumulated (action, observation) units; lossless for turn recall.

READ SIDE — the episodic cache exposed as READ-ONLY VIRTUAL FILES under `history/` (HistoryFS), which the
model reads/lists/greps with its ordinary file tools. Measured 2026-07-06: the model reaches for
read_file/grep (a pretraining reflex) far more readily than a bespoke recall tool, so surfacing sealed
turns as files converts evicted-fact confabulation 47%->0%. The PAGED-OUT HISTORY manifest in the slice
lists each earlier turn (turn · title · note) WITH the exact read_file("history/turn-N.md") call.
  read_file("history/index.md")     -> the full TIMESTAMPED/TITLED index of this session's turns
  read_file("history/turn-N.md")    -> a specific turn's seal snapshot (actions/observations/note)
  grep <pat> path=history/          -> content search over this session's turn files
  search_history("keywords")        -> FTS5 content search: THIS session + PAST sessions (the one thing
                                       the files can't do — other sessions aren't mounted here)
Reaching back is expected, not a failure: the slice is bounded, so an earlier turn genuinely is not in
front of the model. NON-ACCUMULATION (moat): a read turn is TRANSIENT — it enters context for this loop
only and is never written back into slice state (the slice is rebuilt from the durable stores each turn),
so reads can never rebuild the transcript. history/ + search_history are registered when memory is durable.
"""
from __future__ import annotations

import json
import os
import re
from collections import deque

from .events import AssistantText, Event, SliceBuilt, ToolResult, TurnEnd, TurnInterrupted
from .pfc import edited_paths_in_code, paths_in_code  # noqa: F401  (paths_in_code kept for back-compat callers)
from .safety import redact_text
from .text_utils import format_ts, now_iso

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
                # the SEAL artifact: the turn's slice as a clean MARKDOWN snapshot — what history/turn-N.md
                # serves, so paging a past turn back reads like opening a readable doc.
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


_MAX_RECORD_VALUE_BYTES = 256 * 1024  # per-value disk safety valve (one pathological output)


class HippocampusMixin:
    """The durable episodic-cache STORAGE side (lossless turn log on disk). Mixed into MememMemory
    (memory.py) alongside NeocortexMixin (neocortex.py) — `self` at runtime is a concrete MememMemory
    instance, so `self._vault`/`self._idx_lock` (set by MememMemory.__init__) resolve normally via the
    MRO. This is the counterpart EpisodeSink (above) writes through and recall_history's handler
    (below, via `memory.read_episodes`) reads through."""

    def _clamp(self, v):
        if isinstance(v, str) and len(v.encode("utf-8")) > _MAX_RECORD_VALUE_BYTES:
            b = v.encode("utf-8"); h = _MAX_RECORD_VALUE_BYTES // 2   # slice on BYTES (cap is a byte budget);
            # errors="replace" (not "ignore"): a byte cut mid-multibyte-char marks it U+FFFD instead of
            # silently deleting bytes — visible, lossless-ish boundary rather than a quiet corruption.
            head = b[:h].decode("utf-8", "replace"); tail = b[-h:].decode("utf-8", "replace")
            return redact_text(head + f"\n…[truncated {len(v)} chars]…\n" + tail)
        if isinstance(v, str):
            return redact_text(v)  # (c) redact every persisted episodic string on its way to the cache
        # #35: recurse into structured (non-string) tool outputs so str leaves inside a dict/list are
        # still byte-bounded + redacted — a huge or secret-bearing nested payload can't slip past.
        if isinstance(v, dict):
            return {k: self._clamp(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._clamp(x) for x in v]
        return v

    def _clamp_record(self, rec: dict) -> dict:
        # Redact + byte-bound EVERY string leaf of the WHOLE record, not just steps. Earlier this clamped
        # only steps[*].observation/action — so the top-level title/note/markdown/meta (markdown is rendered
        # from the RAW steps + note) reached the durable cache UNREDACTED and could be surfaced by
        # recall_history. _clamp recurses through dict/list, so one call covers the entire record uniformly.
        return self._clamp(rec)

    def append_episode(self, session_id: str, task_id: str, turn: int, record: dict) -> None:
        try:
            d = os.path.join(self._vault, "episodic")
            os.makedirs(d, exist_ok=True)
            ts = now_iso()
            clamped = self._clamp_record(record)
            line = {"v": 1, "session_id": session_id, "task_id": task_id, "turn": turn,
                    "ts": ts, "record": clamped}
            with open(os.path.join(d, f"{session_id}.jsonl"), "a", encoding="utf-8") as f:
                # #36: default=str — a non-serializable value in a tool output must STRINGIFY, never raise
                # and silently drop the whole turn (the except below would eat it = lost episode + index).
                f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except Exception:
            return  # a cache write must never break a session
        self._index_episode(session_id, task_id, turn, ts, clamped)  # additive FTS5 mirror (item 12)

    # --- cross-session FTS5 episode index (item 12; additive, degrades to no-op) ---
    def _episode_index(self):
        """Lazily open the FTS5 episode index (cached). Returns None if FTS5 is
        unavailable — every caller treats None as 'index off' and falls back."""
        idx = getattr(self, "_idx", "unset")
        if idx != "unset":
            return idx                       # fast path: already opened (lock-free)
        with self._idx_lock:                 # double-checked: exactly ONE connection is opened + tracked by close()
            idx = getattr(self, "_idx", "unset")
            if idx == "unset":
                try:
                    from .search_index import make_episode_index
                    idx = make_episode_index()
                    if not idx.is_active:
                        idx = None
                except Exception:
                    idx = None
                self._idx = idx
        return idx

    def close(self) -> None:
        """#33: close the cached FTS5 episode-index connection (WAL checkpoint + release the fd) at
        session end. Idempotent — safe before the index was ever opened ('unset') or after close."""
        idx = getattr(self, "_idx", None)
        if idx is not None and idx != "unset":
            try:
                idx.close()
            except Exception:  # noqa: BLE001
                pass
        self._idx = None

    def _index_episode(self, session_id, task_id, turn, ts, record) -> None:
        idx = self._episode_index()
        if idx is None:
            return
        try:
            from .search_index import episode_searchable_text
            idx.index_episode(session_id=session_id, task_id=task_id, turn=turn, ts=ts,
                              title=record.get("title", ""), note=record.get("note", ""),
                              text=episode_searchable_text(record))
        except Exception:
            pass

    def search_episodes(self, query: str, *, limit: int = 5,
                        exclude_session: str | None = None,
                        only_session: str | None = None) -> list[dict]:
        """Episode discovery (FTS5). `exclude_session` => cross-session recall; `only_session` =>
        within-session content recall of the long tail (turns past the manifest/index window).
        Returns bounded hit dicts (see search_index.EpisodeIndex.search) or [] when unavailable."""
        idx = self._episode_index()
        if idx is None:
            return []
        try:
            return idx.search(query, limit=limit, exclude_session=exclude_session,
                              only_session=only_session)
        except Exception:
            return []

    def read_episodes(self, session_id: str, *, limit: int | None = None) -> list[dict]:
        """Read the session's episodic cache (the read side behind the history/ virtual files and
        search_history). Returns the raw line dicts in turn order; `limit` keeps only the most recent N.
        Never raises."""
        try:
            path = os.path.join(self._vault, "episodic", f"{session_id}.jsonl")
            if not os.path.exists(path):
                return []
            # limit set → keep only the last N parsed dicts via a bounded deque (don't hold the whole
            # session in memory just to slice the tail); limit unset (consolidate) reads all by design.
            out = deque(maxlen=limit) if limit is not None else []   # limit=0 ⇒ deque(maxlen=0) keeps ZERO (was: read all)
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        try:
                            out.append(json.loads(ln))
                        except ValueError:
                            continue
            return list(out)
        except Exception:
            return []

    def episode_manifest(self, session_id: str, k: int) -> tuple[list[dict], int]:
        """(last_k_dicts, total_count) for the PAGED-OUT HISTORY manifest. Reads only the file TAIL and
        parses only ~k records, so the per-turn slice build stays O(k) instead of re-parsing the whole
        session JSONL every turn (which was O(n²) over a long session). Never raises."""
        try:
            path = os.path.join(self._vault, "episodic", f"{session_id}.jsonl")
            if not os.path.exists(path):
                return [], 0
            size = os.path.getsize(path)
            total = 0
            window = min(size, max(65536, (k + 1) * 4096))
            with open(path, "rb") as f:
                for line in f:                 # cheap newline count (no JSON parse) for the '…older' flag
                    if line.strip():
                        total += 1
                f.seek(max(0, size - window))
                tail = f.read()
            rows = tail.splitlines()
            if size > window and rows:
                rows = rows[1:]                # the window may start mid-line → drop the partial leader
            out: list[dict] = []
            for ln in rows:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln.decode("utf-8", "replace")))
                except ValueError:
                    continue
            return out[-k:], total
        except Exception:
            return [], 0


OBS_TAIL = 300         # legacy-record fallback ONLY: per-observation tail when a turn has no stored markdown.
# render_trace returns a fetched turn IN FULL (no read-side content cap) — the bound is the SEAL: the archive
# already excerpts observations at SAVE time (_obs_excerpt), and the turn markdown is a bounded snapshot.


def _short_ts(ts: str) -> str:
    return format_ts(ts)   # "06-16 12:30"


def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else "…" + s[-n:]


def render_trace(lines: list[dict]) -> str:
    """Page sealed turns back as their clean MARKDOWN snapshot (the seal artifact) — returned IN FULL, no
    read-side size cap (see the constants note: the bound is the seal + transience, not a second read cut;
    a cap here truncated the distilled conclusion at the markdown tail). Falls back to a computed
    action→result trace for older records that predate the stored markdown (per-observation tail only — a
    legacy raw-obs guard; the conclusion/note is kept whole)."""
    from .tool_summary import summarize_tool_result   # fallback path only
    out = []
    for ln in lines:
        rec = ln.get("record", {})
        head = f"\n── turn {ln.get('turn')} · {_short_ts(ln.get('ts',''))} · {rec.get('title') or ''}"
        md = rec.get("markdown")
        if md:                                   # the SEAL artifact — return it directly, in full
            out.append(head + "\n" + md)
        else:                                    # older record without a stored markdown → compute a trace
            block = [head]
            for st in rec.get("steps", []):
                for a, o in zip(st.get("action", []), st.get("observation", [])):
                    summary = summarize_tool_result(a.get("name", ""), a.get("args", {}), o,
                                                    failing=bool(a.get("failing")))
                    block.append(f"  • {summary} → {_tail(o, OBS_TAIL)}")
            if rec.get("note"):
                block.append(f"  ↳ note: {rec['note']}")     # conclusion in full
            out.append("\n".join(block))
    return "\n".join(out).strip() or "(no trace)"


def render_search(mine, cross) -> str:
    """Render a content search. THIS session's matching turns come WITH the read_file call to open the
    turn file — the model searched by content and now has the turn number, so any turn is reachable
    without guessing a number. PAST sessions' FTS5 hits follow as read-only context (their turns aren't
    mounted as files in this session)."""
    out = []
    if mine:
        out.append("# THIS SESSION — content matches (read the full turn with the call shown)")
        for r in mine:
            out.append(f'- turn {r.handle}: {r.preview}  → read_file("history/turn-{r.handle}.md")')
    if cross:
        out.append("# CROSS-SESSION RECALL (past sessions — FTS5 over the durable episode index)")
        for r in cross:
            out.append(f"- [{r.handle}] {r.preview}")
    return "\n".join(out) if out else "No content matches found."


# ── history/ — the episodic archive as a read-only VIRTUAL file namespace ─────────────────────────────
# Measured 2026-07-06 (A/B on the gap-detection matrix): the model reaches for read_file/grep — a deeply
# grooved pretraining reflex — but RESISTS the bespoke recall tool, wrongly treating the in-context convo as
# complete. Exposing sealed turns AS files it can read/list/grep converts evicted-fact confabulation 47%→0%
# (recovery 13%→100%). No files hit disk: HistoryFS serves the SAME episodic cache read_episodes reads,
# routed by LocalToolHost whenever a tool path targets `history/`. Read-only (writes rejected upstream); a
# real on-disk file always wins the name (host._history_route), so the virtual view never lies about disk.
HISTORY_MOUNT = "history"
_TURN_FILE = re.compile(r"^turn-(\d+)\.md$")


def _history_leaf(path: str) -> str:
    """The path tail under the history mount: 'history/turn-2.md' -> 'turn-2.md', 'history' or 'history/' -> ''."""
    p = (path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.rstrip("/")
    if p == HISTORY_MOUNT:
        return ""
    return p[len(HISTORY_MOUNT) + 1:] if p.startswith(HISTORY_MOUNT + "/") else p


class HistoryFS:
    """Read-only virtual filesystem over THIS session's sealed turns. `index.md` lists every turn (the
    manifest-as-file); each `turn-<N>.md` is that turn's markdown snapshot (the seal artifact, in full).
    Content is served live from the episodic cache (read_episodes) — nothing is written to disk. Duck-typed:
    the LocalToolHost holds one as `_history` and routes read_file/list_files/grep under `history/` here."""

    def __init__(self, memory, session_id: str):
        self._memory = memory
        self._session_id = session_id

    def _lines(self) -> list:
        return self._memory.read_episodes(self._session_id)

    @staticmethod
    def _turn_names(lines) -> list:
        return [f"turn-{ln.get('turn')}.md" for ln in lines if ln.get("turn") is not None]

    def index(self, lines=None) -> str:
        if lines is None:
            lines = self._lines()
        if not lines:
            return "# HISTORY (this session)\n(no earlier turns yet — this is an early turn.)"
        out = ["# HISTORY — this session's earlier turns "
               "(read one: read_file(\"history/turn-<N>.md\"); find by content: search_history(\"keywords\"))"]
        for ln in lines:
            rec = ln.get("record", {})
            title = (rec.get("title") or "(no title)")[:70]
            note = rec.get("note") or ""
            fail = " FAIL" if (rec.get("meta") or {}).get("failing") else ""
            out.append(f"- turn-{ln.get('turn')}.md · {_short_ts(ln.get('ts', ''))} · {title}{fail}"
                       + (f" — {note[:90]}" if note else ""))
        return "\n".join(out)

    def read_file(self, path: str) -> str:
        lines = self._lines()
        leaf = _history_leaf(path)
        if leaf in ("", "index.md"):
            return self.index(lines)
        m = _TURN_FILE.match(leaf)
        if m:
            n = int(m.group(1))
            sel = [ln for ln in lines if ln.get("turn") == n]
            if sel:
                return render_trace(sel)   # the seal snapshot for that turn, in full (same as recall's compact)
            return (f'history/{leaf}: no such turn this session. '
                    f'read_file("history/index.md") for the list of turns.')
        return (f'history/{leaf}: not a history file. Available: index.md, '
                f'{", ".join(self._turn_names(lines)) or "(no turns yet)"}.')

    def listing(self, path: str = HISTORY_MOUNT) -> str:
        names = ["index.md"] + self._turn_names(self._lines())
        return ("\n".join(names)
                + '\n(read index.md for turn titles, or search_history("keywords") to find a turn by content)')

    def grep(self, pattern: str, *, output_mode: str = "content", context: int = 0,
             offset: int = 0, limit: int = 50) -> str:
        # ponytail: regex over the rendered turn docs (Python re, not ripgrep — these are virtual). `context`
        # is accepted for arg-parity with the real grep but not implemented (turn docs are seal-bounded).
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"grep: invalid regex ({e})."
        lines = self._lines()
        docs = [("index.md", self.index(lines))]
        docs += [(f"turn-{ln.get('turn')}.md", render_trace([ln]))
                 for ln in lines if ln.get("turn") is not None]
        hits, counts = [], {}
        for name, text in docs:
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"history/{name}:{i}:{line}")
                    counts[name] = counts.get(name, 0) + 1
        if not hits:
            return "grep: no matches found."
        if output_mode == "files_with_matches":
            body_lines = [f"history/{n}" for n in counts]
        elif output_mode == "count":
            body_lines = [f"history/{n}:{c}" for n, c in counts.items()]
        else:
            body_lines = hits
        total = len(body_lines)
        window = body_lines[offset:offset + limit]
        if not window:
            return f"grep: no results at offset={offset} ({total} total matches)."
        body = "\n".join(window)
        if offset + limit < total:
            body += f"\n\n[truncated; use offset={offset + limit} to see more]"
        return body


def make_search_history_tool(memory, session_id: str):
    """ToolEntry for search_history: FTS5 content search over PAST sessions AND this session's turns — the
    one thing the greppable history/ files can't do (other sessions aren't mounted as files). Preserves the
    cross-session recall the deleted recall_history(search=…) provided. This-session hits come WITH the
    read_file(\"history/turn-N.md\") call to open them; past-session hits are read-only context. No rein
    needed — each query is a real search returning new info (unlike the old turn-drill loop)."""
    from .pagetable import PageTable
    from .registry import ToolEntry
    # PageTable's episode-xsession backend wraps memory.search_episodes; episode-search-thissession finds
    # this session's turns by content (reachable via read_file once you know the number).
    pages = PageTable(memory=memory, session_id=session_id)

    def _handler(args: dict) -> str:
        q = (args.get("query") or args.get("search") or "")
        q = q.strip() if isinstance(q, str) else ""
        if not q:
            return "search_history: pass a 'query' (FTS5 keywords — AND/OR, \"quoted\", prefix*)."
        mine = pages.lookup(q, kind="episode-search-thissession", k=6)
        cross = pages.lookup(q, kind="episode-xsession", k=6)
        if not mine and not cross:
            return ('No content matches in this or past sessions for that query. Try different keywords, '
                    'or read_file("history/index.md") for this session\'s full turn list.')
        return render_search(mine, cross)

    schema = {"type": "function", "function": {
        "name": "search_history",
        "description": (
            "Find an earlier turn by CONTENT (FTS5 keyword search) across THIS session AND your PAST sessions "
            "— use it when you don't know a turn's number, or to recall how something was solved in an earlier "
            "session. This-session matches come with the read_file(\"history/turn-N.md\") call to open them; "
            "for THIS session you can also read_file(\"history/index.md\") to browse every turn, or grep the "
            "history/ files. Query supports AND/OR, \"quoted phrases\", and prefix*."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "FTS5 keywords to search episode content for."},
        }, "required": ["query"]}}}
    return ToolEntry(name="search_history", schema=schema, handler=_handler, source="builtin")
