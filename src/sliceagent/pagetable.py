"""PageTable — the SINGLE read/retrieval entry point for the slice.

Three scattered retrieval sources used to be wired independently into the slice build:
the code-discovery map (Retriever.retrieve/graph_map), the per-subtree project notes
(SubdirHints.hints_for), and cross-session episode search (Memory.search_episodes).
PageTable unifies them behind ONE call — ``lookup(focus, *, kind, k) -> list[PageRef]`` —
so the slice has a single place that decides WHAT to page in.

The PageTable owns the SubdirHints instance (constructed once per build closure, same
lifetime as before, preserving per-task subtree dedup). It is otherwise stateless: each
backend is a thin adapter over its source. Backends emit RAW text in the PageRef.preview;
fencing (wrap_untrusted) stays at ONE layer — the renderer in seed.py — so there is no
double-wrap.

NO-TRANSCRIPT MOAT: lookup() reads from durable/derived sources each turn; it never
accumulates state across turns (the only per-instance state is SubdirHints' per-task
surfaced-subtree set, which is a bounded durable store, not a transcript).

BRAIN-ANALOGY LEGEND (used in this file's section comments — exactly three memory layers):
  SENSORY CORTEX  — code / project-notes: re-computed live from the filesystem, never persisted;
                    perception of the present, not memory of the past.
  HIPPOCAMPUS/L0  — canonical immutable events and artifact seals. ``episode-*`` backends are only
                    discovery over the legacy episodic JSONL/FTS compatibility mirror.
  PFC/L1          — Active Work, derived from L0 and rebuilt without a PageTable lookup.
  NEOCORTEX/L2    — typed USER/PROJECT/CRAFT knowledge, pushed only after scope/relevance admission.

Memem may extend L2 retrieval as an optional index/legacy bridge; it does not own L2. Roster and skills are
adjacent capabilities, not additional memory layers. Only four of PageTable's six kinds fire per turn inside
``build()``; the two episode-search kinds remain explicit compatibility discovery paths.

DEFERRED (next backends to fold in here):
  - per-file code refs (fan-out of the repo map) — kept as the single '(repo map)' page.
"""
from __future__ import annotations

import os

from .interfaces import PageRef
from .text_utils import format_ts, normalize_ws

# Per-code-map preview cap — a generous PHYSICAL backstop only. The real bound is BREADTH, applied ONCE
# in graph_map (top-N ranked files, each shown complete). This must not re-cut the breadth-bounded map.
CODE_PREVIEW_CHARS = 12000


class PageTable:
    """Single entry for the slice's read/retrieval. ``lookup`` dispatches by ``kind`` to one
    backend; ``k`` is the per-kind budget (k<=0 SKIPS that backend, returning [])."""

    def __init__(self, retriever=None, memory=None, subdir_hints=None,
                 *, session_id: str | None = None):
        self.retriever = retriever
        self.memory = memory
        self.subdir_hints = subdir_hints          # OWNED here (per-task subtree dedup lives on it)
        # ONE concept — the CURRENT session: cross-session reads EXCLUDE it, within-session reads filter
        # ONLY to it. (Was the overloaded `exclude_session`, used as both exclude AND only — a leak waiting
        # to happen the moment a caller left it None.)
        self.session_id = session_id

    # ------------------------------------------------------------------ public
    def lookup(self, focus, *, kind: str, k: int = 6, paths=None) -> list[PageRef]:
        """Page in references relevant to ``focus`` from the ``kind`` backend.

        ``focus`` is the per-kind locator: a discovery QUERY (code), the active-file WORKING
        SET (project-notes), or a search QUERY (episode-xsession). ``k`` bounds the backend
        (k<=0 => skip, [] — honors tighten's discovery_k=0 floor). Returns PageRefs carrying
        RAW text; the caller fences them at render time."""
        if k <= 0:
            return []
        # — SENSORY CORTEX (derived views): re-computed from the LIVE filesystem/code every call,
        # never persisted — there is nothing to "remember" here, only to look at again, more carefully.
        if kind == "code":
            return self._code(focus, k)
        if kind == "project-notes":
            return self._project_notes(focus)
        # — NEOCORTEX / L2: typed native knowledge admitted by scope and relevance. Optional Memem results
        # are legacy/index candidates and never create authority by ranking highly.
        if kind == "memory-lessons":
            return self._lessons(focus, k, paths)
        # — HIPPOCAMPUS / L0 compatibility discovery: these episode backends query the legacy JSONL/FTS
        # mirror. Canonical exact history is served independently from immutable artifact seals.
        if kind == "episode-xsession":
            return self._episodes(focus, k)
        if kind == "episode-thissession":
            return self._episodes_thissession(focus, k)
        if kind == "episode-search-thissession":
            return self._episodes_search_thissession(focus, k)
        return []

    # ----------------------------------------------------------------- backends
    def _code(self, query: str, k: int) -> list[PageRef]:
        """RELATED CODE: the retriever's relevance-ranked repo MAP. KEEP the single '(repo map)'
        shape (one PageRef wrapping the whole map text) — per-file fan-out is a deferred follow-up."""
        if self.retriever is None:
            return []
        snippets = self.retriever.retrieve(query, k=k)
        if not snippets:
            return []
        # The Retriever contract yields a single Snippet(path='(repo map)') today (see
        # code_index.RipgrepCodeIndex.retrieve). Carry its map text as ONE page, RAW.
        sn = snippets[0]
        return [PageRef(handle=sn.path, kind="code", preview=sn.text[:CODE_PREVIEW_CHARS],
                        score=sn.score, untrusted=True)]

    def _project_notes(self, active_files) -> list[PageRef]:
        """SUBDIRECTORY CONTEXT: convention files for any subtree in the working set not yet
        surfaced this task. SubdirHints is owned here; its per-task dedup is preserved."""
        if self.subdir_hints is None:
            return []
        text = self.subdir_hints.hints_for(active_files)
        if not text:
            return []
        return [PageRef(handle="(project notes)", kind="project-notes", preview=text,
                        score=0.0, untrusted=True)]

    def _lessons(self, query: str, k: int, paths=None) -> list[PageRef]:
        """L2 KNOWLEDGE: request-relevant typed records admitted by the memory backend.

        Native USER/PROJECT/CRAFT knowledge is canonical; optional Memem hits are labelled legacy/index
        candidates. ``_episodes`` is a separate L0-compatibility discovery path, never another knowledge
        tier. Each snippet becomes one fenced ``PageRef``; unavailable memory or no hits returns ``[]``.
        """
        if self.memory is None:
            return []
        seed_recall = getattr(self.memory, "seed_recall", None)
        recall = seed_recall if callable(seed_recall) else self.memory.recall
        snippets = recall(query, k=k, paths=paths)   # R1: file-context bonus at topic-start
        return [PageRef(handle=sn.path, kind="memory-lessons", preview=sn.text,
                        score=sn.score, untrusted=True) for sn in snippets]

    def _episodes(self, query: str, k: int) -> list[PageRef]:
        """CROSS-SESSION RECALL: FTS5 episode hits from PAST sessions (the one cross-session read
        path). Each hit row -> one PageRef: `handle` is the session·turn locator, `preview` packs
        ts/title/note/match for the listing. Empty/unavailable index -> []."""
        if self.memory is None or not isinstance(query, str) or not query.strip():
            return []
        hits = self.memory.search_episodes(query.strip(), limit=k,
                                           exclude_session=self.session_id)   # cross-session: drop my lineage
        return [_episode_pageref(h) for h in hits]

    def _episodes_search_thissession(self, query: str, k: int) -> list[PageRef]:
        """WITHIN-SESSION content recall: FTS5 over the CURRENT session's episodes (the long-tail past
        the manifest/index window). Closes the gap where an old turn was reachable only by a turn number
        nobody knew. Each hit -> a PageRef whose handle is the TURN NUMBER, so the model pages the full
        turn with read_file("history/turn-N.md") — search by content, read by the number it just learned."""
        if self.memory is None or not isinstance(query, str) or not query.strip() or not self.session_id:
            return []                              # FAIL CLOSED: no current session → no within-session search
        hits = self.memory.search_episodes(query.strip(), limit=k, only_session=self.session_id)
        # W6': a DELEGATED-WORK hit (task_id 'subagent:sub-N', turn=0 — an FTS mirror row, not a turn)
        # carries its subagents/ handle so the renderer points at the seal, never a bogus turn file.
        return [PageRef(handle=(str(h.get("task_id"))[len("subagent:"):]
                                if str(h.get("task_id") or "").startswith("subagent:")
                                else str(h.get("turn"))),
                        kind="episode-search-thissession",
                        preview=_pack_episode_preview(h), score=float(h.get("score") or 0.0),
                        untrusted=False) for h in hits]

    def _episodes_thissession(self, session_id: str, k: int) -> list[PageRef]:
        """Legacy history manifest: locator-only PageRefs for the last ``k`` compatibility rows in this session.

        This remains the trigger for older ``history/`` locators; new canonical manifests use
        ``@sliceagent/history/``. The single this-session episodic
        READ entry (mirrors ``_episodes`` for cross-session) so the slice has ONE retrieval seam. Locators
        only — turn/title/breadcrumb, NEVER step bodies; content pages in solely when the model calls
        read_file("history/turn-N.md"). Bounded to ``k``; a trailing '…older' ref flags that more exist."""
        if not session_id:
            return []
        # Use the TAIL-only manifest read (O(k)/turn) when available, so a long session doesn't re-parse the
        # whole JSONL every slice build — that was O(n²)/session, eroding the history-bounded-cost moat.
        manifest = getattr(self.memory, "episode_manifest", None)
        if manifest is not None:
            shown, total = manifest(session_id, k)
            older = max(0, total - len(shown))
        else:
            read = getattr(self.memory, "read_episodes", None)
            if read is None:
                return []
            lines = read(session_id)        # fallback: whole-session read
            shown = lines[-k:]
            older = len(lines) - len(shown)
        if not shown:
            return []
        refs = [PageRef(handle=str(ln.get("turn")), kind="episode-thissession",
                        preview=_pack_thissession_preview(ln),
                        score=float(ln.get("turn") or 0), untrusted=False) for ln in shown]
        if older:
            refs.append(PageRef(handle="…older", kind="episode-thissession",
                                preview=(f'{older} earlier turn(s) not shown — read_file("history/index.md") for '
                                         f'the full index, or search_history("keywords") to find an older turn '
                                         f"of THIS session by content (also matches past sessions)"),
                                score=0.0, untrusted=False))
        return refs


def _episode_pageref(h: dict) -> PageRef:
    """Map one cross-session episode hit dict to a PageRef (lossless for the listing's display:
    locator in `handle`, ts/title/note/match packed into `preview`)."""
    tid = str(h.get("task_id") or "")
    sid = str(h.get("session_id") or "")
    if tid.startswith("subagent:"):                                  # a past session's delegated seal,
        handle = f"{sid[:14]} · {tid[len('subagent:'):]}"            # read-only context (not mounted here)
    elif sid and h.get("turn") is not None:                          # a past turn — directly readable now via
        handle = f"history/{sid}/turn-{h.get('turn')}.md"            # the cross-session history/ namespace
    else:
        handle = f"{sid[:14]} · turn {h.get('turn')}"
    return PageRef(handle=handle, kind="episode-xsession",
                   preview=_pack_episode_preview(h),
                   score=float(h.get("score") or 0.0), untrusted=True)


def _pack_episode_preview(h: dict) -> str:
    ts = format_ts(h.get("ts"))                            # "06-16 12:30"
    title = (h.get("title") or "(no title)")[:60]
    note = (h.get("note") or "").strip()
    snip = normalize_ws(h.get("snippet"))
    out = f"{ts} · {title}"
    if note:
        out += f"\n    note: {note[:160]}"
    if snip:
        out += f"\n    match: {snip[:200]}"
    return out


def _pack_thissession_preview(ln: dict) -> str:
    """One locator-line body for the PAGED-OUT HISTORY manifest: the turn's title + a PAYOFF
    breadcrumb (what the turn HOLDS), so the model can decide to page it back informedly. Locators
    only — never step bodies/observations (those page in on demand via recall_history). `ln` is a
    raw line parsed from the on-disk episodic JSONL — a malformed/corrupt record (e.g. an older
    schema, a hand-edited file) must degrade to an empty preview, never crash the manifest build."""
    rec = ln.get("record")
    rec = rec if isinstance(rec, dict) else {}
    meta = rec.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    def clipped(value: str, size: int) -> str:
        value = normalize_ws(value)
        return value if len(value) <= size else value[:max(0, size - 1)].rstrip() + "…"

    request = clipped(rec.get("request"), 68)
    title = clipped(rec.get("title") or "(untitled)", 52)
    locator = f'user: "{request}"' if request else f'task: "{title}"'
    flag = " · FAIL" if meta.get("failing") else ""
    crumb = _thissession_breadcrumb(rec, meta)
    return f"turn {ln.get('turn')} · {locator}{flag}" + (f" · {crumb}" if crumb else "")


def _thissession_breadcrumb(rec: dict, meta: dict) -> str:
    """The payoff breadcrumb (≤60 chars). An empty breadcrumb was the pin/view killer — a locator
    with no visible payoff never gets called — so every line is GUARANTEED a content-derived hint:
    the model's own note if it left one, else the turn's edited files, else its distinct read/grep/run
    actions. All from data already in the record (no extra read, no LLM). `rec`/`meta` are already
    dict-guarded by the caller; `steps`/`action` entries are guarded here since they come from the
    same untrusted on-disk record."""
    def clipped(value: str, size: int) -> str:
        value = normalize_ws(value)
        return value if len(value) <= size else value[:max(0, size - 1)].rstrip() + "…"

    note = normalize_ws(rec.get("note"))
    if note:
        return clipped("assistant preview: " + note, 80)
    files = meta.get("files") or []
    files = files if isinstance(files, list) else []
    if files:
        return ("edited: " + ", ".join(os.path.basename(str(f)) for f in files))[:60]
    acts: list[str] = []
    for st in rec.get("steps", []) or []:
        if not isinstance(st, dict):
            continue
        for a in st.get("action", []) or []:
            if not isinstance(a, dict):
                continue
            name = a.get("name") or ""
            args = a.get("args", {}) if isinstance(a.get("args"), dict) else {}
            arg = args.get("path") or args.get("query") or args.get("command") or ""
            sig = (f"{name} {os.path.basename(str(arg))}").strip() if arg else name
            if sig and sig not in acts:
                acts.append(sig)
    return ("did: " + ", ".join(acts[:3]))[:60] if acts else ""
