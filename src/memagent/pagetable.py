"""PageTable — the SINGLE read/retrieval entry point for the slice.

Three scattered retrieval sources used to be wired independently into the slice build:
the code-discovery map (Retriever.retrieve/graph_map), the per-subtree project notes
(SubdirHints.hints_for), and cross-session episode search (Memory.search_episodes).
PageTable unifies them behind ONE call — ``lookup(focus, *, kind, k) -> list[PageRef]`` —
so the slice has a single place that decides WHAT to page in.

The PageTable owns the SubdirHints instance (constructed once per build closure, same
lifetime as before, preserving per-task subtree dedup). It is otherwise stateless: each
backend is a thin adapter over its source. Backends emit RAW text in the PageRef.preview;
fencing (wrap_untrusted) stays at ONE layer — the renderer in slice.py — so there is no
double-wrap.

NO-TRANSCRIPT MOAT: lookup() reads from durable/derived sources each turn; it never
accumulates state across turns (the only per-instance state is SubdirHints' per-task
surfaced-subtree set, which is a bounded durable store, not a transcript).

DEFERRED (next backends to fold in here):
  - per-file code refs (fan-out of the repo map) — kept as the single '(repo map)' page.
"""
from __future__ import annotations

import os

from .interfaces import PageRef
from .text_utils import format_ts, normalize_ws

# Per-code-map preview cap (signatures are compact; bounded like every tier). Mirrors the
# former slice.DISCOVERY_CHARS so the rendered RELATED CODE block is byte-identical.
CODE_PREVIEW_CHARS = 4000


class PageTable:
    """Single entry for the slice's read/retrieval. ``lookup`` dispatches by ``kind`` to one
    backend; ``k`` is the per-kind budget (k<=0 SKIPS that backend, returning [])."""

    def __init__(self, retriever=None, memory=None, subdir_hints=None,
                 *, exclude_session: str | None = None):
        self.retriever = retriever
        self.memory = memory
        self.subdir_hints = subdir_hints          # OWNED here (per-task subtree dedup lives on it)
        self.exclude_session = exclude_session     # cross-session reads drop the current lineage

    # ------------------------------------------------------------------ public
    def lookup(self, focus, *, kind: str, k: int = 6) -> list[PageRef]:
        """Page in references relevant to ``focus`` from the ``kind`` backend.

        ``focus`` is the per-kind locator: a discovery QUERY (code), the active-file WORKING
        SET (project-notes), or a search QUERY (episode-xsession). ``k`` bounds the backend
        (k<=0 => skip, [] — honors tighten's discovery_k=0 floor). Returns PageRefs carrying
        RAW text; the caller fences them at render time."""
        if k <= 0:
            return []
        if kind == "code":
            return self._code(focus, k)
        if kind == "project-notes":
            return self._project_notes(focus)
        if kind == "memory-lessons":
            return self._lessons(focus, k)
        if kind == "episode-xsession":
            return self._episodes(focus, k)
        if kind == "episode-thissession":
            return self._episodes_thissession(focus, k)
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

    def _lessons(self, query: str, k: int) -> list[PageRef]:
        """RELEVANT MEMORY: distilled cross-session LESSONS (memem's relevance-gated retrieve), the
        always-on per-turn recall. Distinct from `_episodes` (raw FTS5 episode text): lessons are the
        consolidated long-term vault, episodes are the lossless cache. Each Snippet -> one PageRef
        (preview carries the lesson text RAW; the renderer fences it). memory absent / no hits -> []."""
        if self.memory is None:
            return []
        snippets = self.memory.recall(query, k=k)
        return [PageRef(handle=sn.path, kind="memory-lessons", preview=sn.text,
                        score=sn.score, untrusted=True) for sn in snippets]

    def _episodes(self, query: str, k: int) -> list[PageRef]:
        """CROSS-SESSION RECALL: FTS5 episode hits from PAST sessions (the one cross-session read
        path). Each hit row -> one PageRef: `handle` is the session·turn locator, `preview` packs
        ts/title/note/match for the listing. Empty/unavailable index -> []."""
        if self.memory is None or not isinstance(query, str) or not query.strip():
            return []
        hits = self.memory.search_episodes(query.strip(), limit=k,
                                           exclude_session=self.exclude_session)
        return [_episode_pageref(h) for h in hits]

    def _episodes_thissession(self, session_id: str, k: int) -> list[PageRef]:
        """PAGED-OUT HISTORY manifest: locator-only PageRefs for the last ``k`` turns of THIS session —
        the TRIGGER that makes recall_history get called (the model cannot reach for a cache it cannot
        see; pin/view died because their payoff was invisible). The single this-session episodic READ
        entry (mirrors ``_episodes`` for cross-session) so the slice has ONE retrieval seam. Locators
        only — turn/title/breadcrumb, NEVER step bodies; content pages in solely when the model calls
        recall_history(turns=[N]). Bounded to ``k``; a trailing '…older' ref flags that more exist."""
        read = getattr(self.memory, "read_episodes", None)
        if read is None or not session_id:
            return []
        lines = read(session_id)        # whole-session read (same source recall_history uses)
        if not lines:
            return []
        shown = lines[-k:]
        refs = [PageRef(handle=str(ln.get("turn")), kind="episode-thissession",
                        preview=_pack_thissession_preview(ln),
                        score=float(ln.get("turn") or 0), untrusted=False) for ln in shown]
        older = len(lines) - len(shown)
        if older:
            refs.append(PageRef(handle="…older", kind="episode-thissession",
                                preview=(f"{older} earlier turn(s) — recall_history() for the full index, "
                                         f"or recall_history(search=\"…\") for other sessions"),
                                score=0.0, untrusted=False))
        return refs


def _episode_pageref(h: dict) -> PageRef:
    """Map one cross-session episode hit dict to a PageRef (lossless for the listing's display:
    locator in `handle`, ts/title/note/match packed into `preview`)."""
    handle = f"{(h.get('session_id') or '')[:14]} · turn {h.get('turn')}"
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
    only — never step bodies/observations (those page in on demand via recall_history)."""
    rec = ln.get("record", {}) or {}
    meta = rec.get("meta", {}) or {}
    title = normalize_ws(rec.get("title") or "(untitled)")[:52]
    flag = " · FAIL" if meta.get("failing") else ""
    crumb = _thissession_breadcrumb(rec, meta)
    return f"turn {ln.get('turn')} · \"{title}\"{flag}" + (f" · {crumb}" if crumb else "")


def _thissession_breadcrumb(rec: dict, meta: dict) -> str:
    """The payoff breadcrumb (≤60 chars). An empty breadcrumb was the pin/view killer — a locator
    with no visible payoff never gets called — so every line is GUARANTEED a content-derived hint:
    the model's own note if it left one, else the turn's edited files, else its distinct read/grep/run
    actions. All from data already in the record (no extra read, no LLM)."""
    note = normalize_ws(rec.get("note"))
    if note:
        return ("note: " + note)[:60]
    files = meta.get("files") or []
    if files:
        return ("edited: " + ", ".join(os.path.basename(str(f)) for f in files))[:60]
    acts: list[str] = []
    for st in rec.get("steps", []) or []:
        for a in st.get("action", []) or []:
            name = a.get("name") or ""
            args = a.get("args", {}) if isinstance(a.get("args"), dict) else {}
            arg = args.get("path") or args.get("query") or args.get("command") or ""
            sig = (f"{name} {os.path.basename(str(arg))}").strip() if arg else name
            if sig and sig not in acts:
                acts.append(sig)
    return ("did: " + ", ".join(acts[:3]))[:60] if acts else ""
