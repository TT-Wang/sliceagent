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
  - Memory.recall (per-task episodic recall) — still a sibling call in slice.py this step.
  - per-file code refs (fan-out of the repo map) — kept as the single '(repo map)' page.
"""
from __future__ import annotations

import re

from .interfaces import PageRef

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
        if kind == "episode-xsession":
            return self._episodes(focus, k)
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

    def _episodes(self, query: str, k: int) -> list[PageRef]:
        """CROSS-SESSION RECALL: FTS5 episode hits from PAST sessions (the one cross-session read
        path). Each hit row -> one PageRef: `handle` is the session·turn locator, `preview` packs
        ts/title/note/match for the listing. Empty/unavailable index -> []."""
        if self.memory is None or not isinstance(query, str) or not query.strip():
            return []
        hits = self.memory.search_episodes(query.strip(), limit=k,
                                           exclude_session=self.exclude_session)
        return [_episode_pageref(h) for h in hits]


def _episode_pageref(h: dict) -> PageRef:
    """Map one cross-session episode hit dict to a PageRef (lossless for the listing's display:
    locator in `handle`, ts/title/note/match packed into `preview`)."""
    handle = f"{(h.get('session_id') or '')[:14]} · turn {h.get('turn')}"
    return PageRef(handle=handle, kind="episode-xsession",
                   preview=_pack_episode_preview(h),
                   score=float(h.get("score") or 0.0), untrusted=True)


def _pack_episode_preview(h: dict) -> str:
    ts = (h.get("ts") or "")[5:16].replace("T", " ")       # "06-16 12:30"
    title = (h.get("title") or "(no title)")[:60]
    note = (h.get("note") or "").strip()
    snip = re.sub(r"\s+", " ", h.get("snippet") or "").strip()
    out = f"{ts} · {title}"
    if note:
        out += f"\n    note: {note[:160]}"
    if snip:
        out += f"\n    match: {snip[:200]}"
    return out
