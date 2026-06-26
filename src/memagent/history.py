"""recall_history — the model's first-class read into the episodic cache (a NORMAL navigation move).

The cache is never part of the slice (Markov by construction); this tool pages a past turn back IN.
It is the read verb for the PAGED-OUT HISTORY manifest in the slice: that manifest lists each earlier
turn (turn · title · note) WITH the exact call to fetch it, so reaching back is copy-paste, not a
blind guess — a cache the model can't see is a cache it never calls (the manifest is the trigger).
  recall_history()                  -> the full TIMESTAMPED/TITLED index (turns older than the manifest)
  recall_history(last=N | turns=[]) -> a specific turn's compact trace (action/observation/note)
  recall_history(..., full=true)    -> a turn's FULL stored slice (exact past state)
  recall_history(search="…")        -> FTS5 content search: THIS session's long tail + other sessions
Reaching back is expected, not a failure: the slice is bounded, so an earlier turn genuinely is not in
front of the model, and there is no automatic mechanism that can guess WHICH past turn the current
reasoning needs — only the model can. NON-ACCUMULATION (moat): a fetched turn is TRANSIENT — it enters
context for this loop only and is never written back into slice state (the slice is rebuilt from the
durable stores each turn); a single turn's fetches are bounded by DISTINCT_PER_TURN + the exact-repeat
redirect, so 'encourage recall' can never rebuild the transcript. Registered when memory is durable.
"""
from __future__ import annotations

import threading

from .text_utils import format_ts

INDEX_LIMIT = 40       # breadcrumbs shown by the bare index (a LOCATOR bound — titles/notes, not content)
OBS_TAIL = 300         # legacy-record fallback ONLY: per-observation tail when there is no stored markdown
DISTINCT_PER_TURN = 8  # backstop on DISTINCT turn-fetches per turn; repeats are redirected for free
# NO read-side CONTENT cap: a fetched turn is returned IN FULL. The bound is the SEAL, not a second cut at
# read — the archive already excerpts observations at SAVE time (episode._obs_excerpt), a fetched turn is
# TRANSIENT (enters context for this loop only, never written back to slice state, so recall can't rebuild
# the transcript across loops), and the physical context window + overflow is the size backstop for a
# deliberate sweep. The old 4000/8000 caps cut the distilled CONCLUSION — the one thing recall exists to
# return — because the conclusion is appended LAST in the markdown (bound = the seal, not a within-loop cut).

CAPTURE_BACK = ("\n\n↳ Now in context. Record what you need with a `note`, then continue — and "
                "fetch another turn from PAGED-OUT HISTORY whenever you need more.")


def _sig(args: dict):
    """Identity of a fetch, so an exact REPEAT can be redirected (the loop) while DISTINCT fetches
    (a real search — each returns new info) are allowed."""
    if args.get("turns"):
        turns = args["turns"]
        if isinstance(turns, (str, int)):
            turns = [turns]   # a scalar/string turn id is ONE number, not a char/digit iterable
        nums = set()
        for t in turns:
            try:
                nums.add(int(t))
            except (TypeError, ValueError):
                pass
        return ("turns", frozenset(nums), bool(args.get("full")))
    if args.get("last"):
        try:
            return ("last", int(args["last"]), bool(args.get("full")))
        except (TypeError, ValueError):
            return ("last", 5, bool(args.get("full")))   # MATCH the handler's non-numeric fallback (n=5) — else
            #                                              a malformed `last` records sig ('index',) and poisons
            #                                              the real index-fetch slot for the rest of the turn
    return ("index",)


def _short_ts(ts: str) -> str:
    return format_ts(ts)   # "06-16 12:30"


def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else "…" + s[-n:]


def render_index(lines: list[dict]) -> str:
    from .finding_types import badge, classify_finding   # item 14a: typed note badge in the index
    out = ["# CACHED HISTORY (index — fetch a turn with recall_history(turns=[N]) or last=N)"]
    for ln in lines:
        rec = ln.get("record", {})
        meta = rec.get("meta", {})
        failing = bool(meta.get("failing"))
        flag = " FAIL" if failing else ""
        nsteps = len(rec.get("steps", []))
        title = rec.get("title") or "(no title)"
        note = rec.get("note") or ""
        # type the breadcrumb's note so the model scans by KIND (decision / ruled-out / …)
        edited = bool(meta.get("files"))
        tag = badge(classify_finding(note, edited=edited, had_error=failing,
                                     resolved=not failing and meta.get("stop_reason") == "end_turn"))
        out.append(f"- turn {ln.get('turn')} · {_short_ts(ln.get('ts',''))} · [{ln.get('task_id','')}] "
                   f"{title[:60]} · {nsteps}st{flag}" + (f" — {tag}{note[:80]}" if note else ""))
    return "\n".join(out)


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


def render_full(lines: list[dict]) -> str:
    out = []
    for ln in lines:
        rec = ln.get("record", {})
        slices = [st.get("slice", "") for st in rec.get("steps", []) if st.get("slice")]
        body = (slices[-1] if slices else "")   # the turn's last reconstructed slice = its end state
        note = rec.get("note") or ""
        chunk = f"\n══ turn {ln.get('turn')} · {_short_ts(ln.get('ts',''))} · {rec.get('title') or ''}\n{body}"
        if note:                                # the agent's REPLY/conclusion lives in the note, NOT the seed
            chunk += f"\n\n## conclusion\n{note}"   # slice — without this, 'full' never returned the findings
        out.append(chunk)
    return "\n".join(out).strip() or "(no slice stored)"


def render_search(mine, cross) -> str:
    """Render a content search. THIS session's matching turns come WITH the exact fetch call — the
    model searched by content and now has the turn number, so the long tail past the manifest/index
    window is reachable without guessing a number. PAST sessions' FTS5 hits follow as read-only
    context (no in-session turn to fetch)."""
    out = []
    if mine:
        out.append("# THIS SESSION — content matches (page the full turn with the call shown)")
        for r in mine:
            out.append(f"- turn {r.handle}: {r.preview}  → recall_history(turns=[{r.handle}])")
    if cross:
        out.append("# CROSS-SESSION RECALL (past sessions — FTS5 over the durable episode index)")
        for r in cross:
            out.append(f"- [{r.handle}] {r.preview}")
    return "\n".join(out) if out else "No content matches found."


def make_history_tool(memory, session_id: str):
    """ToolEntry for recall_history, reading `memory`'s episodic cache for this session.

    Guardrail reins on REPETITION, not count — so a genuine search (distinct fetches, each returning
    new info) is never blocked, only the useless loop (re-fetching the same thing) is. An exact repeat
    gets a one-line redirect (no re-dump); distinct turn-fetches are allowed up to a generous backstop,
    past which the message points back at the cheap INDEX (which already lists every turn's title+note)
    rather than hard-blocking. Turn boundaries need no plumbing: the cache grows one record per turn,
    so a change in episode count resets the rein. The index fetch is free (the locator); only data
    drills (turns=/last=) count toward the backstop. Each served result carries a capture-back nudge."""
    from .pagetable import PageTable
    from .registry import ToolEntry
    _guards: dict = {}   # thread_id -> rein state; parallel explorers share this closure → isolate per thread
    # The ONE cross-session read path: PageTable's episode-xsession backend wraps
    # memory.search_episodes (the this-session read_episodes drill stays in this handler).
    pages = PageTable(memory=memory, session_id=session_id)

    def _handler(args: dict) -> str:
        guard = _guards.setdefault(threading.get_ident(), {"seen": -1, "served": set(), "distinct": 0})
        # content-search shape: search=... runs FTS5 over THIS session's long tail (turns past the
        # manifest/index window — reachable by content, not just by a turn number nobody knows) AND
        # past sessions. Checked first, no rein (each query is a real search returning new info).
        q = args.get("search")
        if isinstance(q, str) and q.strip():
            mine = pages.lookup(q.strip(), kind="episode-search-thissession", k=6)
            cross = pages.lookup(q.strip(), kind="episode-xsession", k=6)
            if not mine and not cross:
                return ("No content matches in this or past sessions for that query. Try different "
                        "keywords, or recall_history() (no args) for this session's full index.")
            return render_search(mine, cross) + CAPTURE_BACK
        lines = memory.read_episodes(session_id)
        if len(lines) != guard["seen"]:          # cache grew (or first call) → new turn → reset rein
            guard["seen"] = len(lines)
            guard["served"] = set()
            guard["distinct"] = 0
        if not lines:
            return "No cached history yet (this is an early turn)."
        sig = _sig(args)
        if sig in guard["served"]:               # exact repeat → redirect, don't re-dump (kills the loop)
            return ("You already pulled this earlier this turn (it's above). Fetch a DIFFERENT turn, "
                    "use last=N to sweep several at once, or act on what you have (record a note).")
        is_drill = sig[0] != "index"
        if is_drill and guard["distinct"] >= DISTINCT_PER_TURN:   # examined plenty → point at the index
            return (f"You've examined {guard['distinct']} different turns this turn. Use the index "
                    "(recall_history() — titles + notes for ALL turns) to pinpoint the right one, then "
                    "fetch just that turn — or proceed with what you have.")
        turns, last, full = args.get("turns"), args.get("last"), bool(args.get("full"))
        if not turns and not last:
            guard["served"].add(sig)
            return render_index(lines[-INDEX_LIMIT:]) + CAPTURE_BACK
        if turns:
            if isinstance(turns, (str, int)):
                turns = [turns]   # a scalar/string turn id is ONE number, never split into its digits ("23"→23)
            want = set()
            for t in turns:
                try:
                    want.add(int(t))
                except (TypeError, ValueError):
                    pass
            sel = [ln for ln in lines if ln.get("turn") in want]
        else:
            try:
                n = int(last)
            except (TypeError, ValueError):
                n = 5   # a non-numeric `last` must not raise — fall back to a small recent window
            sel = lines[-max(1, n):]   # clamp: a negative `last` would slice a too-broad window
        if not sel:
            return "No matching turns. Call recall_history() with no args for the index."
        guard["served"].add(sig)
        guard["distinct"] += 1
        return (render_full(sel) if full else render_trace(sel)) + CAPTURE_BACK

    schema = {"type": "function", "function": {
        "name": "recall_history",
        "description": (
            "Page an earlier turn of THIS session back into context — normal navigation, since the slice "
            "is bounded. The PAGED-OUT HISTORY section of your slice lists each earlier turn's number, "
            "title and note WITH the exact call to fetch it: copy that — {\"turns\":[N,...]} for the "
            "turn's actions/observations/notes (add {\"full\":true} for its full stored state), or "
            "{\"last\":N} for the most recent N. Call with NO args for the full index of turns older than "
            "the manifest. To find an old turn by CONTENT (this session or past ones) when you don't know "
            "its number, {\"search\":\"keywords\"} (FTS5 — AND/OR/quoted/prefix*). Reach back whenever an "
            "earlier turn holds something you need instead of re-deriving it; record what you find with a note."),
        "parameters": {"type": "object", "properties": {
            "last": {"type": "integer", "description": "fetch the most recent N turns (this session)"},
            "turns": {"type": "array", "items": {"type": "integer"},
                      "description": "fetch these specific turn numbers (from the index, this session)"},
            "full": {"type": "boolean", "description": "return the full stored slice instead of the compact trace"},
            "search": {"type": "string",
                       "description": "Content search (FTS5) over THIS session's earlier turns AND past "
                                      "sessions — find an old turn by what it was ABOUT when you don't know "
                                      "its number; this-session matches come with the call to page them back"},
        }}}}
    return ToolEntry(name="recall_history", schema=schema, handler=_handler, source="builtin")
