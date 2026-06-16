"""recall_history — the model's bounded, on-demand valve into the COLD episodic cache.

The cache is never part of the slice (Markov by construction). This tool lets the model CHOOSE to
look back when reconstruction dropped something it needs — the explicit, bounded equivalent of a
transcript agent's "scroll up". Index-then-drill:
  recall_history()                  -> a cheap TIMESTAMPED/TITLED index of past turns to scan
  recall_history(last=N | turns=[]) -> a specific turn's compact trace (action/observation/note)
  recall_history(..., full=true)    -> a turn's FULL stored slice (exact past state)
Everything is BOUNDED (the model picks depth; the payload is capped) so a lookback can't silently
reintroduce the transcript. Heavy use is a signal the tiers under-reconstruct — a diagnostic, not
just a feature. Registered only when memory is durable (NullMemory/eval path never sees it).
"""
from __future__ import annotations

import json

INDEX_LIMIT = 40       # breadcrumbs shown by the bare index
TRACE_MAX = 4000       # total chars for a compact-trace fetch
FULL_MAX = 8000        # total chars for a full-slice fetch
OBS_TAIL = 300         # per-observation tail kept in a trace
DISTINCT_PER_TURN = 8  # generous backstop on DISTINCT turn-fetches; repeats are redirected for free

CAPTURE_BACK = ("\n\n↳ You now have this in context. Record what you need with a `note` and "
                "CONTINUE — only call recall_history again to fetch a DIFFERENT turn.")


def _sig(args: dict):
    """Identity of a fetch, so an exact REPEAT can be redirected (the loop) while DISTINCT fetches
    (a real search — each returns new info) are allowed."""
    if args.get("turns"):
        return ("turns", frozenset(int(t) for t in args["turns"]), bool(args.get("full")))
    if args.get("last"):
        return ("last", int(args["last"]), bool(args.get("full")))
    return ("index",)


def _short_ts(ts: str) -> str:
    return (ts or "")[5:16].replace("T", " ")   # "06-16 12:30"


def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else "…" + s[-n:]


def _arg_hint(args: dict) -> str:
    if not isinstance(args, dict):
        return ""
    for k in ("path", "command", "task_id", "goal", "query"):
        if args.get(k):
            return f"{k}={str(args[k])[:60]}"
    if args.get("code"):
        return "code=" + (str(args["code"]).strip().splitlines() or [""])[0][:60]
    extra = {k: v for k, v in args.items() if k != "note"}
    return (json.dumps(extra, ensure_ascii=False)[:60]) if extra else ""


def render_index(lines: list[dict]) -> str:
    out = ["# CACHED HISTORY (index — fetch a turn with recall_history(turns=[N]) or last=N)"]
    for ln in lines:
        rec = ln.get("record", {})
        meta = rec.get("meta", {})
        flag = " FAIL" if meta.get("failing") else ""
        nsteps = len(rec.get("steps", []))
        title = rec.get("title") or "(no title)"
        note = rec.get("note") or ""
        out.append(f"- turn {ln.get('turn')} · {_short_ts(ln.get('ts',''))} · [{ln.get('task_id','')}] "
                   f"{title[:60]} · {nsteps}st{flag}" + (f" — {note[:80]}" if note else ""))
    return "\n".join(out)


def render_trace(lines: list[dict], cap: int = TRACE_MAX) -> str:
    out, used = [], 0
    for ln in lines:
        rec = ln.get("record", {})
        head = f"\n── turn {ln.get('turn')} · {_short_ts(ln.get('ts',''))} · {rec.get('title') or ''}"
        block = [head]
        for st in rec.get("steps", []):
            for a, o in zip(st.get("action", []), st.get("observation", [])):
                fail = " ✗" if a.get("failing") else ""
                block.append(f"  • {a.get('name')}({_arg_hint(a.get('args', {}))}){fail} → {_tail(o, OBS_TAIL)}")
        if rec.get("note"):
            block.append(f"  ↳ note: {rec['note'][:200]}")
        chunk = "\n".join(block)
        if used + len(chunk) > cap:
            out.append("\n…[older turns truncated — narrow with turns=[…]]")
            break
        out.append(chunk)
        used += len(chunk)
    return "\n".join(out).strip() or "(no trace)"


def render_full(lines: list[dict], cap: int = FULL_MAX) -> str:
    out, used = [], 0
    for ln in lines:
        rec = ln.get("record", {})
        slices = [st.get("slice", "") for st in rec.get("steps", []) if st.get("slice")]
        body = (slices[-1] if slices else "")   # the turn's last reconstructed slice = its end state
        chunk = f"\n══ turn {ln.get('turn')} · {_short_ts(ln.get('ts',''))} · {rec.get('title') or ''}\n{body}"
        if used + len(chunk) > cap:
            out.append("\n…[truncated — fetch fewer turns for full slices]")
            break
        out.append(chunk)
        used += len(chunk)
    return "\n".join(out).strip() or "(no slice stored)"


def make_history_tool(memory, session_id: str):
    """ToolEntry for recall_history, reading `memory`'s episodic cache for this session.

    Guardrail reins on REPETITION, not count — so a genuine search (distinct fetches, each returning
    new info) is never blocked, only the useless loop (re-fetching the same thing) is. An exact repeat
    gets a one-line redirect (no re-dump); distinct turn-fetches are allowed up to a generous backstop,
    past which the message points back at the cheap INDEX (which already lists every turn's title+note)
    rather than hard-blocking. Turn boundaries need no plumbing: the cache grows one record per turn,
    so a change in episode count resets the rein. The index fetch is free (the locator); only data
    drills (turns=/last=) count toward the backstop. Each served result carries a capture-back nudge."""
    from .registry import ToolEntry
    guard = {"seen": -1, "served": set(), "distinct": 0}   # episode count, fetches served, drills

    def _handler(args: dict) -> str:
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
            want = {int(t) for t in turns}
            sel = [ln for ln in lines if ln.get("turn") in want]
        else:
            sel = lines[-int(last):]
        if not sel:
            return "No matching turns. Call recall_history() with no args for the index."
        guard["served"].add(sig)
        guard["distinct"] += 1
        return (render_full(sel) if full else render_trace(sel)) + CAPTURE_BACK

    schema = {"type": "function", "function": {
        "name": "recall_history",
        "description": (
            "Look back into THIS session's cached history (turns older than what RECENT shows). Call "
            "with NO args for a timestamped, titled INDEX of past turns; then fetch specifics with "
            "{\"turns\":[N,...]} or {\"last\":N} to get their compact trace (actions/observations/"
            "notes), or add {\"full\":true} for a turn's full slice. Use it to recover context you no "
            "longer have — then record what you learned with a note so you don't have to re-read."),
        "parameters": {"type": "object", "properties": {
            "last": {"type": "integer", "description": "fetch the most recent N turns"},
            "turns": {"type": "array", "items": {"type": "integer"},
                      "description": "fetch these specific turn numbers (from the index)"},
            "full": {"type": "boolean", "description": "return the full stored slice instead of the compact trace"},
        }}}}
    return ToolEntry(name="recall_history", schema=schema, handler=_handler, source="builtin")
