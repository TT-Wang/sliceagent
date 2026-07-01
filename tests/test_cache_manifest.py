"""PAGED-OUT HISTORY manifest — the cache made VISIBLE so the model CALLS recall_history.

The active-ask channel was dead because the episodic cache had no manifest in the slice: the model
calls read_file because REPO MAP advertises paths, but never called recall_history because nothing
advertised the cache. These checks prove the manifest now renders as a typed region (locators +
copy-pasteable fetch call), is BOUNDED (moat), carries a payoff breadcrumb for every turn (the
pin/view-killer was invisible payoff), leaks NO content bodies, and self-suppresses with no durable
cache. Deterministic — no model, no disk, no memem (a tiny durable-memory stub).
Run: PYTHONPATH=src python tests/test_cache_manifest.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.memory import NullMemory                              # noqa: E402
from memagent.pagetable import PageTable                           # noqa: E402
from memagent.regions import MANIFEST_TURNS, render_cache_manifest # noqa: E402
from memagent.slice import Slice, make_build_slice                 # noqa: E402
from memagent.tools import LocalToolHost                           # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _line(turn, title, note="", files=None, failing=False, steps=None):
    """One episodic line dict, the shape memory.read_episodes returns (see memory.append_episode)."""
    rec = {"title": title, "note": note, "steps": steps or [],
           "meta": {"failing": failing, "files": files or [], "stop_reason": "end_turn"}}
    return {"v": 1, "session_id": "s1", "task_id": "t1", "turn": turn, "ts": "2026-06-18T12:00:00", "record": rec}


class _DurableMem:
    """Minimal durable memory: only read_episodes is needed for the manifest (the single read seam)."""
    is_durable = True
    def __init__(self, lines): self._lines = lines
    def read_episodes(self, session_id, *, limit=None):
        out = self._lines if session_id == "s1" else []
        return out[-limit:] if limit else out
    def recall(self, *a, **k): return []
    def search_episodes(self, *a, **k): return []


def _build_user(mem, session_id="s1", tmp=None):
    """Build the slice user-message string via the real make_build_slice seam."""
    wd = tmp or os.path.join(os.path.dirname(__file__), "..")
    state = Slice(); state.reset("rename Config.load to from_file everywhere")
    tools = LocalToolHost(os.path.abspath(wd))
    build = make_build_slice(state, tools, None, mem, state.goal, session_id)
    return build()[1]["content"]


# ── the manifest renders as a region, with copy-pasteable fetch calls ─────────
@check
def manifest_renders_with_fetch_calls():
    mem = _DurableMem([
        _line(1, "find Config.load callers", note="3 callers: api.py:40, cli.py:88, test_cfg.py:12"),
        _line(2, "read the settings schema", note="schema rejects null timeout"),
        _line(3, "repro the KeyError", note="fails only when MEMAGENT_VAULT unset", failing=True),
    ])
    user = _build_user(mem)
    assert "# PAGED-OUT HISTORY" in user, "manifest region missing from slice"
    # each turn is a locator line ENDING in its own fetch call (the copy-paste win)
    for n in (1, 2, 3):
        assert f"recall_history(turns=[{n}])" in user, f"turn {n} fetch call missing"
    assert "3 callers: api.py:40" in user, "the payoff breadcrumb (note) is missing"
    assert "· FAIL" in user, "failing turn not flagged"


# ── bounded to MANIFEST_TURNS (the moat: constant size regardless of session length) ──
@check
def manifest_bounded_with_older_tail():
    lines = [_line(i, f"turn {i} work", note=f"did thing {i}") for i in range(1, 13)]  # 12 turns
    user = _build_user(_DurableMem(lines))
    shown = [n for n in range(1, 13) if f"recall_history(turns=[{n}])" in user]
    assert len(shown) == MANIFEST_TURNS, f"expected {MANIFEST_TURNS} locators, got {len(shown)}: {shown}"
    assert shown == list(range(13 - MANIFEST_TURNS, 13)), f"must show the LAST {MANIFEST_TURNS}: {shown}"
    older = 12 - MANIFEST_TURNS
    assert f"{older} earlier turn(s)" in user, "the '+N earlier' tail is missing"
    assert "recall_history() for the full index" in user, "older tail should point at the bare index"


# ── every line carries a payoff breadcrumb, even with no note (adoption requirement) ──
@check
def breadcrumb_falls_back_to_files_then_actions():
    steps = [{"slice": "", "observation": ["..."],
              "action": [{"name": "grep", "args": {"query": "Config.load"}, "failing": False},
                         {"name": "read_file", "args": {"path": "pkg/api.py"}, "failing": False}]}]
    mem = _DurableMem([
        _line(1, "edit pass", note="", files=["pkg/api.py", "pkg/cli.py"]),  # no note -> files
        _line(2, "explore pass", note="", files=[], steps=steps),            # no note/files -> actions
    ])
    user = _build_user(mem)
    assert "edited: api.py, cli.py" in user, "breadcrumb must fall back to edited files"
    assert "did: grep Config.load" in user, "breadcrumb must fall back to the turn's actions"


# ── moat: the manifest is LOCATORS, never content bodies (no step observations leak) ──
@check
def manifest_leaks_no_content_bodies():
    huge = "SENTINEL_BODY_" + "x" * 5000   # a big observation that must NOT enter the slice
    steps = [{"slice": "WHOLE_PAST_SLICE_" + "y" * 5000,
              "action": [{"name": "read_file", "args": {"path": "big.py"}, "failing": False}],
              "observation": [huge]}]
    user = _build_user(_DurableMem([_line(1, "read big file", note="big.py is 5k lines", steps=steps)]))
    assert "# PAGED-OUT HISTORY" in user
    assert "SENTINEL_BODY_" not in user, "observation body leaked into the manifest (moat violation)"
    assert "WHOLE_PAST_SLICE_" not in user, "a past slice body leaked into the manifest (moat violation)"
    assert "big.py is 5k lines" in user, "the bounded note breadcrumb should still show"


# ── self-suppress when there is no durable cache (eval/NullMemory path unchanged) ─────
@check
def nullmemory_renders_no_manifest():
    user = _build_user(NullMemory())
    assert "# PAGED-OUT HISTORY" not in user, "NullMemory must produce no manifest"


@check
def empty_session_renders_no_manifest():
    user = _build_user(_DurableMem([]), session_id="other")  # no episodes for this session id
    assert "# PAGED-OUT HISTORY" not in user


# ── PageTable is the single read seam; it returns locator-only PageRefs ───────────────
@check
def pagetable_thissession_returns_locator_refs():
    lines = [_line(i, f"t{i}", note=f"n{i}") for i in range(1, 12)]  # 11 turns
    refs = PageTable(memory=_DurableMem(lines)).lookup("s1", kind="episode-thissession", k=MANIFEST_TURNS)
    assert len(refs) == MANIFEST_TURNS + 1, f"expected {MANIFEST_TURNS} + older sentinel, got {len(refs)}"
    assert refs[-1].handle == "…older", "last ref must be the older-tail sentinel"
    assert all(r.kind == "episode-thissession" for r in refs)
    # locators only: the preview is a one-line body, never a step/observation dump
    assert all("\n" not in r.preview for r in refs[:-1]), "a locator preview must be a single line"
    # render is a no-op on empty
    assert render_cache_manifest([]) == ""
    assert PageTable(memory=NullMemory()).lookup("s1", kind="episode-thissession", k=8) == []


@check
def pagetable_memory_lessons_unifies_recall():
    """The RELEVANT MEMORY recall now flows through the ONE read seam (kind='memory-lessons'),
    distinct from raw episode-xsession. PageRefs wrap recall's Snippets; render is byte-identical
    to the former render_memory(memory.recall(...)) path; absent/empty/skip all collapse to ''."""
    from memagent.interfaces import Snippet
    from memagent.slice import render_memory

    class _LessonMem:
        def recall(self, q, k=6, paths=None):
            return [Snippet(path="mem://a", text="lesson  one", score=0.9),
                    Snippet(path="mem://b", text="lesson two", score=0.4)]
        def read_episodes(self, *a, **k): return []
        def search_episodes(self, *a, **k): return []

    refs = PageTable(memory=_LessonMem()).lookup("goal", kind="memory-lessons", k=6)
    assert [r.kind for r in refs] == ["memory-lessons", "memory-lessons"]
    assert [r.handle for r in refs] == ["mem://a", "mem://b"]      # Snippet.path -> handle
    assert [r.preview for r in refs] == ["lesson  one", "lesson two"]  # RAW text -> preview
    # render matches the OLD Snippet path exactly (one_line collapses ws, wrap_untrusted fences once)
    rendered = render_memory(refs)
    assert "- lesson one" in rendered and "- lesson two" in rendered
    assert rendered.startswith("<untrusted-data kind=\"memory\">")
    # empty / absent / skip all suppress the tier (== the old `memory is None -> ""` branch)
    assert render_memory([]) == ""
    assert PageTable(memory=None).lookup("g", kind="memory-lessons", k=6) == []
    assert PageTable(memory=_LessonMem()).lookup("g", kind="memory-lessons", k=0) == []


@check
def truncated_prior_reply_advertises_recall_so_the_model_does_not_confabulate():
    """The 'explain item 2' bug: a long prior reply is stored in the RECENT CONVERSATION ring as an
    800-char gist. If the ring doesn't SIGNAL the cut + how to page the rest, the model reads the gist as
    the whole reply and confabulates the part it can't see (a cross-turn-continuity failure = the moat).
    Fix: a truncated ring reply carries a recall_history(last=K) marker, and that call returns the full text."""
    from memagent.episode import turn_markdown
    from memagent.events import AssistantText
    from memagent.history import make_history_tool
    from memagent.slice import record_user, slice_sink

    item2 = "2. lib/pipeline.ts:131,142 — jobs_scored column used for the jobs_updated value."
    report = ("Bug Hunt Report\n\n1. queries.ts:233 build-blocker. "
              + ("long item-1 detail that pushes item 2 well past the gist cap. " * 20)
              + "\n\n" + item2 + "\n\n3. llm.ts:87 drops systemPrompt.\n")

    class Mem:
        is_durable = True
        def __init__(self): self.eps = []
        def append_episode(self, s, t, turn, rec):
            self.eps.append({"v": 1, "session_id": s, "task_id": t, "turn": turn, "ts": "2026-07-01T12:00:00", "record": rec})
        def read_episodes(self, s, *, limit=None):
            out = [e for e in self.eps if e["session_id"] == s]; return out[-limit:] if limit else out
        def episode_manifest(self, s, k):
            out = [e for e in self.eps if e["session_id"] == s]; return out[-k:], len(out)
        def recall(self, *a, **k): return []
        def search_episodes(self, *a, **k): return []

    mem, sid = Mem(), "s1"
    st = Slice(); st.reset("do a bug hunt"); sink = slice_sink(st)
    record_user(st, "do a bug hunt"); sink(AssistantText(report))    # turn 1: long report
    mem.append_episode(sid, "t1", 1, {"title": "bug hunt", "note": report,
        "steps": [{"slice": "", "action": [], "observation": []}],
        "markdown": turn_markdown("bug hunt", [{"slice": "", "action": [], "observation": []}], report,
                                  {"files": [], "stop_reason": "end_turn"}),
        "meta": {"failing": False, "files": [], "stop_reason": "end_turn"}})

    record_user(st, "explain item 2")                                # turn 2: build the slice
    tools = LocalToolHost(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    user = make_build_slice(st, tools, None, mem, "explain item 2", sid)()[1]["content"]

    # item 2 is NOT in the static slice (it was cut from the gist) — recall is REQUIRED to answer
    assert "jobs_scored" not in user, "test premise broken: item 2 should be past the gist cap"
    # ...but the ring advertises the exact recall call, so the model pages it back instead of guessing
    assert "recall_history(last=1)" in user, "truncated ring reply must advertise recall_history(last=1)"
    # and following that call returns the full reply, item 2 included
    out = make_history_tool(mem, sid).handler({"last": 1})
    assert "jobs_scored" in out and "131,142" in out, "recall_history(last=1) must return item 2's full text"


@check
def truncated_finding_advertises_recall_instead_of_silently_dropping_content():
    # SECOND, separate instance of the same bug class: a long AssistantText (e.g. a multi-item bug-hunt
    # report) also folds into a FINDINGS entry via record_note(text, source="claim") — cut to
    # MAX_FINDING_CHARS=300 with NO signal. TT hit this for real: 3 filler turns pushed the bug-hunt reply
    # out of the RECENT CONVERSATION ring entirely, leaving ONLY this findings fragment — with no marker,
    # the model saw a snippet of bug #1 and FABRICATED 3 replacement bugs instead of recalling the rest.
    # Findings carry no turn number, so the fix points at the two GENERAL recall paths (the manifest, or
    # recall_history(search=...)) rather than a specific turns=[N] call.
    from memagent.slice import Slice, record_note

    long_report = ("Bug Hunt: lib/db.ts\n\n" + "1. (BUG) jobs_updated column has broken indentation. " * 6
                   + "\n\n2. (BUG) archetypeCounts() missing bd, finance, strategy. Build-breaking.\n"
                   + "\n3. (BUG) no closeDb() on the error path.\n\n4. (BUG) duplicate closeDb() calls.\n")
    from memagent.text_utils import normalize_ws
    assert len(normalize_ws(long_report)) > 300, "test premise broken: the NORMALIZED report must exceed MAX_FINDING_CHARS"

    s = Slice(); s.reset("bug hunt lib/db.ts")
    is_new = record_note(s, long_report, source="claim")
    assert is_new and s.findings, "a genuinely new claim must be recorded"
    stored = s.findings[-1]

    # bug #2's specifics (past the cut) must NOT silently appear as if part of the visible fragment
    assert "archetypeCounts" not in stored, "test premise broken: bug #2 should be past the cut"
    # the stored finding must clearly mark itself as partial and point at BOTH real recall paths
    assert "PARTIAL" in stored, "a truncated finding must say it is partial, not silently drop the rest"
    assert "PAGED-OUT HISTORY" in stored and "recall_history(search=" in stored, stored
    assert "don't guess" in stored, "must explicitly warn against re-deriving instead of recalling"
    # a SHORT note that fits within the cap must be stored VERBATIM, with no marker (no false positives)
    s2 = Slice(); record_note(s2, "a short claim under the cap", source="claim")
    assert s2.findings[-1] == "a short claim under the cap", s2.findings


@check
def truncated_user_report_also_advertises_recall():
    # THIRD site sharing the same class of bug: capture_user_report stores the user's OWN detailed bug
    # report (verbatim, bounded to MAX_REPORT_CHARS=280) as the OPEN USER REPORT blocker. The capturing
    # turn also shows the message in full via CURRENT REQUEST, but a LATER turn only sees this bounded
    # field — if it was silently cut, part of the user's own spec of what's broken would be lost with no
    # recovery path. Both this and the findings fix now share the SAME _cut_with_recall_marker helper.
    from memagent.regions import capture_user_report
    from memagent.slice import Slice

    long_report = "it's broken - " + ("when I click X, Y happens instead of Z. " * 8)
    from memagent.text_utils import normalize_ws
    assert len(normalize_ws(long_report)) > 280, "test premise broken: must exceed MAX_REPORT_CHARS"

    s = Slice(); s.reset("x")
    assert capture_user_report(s, long_report) is True, "a failure-report message must be captured"
    assert "PARTIAL" in s.open_report and "don't guess" in s.open_report, s.open_report
    assert "recall_history(search=" in s.open_report, s.open_report

    # a short report fits verbatim, no marker
    s2 = Slice(); s2.reset("y")
    short = "it's broken - the button doesn't work"
    capture_user_report(s2, short)
    assert s2.open_report == short, s2.open_report


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
