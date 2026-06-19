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
        def recall(self, q, k=6):
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


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
