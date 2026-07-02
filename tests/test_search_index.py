"""Item 12 — cross-session FTS5 episode index. No model, no pytest.
Run: python tests/test_search_index.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.search_index import (  # noqa: E402
    EpisodeIndex, episode_searchable_text, fts5_available, make_episode_index,
)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _rec(title="", note="", actions=(), obs=(), files=()):
    return {"title": title, "note": note,
            "steps": [{"slice": "", "action": list(actions), "observation": list(obs)}],
            "meta": {"files": list(files)}}


@check
def fts5_is_available_here():
    # the test environment must support FTS5 for the rest to be meaningful
    assert fts5_available() is True


@check
def searchable_text_flattens_record():
    rec = _rec(title="fix parser", note="off-by-one in tokenizer",
               actions=[{"name": "edit_file", "args": {"path": "parser.py"}}],
               obs=["Error: index out of range"], files=["parser.py"])
    blob = episode_searchable_text(rec)
    assert "fix parser" in blob and "tokenizer" in blob
    assert "parser.py" in blob and "index out of range" in blob


@check
def index_and_search_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        idx = make_episode_index(os.path.join(d, "idx.db"))
        assert idx.is_active
        idx.index_episode(session_id="s-A", task_id="t1", turn=1, ts="2026-06-16T10:00:00Z",
                          title="auth refactor", note="switched to JWT",
                          text=episode_searchable_text(_rec(note="switched to JWT token flow")))
        idx.index_episode(session_id="s-B", task_id="t2", turn=1, ts="2026-06-16T11:00:00Z",
                          title="css tweak", note="padding fix",
                          text="adjusted padding on the header")
        hits = idx.search("JWT")
        assert len(hits) == 1 and hits[0]["session_id"] == "s-A"
        assert hits[0]["turn"] == 1 and hits[0]["title"] == "auth refactor"
        idx.close()


@check
def search_crosses_sessions_and_excludes_current():
    with tempfile.TemporaryDirectory() as d:
        idx = make_episode_index(os.path.join(d, "idx.db"))
        for sid in ("s-1", "s-2", "s-3"):
            idx.index_episode(session_id=sid, task_id="t", turn=1, ts="2026-06-16T10:00:00Z",
                              title="docker networking", note="bridge mode",
                              text="docker networking bridge mode notes")
        all_hits = idx.search("docker networking", limit=10)
        assert {h["session_id"] for h in all_hits} == {"s-1", "s-2", "s-3"}
        excl = idx.search("docker networking", limit=10, exclude_session="s-2")
        assert "s-2" not in {h["session_id"] for h in excl}
        assert len(excl) == 2
        idx.close()


@check
def reindex_is_idempotent_per_session_turn():
    with tempfile.TemporaryDirectory() as d:
        idx = make_episode_index(os.path.join(d, "idx.db"))
        for _ in range(3):  # same (session, turn) indexed 3x → must remain ONE row
            idx.index_episode(session_id="s-X", task_id="t", turn=2, ts="2026-06-16T10:00:00Z",
                              title="dup", note="n", text="kafka consumer lag")
        hits = idx.search("kafka")
        assert len(hits) == 1
        idx.close()


@check
def bad_query_and_empty_query_return_empty():
    with tempfile.TemporaryDirectory() as d:
        idx = make_episode_index(os.path.join(d, "idx.db"))
        idx.index_episode(session_id="s", task_id="t", turn=1, ts="t", title="x", note="",
                          text="hello world")
        assert idx.search("") == []
        assert idx.search('"unterminated') == []   # malformed FTS5 → [] not a raise
        idx.close()


@check
def inactive_index_is_a_noop():
    # simulate FTS5-unavailable: an index whose connection never opened
    idx = EpisodeIndex.__new__(EpisodeIndex)
    idx.is_active = False
    idx._con = None
    idx.index_episode(session_id="s", task_id="t", turn=1, ts="t", title="x", note="", text="y")
    assert idx.search("x") == []


@check
def recency_breaks_ties_among_comparable_hits():
    # the live-use bug: "no.3 findings for sliceagent" recalled the OLDER 'investigate sliceagent' turn over
    # the NEWER 'core loop review' turn because BM25 is purely lexical. Among comparably-relevant hits the
    # recency blend must now prefer the more recent turn (the latest review).
    with tempfile.TemporaryDirectory() as d:
        idx = make_episode_index(os.path.join(d, "idx.db"))
        sid = "s-1"
        idx.index_episode(session_id=sid, task_id="t", turn=3, ts="2026-06-25T11:49:00Z",
                          title="investigate the sliceagent project", note="findings about sliceagent",
                          text="sliceagent findings investigation sliceagent project findings notable")
        idx.index_episode(session_id=sid, task_id="t", turn=8, ts="2026-06-25T11:55:00Z",
                          title="close review of the core agent loop", note="sliceagent loop findings",
                          text="sliceagent findings review core agent loop sliceagent guardrail findings")
        hits = idx.search("sliceagent findings", only_session=sid, limit=5)
        assert hits and hits[0]["turn"] == 8, [h["turn"] for h in hits]   # newest review wins the tie
        idx.close()


@check
def recency_does_not_override_stronger_relevance():
    # the moat guard: recency is only a tie-break WITHIN the relevance band — a clearly stronger (but
    # older) lexical match must still beat a weak recent one, so recency can never smuggle in noise.
    with tempfile.TemporaryDirectory() as d:
        idx = make_episode_index(os.path.join(d, "idx.db"))
        sid = "s-1"
        idx.index_episode(session_id=sid, task_id="t", turn=1, ts="2026-06-20T10:00:00Z",
                          title="jwt auth", note="jwt refresh",
                          text="jwt auth token refresh jwt rotation jwt secret findings jwt jwt")
        idx.index_episode(session_id=sid, task_id="t", turn=2, ts="2026-06-25T10:00:00Z",
                          title="css padding", note="header", text="padding header findings")
        hits = idx.search("jwt findings", only_session=sid, limit=5)
        assert hits[0]["turn"] == 1, [h["turn"] for h in hits]   # strong OLD match beats weak recent
        idx.close()


@check
def memory_search_episodes_noop_without_index():
    # NullMemory must expose search_episodes returning [] (contract parity)
    from sliceagent.memory import NullMemory
    assert NullMemory().search_episodes("anything") == []


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
