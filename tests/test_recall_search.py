"""Within-session content recall: search_history("…") must find an OLD turn of THIS session by what it
was ABOUT — closing the long-tail gap where a turn past the manifest/index window was reachable only by
a turn number nobody knew. Two layers:
  1. WIRING (no FTS5 dep): the PageTable backend + render_search produce turn-numbered hits that come
     WITH the exact read_file("history/turn-N.md") call, and only_session/exclude_session scope correctly.
  2. FTS5 (guarded): the real EpisodeIndex.only_session filter restricts to one session; exclude drops it.
No model, no pytest. Run: PYTHONPATH=src python tests/test_recall_search.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hippocampus import render_search                 # noqa: E402
from sliceagent.pagetable import PageTable                   # noqa: E402
from sliceagent.search_index import EpisodeIndex, _fts_match_query, fts5_available  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _FakeMem:
    """Honors the only_session / exclude_session contract over canned rows, plus a content match —
    exactly what MememMemory.search_episodes promises, with zero FTS5/vault setup."""
    def __init__(self, rows):
        self.rows = rows

    def search_episodes(self, query, *, limit=5, exclude_session=None, only_session=None):
        out = []
        for r in self.rows:
            if exclude_session and r["session_id"] == exclude_session:
                continue
            if only_session and r["session_id"] != only_session:
                continue
            hay = f"{r.get('title','')} {r.get('note','')} {r.get('snippet','')}".lower()
            if query.lower() in hay:
                out.append(r)
        return out[:limit]


_ROWS = [
    {"session_id": "S", "turn": 7, "title": "database schema", "note": "chose postgres + jsonb",
     "snippet": "we picked «postgres»", "score": 1.0, "ts": None},
    {"session_id": "S", "turn": 3, "title": "logging setup", "note": "structured logs", "snippet": "",
     "score": 0.5, "ts": None},
    {"session_id": "OTHER", "turn": 9, "title": "database migration", "note": "alembic",
     "snippet": "ran «database» migration", "score": 0.8, "ts": None},
]


@check
def thissession_search_returns_turn_numbered_hits_only_for_this_session():
    pt = PageTable(memory=_FakeMem(_ROWS), session_id="S")        # the CURRENT session id
    mine = pt.lookup("database", kind="episode-search-thissession", k=6)
    # only THIS session (S) matches the content "database" — turn 7. OTHER session is filtered out.
    assert [r.handle for r in mine] == ["7"], [r.handle for r in mine]
    assert all(r.kind == "episode-search-thissession" for r in mine)


@check
def xsession_search_still_excludes_this_session():
    pt = PageTable(memory=_FakeMem(_ROWS), session_id="S")
    cross = pt.lookup("database", kind="episode-xsession", k=6)
    # cross-session must NOT leak THIS session's turns; only OTHER's turn 9 surfaces.
    assert cross and all("OTHER" in r.handle for r in cross), [r.handle for r in cross]
    assert all("S · turn 7" not in r.handle for r in cross)


@check
def render_search_emits_the_exact_fetch_call_for_this_session_hits():
    pt = PageTable(memory=_FakeMem(_ROWS), session_id="S")
    mine = pt.lookup("database", kind="episode-search-thissession", k=6)
    cross = pt.lookup("database", kind="episode-xsession", k=6)
    out = render_search(mine, cross)
    # the model searched by CONTENT and gets back the COPY-PASTE read_file call — no turn number to guess.
    assert 'read_file("history/turn-7.md")' in out, out
    assert "THIS SESSION" in out and "CROSS-SESSION" in out, out


@check
def empty_query_is_safe():
    pt = PageTable(memory=_FakeMem(_ROWS), session_id="S")
    assert pt.lookup("   ", kind="episode-search-thissession", k=6) == []
    assert pt.lookup(None, kind="episode-search-thissession", k=6) == []


@check
def thissession_search_fails_closed_without_a_session():
    # REGRESSION (the cross-session leak the review caught): when PageTable has no current session, a
    # "this-session" content search must return NOTHING — it must NOT fall through to only_session=None
    # and leak EVERY session's turns. (Old code overloaded one field as both exclude & only → leaked.)
    pt = PageTable(memory=_FakeMem(_ROWS))                  # no session_id
    assert pt.lookup("database", kind="episode-search-thissession", k=6) == [], "within-session must fail closed"


@check
def fts5_only_session_filter_is_real():
    # Guarded: only runs where sqlite has FTS5. Proves the SQL-layer scope, not just the fake.
    if not fts5_available():
        print("  (skipped: no FTS5 in this sqlite)")
        return
    idx = EpisodeIndex(":memory:")
    idx.index_episode(session_id="S", task_id="t", turn=7, ts="", title="database schema",
                      note="postgres jsonb", text="we chose postgres for the schema")
    idx.index_episode(session_id="S", task_id="t", turn=3, ts="", title="logging", note="",
                      text="structured logging setup")
    idx.index_episode(session_id="OTHER", task_id="t2", turn=9, ts="", title="db migration",
                      note="", text="ran a postgres database migration")
    only = idx.search("postgres", only_session="S")
    assert {h["turn"] for h in only} == {7}, only          # within THIS session, by content
    excl = idx.search("postgres", exclude_session="S")
    assert all(h["session_id"] != "S" for h in excl) and excl, excl   # cross drops S, keeps OTHER
    idx.close()


# --- OR-join + relative floor (the 'can't locate my second finding' production bug) --------------
@check
def query_tokens_are_or_joined_not_and_joined():
    # AND-join meant one absent token (the ordinal 'second') zeroed the whole result. OR is recall-correct.
    assert _fts_match_query("a b c") == '"a" OR "b" OR "c"', _fts_match_query("a b c")
    assert _fts_match_query("solo") == '"solo"'
    assert _fts_match_query("") == ""


@check
def or_join_finds_the_review_despite_ordinal_meta_terms():
    if not fts5_available():
        print("  (skipped: no FTS5)"); return
    idx = EpisodeIndex(":memory:")
    idx.index_episode(session_id="S", task_id="t", turn=7, ts="", title="review page.tsx",
        note="Finding 2: invalid query-string archetype/coding values are blindly cast in page.tsx.",
        text="app/page.tsx casts arrayParam to RoleArchetype; archetypeCounts missing bd finance strategy.")
    idx.index_episode(session_id="S", task_id="t", turn=8, ts="", title="unrelated",
        note="we discussed the deployment pipeline and CI config.", text="ci yaml deploy kube")
    # 'second'/'your'/'is' are NOT in the review text → AND-join returned nothing (the bug); OR-join finds it
    hits = idx.search("what is your second finding for page.tsx", only_session="S")
    turns = [h["turn"] for h in hits]
    assert 7 in turns and hits[0]["turn"] == 7, f"OR-join must surface + rank the review turn; got {turns}"
    idx.close()


@check
def relative_floor_trims_the_weak_single_term_tail():
    if not fts5_available():
        print("  (skipped: no FTS5)"); return
    idx = EpisodeIndex(":memory:")
    idx.index_episode(session_id="S", task_id="t", turn=1, ts="", title="strong",
                      note="alpha beta gamma delta page", text="alpha beta gamma delta page")
    for i in range(2, 6):
        idx.index_episode(session_id="S", task_id="t", turn=i, ts="", title=f"weak{i}",
                          note="page", text="page sidebar layout grid")
    hits = idx.search("alpha beta gamma delta page", only_session="S")
    assert hits and hits[0]["turn"] == 1, "the strong multi-term match must rank first"
    assert len(hits) < 5, f"the relative floor should trim the weak single-term tail; kept {len(hits)}"
    idx.close()


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
