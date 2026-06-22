"""Within-session content recall: recall_history(search="…") must find an OLD turn of THIS session
by what it was ABOUT — closing the long-tail gap where a turn past the manifest/index window was
reachable only by a turn number nobody knew. Two layers:
  1. WIRING (no FTS5 dep): the PageTable backend + render_search produce turn-numbered hits that come
     WITH the exact recall_history(turns=[N]) call, and only_session/exclude_session scope correctly.
  2. FTS5 (guarded): the real EpisodeIndex.only_session filter restricts to one session; exclude drops it.
No model, no pytest. Run: PYTHONPATH=src python tests/test_recall_search.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.history import render_search                 # noqa: E402
from memagent.pagetable import PageTable                   # noqa: E402
from memagent.search_index import EpisodeIndex, fts5_available  # noqa: E402

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
    pt = PageTable(memory=_FakeMem(_ROWS), exclude_session="S")   # exclude_session carries the current sid
    mine = pt.lookup("database", kind="episode-search-thissession", k=6)
    # only THIS session (S) matches the content "database" — turn 7. OTHER session is filtered out.
    assert [r.handle for r in mine] == ["7"], [r.handle for r in mine]
    assert all(r.kind == "episode-search-thissession" for r in mine)


@check
def xsession_search_still_excludes_this_session():
    pt = PageTable(memory=_FakeMem(_ROWS), exclude_session="S")
    cross = pt.lookup("database", kind="episode-xsession", k=6)
    # cross-session must NOT leak THIS session's turns; only OTHER's turn 9 surfaces.
    assert cross and all("OTHER" in r.handle for r in cross), [r.handle for r in cross]
    assert all("S · turn 7" not in r.handle for r in cross)


@check
def render_search_emits_the_exact_fetch_call_for_this_session_hits():
    pt = PageTable(memory=_FakeMem(_ROWS), exclude_session="S")
    mine = pt.lookup("database", kind="episode-search-thissession", k=6)
    cross = pt.lookup("database", kind="episode-xsession", k=6)
    out = render_search(mine, cross)
    # the model searched by CONTENT and gets back the COPY-PASTE call — no turn number to guess.
    assert "recall_history(turns=[7])" in out, out
    assert "THIS SESSION" in out and "CROSS-SESSION" in out, out


@check
def empty_query_is_safe():
    pt = PageTable(memory=_FakeMem(_ROWS), exclude_session="S")
    assert pt.lookup("   ", kind="episode-search-thissession", k=6) == []
    assert pt.lookup(None, kind="episode-search-thissession", k=6) == []


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
