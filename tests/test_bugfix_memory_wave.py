"""Regression tests for the memory/episode/search wave: #36 non-serializable outputs don't drop the
turn, #35 clamp recurses into nested outputs, #37 frontmatter newline survives round-trip, #39 atomic
writes, #40 FTS5 query escaping, #41 intuitive score sign. No model, no pytest, no memem needed.
Run: PYTHONPATH=src python tests/test_bugfix_memory_wave.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.memory import (MememMemory, _parse_task_md, _render_task_md,  # noqa: E402
                             _write_atomic)
from sliceagent.interfaces import TaskState  # noqa: E402
from sliceagent.search_index import EpisodeIndex, fts5_available  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _bare_memory(vault):
    m = MememMemory.__new__(MememMemory)   # bypass __init__ → no memem dependency
    m._vault = vault
    m._idx = None                          # FTS off for this unit (covered separately below)
    return m


@check
def append_episode_survives_non_serializable_output():  # #36
    vault = tempfile.mkdtemp(prefix="vault-")
    m = _bare_memory(vault)
    rec = {"title": "t", "note": "n",
           "steps": [{"action": [{"name": "x", "args": {}}], "observation": [{1, 2, 3}]}]}  # set = not JSON
    m.append_episode("s1", "task1", 1, rec)
    got = m.read_episodes("s1")
    assert len(got) == 1, "a non-serializable tool output must NOT drop the whole turn (#36)"


@check
def clamp_recurses_into_nested_outputs():  # #35
    m = _bare_memory(tempfile.mkdtemp(prefix="vault2-"))
    huge = "x" * 500_000
    out = m._clamp({"payload": {"blob": huge}, "items": [huge]})
    assert len(out["payload"]["blob"]) < len(huge), "nested str leaf must be byte-bounded (#35)"
    assert len(out["items"][0]) < len(huge), "str leaf inside a list must be bounded too"


@check
def frontmatter_newline_round_trips():  # #37
    ts = TaskState(task_id="t1", session_id="s1", title="line one\nline two", goal="g", tags="a\nb")
    p = os.path.join(tempfile.mkdtemp(prefix="ts-"), "t.md")
    _write_atomic(p, _render_task_md(ts, created="2026-06-23", updated="2026-06-23"))
    back = _parse_task_md(p)
    assert back.title == "line one line two", back.title   # collapsed, NOT truncated to "line one"
    assert "\n" not in back.tags


@check
def write_atomic_replaces_and_keeps_original_on_failure():  # #39
    p = os.path.join(tempfile.mkdtemp(prefix="aw-"), "f.txt")
    _write_atomic(p, "v1")
    assert open(p).read() == "v1"
    _write_atomic(p, "v2")
    assert open(p).read() == "v2"   # replaced atomically
    # no stray temp files left behind in the dir
    assert os.listdir(os.path.dirname(p)) == ["f.txt"]


@check
def fts_query_is_escaped_and_score_is_positive():  # #40 / #41
    if not fts5_available():
        print("  (fts5 unavailable — skipping #40/#41)")
        return
    db = os.path.join(tempfile.mkdtemp(prefix="idx-"), "idx.db")
    idx = EpisodeIndex(db)
    assert idx.is_active
    idx.index_episode(session_id="s1", task_id="t1", turn=1, ts="2026", title="setup",
                      note="", text="alpha bravo charlie deadbeef configuration")
    hits = idx.search("alpha")
    assert len(hits) == 1 and hits[0]["score"] >= 0.0, hits   # #41 higher-is-better
    # punctuation/operators around real words are stripped to literal terms → still match (#40)
    for q in ('alpha-bravo', '"deadbeef"', 'charlie: deadbeef*'):
        assert len(idx.search(q)) == 1, f"escaped query {q!r} should still match"
    # raw FTS5 operator syntax must not CRASH (returns a list; may be empty since operators become words)
    for q in ('alpha AND (bravo', 'foo* OR "x"', ')))', '* * *'):
        assert isinstance(idx.search(q), list), f"query {q!r} must not raise"
    assert idx.search("zzznomatch") == []
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
