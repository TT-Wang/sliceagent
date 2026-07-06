"""history/ virtual files (HistoryFS) + search_history + read_episodes tests. No model, no pytest.
Run: python tests/test_history.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hippocampus import render_search, render_trace  # noqa: E402
from sliceagent.memory import NullMemory  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _rec(title, note, steps, failing=False):
    return {"title": title, "note": note, "steps": steps,
            "meta": {"failing": failing, "stop_reason": "end_turn", "files": []}}


def _memem():
    from sliceagent.memory import MememMemory
    m = MememMemory()
    m._vault = tempfile.mkdtemp()
    return m


@check
def append_and_read_roundtrip_with_ts_topic_title():
    try:
        m = _memem()
    except Exception:
        print("  (skip: memem not importable)"); return
    m.append_episode("s1", "t-aaa", 1, _rec("fix the parser", "root cause: off-by-one",
                                            [{"slice": "SLICE1", "action": [{"name": "read_file", "args": {"path": "p.py"}, "failing": False}],
                                              "observation": ["file contents"]}]))
    m.append_episode("s1", "t-bbb", 2, _rec("add range query", "done", [{"slice": "S2", "action": [], "observation": []}]))
    lines = m.read_episodes("s1")
    assert len(lines) == 2
    assert lines[0]["task_id"] == "t-aaa" and lines[0]["ts"] and lines[0]["record"]["title"] == "fix the parser"
    assert m.read_episodes("s1", limit=1)[0]["turn"] == 2          # most-recent N
    assert m.read_episodes("nope") == []                           # missing session → []


@check
def historyfs_over_durable_memem_roundtrip():
    """HistoryFS reads the SAME durable episodic cache the write side flushes — index + turn file."""
    try:
        m = _memem()
    except Exception:
        print("  (skip: memem not importable)"); return
    from sliceagent.hippocampus import HistoryFS
    m.append_episode("s1", "t1", 1, _rec("task one", "did a thing",
                     [{"slice": "", "action": [{"name": "run_command", "args": {"command": "pytest"},
                                                "failing": True}], "observation": ["FAILED 3 tests"]}]))
    fs = HistoryFS(m, "s1")
    assert "turn-1.md" in fs.read_file("history/index.md") and "task one" in fs.read_file("history/index.md")
    t1 = fs.read_file("history/turn-1.md")
    assert "run_command" in t1 and "FAILED 3 tests" in t1 and "did a thing" in t1   # action+obs+note


@check
def trace_has_no_read_cap_but_bounds_per_obs_at_the_seal():
    # NO read-side total cap: every requested turn comes back (a cap dropped whole turns / cut conclusions).
    # The bound is the SEAL — a legacy record's raw obs is tailed per-observation (OBS_TAIL), nothing dropped.
    lines = [{"task_id": "t", "turn": i, "ts": "2026-06-16T12:00:00",
              "record": _rec(f"turn{i}", "", [{"slice": "", "action": [{"name": "run", "args": {}, "failing": False}],
                                               "observation": ["x" * 5000]}])} for i in range(1, 20)]
    out = render_trace(lines)
    assert "turn 1" in out and "turn 19" in out and "truncated" not in out   # all turns, no total cap
    assert "x" * 5000 not in out                                             # raw obs bounded per-obs at the seal


@check
def trace_returns_the_full_conclusion_for_a_large_turn():
    # the 'can't locate my second finding' read-side bug: a big turn's conclusion (markdown tail) must
    # survive with no read cap — render_trace backs each history/turn-N.md file.
    concl = "Finding 6: tab links drop string[] params."
    big_md = "## seed\n" + ("filler " * 5000) + f"\n\n## conclusion\n{concl}"
    line = {"task_id": "t", "turn": 7, "ts": "2026-06-16T12:00:00",
            "record": {"title": "review page.tsx", "note": concl, "markdown": big_md,
                       "steps": [{"slice": "seed-slice-body"}], "meta": {}}}
    assert concl in render_trace([line]), "the conclusion at the markdown tail must survive (no read cap)"


class _FakeMem:
    def __init__(self, lines, hits=None):
        self._lines = lines
        self._hits = hits or []          # search_episodes results (for search_history tests)
    def read_episodes(self, session_id, *, limit=None):
        return self._lines
    def search_episodes(self, query, *, limit=8, exclude_session=None, only_session=None):
        return list(self._hits)          # ignores query — the test controls the hits directly


@check
def nullmemory_has_no_history():
    from sliceagent.hippocampus import HistoryFS
    m = NullMemory()
    assert m.read_episodes("s1") == []
    assert "no earlier turns yet" in HistoryFS(m, "s1").read_file("history/index.md")


# ── history/ virtual read-only namespace (HistoryFS + LocalToolHost routing) ──────────────────────────
def _hist_lines():
    return [
        {"task_id": "t1", "turn": 1, "ts": "2026-07-06T12:30:00",
         "record": _rec("fix the parser", "off-by-one in tokenizer",
                        [{"slice": "SL1", "action": [{"name": "read_file", "args": {"path": "p.py"}, "failing": False}],
                          "observation": ["contents"]}])},
        {"task_id": "t1", "turn": 2, "ts": "2026-07-06T12:40:00",
         "record": {"title": "provision box", "note": "pass",
                    "markdown": "# provision box\n## what happened\n- [run_command] ./provision.sh -> ok\n"
                                "    provisioned id: needle-pv-42\n",
                    "steps": [{"slice": "S2", "action": [], "observation": []}],
                    "meta": {"failing": False, "stop_reason": "end_turn", "files": []}}},
    ]


@check
def historyfs_index_lists_turns_as_files():
    from sliceagent.hippocampus import HistoryFS
    fs = HistoryFS(_FakeMem(_hist_lines()), "s")
    idx = fs.read_file("history/index.md")
    assert "turn-1.md" in idx and "turn-2.md" in idx           # each turn is a file
    assert "fix the parser" in idx and "provision box" in idx  # titles
    assert 'read_file("history/turn-<N>.md")' in idx           # tells the model how to read one
    assert fs.read_file("history") == idx and fs.read_file("history/") == idx   # dir/root ⇒ index


@check
def historyfs_turn_read_returns_the_seal_markdown():
    from sliceagent.hippocampus import HistoryFS
    fs = HistoryFS(_FakeMem(_hist_lines()), "s")
    t2 = fs.read_file("history/turn-2.md")
    assert "needle-pv-42" in t2 and "provision box" in t2      # the sealed turn's markdown, in full
    assert "no such turn" in fs.read_file("history/turn-99.md")
    assert "not a history file" in fs.read_file("history/notes.txt")


@check
def historyfs_listing_and_grep():
    from sliceagent.hippocampus import HistoryFS
    fs = HistoryFS(_FakeMem(_hist_lines()), "s")
    ls = fs.listing("history")
    assert "index.md" in ls and "turn-1.md" in ls and "turn-2.md" in ls
    g = fs.grep("needle-pv-\\d+")
    assert "history/turn-2.md:" in g and "needle-pv-42" in g   # ripgrep-shaped file:line:text
    assert "history/turn-2.md" in fs.grep("needle", output_mode="files_with_matches")
    assert "no matches" in fs.grep("zzz-not-present")
    assert "invalid regex" in fs.grep("(unclosed")


@check
def historyfs_grep_scopes_to_the_requested_path():
    # ripgrep semantics: grep of a specific turn file searches ONLY that file, not the whole namespace.
    from sliceagent.hippocampus import HistoryFS
    fs = HistoryFS(_FakeMem(_hist_lines()), "s")
    assert "needle-pv-42" in fs.grep("needle", path="history")                 # dir → all turns
    assert "needle-pv-42" in fs.grep("needle", path="history/turn-2.md")       # the turn that has it
    assert "no matches" in fs.grep("needle", path="history/turn-1.md")         # a turn that does NOT
    assert "no matches" in fs.grep("needle", path="history/turn-99.md")        # a non-existent turn


@check
def historyfs_normalizes_messy_paths():
    # _history_leaf must collapse stray '//' and './' so a slightly-malformed path still resolves the turn.
    from sliceagent.hippocampus import HistoryFS
    fs = HistoryFS(_FakeMem(_hist_lines()), "s")
    for p in ("history//turn-2.md", "history/./turn-2.md", "./history/turn-2.md", "history/turn-2.md/"):
        assert "needle-pv-42" in fs.read_file(p), p
    # a '..'-escape normalizes to a non-history file → safely misses (no traversal)
    assert "not a history file" in fs.read_file("history/../secret.md")


@check
def host_routes_read_list_grep_to_history():
    from sliceagent.tools import LocalToolHost
    from sliceagent.code_grep import make_grep_tool
    host = LocalToolHost(root=tempfile.mkdtemp())
    from sliceagent.hippocampus import HistoryFS
    host._history = HistoryFS(_FakeMem(_hist_lines()), "s")
    assert "turn-2.md" in host._t_read_file({"path": "history/index.md"})
    assert "needle-pv-42" in host._t_read_file({"path": "history/turn-2.md"})
    assert "turn-1.md" in host._t_list_files({"path": "history"})
    grep = make_grep_tool(host)
    assert "needle-pv-42" in grep.handler({"pattern": "needle-pv-\\d+", "path": "history"})
    # a normal path is NOT routed to history — listing "." serves the real (empty) workspace, not the index
    assert host._t_list_files({"path": "."}) == "(empty)"


@check
def host_rejects_writes_to_history():
    from sliceagent.tools import LocalToolHost
    host = LocalToolHost(root=tempfile.mkdtemp())
    from sliceagent.hippocampus import HistoryFS
    host._history = HistoryFS(_FakeMem(_hist_lines()), "s")
    for res in (host._t_edit_file({"path": "history/turn-1.md", "content": "x"}),
                host._t_append({"path": "history/new.md", "content": "x"}),
                host._t_str_replace({"path": "history/turn-1.md", "old_str": "a", "new_str": "b"})):
        assert getattr(res, "ok", True) is False and "read-only" in res, res


@check
def host_real_file_wins_over_virtual_history():
    # I2: a REAL on-disk file under history/ must never be shadowed by the virtual view (no lying about disk).
    from sliceagent.tools import LocalToolHost
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "history"), exist_ok=True)
    with open(os.path.join(d, "history", "real.md"), "w") as f:
        f.write("REAL ON DISK CONTENT")
    host = LocalToolHost(root=d)
    from sliceagent.hippocampus import HistoryFS
    host._history = HistoryFS(_FakeMem(_hist_lines()), "s")
    assert "REAL ON DISK CONTENT" in host._t_read_file({"path": "history/real.md"})   # real wins
    # a real history/ DIR exists → list_files shows the real dir (real wins), not the virtual listing
    assert "real.md" in host._t_list_files({"path": "history"})
    # a real file also can be WRITTEN (guard only rejects the virtual case)
    assert getattr(host._t_append({"path": "history/real.md", "content": "!"}), "ok", True) is not False


@check
def render_search_points_at_history_files():
    # this-session content hits come WITH the read_file("history/turn-N.md") call (not the deleted recall tool)
    from types import SimpleNamespace as NS
    mine = [NS(handle=3, preview="fixed the tokenizer off-by-one")]
    cross = [NS(handle="sess-9", preview="how retries were tuned last week")]
    out = render_search(mine, cross)
    assert 'read_file("history/turn-3.md")' in out and "recall_history" not in out
    assert "CROSS-SESSION" in out and "sess-9" in out


@check
def search_history_tool_empty_query_and_no_matches():
    from sliceagent.hippocampus import make_search_history_tool
    tool = make_search_history_tool(_FakeMem([]), "s")
    assert tool.schema["function"]["name"] == "search_history"
    assert "pass a 'query'" in tool.handler({})                       # empty query → usage
    out = tool.handler({"query": "nonexistent-token-xyz"})            # FakeMem returns no hits
    assert "No content matches" in out and 'history/index.md' in out  # points back at the file namespace


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
