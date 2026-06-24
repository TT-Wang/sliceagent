"""The seal saves each turn's slice into the cache as a clean MARKDOWN snapshot, and recall_history
returns it smoothly (read a past turn like opening a doc). No model. Run:
  PYTHONPATH=src python tests/test_seal_markdown.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.episode import make_episode_sink, turn_markdown  # noqa: E402
from memagent.history import make_history_tool  # noqa: E402
from memagent.events import SliceBuilt, ToolResult, TurnEnd  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Cache:
    is_durable = True
    def __init__(self):
        self._eps = {}
    def append_episode(self, sid, tid, turn, record):
        self._eps.setdefault(sid, []).append({"turn": turn, "ts": "", "task_id": tid, "record": record})
    def read_episodes(self, sid, *, limit=None):
        out = self._eps.get(sid, [])
        return out[-limit:] if limit else out
    def recall(self, *a, **k):
        return []
    def search_episodes(self, *a, **k):
        return []


def _run_a_turn(sink):
    sink(SliceBuilt(rendered="# OPEN FILES\n..."))
    sink(ToolResult(name="read_file", args={"path": "calc.py"}, output="def add(a,b): return a+b", failing=False))
    sink(ToolResult(name="str_replace", args={"path": "calc.py", "note": "added subtract"}, output="edited", failing=False))
    sink(ToolResult(name="run_command", args={"command": "pytest -q"}, output="2 passed", failing=False))
    sink(TurnEnd("end_turn", 3, {"prompt_tokens": 4000, "completion_tokens": 200}))


@check
def seal_stores_clean_markdown():
    mem = _Cache()
    sink = make_episode_sink(mem, session_id="s", task_id_fn=lambda: "t", title_fn=lambda: "Add subtract to calc.py")
    _run_a_turn(sink)
    rec = mem.read_episodes("s")[0]["record"]
    md = rec.get("markdown")
    assert md, "the seal must store a markdown snapshot of the turn"
    assert md.startswith("# Add subtract to calc.py"), "markdown leads with the turn heading"
    assert "**changed files:** calc.py" in md, "markdown names the changed files"
    assert "## what happened" in md and "pytest" in md, "markdown carries the action trace"
    assert "## conclusion" in md and "added subtract" in md, "markdown carries the conclusion"


@check
def recall_returns_the_markdown_smoothly():
    mem = _Cache()
    sink = make_episode_sink(mem, session_id="s", task_id_fn=lambda: "t", title_fn=lambda: "Turn one")
    _run_a_turn(sink)
    tool = make_history_tool(mem, "s")
    out = tool.handler({"turns": [1]})
    assert "## what happened" in out and "## conclusion" in out, \
        "recall_history returns the clean markdown snapshot (smooth read), not a raw dump"


@check
def recall_returns_the_observed_value_end_to_end():
    # THE channel, model-free: a value OBSERVED in a turn (not reported, not noted) must come BACK through
    # recall_history. Proves precision is a property of the cache+recall path itself — any live flakiness
    # on top is the model's recall reliability, not the channel losing data.
    mem = _Cache()
    sink = make_episode_sink(mem, session_id="s", task_id_fn=lambda: "t", title_fn=lambda: "Read config")
    sink(SliceBuilt(rendered="# seed"))
    sink(ToolResult(name="read_file", args={"path": "config.env"},
                    output="HOST=localhost\nSECRET=VAL_XYZ_42\nPORT=80\nDEBUG=false", failing=False))
    sink(TurnEnd("end_turn", 1, {"prompt_tokens": 10, "completion_tokens": 2}))
    out = make_history_tool(mem, "s").handler({"turns": [1]})
    assert "VAL_XYZ_42" in out, "recall_history must return the OBSERVED value (the cross-slice channel)"


@check
def turn_markdown_is_pure_and_self_contained():
    # the renderer builds from buffered turn data alone (no Slice) — Markov, no coupling.
    md = turn_markdown("My Turn", [{"action": [{"name": "run_command", "args": {"command": "ls"}, "failing": False}],
                                    "observation": ["a.py\nb.py"]}], "did the thing",
                       {"files": ["a.py"], "stop_reason": "end_turn"})
    assert md.startswith("# My Turn") and "a.py" in md and "did the thing" in md


@check
def markdown_carries_observed_data_for_recall():
    # goal-2 precision (previous slice → cache → current slice): the cache must preserve the DATA a turn
    # SAW (a value read from a file), not just a count-summary — else cross-slice recall returns
    # "read_file -> N chars, N lines" and the agent can't answer "what was the value?" later. A bounded
    # observation excerpt now rides each trace line.
    secret = "zk9q-7r4t"
    md = turn_markdown("Read config", [{
        "action": [{"name": "read_file", "args": {"path": "build_meta.txt"}, "failing": False}],
        "observation": [f"     1\tA=1\n     2\tBUILD_FINGERPRINT={secret}\n     3\tB=2\n"]}],
        "file is readable", {})
    assert "[read_file]" in md, "still carries the summary line"
    assert secret in md, "the OBSERVED VALUE must survive into the recallable markdown (the goal-2 precision fix)"


@check
def observation_excerpt_is_bounded_not_a_raw_dump():
    # the archive stays distilled: a huge observation is reduced to a bounded head+tail (moat — the cache
    # is L2 but must not become a raw transcript).
    big = "X" * 6000 + "Y" * 6000
    md = turn_markdown("big read", [{
        "action": [{"name": "read_file", "args": {"path": "big.txt"}, "failing": False}],
        "observation": [big]}], "", {})
    assert len(md) < 2500, "a large observation must be bounded head+tail, not dumped whole (12k here)"
    assert "…⋯…" in md, "a large observation is elided in the middle"


@check
def medium_observation_kept_whole_so_middle_survives():
    # the S6 gap: a value in the MIDDLE of a ~100-line file must survive (kept whole under _OBS_KEEP_WHOLE),
    # not be elided by a tight head+tail.
    lines = [f"k{i:03d}=v{i:03d}" for i in range(100)]
    lines[50] = "DEEP_KEY=MIDDLE_NEEDLE_42"
    md = turn_markdown("read bigconf", [{
        "action": [{"name": "read_file", "args": {"path": "bigconf.txt"}, "failing": False}],
        "observation": ["\n".join(lines)]}], "", {})
    assert "MIDDLE_NEEDLE_42" in md, "a value in the middle of a normal file must survive into the cache"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
