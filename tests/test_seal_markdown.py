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
def turn_markdown_is_pure_and_self_contained():
    # the renderer builds from buffered turn data alone (no Slice) — Markov, no coupling.
    md = turn_markdown("My Turn", [{"action": [{"name": "run_command", "args": {"command": "ls"}, "failing": False}],
                                    "observation": ["a.py\nb.py"]}], "did the thing",
                       {"files": ["a.py"], "stop_reason": "end_turn"})
    assert md.startswith("# My Turn") and "a.py" in md and "did the thing" in md


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
