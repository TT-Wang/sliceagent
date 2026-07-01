"""Agent-writable WORLD MODEL region — the slice generalized beyond source files (maze map /
inventory / system state). State lives in the Slice, folded by slice_sink from world_set/world_clear
events (the note→findings seam), READ straight from the rendered region, survives the seal, clears on
reset. Deterministic, no model, no pytest. Run: PYTHONPATH=src python tests/test_world_region.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.pfc import Slice, slice_sink  # noqa: E402
from memagent.seed import make_build_slice  # noqa: E402
from memagent.events import ToolResult                          # noqa: E402
from memagent.tools import LocalToolHost                        # noqa: E402
from memagent.memory import NullMemory                          # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _emit(sink, name, args, failing=False):
    sink(ToolResult(name=name, args=args, output="ok", failing=failing))


@check
def world_set_folds_into_slice():
    s = Slice(); s.reset("solve the maze")
    sink = slice_sink(s)
    _emit(sink, "world_set", {"key": "pos", "value": "2,3"})
    _emit(sink, "world_set", {"key": "map", "value": "#.#\n. .\n#E#"})
    assert s.world["pos"] == "2,3", s.world
    assert "\n" in s.world["map"], "multiline value must be preserved"


@check
def failing_world_set_is_ignored():
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    _emit(sink, "world_set", {"key": "k", "value": "v"}, failing=True)
    assert "k" not in s.world, "a failing world_set must not mutate state"


@check
def world_renders_in_the_slice():
    wd = tempfile.mkdtemp(prefix="world-")
    s = Slice(); s.reset("solve the maze")
    sink = slice_sink(s)
    _emit(sink, "world_set", {"key": "pos", "value": "2,3"})
    tools = LocalToolHost(root=wd)
    build = make_build_slice(s, tools, None, NullMemory(), s.goal)
    user = build()[1]["content"]
    assert "# WORLD MODEL" in user and "pos: 2,3" in user, "world not rendered into slice"


@check
def world_unbounded_within_loop():
    """Unlike findings (capped ring), the world model is NOT truncated within a loop."""
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    for i in range(50):
        _emit(sink, "world_set", {"key": f"cell{i}", "value": str(i)})
    assert len(s.world) == 50, f"world model must not be capped: {len(s.world)}"


@check
def world_survives_seal_clears_on_reset():
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    _emit(sink, "world_set", {"key": "k", "value": "v"})
    s.seal()
    assert s.world.get("k") == "v", "world model must SURVIVE the seal (distilled task state)"
    s.reset("a brand new task")
    assert s.world == {}, "world model must CLEAR on reset (a new task)"


@check
def world_clear_key_and_all():
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    _emit(sink, "world_set", {"key": "a", "value": "1"})
    _emit(sink, "world_set", {"key": "b", "value": "2"})
    _emit(sink, "world_clear", {"key": "a"})
    assert "a" not in s.world and s.world.get("b") == "2", s.world
    _emit(sink, "world_clear", {})
    assert s.world == {}, "world_clear with no key clears all"


@check
def tools_registered_and_confirm():
    wd = tempfile.mkdtemp(prefix="world-")
    h = LocalToolHost(root=wd)
    names = {sc["function"]["name"] for sc in h.schemas()}
    assert "world_set" in names and "world_clear" in names
    out = h.run("world_set", {"key": "x", "value": "y"})
    assert "WORLD MODEL" in out and not out.startswith("Error:"), out


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
