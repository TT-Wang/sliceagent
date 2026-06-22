"""TUI rendering of the new tiers (Path-A borrow): the RichSink surfaces the PLAN checklist + MISSION
line and tracks FRESH-input cost. Skips cleanly if the `tui` extra (rich) isn't installed. No model.
Run: PYTHONPATH=src python tests/test_tui_render.py
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from rich.console import Console  # noqa: E402
    from memagent.tui import RichSink, _render_plan  # noqa: E402
    from memagent.events import StepEnd, ToolResult   # noqa: E402
except Exception as _e:  # noqa: BLE001 — tui extra (rich) not installed → skip, don't fail the suite
    print(f"SKIP test_tui_render (tui extra not available: {_e})")
    sys.exit(0)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _sink_capture():
    buf = io.StringIO()
    con = Console(file=buf, width=100, force_terminal=False, color_system=None)
    return RichSink(con, {}), buf


@check
def plan_renders_as_a_checklist():
    sink, buf = _sink_capture()
    sink(ToolResult("update_plan", {"steps": [
        {"step": "write the parser", "status": "done"},
        {"step": "add error handling", "status": "in_progress"},
        {"step": "write tests", "status": "pending"}]}, "PLAN updated", failing=False))
    out = buf.getvalue()
    assert "plan" in out and "1/3 done" in out, out
    assert "write the parser" in out and "add error handling" in out and "write tests" in out
    assert "✓" in out and "▶" in out and "○" in out, "status glyphs must render"


@check
def mission_renders_as_a_line():
    sink, buf = _sink_capture()
    sink(ToolResult("set_mission", {"text": "ship the v2 release"}, "MISSION set", failing=False))
    out = buf.getvalue()
    assert "mission" in out and "ship the v2 release" in out, out


@check
def fresh_tokens_tracked_for_toolbar():
    sink, _ = _sink_capture()
    sink(StepEnd(1, {"prompt_tokens": 1000, "completion_tokens": 20, "input_other": 200}, "tool_use"))
    sink(StepEnd(2, {"prompt_tokens": 1000, "completion_tokens": 10, "input_other": 50}, "stop"))
    assert sink.stats["fresh"] == 250, sink.stats
    assert sink.stats["tokens"] == 2030, sink.stats


@check
def render_plan_handles_empty_and_bad_input():
    assert _render_plan([]) is not None                 # empty → "(empty plan)" panel, no crash
    assert _render_plan([{"step": "x", "status": "weird"}, "not a dict"]) is not None


@check
def completer_does_slash_and_files():
    from prompt_toolkit.document import Document
    from memagent.tui import _InputCompleter
    comp = _InputCompleter(files=["src/memagent/util.py", "tests/test_util.py", "README.md"])
    # slash palette at line start
    slash = [c.text for c in comp.get_completions(Document("/pl"), None)]
    assert "/plan" in slash, slash
    # filename completion on the current word (basename-prefix first)
    files = [c.text for c in comp.get_completions(Document("please edit util"), None)]
    assert "src/memagent/util.py" in files and "tests/test_util.py" in files, files
    # short words / mid-prose don't spam
    assert list(comp.get_completions(Document("a"), None)) == []


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
