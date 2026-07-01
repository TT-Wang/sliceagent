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
    # filename completion on an explicit @mention (basename-prefix first) — the ONLY trigger; matches
    # the @path syntax cli.py's message parser already recognizes for pinning/attaching a file.
    files = [c.text for c in comp.get_completions(Document("please edit @util"), None)]
    assert "src/memagent/util.py" in files and "tests/test_util.py" in files, files
    # plain prose (no @) must NOT pop a completion menu, even on a word that matches a real file —
    # this is the exact annoyance the @-gating fixes.
    assert list(comp.get_completions(Document("please edit util"), None)) == []
    # short words / mid-prose don't spam
    assert list(comp.get_completions(Document("a"), None)) == []


@check
def banner_always_uses_the_big_block_wordmark():
    # User preference: the full ansi_shadow BLOCK wordmark every time — never the compact one-line fallback.
    # Each art row is no-wrap+crop, so a wide terminal shows it in full and a narrow one clips it cleanly on
    # the right (a single line per row), never wrapping into a staircase.
    from rich.console import Console
    from memagent.tui import banner_panel, _WORDMARK
    def render(width):
        buf = io.StringIO()
        c = Console(file=buf, width=width, force_terminal=True, color_system=None)
        c.print(banner_panel(c, "info"))
        return buf.getvalue()
    wide = render(120)
    assert _WORDMARK[0] in wide and _WORDMARK[-1] in wide, "a wide terminal must show the full block wordmark"
    assert "m e m a g e n t" not in wide, "the compact fallback must be gone"
    narrow = render(60)                               # narrower than the art → still big, just clipped
    assert "█" in narrow and "m e m a g e n t" not in narrow, \
        "even a narrow terminal must show the big block wordmark (clipped), never the compact one"
    # each big-art row is ONE display line (no_wrap): the 6 emblem-prefixed rows stay 6 lines, never split.
    # (Doubled emblems ▓▓/▒▒/░░ mark exactly the art rows; the tagline uses single ▓/▒/░, so it's excluded.)
    art_lines = [ln for ln in narrow.splitlines() if any(e in ln for e in ("▓▓", "▒▒", "░░"))]
    assert len(art_lines) == len(_WORDMARK), f"art must not wrap: {len(_WORDMARK)} rows, got {len(art_lines)}"


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
