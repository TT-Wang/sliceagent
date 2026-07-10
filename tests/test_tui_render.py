"""TUI rendering checks for calm rails, responsive progress furniture, and token accounting.
Skips cleanly if the `tui` extra (rich) isn't installed. No model.
Run: PYTHONPATH=src python tests/test_tui_render.py
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from rich.cells import cell_len  # noqa: E402
    from rich.console import Console  # noqa: E402
    from sliceagent.tui import (RichSink, _box_width, _render_plan, _render_read_summary,  # noqa: E402
                                _render_tool_result, _response_panel, _toolbar)
    from sliceagent.events import StepEnd, ToolResult   # noqa: E402
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
def plan_renders_as_one_settled_summary():
    sink, buf = _sink_capture()
    sink(ToolResult("update_plan", {"steps": [
        {"step": "write the parser", "status": "done"},
        {"step": "add error handling", "status": "in_progress"},
        {"step": "write tests", "status": "pending"}]}, "PLAN updated", failing=False))
    out = buf.getvalue()
    assert out == "│ plan 1/3 · add error handling\n", out
    assert "write the parser" not in out and "write tests" not in out, \
        "full plan history belongs in /plan, not scrollback"


@check
def fresh_tokens_tracked_for_toolbar():
    sink, _ = _sink_capture()
    sink(StepEnd(1, {"prompt_tokens": 1000, "completion_tokens": 20, "input_other": 200}, "tool_use"))
    sink(StepEnd(2, {"prompt_tokens": 1000, "completion_tokens": 10, "input_other": 50}, "stop"))
    assert sink.stats["fresh"] == 250, sink.stats
    assert sink.stats["tokens"] == 2030, sink.stats


@check
def render_plan_handles_empty_and_bad_input():
    assert _render_plan([]) is not None                 # empty → compact "plan 0/0" row, no crash
    assert _render_plan([{"step": "x", "status": "weird"}, "not a dict"]) is not None


@check
def settled_rows_never_wrap_at_common_terminal_widths():
    event = ToolResult(
        "run_command", {"command": "pytest -q " + "very_long_test_name_" * 8},
        "128 passed; " + "verification detail " * 12, False,
    )
    reads = [("read_file", "src/" + "deeply_nested/" * 8 + "parser.py") for _ in range(4)]
    plan = [
        {"step": "inspect", "status": "done"},
        {"step": "add a deliberately long regression test " * 5, "status": "in_progress"},
    ]
    for width in (60, 80, 120):
        assert cell_len(_render_plan(plan, width).plain) <= width
        summary = _render_read_summary(reads, width)
        assert summary is not None and cell_len(summary.plain) <= width
        buf = io.StringIO()
        console = Console(file=buf, width=width, force_terminal=False, color_system=None, soft_wrap=False)
        console.print(_render_tool_result(event, width))
        assert all(cell_len(line) <= width for line in buf.getvalue().splitlines()), buf.getvalue()


@check
def response_panel_and_footer_are_responsive_at_60_80_and_120():
    stats = {
        "workspace": "demo", "model": "model-x", "policy": "ask-before-write",
        "topic": "fix retry handling", "tokens": 999_999, "saved_cached_tok": 42_000,
    }
    for width, expected_box in ((60, 58), (80, 78), (120, 96)):
        buf = io.StringIO()
        console = Console(file=buf, width=width, force_terminal=False, color_system=None, soft_wrap=False)
        assert _box_width(console) == expected_box
        console.print(_response_panel("A response with `code` and enough prose to wrap cleanly.", console))
        assert all(cell_len(line) <= width for line in buf.getvalue().splitlines()), buf.getvalue()

        footer = _toolbar(stats, lambda width=width: width)()
        plain = "".join(fragment[1] for fragment in footer)
        assert cell_len(plain) <= width and "sliceagent" in plain and "demo" in plain and "model-x" in plain
        assert ("ask-before-write" in plain) is (width >= 80)
        assert ("fix retry handling" in plain) is (width >= 120)
        assert "999" not in plain and "saved" not in plain, "volatile cost detail belongs in /cost"

        unicode_stats = dict(stats, workspace="切片代理工作区非常长", topic="修复重试逻辑并验证")
        unicode_footer = _toolbar(unicode_stats, lambda width=width: width)()
        unicode_plain = "".join(fragment[1] for fragment in unicode_footer)
        assert cell_len(unicode_plain) <= width, (width, unicode_plain)


@check
def completer_does_slash_and_files():
    from prompt_toolkit.document import Document
    from sliceagent.tui import _InputCompleter
    comp = _InputCompleter(files=["src/sliceagent/util.py", "tests/test_util.py", "README.md"])
    # slash palette at line start
    slash = [c.text for c in comp.get_completions(Document("/pl"), None)]
    assert "/plan" in slash, slash
    # filename completion on an explicit @mention (basename-prefix first) — the ONLY trigger; matches
    # the @path syntax cli.py's message parser already recognizes for pinning/attaching a file.
    files = [c.text for c in comp.get_completions(Document("please edit @util"), None)]
    assert "src/sliceagent/util.py" in files and "tests/test_util.py" in files, files
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
    from sliceagent.tui import banner_panel, _WORDMARK
    def render(width):
        buf = io.StringIO()
        # Rich's TERM=dumb fallback reports a fixed 80×25 unless both dimensions and TERM are pinned,
        # which would make this width-sensitive rendering test assert against the test runner environment.
        c = Console(file=buf, width=width, height=25, force_terminal=True, color_system=None,
                    _environ={"TERM": "xterm"})
        c.print(banner_panel(c, "info"))
        return buf.getvalue()
    wide = render(120)
    assert _WORDMARK[0] in wide and _WORDMARK[-1] in wide, "a wide terminal must show the full block wordmark"
    assert "m e m a g e n t" not in wide, "the compact fallback must be gone"
    # a typical ~86-col window (just under the roomy-frame threshold) must still show the FULL wordmark —
    # the last row present uncropped = the final 't' isn't clipped (regression: it clipped up to width 90,
    # because full chrome needed ~91 cols; the adaptive layout now fits the 79-col art from width 85).
    # The tight fit (full wordmark from ~85 cols) is calibrated on POSIX cell metrics; prompt_toolkit's
    # get_cwidth measures the ansi_shadow block glyphs wider on Windows, so the art needs a roomier terminal
    # there. The full wordmark itself is already asserted cross-platform at width 120 above.
    for w in ((85, 86, 90) if os.name != "nt" else ()):
        out = render(w)
        assert _WORDMARK[-1] in out, f"full wordmark (final 't') must show at width {w}, not clip"
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
