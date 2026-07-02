"""TUI sink rendering + input plumbing. Requires the `tui` extra (rich + prompt_toolkit); if those
are absent (e.g. the py3.10 headless test env) the whole file SKIPS — the TUI is optional and the
core/eval path never imports it. Run: PYTHONPATH=src <venv>/python tests/test_tui.py
"""
import os
import sys
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import rich  # noqa: F401
    import prompt_toolkit  # noqa: F401
except Exception:
    print("SKIP test_tui — `tui` extra (rich+prompt_toolkit) not installed")
    sys.exit(0)

from rich.console import Console  # noqa: E402
from sliceagent import tui  # noqa: E402
from sliceagent.events import (AssistantText, ApiRetry, StepEnd, ToolResult,  # noqa: E402
                             TurnEnd, TurnInterrupted)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _render(events):
    c = Console(file=StringIO(), force_terminal=True, width=100, color_system=None)
    stats = {"model": "m", "policy": "allow", "topic": "", "tokens": 0}
    sink = tui.make_rich_sink(c, stats)
    for e in events:
        sink(e)
    sink._flush_reads()   # a real turn always ends with TurnEnd, which flushes buffered read-only cards
    sink._stop()
    return c.file.getvalue(), stats


@check
def tool_result_ok_shows_check_and_header():
    # a MUTATING tool renders its own ✓ card (read-only tools coalesce — see the next test)
    out, _ = _render([ToolResult("edit_file", {"path": "src/foo.py"}, "ok", False)])
    assert "✓" in out and "write" in out and "foo.py" in out, out


@check
def read_only_tools_coalesce_into_one_dim_line():
    out, _ = _render([
        ToolResult("read_file", {"path": "a.py"}, "ok", False),
        ToolResult("read_file", {"path": "b.py"}, "ok", False),
        ToolResult("grep", {"pattern": "foo"}, "matches", False),
        TurnEnd("end_turn", 1, {}),
    ])
    assert "2 read" in out and "1 grep" in out, out      # one coalesced summary…
    assert "a.py" in out and "b.py" in out, out          # …that still names the files
    assert "✓ read" not in out, out                      # …not three separate ✓ cards


@check
def tool_result_fail_shows_cross():
    out, _ = _render([ToolResult("run_command", {"command": "pytest"}, "Exit code 1", True)])
    assert "✗" in out and "run" in out, out


@check
def str_replace_renders_a_diff():
    out, _ = _render([ToolResult("str_replace",
                                 {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}, "ok", False)])
    assert "- x = 1" in out and "+ x = 2" in out, out


@check
def assistant_text_renders_markdown():
    out, _ = _render([AssistantText("Done. The **fix** works.")])
    assert "Done" in out and "fix" in out, out


@check
def turn_end_and_retry_render():
    out, _ = _render([ApiRetry(1, "timeout"), TurnEnd("end_turn", 3, {"prompt_tokens": 100, "completion_tokens": 20})])
    assert "retry" in out and "done" in out and "3 steps" in out, out


@check
def step_end_accumulates_tokens():
    _, stats = _render([StepEnd(1, {"prompt_tokens": 50, "completion_tokens": 10}, "tool_use"),
                        StepEnd(2, {"prompt_tokens": 30, "completion_tokens": 5}, "end_turn")])
    assert stats["tokens"] == 95, stats


@check
def interrupt_renders_warning():
    out, _ = _render([TurnInterrupted("aborted", "stopped by user")])
    assert "interrupted" in out, out


@check
def slash_completer_offers_navigation_commands():
    from prompt_toolkit.document import Document
    comp = tui._InputCompleter()   # no repo files wired → behaves slash-only here
    got = [c.text for c in comp.get_completions(Document("/mo", len("/mo")), None)]
    assert "/model" in got and "/mode" in got, got
    # /switch was removed from the palette in the menu redesign (parked-topic resume dropped)
    assert "/switch" not in [c.text for c in comp.get_completions(Document("/sw", len("/sw")), None)]
    # a non-slash line offers nothing when no repo files are wired
    assert list(comp.get_completions(Document("hello", 5), None)) == []


@check
def tui_enabled_honors_the_flag():
    os.environ["AGENT_TUI"] = "0"
    assert tui.tui_enabled() is False
    os.environ["AGENT_TUI"] = "1"
    assert tui.tui_enabled() is True
    del os.environ["AGENT_TUI"]


@check
def tool_header_picks_primary_arg():
    assert "grep" in tui._tool_header("grep", {"pattern": "TODO", "note": "x"})
    assert "TODO" in tui._tool_header("grep", {"pattern": "TODO"})


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
