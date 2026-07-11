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
from sliceagent.events import (AssistantText, ApiRetry, StepBegin, StepEnd, ToolResult,  # noqa: E402
                               TurnCommitted, TurnEnd, TurnInterrupted, TurnPhaseChanged, TurnStarted)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _render(events, *, width=100):
    c = Console(file=StringIO(), force_terminal=True, width=width, color_system=None)
    stats = {"model": "m", "policy": "allow", "topic": "", "tokens": 0}
    sink = tui.make_rich_sink(c, stats, await_commit=True)
    for e in events:
        sink(e)
    sink._flush_reads()   # a real turn always ends with TurnEnd, which flushes buffered read-only cards
    sink._stop()
    return c.file.getvalue(), stats


@check
def tool_result_ok_uses_a_quiet_activity_rail():
    # Ordinary success stays quiet; green ✓ is reserved for the durable turn boundary.
    out, _ = _render([ToolResult("edit_file", {"path": "src/foo.py"}, "ok", False)])
    assert "│ write" in out and "foo.py" in out and "✓" not in out, out


@check
def read_only_tools_coalesce_into_one_dim_line():
    out, _ = _render([
        ToolResult("read_file", {"path": "a.py"}, "ok", False),
        ToolResult("read_file", {"path": "b.py"}, "ok", False),
        ToolResult("grep", {"pattern": "foo"}, "matches", False),
        TurnEnd("end_turn", 1, {}),
    ])
    assert "2 read" in out and "1 search" in out, out    # one coalesced summary…
    assert "a.py" in out and "b.py" in out, out          # …that still names the files
    assert "✓ read" not in out, out                      # …not three separate ✓ cards


@check
def tool_result_fail_shows_cross():
    out, _ = _render([ToolResult("run_command", {"command": "pytest"}, "Exit code 1", True)])
    assert "✗" in out and "run" in out, out


@check
def successful_tool_notes_become_deduplicated_milestones():
    out, _ = _render([
        TurnStarted("verify retry behavior"),
        ToolResult("run_command", {"command": "pytest", "note": "The retry regression now passes."},
                   "1 passed", False),
        ToolResult("run_command", {"command": "pytest", "note": "The retry regression now passes."},
                   "1 passed", False),
        ToolResult("run_command", {"command": "pytest", "note": "A failed claim must stay hidden."},
                   "exit 1", True),
    ])
    assert out.count("◆ The retry regression now passes.") == 1, out
    assert "A failed claim must stay hidden." not in out, out


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
    out, _ = _render([
        TurnStarted("fix retries"),
        ApiRetry(1, "timeout"),
        TurnEnd("end_turn", 3, {"prompt_tokens": 100, "completion_tokens": 20}),
        TurnCommitted(True, "end_turn", detail="checkpoint saved"),
    ])
    assert "retry" in out and "turn saved" in out, out


@check
def committed_receipt_distinguishes_agent_rejection_from_execution_failure():
    receipt = {
        "disposition": "completed_with_warnings",
        "counts": {
            "requested": 1, "rejected_before_execution": 1, "execution_started": 0,
            "settled": 1, "succeeded": 0, "failed": 0, "cancelled": 0,
            "indeterminate": 0, "not_started": 0,
        },
        "agents": {
            "requested": 1, "rejected_before_execution": 1, "execution_started": 0,
            "settled": 1, "succeeded": 0, "failed": 0, "cancelled": 0,
            "indeterminate": 0, "not_started": 0,
        },
    }
    out, _ = _render([
        TurnStarted("delegate"), TurnEnd("end_turn", 1, {}),
        TurnCommitted(True, "end_turn", receipt=receipt),
    ])
    assert "turn saved with warnings" in out, out
    assert "1 agent rejected before start" in out, out
    assert "agent failed" not in out and "agent started" not in out, out


@check
def adverse_receipt_facts_survive_an_ordinary_width_completion():
    receipt = {
        "turn_status": "end_turn", "disposition": "completed_with_warnings",
        "counts": {
            "requested": 24, "rejected_before_execution": 13, "execution_started": 11,
            "settled": 24, "succeeded": 10, "failed": 1, "cancelled": 0,
            "indeterminate": 0, "not_started": 0,
        },
        "agents": {
            "requested": 24, "rejected_before_execution": 13, "execution_started": 11,
            "settled": 24, "succeeded": 10, "failed": 1, "cancelled": 0,
            "indeterminate": 0, "not_started": 0,
        },
    }
    out, _ = _render([
        TurnStarted("delegate", plan=[{"step": "inspect", "status": "done"}]),
        StepBegin(1), TurnEnd("end_turn", 1, {}),
        TurnCommitted(True, "end_turn", receipt=receipt),
    ], width=80)
    assert "turn saved with warnings" in out, out
    assert "13 rejected before start" in out and "1 failed" in out, out
    assert "plan 1/1" not in out and "1 pass" not in out, \
        "cosmetic progress must yield space to adverse lifecycle truth"


@check
def read_wave_flushes_at_step_end_before_any_answer_or_turn_end():
    out, _ = _render([
        TurnStarted("inspect files"),
        ToolResult("read_file", {"path": "a.py"}, "ok", False),
        ToolResult("grep", {"pattern": "needle"}, "matches", False),
        StepEnd(1, {}, "tool_use"),
    ])
    assert "1 read" in out and "1 search" in out and "a.py" in out, out


@check
def rejected_completion_draft_is_replaced_before_commit():
    out, _ = _render([
        TurnStarted("fix and verify"),
        StepBegin(1),
        AssistantText("draft that failed verification"),
        StepEnd(1, {}, "end_turn"),
        TurnPhaseChanged("checking_completion", "running checks"),
        StepBegin(2),                 # completion gate requested another pass → discard prior draft
        AssistantText("final verified response"),
        StepEnd(2, {}, "end_turn"),
        TurnEnd("end_turn", 2, {}),
        TurnCommitted(True, "end_turn"),
    ])
    assert "draft that failed verification" not in out, out
    assert out.count("final verified response") == 1 and "turn saved" in out, out


@check
def committed_turn_cannot_swallow_chitchat_or_be_resurrected_by_late_deltas():
    c = Console(file=StringIO(), force_terminal=False, width=100, color_system=None)
    sink = tui.make_rich_sink(c, {}, await_commit=True)
    sink(TurnStarted("finish the task"))
    sink(AssistantText("task response"))
    sink(TurnEnd("end_turn", 1, {}))
    sink(TurnCommitted(True, "end_turn"))
    terminal = sink.progress.snapshot()
    assert terminal.committed and not terminal.active

    sink(AssistantText("You are welcome!"))     # cheap chitchat has no TurnStarted/TurnCommitted pair
    assert "You are welcome!" in c.file.getvalue()
    assert sink._pending_answer == "" and not sink.progress.snapshot().active
    sink.on_delta("content", "late stale bytes")
    assert sink.progress.snapshot() == terminal, "late deltas must not reopen a committed turn"
    assert sink._status is None


@check
def partial_and_unsaved_answers_are_never_styled_as_normal_final_responses():
    interrupted, _ = _render([
        TurnStarted("long response"),
        AssistantText("PARTIAL RESPONSE"),
        TurnInterrupted("max_tokens", "response was truncated"),
    ])
    assert "assistant · partial" in interrupted and "PARTIAL RESPONSE" in interrupted, interrupted

    unsaved, _ = _render([
        TurnStarted("save response"),
        AssistantText("UNSAVED RESPONSE"),
        TurnEnd("end_turn", 1, {}),
        TurnCommitted(False, "end_turn", detail="disk full"),
    ])
    assert "assistant · unsaved" in unsaved and "save failed" in unsaved, unsaved

    update, _ = _render([
        TurnStarted("use a tool"),
        AssistantText("I will inspect that now.", final=False),
    ])
    assert "assistant · update" in update and "I will inspect" in update, update


@check
def step_end_accumulates_tokens():
    _, stats = _render([StepEnd(1, {"prompt_tokens": 50, "completion_tokens": 10}, "tool_use"),
                        StepEnd(2, {"prompt_tokens": 30, "completion_tokens": 5}, "end_turn")])
    assert stats["tokens"] == 95, stats


@check
def interrupt_renders_warning():
    out, _ = _render([TurnInterrupted("aborted", "stopped by user")])
    assert "interrupted" in out, out
    blocked, _ = _render([TurnInterrupted("stuck", "read_file failed repeatedly")])
    assert "stopped" in blocked and "interrupted" not in blocked, blocked


@check
def slash_completer_offers_navigation_commands():
    from prompt_toolkit.document import Document
    comp = tui._InputCompleter()   # no repo files wired → behaves slash-only here
    got = [c.text for c in comp.get_completions(Document("/mo", len("/mo")), None)]
    assert "/model" in got and "/mode" in got, got
    update = [c.text for c in comp.get_completions(Document("/up", len("/up")), None)]
    assert "/update" in update, update
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
    assert "search" in tui._tool_header("grep", {"pattern": "TODO", "note": "x"})
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
