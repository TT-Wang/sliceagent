"""The LIVE composer (AGENT_TUI=live): the always-pinned box whose turns run in a worker thread while
output streams above. Driven HEADLESSLY — LiveSink fed events directly, and build_live_app driven with a
prompt_toolkit pipe input — so we verify the real logic (status transitions, static prints above, Enter→turn
dispatch in a worker, ctrl-d quit) without a tty. The pinned-during-streaming RENDERING still needs a live
terminal; this covers everything testable offline.

No model, no pytest. Run: PYTHONPATH=src python tests/test_live_composer.py
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _rec_console():
    from rich.console import Console
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=80, soft_wrap=False), buf


@check
def livesink_status_transitions_and_prints_above():
    from sliceagent.tui import LiveSink
    from sliceagent.events import SliceBuilt, ToolStarted, ToolResult, AssistantText, TurnEnd
    console, buf = _rec_console()
    statuses = []
    sink = LiveSink(console, {}, lambda s: statuses.append(s))

    sink(SliceBuilt("the request"))                                   # → thinking…
    sink(ToolStarted("read_file", {"path": "parser.py"}))            # → 'read parser.py …'
    sink(ToolResult("read_file", {"path": "parser.py"}, "code", False))   # prints card; → working…
    sink.on_delta("content", "here is the answer")                    # → writing… tail
    sink(AssistantText("the final answer text"))                     # prints reply panel
    sink(TurnEnd("end_turn", 2, {}))                                  # → None (idle)

    assert "thinking…" in statuses, statuses
    assert any("read" in (s or "") and "parser.py" in (s or "") for s in statuses), statuses
    assert "working…" in statuses, statuses
    assert any((s or "").startswith("writing…") and "answer" in (s or "") for s in statuses), statuses
    assert statuses[-1] is None, "TurnEnd must clear the status to idle"
    out = buf.getvalue()
    assert "the final answer text" in out, "the reply must print ABOVE the box"
    assert "parser.py" in out, "the tool card must print above the box"


@check
def livesink_clears_the_writing_status_when_the_response_arrives():
    # AssistantText means the reply is done → the "writing…" status must clear THEN, not linger until
    # TurnEnd, so the finished panel isn't shown above a stale streaming spinner. (RichSink already stops its
    # transient status on AssistantText; this gives LiveSink the same guarantee.)
    from sliceagent.tui import LiveSink
    from sliceagent.events import SliceBuilt, AssistantText
    console, _ = _rec_console()
    statuses = []
    sink = LiveSink(console, {}, lambda s: statuses.append(s))
    sink(SliceBuilt("q"))
    sink.on_delta("content", "streaming the answer now")     # → writing… …
    assert (statuses[-1] or "").startswith("writing…"), statuses
    sink(AssistantText("the final answer"))                  # must clear the status right here
    assert statuses[-1] is None, f"AssistantText must clear the streaming status, got {statuses[-1]!r}"


@check
def livesink_read_card_is_header_only_like_richsink():
    # parity with RichSink: read/list cards show no content dump (shared _render_tool_result)
    from sliceagent.tui import LiveSink
    from sliceagent.events import ToolResult
    console, buf = _rec_console()
    LiveSink(console, {}, lambda s: None)(ToolResult("read_file", {"path": "x.py"}, "SECRET-CONTENT", False))
    assert "SECRET-CONTENT" not in buf.getvalue(), "read card should not dump file content"
    assert "x.py" in buf.getvalue()


def _drive_live(keys, run_one_turn, handle_slash=None):
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.tui import build_live_app
    console, buf = _rec_console()
    with create_pipe_input() as pinp:
        pinp.send_text(keys)
        app, state = build_live_app(console=console, stats={"model": "test-model", "topic": "demo"}, root=None,
                                    run_one_turn=run_one_turn, handle_slash=handle_slash,
                                    pt_input=pinp, pt_output=DummyOutput())
        app.run()
    for th in state.get("threads", []):
        th.join(timeout=3)
    return state, buf.getvalue()


@check
def live_app_submit_dispatches_a_turn_in_a_worker():
    calls = []
    def fake_turn(text, sink, signal):
        from sliceagent.events import AssistantText
        calls.append((text, signal))
        sink(AssistantText("worker reply"))
    state, out = _drive_live("explain the parser\r\x04", fake_turn)   # submit, then ctrl-d to quit
    assert calls and calls[0][0] == "explain the parser", calls
    assert state["last"] == "explain the parser"
    assert "explain the parser" in out, "the user message must be echoed above the box on Enter"
    assert "worker reply" in out, "the turn's output must print above the box"
    # the turn got a real abort signal object (Event-like: has .set / .is_set)
    sig = calls[0][1]
    assert hasattr(sig, "set") and hasattr(sig, "is_set"), "run_turn must receive an abort signal"


@check
def live_app_escape_aborts_a_running_turn_like_ctrl_c():
    # Esc used to be a no-op mid-turn ("never undo mid-turn"); it must now abort exactly like ctrl-c does
    # (same state["signal"].set() call) instead of being silently swallowed.
    import time
    seen = {}

    def fake_turn(text, sink, signal):
        seen["sig"] = signal
        for _ in range(40):             # poll up to ~2s, return as soon as the abort signal is set
            if signal.is_set():
                return
            time.sleep(0.05)

    state, _ = _drive_live("do something slow\r\x1b\x04", fake_turn)   # submit, Esc, then ctrl-d to quit
    for th in state.get("threads", []):
        th.join(timeout=3)
    assert "sig" in seen, "the turn must have started before Esc was sent"
    assert seen["sig"].is_set(), "Esc mid-turn must set the SAME abort signal ctrl-c uses"


@check
def live_app_ctrl_d_quits_without_a_turn():
    calls = []
    state, _ = _drive_live("\x04", lambda *a: calls.append(a))   # bare ctrl-d
    assert not calls, "ctrl-d at the idle box must quit, not run a turn"
    assert state["running"] is False


@check
def live_app_slash_is_handled_not_run_as_a_turn():
    seen, turns = [], []
    state, _ = _drive_live("/threads\r\x04", lambda *a: turns.append(a), handle_slash=lambda s: seen.append(s))
    assert seen == ["/threads"], seen
    assert not turns, "a slash command must NOT be dispatched as a turn"


@check
def richsink_refactor_is_render_identical():
    # the shared _render_tool_result must produce the SAME output RichSink did before extraction:
    # plan panel, mission line, gutter card with ✓/✗, and read-card header-only.
    from sliceagent.tui import _render_tool_result
    from sliceagent.events import ToolResult
    from rich.console import Console

    def render(e):
        c = Console(file=io.StringIO(), force_terminal=False, width=80, soft_wrap=False)
        c.print(_render_tool_result(e))
        return c.file.getvalue()

    assert "plan ·" in render(ToolResult("update_plan", {"steps": [{"step": "a", "status": "done"}]}, "", False))
    assert "mission" in render(ToolResult("set_mission", {"text": "ship it"}, "", False))
    run_ok = render(ToolResult("run_command", {"command": "pytest"}, "3 passed", False))
    assert "✓" in run_ok and "pytest" in run_ok and "3 passed" in run_ok, run_ok
    run_bad = render(ToolResult("run_command", {"command": "x"}, "boom", True))
    assert "✗" in run_bad and "boom" in run_bad


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
