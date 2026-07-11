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
import tempfile

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
    from sliceagent.events import (AssistantText, StepBegin, StepEnd, ToolResult, ToolStarted,
                                   TurnCommitted, TurnEnd, TurnStarted)
    console, buf = _rec_console()
    statuses = []
    sink = LiveSink(console, {}, lambda s: statuses.append(s), await_commit=True)

    sink(TurnStarted("the request"))                                  # → preparing
    sink(StepBegin(1))                                                 # → thinking
    sink(ToolStarted("read_file", {"path": "parser.py"}))            # → inspecting
    sink(ToolResult("read_file", {"path": "parser.py"}, "code", False))
    sink(StepEnd(1, {}, "tool_use"))                                  # publishes read wave → integrating
    sink.on_delta("content", "here is the answer")                    # → writing, no response tail
    sink(AssistantText("the final answer text"))                      # held until durable commit
    sink(TurnEnd("end_turn", 2, {}))                                  # → finalizing, not done
    sink(TurnCommitted(True, "end_turn", detail="checkpoint saved"))  # → reply + saved + idle

    assert any("Thinking" in (s or "") for s in statuses), statuses
    assert any("Inspecting" in (s or "") and "parser.py" in (s or "") for s in statuses), statuses
    assert any("Integrating" in (s or "") for s in statuses), statuses
    assert any("Writing" in (s or "") for s in statuses), statuses
    assert any("Finalizing" in (s or "") for s in statuses), statuses
    assert all("the request" not in (s or "") for s in statuses), \
        "the submitted prompt must not be pinned in the reasoning/status row"
    assert statuses[-1] is None, "only durable commit may clear the active status"
    out = buf.getvalue()
    assert "the final answer text" in out, "the reply must print ABOVE the box"
    assert "parser.py" in out, "the tool card must print above the box"
    assert "turn saved" in out, out


@check
def livesink_keeps_finalizing_until_the_response_is_committed():
    from sliceagent.tui import LiveSink
    from sliceagent.events import AssistantText, TurnCommitted, TurnEnd, TurnStarted
    console, buf = _rec_console()
    statuses = []
    sink = LiveSink(console, {}, lambda s: statuses.append(s), await_commit=True)
    sink(TurnStarted("q"))
    sink.on_delta("content", "streaming the answer now")
    assert "Writing" in (statuses[-1] or ""), statuses
    sink(AssistantText("the final answer"))
    assert "Finalizing" in (statuses[-1] or ""), statuses
    assert "the final answer" not in buf.getvalue(), "terminal answer must wait for commit"
    sink(TurnEnd("end_turn", 1, {}))
    assert statuses[-1] is not None, "TurnEnd is not durable completion"
    sink(TurnCommitted(True, "end_turn"))
    assert statuses[-1] is None and "the final answer" in buf.getvalue()


@check
def livesink_completion_uses_the_same_sealed_receipt_truth():
    from sliceagent.tui import LiveSink
    from sliceagent.events import TurnCommitted, TurnEnd, TurnStarted
    console, buf = _rec_console()
    sink = LiveSink(console, {}, lambda _status: None, await_commit=True)
    receipt = {
        "disposition": "indeterminate",
        "counts": {
            "requested": 1, "rejected_before_execution": 0, "execution_started": 1,
            "settled": 0, "succeeded": 0, "failed": 0, "cancelled": 0,
            "indeterminate": 1, "not_started": 0,
        },
        "agents": {},
    }
    sink(TurnStarted("run operation"))
    sink(TurnEnd("indeterminate", 1, {}))
    sink(TurnCommitted(True, "indeterminate", receipt=receipt))
    rendered = buf.getvalue()
    assert "indeterminate state saved" in rendered, rendered
    assert "1 started" in rendered and "1 indeterminate" in rendered, rendered


@check
def livesink_preserves_the_same_adverse_counts_at_eighty_columns():
    from sliceagent.tui import LiveSink
    from sliceagent.events import TurnCommitted, TurnEnd, TurnStarted
    console, buf = _rec_console()
    sink = LiveSink(console, {}, lambda _status: None, await_commit=True)
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
    sink(TurnStarted("delegate"))
    sink(TurnEnd("end_turn", 1, {}))
    sink(TurnCommitted(True, "end_turn", receipt=receipt))
    rendered = buf.getvalue()
    assert "13 rejected before start" in rendered and "1 failed" in rendered, rendered


@check
def livesink_read_card_is_header_only_like_richsink():
    # parity with RichSink: read/list cards show no content dump (shared _render_tool_result)
    from sliceagent.tui import LiveSink
    from sliceagent.events import StepEnd, ToolResult, TurnStarted
    console, buf = _rec_console()
    sink = LiveSink(console, {}, lambda s: None)
    sink(TurnStarted("inspect x.py"))
    sink(ToolResult("read_file", {"path": "x.py"}, "SECRET-CONTENT", False))
    sink(StepEnd(1, {}, "tool_use"))
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
def live_app_cwd_switch_stays_open_for_the_next_turn():
    turn_calls, slash_calls = [], []

    def switch_slash(text):
        slash_calls.append(text)
        return "switched"                 # ordinary handled state, deliberately NOT the old "restart"

    state, out = _drive_live(
        "/cwd /tmp/next\rinspect target\r\x04",
        lambda text, *_: turn_calls.append(text),
        handle_slash=switch_slash,
    )
    assert slash_calls == ["/cwd /tmp/next"]
    assert turn_calls == ["inspect target"], \
        "the same live Application must accept another turn after /cwd"
    assert state["last"] == "inspect target" and state["running"] is False
    assert "turn error" not in out.lower()


@check
def live_workspace_refresh_preserves_application_and_rebinds_completion():
    from prompt_toolkit.document import Document
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.tui import build_live_app

    current = tempfile.mkdtemp(prefix="live-current-")
    target = tempfile.mkdtemp(prefix="live-target-")
    with open(os.path.join(current, "current_only.py"), "w", encoding="utf-8") as f:
        f.write("CURRENT = True\n")
    with open(os.path.join(target, "target_only.py"), "w", encoding="utf-8") as f:
        f.write("TARGET = True\n")
    console, _buf = _rec_console()
    with create_pipe_input() as pinp:
        app, state = build_live_app(
            console=console, stats={"model": "test-model", "workspace": "current"}, root=current,
            run_one_turn=lambda *_: None, pt_input=pinp, pt_output=DummyOutput(),
        )
        app_id = id(app)
        before = {item.text for item in app.current_buffer.completer.get_completions(
            Document("@current"), None,
        )}
        assert "current_only.py" in before
        state["set_workspace"](target)
        after = {item.text for item in app.current_buffer.completer.get_completions(
            Document("@target"), None,
        )}
        stale = {item.text for item in app.current_buffer.completer.get_completions(
            Document("@current"), None,
        )}
        assert id(app) == app_id, "completion refresh must not recreate/reconnect the live terminal app"
        assert "target_only.py" in after and "current_only.py" not in stale


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
def shared_tool_renderer_uses_the_same_visual_grammar():
    # Both adapters share the same calm rails: compact plan, neutral success, explicit failure.
    from sliceagent.tui import _render_tool_result
    from sliceagent.events import ToolResult
    from rich.console import Console

    def render(e):
        c = Console(file=io.StringIO(), force_terminal=False, width=80, soft_wrap=False)
        c.print(_render_tool_result(e))
        return c.file.getvalue()

    assert "│ plan 1/1 · complete" in render(
        ToolResult("update_plan", {"steps": [{"step": "a", "status": "done"}]}, "", False)
    )
    run_ok = render(ToolResult("run_command", {"command": "pytest"}, "3 passed", False))
    assert "│ run pytest" in run_ok and "3 passed" in run_ok and "✓" not in run_ok, run_ok
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
