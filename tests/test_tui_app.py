"""Tests for the full-screen Textual TUI (src/memagent/tui_app.py).

Uses Textual's `Pilot` async test harness. The agent loop is mocked so tests run without an LLM.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from textual.widgets import Input, ListView, RichLog, Static, TextArea, Tree
except Exception as exc:  # pragma: no cover
    print(f"SKIP: Textual not installed ({exc})")
    raise SystemExit(0)

from rich.console import Console

from memagent import events as E
from memagent.session import Session
from memagent.slice import record_user
from memagent.tui_app import (
    MemagentTui,
    PaletteScreen,
    ConfirmScreen,
    AskUserScreen,
    DiffScreen,
)


def _render_to_text(renderable) -> str:
    console = Console(force_terminal=False, width=120)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()



def _make_session() -> Session:
    """A session with a pre-seeded active topic so the UI has state to render."""
    s = Session(None)
    s.new_topic("refactor parser")
    s.active().mission = "make the parser robust"
    s.active().plan = [
        {"step": "find grammar bug", "status": "done"},
        {"step": "rewrite rule", "status": "in_progress"},
        {"step": "add tests", "status": "pending"},
    ]
    s.active().requirements = [{"text": "use exact token names", "done": False}]
    s.active().active_files = ["src/parser.py", "src/lexer.py"]
    s.active().edited_files = {"src/parser.py"}
    s.active().findings.append("the bug is in rule X")
    return s


def _make_app(run_turn=None, make_build_slice=None, route_topic=None) -> MemagentTui:
    """Factory that wires mocked callbacks so no real LLM or loop is invoked."""
    session = _make_session()

    async def _fake_run_turn(**kwargs):
        dispatch = kwargs.get("dispatch")
        # Emit a few representative events to exercise rendering.
        dispatch(E.SliceBuilt(rendered="system prompt", messages=[]))
        dispatch(E.AssistantText(content="I will rewrite rule X."))
        dispatch(E.ToolStarted(name="str_replace", args={"path": "src/parser.py", "old_string": "old", "new_string": "new"}))
        dispatch(E.ToolResult(name="str_replace", args={"path": "src/parser.py", "old_string": "old", "new_string": "new"}, output="ok", failing=False))
        dispatch(E.StepEnd(step=1, usage={"prompt_tokens": 100, "completion_tokens": 20, "input_other": 80}, stop_reason="tool_use"))
        dispatch(E.TurnEnd(stop_reason="end_turn", steps=1, usage={"prompt_tokens": 100, "completion_tokens": 20}))
        class _R:
            stop_reason = "end_turn"
        return _R()

    def _fake_build_slice(*args, **kwargs):
        def build():
            return [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        return build

    def _fake_route_topic(llm, line, session):
        return ("continue", "")

    app = MemagentTui(
        session=session,
        tools=None,
        retriever=None,
        memory=None,
        llm=None,
        hooks=None,
        dispatch=lambda e: None,
        run_turn=run_turn or _fake_run_turn,
        make_build_slice=make_build_slice or _fake_build_slice,
        record_user=record_user,
        route_topic=route_topic or _fake_route_topic,
        stats={"model": "test-model", "policy": "guard", "topic": "", "tokens": 0, "fresh": 0},
    )
    return app


def test_textual_available():
    from memagent.tui_app import textual_available
    assert textual_available() is True


async def test_app_composes():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#tree", Tree)
        assert app.query_one("#plan", Static)
        assert app.query_one("#conversation", RichLog)
        assert app.query_one("#input", TextArea)


async def test_sidebar_shows_working_set():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        tree = app.query_one("#tree", Tree)
        labels = [str(node.label) for node in tree.root.children]
        assert any("active files" in label for label in labels)
        assert any("edited" in label for label in labels)


async def test_plan_panel_renders():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        plan = app.query_one("#plan", Static)
        text = _render_to_text(plan.render()._renderable)
        assert "make the parser robust" in text
        assert "find grammar bug" in text
        assert "use exact token names" in text


async def test_submit_input_runs_turn():
    calls = {"turns": 0}

    async def fake_run(**kwargs):
        calls["turns"] += 1
        dispatch = kwargs.get("dispatch")
        dispatch(E.SliceBuilt(rendered="", messages=[]))
        dispatch(E.TurnEnd(stop_reason="end_turn", steps=1, usage={"prompt_tokens": 1, "completion_tokens": 1}))
        class _R:
            stop_reason = "end_turn"
        return _R()

    app = _make_app(run_turn=fake_run)
    async with app.run_test() as pilot:
        await pilot.pause()
        input_widget = app.query_one("#input", TextArea)
        input_widget.text = "continue refactoring"
        await pilot.press("ctrl+enter")
        await pilot.pause()
        # Allow the worker to complete.
        await asyncio.sleep(0.1)
        await pilot.pause()
    assert calls["turns"] == 1


async def test_events_render_in_conversation():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Manually feed the sink a complete mocked turn.
        sink = MemagentTui.make_sink(app)
        sink(E.SliceBuilt(rendered="system prompt", messages=[]))
        sink(E.AssistantText(content="I will rewrite rule X."))
        sink(E.ToolStarted(name="str_replace", args={"path": "src/parser.py", "old_string": "old", "new_string": "new"}))
        sink(E.ToolResult(name="str_replace", args={"path": "src/parser.py", "old_string": "old", "new_string": "new"}, output="ok", failing=False))
        sink(E.StepEnd(step=1, usage={"prompt_tokens": 100, "completion_tokens": 20, "input_other": 80}, stop_reason="tool_use"))
        sink(E.TurnEnd(stop_reason="end_turn", steps=1, usage={"prompt_tokens": 100, "completion_tokens": 20}))
        await pilot.pause()
        log = app.query_one("#conversation", RichLog)
        lines = log.lines
        text = "\n".join(str(line) for line in lines)
        assert "I will rewrite rule X" in text
        assert "done" in text


async def test_new_topic_action():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert app._session.active_id is not None


async def test_stats_update_after_step():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        sink = MemagentTui.make_sink(app)
        sink(E.StepEnd(step=1, usage={"prompt_tokens": 100, "completion_tokens": 20, "input_other": 80}, stop_reason="tool_use"))
        await pilot.pause()
        assert app._stats["tokens"] == 120
        assert app._stats["fresh"] == 80


async def test_tool_output_folding():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        sink = MemagentTui.make_sink(app)
        long_output = "\n".join(f"line {i}" for i in range(20))
        sink(E.ToolResult(name="run_command", args={"command": "cat big.log"}, output=long_output, failing=False))
        await pilot.pause()
        log = app.query_one("#conversation", RichLog)
        text = "\n".join(str(line) for line in log.lines)
        assert "more lines" in text
        assert "line 0" in text
        assert "line 19" not in text  # folded away


async def test_fuzzy_palette_filters():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(PaletteScreen())
        await pilot.pause()
        inp = app.screen.query_one("#pinput", Input)
        inp.value = "diff"
        await pilot.pause()
        lst = app.screen.query_one("#plist", ListView)
        labels = [str(child.children[0].render()) for child in lst.children]
        assert any("/diff" in label for label in labels)
        assert not any("/exit" in label for label in labels)


async def test_confirm_dialog_returns_value():
    import threading
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._dialog_event = threading.Event()
        app.push_screen(ConfirmScreen("run_command", "rm -rf /", "destructive command"))
        await pilot.pause()
        await pilot.click("#yes")
        await pilot.pause()
        assert app._dialog_result[0] == "yes"


async def test_ask_user_dialog_returns_value():
    import threading
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._dialog_event = threading.Event()
        app.push_screen(AskUserScreen("Which color?", ["red", "blue"]))
        await pilot.pause()
        await pilot.click("#opt-red")
        await pilot.pause()
        assert app._dialog_result[0] == "red"


async def test_diff_screen_unified():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(DiffScreen("parser.py", "old line\nsecond", "new line\nsecond"))
        await pilot.pause()
        ta = app.screen.query_one("#diff", TextArea)
        text = ta.text
        assert "--- a/parser.py" in text
        assert "+++ b/parser.py" in text
        assert "-old line" in text
        assert "+new line" in text


async def test_slash_diff_command():
    app = _make_app()
    # Provide a fake tool host with a root method so /diff can try to read files.
    class FakeTools:
        def root(self):
            return "/tmp"
    app._tools = FakeTools()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._session.active().edited_files = {"src/parser.py"}
        app._session.active().edit_anchor["src/parser.py"] = "original"
        # Create the file so read succeeds.
        os.makedirs("/tmp/src", exist_ok=True)
        with open("/tmp/src/parser.py", "w", encoding="utf-8") as f:
            f.write("changed content\n")
        app._handle_slash("/diff")
        await pilot.pause()
        assert isinstance(app.screen, DiffScreen)


if __name__ == "__main__":
    import asyncio
    tests = [
        test_textual_available,
        test_app_composes,
        test_sidebar_shows_working_set,
        test_plan_panel_renders,
        test_submit_input_runs_turn,
        test_events_render_in_conversation,
        test_new_topic_action,
        test_stats_update_after_step,
        test_tool_output_folding,
        test_fuzzy_palette_filters,
        test_confirm_dialog_returns_value,
        test_ask_user_dialog_returns_value,
        test_diff_screen_unified,
        test_slash_diff_command,
    ]
    for t in tests:
        try:
            if asyncio.iscoroutinefunction(t):
                asyncio.run(t())
            else:
                t()
            print(f"ok: {t.__name__}")
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
