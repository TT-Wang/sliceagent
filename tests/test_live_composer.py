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
def live_status_keeps_token_and_savings_orientation_while_busy():
    from sliceagent.tui import _live_status_line
    rendered = "".join(fragment[1] for fragment in _live_status_line(
        "◌ Delegating — 3 agents running · ui · grep renderer",
        {"model": "unknown-model", "tokens": 12_345, "saved_cached_tok": 67_890},
        80,
    ))
    assert "Delegating" in rendered, rendered
    assert "12.3k tok" in rendered and "67.9k tok saved" in rendered, rendered


@check
def livesink_accepts_typed_child_progress_and_rejects_old_turns():
    from sliceagent.events import SubagentProgress, ToolStarted, TurnStarted
    from sliceagent.execution import ToolInvocation
    from sliceagent.tui import LiveSink
    console, _buf = _rec_console()
    statuses = []
    sink = LiveSink(console, {}, statuses.append, await_commit=True)
    sink(TurnStarted("review", turn_id="turn-live"))
    invocation = ToolInvocation("spawn-live", "spawn_agent", {
        "agent": "explorer", "task": "audit renderer",
    }, 0)
    sink(ToolStarted(invocation.name, dict(invocation.args), invocation))
    sink.subagent_notify(SubagentProgress(
        "child-live", "turn-live", 1, "explorer", "ui", 1,
        "running", "grep tui.py", 4, 4,
    ))
    assert "1 agent running" in (statuses[-1] or "") and "ui" in (statuses[-1] or ""), statuses
    before = statuses[-1]
    sink.subagent_notify(SubagentProgress(
        "child-old", "turn-old", 1, "explorer", "old", 1,
        "running", "wrong turn", 1, 1,
    ))
    assert statuses[-1] == before and "wrong turn" not in (statuses[-1] or ""), statuses


@check
def live_fanout_matrix_heartbeats_and_transitions_once_to_results():
    import threading
    import time
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.events import (AssistantText, StepBegin, StepEnd, SubagentProgress, ToolResult,
                                   ToolStarted, TurnCommitted, TurnEnd, TurnStarted)
    from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
    from sliceagent.tui import build_live_app

    console, buf = _rec_console()
    release = threading.Event()
    observations = []

    def fake_turn(_text, sink, _signal):
        sink(TurnStarted("fan out", turn_id="turn-matrix"))
        sink(StepBegin(1))
        invocations = []
        for index, name in enumerate(("ui", "memory", "scheduler"), 1):
            invocation = ToolInvocation(
                f"spawn-{index}", "spawn_agent",
                {"agent": "explorer", "name": name, "task": f"audit {name}"}, index - 1,
            )
            invocations.append(invocation)
            sink(ToolStarted(invocation.name, dict(invocation.args), invocation))
            sink.subagent_notify(SubagentProgress(
                f"child-{index}", "turn-matrix", index, "explorer", name, 1,
                "running", f"read {name}.py", index, 1,
                invocation_id=invocation.id, request_ordinal=index, objective=f"audit {name}",
            ))
        for index in (1, 2):
            sink.subagent_notify(SubagentProgress(
                f"child-{index}", "turn-matrix", index, "explorer", ("ui", "memory")[index - 1], 1,
                "report_ready", "report ready", index + 2, 2,
                invocation_id=f"spawn-{index}", request_ordinal=index,
            ))
        release.wait(timeout=3)
        sink.subagent_notify(SubagentProgress(
            "child-3", "turn-matrix", 3, "explorer", "scheduler", 1,
            "report_ready", "report ready", 5, 2,
            invocation_id="spawn-3", request_ordinal=3,
        ))
        for index, invocation in enumerate(invocations, 1):
            effect = ToolEffect(f"effect-{index}", "child_outcome", {
                "artifact_id": f"child-{index}", "report_completion": "complete",
            })
            outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "report ready", (effect,))
            sink(ToolResult(
                invocation.name, dict(invocation.args), outcome.text, False,
                status="succeeded", invocation_id=invocation.id, outcome=outcome,
            ))
        sink(StepEnd(1, {}, "tool_use"))
        sink(AssistantText("done", final=True))
        sink(TurnEnd("end_turn", 1, {}))
        sink(TurnCommitted(True, "end_turn"))

    with create_pipe_input() as pinp:
        app, state = build_live_app(
            console=console, stats={"model": "test"}, root=None, run_one_turn=fake_turn,
            pt_input=pinp, pt_output=DummyOutput(),
        )
        pinp.send_text("review\r")

        def inspect_and_release():
            deadline = time.monotonic() + 2
            while state.get("progress") is None and time.monotonic() < deadline:
                time.sleep(0.01)
            first = "".join(fragment[1] for fragment in state["render_status"]())
            first_height = state["status_height"]()
            time.sleep(1.05)  # no child event: the prompt-toolkit heartbeat must still advance elapsed time
            second = "".join(fragment[1] for fragment in state["render_status"]())
            observations.extend((first, second, first_height))
            release.set()
            deadline = time.monotonic() + 2
            while state.get("running") and time.monotonic() < deadline:
                time.sleep(0.01)
            pinp.send_text("\x04")

        inspector = threading.Thread(target=inspect_and_release, daemon=True)
        inspector.start()
        app.run()
        inspector.join(timeout=3)
    for thread in state.get("threads", []):
        thread.join(timeout=2)

    first, second, height = observations
    assert height >= 5 and all(name in first for name in ("ui", "memory", "scheduler")), first
    assert "2 reports ready" in first and "1 working" in first, first
    assert first != second and "00:01" in second, (first, second)
    rendered = buf.getvalue()
    assert rendered.count("agents · 3/3 reports ready") == 1, rendered
    assert "done" in rendered and state["running"] is False


@check
def live_matrix_respects_tiny_terminal_height():
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output import DummyOutput
    from sliceagent.events import SubagentProgress, ToolStarted, TurnStarted
    from sliceagent.execution import ToolInvocation
    from sliceagent.progress import TurnProgress
    from sliceagent.tui import build_live_app

    class SizedOutput(DummyOutput):
        def __init__(self, rows):
            self._rows = rows

        def get_size(self):
            return Size(rows=self._rows, columns=80)

    progress = TurnProgress(await_commit=True)
    progress.reduce(TurnStarted("fanout", turn_id="turn-tiny"))
    for index in range(1, 13):
        invocation = ToolInvocation(
            f"spawn-{index}", "spawn_agent", {"agent": "explorer", "task": str(index)}, index - 1,
        )
        progress.reduce(ToolStarted(invocation.name, dict(invocation.args), invocation))
        progress.subagent_activity(SubagentProgress(
            f"child-{index}", "turn-tiny", index, "explorer", f"agent-{index}", 1,
            "running", "working", index, 1,
            invocation_id=invocation.id, request_ordinal=index,
        ))

    for rows in (4, 6, 8, 12):
        console, _buf = _rec_console()
        _app, state = build_live_app(
            console=console, stats={}, root=None,
            run_one_turn=lambda *_args: None, pt_output=SizedOutput(rows),
        )
        state["running"] = True
        state["status"] = "◌ Delegating"
        state["progress"] = progress.snapshot()
        rendered = "".join(fragment[1] for fragment in state["render_status"]())
        height = state["status_height"]()
        assert height + 3 <= rows, (rows, height, rendered)
        assert rendered.count("\n") + 1 == height, (rows, height, rendered)
        if rows >= 6:
            assert "agents 12" in rendered, (rows, rendered)


@check
def replayed_terminal_tool_result_does_not_duplicate_the_agent_group():
    from sliceagent.events import StepBegin, StepEnd, SubagentProgress, ToolResult, ToolStarted, TurnStarted
    from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
    from sliceagent.tui import LiveSink

    console, buf = _rec_console()
    sink = LiveSink(console, {}, lambda _status: None, await_commit=True)
    invocation = ToolInvocation(
        "spawn-one", "spawn_agent", {"agent": "explorer", "task": "audit"}, 0,
    )
    sink(TurnStarted("audit", turn_id="turn-replay"))
    sink(StepBegin(1))
    sink(ToolStarted(invocation.name, dict(invocation.args), invocation))
    sink.subagent_notify(SubagentProgress(
        "child-one", "turn-replay", 1, "explorer", "", 1,
        "report_ready", "report ready", 2, 2,
        invocation_id=invocation.id, request_ordinal=1,
    ))
    effect = ToolEffect("child-one:effect", "child_outcome", {
        "artifact_id": "child-one", "report_completion": "complete",
    })
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "report", (effect,))
    event = ToolResult(
        invocation.name, dict(invocation.args), "report", False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    )
    sink(event)
    sink(event)  # dispatcher interruption recovery is deliberately at-least-once
    sink(StepEnd(1, {}, "tool_use"))
    rendered = buf.getvalue()
    assert rendered.count("agents · 1/1 reports ready") == 1, rendered
    assert rendered.count("1 explorer — audit") == 1, rendered


@check
def replayed_nonagent_result_is_presented_once():
    from sliceagent.events import StepBegin, StepEnd, ToolResult, ToolStarted, TurnStarted
    from sliceagent.execution import ToolInvocation, ToolOutcome, ToolStatus
    from sliceagent.tui import LiveSink

    console, buf = _rec_console()
    sink = LiveSink(console, {}, lambda _status: None, await_commit=True)
    invocation = ToolInvocation("read-once", "read_file", {"path": "a.py"}, 0)
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "contents")
    event = ToolResult(
        invocation.name, dict(invocation.args), outcome.text, False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    )
    sink(TurnStarted("read", turn_id="turn-read-replay"))
    sink(StepBegin(1))
    sink(ToolStarted(invocation.name, dict(invocation.args), invocation))
    sink(event)
    sink(event)
    sink(StepEnd(1, {}, "tool_use"))
    rendered = buf.getvalue()
    assert rendered.count("read") == 1 and rendered.count("a.py") == 1, rendered


@check
def rejected_terminal_callback_cannot_poison_agent_duration():
    import time
    from sliceagent.events import StepBegin, StepEnd, SubagentProgress, ToolResult, ToolStarted, TurnStarted
    from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
    from sliceagent.tui import LiveSink

    console, buf = _rec_console()
    sink = LiveSink(console, {}, lambda _status: None, await_commit=True)
    invocation = ToolInvocation(
        "spawn-duration", "spawn_agent", {"agent": "explorer", "task": "audit"}, 0,
    )
    sink(TurnStarted("audit", turn_id="turn-duration"))
    sink(StepBegin(1))
    sink(ToolStarted(invocation.name, dict(invocation.args), invocation))
    time.sleep(0.06)
    # Wrong-turn terminal hints are rejected by the reducer and therefore cannot become timing truth.
    sink.subagent_notify(SubagentProgress(
        "child-duration", "old-turn", 1, "explorer", "", 1,
        "report_ready", "wrong turn", 0, 1,
        invocation_id=invocation.id, request_ordinal=1,
    ))
    effect = ToolEffect("duration:effect", "child_outcome", {
        "artifact_id": "child-duration", "report_completion": "complete",
    })
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "report", (effect,))
    sink(ToolResult(
        invocation.name, dict(invocation.args), outcome.text, False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    ))
    sink(StepEnd(1, {}, "tool_use"))
    rendered = buf.getvalue()
    assert "00:00" not in rendered and "wrong turn" not in rendered, rendered


@check
def retired_livesink_ignores_late_model_and_event_callbacks():
    from sliceagent.events import ToolResult, TurnStarted
    from sliceagent.tui import LiveSink
    console, _buf = _rec_console()
    statuses = []
    sink = LiveSink(console, {}, statuses.append, await_commit=True)
    sink(TurnStarted("old turn", turn_id="old"))
    sink.retire()
    before = list(statuses)
    sink.on_delta("content", "late old bytes")
    sink(ToolResult("run_command", {"command": "old"}, "late", False))
    assert statuses == before and sink.progress.snapshot().turn_id == "old"


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
    assert "operations · 0/1 succeeded" in rendered and "operations · 1 indeterminate" in rendered, rendered


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
def live_ctrl_j_inserts_a_newline_without_submitting_early():
    calls = []
    state, _out = _drive_live(
        "inspect parser.py\nthen run its tests\r\x04",
        lambda text, *_: calls.append(text),
    )
    assert calls == ["inspect parser.py\nthen run its tests"], calls
    assert state["last"] == calls[0]


@check
def live_composer_round_trips_mid_turn_input_without_a_second_turn():
    import threading
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.tui import build_live_app

    console, buf = _rec_console()
    holder, answers, turns = {}, [], []

    def fake_turn(text, _sink, _signal):
        turns.append(text)
        answers.append(holder["state"]["request_input"]("Allow this exact call?", ("Yes", "No", "Always")))

    with create_pipe_input() as pinp:
        app, state = build_live_app(
            console=console, stats={"model": "test-model", "topic": "demo"}, root=None,
            run_one_turn=fake_turn, pt_input=pinp, pt_output=DummyOutput(),
        )
        holder["state"] = state
        pinp.send_text("start work\r")
        answer_timer = threading.Timer(0.15, lambda: pinp.send_text("1\r"))
        exit_timer = threading.Timer(0.35, lambda: pinp.send_text("\x04"))
        answer_timer.start(); exit_timer.start()
        app.run()
        answer_timer.join(timeout=1); exit_timer.join(timeout=1)
    for thread in state.get("threads", []):
        thread.join(timeout=2)
    assert turns == ["start work"], "the answer must resolve the pending request, not launch another turn"
    assert answers == ["Yes"]
    assert "Allow this exact call?" in buf.getvalue()


@check
def live_ctrl_d_releases_pending_input_and_waits_for_the_worker():
    import threading
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.tui import build_live_app

    console, _buf = _rec_console()
    holder, answers = {}, []

    def fake_turn(_text, _sink, _signal):
        answers.append(holder["state"]["request_input"]("Need approval", ("Yes", "No")))

    with create_pipe_input() as pinp:
        app, state = build_live_app(
            console=console, stats={"model": "test-model"}, root=None,
            run_one_turn=fake_turn, pt_input=pinp, pt_output=DummyOutput(),
        )
        holder["state"] = state
        pinp.send_text("start\r")
        exit_timer = threading.Timer(0.15, lambda: pinp.send_text("\x04"))
        exit_timer.start()
        app.run()
        exit_timer.join(timeout=1)
    for thread in state.get("threads", []):
        thread.join(timeout=2)
    assert answers == [""], answers
    assert state["running"] is False
    assert not any(thread.is_alive() for thread in state.get("threads", []))


@check
def live_ui_handoff_does_not_deadlock_when_the_app_closes_before_callback():
    import threading
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.tui import build_live_app

    console, _buf = _rec_console()

    class _DroppingLoop:
        def call_soon_threadsafe(self, _callback):
            # Reproduce the shutdown race: scheduling succeeds, but Application.run retires before dispatch.
            state["closing"] = True

        @staticmethod
        def is_closed():
            return False

    with create_pipe_input() as pinp:
        app, state = build_live_app(
            console=console, stats={"model": "test-model"}, root=None,
            run_one_turn=lambda *_: None, pt_input=pinp, pt_output=DummyOutput(),
        )
        app.loop = _DroppingLoop()
        app._loop_thread = object()
        answers = []
        worker = threading.Thread(
            target=lambda: answers.append(state["request_input"]("Allow?", ("Yes", "No"))),
        )
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive(), "a dropped UI callback must not strand the worker during shutdown"
        assert answers == [""]
        assert state.get("input_request") is None


@check
def live_prompt_never_consumes_a_preexisting_draft_as_consent():
    import threading
    import time
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.tui import build_live_app

    console, _buf = _rec_console()
    with create_pipe_input() as pinp:
        app, state = build_live_app(
            console=console, stats={"model": "test-model"}, root=None,
            run_one_turn=lambda *_: None, pt_input=pinp, pt_output=DummyOutput(),
        )
        app.current_buffer.text = "yes"
        answers = []
        worker = threading.Thread(
            target=lambda: answers.append(state["request_input"]("Allow?", ("Yes", "No"))),
        )
        worker.start()
        for _ in range(50):
            if state.get("input_request") is not None:
                break
            time.sleep(0.01)
        pending = state["input_request"]
        assert app.current_buffer.text == "", "draft text must be cleared before the prompt can accept Enter"
        pending["answer"] = "No"
        pending["event"].set()
        worker.join(timeout=2)
        assert answers == ["No"]
        assert app.current_buffer.text == "yes", "the unrelated draft should be restored after the answer"


@check
def live_prompt_rejects_typing_that_arrives_before_the_question_is_visible():
    import threading
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.tui import build_live_app

    console, _buf = _rec_console()
    entered_print, release_print = threading.Event(), threading.Event()
    original_print = console.print
    first = {"value": True}

    def slow_first_print(*args, **kwargs):
        if first["value"]:
            first["value"] = False
            entered_print.set()
            release_print.wait(timeout=2)
        return original_print(*args, **kwargs)

    console.print = slow_first_print
    with create_pipe_input() as pinp:
        app, state = build_live_app(
            console=console, stats={"model": "test-model"}, root=None,
            run_one_turn=lambda *_: None, pt_input=pinp, pt_output=DummyOutput(),
        )
        answers = []
        worker = threading.Thread(
            target=lambda: answers.append(state["request_input"]("Allow?", ("Yes", "No"))),
        )
        worker.start()
        assert entered_print.wait(timeout=2)
        pending = state["input_request"]
        assert pending["accepting"] is False
        app.current_buffer.text = "yes"  # typed while output is still blocked; never valid consent
        release_print.set()
        for _ in range(100):
            if pending["accepting"]:
                break
            threading.Event().wait(0.01)
        assert pending["accepting"] is True and app.current_buffer.text == ""
        pending["answer"] = "No"
        pending["event"].set()
        worker.join(timeout=2)
        assert answers == ["No"]
        assert app.current_buffer.text == "yes", "early typing should return as a draft, not authorize"


@check
def live_banner_failure_retires_installed_bridges():
    """The banner is startup work too: a renderer failure there must execute run_live's finalizer."""
    import sliceagent.tui as tui

    console, _buf = _rec_console()
    calls = []
    state = {
        "set_workspace": object(), "request_input": object(), "threads": [],
        "signal": None, "input_request": None,
    }

    class _NeverRun:
        def run(self):
            raise AssertionError("app.run must not be reached after a banner failure")

    old_build, old_banner = tui.build_live_app, tui.banner
    tui.build_live_app = lambda **_kwargs: (_NeverRun(), state)
    tui.banner = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("banner failed"))
    raised = False
    try:
        tui.run_live(
            console=console, stats={}, banner_info="probe", root=None,
            run_one_turn=lambda *_args: None,
            on_ready=lambda workspace, requester: calls.append((workspace, requester)),
        )
    except RuntimeError as exc:
        raised = str(exc) == "banner failed"
    finally:
        tui.build_live_app, tui.banner = old_build, old_banner
    assert raised
    assert state.get("closing") is True
    assert len(calls) == 2 and calls[0][0] is not None and calls[0][1] is not None
    assert calls[-1] == (None, None), calls


@check
def live_on_ready_failure_still_runs_bridge_retirement():
    import sliceagent.tui as tui

    console, _buf = _rec_console()
    state = {
        "set_workspace": object(), "request_input": object(), "threads": [],
        "signal": None, "input_request": None,
    }

    class _NeverRun:
        def run(self):
            raise AssertionError("app.run must not be reached")

    calls = []
    def on_ready(workspace, requester):
        calls.append((workspace, requester))
        if len(calls) == 1:
            raise RuntimeError("bridge install failed")

    old_build, old_banner = tui.build_live_app, tui.banner
    tui.build_live_app = lambda **_kwargs: (_NeverRun(), state)
    tui.banner = lambda *_args, **_kwargs: None
    raised = False
    try:
        tui.run_live(console=console, stats={}, banner_info="probe", root=None,
                     run_one_turn=lambda *_args: None, on_ready=on_ready)
    except RuntimeError as exc:
        raised = str(exc) == "bridge install failed"
    finally:
        tui.build_live_app, tui.banner = old_build, old_banner
    assert raised and state.get("closing") is True
    assert len(calls) == 2 and calls[-1] == (None, None), calls


@check
def live_renderer_failure_never_waits_forever_or_starts_a_second_owner():
    import threading
    import time
    import sliceagent.tui as tui

    console, _buf = _rec_console()
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()
    state = {
        "set_workspace": object(), "request_input": object(), "threads": [worker],
        "signal": threading.Event(), "input_request": None,
    }

    class _BrokenApp:
        def run(self):
            raise RuntimeError("renderer failed")

    calls = []
    old_build, old_banner = tui.build_live_app, tui.banner
    tui.build_live_app = lambda **_kwargs: (_BrokenApp(), state)
    tui.banner = lambda *_args, **_kwargs: None
    started = time.monotonic()
    error = None
    try:
        tui.run_live(
            console=console, stats={}, banner_info="probe", root=None,
            run_one_turn=lambda *_args: None,
            on_ready=lambda workspace, requester: calls.append((workspace, requester)),
            worker_retire_timeout=0.05,
        )
    except Exception as exc:  # noqa: BLE001
        error = exc
    finally:
        tui.build_live_app, tui.banner = old_build, old_banner
        release.set(); worker.join(timeout=1)
    assert isinstance(error, tui.LiveWorkerRetirementError), error
    assert time.monotonic() - started < 0.5, "retirement must have one bounded deadline"
    assert state["signal"].is_set() and calls[-1] == (None, None)


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
def live_modal_slash_suspends_instead_of_nesting_an_application():
    seen, turns = [], []
    state, _ = _drive_live(
        "/model\r", lambda *args: turns.append(args), handle_slash=lambda text: seen.append(text),
    )
    assert state["suspended_slash"] == "/model"
    assert not seen and not turns, "the selector must run only after this composer has retired"


@check
def run_live_executes_modal_then_resumes_with_the_current_workspace():
    import sliceagent.tui as tui

    console, _buf = _rec_console()
    initial, target = "/tmp/initial", "/tmp/after-switch"
    roots, modal_calls, ready_calls, banners = [], [], [], []
    states = [
        {
            "set_workspace": object(), "request_input": object(), "threads": [],
            "signal": None, "input_request": None, "root": target,
            "suspended_slash": "/model",
        },
        {
            "set_workspace": object(), "request_input": object(), "threads": [],
            "signal": None, "input_request": None, "root": target,
            "suspended_slash": "",
        },
    ]

    class App:
        @staticmethod
        def run():
            return None

    old_build, old_banner = tui.build_live_app, tui.banner
    def build(**kwargs):
        roots.append(kwargs["root"])
        return App(), states[len(roots) - 1]
    tui.build_live_app = build
    tui.banner = lambda *_args, **_kwargs: banners.append("banner")
    try:
        tui.run_live(
            console=console, stats={}, banner_info="probe", root=initial,
            run_one_turn=lambda *_args: None, handle_slash=lambda _line: None,
            handle_modal_slash=modal_calls.append,
            on_ready=lambda workspace, requester: ready_calls.append((workspace, requester)),
        )
    finally:
        tui.build_live_app, tui.banner = old_build, old_banner
    assert modal_calls == ["/model"]
    assert roots == [initial, target], "resumed completion must not revert to the launch workspace"
    assert banners == ["banner"], "a modal resume must not repaint startup chrome"
    assert ready_calls[1] == (None, None) and ready_calls[-1] == (None, None)


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


@check
def rich_and_pinned_adapters_share_one_event_orchestrator():
    """The two surfaces may own status differently, but durable output must not drift."""
    from sliceagent.events import (AssistantText, StepEnd, ToolResult, TurnEnd, TurnStarted)
    from sliceagent.tui import LiveSink, RichSink

    rich_console, rich_buf = _rec_console()
    live_console, live_buf = _rec_console()
    rich = RichSink(rich_console, {}, await_commit=False)
    live = LiveSink(live_console, {}, lambda _status: None, await_commit=False)
    # Status ownership is intentionally adapter-specific; compare only the shared event projection.
    rich._sync_status = lambda: None
    live._sync_status = lambda: None
    events = (
        TurnStarted("inspect"),
        ToolResult("read_file", {"path": "parser.py"}, "source", False),
        StepEnd(1, {}, "tool_use"),
        ToolResult("run_command", {"command": "tests"}, "3 passed", False),
        AssistantText("still checking", final=False),
        AssistantText("done", final=True),
        TurnEnd("end_turn", 2, {}),
    )
    for event in events:
        rich(event)
        live(event)
    assert rich_buf.getvalue() == live_buf.getvalue()


@check
def typed_tool_projection_normalizes_legacy_and_outcome_status():
    from sliceagent.events import ToolResult
    from sliceagent.execution import ToolInvocation, ToolOutcome, ToolStatus
    from sliceagent.tui_projection import project_tool_result

    legacy = project_tool_result(ToolResult("read_file", {}, "", False, invocation_id="legacy"))
    assert legacy.invocation_id == "legacy" and legacy.succeeded
    invocation = ToolInvocation("typed", "spawn_agent", {}, 0)
    outcome = ToolOutcome(invocation, ToolStatus.CANCELLED, "cancelled")
    typed = project_tool_result(ToolResult(
        "spawn_agent", {}, "cancelled", False, outcome=outcome,
    ))
    assert typed.invocation_id == "typed" and typed.status == "cancelled" and typed.is_delegation


@check
def queued_agents_project_before_physical_start_on_both_surfaces():
    from sliceagent.events import ToolQueued, ToolResult, ToolStarted, TurnStarted
    from sliceagent.execution import ToolInvocation
    from sliceagent.progress import TurnProgress
    from sliceagent.tui import LiveSink, RichSink, _agent_matrix_plain_lines

    invocation = ToolInvocation(
        "queued-child", "spawn_agent",
        {"agent": "explorer", "name": "capacity", "task": "inspect queueing"}, 4,
    )
    for sink in (
        RichSink(_rec_console()[0], {}, await_commit=False),
        LiveSink(_rec_console()[0], {}, lambda _status: None, await_commit=False),
    ):
        ticks = iter((10.0, 20.0, 30.0))
        sink.progress = TurnProgress(clock=lambda: next(ticks), await_commit=False)
        sink._sync_status = lambda: None
        sink(TurnStarted("delegate", turn_id="turn-queued"))
        sink(ToolQueued(invocation, "waiting for agent slot"))
        queued = sink.progress.snapshot()
        assert queued.active_tools == () and queued.counts == {}, queued
        assert len(queued.subagents) == 1
        row = queued.subagents[0]
        assert row.agent_id == "invocation:queued-child" and row.phase == "queued", row
        assert row.started_at is None and row.queued_at == 20.0, row
        assert "1 agent queued" in queued.detail and "waiting for agent slot" in queued.detail
        rendered = "\n".join(line for _style, line in _agent_matrix_plain_lines(
            queued, 100, now=25.0,
        ))
        assert "queued" in rendered and "waiting for agent slot" in rendered, rendered

        sink(ToolStarted(invocation.name, dict(invocation.args), invocation))
        started = sink.progress.snapshot()
        assert len(started.active_tools) == 1 and started.counts == {}, started
        row = started.subagents[0]
        assert row.agent_id == "invocation:queued-child" and row.phase == "starting", row
        assert row.started_at == 30.0 and row.queued_at == 20.0, row

    cancelled_invocation = ToolInvocation(
        "queued-cancelled", "spawn_agent", {"agent": "explorer", "task": "wait"}, 0,
    )
    machine = TurnProgress(clock=iter((1.0, 2.0, 3.0)).__next__, await_commit=False)
    machine.reduce(TurnStarted("delegate"))
    machine.reduce(ToolQueued(cancelled_invocation, "waiting for global capacity"))
    cancelled = machine.reduce(ToolResult(
        cancelled_invocation.name, dict(cancelled_invocation.args), "cancelled", True,
        status="cancelled", invocation_id=cancelled_invocation.id,
    ))
    assert cancelled.active_tools == () and cancelled.counts == {}, cancelled
    assert cancelled.subagents[0].phase == "cancelled"
    assert cancelled.subagents[0].started_at is None


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
