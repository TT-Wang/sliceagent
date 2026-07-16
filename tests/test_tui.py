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
from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _render(events, *, width=100):
    # This helper captures settled scrollback, not Rich's live cursor protocol.  An
    # xterm-like TERM can otherwise make a forced-terminal StringIO interactive on
    # Linux, adding Rich-owned cursor-hide/redraw escapes that the terminal-safety
    # assertions correctly reject when they come from model or tool text.
    c = Console(
        file=StringIO(), force_terminal=True, force_interactive=False,
        width=width, color_system=None,
    )
    stats = {"model": "m", "topic": "", "tokens": 0}
    sink = tui.make_rich_sink(c, stats, await_commit=True)
    sink._spinner_on = False
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
def indeterminate_tool_is_unknown_not_an_ordinary_failure():
    invocation = ToolInvocation("cmd-1", "run_command", {"command": "deploy"}, 0)
    outcome = ToolOutcome(invocation, ToolStatus.INDETERMINATE, "connection dropped\ncheck remote state")
    out, _ = _render([ToolResult(
        "run_command", {"command": "deploy"}, outcome.text, True,
        status="indeterminate", invocation_id="cmd-1", outcome=outcome,
    )])
    assert "! run deploy · state unknown" in out, out
    assert "connection dropped" in out and "check remote state" in out, out


def _agent_result(call_id, ordinal, task, status, *, cause="", report="report", source_coverage="",
                  evidence_status="", evidence_account=None, report_completion="complete"):
    invocation = ToolInvocation(call_id, "spawn_agent", {"agent": "explorer", "task": task}, ordinal - 1)
    effect = ToolEffect(f"{call_id}:child", "child_outcome", {
        "artifact_id": f"child-{ordinal}", "kind": "explorer", "launch_ordinal": ordinal,
        "status": "ok" if status is ToolStatus.SUCCEEDED else "error",
        "stop_reason": "end_turn" if status is ToolStatus.SUCCEEDED else "error",
        "stop_cause": cause or ("complete" if status is ToolStatus.SUCCEEDED else "error"),
        "report_completion": report_completion,
        **({"source_coverage_status": source_coverage} if source_coverage else {}),
        **({"explorer_evidence_status": evidence_status} if evidence_status else {}),
        **({"explorer_evidence": evidence_account} if evidence_account is not None else {}),
    })
    outcome = ToolOutcome(invocation, status, report, (effect,))
    return ToolResult(
        "spawn_agent", dict(invocation.args), report, status is not ToolStatus.SUCCEEDED,
        status=status.value, invocation_id=call_id, outcome=outcome,
    )


@check
def parallel_agents_render_as_one_typed_outcome_group():
    out, _ = _render([
        _agent_result("agent-2", 2, "audit UI", ToolStatus.FAILED,
                      cause="provider_timeout", report="Error: provider timed out"),
        _agent_result("agent-1", 1, "audit persistence", ToolStatus.SUCCEEDED,
                      report="a very long child report that belongs in its artifact"),
        StepEnd(1, {}, "tool_use"),
    ])
    assert "agents · 1/2 reports ready" in out and "1 timed out" in out, out
    assert out.index("1 explorer") < out.index("2 explorer"), "launch order must survive reverse completion"
    assert "audit persistence" in out and "audit UI" in out, out
    assert "provider timeout" in out and "Error: provider timed out" in out, out
    assert "a very long child report" not in out, \
        "the TUI must not duplicate a direct report when its durable locator is available"
    assert "artifacts · artifacts/index.md · 2 stored" in out, out


@check
def operational_success_and_partial_source_coverage_render_as_separate_facts():
    result = _agent_result(
        "synth-1", 1, "merge reports", ToolStatus.SUCCEEDED,
        source_coverage="source_partial",
    )
    out, _ = _render([result, StepEnd(1, {}, "tool_use")])
    assert "agents · 1/1 reports ready" in out, out
    assert "source coverage · 1 source partial" in out, out
    assert "source partial" in out and "ground" not in out and "verified" not in out, out


@check
def typed_evidence_quality_is_visible_without_becoming_execution_failure():
    out, _ = _render([
        _agent_result(
            "nav", 1, "map files", ToolStatus.SUCCEEDED,
            evidence_status="navigation_only",
            evidence_account={"navigation_success_count": 4},
        ),
        _agent_result(
            "partial", 2, "inspect code", ToolStatus.SUCCEEDED,
            evidence_status="content_partial",
            evidence_account={"content_success_count": 5, "retained_content_view_count": 2},
        ),
        _agent_result(
            "none", 3, "inspect missing area", ToolStatus.SUCCEEDED,
            evidence_status="none", report="content retained according to prose",
        ),
        _agent_result(
            "legacy", 4, "inspect legacy child", ToolStatus.SUCCEEDED,
            report="navigation_only according to prose",
        ),
        StepEnd(1, {}, "tool_use"),
    ])
    assert "agents · 4/4 reports ready" in out, out
    assert "1 navigation only" in out and "1 content partial" in out and "1 no evidence" in out, out
    assert "1 not assessed" in out, "missing typed evidence must not be inferred from report prose"
    assert "evidence nav 4" in out and "evidence partial 2/5" in out, out
    assert "✗" not in out, "weak/absent evidence is not an execution failure"


@check
def multiline_tool_output_is_bounded_without_flattening_the_cause():
    output = "headline\nframe one\nframe two\nframe three\nframe four\nroot cause"
    out, _ = _render([ToolResult("run_command", {"command": "pytest"}, output, True)])
    assert "headline" in out and "root cause" in out and "… 1 line omitted" in out, out


@check
def tool_preview_cannot_replay_terminal_control_sequences():
    out, _ = _render([ToolResult(
        "run_command", {"command": "demo"},
        "\x1b[31mred\x1b[0m\n\x1b]0;spoof-title\x07done", False,
    )])
    assert "red" in out and "done" in out, out
    assert "\x1b" not in out and "\x07" not in out, repr(out)


@check
def all_model_and_argument_text_is_terminal_safe():
    out, _ = _render([
        ToolResult("read_file", {"path": "src/\x1b[2Jspoof.py"}, "ok", False),
        StepEnd(1, {}, "tool_use"),
        AssistantText(
            "Answer \x1b]0;fake-title\x07with **markdown** and \x1b[2Jclear "
            "plus C1 \u009b2J and bidi left\u202eright"
        ),
        TurnEnd("end_turn", 1, {}), TurnCommitted(True, "end_turn"),
    ])
    assert "spoof.py" in out and "markdown" in out and "clear" in out, out
    assert "\x1b" not in out and "\x07" not in out and "\u009b" not in out and "\u202e" not in out, repr(out)


@check
def unknown_explicit_tool_status_is_indeterminate():
    out, _ = _render([ToolResult(
        "run_command", {"command": "deploy"}, "provider extension status", False,
        status="timed_out",
    )])
    assert "! run deploy · state unknown" in out, out


@check
def failed_agent_fanout_has_a_hard_row_cap_and_manifest():
    events = [
        _agent_result(f"agent-{index}", index, f"audit area {index}", ToolStatus.FAILED,
                      cause="provider_timeout", report=f"Error: failure detail {index}")
        for index in range(1, 101)
    ]
    out, _ = _render([*events, StepEnd(1, {}, "tool_use")], width=80)
    lines = out.splitlines()
    assert len(lines) <= 20, f"agent group flooded scrollback with {len(lines)} rows"
    assert "100 timed out" in out and "… 92 more agents" in out, out
    assert "artifacts · artifacts/index.md · 100 stored" in out, out


@check
def mixed_rejected_and_started_agents_keep_distinct_request_ordinals():
    rejected_invocation = ToolInvocation(
        "agent-rejected", "spawn_agent", {"agent": "explorer", "task": "invalid request"}, 0,
    )
    rejected = ToolOutcome(rejected_invocation, ToolStatus.FAILED, "rejected before start")
    started_invocation = ToolInvocation(
        "agent-started", "spawn_agent", {"agent": "explorer", "task": "real audit"}, 1,
    )
    child = ToolEffect("agent-started:child", "child_outcome", {
        "artifact_id": "child-real", "kind": "explorer", "launch_ordinal": 1,
        "stop_cause": "complete",
    })
    started = ToolOutcome(started_invocation, ToolStatus.SUCCEEDED, "report", (child,))
    out, _ = _render([
        ToolResult("spawn_agent", dict(rejected_invocation.args), rejected.text, True,
                   status="failed", invocation_id=rejected_invocation.id, outcome=rejected),
        ToolResult("spawn_agent", dict(started_invocation.args), started.text, False,
                   status="succeeded", invocation_id=started_invocation.id, outcome=started),
        StepEnd(1, {}, "tool_use"),
    ])
    assert "1 explorer — invalid request" in out and "2 explorer — real audit" in out, out


@check
def legacy_success_without_an_artifact_keeps_its_only_report_visible():
    out, _ = _render([
        ToolResult("spawn_agent", {"agent": "explorer", "task": "legacy audit"},
                   "legacy inline report", False, status="succeeded"),
        StepEnd(1, {}, "tool_use"),
    ])
    assert "legacy audit" in out and "legacy inline report" in out, out


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
    assert out.count("agent note · The retry regression now passes.") == 1, out
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
    assert "agents · 0/1 succeeded" in out and "agents · 1 rejected before start" in out, out
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
def every_adverse_receipt_state_survives_narrow_completion_rows():
    receipt = {
        "turn_status": "indeterminate", "disposition": "indeterminate",
        "counts": {
            "requested": 7, "rejected_before_execution": 1, "execution_started": 4,
            "settled": 6, "succeeded": 1, "failed": 1, "cancelled": 2,
            "lifecycle_not_run": 1, "indeterminate": 1, "not_started": 1,
        },
        "agents": {
            "requested": 7, "rejected_before_execution": 1, "execution_started": 4,
            "settled": 6, "succeeded": 1, "failed": 1, "cancelled": 2,
            "lifecycle_not_run": 1, "indeterminate": 1, "not_started": 1,
        },
    }
    out, _ = _render([
        TurnStarted("delegate"), TurnEnd("indeterminate", 1, {}),
        TurnCommitted(True, "indeterminate", receipt=receipt),
    ], width=60)
    for fact in (
        "agents · 1/7 succeeded", "agents · 1 rejected before start", "agents · 1 failed",
        "agents · 1 cancelled", "agents · 1 indeterminate", "agents · 1 not started",
        "agents · 1 not run",
    ):
        assert fact in out, (fact, out)


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
        StepBegin(2),                 # optional oracle requested another pass → discard prior draft
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
    assert "assistant update" in update and "I will inspect" in update, update


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
    assert got == ["/model"], got
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
