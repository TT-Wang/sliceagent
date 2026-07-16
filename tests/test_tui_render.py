"""TUI rendering checks for calm rails, responsive progress furniture, and token accounting.
Skips cleanly if the `tui` extra (rich) isn't installed. No model.
Run: PYTHONPATH=src python tests/test_tui_render.py
"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from rich.cells import cell_len  # noqa: E402
    from rich.console import Console  # noqa: E402
    from sliceagent.tui import (RichSink, _ToolTiming, _agent_matrix_plain_lines, _box_width,  # noqa: E402
                                _live_status_line, _render_agent_batch, _render_plan, _render_read_summary,
                                _private_prompt_history, _record_usage, _render_tool_result,
                                _response_panel, _toolbar)
    from sliceagent.tui_projection import AgentResultView, output_preview  # noqa: E402
    from sliceagent.events import (StepBegin, StepEnd, SubagentProgress, ToolResult, ToolStarted,
                                   TurnStarted)  # noqa: E402
    from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus  # noqa: E402
    from sliceagent.progress import TurnProgress  # noqa: E402
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
    sink(StepEnd(1, {"prompt_tokens": 1000, "completion_tokens": 20,
                          "input_other": 200, "input_cache_creation": 30}, "tool_use"))
    sink(StepEnd(2, {"prompt_tokens": 1000, "completion_tokens": 10, "input_other": 50}, "stop"))
    assert sink.stats["fresh"] == 280, sink.stats
    assert sink.stats["tokens"] == 2030, sink.stats

    chitchat_stats = {"model": "deepseek-reasoner"}
    _record_usage(chitchat_stats, {
        "prompt_tokens": 12, "completion_tokens": 4, "input_other": 12, "output": 4,
    })
    assert chitchat_stats["tokens"] == 16 and chitchat_stats["cost"] > 0, chitchat_stats


@check
def prompt_history_is_created_and_repaired_private():
    import stat
    import tempfile

    home = tempfile.mkdtemp(prefix="tui-private-history-")
    old_home = os.environ.get("HOME")
    old_userprofile = os.environ.get("USERPROFILE")
    old_umask = os.umask(0o022)
    try:
        os.environ["HOME"] = home
        os.environ["USERPROFILE"] = home
        state = os.path.join(home, ".sliceagent")
        os.makedirs(state)
        path = os.path.join(state, "history")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("legacy\n")
        os.chmod(path, 0o644)
        history = _private_prompt_history()
        history.append_string("private request")
        if os.name != "nt":
            assert stat.S_IMODE(os.stat(state).st_mode) == 0o700
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    finally:
        os.umask(old_umask)
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_userprofile is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = old_userprofile


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
def agent_groups_and_busy_meter_are_width_safe_with_wide_text():
    agents = [AgentResultView(
        invocation_id=f"agent-{index}", launch_ordinal=index, kind="explorer",
        name=f"审查员-{index}", task="检查渲染器并验证并发状态不会回退" * 3,
        status="failed" if index == 2 else "succeeded",
        stop_cause="provider_timeout" if index == 2 else "complete",
        recovered_from=(), artifact_id=f"child-{index}", detail="详细错误" * 30,
        duration_s=12.4, report_completion="complete",
    ) for index in range(1, 5)]
    for width in (60, 80, 120):
        buf = io.StringIO()
        console = Console(file=buf, width=width, force_terminal=False, color_system=None, soft_wrap=False)
        console.print(_render_agent_batch(agents, width))
        assert all(cell_len(line) <= width for line in buf.getvalue().splitlines()), buf.getvalue()
        status = "".join(fragment[1] for fragment in _live_status_line(
            "◌ Delegating — 4 agents running · 审查员-4 · grep renderer",
            {"model": "unknown", "tokens": 987_654, "saved_cached_tok": 123_456}, width,
        ))
        assert cell_len(status) <= width, (width, status)


@check
def live_agent_matrix_keeps_report_readiness_separate_from_source_partial():
    machine = TurnProgress(await_commit=True)
    machine.reduce(TurnStarted("merge reports", turn_id="turn-source"))
    machine.reduce(StepBegin(1))
    invocation = ToolInvocation(
        "synth-1", "spawn_agent", {"agent": "synthesiser", "task": "merge reports"}, 0,
    )
    machine.reduce(ToolStarted(invocation.name, dict(invocation.args), invocation))
    outcome_effect = ToolEffect("child-source:outcome", "child_outcome", {
        "kind": "synthesiser", "status": "ok",
        "report_completion": "complete",
        "source_coverage_status": "source_partial",
    })
    artifact_effect = ToolEffect("child-source:artifact", "child_artifact", {
        "artifact_id": "child-source",
    })
    outcome = ToolOutcome(
        invocation, ToolStatus.SUCCEEDED, "report", (outcome_effect, artifact_effect),
    )
    machine.reduce(ToolResult(
        invocation.name, dict(invocation.args), outcome.text, False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    ))

    rendered = "\n".join(line for _, line in _agent_matrix_plain_lines(machine.snapshot(), 120))
    assert "1 ready" in rendered and "1 source partial" in rendered, rendered
    assert "✓ ready" in rendered and "source partial" in rendered, rendered
    assert "ground" not in rendered and "verified" not in rendered, rendered


@check
def live_agent_matrix_is_stable_bounded_and_cell_safe():
    machine = TurnProgress(await_commit=True)
    machine.reduce(TurnStarted("review", turn_id="turn-matrix"))
    machine.reduce(StepBegin(1))
    # Reverse callback order and wide text exercise stable physical request order and cell cropping.
    calls = []
    for index in range(1, 16):
        invocation = ToolInvocation(
            f"spawn-{index}", "spawn_agent",
            {"agent": "explorer", "name": f"审查员-{index}", "task": f"检查区域 {index}"},
            index - 1,
        )
        calls.append(invocation)
        machine.reduce(ToolStarted(invocation.name, dict(invocation.args), invocation))
    for index in reversed(range(1, 16)):
        phase = "failed" if index in {3, 14} else ("report_ready" if index % 3 == 0 else "running")
        machine.subagent_activity(SubagentProgress(
            f"child-{index}", "turn-matrix", index, "explorer", f"审查员-{index}", 1,
            phase, "检查渲染器并验证并发状态" * 4, index, index,
            invocation_id=f"spawn-{index}", request_ordinal=index, objective=f"检查区域 {index}",
        ))
    snap = machine.snapshot()
    for width in (40, 60, 80, 120):
        lines = _agent_matrix_plain_lines(snap, width, now=time.monotonic() + 2)
        assert len(lines) <= 12, "summary + header + 8 rows/overflow must stay bounded"
        assert all(cell_len(line) <= width for _, line in lines), (width, lines)
        joined = "\n".join(line for _, line in lines)
        assert "15" in joined and "failed" in joined and "hidden" in joined, joined
        visible_ids = [line.split()[0] for _, line in lines if line.strip()[:1].isdigit()]
        assert visible_ids == sorted(visible_ids, key=int), visible_ids


@check
def nested_matrix_ids_follow_request_order_not_callback_order():
    machine = TurnProgress(await_commit=True)
    machine.reduce(TurnStarted("review", turn_id="turn-tree"))
    machine.reduce(StepBegin(1))
    root = ToolInvocation(
        "spawn-root", "spawn_agent", {"agent": "explorer", "task": "root"}, 0,
    )
    machine.reduce(ToolStarted(root.name, dict(root.args), root))
    machine.subagent_activity(SubagentProgress(
        "child-root", "turn-tree", 1, "explorer", "root", 1,
        "running", "root work", 1, 1,
        invocation_id=root.id, request_ordinal=1,
    ))
    for ordinal in (2, 1):
        machine.subagent_activity(SubagentProgress(
            f"nested-{ordinal}", "turn-tree", ordinal, "explorer", f"nested-{ordinal}", 2,
            "running", "nested work", 1, 1,
            parent_agent_id="child-root", invocation_id=f"child-root/call_1_{ordinal}",
            request_ordinal=ordinal,
        ))
    joined = "\n".join(
        line for _, line in _agent_matrix_plain_lines(machine.snapshot(), 100)
    )
    first, second = joined.index("1.1"), joined.index("1.2")
    assert first < second and "nested-1" in joined and "nested-2" in joined, joined


@check
def typed_timing_never_falls_back_to_a_same_name_sibling():
    now = [0.0]
    timing = _ToolTiming(clock=lambda: now[0])
    real = ToolInvocation("real", "spawn_agent", {"task": "real"}, 0)
    timing.start(ToolStarted(real.name, dict(real.args), real))
    now[0] = 5.0
    assert timing.settle(ToolResult(
        "spawn_agent", {"task": "rejected"}, "not started", True,
        invocation_id="other",
    )) is None
    now[0] = 10.0
    assert timing.settle(ToolResult(
        "spawn_agent", dict(real.args), "done", False, invocation_id="real",
    ), ended_at=2.5) == 2.5


@check
def output_preview_scans_large_results_with_constant_retained_rows():
    result = output_preview(("frame\n" * 100_000) + "root cause", max_rows=3)
    assert result.lines == ("frame", "frame", "root cause")
    assert result.hidden_lines == 99_998 and result.tail_retained
    started = time.monotonic()
    blank_heavy = output_preview(("\n" * 100_000) + "tail", max_rows=3)
    assert time.monotonic() - started < 2.0, "newline-heavy preview regressed to superlinear scanning"
    assert blank_heavy.lines[-1:] == ("tail",)


@check
def narrow_single_agent_uses_a_complete_manifest_handle():
    view = AgentResultView(
        invocation_id="agent-1", launch_ordinal=1, kind="explorer", name="", task="audit",
        status="succeeded", stop_cause="complete", recovered_from=(),
        artifact_id="subagent-" + ("a" * 32), detail="report", request_ordinal=1,
        report_completion="complete",
    )
    buf = io.StringIO()
    console = Console(file=buf, width=58, force_terminal=False, color_system=None, soft_wrap=False)
    console.print(_render_agent_batch([view], 58))
    rendered = buf.getvalue()
    assert "artifact · artifacts/index.md" in rendered and "…" not in rendered, rendered


@check
def response_panel_and_footer_are_responsive_at_60_80_and_120():
    stats = {
        "workspace": "demo", "model": "deepseek-reasoner",
        "topic": "fix retry handling", "tokens": 999_999, "fresh": 1_234,
        "saved_cached_tok": 42_000, "last_turn_s": 65,
    }
    for width, expected_box in ((60, 58), (80, 78), (120, 108)):
        buf = io.StringIO()
        console = Console(file=buf, width=width, force_terminal=False, color_system=None, soft_wrap=False)
        assert _box_width(console) == expected_box
        console.print(_response_panel("A response with `code` and enough prose to wrap cleanly.", console))
        response_lines = buf.getvalue().splitlines()
        assert all(cell_len(line) <= width for line in response_lines), buf.getvalue()
        assert not response_lines[1].strip() and not response_lines[-2].strip(), \
            "assistant prose needs one blank row above and below"

        footer = _toolbar(stats, lambda width=width: width)()
        plain = "".join(fragment[1] for fragment in footer)
        assert cell_len(plain) <= width and "sliceagent" in plain
        assert ("demo" in plain) is (width >= 72)
        assert ("deepseek-reasoner" in plain) is (width >= 72)
        # The legacy DeepSeek alias is priced as its current V4 Flash target until retirement.
        assert "1.0M tok" in plain and "$0.0001 saved" in plain, plain
        assert ("1.2k fresh" in plain) is (width >= 72), plain
        assert ("⏲ 01:05" in plain) is (width >= 110), plain
        assert "fix retry handling" not in plain, "task prompts must never become pinned footer chrome"

        unicode_stats = dict(stats, workspace="切片代理工作区非常长", topic="修复重试逻辑并验证")
        unicode_footer = _toolbar(unicode_stats, lambda width=width: width)()
        unicode_plain = "".join(fragment[1] for fragment in unicode_footer)
        assert cell_len(unicode_plain) <= width, (width, unicode_plain)


@check
def completer_does_slash_and_files():
    from prompt_toolkit.document import Document
    from sliceagent.tui import _InputCompleter
    comp = _InputCompleter(files=["src/sliceagent/util.py", "tests/test_util.py", "README.md",
                                  "app/jobs/[id]/page.tsx", "docs/my guide.md", "bad\nname.py"])
    # slash palette at line start
    slash = [c.text for c in comp.get_completions(Document("/pl"), None)]
    assert "/plan" in slash, slash
    # filename completion on an explicit @mention (basename-prefix first) — the ONLY trigger; matches
    # the @path syntax cli.py's message parser already recognizes for pinning/attaching a file.
    files = [c.text for c in comp.get_completions(Document("please edit @util"), None)]
    assert "src/sliceagent/util.py" in files and "tests/test_util.py" in files, files
    bracketed = [c.text for c in comp.get_completions(Document("inspect @[id]"), None)]
    assert "app/jobs/[id]/page.tsx" in bracketed, bracketed
    spaced = [c.text for c in comp.get_completions(Document("read @guide"), None)]
    assert '"docs/my guide.md"' in spaced, spaced
    assert list(comp.get_completions(Document("read @bad"), None)) == [], \
        "control-character filenames must never be inserted into the prompt"
    # plain prose (no @) must NOT pop a completion menu, even on a word that matches a real file —
    # this is the exact annoyance the @-gating fixes.
    assert list(comp.get_completions(Document("please edit util"), None)) == []
    # short words / mid-prose don't spam
    assert list(comp.get_completions(Document("a"), None)) == []


@check
def mention_parser_accepts_completion_syntax_and_confines_resolution():
    import tempfile
    from sliceagent.mentions import parse_mentions, workspace_mentions

    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
        os.makedirs(os.path.join(root, "app", "jobs", "[id]"))
        os.makedirs(os.path.join(root, "docs"))
        bracketed = os.path.join(root, "app", "jobs", "[id]", "page.tsx")
        spaced = os.path.join(root, "docs", "my guide.md")
        comma = os.path.join(root, "literal,")
        for path in (bracketed, spaced, comma):
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("x")
        secret = os.path.join(outside, "secret.txt")
        with open(secret, "w", encoding="utf-8") as stream:
            stream.write("secret")
        os.symlink(secret, os.path.join(root, "escape.txt"))

        text = ('inspect @app/jobs/[id]/page.tsx and @"docs/my guide.md", '
                'then @app/jobs/[id]/page.tsx; ignore user@example.com')
        assert parse_mentions(text) == ["app/jobs/[id]/page.tsx", "docs/my guide.md",
                                          "app/jobs/[id]/page.tsx;"]
        assert workspace_mentions(text, root) == ["app/jobs/[id]/page.tsx", "docs/my guide.md"]
        # Exact filename wins over punctuation fallback; traversal and symlink escapes never resolve.
        assert workspace_mentions("@literal, @../secret.txt @escape.txt", root) == ["literal,"]
        assert workspace_mentions("please inspect (@app/jobs/[id]/page.tsx)", root) == [
            "app/jobs/[id]/page.tsx",
        ]
        assert parse_mentions('@"bad\nname.py"') == [], "quoted controls must not enter mention paths"


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
