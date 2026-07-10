"""Rich projection checks for the shared, UI-neutral turn progress reducer.

Run: PYTHONPATH=src python tests/test_progress_status.py
"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console                                        # noqa: E402
from rich.cells import cell_len                                         # noqa: E402
from rich.live import Live                                             # noqa: E402
from rich.text import Text                                             # noqa: E402

from sliceagent.events import (ApiRetry, ModelCallPrepared, SliceBuilt, StepBegin, StepEnd,  # noqa: E402
                               ToolResult, ToolStarted, TurnStarted)
from sliceagent.progress import TurnProgress                            # noqa: E402
from sliceagent.tui import RichSink, _LiveStatus, _fmt_tally, _render_progress  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _sink(force=True):
    return RichSink(Console(file=io.StringIO(), force_terminal=force, width=100, height=25,
                            _environ={"TERM": "xterm"}), {})


@check
def live_status_ticks_and_a_text_body_would_freeze():
    s = _sink()
    s(TurnStarted("review the parser", task_title="Parser review"))
    s._stop()  # drive the body directly below; do not run two Rich Live regions on one console
    calls = {"n": 0}

    class CountingStatus(_LiveStatus):
        def __rich__(self):
            calls["n"] += 1
            return super().__rich__()

    body = CountingStatus(s)
    lv = Live(body, console=s.c, refresh_per_second=12, transient=True); lv.start()
    time.sleep(0.3); lv.stop()
    assert calls["n"] > 1, f"status body must re-render each frame (ticks), got {calls['n']}"
    calls["n"] = 0
    lv2 = Live(Text("frozen"), console=s.c, refresh_per_second=12, transient=True); lv2.start()
    time.sleep(0.2); lv2.stop()


@check
def render_is_markup_safe_and_shows_task_pass_and_elapsed():
    s = RichSink(Console(file=io.StringIO(), force_terminal=True, width=124, height=25,
                         _environ={"TERM": "xterm"}), {})
    try:
        s(TurnStarted("edit [id] and [/learn]"))
        s(StepBegin(7))
        rendered = _LiveStatus(s).__rich__()
        out = rendered.plain
        assert "pass 7" in out, out
        assert "[id]" in out and "[/learn]" in out, f"brackets must survive literally: {out!r}"
        assert ":" in out, "elapsed time must be present"
        assert rendered.no_wrap and rendered.overflow == "ellipsis", \
            "the transient status must stay one line"
    finally:
        s._stop()


@check
def responsive_progress_preserves_truth_at_60_80_and_120_columns():
    machine = TurnProgress(clock=lambda: 100.0, await_commit=True)
    machine.reduce(TurnStarted("fix retry handling", task_title="Retry fix", plan=[
        {"step": "inspect retry path", "status": "done"},
        {"step": "add regression test", "status": "in_progress"},
        {"step": "verify full suite", "status": "pending"},
    ]))
    machine.reduce(StepBegin(7))
    machine.reduce(ModelCallPrepared(7, 2, []))
    machine.reduce(ToolResult("read_file", {"path": "retry.py"}, "ok", False))
    machine.reduce(ToolStarted("run_command", {"command": "pytest -q tests/test_retry.py"}))
    snap = machine.snapshot()

    rows = {width: _render_progress(snap, width, now=142.0) for width in (60, 80, 120)}
    for width, row in rows.items():
        assert cell_len(row.plain) <= width, (width, row.plain)
        assert "Running" in row.plain and "00:42" in row.plain, row.plain
        assert row.no_wrap and row.overflow == "ellipsis"
    assert "add regression test" not in rows[60].plain
    assert "add regression" in rows[80].plain and "add regression" in rows[120].plain
    assert "pass 7" not in rows[60].plain and "pass 7" not in rows[80].plain
    assert "pass 7" in rows[120].plain and "attempt 2" in rows[120].plain
    assert "1 read" in rows[120].plain


@check
def production_order_never_resets_the_first_model_pass_or_clock():
    s = _sink()
    s(TurnStarted("fix retry progress", task_title="Retry progress"))
    started = s.progress.snapshot().started_at
    s(StepBegin(1))
    s(SliceBuilt("seed"))
    s(ModelCallPrepared(1, 1, [{"role": "user", "content": "seed"}]))
    snap = s.progress.snapshot()
    assert snap.model_pass == 1 and snap.provider_attempt == 1, snap
    assert snap.started_at == started, "late SliceBuilt must not restart the whole-turn clock"
    assert snap.phase.value == "thinking", snap
    s._stop()


@check
def status_mutates_in_place_and_reads_flush_at_the_tool_boundary():
    s = _sink()
    s(TurnStarted("inspect files"))
    first = s._status
    s(StepBegin(1))
    s(ToolStarted("read_file", {"path": "a.ts"}))
    assert s._status is first, "the Status must mutate in place, not flicker through recreation"
    s(ToolResult("read_file", {"path": "a.ts"}, "ok", False))
    assert s._reads, "successful reads should coalesce within one tool wave"
    s(StepEnd(1, {}, "tool_use"))
    assert not s._reads, "StepEnd(tool_use) must publish the completed read wave before model wait"
    assert "a.ts" in s.c.file.getvalue(), s.c.file.getvalue()
    counts = s.progress.snapshot().counts
    assert counts == {"read": 1}, counts
    assert _fmt_tally({"read": 1, "edit": 1, "cmd": 1, "fail": 1}) == \
        "1 read · 1 edit · 1 cmd · 1 fail"
    s._stop()


@check
def retry_and_next_provider_attempt_keep_one_live_status():
    s = _sink()
    s(TurnStarted("retry safely"))
    s(StepBegin(1))
    s(ModelCallPrepared(1, 1, []))
    first = s._status
    s(ApiRetry(1, "temporary timeout", delay_s=1.2, max_attempts=3))
    assert s._status is first and s._status is not None
    assert s.progress.snapshot().phase.value == "retrying"
    s(ModelCallPrepared(1, 2, []))
    assert s._status is first and s.progress.snapshot().provider_attempt == 2
    s._stop()


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
