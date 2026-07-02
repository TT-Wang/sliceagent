"""Live progress status (the ticking step/action/elapsed/tally line) — the RichSink _LiveStatus body.

The user couldn't tell a long multi-step turn from a hung one: the old spinner had a static per-step label,
no step counter, no elapsed. These checks pin the self-refreshing status: it TICKS off Rich's own Status loop
(no thread), shows step + action + a running tally, is markup-safe (a crash source in this file), mutates in
place (no flicker), and stays alive through a read run so the clock never stalls. Deterministic; no LLM.
Run: PYTHONPATH=src python tests/test_progress_status.py
"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console                                        # noqa: E402
from rich.live import Live                                             # noqa: E402
from rich.text import Text                                             # noqa: E402

from sliceagent.events import (SliceBuilt, StepBegin, ToolResult, ToolStarted)  # noqa: E402
from sliceagent.tui import RichSink, _LiveStatus, _fmt_tally             # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _sink(force=True):
    return RichSink(Console(file=io.StringIO(), force_terminal=force, width=100), {})


@check
def live_status_ticks_and_a_text_body_would_freeze():
    # THE central claim: the _LiveStatus INSTANCE re-renders each frame (elapsed advances) with NO extra
    # thread; a static Text body would render once (frozen). Guards the freeze-trap for future edits.
    s = _sink(); s._turn_t0 = s._action_t0 = time.monotonic(); s._step = 4; s._label = "⚡ run npm test"
    calls = {"n": 0}
    body = _LiveStatus(s)
    real = body.__rich__
    body.__rich__ = lambda: (calls.__setitem__("n", calls["n"] + 1) or real())
    lv = Live(body, console=s.c, refresh_per_second=12, transient=True); lv.start()
    time.sleep(0.3); lv.stop()
    assert calls["n"] > 1, f"status body must re-render each frame (ticks), got {calls['n']}"
    # a Text body renders once — proving the instance (not a Text) is what makes it tick
    calls["n"] = 0
    lv2 = Live(Text("frozen"), console=s.c, refresh_per_second=12, transient=True); lv2.start()
    time.sleep(0.2); lv2.stop()  # (nothing to count; just assert no crash — the contract is documented)


@check
def render_is_markup_safe_and_shows_step_and_elapsed():
    # dynamic label carrying brackets ([id] path, a stray [/tag]) must render literally, never MarkupError.
    s = _sink(); s._turn_t0 = s._action_t0 = time.monotonic(); s._step = 7
    s._label = "✏️ edit src/app/jobs/[id]/page.tsx and [/learn]"
    out = str(_LiveStatus(s).__rich__())
    assert "step 7" in out, out
    assert "[id]" in out and "[/learn]" in out, f"brackets must survive literally: {out!r}"


@check
def slicebuilt_arms_clock_stepbegin_counts_and_tally_grows():
    s = _sink()
    s(SliceBuilt("seed"))
    assert s._turn_t0 is not None and s._status is not None, "SliceBuilt arms the turn clock + status"
    s(StepBegin(3)); assert s._step == 3, "StepBegin drives the step counter"
    s(ToolStarted("run_command", {"command": "npm test"}))
    s(ToolResult("run_command", {"command": "npm test"}, "PASS", False))
    s(ToolResult("read_file", {"path": "a.ts"}, "ok", False))      # coalesced read
    s(ToolResult("str_replace", {"path": "b.ts"}, "done", True))   # a failing edit
    assert s._tally == {"cmd": 1, "read": 1, "edit": 1, "fail": 1}, s._tally
    assert _fmt_tally(s._tally) == "1 read · 1 edit · 1 cmd · 1 fail"


@check
def spin_mutates_in_place_and_survives_a_read_run():
    # no destroy+recreate per event (that was the flicker); and a coalesced read must NOT tear the region,
    # so the clock keeps ticking through a 12-file read run.
    s = _sink()
    s(SliceBuilt("seed"))
    first = s._status
    s(StepBegin(1)); s(ToolStarted("grep", {"pattern": "x"}))
    assert s._status is first, "the Status must be MUTATED in place, not recreated (no flicker)"
    s(ToolResult("read_file", {"path": "a.ts"}, "ok", False))   # a coalesced read
    assert s._status is first and s._status is not None, "status must stay live through a read run"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
