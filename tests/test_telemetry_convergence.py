"""Core-review #8 (reconstruction telemetry) + #5 (convergence resets on a new finding, not just an edit).
No model, no pytest. Run: PYTHONPATH=src python tests/test_telemetry_convergence.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import StepEnd, ToolResult            # noqa: E402
from sliceagent.pfc import Slice, slice_sink  # noqa: E402
from sliceagent.regions import record_note  # noqa: E402
from sliceagent.telemetry import Telemetry                   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── #8: reconstruction telemetry ─────────────────────────────────────────────
@check
def telemetry_counts_rereads_recalls_and_reads():
    t = Telemetry(window=6)
    t(ToolResult("read_file", {"path": "a.py"}, "data", False))   # read 1
    t(StepEnd(0, {}, "x"))
    t(ToolResult("read_file", {"path": "a.py"}, "data", False))   # re-read within window
    t(ToolResult("read_file", {"path": "b.py"}, "data", False))   # read of b
    t(ToolResult("search_history", {"query": "x"}, "...", False))  # a recall (cross-session content search)
    t(ToolResult("read_file", {"path": "c.py"}, "Error", True))   # FAILED read — not counted
    s = t.summary()
    assert s["reads"] == 3 and s["re_reads"] == 1 and s["recalls"] == 1, s


@check
def telemetry_respects_the_window():
    t = Telemetry(window=2)
    t(ToolResult("read_file", {"path": "x"}, "d", False))
    for _ in range(3):
        t(StepEnd(0, {}, "x"))                                    # advance 3 steps (> window 2)
    t(ToolResult("read_file", {"path": "x"}, "d", False))         # gap 3 > 2 → NOT a re-read
    assert t.summary()["re_reads"] == 0, t.summary()


# ── #5: convergence counter resets on a genuinely-new finding ────────────────
@check
def record_note_reports_new_vs_dup():
    s = Slice(); s.reset("t")
    assert record_note(s, "root cause is the null check") is True     # new
    assert record_note(s, "root cause is the null check") is False    # dup refresh
    assert record_note(s, "Let me check the file") is False           # narration dropped
    assert record_note(s, "") is False                                # empty


@check
def since_edit_resets_on_new_finding_not_on_dup_or_noop():
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    s.since_edit = 3
    sink(ToolResult("read_file", {"path": "a.py", "note": "the bug is in parse()"}, "data", False))
    assert s.since_edit == 0, "a NEW finding (active learning) must reset the convergence counter"
    sink(ToolResult("read_file", {"path": "b.py"}, "data", False))    # no note, no edit → no progress
    assert s.since_edit == 1
    sink(ToolResult("read_file", {"path": "c.py", "note": "the bug is in parse()"}, "data", False))  # dup
    assert s.since_edit == 2, "a DUPLICATE note is not new knowledge → must NOT reset"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
