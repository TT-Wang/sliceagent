"""bounded = SEAL the within-loop info, NOT a size/relevance cut. Under the accumulate-only loop the
raw trajectory IS the native transcript (kept whole within the turn, sealed at the boundary — see
test_loop_overflow); the distilled FINDINGS tier embodies the same principle inside the slice:
(1) WITHIN a loop, NO slice section is size-bounded — every distinct finding stays whole (a cut harms
    the LLM).
(2) the bound is the loop-boundary SEAL — seal() CARRIES the distilled context (findings + in-progress
    change-set) and drops the raw exploratory trajectory; reset() (a new task) wipes everything.
No model. Run: PYTHONPATH=src python tests/test_bound_is_relevance.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.regions import record_note  # noqa: E402
from memagent.slice import Slice, touch_file  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def within_loop_keeps_all_findings():
    # NO within-loop bound: 30 distinct conclusions all stay (an old MAX_FINDINGS=8 cap would cut 22).
    s = Slice(); s.reset("t")
    for i in range(30):
        record_note(s, f"distinct conclusion number {i} about subsystem {i}")
    assert len(s.findings) == 30, f"every distinct finding must stay within the loop, got {len(s.findings)}"


@check
def seal_carries_distilled_drops_raw():
    # the BOUND is the seal at a TURN boundary: CARRY the distilled context, SEAL the raw trajectory.
    s = Slice(); s.reset("loop one")
    touch_file(s, "feature.py", edited=True)        # in-progress change-set
    touch_file(s, "explored.py")                    # exploratory read
    for i in range(6):
        record_note(s, f"conclusion {i} about the feature")
    assert s.findings and "explored.py" in s.active_files, "loop one accumulated complete info"
    s.seal()                                        # the turn-boundary seal (continue_topic calls this)
    # SEALED (raw exploratory reads → re-readable / recallable on demand):
    assert "explored.py" not in s.active_files, "exploratory reads are sealed (re-readable on demand)"
    # CARRIED (distilled continuity into the next loop):
    assert len(s.findings) == 6, "distilled findings carry across the seal"
    assert "feature.py" in s.active_files and "feature.py" in s.edited_files, "in-progress change-set carries"


@check
def reset_wipes_everything():
    # distinct from seal(): a brand-NEW task wipes all carry too.
    s = Slice(); s.reset("loop one")
    touch_file(s, "feature.py", edited=True)
    record_note(s, "a conclusion")
    s.reset("a totally new task")
    assert s.findings == [] and s.active_files == [] and s.edited_files == set(), \
        "reset() is a full wipe (new task), unlike seal() which preserves the distilled carry"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
