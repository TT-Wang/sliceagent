"""bounded = SEAL the within-loop info, NOT a size/relevance cut. Under the accumulate-only loop the
raw trajectory IS the native transcript (kept whole within the turn, sealed at the boundary — see
test_loop_overflow); the semantic slice embodies the same principle without applying a second blunt cut:
(1) WITHIN a loop, NO slice section is size-bounded — every distinct finding stays whole (a cut harms
    the LLM).
(2) the bound is history, not useful task state — seal() CARRIES findings and the adaptive working set;
    physical context projection and SwapManager evict under real pressure. reset() wipes a new task.
No model. Run: PYTHONPATH=src python tests/test_bound_is_relevance.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.regions import record_note  # noqa: E402
from sliceagent.pfc import Slice, touch_file  # noqa: E402

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
def seal_carries_distilled_and_adaptive_working_set():
    # A turn seal drops the raw model trajectory, but does not blindly discard still-useful task state.
    s = Slice(); s.reset("loop one")
    touch_file(s, "feature.py", edited=True)        # in-progress change-set
    touch_file(s, "explored.py")                    # exploratory read
    for i in range(6):
        record_note(s, f"conclusion {i} about the feature")
    assert s.findings and "explored.py" in s.active_files, "loop one accumulated complete info"
    s.seal()                                        # the turn-boundary seal (continue_topic calls this)
    # CARRIED: SwapManager owns pressure-aware eviction; seal does not impose an edited-files-only policy.
    assert "explored.py" in s.active_files, "the adaptive working set survives an arbitrary turn boundary"
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
