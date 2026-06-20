"""bounded = SEAL the within-loop info, NOT a size/relevance cut. Two guarantees:
(1) WITHIN a loop, NO section is bounded — every distinct finding/observation stays whole (a cut harms
    the LLM); the only reduction is non-lossy dedup (a re-run supersedes its obsolete output; a read of a
    resident file is shown in full by OPEN FILES).
(2) the bound is the loop-boundary SEAL — reset() (a new loop) starts fresh, not carrying raw accumulation.
No model. Run: PYTHONPATH=src python tests/test_bound_is_relevance.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.regions import record_note, record_action  # noqa: E402
from memagent.slice import Slice, build_artifacts, render_slice, touch_file  # noqa: E402
from memagent.regions import _NO_CAP  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _T:
    def read_text(self, p):
        return f"# {p}\nvalue = 1\n"
    def root(self):
        return "."


@check
def within_loop_keeps_all_findings():
    # NO within-loop bound: 30 distinct conclusions all stay (old MAX_FINDINGS=8 would have cut 22).
    s = Slice(); s.reset("t")
    for i in range(30):
        record_note(s, f"distinct conclusion number {i} about subsystem {i}")
    assert len(s.findings) == 30, f"every distinct finding must stay within the loop, got {len(s.findings)}"


@check
def within_loop_keeps_all_observations():
    # NO within-loop bound on RECENT: 30 distinct command outputs all stay (old K=4 would have cut 26).
    s = Slice(); s.reset("t")
    for i in range(30):
        record_action(s, "run_command", {"command": f"echo {i}"}, f"output line for command {i}")
    assert len(s.recent) == 30, f"every distinct observation must stay within the loop, got {len(s.recent)}"
    # and the renderer (real loop, _NO_CAP) shows them all — no render-side cut either
    out = render_slice(s, "ART", window=_NO_CAP, max_findings=_NO_CAP)
    assert "command 0" in out and "command 29" in out, "render must include the whole loop's RECENT"


@check
def dedup_is_nonlossy():
    # a re-run supersedes its own obsolete output (info not lost — the latest IS the truth)
    s = Slice(); s.reset("t")
    for _ in range(5):
        record_action(s, "run_command", {"command": "pytest"}, "1 failed")
    record_action(s, "run_command", {"command": "pytest"}, "all passed")
    pytest_entries = [e for e in s.recent if "pytest" in e["action"]]
    assert len(pytest_entries) == 1 and "all passed" in pytest_entries[0]["observation"], \
        "an exact re-run supersedes its obsolete output (non-lossy dedup, not a within-loop cut)"
    # a read of a now-resident file is shown IN FULL by OPEN FILES, so its raw RECENT obs is redundant
    touch_file(s, "mod.py")
    record_action(s, "read_file", {"path": "mod.py"}, "# mod.py\nvalue = 1\n")
    assert not any(e.get("path") == "mod.py" for e in s.recent), \
        "a resident file's read obs is redundant with OPEN FILES (non-lossy), so it isn't duplicated in RECENT"
    assert "mod.py" in s.active_files, "the file itself stays resident (shown in full by OPEN FILES)"


@check
def seal_carries_distilled_seals_raw():
    # the BOUND is the seal at a TURN boundary: CARRY the distilled context, SEAL the raw trajectory.
    s = Slice(); s.reset("loop one")
    touch_file(s, "feature.py", edited=True)        # in-progress change-set
    touch_file(s, "explored.py")                    # exploratory read
    for i in range(6):
        record_note(s, f"conclusion {i} about the feature")
        record_action(s, "run_command", {"command": f"cmd {i}"}, f"out {i}")
    assert s.findings and s.recent and "explored.py" in s.active_files, "loop one accumulated complete info"
    s.seal()                                        # the turn-boundary seal (continue_topic calls this)
    # SEALED (raw → archived + recall-on-demand):
    assert s.recent == [], "raw RECENT trajectory is sealed"
    assert s.step_log == [], "raw step cache is sealed"
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
    assert s.findings == [] and s.recent == [] and s.active_files == [] and s.edited_files == set(), \
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
