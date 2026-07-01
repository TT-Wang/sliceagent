"""Read-only convergence nudge (render_convergence) — the 'show path' spin bug: a read-only task that
keeps exploring without answering gets nudged to answer/ask, but an edit task, a below-threshold turn,
an errored turn, and a delegated explorer do NOT. (The lesson-mining half of this file moved to
test_consolidate.py — distillation is now CACHE-only; the per-turn LessonMiner was removed.)
No model, no pytest. Run: python tests/test_mining.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.pfc import Slice  # noqa: E402
from memagent.regions import EXPLORE_NUDGE_AFTER, render_convergence  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def readonly_spin_nudges_to_answer():
    s = Slice(); s.reset("show me the path")                       # no edits ever
    s.turn_actions = EXPLORE_NUDGE_AFTER                            # explored this turn without answering
    out = render_convergence(s)
    assert "answer" in out.lower() and ("ask_user" in out or "tool calls this turn" in out)


@check
def readonly_nudge_quiet_below_threshold_and_on_error():
    s = Slice(); s.reset("t")
    s.turn_actions = 2                                            # below EXPLORE_NUDGE_AFTER → no nudge yet
    assert render_convergence(s) == ""
    s.turn_actions = 9; s.last_error = "boom"                     # an error gates the nudge even when explored a lot
    assert render_convergence(s) == ""


@check
def edit_task_uses_postedit_path_not_readonly():
    # once anything is edited, the read-only nudge is dormant — the post-edit convergence path applies
    s = Slice(); s.reset("t"); s.edited_files = {"a.py"}; s.since_edit = 3
    out = render_convergence(s)
    assert "read-only" not in out and "edited 1 file" in out


@check
def explore_mode_suppresses_readonly_nudge():
    # a delegated EXPLORER must NOT be told to stop exploring — its job IS read-only investigation, and the
    # nudge was cutting reviews short before the key (large) files were read. max_steps bounds it instead.
    s = Slice(); s.reset("review the repo"); s.turn_actions = EXPLORE_NUDGE_AFTER + 5
    assert render_convergence(s) != ""        # a normal (top-level) agent WOULD be nudged here
    s.explore_mode = True
    assert render_convergence(s) == ""        # explore_mode suppresses it


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
