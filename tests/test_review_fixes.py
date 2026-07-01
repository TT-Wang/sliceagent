"""Kimi-review fixes #1 (OPEN USER REPORT detection) and #2 (BudgetHook per-turn reset).
No model, no pytest. Run: PYTHONPATH=src python tests/test_review_fixes.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.hooks import BudgetHook            # noqa: E402
from memagent.regions import is_user_report  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── #1: failure-report detection (catch the misses WITHOUT false-positiving benign follow-ups) ──
_REPORTS = [
    "tests are red", "the tests are failing", "it hangs", "it's frozen now",
    "nothing happens when I click run", "that didn't fix it", "did not fix the bug",
    "still the same error", "same issue", "I get a 500", "HTTP 500 error",
    "the endpoint returns a 404", "ModuleNotFoundError: No module named foo",
    "the build is broken", "it doesn't run", "it can't play at all",
    "cd: no such file or directory",
]
_BENIGN = [
    "add a 500ms timeout", "also build the docs", "now add some tests",
    "the function works, optimize it", "make the button red", "rebuild the index",
    "let's continue with the next feature", "same approach as the other file",
    "explain how the parser works", "create calc.py", "run the tests please",
    "the tests pass — add one more",
]

@check
def reports_are_detected():
    missed = [m for m in _REPORTS if not is_user_report(m)]
    assert not missed, f"failure reports NOT detected: {missed}"

@check
def benign_followups_are_not_flagged():
    false_pos = [m for m in _BENIGN if is_user_report(m)]
    assert not false_pos, f"benign follow-ups WRONGLY flagged as failure reports: {false_pos}"


# ── #2: BudgetHook is a PER-TURN cap (resets each turn), not a silent whole-session cap ──
@check
def budget_stops_the_turn_at_the_cap():
    h = BudgetHook(100); h.reset_for_turn()
    assert h.record_step_usage({"prompt_tokens": 60, "completion_tokens": 0}) is None  # 60 < 100
    r = h.record_step_usage({"prompt_tokens": 60, "completion_tokens": 0})             # 120 >= 100
    assert r and r.get("stop_turn"), r

@check
def budget_resets_each_turn():
    h = BudgetHook(100); h.reset_for_turn()
    h.record_step_usage({"prompt_tokens": 90, "completion_tokens": 20})  # spent 110, turn 1 done
    assert h.spent >= 100
    h.reset_for_turn()                                                   # next user task
    assert h.spent == 0, "budget must reset per turn (not accumulate across the session)"
    assert h.record_step_usage({"prompt_tokens": 50, "completion_tokens": 0}) is None  # fresh budget


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
