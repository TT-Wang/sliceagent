"""Guidance strings/functions for budget-ceiling wording.
No model, no pytest. Run: python tests/test_guidance.py

Module-level assertions only. The run_turn integration case is owned by W1.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.guidance import BUDGET_EXHAUSTED  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def max_steps_mentions_step_ceiling_and_summarize_next():
    msg = BUDGET_EXHAUSTED("max_steps")
    low = msg.lower()
    assert "step" in low                                     # names the step ceiling
    assert "summarize" in low                                # asks to summarize progress
    assert "next" in low                                     # + the single most useful next action


@check
def token_budget_mentions_token_ceiling():
    msg = BUDGET_EXHAUSTED("token_budget")
    assert "token" in msg.lower()
    assert "summarize" in msg.lower()


@check
def budget_does_not_silently_retry():
    # both kinds must steer away from silent retry / looping
    for kind in ("max_steps", "token_budget"):
        low = BUDGET_EXHAUSTED(kind).lower()
        assert "retry" in low or "loop" in low or "keep working" in low


@check
def budget_unknown_kind_is_safe():
    # never raises / never KeyErrors on an unexpected kind
    msg = BUDGET_EXHAUSTED("something_else")
    assert isinstance(msg, str) and msg
    assert "summarize" in msg.lower()


@check
def budget_is_stable_per_kind():
    # pure function of its argument — same input, byte-identical output (cache-safe)
    assert BUDGET_EXHAUSTED("max_steps") == BUDGET_EXHAUSTED("max_steps")
    assert BUDGET_EXHAUSTED("max_steps") != BUDGET_EXHAUSTED("token_budget")


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
