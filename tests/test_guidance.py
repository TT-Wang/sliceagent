"""Guidance strings/functions — denial + budget-ceiling wording.
No model, no pytest. Run: python tests/test_guidance.py

Module-level assertions only. The integration cases (denied tool -> Error in
slice.last_error; run_turn max_steps dispatches TurnInterrupted with this message)
are owned by W1 and asserted in their wiring tests, not here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.guidance import (                              # noqa: E402
    BUDGET_EXHAUSTED,
    DENIAL_NO_PROMPT,
    DENIAL_USER,
)

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


@check
def denial_no_prompt_has_do_not_retry_guidance():
    low = DENIAL_NO_PROMPT.lower()
    assert isinstance(DENIAL_NO_PROMPT, str) and DENIAL_NO_PROMPT
    assert "do not retry" in low                             # action-oriented: don't spin
    assert "instead" in low                                  # + do X instead
    assert "permission" in low or "approv" in low            # names the cause


@check
def denial_user_has_do_not_retry_guidance():
    low = DENIAL_USER.lower()
    assert isinstance(DENIAL_USER, str) and DENIAL_USER
    assert "do not retry" in low                             # action-oriented: don't spin
    assert "instead" in low or "different" in low            # + do X instead
    assert "declin" in low or "user" in low                  # names the cause


@check
def denials_are_distinct():
    assert DENIAL_NO_PROMPT != DENIAL_USER


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
