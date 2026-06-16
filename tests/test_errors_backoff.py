"""Jittered backoff + retry behaviour (W5 errors.py).
No model, no pytest. Run: python tests/test_errors_backoff.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent import errors                                # noqa: E402
from memagent.errors import jittered_backoff, with_retry   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def jittered_in_range_attempt_1():
    # attempt 1 -> delay = base_delay (0.5); jitter uniform in [0, jitter_ratio*delay]
    base, jr = 0.5, 0.5
    lo = base
    hi = base * (1 + jr)
    for _ in range(200):
        v = jittered_backoff(1, base_delay=base, max_delay=5.0, jitter_ratio=jr)
        assert lo <= v <= hi, f"{v} not in [{lo}, {hi}]"


@check
def grows_exponentially_then_capped():
    # base part (delay before jitter) doubles each attempt until max_delay caps it.
    # Use jitter_ratio=0 so the value IS exactly the deterministic delay.
    assert jittered_backoff(1, base_delay=0.5, max_delay=5.0, jitter_ratio=0.0) == 0.5
    assert jittered_backoff(2, base_delay=0.5, max_delay=5.0, jitter_ratio=0.0) == 1.0
    assert jittered_backoff(3, base_delay=0.5, max_delay=5.0, jitter_ratio=0.0) == 2.0
    assert jittered_backoff(4, base_delay=0.5, max_delay=5.0, jitter_ratio=0.0) == 4.0
    # 0.5 * 2^4 = 8.0 -> capped at max_delay=5.0
    assert jittered_backoff(5, base_delay=0.5, max_delay=5.0, jitter_ratio=0.0) == 5.0
    assert jittered_backoff(20, base_delay=0.5, max_delay=5.0, jitter_ratio=0.0) == 5.0


@check
def two_same_attempt_calls_differ():
    # jitter (counter + time seed) decorrelates two calls at the same attempt
    seen = {jittered_backoff(3, base_delay=0.5, max_delay=5.0, jitter_ratio=0.5) for _ in range(50)}
    assert len(seen) > 1, "jitter not applied — identical delays across calls"


@check
def with_retry_retries_then_reraises():
    slept = []
    orig_sleep = errors.time.sleep
    errors.time.sleep = lambda s: slept.append(s)        # monkeypatch: don't actually wait
    try:
        calls = {"n": 0}
        def boom():
            calls["n"] += 1
            raise RuntimeError("rate limit exceeded")    # classify -> retryable
        raised = False
        try:
            with_retry(boom, max_attempts=3)
        except RuntimeError:
            raised = True
        assert raised, "should re-raise after exhausting attempts"
        assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"
        assert len(slept) == 2, f"expected 2 backoff sleeps (between 3 attempts), got {len(slept)}"
    finally:
        errors.time.sleep = orig_sleep


@check
def no_retry_on_non_retryable():
    slept = []
    orig_sleep = errors.time.sleep
    errors.time.sleep = lambda s: slept.append(s)
    try:
        calls = {"n": 0}
        class Forbidden(Exception):
            status_code = 403
        def denied():
            calls["n"] += 1
            raise Forbidden("denied")                    # classify -> retryable=False
        raised = False
        try:
            with_retry(denied, max_attempts=3)
        except Forbidden:
            raised = True
        assert raised, "non-retryable should propagate immediately"
        assert calls["n"] == 1, f"non-retryable must not retry, got {calls['n']} calls"
        assert slept == [], "non-retryable must not sleep"
    finally:
        errors.time.sleep = orig_sleep


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
