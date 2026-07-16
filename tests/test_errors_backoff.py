"""Jittered backoff + retry behaviour (W5 errors.py).
No model, no pytest. Run: python tests/test_errors_backoff.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent import errors                                # noqa: E402
from sliceagent.errors import (ImmediateRetryError, IndeterminateModelCallError,  # noqa: E402
                               PreFirstByteTimeoutError, ProviderCapacityError, RetryCancelledError,
                               TransportStartupError, jittered_backoff, with_retry)

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


@check
def compatibility_rewrite_retries_immediately_but_stays_inside_the_physical_attempt_cap():
    calls, retries = {"n": 0}, []

    def negotiate():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImmediateRetryError("drop one unsupported optional field")
        return "ok"

    assert with_retry(negotiate, max_attempts=3, dispatch=retries.append) == "ok"
    assert calls["n"] == 2
    assert len(retries) == 1 and retries[0].delay_s == 0.0


@check
def compatibility_rewrite_does_not_mint_a_fourth_request_after_two_transient_failures():
    calls, retries = {"n": 0}, []
    original_sleep = errors.time.sleep
    errors.time.sleep = lambda _seconds: None

    def fail_three_ways():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient provider timeout")
        raise ImmediateRetryError("deterministic compatibility downgrade discovered")

    try:
        with_retry(fail_three_ways, max_attempts=3, dispatch=retries.append)
        assert False, "the third physical request must exhaust the fixed request ceiling"
    except ImmediateRetryError:
        pass
    finally:
        errors.time.sleep = original_sleep
    assert calls["n"] == 3
    assert len(retries) == 2, "a terminal third failure must not announce a nonexistent fourth attempt"


@check
def silent_first_byte_timeouts_have_a_narrow_two_attempt_replay_ceiling():
    calls, retries = {"n": 0}, []
    original_sleep = errors.time.sleep
    errors.time.sleep = lambda _seconds: None

    def silent_reasoner():
        calls["n"] += 1
        raise PreFirstByteTimeoutError("provider stream timed out before its first response byte")

    try:
        with_retry(silent_reasoner, max_attempts=3, dispatch=retries.append)
        assert False, "a silent expensive request must not be replayed three times"
    except PreFirstByteTimeoutError:
        pass
    finally:
        errors.time.sleep = original_sleep
    assert calls["n"] == 2
    assert len(retries) == 1 and retries[0].max_attempts == 2
    classified = errors.classify(PreFirstByteTimeoutError("silent"))
    assert classified["retryable"] is True and classified["kind"] == errors.ErrorKind.TIMEOUT


@check
def ordinary_quick_timeouts_keep_the_general_retry_budget():
    calls = {"n": 0}
    original_sleep = errors.time.sleep
    errors.time.sleep = lambda _seconds: None

    def ordinary_timeout():
        calls["n"] += 1
        raise TimeoutError("quick transport timeout")

    try:
        with_retry(ordinary_timeout, max_attempts=3)
        assert False
    except TimeoutError:
        pass
    finally:
        errors.time.sleep = original_sleep
    assert calls["n"] == 3, "the special ceiling must not weaken ordinary transient recovery"


@check
def indeterminate_provider_call_never_retries_an_abandoned_socket():
    calls = {"n": 0}

    def abandoned():
        calls["n"] += 1
        raise IndeterminateModelCallError("provider request may still be in flight")

    try:
        with_retry(abandoned, max_attempts=3)
        assert False, "indeterminate call must surface"
    except IndeterminateModelCallError:
        pass
    assert calls["n"] == 1
    classified = errors.classify(IndeterminateModelCallError("still in flight"))
    assert classified["retryable"] is False and classified["kind"] == errors.ErrorKind.TIMEOUT


@check
def lifecycle_terminal_markers_override_a_permissive_adapter_classifier():
    for marker in (
        IndeterminateModelCallError("physical request may still be open"),
        ProviderCapacityError("all physical slots remain occupied"),
        RetryCancelledError("owning turn retired"),
        TransportStartupError("local transport generation is stuck"),
    ):
        calls = {"n": 0}

        def terminal(error=marker):
            calls["n"] += 1
            raise error

        try:
            with_retry(terminal, max_attempts=3, is_retryable=lambda _error: True)
            assert False, f"{type(marker).__name__} must surface without adapter reclassification"
        except type(marker):
            pass
        assert calls["n"] == 1, f"{type(marker).__name__} was retried by a permissive adapter"

    cancel = threading.Event()
    def indeterminate_then_owner_cancels():
        cancel.set()
        raise IndeterminateModelCallError("physical closure is still unknown")
    try:
        with_retry(
            indeterminate_then_owner_cancels,
            is_retryable=lambda _error: True,
            should_cancel=cancel.is_set,
        )
        assert False, "physical uncertainty must not be relabelled by a simultaneous owner cancellation"
    except IndeterminateModelCallError:
        pass


@check
def string_http_statuses_use_the_same_retry_taxonomy():
    class StringStatus(Exception):
        def __init__(self, code):
            self.status_code = code
            super().__init__("opaque provider failure")

    assert errors.classify(StringStatus("429"))["kind"] == errors.ErrorKind.RATE_LIMIT
    assert errors.classify(StringStatus("503"))["retryable"] is True
    auth = errors.classify(StringStatus("401"))
    assert auth["kind"] == errors.ErrorKind.AUTH and auth["retryable"] is False


@check
def cancellation_wakes_retry_backoff_without_starting_another_attempt():
    cancel = threading.Event()
    calls = {"n": 0}

    def timeout():
        calls["n"] += 1
        raise TimeoutError("provider timed out")

    started = time.monotonic()
    try:
        with_retry(
            timeout, max_attempts=3, should_cancel=cancel.is_set,
            dispatch=lambda _event: cancel.set(),
        )
        assert False, "cancellation must interrupt the pending retry"
    except RetryCancelledError:
        pass
    assert calls["n"] == 1
    assert time.monotonic() - started < 0.25, "Event.wait should wake instead of sleeping through backoff"


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
