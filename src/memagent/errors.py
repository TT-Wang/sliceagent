"""Error classification + retry (lifted in spirit from Hermes `error_classifier` + Kimi `chatWithRetry`).

`classify` maps an exception to structured recovery hints; `with_retry` does
abort-aware jittered backoff on retryable errors.

The backoff is jittered (ported from Hermes `agent/retry_utils.py:19`): a
lock-guarded monotonic counter seeds a per-call RNG so concurrent sessions
hitting the same rate-limited provider don't all retry at the same instant
(decorrelated uniform jitter).
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable

from .context_overflow import is_context_overflow
from .events import ApiRetry, Dispatcher

# Monotonic counter for jitter-seed uniqueness within the same process.
# Lock-guarded to avoid races in concurrent retry paths (Hermes retry_utils.py:12).
_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute a jittered exponential backoff delay (Hermes retry_utils.py:19).

    Args:
        attempt: 1-based retry attempt number.
        base_delay: Base delay in seconds for attempt 1.
        max_delay: Maximum delay cap in seconds.
        jitter_ratio: Fraction of the computed delay to use as random jitter
            range. 0.5 means jitter is uniform in [0, 0.5 * delay].

    Returns:
        Delay in seconds: min(base * 2^(attempt-1), max_delay) + jitter.

    The jitter decorrelates concurrent retries so multiple sessions hitting the
    same provider don't all retry at the same instant.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)

    # Seed from time + counter for decorrelation even with coarse clocks.
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)

    return delay + jitter


def classify(error: Exception) -> dict:
    msg = str(error).lower()
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    overflow = is_context_overflow(error)
    retryable = False
    if status == 429 or "rate limit" in msg or "overloaded" in msg or "503" in msg:
        retryable = True
    if isinstance(status, int) and 500 <= status < 600:
        retryable = True
    if "timeout" in msg or "timed out" in msg or "connection error" in msg or "econn" in msg:
        retryable = True
    if status in (401, 403):
        retryable = False  # auth — never retry
    if overflow:
        retryable = False  # tighten the slice, don't blindly re-send the oversized request
    return {"retryable": retryable, "is_context_overflow": overflow, "status": status}


def with_retry(
    fn: Callable[[], object],
    *,
    max_attempts: int = 3,
    is_retryable: Callable[[Exception], bool] | None = None,
    dispatch: Dispatcher | None = None,
):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            retry = is_retryable(e) if is_retryable else classify(e)["retryable"]
            if not retry or attempt == max_attempts:
                raise
            if dispatch:
                dispatch(ApiRetry(attempt=attempt, error=str(e)[:200]))
            time.sleep(jittered_backoff(attempt))
