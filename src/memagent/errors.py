"""Error classification + retry.

`classify` maps an exception to structured recovery hints; `with_retry` does
abort-aware jittered backoff on retryable errors.

The backoff is jittered: a
lock-guarded monotonic counter seeds a per-call RNG so concurrent sessions
hitting the same rate-limited provider don't all retry at the same instant
(decorrelated uniform jitter).
"""
from __future__ import annotations

import random
import threading
import time
from enum import Enum
from typing import Callable

from .context_overflow import is_context_overflow
from .events import ApiRetry, Dispatcher


class ErrorKind(str, Enum):
    """Typed failure taxonomy. str-based so it stays
    backward-compatible: `classify(e)["kind"] == "rate_limit"` still works, and it's JSON-serializable for
    telemetry. ALL members let the metrics layer pre-register a counter per kind."""
    CONTEXT_OVERFLOW = "context_overflow"
    AUTH = "auth"
    EMPTY_RESPONSE = "empty_response"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


class EmptyResponseError(Exception):
    """The provider returned a degenerate completion — no content AND no tool calls. Some
    providers/proxies occasionally emit an empty body; returning it stalls the loop. Classified
    RETRYABLE so `with_retry` re-rolls instead."""


# Monotonic counter for jitter-seed uniqueness within the same process.
# Lock-guarded to avoid races in concurrent retry paths.
_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute a jittered exponential backoff delay.

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
    empty = isinstance(error, EmptyResponseError)
    retryable = False
    if status == 429 or "rate limit" in msg or "too many requests" in msg or "overloaded" in msg or "503" in msg:
        retryable = True
    if isinstance(status, int) and 500 <= status < 600:
        retryable = True
    if "timeout" in msg or "timed out" in msg or "connection error" in msg or "econn" in msg:
        retryable = True
    if empty:
        retryable = True  # degenerate empty completion — re-roll
    if status in (401, 403):
        retryable = False  # auth — never retry
    if overflow:
        retryable = False  # tighten the slice, don't blindly re-send the oversized request
    # Bucket the failure for telemetry (orthogonal to `retryable`; lets the metrics layer count
    # rate-limit vs timeout vs overflow vs empty).
    if overflow:
        kind = ErrorKind.CONTEXT_OVERFLOW
    elif status in (401, 403):
        kind = ErrorKind.AUTH
    elif empty:
        kind = ErrorKind.EMPTY_RESPONSE
    elif status == 429 or "rate limit" in msg or "too many requests" in msg or "overloaded" in msg:
        kind = ErrorKind.RATE_LIMIT
    elif (isinstance(status, int) and 500 <= status < 600) or "503" in msg:
        kind = ErrorKind.SERVER
    elif "timeout" in msg or "timed out" in msg:
        kind = ErrorKind.TIMEOUT
    elif "connection" in msg or "econn" in msg:
        kind = ErrorKind.CONNECTION
    else:
        kind = ErrorKind.UNKNOWN
    return {"retryable": retryable, "is_context_overflow": overflow, "status": status, "kind": kind}


def _retry_after_seconds(error: Exception) -> "float | None":
    """Best-effort: a 429/503 may carry a Retry-After (SDK `.retry_after` or a response header) telling us
    EXACTLY how long to wait — honor it instead of guessing. Returns seconds, or None to fall back to
    backoff (incl. when Retry-After is an HTTP-date, which we don't parse). Never raises."""
    try:
        val = getattr(error, "retry_after", None)
        if val is None:
            hdrs = getattr(getattr(error, "response", None), "headers", None)
            if hdrs is not None:
                val = hdrs.get("retry-after") or hdrs.get("Retry-After")
        if val is None:
            return None
        secs = float(val)
        return secs if secs >= 0 else None
    except (TypeError, ValueError):
        return None


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
            delay = jittered_backoff(attempt)
            ra = _retry_after_seconds(e)
            if ra is not None:
                delay = max(delay, min(ra, 60.0))   # honor server Retry-After, capped so a huge value can't stall the turn
            time.sleep(delay)
