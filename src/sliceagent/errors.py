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

from .context_overflow import _extract_status_code, is_context_overflow
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


class RetryCancelledError(Exception):
    """The owning turn was cancelled before a provider retry could start."""


class ImmediateRetryError(Exception):
    """Adapter state changed and the next physical request should be retried visibly, without backoff.

    Compatibility negotiation must return to :func:`with_retry` instead of issuing a hidden replacement
    request inside one adapter call.  This marker is intentionally narrow: transient provider failures keep
    their ordinary jittered backoff.
    """


class IndeterminateModelCallError(Exception):
    """A watchdog returned while the physical provider request may still be in flight.

    Retrying would overlap an abandoned socket and make request count, latency, and spend untruthful.  This
    error is therefore non-retryable; callers may seal the failure but must not launch a recovery model call.
    """


class ProviderCapacityError(Exception):
    """No physical provider slot became available before this call's admission deadline.

    This logical call provably never opened a request, while existing indeterminate calls may still own every
    provider slot. Blindly retrying here only waits the same whole deadline again and cannot create capacity;
    a later user turn may retry after those physical calls close.
    """


class TransportStartupError(Exception):
    """The local streaming transport did not become ready before this call's deadline.

    No provider request was opened, but replaying immediately against the same stuck loop-generation only
    repeats the wait.  A later turn may use a generation that eventually became ready or was restarted.
    """


class PreFirstByteTimeoutError(Exception):
    """A streaming provider call closed cleanly before yielding its first SSE item.

    This is safe to retry because no model output was observed and physical closure is confirmed, but it is
    expensive in a different way from a quick connection failure: a reasoning provider may have spent the
    whole interval computing before a router/SDK timed out.  ``with_retry`` therefore gives this narrow class
    a smaller replay ceiling while leaving rate-limit, server, and connection retry policy unchanged.
    """


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
    status = _extract_status_code(error)
    overflow = is_context_overflow(error)
    empty = isinstance(error, EmptyResponseError)
    immediate = isinstance(error, ImmediateRetryError)
    indeterminate = isinstance(error, IndeterminateModelCallError)
    capacity = isinstance(error, ProviderCapacityError)
    startup = isinstance(error, TransportStartupError)
    pre_first_byte = isinstance(error, PreFirstByteTimeoutError)
    retryable = False
    if status == 429 or "rate limit" in msg or "too many requests" in msg or "overloaded" in msg or "503" in msg:
        retryable = True
    if isinstance(status, int) and 500 <= status < 600:
        retryable = True
    if pre_first_byte or "timeout" in msg or "timed out" in msg or "connection error" in msg or "econn" in msg:
        retryable = True
    if empty:
        retryable = True  # degenerate empty completion — re-roll
    if immediate:
        retryable = True
    if indeterminate or capacity or startup:
        retryable = False
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
    elif indeterminate or capacity or startup or pre_first_byte or "timeout" in msg or "timed out" in msg:
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
    should_cancel: Callable[[], bool] | None = None,
    max_pre_first_byte_attempts: int = 2,
):
    def cancelled() -> bool:
        if should_cancel is None:
            return False
        try:
            return bool(should_cancel())
        except Exception:
            return False

    def wait_or_cancel(delay: float) -> None:
        if should_cancel is None:
            time.sleep(delay)
            return
        # threading.Event.is_set is the production callback. Its bound owner gives us a wakeable wait rather
        # than polling through the retry delay; generic callbacks retain a short bounded poll fallback.
        owner = getattr(should_cancel, "__self__", None)
        waiter = getattr(owner, "wait", None)
        if callable(waiter):
            if waiter(delay):
                raise RetryCancelledError("model retry cancelled by the owning turn")
            return
        deadline = time.monotonic() + delay
        while True:
            if cancelled():
                raise RetryCancelledError("model retry cancelled by the owning turn")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.05, remaining))

    # A provider/router can remain silent while doing expensive hidden reasoning, then close on its first-byte
    # timeout. Replaying that shape three times is the historical child retry storm. Count only this explicit
    # adapter marker; ordinary quick 429/5xx/connection recovery retains the caller's full attempt budget.
    pre_first_byte_failures = 0
    try:
        pre_first_byte_ceiling = max(1, min(int(max_attempts), int(max_pre_first_byte_attempts)))
    except (TypeError, ValueError, OverflowError):
        pre_first_byte_ceiling = min(max_attempts, 2)

    for attempt in range(1, max_attempts + 1):
        if cancelled():
            raise RetryCancelledError("model call cancelled by the owning turn")
        try:
            return fn()
        except Exception as e:
            # These markers describe lifecycle truth, not provider taxonomy.  An adapter-supplied
            # classifier must never turn either one back into a retryable transport error: an
            # indeterminate call may still own a live socket, while cancellation has already retired
            # the owning turn.
            if isinstance(e, (
                IndeterminateModelCallError, ProviderCapacityError, RetryCancelledError,
                TransportStartupError,
            )):
                raise
            if cancelled():
                raise RetryCancelledError("model retry cancelled by the owning turn") from e
            retry = is_retryable(e) if is_retryable else classify(e)["retryable"]
            event_max_attempts = max_attempts
            if isinstance(e, PreFirstByteTimeoutError):
                pre_first_byte_failures += 1
                remaining = pre_first_byte_ceiling - pre_first_byte_failures
                event_max_attempts = min(max_attempts, attempt + max(0, remaining))
                if remaining <= 0:
                    raise
            if not retry or attempt == max_attempts:
                raise
            delay = 0.0 if isinstance(e, ImmediateRetryError) else jittered_backoff(attempt)
            ra = _retry_after_seconds(e)
            if ra is not None:
                delay = max(delay, min(ra, 60.0))   # honor server Retry-After, capped so a huge value can't stall the turn
            if dispatch:
                dispatch(ApiRetry(
                    attempt=attempt, error=str(e)[:200], delay_s=delay, max_attempts=event_max_attempts,
                ))
            wait_or_cancel(delay)
