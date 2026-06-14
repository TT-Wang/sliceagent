"""Error classification + retry (lifted in spirit from Hermes `error_classifier` + Kimi `chatWithRetry`).

`classify` maps an exception to structured recovery hints; `with_retry` does
abort-aware exponential backoff on retryable errors.
"""
from __future__ import annotations

import time
from typing import Callable

from .events import ApiRetry, Dispatcher


def classify(error: Exception) -> dict:
    msg = str(error).lower()
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    retryable = False
    if status == 429 or "rate limit" in msg or "overloaded" in msg or "503" in msg:
        retryable = True
    if isinstance(status, int) and 500 <= status < 600:
        retryable = True
    if "timeout" in msg or "timed out" in msg or "connection error" in msg or "econn" in msg:
        retryable = True
    if status in (401, 403):
        retryable = False  # auth — never retry
    return {"retryable": retryable, "status": status}


def with_retry(
    fn: Callable[[], object],
    *,
    max_attempts: int = 3,
    is_retryable: Callable[[Exception], bool] | None = None,
    dispatch: Dispatcher | None = None,
):
    delay = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            retry = is_retryable(e) if is_retryable else classify(e)["retryable"]
            if not retry or attempt == max_attempts:
                raise
            if dispatch:
                dispatch(ApiRetry(attempt=attempt, error=str(e)[:200]))
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
