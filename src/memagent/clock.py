"""Clock abstraction (borrowed from Kimi's clock seam).

Wraps wall-clock time behind a tiny interface so timed logic (cron) is DETERMINISTICALLY testable:
SystemClock in production, FakeClock in tests. Anything that needs "now" takes a clock instead of
calling time.time() directly.
"""
from __future__ import annotations

import time


class SystemClock:
    """Real wall-clock time (epoch seconds)."""

    def now(self) -> float:
        return time.time()


class FakeClock:
    """Controllable clock for tests — `now()` returns a fixed value you `advance()` explicitly."""

    def __init__(self, t: float = 0.0):
        self._t = float(t)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += float(seconds)


SYSTEM_CLOCK = SystemClock()
