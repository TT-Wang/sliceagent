"""Oracle implementations — ground-truth verification, independent of retrieval accuracy.

The loop can gate "done" on this so a retrieval miss can't masquerade as completion.
"""
from __future__ import annotations

import subprocess


class CommandOracle:
    """Runs a verification command (e.g. the project's test suite). Pass/fail by exit code."""

    def __init__(self, cmd: str, timeout: int = 120):
        self.cmd = cmd
        self.timeout = timeout

    def verify(self) -> tuple[bool, str]:
        try:
            r = subprocess.run(self.cmd, shell=True, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            # A timed-out verification is a FAILURE, not a thrown exception — otherwise it propagates out
            # of the oracle and silently BYPASSES the done-gate (a hung test would mark the task complete).
            # On timeout, .stdout/.stderr may each be bytes OR str OR None (version/stream dependent) —
            # decode EACH before concat, else a bytes+str mix (e.g. stdout bytes, stderr None→"") raises
            # TypeError and the crash bypasses the done-gate. Coerce per-operand.
            def _s(x):
                return x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
            out = _s(e.stdout) + _s(e.stderr)
            return False, (out + f"\n[verification timed out after {self.timeout}s]").strip()
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return r.returncode == 0, out


class NullOracle:
    def verify(self) -> tuple[bool, str]:
        return True, ""
