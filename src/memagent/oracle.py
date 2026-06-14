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
        r = subprocess.run(self.cmd, shell=True, capture_output=True, text=True, timeout=self.timeout)
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return r.returncode == 0, out


class NullOracle:
    def verify(self) -> tuple[bool, str]:
        return True, ""
