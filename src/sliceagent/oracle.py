"""Oracle implementations — ground-truth verification, independent of retrieval accuracy.

The loop can gate "done" on this so a retrieval miss can't masquerade as completion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .execution import ToolStatus
from .sandbox import SANDBOX_TIMEOUT, LocalSandbox


@dataclass(frozen=True)
class OracleResult:
    """Typed completion-gate result with tuple-unpacking compatibility."""

    status: ToolStatus
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.SUCCEEDED

    def __iter__(self):
        yield self.ok
        yield self.output


class CommandOracle:
    """Runs a verification command (e.g. the project's test suite). Pass/fail by exit code."""

    def __init__(self, cmd: str, timeout: int = 120):
        self.cmd = cmd
        self.timeout = timeout

    def verify(self) -> OracleResult:
        # Verification inherits the caller environment for compatibility, but shares the same owned
        # process-group lifecycle as command tools. A timeout is still conservatively indeterminate:
        # ordinary descendants are reaped, yet a deliberately detached process cannot be disproved.
        code, output = LocalSandbox(scrub_secrets=False).run(
            self.cmd, cwd=os.getcwd(), timeout=self.timeout,
        )
        output = output.strip()
        if code == SANDBOX_TIMEOUT:
            return OracleResult(ToolStatus.INDETERMINATE, output)
        return OracleResult(ToolStatus.SUCCEEDED if code == 0 else ToolStatus.FAILED, output)


class NullOracle:
    def verify(self) -> OracleResult:
        return OracleResult(ToolStatus.SUCCEEDED)
