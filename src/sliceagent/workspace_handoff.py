"""Typed decisions for staging an in-process workspace handoff.

The scheduler owns navigation truth.  A string-only callback made the tool host
collapse three different outcomes (scheduled, recoverably refused, and failed)
into one generic error.  Keep this small value object independent of the CLI so
the model tool and slash-command surfaces project the same decision.
"""
from __future__ import annotations

from dataclasses import dataclass

from .execution import ToolStatus


@dataclass(frozen=True)
class WorkspaceScheduleDecision:
    """Conclusive result of asking the live host to stage one workspace switch."""

    status: ToolStatus
    message: str = ""

    @classmethod
    def scheduled(cls) -> "WorkspaceScheduleDecision":
        return cls(ToolStatus.SUCCEEDED)

    @classmethod
    def steered(cls, message: str) -> "WorkspaceScheduleDecision":
        return cls(ToolStatus.STEERED, str(message))

    @classmethod
    def failed(cls, message: str) -> "WorkspaceScheduleDecision":
        return cls(ToolStatus.FAILED, str(message))

    @property
    def accepted(self) -> bool:
        return self.status is ToolStatus.SUCCEEDED


__all__ = ["WorkspaceScheduleDecision"]
