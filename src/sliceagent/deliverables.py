"""Narrow output reminders for self-contained responses and typed procedure deliverables.

Execution truth and response delivery are different facts.  This module detects only an
obvious *non-delivery* (empty prose, a pointer to private output, or a deferred-work update).
Those shapes are invalid for every terminal response because tool/child/update text is private;
procedure contracts add scope without making the basic visibility invariant optional.
It deliberately does not grade report structure, evidence quality, headings, or completeness.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import re


DELIVERABLE_VERSION = 1
CODE_REVIEW_REPORT = "code_review_report"


@dataclass(frozen=True)
class DeliverableRequirement:
    """One procedure-bound deliverable scoped to one exact logical request."""

    kind: str
    logical_id: str
    source: str
    version: int = DELIVERABLE_VERSION

    def __post_init__(self) -> None:
        if self.kind not in {CODE_REVIEW_REPORT}:
            raise ValueError(f"unsupported deliverable kind: {self.kind!r}")
        if not isinstance(self.logical_id, str) or not self.logical_id.strip():
            raise ValueError("deliverable logical_id must be non-empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("deliverable source must be non-empty")
        if self.version != DELIVERABLE_VERSION:
            raise ValueError(f"unsupported deliverable version: {self.version!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "v": self.version,
            "kind": self.kind,
            "logical_id": self.logical_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DeliverableRequirement | None":
        if not isinstance(value, Mapping) or not value:
            return None
        try:
            return cls(
                kind=str(value.get("kind") or ""),
                logical_id=str(value.get("logical_id") or ""),
                source=str(value.get("source") or ""),
                version=int(value.get("v") or DELIVERABLE_VERSION),
            )
        except (TypeError, ValueError, OverflowError):
            return None


@dataclass(frozen=True)
class DeliverableAssessment:
    complete: bool
    shape: str
    reason: str = ""


def requirement_for_contract(
    contract: object,
    *,
    logical_id: str,
    source: str,
) -> DeliverableRequirement | None:
    """Compile declarative procedure metadata into one supported output contract.

    The kernel never branches on a skill's brand/name.  A skill opts in through its
    frontmatter and a successful ``skill_loaded`` effect carries that declaration to L1.
    """
    normalized = str(contract or "").strip().casefold().replace("-", "_").replace("/", "_")
    if normalized not in {CODE_REVIEW_REPORT, f"{CODE_REVIEW_REPORT}_v1"}:
        return None
    return DeliverableRequirement(
        kind=CODE_REVIEW_REPORT,
        logical_id=str(logical_id or "").strip(),
        source=str(source or "").strip(),
    )


_DEICTIC_PLACEHOLDER = re.compile(
    r"\b(?:findings?|report|results?)\s+(?:above|below)\b|"
    r"\bas\s+(?:shown|listed|reported)\s+above\b|"
    r"(?:上述|以上|下述)(?:发现|报告|结果)",
    re.IGNORECASE,
)
_DEFERRED_WORK_UPDATE = re.compile(
    r"\b(?:let\s+me|i\s+(?:still\s+)?need\s+to|i(?:'ll|\s+will)\s+now)\s+"
    r"(?:re-?read|read|search|inspect|check|confirm|verify|open|fetch|review)\b|"
    r"(?:让我|我还需要|我现在(?:会|将))(?:重新)?(?:读取|搜索|检查|确认|验证|打开|审查)",
    re.IGNORECASE,
)


def assess_deliverable(
    requirement: DeliverableRequirement | None,
    candidate: object,
) -> DeliverableAssessment:
    """Recognize obvious non-delivery without grading the response itself."""
    text = str(candidate or "").strip()
    if not text:
        return DeliverableAssessment(False, "missing", "the terminal response is empty")
    if _DEICTIC_PLACEHOLDER.search(text) is not None:
        return DeliverableAssessment(
            False,
            "placeholder",
            "the response points to private findings 'above' instead of containing the report",
        )
    if _DEFERRED_WORK_UPDATE.search(text) is not None:
        return DeliverableAssessment(
            False,
            "deferred_update",
            "the terminal response is a progress update promising more inspection, not an answer",
        )
    if requirement is None:
        return DeliverableAssessment(True, "visible_response")
    if requirement.kind != CODE_REVIEW_REPORT:
        return DeliverableAssessment(True, "unsupported_kind")
    return DeliverableAssessment(True, "visible_response")


__all__ = [
    "CODE_REVIEW_REPORT", "DeliverableAssessment", "DeliverableRequirement",
    "assess_deliverable", "requirement_for_contract",
]
