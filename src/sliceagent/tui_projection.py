"""Pure projections from execution events into bounded terminal-facing data.

The terminal renderer should not reverse-engineer lifecycle truth from prose.  This
module deliberately has no Rich or prompt-toolkit dependency: it extracts typed tool
status, child-artifact identity, and small output previews that any frontend can render.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from .execution import coerce_tool_status


DELEGATION_TOOLS = frozenset({"spawn_agent", "spawn_explore", "spawn_subagent"})
EDIT_TOOLS = frozenset({"edit_file", "append_to_file", "str_replace"})
QUIET_OUTPUT_TOOLS = frozenset({"read_file", "list_files", *EDIT_TOOLS})
EVIDENCE_STATUSES = frozenset({
    "not_assessed", "none", "navigation_only", "content_partial", "content_retained",
})
REPORT_COMPLETIONS = frozenset({"complete", "partial", "absent", "unknown"})
EVIDENCE_COUNT_FIELDS = (
    "scope_path_count", "navigation_success_count", "content_success_count",
    "gap_observation_count", "retained_navigation_view_count",
    "retained_content_view_count", "omitted_navigation_view_count",
    "omitted_content_view_count", "truncated_content_view_count",
)
_OSC_ESCAPE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_BIDI_CONTROLS = frozenset({
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
})


def safe_terminal_text(value: object, *, multiline: bool = True) -> str:
    """Neutralize terminal control sequences while preserving ordinary Unicode text."""
    text = str(value or "")
    text = _OSC_ESCAPE.sub("", text)
    text = _ANSI_ESCAPE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    allowed = {"\n", "\t"} if multiline else {"\t"}
    return "".join(
        char if (
            char in allowed
            or (32 <= ord(char) < 127)
            or (ord(char) >= 160 and char not in _BIDI_CONTROLS)
        ) else "�"
        for char in text
    )


def normalized_tool_status(event: object) -> str:
    """Return one of the typed execution statuses, with a legacy fallback."""
    outcome = getattr(event, "outcome", None)
    status = getattr(outcome, "status", None)
    status = getattr(status, "value", status)
    if status is None or status == "":
        status = getattr(event, "status", None)
    if status is not None and status != "":
        return coerce_tool_status(status).value
    return "failed" if bool(getattr(event, "failing", False)) else "succeeded"


def normalized_evidence_status(value: object) -> str:
    """Normalize only the typed explorer-evidence vocabulary; malformed input stays unassessed."""
    status = str(value or "not_assessed").strip().casefold().replace("-", "_").replace(" ", "_")
    return status if status in EVIDENCE_STATUSES else "not_assessed"


def normalized_report_completion(value: object) -> str:
    """Normalize report-byte completeness independently of child execution state."""
    completion = str(value or "unknown").strip().casefold().replace("-", "_").replace(" ", "_")
    return completion if completion in REPORT_COMPLETIONS else "unknown"


def child_incompleteness_label(report_completion: object, operational_partial: bool) -> str:
    """Keep report truncation distinct from an operationally incomplete child run."""
    if normalized_report_completion(report_completion) == "partial":
        return "partial report"
    return "work incomplete" if operational_partial else ""


def evidence_account_counts(value: object) -> tuple[tuple[str, int], ...]:
    """Keep the optional exact evidence counters bounded and immutable for presentation."""
    if not isinstance(value, Mapping):
        return ()
    counts = []
    for key in EVIDENCE_COUNT_FIELDS:
        raw = value.get(key)
        if not isinstance(raw, int) or isinstance(raw, bool):
            continue
        counts.append((key, max(0, min(raw, 1_000_000))))
    return tuple(counts)


def invocation_id(event: object) -> str:
    """Stable call identity shared by start/result timing and concurrent rendering."""
    value = str(getattr(event, "invocation_id", "") or "")
    if value:
        return value
    invocation = getattr(event, "invocation", None)
    if invocation is None:
        invocation = getattr(getattr(event, "outcome", None), "invocation", None)
    return str(getattr(invocation, "id", "") or "")


@dataclass(frozen=True)
class OutputPreview:
    lines: tuple[str, ...] = ()
    hidden_lines: int = 0
    tail_retained: bool = False


@dataclass(frozen=True)
class ToolResultView:
    """Small typed lifecycle projection shared by every terminal adapter."""
    invocation_id: str
    name: str
    status: str

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    @property
    def is_delegation(self) -> bool:
        return self.name in DELEGATION_TOOLS


def project_tool_result(event: object) -> ToolResultView:
    return ToolResultView(
        invocation_id=invocation_id(event),
        name=str(getattr(event, "name", "") or ""),
        status=normalized_tool_status(event),
    )


def _next_line(raw: str, start: int) -> tuple[str, int]:
    """Return one physical line and the next offset using C-level string searches."""
    end = raw.find("\n", start)
    if end < 0:
        return raw[start:], len(raw)
    return raw[start:end], end + 1


def _last_nonempty_line(raw: str, max_chars: int) -> str:
    end = len(raw)
    while end:
        while end and raw[end - 1] in "\r\n":
            end -= 1
        if not end:
            return ""
        newline = raw.rfind("\n", 0, end)
        start = newline + 1
        line = _preview_line(raw[start:end], max_chars)
        if line:
            return line
        end = newline
    return ""


def _preview_line(value: str, max_chars: int) -> str:
    line = " ".join(safe_terminal_text(value, multiline=False).expandtabs(4).split())
    return line if len(line) <= max_chars else line[: max_chars - 1] + "…"


def output_preview(value: object, *, max_rows: int = 3, max_chars: int = 320) -> OutputPreview:
    """Keep useful line shape while bounding terminal noise.

    When output is longer than the row budget, retain the final non-empty line as well
    as the beginning.  That preserves both an error's headline and its concluding cause.
    """
    raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    max_rows = max(1, int(max_rows))
    head: list[str] = []
    offset = 0
    while offset < len(raw) and len(head) < max_rows:
        raw_line, next_offset = _next_line(raw, offset)
        line = _preview_line(raw_line, max_chars)
        if line:
            head.append(line)
        if next_offset <= offset:
            break
        offset = next_offset
    physical_lines = raw.count("\n") + (1 if raw and raw[-1] != "\n" else 0)
    clipped = tuple(head)
    if physical_lines <= max_rows or offset >= len(raw):
        return OutputPreview(clipped, 0, False)
    if max_rows == 1:
        shown = clipped[:1]
        tail_retained = False
    else:
        tail = _last_nonempty_line(raw, max_chars)
        shown = (*clipped[: max_rows - 1], tail)
        tail_retained = True
    shown = tuple(line for line in shown if line)
    return OutputPreview(shown, max(0, physical_lines - len(shown)), tail_retained)


def _child_payload(event: object) -> Mapping[str, object]:
    merged: dict[str, object] = {}
    for effect in (getattr(getattr(event, "outcome", None), "effects", ()) or ()):
        if str(getattr(effect, "kind", "") or "") not in {"child_outcome", "child_artifact"}:
            continue
        payload = getattr(effect, "payload", None)
        if isinstance(payload, Mapping):
            merged.update(payload)
    return merged


@dataclass(frozen=True)
class AgentResultView:
    invocation_id: str
    launch_ordinal: int
    kind: str
    name: str
    task: str
    status: str
    stop_cause: str
    recovered_from: tuple[str, ...]
    artifact_id: str
    detail: str
    source_coverage_status: str = "not_assessed"
    evidence_status: str = "not_assessed"
    evidence_account: tuple[tuple[str, int], ...] = ()
    duration_s: float | None = None
    request_ordinal: int = 0
    stop_reason: str = ""
    terminal_reason: str = ""
    partial: bool = False
    report_completion: str = "unknown"

    @property
    def sealed(self) -> bool:
        """Whether optional canonical artifact persistence actually committed."""
        return bool(self.artifact_id)

    @property
    def report_ready(self) -> bool:
        """Whether successful computation actually returned report bytes to the parent."""
        return self.status == "succeeded" and self.report_completion in {"complete", "partial"}

    @property
    def timed_out(self) -> bool:
        """Timeout is a typed child-outcome fact, never an inference from rendered prose."""
        causes = {self.stop_cause.strip().casefold(), self.stop_reason.strip().casefold()}
        return any(cause.endswith("_timeout") for cause in causes if cause)


def project_agent_result(event: object, *, duration_s: float | None = None) -> AgentResultView:
    """Project a spawn result from typed effects; prose is only a bounded detail fallback."""
    args = getattr(event, "args", None)
    args = args if isinstance(args, Mapping) else {}
    payload = _child_payload(event)
    raw_recovered = payload.get("recovered_from") or ()
    recovered = tuple(str(item) for item in raw_recovered if item) \
        if isinstance(raw_recovered, (list, tuple)) else ()
    try:
        ordinal = max(0, int(payload.get("launch_ordinal") or 0))
    except (TypeError, ValueError):
        ordinal = 0
    source_coverage = str(
        payload.get("source_coverage_status") or payload.get("epistemic_status") or "not_assessed"
    ).strip().casefold()
    source_coverage = {
        "grounded": "source_complete", "partial": "source_partial", "unsupported": "source_unsupported",
    }.get(source_coverage, source_coverage)
    if source_coverage not in {
        "source_complete", "source_partial", "source_unsupported", "not_assessed",
    }:
        source_coverage = "not_assessed"
    tool_name = str(getattr(event, "name", "") or "")
    kind = str(payload.get("kind") or args.get("agent") or (
        "explorer" if tool_name == "spawn_explore" else "general"
    ))
    preview = output_preview(getattr(event, "output", ""), max_rows=1, max_chars=240)
    invocation = getattr(getattr(event, "outcome", None), "invocation", None)
    try:
        request_ordinal = max(0, int(getattr(invocation, "provider_index", -1)) + 1)
    except (TypeError, ValueError):
        request_ordinal = 0
    status = normalized_tool_status(event)
    stop_cause = safe_terminal_text(payload.get("stop_cause") or "", multiline=False)
    stop_reason = safe_terminal_text(payload.get("stop_reason") or "", multiline=False)
    explicit_partial = payload.get("partial")
    partial = bool(explicit_partial) if isinstance(explicit_partial, bool) else False
    # Explorer-prefixed fields are the canonical child-artifact wire. Generic names remain a typed
    # compatibility alias for older/third-party producers; neither path consults report prose.
    evidence_status_value = (
        payload.get("explorer_evidence_status")
        if "explorer_evidence_status" in payload else payload.get("evidence_status")
    )
    evidence_account_value = (
        payload.get("explorer_evidence")
        if "explorer_evidence" in payload else payload.get("evidence_account")
    )
    return AgentResultView(
        invocation_id=invocation_id(event),
        launch_ordinal=ordinal,
        kind=safe_terminal_text(kind, multiline=False),
        name=safe_terminal_text(payload.get("name") or args.get("name") or "", multiline=False),
        task=" ".join(safe_terminal_text(args.get("task") or "", multiline=True).split()),
        status=status,
        stop_cause=stop_cause,
        recovered_from=tuple(safe_terminal_text(item, multiline=False) for item in recovered),
        artifact_id=safe_terminal_text(payload.get("artifact_id") or "", multiline=False),
        detail=preview.lines[0] if preview.lines else "",
        source_coverage_status=source_coverage,
        evidence_status=normalized_evidence_status(evidence_status_value),
        evidence_account=evidence_account_counts(evidence_account_value),
        duration_s=duration_s,
        request_ordinal=request_ordinal,
        stop_reason=stop_reason,
        terminal_reason=stop_cause or stop_reason,
        partial=partial,
        report_completion=normalized_report_completion(payload.get("report_completion")),
    )


__all__ = [
    "DELEGATION_TOOLS", "EDIT_TOOLS", "EVIDENCE_COUNT_FIELDS", "EVIDENCE_STATUSES",
    "QUIET_OUTPUT_TOOLS", "REPORT_COMPLETIONS", "AgentResultView", "OutputPreview", "ToolResultView",
    "child_incompleteness_label", "evidence_account_counts", "invocation_id", "normalized_evidence_status",
    "normalized_report_completion",
    "normalized_tool_status", "output_preview", "project_agent_result", "project_tool_result",
    "safe_terminal_text",
]
