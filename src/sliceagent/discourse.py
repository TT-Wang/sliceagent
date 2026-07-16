"""Addressable discourse anchors for user-visible assistant output.

The archive keeps the full turn.  This module derives only small, source-linked
addresses for things users naturally refer to later ("number 2", "the first
subagent", "your original findings").  The anchors are a pageable index, not a
second copy of the conversation and never a source of factual truth about the
live workspace.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .intent import (EvidenceQuery, QualityEvidenceQuery,
                     derive_evidence_query)
from .persistence import artifact_order_key


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD_HEADING = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?[\s-]*$")
_NUMBERED = re.compile(r"^\s*\*{0,2}(\d{1,3})[.)]\*{0,2}\s+(.+?)\s*$")
_STABLE_ID = re.compile(r"\b(?:sub|agent|finding|bug)-\d+\b", re.IGNORECASE)
_PROPOSAL = re.compile(
    r"(?:^|(?<=[.!?]))\s*((?:would\s+you\s+like\s+me\s+to|"
    r"do\s+you\s+want\s+me\s+to|want\s+me\s+to|shall\s+i|should\s+i|i\s+can)"
    r"\b[^?.!\n]{1,300}\?)",
    re.IGNORECASE,
)
_CHOICE_QUESTION = re.compile(
    r"((?:which\b[^?\n]{0,100}\b(?:prefer|choose|pick)|"
    r"(?:please\s+)?(?:choose|pick)\s+(?:one|an?\s+option))[^?\n]*\?)",
    re.IGNORECASE,
)
_QUESTION_SENTENCE = re.compile(r"(?:^|(?<=[.!?]))\s*([^?\n]{1,300}\?)")
_PATH_TOKEN = re.compile(
    r"`((?:~[/\\]|/|[A-Za-z]:[/\\])[^`\r\n?]+)`|"
    r"((?:~[/\\]|/|[A-Za-z]:[/\\])[^\s?]+)"
)
_PATH_CONFIRMATION = re.compile(r"\b(?:is\s+(?:it|that)|confirm\b|correct\b|right\b)", re.IGNORECASE)
_WORKSPACE_CONTEXT = re.compile(
    r"\b(?:workspace|project|repo(?:sitory)?|directory|folder)\b", re.IGNORECASE,
)
# The assistant's own "which one should I navigate to — loom-app or loom-engine?" question names bare
# directory options but no absolute path (so the path-confirmation branch above cannot fire). A next-turn
# reply naming one option confers scoped navigation authority (resolved in intent._selected_nav_target_grant).
_NAV_DISAMBIGUATION = re.compile(
    r"\bwhich\b[^?\n]{0,120}\b(?:navigate|switch\s+to|go\s+(?:to|into)|move\s+(?:to|into)|"
    r"open|cd|chdir|work\s+in)\b", re.IGNORECASE)
_DIR_OPTION = re.compile(r"\b([A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)+)\b")
# A single-target navigation OFFER inside an assistant proposal ("do you want me to switch to loom-app?").
# The bare directory name (not an absolute path) means the path-confirmation branch cannot fire; a bare
# "yes" then continues this one navigation (intent.analyze_turn treats a single nav_target as acceptable).
_NAV_OFFER = re.compile(
    r"\b(?:navigate|switch|go|move|cd|chdir|change(?:\s+(?:the\s+)?workspace)?)\s+(?:to|into)\s+"
    r"(?:the\s+)?(?:workspace\s+|folder\s+|directory\s+|project\s+)?"
    r"([A-Za-z0-9~][A-Za-z0-9._/-]*)", re.IGNORECASE)
_OPTION_SELECTION = re.compile(
    r"(?:go\s+with|choose|pick|take|let'?s\s+do|option)\s+(?:option\s+)?(\d{1,3})\b"
    r"|^\s*(\d{1,3})\s*[.!]*\s*$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_REFERENCE_STOP = frozenset({
    "a", "again", "all", "and", "back", "bug", "bugs", "finding", "findings",
    "first", "from", "give", "i", "in", "is", "it", "just", "list", "listed",
    "me", "my", "no", "number", "of", "one", "original", "please", "report",
    "review", "said", "second", "subagent", "subagents", "summarize", "summary",
    "the", "that", "this", "to", "told", "was", "what", "which", "you", "your",
})
_FOCUS_REFERENCE = re.compile(
    r"\b(?:why\b[^\n]{0,80}\bmistake|mistake\b[^\n]{0,40}\bagain|"
    r"wrong\b[^\n]{0,40}\bagain|remind\s+me\b[^\n]{0,40}\bagain|"
    r"that\s+(?:one|item|finding|subagent)\b[^\n]{0,40}\bagain)\b",
    re.IGNORECASE,
)
_NAMED_SUBJECT = re.compile(
    r"\b(?:the\s+)?([A-Za-z][A-Za-z0-9_.-]{1,80})\s+"
    r"(project|workspace|repo(?:sitory)?|codebase)\b",
    re.IGNORECASE,
)
_SUBJECT_REPAIR = re.compile(
    r"\b(?:i\s+(?:mean|meant)|no\s*,\s*i\s+(?:mean|meant))\s+(?:the\s+)?"
    r"([A-Za-z][A-Za-z0-9_.-]{1,80})(?:\s+(project|workspace|repo(?:sitory)?|codebase))?\b",
    re.IGNORECASE,
)
_EXPLICIT_SELF_TARGET = re.compile(
    r"\b(?:yourself|sliceagent(?:\s+itself)?|the\s+agent\s+itself|your\s+own\s+"
    r"(?:mind|machinery|design|behavior|behaviour|reasoning|memory|intent))\b",
    re.IGNORECASE,
)
_GENERIC_SUBJECTS = frozenset({"a", "active", "current", "my", "our", "that", "the", "this", "your"})
_DELEGATION_TOOLS = frozenset({"spawn_agent", "spawn_explore", "spawn_subagent"})
_COMMAND_TOOLS = frozenset({
    "run_command", "execute_code",
    "proc_start", "proc_poll", "proc_tail", "proc_wait", "proc_kill",
    "terminal_open", "terminal_send", "terminal_read", "terminal_wait", "terminal_close",
    # Legacy receipt names remain readable without weakening current-name coverage.
    "proc_write", "proc_read", "terminal_start", "terminal_write", "terminal_kill",
})
_FILE_READ_TOOLS = frozenset({"read_file"})
_FILE_OBSERVE_TOOLS = frozenset({"read_file", "list_files", "grep", "glob", "code_review"})
_FILE_WRITE_TOOLS = frozenset({"edit_file", "append_to_file", "str_replace", "write_file"})
_FILE_TOOLS = _FILE_OBSERVE_TOOLS | _FILE_WRITE_TOOLS
_TOOLS_BY_EVIDENCE_FAMILY = {
    "delegation": _DELEGATION_TOOLS,
    "command": _COMMAND_TOOLS,
    "file": _FILE_TOOLS,
    "file_read": _FILE_READ_TOOLS,
    "file_write": _FILE_WRITE_TOOLS,
}


def _plain(text: str) -> str:
    value = re.sub(r"[`*_~]", "", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n:-")
    return value


def _proposal_scan(text: str) -> str:
    """Blank example/code spans while preserving offsets into the visible assistant response."""
    chars = list(str(text or ""))

    def blank(start: int, end: int) -> None:
        for index in range(max(0, start), min(len(chars), end)):
            if chars[index] not in "\r\n":
                chars[index] = " "

    source = str(text or "")
    offset = 0
    fence: tuple[str, int] | None = None
    for line in source.splitlines(keepends=True):
        marker = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        in_fence = fence is not None
        if marker is not None:
            token = marker.group(1)
            if fence is None:
                fence = (token[0], len(token))
            elif (token[0] == fence[0] and len(token) >= fence[1]
                  and not line[marker.end():].strip()):
                fence = None
            blank(offset, offset + len(line))
        elif in_fence:
            blank(offset, offset + len(line))
        elif re.match(r"^(?:\t| {4,}|\s*>)", line):
            blank(offset, offset + len(line))
        offset += len(line)
    # A whole question shown as quoted/inline-code data is not the assistant asking it. A quoted path inside
    # a real surrounding question remains visible because the question mark sits outside the quote.
    for pattern in (r'"[^"\r\n]*\?[^"\r\n]*"', r"'[^'\r\n]*\?[^'\r\n]*'", r"`[^`\r\n]*\?[^`\r\n]*`"):
        for match in re.finditer(pattern, source):
            blank(*match.span())
    return "".join(chars)


def extract_pending_proposal(text: str) -> dict | None:
    """Return one immediate assistant action offer, or ``None``.

    A bare user assent may inherit effect authority only from this explicit, source-linked continuity object.
    Merely mentioning that a fix exists is not a proposal.
    """
    source = str(text or "")
    scan = _proposal_scan(source)
    choices = list(_CHOICE_QUESTION.finditer(scan))
    anchors = extract_addressable_anchors(scan) if choices else ()
    option_anchors = []
    if choices and anchors:
        question = choices[-1]
        candidates = [anchor for anchor in anchors if anchor.source_range[0] < question.start(1)]
        if candidates:
            option_anchors = [candidates[-1]]
            expected = candidates[-1].ordinal - 1
            for anchor in reversed(candidates[:-1]):
                if anchor.collection != option_anchors[-1].collection or anchor.ordinal != expected:
                    break
                option_anchors.append(anchor)
                expected -= 1
            option_anchors.reverse()
    if choices and len(option_anchors) >= 2:
        question = choices[-1]
        start = option_anchors[0].source_range[0]
        return {
            "text": source[start:question.end(1)],
            "source_range": [start, question.end(1)],
            "options": [anchor.to_dict() for anchor in option_anchors],
        }
    # A clarification of the exact target for an already-requested workspace navigation is itself a typed
    # pending action. This is deliberately narrow: an arbitrary yes/no question (or even an arbitrary path
    # question) cannot confer effect authority. The surrounding assistant text must identify a workspace-like
    # frame, and the question must contain one concrete absolute/home path.
    for question in reversed(list(_QUESTION_SENTENCE.finditer(scan))):
        sentence = question.group(1)
        if _PATH_CONFIRMATION.search(sentence) is None:
            continue
        context = scan[max(0, question.start(1) - 400):question.end(1)]
        paths = list(_PATH_TOKEN.finditer(sentence))
        if not paths:
            paths = list(_PATH_TOKEN.finditer(context))
        if not paths:
            continue
        if _WORKSPACE_CONTEXT.search(context) is None:
            continue
        path_match = paths[-1]
        path = (path_match.group(1) or path_match.group(2) or "").rstrip(".,;:!)]}\"'*_~")
        if not path:
            continue
        start, end = question.span(1)
        return {
            "text": source[start:end],
            "source_range": [start, end],
            "action": {"tool": "change_workspace", "args": {"path": path}},
        }
    # A workspace-navigation disambiguation question offering named directory options. No absolute path is
    # present, so this is the naming analogue of the path-confirmation branch above: the reply naming an
    # option is a typed navigate selection, not an arbitrary yes/no continuation.
    for question in reversed(list(_QUESTION_SENTENCE.finditer(scan))):
        sentence = question.group(1)
        if _NAV_DISAMBIGUATION.search(sentence) is None:
            continue
        # A strong nav verb (navigate/switch to/cd) + >=2 directory-like options is signal enough; no
        # extra workspace-context word is required (it wrongly rejected the plural "directories").
        context = scan[max(0, question.start(1) - 400):question.end(1)]
        names: list[str] = []
        for option in _DIR_OPTION.finditer(context):
            name = option.group(1)
            if name.casefold() not in {existing.casefold() for existing in names}:
                names.append(name)
        if len(names) < 2:
            continue
        start, end = question.span(1)
        return {
            "text": source[start:end],
            "source_range": [start, end],
            "nav_targets": names,
        }
    matches = list(_PROPOSAL.finditer(scan))
    if not matches:
        return None
    match = matches[-1]
    start, end = match.span(1)
    proposal_text = source[start:end]
    # A single-target navigation offer ("do you want me to switch to loom-app?") is a typed navigate
    # selection: a bare "yes" then authorizes navigating to that one named target.
    nav = _NAV_OFFER.search(proposal_text)
    if nav is not None:
        name = nav.group(1).rstrip("?.,;:!)/'\"")
        if name:
            return {"text": proposal_text, "source_range": [start, end], "nav_targets": [name]}
    return {
        "text": proposal_text,
        "source_range": [start, end],
    }


def _selected_pending_option(request: str, proposal: Mapping | None) -> dict | None:
    if not isinstance(proposal, Mapping):
        return None
    options = proposal.get("options")
    if not isinstance(options, (list, tuple)):
        return None
    match = _OPTION_SELECTION.search(str(request or ""))
    if match is None:
        return None
    ordinal = int(next(part for part in match.groups() if part))
    for option in options:
        if not isinstance(option, Mapping):
            continue
        try:
            if int(option.get("ordinal") or 0) == ordinal:
                return dict(option)
        except (TypeError, ValueError):
            continue
    return None


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(_plain(text).casefold()))


@dataclass(frozen=True)
class DiscourseAnchor:
    """One addressable item in a sealed, user-visible assistant response."""

    collection: str
    ordinal: int
    label: str
    excerpt: str
    source_range: tuple[int, int]
    stable_id: str = ""
    artifact_id: str = ""
    task_id: str = ""
    sequence: int = 0

    def to_dict(self) -> dict:
        return {
            "collection": self.collection,
            "ordinal": self.ordinal,
            "label": self.label,
            "excerpt": self.excerpt,
            "source_range": list(self.source_range),
            "stable_id": self.stable_id,
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "DiscourseAnchor | None":
        if not isinstance(value, Mapping):
            return None
        try:
            ordinal = int(value.get("ordinal") or 0)
        except (TypeError, ValueError):
            return None
        raw_range = value.get("source_range")
        if not (isinstance(raw_range, (list, tuple)) and len(raw_range) == 2
                and all(isinstance(item, int) for item in raw_range)):
            return None
        label = str(value.get("label") or "").strip()
        excerpt = str(value.get("excerpt") or "").strip()
        if ordinal <= 0 or not (label or excerpt):
            return None
        return cls(
            collection=str(value.get("collection") or "numbered list"),
            ordinal=ordinal,
            label=label or _plain(excerpt.splitlines()[0]),
            excerpt=excerpt or label,
            source_range=(raw_range[0], raw_range[1]),
            stable_id=str(value.get("stable_id") or ""),
            artifact_id=str(value.get("artifact_id") or ""),
            task_id=str(value.get("task_id") or ""),
            sequence=int(value.get("sequence") or 0),
        )


@dataclass(frozen=True)
class ResolvedAnchor:
    """A current-turn reference resolved to one immutable discourse anchor."""

    mention: str
    anchor: DiscourseAnchor
    score: int

    def to_dict(self) -> dict:
        return {"mention": self.mention, "score": self.score, "anchor": self.anchor.to_dict()}


@dataclass(frozen=True)
class ResolutionResult:
    resolved: tuple[ResolvedAnchor, ...] = ()
    ambiguous: bool = False
    grounding: str = "none"
    candidates: tuple[DiscourseAnchor, ...] = ()


@dataclass(frozen=True)
class AdmissionPreview:
    """Pure, immutable result of orienting one request before any session mutation.

    The host may route first, compute this preview against the prospective task, begin its durable journal,
    then apply ``admission`` and ``focus`` exactly once. ``contract`` remains a read-only compatibility name.
    """

    admission: object
    focus: tuple[dict, ...] = ()
    referenced_artifact_ids: tuple[str, ...] = ()
    # Exact source projections are turn-ephemeral. They may accumulate elastically inside the active slice,
    # but are deliberately excluded from the admission journal and cross-session task state.
    projections: tuple[dict, ...] = ()
    # Source identities + digests needed to reconstruct the same canonical evidence on an adjacent challenge.
    # Unlike ``projections``, this contains no utterance or receipt payload bytes.
    snapshot_basis: dict | None = None
    ambiguous: bool = False
    consume_pending_proposal: bool = True

    @property
    def contract(self):
        return self.admission

    def to_dict(self) -> dict:
        admission = self.admission
        return {
            "admission": admission.to_dict() if hasattr(admission, "to_dict") else admission,
            "focus": [dict(item) for item in self.focus],
            "referenced_artifact_ids": list(self.referenced_artifact_ids),
            "projection_kinds": [str(item.get("kind") or "") for item in self.projections],
            "ambiguous": self.ambiguous,
            "consume_pending_proposal": self.consume_pending_proposal,
        }


# Compatibility alias: there is one preview object, not parallel interpretation/admission writers.
TurnInterpretation = AdmissionPreview


def extract_addressable_anchors(text: str) -> tuple[DiscourseAnchor, ...]:
    """Extract Markdown numbered items with exact source ranges.

    Continuation lines belong to the item until the next numbered item or heading,
    so the archived excerpt remains useful rather than retaining only a lossy title.
    """
    source = str(text or "")
    if not source.strip():
        return ()
    heading = "numbered list"
    lines = source.splitlines(keepends=True)
    anchors: list[DiscourseAnchor] = []
    active: dict | None = None
    list_indent: int | None = None
    offset = 0

    def finish(end: int) -> None:
        nonlocal active
        if active is None:
            return
        excerpt = source[active["start"]:end].strip()
        first = _NUMBERED.match(source[active["start"]:active["line_end"]].rstrip("\r\n"))
        label = _plain(first.group(2) if first is not None else excerpt.splitlines()[0])
        stable = _STABLE_ID.search(excerpt)
        anchors.append(DiscourseAnchor(
            collection=active["collection"], ordinal=active["ordinal"],
            label=label[:300], excerpt=excerpt,
            source_range=(active["start"], end),
            stable_id=stable.group(0).casefold() if stable else "",
        ))
        active = None

    for line in lines:
        body = line.rstrip("\r\n")
        head = _HEADING.match(body) or _BOLD_HEADING.match(body)
        numbered = _NUMBERED.match(body)
        if head is not None:
            finish(offset)
            heading = _plain(head.group(1)) or "numbered list"
            list_indent = None
        elif numbered is not None:
            indent = len(body.expandtabs(4)) - len(body.expandtabs(4).lstrip(" "))
            if active is not None and indent > active["indent"]:
                pass  # nested item: retain it as detail inside the enclosing top-level item
            elif list_indent is not None and indent > list_indent:
                pass
            else:
                finish(offset)
                if list_indent is None:
                    list_indent = indent
                active = {
                    "start": offset, "line_end": offset + len(line), "indent": indent,
                    "ordinal": int(numbered.group(1)), "collection": heading,
                }
        offset += len(line)
    finish(len(source))
    return tuple(anchors)


def _artifact_text(artifact) -> str:
    body = dict(getattr(artifact, "structured_body", {}) or {})
    return str(
        body.get("assistant") or body.get("report") or
        getattr(artifact, "summary", "") or body.get("markdown") or ""
    )


def _receipt_operations_for_query(
    query: EvidenceQuery, operations: Iterable[Mapping],
) -> tuple[dict, ...]:
    """Select and shape receipt operations from the already-typed evidence query."""
    selected_names = _TOOLS_BY_EVIDENCE_FAMILY.get(query.family, frozenset())
    selected = []
    for raw in operations or ():
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "")
        if selected_names and name not in selected_names:
            continue
        args = raw.get("args") if isinstance(raw.get("args"), Mapping) else {}
        # Only source-identifying arguments are projected. Tool output and free-form command/task text remain
        # in the sealed artifact and can be opened explicitly if needed; lifecycle truth stays compact here.
        identity_args = {
            key: args[key] for key in ("agent", "name", "path")
            if key in args and isinstance(args[key], (str, int, float, bool))
        }
        raw_disposition = str(raw.get("disposition") or "unknown")
        known_dispositions = {
            "succeeded", "steered", "failed", "cancelled", "indeterminate", "not_started", "rejected",
        }
        disposition = raw_disposition if raw_disposition in known_dispositions else "unknown"
        reason_text = _plain(raw.get("rejection_reason") or raw.get("outcome_text") or "")
        reason_excerpt = reason_text[:240]
        effect_ids = tuple(dict.fromkeys(
            str(item) for item in (raw.get("effect_ids") or ()) if str(item or "")
        ))
        applied_effect_ids = tuple(dict.fromkeys(
            str(item) for item in (raw.get("applied_effect_ids") or ()) if str(item or "")
        ))
        artifact_refs = tuple(dict.fromkeys(
            str(item) for item in (raw.get("artifact_refs") or ()) if str(item or "")
        ))
        selected.append({
            "invocation_id": str(raw.get("invocation_id") or ""),
            "name": name or "(unknown tool)",
            "identity_args": identity_args,
            "requested": bool(raw.get("requested")),
            "rejected_before_execution": bool(raw.get("rejected_before_execution")),
            "execution_started": bool(raw.get("execution_started")),
            "settled": bool(raw.get("settled")),
            "disposition": disposition,
            **({"recorded_disposition": raw_disposition} if raw_disposition != disposition else {}),
            "effects_declared": len(effect_ids),
            "effects_applied": len(applied_effect_ids),
            "child_artifacts": len(artifact_refs),
            **({"reason": reason_excerpt, "reason_truncated": len(reason_text) > len(reason_excerpt)}
               if query.predicate == "failure_detail"
               and disposition in {
                   "rejected", "failed", "cancelled", "indeterminate", "not_started", "unknown",
               } and reason_text else {}),
        })
    return tuple(selected)


def _operation_counts(operations: Iterable[Mapping]) -> dict[str, int]:
    operations = tuple(operations or ())
    return {
        "requested": sum(bool(item.get("requested")) for item in operations),
        "rejected_before_execution": sum(bool(item.get("rejected_before_execution")) for item in operations),
        "steered_before_execution": sum(
            bool(item.get("rejected_before_execution"))
            and str(item.get("disposition")) == "steered"
            for item in operations
        ),
        "execution_started": sum(bool(item.get("execution_started")) for item in operations),
        "settled": sum(bool(item.get("settled")) for item in operations),
        "succeeded": sum(str(item.get("disposition")) == "succeeded" for item in operations),
        "steered": sum(str(item.get("disposition")) == "steered" for item in operations),
        "failed": sum(str(item.get("disposition")) == "failed" for item in operations),
        "cancelled": sum(str(item.get("disposition")) == "cancelled" for item in operations),
        "indeterminate": sum(str(item.get("disposition")) == "indeterminate" for item in operations),
        "not_started": sum(str(item.get("disposition")) == "not_started" for item in operations),
        "unknown": sum(str(item.get("disposition")) not in {
            "succeeded", "steered", "failed", "cancelled", "indeterminate", "not_started", "rejected",
        } or str(item.get("disposition")) == "unknown" for item in operations),
        "effects_declared": sum(int(item.get("effects_declared") or 0) for item in operations),
        "effects_applied": sum(int(item.get("effects_applied") or 0) for item in operations),
        "child_artifacts": sum(int(item.get("child_artifacts") or 0) for item in operations),
    }


_TURN_DISPOSITIONS = (
    "completed", "completed_with_warnings", "paused", "blocked", "interrupted", "indeterminate", "unknown",
)


def _turn_disposition(value: object) -> str:
    value = str(value or "unknown")
    return value if value in _TURN_DISPOSITIONS else "unknown"


def _digest_ids(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(dict.fromkeys(str(item) for item in values if str(item or ""))):
        digest.update(value.encode("utf-8")); digest.update(b"\0")
    return digest.hexdigest()


def _artifact_gap_ids(gaps: Iterable) -> tuple[str, ...]:
    """Normalize persistence discovery gaps without coupling evidence code to one store type."""
    out = []
    for gap in gaps or ():
        if isinstance(gap, Mapping):
            value = gap.get("artifact_id") or gap.get("id")
        else:
            value = getattr(gap, "artifact_id", "")
        if value:
            out.append(str(value))
    return tuple(dict.fromkeys(out))


def _coerce_evidence_query(value) -> EvidenceQuery:
    if isinstance(value, EvidenceQuery):
        return value
    derived = derive_evidence_query(str(value or ""))
    return derived or EvidenceQuery(source="execution_receipt")


def _has_durable_order(artifact) -> bool:
    body = getattr(artifact, "structured_body", {}) or {}
    meta = body.get("meta") if isinstance(body, Mapping) else None
    value = meta.get("order_ns") if isinstance(meta, Mapping) else None
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _latest_order_candidates(items: Iterable) -> tuple:
    """Select latest, expanding an unresolved legacy same-second tie instead of inventing an order."""
    items = tuple(items)
    if not items:
        return ()
    latest = max(items, key=artifact_order_key)
    if _has_durable_order(latest):
        return (latest,)
    timestamp = str(getattr(latest, "timestamp", "") or "")
    return tuple(item for item in items if not _has_durable_order(item)
                 and str(getattr(item, "timestamp", "") or "") == timestamp)


def _artifacts_in_query_scope(query: EvidenceQuery, artifacts: Iterable) -> tuple:
    items = tuple(artifacts or ())
    if query.scope not in {"latest_turn", "latest_matching_execution"}:
        return items
    turns = tuple(item for item in items if str(getattr(item, "kind", "") or "") == "turn")
    if not turns:
        return ()
    order_key = artifact_order_key
    if query.scope == "latest_turn":
        return _latest_order_candidates(turns)

    # "last attempt/run/execution" means the latest turn containing an operation in the selected family,
    # not the latest conversational filler. Keep every newer turn in the proof domain: a receiptless newer
    # turn makes coverage partial, while a receipt-bearing unrelated turn establishes that no newer matching
    # execution occurred. If there is no known match, scan the full domain so zero remains exact/partial.
    matches = []
    for artifact in turns:
        body = dict(getattr(artifact, "structured_body", {}) or {})
        receipt = body.get("turn_receipt")
        if not isinstance(receipt, Mapping):
            continue
        raw_operations = receipt.get("operations")
        if isinstance(raw_operations, (list, tuple)) \
                and _receipt_operations_for_query(query, raw_operations):
            matches.append(artifact)
    if not matches:
        return turns
    latest_match = max(matches, key=order_key)
    boundary = order_key(latest_match)
    if not _has_durable_order(latest_match):
        boundary_timestamp = str(getattr(latest_match, "timestamp", "") or "")
        return tuple(item for item in turns if (
            _has_durable_order(item) and order_key(item) >= boundary
        ) or (
            not _has_durable_order(item)
            and str(getattr(item, "timestamp", "") or "") >= boundary_timestamp
        ))
    return tuple(item for item in turns if order_key(item) >= boundary)


def execution_receipts_from_artifacts(query: EvidenceQuery | str, artifacts: Iterable) -> tuple[dict, ...]:
    """Project only operation-level detail that the request needs.

    Aggregate/zero-match truth is compiled separately by
    :func:`aggregate_execution_receipts_from_artifacts`, so 10,000 successful or irrelevant turns do not
    become 10,000 prompt referents. Detail remains elastic: an explicit operations/failure-detail request gets
    every matching operation and its stable artifact handle.
    """
    query = _coerce_evidence_query(query)
    out = []
    for artifact in _artifacts_in_query_scope(query, artifacts):
        body = dict(getattr(artifact, "structured_body", {}) or {})
        receipt = body.get("turn_receipt")
        if not isinstance(receipt, Mapping):
            continue
        raw_operations = receipt.get("operations")
        if not isinstance(raw_operations, (list, tuple)):
            raw_operations = ()
        operations = _receipt_operations_for_query(query, raw_operations)
        turn_disposition = _turn_disposition(receipt.get("disposition"))
        warning_values = tuple(
            _plain(item) for item in (receipt.get("warnings") or ()) if _plain(item)
        )
        turn_level_adverse = bool(
            query.family == "all"
            and (turn_disposition != "completed" or warning_values)
        )
        projected_operations = operations
        if query.predicate == "aggregate":
            continue
        elif query.predicate == "failure_detail":
            projected_operations = tuple(
                operation for operation in operations
                if operation.get("disposition") in {
                    "rejected", "failed", "cancelled", "indeterminate", "not_started", "unknown",
                }
            )
            if not projected_operations and not turn_level_adverse:
                continue
        elif not projected_operations:
            continue
        warning_excerpts = tuple(value[:240] for value in warning_values[:8])
        out.append({
            "kind": "execution_receipt",
            "artifact_id": str(getattr(artifact, "id", "") or ""),
            "timestamp": str(getattr(artifact, "timestamp", "") or ""),
            "turn_id": str(receipt.get("turn_id") or getattr(artifact, "id", "") or ""),
            "turn_disposition": turn_disposition,
            "turn_warning_count": len(warning_values),
            "turn_warning_excerpts": list(warning_excerpts),
            "turn_warnings_truncated": (
                len(warning_values) > len(warning_excerpts)
                or any(len(value) > 240 for value in warning_values[:8])
            ),
            "counts": _operation_counts(operations),
            "operations": list(projected_operations),
            "projection": "detail",
        })
    return tuple(out)


def aggregate_execution_receipts(
    query: EvidenceQuery, receipts: Iterable[Mapping],
) -> dict | None:
    """Compatibility fold for callers that already hold projected receipt rows."""
    receipts = tuple(item for item in receipts if isinstance(item, Mapping))
    if not receipts:
        return None
    keys = tuple(_operation_counts(()))
    counts = {
        key: sum(int((item.get("counts") or {}).get(key, 0))
                 for item in receipts if isinstance(item.get("counts"), Mapping))
        for key in keys
    }
    serialized = json.dumps(receipts, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "kind": "execution_receipt_aggregate",
        "query": query.to_dict(),
        "counts": counts,
        "receipt_count": len(receipts),
        "matching_receipt_count": len(receipts),
        "source_set_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "source_index_handle": "history/index.md",
    }


def aggregate_execution_receipts_from_artifacts(
    query: EvidenceQuery | str, artifacts: Iterable,
) -> dict | None:
    """Compile exact bounded counts over every receipt in scope, including an exact zero match.

    The result size is constant in task length. ``receipt_count`` proves how many canonical turn receipts were
    scanned, ``matching_receipt_count`` distinguishes zero relevant operations from no receipt source, and the
    digest binds the derived result to the immutable source set without copying every source ID into the slice.
    """
    query = _coerce_evidence_query(query)
    totals = {key: 0 for key in _operation_counts(())}
    receipt_count = 0
    matching_receipt_count = 0
    operation_count = 0
    source_digest = hashlib.sha256()
    projection_digest = hashlib.sha256(
        json.dumps(query.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    turn_counts = {key: 0 for key in _TURN_DISPOSITIONS}
    warning_count = 0
    nonclean_turn_count = 0
    child_artifact_ids: set[str] = set()
    sealed_artifact_ids: set[str] = set()
    direct_file_paths: set[str] = set()
    file_operations_without_path = 0
    opaque_command_operations = 0
    scoped_artifacts = sorted(
        _artifacts_in_query_scope(query, artifacts),
        key=lambda item: str(getattr(item, "id", "") or ""),
    )
    for artifact in scoped_artifacts:
        if str(getattr(artifact, "kind", "") or "") != "turn":
            continue
        artifact_id = str(getattr(artifact, "id", "") or "")
        projection_digest.update(artifact_id.encode("utf-8")); projection_digest.update(b"\0")
        body = dict(getattr(artifact, "structured_body", {}) or {})
        receipt = body.get("turn_receipt")
        if not isinstance(receipt, Mapping):
            projection_digest.update(b"missing-receipt\0")
            continue
        receipt_count += 1
        turn_disposition = _turn_disposition(receipt.get("disposition"))
        turn_counts[turn_disposition] += 1
        warnings = tuple(receipt.get("warnings") or ())
        warning_count += len(warnings)
        if turn_disposition != "completed" or warnings:
            nonclean_turn_count += 1
        raw_operations = receipt.get("operations")
        if not isinstance(raw_operations, (list, tuple)):
            raw_operations = ()
        opaque_command_operations += sum(
            str(raw.get("name") or "") in {"execute_code", "run_command"}
            for raw in raw_operations if isinstance(raw, Mapping)
        )
        operations = _receipt_operations_for_query(query, raw_operations)
        if operations:
            matching_receipt_count += 1
            operation_count += len(operations)
        counts = _operation_counts(operations)
        for key in totals:
            totals[key] += counts[key]
        if query.family in {"file", "file_read", "file_write"}:
            for operation in operations:
                identity_args = operation.get("identity_args")
                path = identity_args.get("path") if isinstance(identity_args, Mapping) else None
                if isinstance(path, str) and path:
                    direct_file_paths.add(path)
                else:
                    file_operations_without_path += 1
        selected_names = _TOOLS_BY_EVIDENCE_FAMILY.get(query.family, frozenset())
        for raw in raw_operations:
            if not isinstance(raw, Mapping):
                continue
            if selected_names and str(raw.get("name") or "") not in selected_names:
                continue
            child_artifact_ids.update(
                str(item) for item in (raw.get("artifact_refs") or ()) if str(item or "")
            )
        sealed_artifact_ids.update(
            str(item) for item in (receipt.get("artifact_refs") or ()) if str(item or "")
        )
        receipt_bytes = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
        source_digest.update(artifact_id.encode("utf-8")); source_digest.update(b"\0")
        source_digest.update(receipt_bytes); source_digest.update(b"\0")
        projection_digest.update(receipt_bytes); projection_digest.update(b"\0")
    if not receipt_count:
        return None
    return {
        "kind": "execution_receipt_aggregate",
        "query": query.to_dict(),
        "counts": totals,
        "receipt_count": receipt_count,
        "matching_receipt_count": matching_receipt_count,
        "operation_count": operation_count,
        "turn_counts": turn_counts,
        "turn_warning_count": warning_count,
        "nonclean_turn_count": nonclean_turn_count,
        "child_artifact_count": len(child_artifact_ids),
        "sealed_artifact_ref_count": len(sealed_artifact_ids),
        "distinct_direct_file_path_count": len(direct_file_paths),
        "direct_file_path_set_sha256": _digest_ids(direct_file_paths),
        "file_operations_without_path": file_operations_without_path,
        "opaque_command_operation_count": opaque_command_operations,
        "source_set_sha256": source_digest.hexdigest(),
        "projection_sha256": projection_digest.hexdigest(),
        "source_index_handle": "artifacts/index.md",
    }


def execution_receipt_coverage(
    artifacts: Iterable, query: EvidenceQuery | str | None = None, *, gaps: Iterable = (),
) -> dict:
    """Describe receipt coverage across the exact task or latest-turn scope selected by the query."""
    query = _coerce_evidence_query(query or EvidenceQuery(source="execution_receipt"))
    turns = sorted(
        (artifact for artifact in _artifacts_in_query_scope(query, artifacts)
         if str(getattr(artifact, "kind", "") or "") == "turn"),
        key=lambda artifact: str(getattr(artifact, "id", "") or ""),
    )
    receipt_bearing = []
    missing = []
    corrupt = _artifact_gap_ids(gaps)
    ambiguous = ()
    if query.scope in {"latest_turn", "latest_matching_execution"}:
        legacy_by_timestamp: dict[str, list[str]] = {}
        for artifact in turns:
            if _has_durable_order(artifact):
                continue
            timestamp = str(getattr(artifact, "timestamp", "") or "")
            legacy_by_timestamp.setdefault(timestamp, []).append(
                str(getattr(artifact, "id", "") or "")
            )
        ambiguous = tuple(
            artifact_id for ids in legacy_by_timestamp.values() if len(ids) > 1
            for artifact_id in ids if artifact_id
        )
    for artifact in turns:
        body = dict(getattr(artifact, "structured_body", {}) or {})
        artifact_id = str(getattr(artifact, "id", "") or "")
        if isinstance(body.get("turn_receipt"), Mapping):
            receipt_bearing.append(artifact_id)
        elif artifact_id:
            missing.append(artifact_id)
    return {
        "kind": "execution_receipt_coverage",
        "candidate_turn_artifacts": len(turns),
        "receipt_bearing": len(receipt_bearing),
        "coverage": "partial" if missing or corrupt or ambiguous else "complete",
        "scope": query.scope,
        "candidate_set_sha256": _digest_ids(
            str(getattr(artifact, "id", "") or "") for artifact in turns
        ),
        "missing_receipt_count": len(missing),
        "missing_receipt_sample": missing[:3],
        "missing_set_sha256": _digest_ids(missing),
        "corrupt_artifact_count": len(corrupt),
        "corrupt_artifact_sample": list(corrupt[:3]),
        "corrupt_set_sha256": _digest_ids(corrupt),
        "ambiguous_order_count": len(ambiguous),
        "ambiguous_order_sample": list(ambiguous[:3]),
        "ambiguous_order_set_sha256": _digest_ids(ambiguous),
        "source_index_handle": "artifacts/index.md",
    }


_EXECUTION_EVIDENCE_KINDS = frozenset({
    "execution_receipt", "execution_receipt_aggregate", "execution_receipt_coverage",
    "execution_receipt_absence",
})
_QUALITY_EVIDENCE_KINDS = frozenset({"quality_exchange", "quality_exchange_coverage"})


def _quality_scope_artifacts(query: QualityEvidenceQuery, artifacts: Iterable) -> tuple:
    turns = sorted(
        (artifact for artifact in artifacts
         if str(getattr(artifact, "kind", "") or "") == "turn"),
        key=artifact_order_key,
    )
    return _latest_order_candidates(turns) if query.scope == "latest_response" and turns else turns


def _json_value(value):
    """Return the exact JSON value carried by an immutable artifact record."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sealed_artifact_record(artifact) -> dict:
    """Canonical field-for-field projection of a sealed core artifact."""
    try:
        version = int(getattr(artifact, "schema_version", 1) or 1)
    except (TypeError, ValueError):
        version = 1
    return {
        "v": version,
        "id": str(getattr(artifact, "id", "") or ""),
        "kind": str(getattr(artifact, "kind", "") or ""),
        "workspace_id": str(getattr(artifact, "workspace_id", "") or ""),
        "session_id": str(getattr(artifact, "session_id", "") or ""),
        "task_id": str(getattr(artifact, "task_id", "") or ""),
        "parent_id": str(getattr(artifact, "parent_id", "") or ""),
        "timestamp": str(getattr(artifact, "timestamp", "") or ""),
        "status": str(getattr(artifact, "status", "unknown") or "unknown"),
        "title": str(getattr(artifact, "title", "") or ""),
        "brief": _json_value(getattr(artifact, "brief", {}) or {}),
        "summary": str(getattr(artifact, "summary", "") or ""),
        "structured_body": _json_value(getattr(artifact, "structured_body", {}) or {}),
        "files": _json_value(getattr(artifact, "files", ()) or ()),
        "refs": _json_value(getattr(artifact, "refs", ()) or ()),
        "uncertainty": _json_value(getattr(artifact, "uncertainty", ()) or ()),
        "error": str(getattr(artifact, "error", "") or ""),
    }


def _turn_grounding_artifact_ids(artifact) -> tuple[str, ...]:
    """Collect operation-local evidence relationships in stable receipt order.

    A receipt's top-level ``artifact_refs`` are checkpoint/dependency handoffs. They often carry a prior turn
    forward merely for continuity, which does not make that turn factual support for every later response.
    Operation-local refs currently identify sealed child reports and are the typed provenance edge that the
    response actually consumed. Future observation artifacts can use the same operation-local edge without
    reintroducing generic dependency inheritance.
    """
    body = dict(getattr(artifact, "structured_body", {}) or {})
    receipt = body.get("turn_receipt")
    if not isinstance(receipt, Mapping):
        return ()
    refs = []
    for operation in receipt.get("operations") or ():
        if not isinstance(operation, Mapping):
            continue
        for raw in operation.get("artifact_refs") or ():
            ref = str(raw or "")
            if ref:
                refs.append(ref)
    return tuple(dict.fromkeys(refs))


def _quality_grounding_artifact_ids(turns: Iterable) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        artifact_id
        for turn in turns
        for artifact_id in _turn_grounding_artifact_ids(turn)
    ))


def _subagent_grounding_envelope(record: Mapping) -> dict:
    """Project child claims and the explicitly bounded observation preview as distinct evidence layers.

    Current child artifacts may retain a much larger page-backed observation archive. That archive is never
    copied wholesale into a quality-audit prompt; the exact artifact/evidence locators remain the refinement
    path. Legacy artifacts had only ``observations``, which was already bounded at capture time.
    """
    body = record.get("structured_body") if isinstance(record.get("structured_body"), Mapping) else {}
    brief = body.get("brief") if isinstance(body.get("brief"), Mapping) else record.get("brief")
    brief = brief if isinstance(brief, Mapping) else {}
    archived_observations = body.get("observations") or ()
    if not isinstance(archived_observations, (list, tuple)):
        archived_observations = ()
    preview_rows = (
        body.get("observation_preview")
        if "observation_preview" in body else archived_observations
    ) or ()
    if not isinstance(preview_rows, (list, tuple)):
        preview_rows = ()
    observations = [
        {
            "v": int(item.get("v") or 1),
            "tool": str(item.get("tool") or "unknown"),
            "args": _json_value(item.get("args") or {}),
            "status": str(item.get("status") or "unknown"),
            "view": str(item.get("view") or ""),
            "raw_sha256": str(item.get("raw_sha256") or ""),
            "view_sha256": str(item.get("view_sha256") or ""),
            "raw_bytes": int(item.get("raw_bytes") or 0),
            "view_bytes": int(item.get("view_bytes") or 0),
            "redacted": bool(item.get("redacted")),
            "truncated": bool(item.get("truncated")),
        }
        for item in preview_rows
        if isinstance(item, Mapping)
    ]
    report = str(body.get("report") or "")
    # Claims bind to the full archived views. Only claims whose references resolve there are well-formed; the
    # bounded preview may omit the supporting bytes, in which case it remains a locator rather than support.
    observation_hashes = {
        str(item.get("view_sha256") or "")
        for item in archived_observations
        if isinstance(item, Mapping) and str(item.get("status") or "") == "succeeded"
    }
    claims = []
    from .subagent_contract import SubagentClaim
    raw_claims = body.get("claims") or ()
    if not isinstance(raw_claims, (list, tuple)):
        raw_claims = ()
    for item in raw_claims[:8]:
        if not isinstance(item, Mapping):
            continue
        try:
            claim = SubagentClaim.from_dict(item)
        except (TypeError, ValueError):
            continue
        if claim.report_exact not in report or not set(claim.observation_refs).issubset(observation_hashes):
            continue
        claims.append(claim.to_dict())
    revision = body.get("workspace_revision")
    dependencies = (
        revision.get("dependencies") if isinstance(revision, Mapping)
        and isinstance(revision.get("dependencies"), (list, tuple)) else ()
    )
    return {
        "schema": "subagent-grounding-v1",
        "brief": {
            "objective": str(brief.get("objective") or brief.get("task") or ""),
            "scope": _json_value(brief.get("scope") or ()),
            "exclusions": _json_value(brief.get("exclusions") or ()),
            "report_shape": str(brief.get("report_shape") or ""),
        },
        "status": str(record.get("status") or body.get("status") or "unknown"),
        "coverage": str(body.get("coverage") or ""),
        # A report is the child's claim layer. Workspace-fact support lives only in observations below.
        "report": report,
        # Each entry is exact child testimony indexed from `report`; observation_refs are candidate locators,
        # not proof that the interpretation follows from those bytes.
        "claims": claims,
        "findings": _json_value(body.get("findings") or ()),
        "files": _json_value(body.get("files") or record.get("files") or ()),
        "workspace_dependencies": _json_value(dependencies),
        "gaps": _json_value(body.get("gaps") or ()),
        "uncertainty": _json_value(body.get("uncertainty") or record.get("uncertainty") or ()),
        "conflicts": _json_value(body.get("conflicts") or ()),
        "observations": observations,
    }


def _grounding_projection(artifact) -> dict:
    record = _sealed_artifact_record(artifact)
    body = record["structured_body"]
    source_text = ""
    source_text_kind = "canonical_record_json"
    observation_count = 0
    complete_observation_count = 0
    if record["kind"] == "subagent":
        envelope = _subagent_grounding_envelope(record)
        observations = envelope["observations"]
        observation_count = len(observations)
        complete_observation_count = sum(
            not item["redacted"] and not item["truncated"] and item["status"] == "succeeded"
            for item in observations
        )
        source_text = json.dumps(envelope, sort_keys=True, indent=2, ensure_ascii=False)
        source_text_kind = "subagent_grounding_v1"
    elif isinstance(body, Mapping):
        for key in ("report", "assistant", "markdown"):
            value = body.get(key)
            if isinstance(value, str) and value:
                source_text = value
                source_text_kind = key
                break
    if not source_text:
        source_text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    record_bytes = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return {
        "artifact_id": record["id"],
        "artifact_kind": record["kind"],
        "source_text_kind": source_text_kind,
        "source_text": source_text,
        "observation_count": observation_count,
        "complete_observation_count": complete_observation_count,
        # Bind the projection to the entire immutable record without copying that often-large record into every
        # model slice. The exact source text is sufficient for claim checking; the artifact handle is available
        # for deeper inspection and this digest makes same-ID metadata drift invalidate frozen reuse.
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
    }


_BRIEF_RESPONSE_REQUEST = re.compile(
    r"\b(?:briefly|in\s+brief|be\s+(?:very\s+)?brief|keep\s+(?:it|the\s+(?:answer|response))\s+"
    r"(?:very\s+)?(?:brief|concise))\b",
    re.IGNORECASE,
)
_NEGATED_BRIEF_REQUEST = re.compile(
    r"\b(?:do\s+not|don't|dont|need\s+not|needn't)\s+(?:be\s+)?(?:brief|concise)\b",
    re.IGNORECASE,
)
_EXACT_LINE_REQUEST = re.compile(
    r"\b(?:(?:exactly|in)\s+)?(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"[ -]?(?:physical[ -]?)?lines?\b",
    re.IGNORECASE,
)
_JSON_RESPONSE_REQUEST = re.compile(
    r"\b(?:return|respond(?:\s+with)?|output|format(?:\s+the\s+(?:answer|response))?\s+as)\b"
    r"[^.\n]{0,48}\bjson\b",
    re.IGNORECASE,
)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _produced_exact_sample(assistant: str, *, limit: int = 360) -> str:
    """A bounded verbatim prefix; no ellipsis is appended because the field must remain an exact substring."""
    value = str(assistant or "")
    if len(value) <= limit:
        return value
    head = value[:limit]
    # Prefer a complete displayed line, then a complete word. `rstrip` still leaves an exact source prefix and
    # avoids the misleading mid-token cut that a raw character slice produced in early live checkpoints.
    line_cut = head.rfind("\n")
    word_cut = max(head.rfind(" "), head.rfind("\t"))
    cut = line_cut if line_cut >= limit // 2 else word_cut
    return head[:cut].rstrip() if cut > 0 else head


def _deterministic_response_constraint_mismatches(request: str, assistant: str) -> tuple[dict, ...]:
    """Check only explicit constraints whose satisfaction is mechanically decidable.

    This is intentionally not a general quality judge. It closes the gap between retained user intent and a
    later self-audit for strict shape/format constraints, while leaving semantic usefulness to the source-exact
    model protocol. Conservative thresholds avoid turning ordinary style preferences into historical defects.
    """
    request = str(request or "")
    assistant = str(assistant or "")
    mismatches = []
    words = re.findall(r"\b\w+(?:[-']\w+)*\b", assistant, re.UNICODE)
    nonempty_lines = [line for line in assistant.splitlines() if line.strip()]
    produced_sample = _produced_exact_sample(assistant)
    sample_is_prefix = len(produced_sample) < len(assistant)
    brief_match = _BRIEF_RESPONSE_REQUEST.search(request)
    if brief_match and not _NEGATED_BRIEF_REQUEST.search(request) \
            and (len(words) > 80 or len(assistant) > 600):
        mismatches.append({
            "kind": "deterministic_quality_mismatch",
            "constraint": "brief_response",
            "category": "violated explicit format or constraint",
            "requested_exact": request,
            "produced_exact": produced_sample,
            "produced_exact_is_bounded_prefix": sample_is_prefix,
            "measurements": {
                "words": len(words), "characters": len(assistant),
                "nonempty_lines": len(nonempty_lines), "brief_word_ceiling": 80,
                "brief_character_ceiling": 600,
            },
            "explanation": (
                f"the request explicitly says {brief_match.group(0)!r}, but the sealed response contains "
                f"{len(words)} words and {len(assistant)} characters; either measure exceeds the conservative "
                "brief-response ceiling (80 words / 600 characters)"
            ),
        })

    line_match = _EXACT_LINE_REQUEST.search(request)
    if line_match and re.search(r"\b(?:exactly|summary|response|answer|return|give)\b", request, re.IGNORECASE):
        token = line_match.group("count").casefold()
        expected = int(token) if token.isdigit() else _COUNT_WORDS[token]
        actual = len(nonempty_lines)
        if actual != expected:
            mismatches.append({
                "kind": "deterministic_quality_mismatch",
                "constraint": "exact_nonempty_lines",
                "category": "violated explicit format or constraint",
                "requested_exact": request,
                "produced_exact": produced_sample,
                "produced_exact_is_bounded_prefix": sample_is_prefix,
                "measurements": {"expected_nonempty_lines": expected, "actual_nonempty_lines": actual},
                "explanation": (
                    f"the request specifies {expected} line(s), while the sealed response contains "
                    f"{actual} non-empty physical line(s)"
                ),
            })

    if _JSON_RESPONSE_REQUEST.search(request):
        try:
            json.loads(assistant)
            valid_json = True
        except (TypeError, ValueError, json.JSONDecodeError):
            valid_json = False
        if not valid_json:
            mismatches.append({
                "kind": "deterministic_quality_mismatch",
                "constraint": "valid_json",
                "category": "violated explicit format or constraint",
                "requested_exact": request,
                "produced_exact": produced_sample,
                "produced_exact_is_bounded_prefix": sample_is_prefix,
                "measurements": {"valid_json": False},
                "explanation": "the request explicitly requires JSON, but the complete sealed response is not valid JSON",
            })
    return tuple(mismatches)


def quality_exchanges_from_artifacts(
    query: QualityEvidenceQuery, artifacts: Iterable, *, grounding_artifacts: Iterable = (),
    gaps: Iterable = (),
) -> tuple[dict, ...]:
    """Project exact sealed request/assistant pairs for an observed response-quality judgment.

    This projection deliberately derives no quality verdict. It makes the two utterances and every exact sealed
    artifact named by their receipt co-resident so the model can prove a concrete mismatch instead of judging a
    paged-out response or its factual grounding from a lossy manifest preview. Direct grounding references may
    cross the selected turn scope, so their lookup domain is supplied separately from the candidate turn domain.
    """
    artifacts = tuple(artifacts or ())
    turns = _quality_scope_artifacts(query, artifacts)
    artifacts_by_id = {
        str(getattr(artifact, "id", "") or ""): artifact
        for artifact in (*artifacts, *tuple(grounding_artifacts or ()))
        if str(getattr(artifact, "id", "") or "")
    }
    rows = []
    missing = []
    missing_grounding = []
    projected_grounding_ids = []
    partial_responses = 0
    candidate_ids = []
    source_digest = hashlib.sha256()
    grounding_digest = hashlib.sha256()
    for artifact in turns:
        artifact_id = str(getattr(artifact, "id", "") or "")
        candidate_ids.append(artifact_id)
        body = dict(getattr(artifact, "structured_body", {}) or {})
        brief = dict(getattr(artifact, "brief", {}) or {})
        request = body.get("request")
        if not isinstance(request, str):
            request = brief.get("request")
        assistant = body.get("assistant")
        status = str(getattr(artifact, "status", "") or "")
        provenance = str(body.get("assistant_provenance") or "")
        if not provenance:
            provenance = "final_response" if status == "end_turn" else "unknown"
        if not isinstance(request, str) or not isinstance(assistant, str) \
                or not request.strip() or not assistant.strip():
            missing.append(artifact_id)
            continue
        if provenance not in {"final_response", "partial_or_note"}:
            missing.append(artifact_id)
            continue
        if provenance == "partial_or_note":
            partial_responses += 1
        constraint_mismatches = (
            _deterministic_response_constraint_mismatches(request, assistant)
            if provenance == "final_response" else ()
        )
        grounding_ids = _turn_grounding_artifact_ids(artifact)
        grounding_artifacts = []
        row_missing_grounding = []
        for grounding_id in grounding_ids:
            grounding = artifacts_by_id.get(grounding_id)
            if grounding is None:
                missing_grounding.append(grounding_id)
                row_missing_grounding.append(grounding_id)
                continue
            projected_grounding_ids.append(grounding_id)
            grounding_artifacts.append(_grounding_projection(grounding))
        row = {
            "kind": "quality_exchange",
            "artifact_id": artifact_id,
            "timestamp": str(getattr(artifact, "timestamp", "") or ""),
            "request": request,
            "assistant": assistant,
            "assistant_provenance": provenance,
            "turn_status": status,
            "grounding_artifact_ids": list(grounding_ids),
            "grounding_artifacts": grounding_artifacts,
            "missing_grounding_artifact_ids": row_missing_grounding,
            "deterministic_mismatches": list(constraint_mismatches),
        }
        rows.append(row)
        row_bytes = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        source_digest.update(row_bytes); source_digest.update(b"\0")
        grounding_digest.update(artifact_id.encode("utf-8")); grounding_digest.update(b"\0")
        grounding_digest.update(json.dumps({
            "grounding_artifact_ids": list(grounding_ids),
            "grounding_artifacts": grounding_artifacts,
            "missing_grounding_artifact_ids": row_missing_grounding,
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        grounding_digest.update(b"\0")
    unique_projected_grounding = tuple(dict.fromkeys(projected_grounding_ids))
    unique_missing_grounding = tuple(dict.fromkeys(missing_grounding))
    corrupt = _artifact_gap_ids(gaps)
    ambiguous = tuple(candidate_ids) if query.scope == "latest_response" and len(turns) > 1 else ()
    coverage = (
        "unavailable" if not turns else
        "partial" if missing or unique_missing_grounding or corrupt or ambiguous else
        "complete"
    )
    header = {
        "kind": "quality_exchange_coverage",
        "query": query.to_dict(),
        "coverage": coverage,
        "candidate_turn_artifacts": len(turns),
        "complete_exchange_pairs": len(rows),
        "missing_exchange_count": len(missing),
        "partial_response_pairs": partial_responses,
        "deterministic_mismatch_count": sum(
            len(row.get("deterministic_mismatches") or ()) for row in rows
        ),
        "missing_exchange_sample": missing[:3],
        "grounding_artifact_count": len(unique_projected_grounding),
        "missing_grounding_artifact_count": len(unique_missing_grounding),
        "missing_grounding_artifact_sample": list(unique_missing_grounding[:3]),
        "corrupt_artifact_count": len(corrupt),
        "corrupt_artifact_sample": list(corrupt[:3]),
        "corrupt_set_sha256": _digest_ids(corrupt),
        "ambiguous_order_count": len(ambiguous),
        "ambiguous_order_sample": list(ambiguous[:3]),
        "ambiguous_order_set_sha256": _digest_ids(ambiguous),
        "candidate_set_sha256": _digest_ids(candidate_ids),
        "grounding_set_sha256": grounding_digest.hexdigest(),
        "source_set_sha256": source_digest.hexdigest(),
        "source_index_handle": "artifacts/index.md",
    }
    return (header, *rows)


def _execution_projection_signature(referents: Iterable[Mapping]) -> dict:
    aggregate = next((item for item in referents
                      if item.get("kind") == "execution_receipt_aggregate"), None)
    coverage = next((item for item in referents
                     if item.get("kind") == "execution_receipt_coverage"), None)
    absence = next((item for item in referents
                    if item.get("kind") == "execution_receipt_absence"), None)
    return {
        "state": "aggregate" if aggregate is not None else "absence" if absence is not None else "missing",
        "projection_sha256": str((aggregate or {}).get("projection_sha256") or ""),
        "source_set_sha256": str((aggregate or {}).get("source_set_sha256") or ""),
        "receipt_count": int((aggregate or {}).get("receipt_count", 0) or 0),
        "candidate_set_sha256": str((coverage or {}).get("candidate_set_sha256") or ""),
        "missing_set_sha256": str((coverage or {}).get("missing_set_sha256") or ""),
        "corrupt_set_sha256": str((coverage or {}).get("corrupt_set_sha256") or ""),
        "corrupt_artifact_count": int((coverage or {}).get("corrupt_artifact_count", 0) or 0),
        "ambiguous_order_set_sha256": str(
            (coverage or {}).get("ambiguous_order_set_sha256") or ""
        ),
        "coverage": str((coverage or {}).get("coverage") or "unavailable"),
    }


def _quality_projection_signature(projections: Iterable[Mapping]) -> dict:
    coverage = next((item for item in projections
                     if item.get("kind") == "quality_exchange_coverage"), None)
    return {
        "source_set_sha256": str((coverage or {}).get("source_set_sha256") or ""),
        "candidate_set_sha256": str((coverage or {}).get("candidate_set_sha256") or ""),
        "candidate_turn_artifacts": int((coverage or {}).get("candidate_turn_artifacts", 0) or 0),
        "complete_exchange_pairs": int((coverage or {}).get("complete_exchange_pairs", 0) or 0),
        "missing_exchange_count": int((coverage or {}).get("missing_exchange_count", 0) or 0),
        "partial_response_pairs": int((coverage or {}).get("partial_response_pairs", 0) or 0),
        "grounding_artifact_count": int((coverage or {}).get("grounding_artifact_count", 0) or 0),
        "missing_grounding_artifact_count": int(
            (coverage or {}).get("missing_grounding_artifact_count", 0) or 0
        ),
        "corrupt_artifact_count": int((coverage or {}).get("corrupt_artifact_count", 0) or 0),
        "corrupt_set_sha256": str((coverage or {}).get("corrupt_set_sha256") or ""),
        "ambiguous_order_set_sha256": str(
            (coverage or {}).get("ambiguous_order_set_sha256") or ""
        ),
        "grounding_set_sha256": str((coverage or {}).get("grounding_set_sha256") or ""),
        "coverage": str((coverage or {}).get("coverage") or "unavailable"),
    }


def make_evidence_snapshot(
    admission, projections: Iterable[Mapping], source_turn_id: str, *,
    snapshot_basis: Mapping | None = None, source_generation: int | None = None,
) -> dict | None:
    """Freeze source identities and digests—not payload bytes—for an adjacent verification turn."""
    execution_query = getattr(admission, "evidence_query", None)
    quality_query = getattr(admission, "quality_evidence_query", None)
    if (execution_query is None and quality_query is None) or not isinstance(snapshot_basis, Mapping):
        return None
    execution_refs = tuple(
        dict(ref) for ref in (getattr(admission, "referents", ()) or ())
        if isinstance(ref, Mapping) and str(ref.get("kind") or "") in _EXECUTION_EVIDENCE_KINDS
    )
    quality_rows = tuple(
        dict(item) for item in (projections or ())
        if isinstance(item, Mapping) and str(item.get("kind") or "") in _QUALITY_EVIDENCE_KINDS
    )
    basis = dict(snapshot_basis)
    if execution_query is not None \
            and basis.get("execution_signature") != _execution_projection_signature(execution_refs):
        return None
    if quality_query is not None \
            and basis.get("quality_signature") != _quality_projection_signature(quality_rows):
        return None
    return json.loads(json.dumps({
        "v": 2,
        "source_turn_id": str(source_turn_id or ""),
        "source_generation": int(source_generation or 0),
        "execution_query": execution_query.to_dict() if execution_query is not None else None,
        "quality_query": quality_query.to_dict() if quality_query is not None else None,
        "basis": basis,
    }, ensure_ascii=False))


def _snapshot_queries(snapshot: Mapping | None) \
        -> tuple[EvidenceQuery | None, QualityEvidenceQuery | None]:
    if not isinstance(snapshot, Mapping):
        return None, None
    try:
        version = int(snapshot.get("v") or 0)
    except (TypeError, ValueError):
        return None, None
    if version != 2:
        return None, None
    return (
        EvidenceQuery.from_dict(snapshot.get("execution_query") or {}),
        QualityEvidenceQuery.from_dict(snapshot.get("quality_query") or {}),
    )


def _valid_adjacent_snapshot(
    snapshot: Mapping | None, *, artifacts: Iterable, task_id: str,
    execution_query: EvidenceQuery | None, quality_query: QualityEvidenceQuery | None,
    current_generation: int | None = None,
) -> bool:
    """Validate source adjacency and query identity before canonical re-materialization."""
    if not isinstance(snapshot, Mapping):
        return False
    try:
        version = int(snapshot.get("v") or 0)
    except (TypeError, ValueError):
        return False
    if version != 2:
        return False
    source_turn_id = str(snapshot.get("source_turn_id") or "")
    turn_ids = {
        str(getattr(artifact, "id", "") or "") for artifact in artifacts
        if str(getattr(artifact, "kind", "") or "") == "turn"
        and (not task_id or str(getattr(artifact, "task_id", "") or "") == task_id)
    }
    if not source_turn_id or source_turn_id not in turn_ids:
        return False
    if current_generation is not None:
        try:
            if int(snapshot.get("source_generation") or 0) != int(current_generation):
                return False
        except (TypeError, ValueError):
            return False
    basis = snapshot.get("basis")
    if not isinstance(basis, Mapping):
        return False
    source_ids = tuple(basis.get("execution_artifact_ids") or ()) \
        + tuple(basis.get("quality_artifact_ids") or ()) \
        + tuple(basis.get("quality_grounding_artifact_ids") or ())
    if source_turn_id in source_ids:
        return False
    prior_execution, prior_quality = _snapshot_queries(snapshot)
    if execution_query != prior_execution:
        return False
    if quality_query is None:
        return prior_quality is None
    if prior_quality is None:
        return False
    return (
        quality_query.scope == prior_quality.scope
        and quality_query.prospective_requested == prior_quality.prospective_requested
        and quality_query.purpose == "verify_assessment"
    )


def _materialize_frozen_snapshot(
    snapshot: Mapping, *, artifacts: Iterable,
    execution_query: EvidenceQuery | None, quality_query: QualityEvidenceQuery | None,
) -> tuple[tuple[dict, ...], tuple[dict, ...]] | None:
    """Re-derive a frozen projection from exact immutable source handles and verify all stored digests."""
    basis = snapshot.get("basis")
    if not isinstance(basis, Mapping):
        return None
    by_id = {
        str(getattr(artifact, "id", "") or ""): artifact for artifact in artifacts
        if str(getattr(artifact, "id", "") or "")
    }
    source_turn_id = str(snapshot.get("source_turn_id") or "")
    source_turn = by_id.get(source_turn_id)
    if source_turn is None or str(getattr(source_turn, "kind", "") or "") != "turn":
        return None

    def sources(name: str, *, required_kind: str = "") -> tuple | None:
        raw = basis.get(name)
        if not isinstance(raw, (list, tuple)) or any(not isinstance(item, str) or not item for item in raw):
            return None
        if len(set(raw)) != len(raw) or any(item not in by_id for item in raw):
            return None
        selected = tuple(by_id[item] for item in raw)
        if required_kind and any(
            str(getattr(item, "kind", "") or "") != required_kind for item in selected
        ):
            return None
        return selected

    execution_refs: tuple[dict, ...] = ()
    if execution_query is not None:
        selected = sources("execution_artifact_ids", required_kind="turn")
        if selected is None:
            return None
        domain = tuple(
            artifact for artifact in by_id.values()
            if str(getattr(artifact, "kind", "") or "") == "turn"
            and str(getattr(artifact, "id", "") or "") != source_turn_id
            and (
                str(getattr(artifact, "session_id", "") or "")
                == str(getattr(source_turn, "session_id", "") or "")
                if execution_query.scope == "session" else
                str(getattr(artifact, "task_id", "") or "")
                == str(getattr(source_turn, "task_id", "") or "")
            )
        )
        expected = _artifacts_in_query_scope(execution_query, domain)
        if tuple(str(getattr(item, "id", "") or "") for item in selected) != tuple(
            str(getattr(item, "id", "") or "") for item in expected
        ):
            return None
        details = execution_receipts_from_artifacts(execution_query, selected)
        aggregate = aggregate_execution_receipts_from_artifacts(execution_query, selected)
        coverage = execution_receipt_coverage(selected, execution_query)
        if aggregate is not None:
            execution_refs = (aggregate, *details, coverage)
        else:
            execution_refs = ({
                "kind": "execution_receipt_absence",
                "query": execution_query.to_dict(),
            }, coverage)
        if basis.get("execution_signature") != _execution_projection_signature(execution_refs):
            return None
    elif basis.get("execution_artifact_ids"):
        return None

    quality_rows: tuple[dict, ...] = ()
    if quality_query is not None:
        _prior_execution_query, frozen_quality_query = _snapshot_queries(snapshot)
        if frozen_quality_query is None:
            return None
        selected = sources("quality_artifact_ids", required_kind="turn")
        if selected is None:
            return None
        selected_grounding = sources("quality_grounding_artifact_ids")
        if selected_grounding is None:
            return None
        domain = tuple(
            artifact for artifact in by_id.values()
            if str(getattr(artifact, "kind", "") or "") == "turn"
            and str(getattr(artifact, "id", "") or "") != source_turn_id
            and (
                str(getattr(artifact, "session_id", "") or "")
                == str(getattr(source_turn, "session_id", "") or "")
                if quality_query.scope == "session" else
                str(getattr(artifact, "task_id", "") or "")
                == str(getattr(source_turn, "task_id", "") or "")
            )
        )
        expected = _quality_scope_artifacts(quality_query, domain)
        if tuple(str(getattr(item, "id", "") or "") for item in selected) != tuple(
            str(getattr(item, "id", "") or "") for item in expected
        ):
            return None
        if tuple(str(getattr(item, "id", "") or "") for item in selected_grounding) != \
                _quality_grounding_artifact_ids(selected):
            return None
        quality_rows = quality_exchanges_from_artifacts(
            frozen_quality_query, selected, grounding_artifacts=selected_grounding,
        )
        if basis.get("quality_signature") != _quality_projection_signature(quality_rows):
            return None
    elif basis.get("quality_artifact_ids") or basis.get("quality_grounding_artifact_ids"):
        return None
    return execution_refs, quality_rows


def anchors_from_artifacts(artifacts: Iterable, *, task_id: str = "") -> tuple[DiscourseAnchor, ...]:
    """Load stored anchors (or derive them for older artifacts) with stable handles."""
    out: list[DiscourseAnchor] = []
    for sequence, artifact in enumerate(artifacts):
        artifact_task = str(getattr(artifact, "task_id", "") or "")
        if task_id and artifact_task != task_id:
            continue
        body = dict(getattr(artifact, "structured_body", {}) or {})
        stored = body.get("anchors")
        parsed = []
        if isinstance(stored, (list, tuple)):
            parsed = [anchor for raw in stored if (anchor := DiscourseAnchor.from_dict(raw)) is not None]
        if not parsed:
            parsed = list(extract_addressable_anchors(_artifact_text(artifact)))
        # A child is addressable by the order in which its spawn was accepted, not by completion time.
        # Core artifact IDs are immutable but opaque; legacy sub-N handles are completion-ordered.  This
        # synthetic anchor makes "the first subagent" stable even when concurrent child #2 finishes first.
        if str(getattr(artifact, "kind", "") or "") == "subagent":
            try:
                launch_ordinal = int(body.get("launch_ordinal") or 0)
            except (TypeError, ValueError):
                launch_ordinal = 0
            if launch_ordinal > 0:
                text = _artifact_text(artifact)
                brief = dict(getattr(artifact, "brief", {}) or {})
                label = str(
                    getattr(artifact, "title", "") or brief.get("objective") or
                    brief.get("task") or getattr(artifact, "summary", "") or
                    f"subagent {launch_ordinal}"
                )
                parent = str(getattr(artifact, "parent_id", "") or body.get("parent_id") or "this parent")
                parsed.append(DiscourseAnchor(
                    collection=f"subagents launched by {parent}", ordinal=launch_ordinal,
                    label=_plain(label)[:300], excerpt=text[:2000] or _plain(label),
                    source_range=(0, len(text)),
                ))
        for anchor in parsed:
            out.append(replace(
                anchor,
                artifact_id=str(getattr(artifact, "id", "") or anchor.artifact_id),
                task_id=artifact_task or anchor.task_id,
                sequence=sequence,
            ))
    return tuple(out)


def _mentions(request: str) -> tuple[tuple[str, int], ...]:
    text = str(request or "")
    low = text.casefold()
    # "top 2" is a requested cardinality, not a reference to item number 2.
    blocked = [(match.start(), match.end()) for match in re.finditer(r"\btop\s+\d+\b", low)]

    def inside(start: int) -> bool:
        return any(left <= start < right for left, right in blocked)

    hits: list[tuple[int, int, str, int]] = []
    patterns = (
        re.compile(r"(?:\bnumber\s+|\bno\.?\s*|#)\s*(\d{1,3})\b", re.IGNORECASE),
        re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\b", re.IGNORECASE),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            if not inside(match.start()):
                hits.append((match.start(), match.end(), match.group(0), int(match.group(1))))
    for word, ordinal in _ORDINAL_WORDS.items():
        for match in re.finditer(rf"\b{word}\b", low):
            if not inside(match.start()):
                hits.append((match.start(), match.end(), match.group(0), ordinal))
    hits.sort(key=lambda item: item[0])
    unique = []
    seen = set()
    for _start, _end, mention, ordinal in hits:
        key = (mention.casefold(), ordinal)
        if key not in seen:
            unique.append((mention, ordinal)); seen.add(key)
    return tuple(unique)


def _grounding(request: str, has_reference: bool) -> str:
    low = str(request or "").casefold()
    past = has_reference or bool(re.search(
        r"\b(?:original|earlier|previous|you\s+(?:said|told|listed|reported)|"
        r"what\s+(?:was|were)|again)\b", low,
    ))
    live = bool(re.search(r"\b(?:now|current(?:ly)?|still|today|fixed|exists?|present|true)\b", low))
    if past and live:
        return "both"
    if past:
        return "sealed_past"
    if live:
        return "live_present"
    return "none"


def resolve_discourse_references(
    request: str,
    anchors: Iterable[DiscourseAnchor],
    *,
    focus: Iterable[Mapping | DiscourseAnchor] = (),
) -> ResolutionResult:
    """Resolve ordinals against user-visible collections without re-deriving them.

    The resolver is deliberately conservative.  A tie between different source
    collections is returned as ambiguity; callers may refault more context or ask.
    """
    request = str(request or "")
    mentions = _mentions(request)
    pool = tuple(anchors)
    stable_mentions = tuple(dict.fromkeys(match.group(0).casefold()
                                          for match in _STABLE_ID.finditer(request)))
    grounding = _grounding(request, bool(mentions or stable_mentions))
    focus_keys = set()
    focus_anchors = []
    for raw in focus or ():
        anchor = raw if isinstance(raw, DiscourseAnchor) else DiscourseAnchor.from_dict(raw)
        if anchor is not None:
            focus_keys.add((anchor.artifact_id, anchor.collection, anchor.ordinal, anchor.source_range))
            focus_anchors.append(anchor)
    stable_resolved = []
    for mention in stable_mentions:
        matches = [anchor for anchor in pool if anchor.stable_id == mention]
        identities = {
            (anchor.artifact_id, anchor.collection, anchor.ordinal, anchor.source_range)
            for anchor in matches
        }
        if len(identities) == 1:
            stable_resolved.append(ResolvedAnchor(mention, matches[0], 100))
    if not mentions:
        if stable_resolved:
            return ResolutionResult(resolved=tuple(stable_resolved), grounding="sealed_past")
        if focus_anchors and _FOCUS_REFERENCE.search(request):
            return ResolutionResult(
                resolved=tuple(ResolvedAnchor("current discourse focus", anchor, 100)
                               for anchor in focus_anchors),
                grounding="sealed_past",
            )
        return ResolutionResult(grounding=grounding)
    req_tokens = _tokens(request) - _REFERENCE_STOP
    low = request.casefold()
    original = "original" in low or "first report" in low
    again = "again" in low or "one more time" in low
    resolved: list[ResolvedAnchor] = list(stable_resolved)
    ambiguous_candidates: list[DiscourseAnchor] = []

    for mention, ordinal in mentions:
        candidates = [anchor for anchor in pool if anchor.ordinal == ordinal]
        if not candidates:
            continue
        oldest = min((anchor.sequence for anchor in candidates), default=0)
        scored: list[tuple[int, DiscourseAnchor]] = []
        for anchor in candidates:
            collection_tokens = _tokens(anchor.collection)
            label_tokens = _tokens(anchor.label)
            score = 4 * len(req_tokens & collection_tokens) + len(req_tokens & label_tokens)
            haystack = f"{anchor.collection} {anchor.label} {anchor.stable_id}".casefold()
            if "subagent" in low and ("subagent" in haystack or anchor.stable_id.startswith("sub-")):
                score += 8
            if "high" in low and "high" in haystack:
                score += 5
            if anchor.stable_id and anchor.stable_id in low:
                score += 20
            if (anchor.artifact_id, anchor.collection, anchor.ordinal, anchor.source_range) in focus_keys:
                score += 10 if again else 4
            if original and anchor.sequence == oldest:
                score += 10
            scored.append((score, anchor))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][0]
        winners = [anchor for score, anchor in scored if score == best]
        identities = {(anchor.artifact_id, anchor.collection, anchor.source_range) for anchor in winners}
        if len(identities) != 1:
            ambiguous_candidates.extend(winners)
            continue
        resolved.append(ResolvedAnchor(mention, winners[0], best))

    return ResolutionResult(
        resolved=tuple(resolved), ambiguous=bool(ambiguous_candidates), grounding=grounding,
        candidates=tuple(ambiguous_candidates),
    )


def render_resolved_references(result: ResolutionResult) -> str:
    if not result.resolved:
        return ""
    lines = []
    for item in result.resolved:
        anchor = item.anchor
        source = f"artifacts/{anchor.artifact_id}.md" if anchor.artifact_id else "sealed turn artifact"
        lines.append(
            f"- {item.mention} → {anchor.collection} item {anchor.ordinal} "
            f"(sealed source: {source})\n{anchor.excerpt}"
        )
    return "\n".join(lines)


def _subject_from_focus(focus: Iterable[Mapping | DiscourseAnchor]):
    from .intent import EntityRef

    for raw in reversed(tuple(focus or ())):
        if not isinstance(raw, Mapping) or raw.get("kind") != "subject_focus":
            continue
        entity = EntityRef.from_dict(raw.get("entity") or {})
        if entity is not None:
            return entity
    return None


def _explicit_subject(request: str):
    from .intent import EntityRef

    if (match := _EXPLICIT_SELF_TARGET.search(request)) is not None:
        return EntityRef(
            "SliceAgent", kind="agent", source="explicit",
            source_span=(match.start(), match.end()),
        )
    matches = list(_NAMED_SUBJECT.finditer(request))
    for match in reversed(matches):
        label = match.group(1)
        if label.casefold() in _GENERIC_SUBJECTS:
            continue
        kind = match.group(2).casefold()
        if kind.startswith("repo"):
            kind = "repository"
        return EntityRef(
            label, kind=kind, source="explicit",
            source_span=(match.start(1), match.end(2)),
        )
    return None


def _focus_repair(request: str):
    from .intent import EntityRef, FocusRepair

    match = _SUBJECT_REPAIR.search(request)
    if match is None:
        return None
    label = match.group(1)
    kind = str(match.group(2) or "entity").casefold()
    # Without a subject noun, accept only a proper-name-shaped replacement. This keeps ordinary phrases like
    # "I mean the correction" from silently replacing task focus.
    if match.group(2) is None and not label[:1].isupper():
        return None
    if kind.startswith("repo"):
        kind = "repository"
    entity = EntityRef(
        label, kind=kind, source="repair",
        source_span=(match.start(1), match.end(0)),
    )
    return FocusRepair(entity, (match.start(), match.end()), field="target")


def interpret_turn(
    request: str,
    artifacts: Iterable,
    *,
    task_id: str = "",
    session_id: str = "",
    recent_assistant: Iterable[str] = (),
    focus: Iterable[Mapping | DiscourseAnchor] = (),
    pending_proposal: Mapping | None = None,
    previous_evidence_snapshot: Mapping | None = None,
    current_generation: int | None = None,
    # Compatibility inputs for pure lexical callers. Production continuation uses the frozen snapshot above.
    previous_evidence_query: EvidenceQuery | Mapping | None = None,
    previous_quality_evidence_query: QualityEvidenceQuery | Mapping | None = None,
) -> TurnInterpretation:
    """Build the single shared turn interpretation from durable discourse candidates."""
    from .intent import analyze_turn

    inherited_subject = _subject_from_focus(focus)
    explicit_subject = _explicit_subject(request)
    repair = _focus_repair(request)
    artifact_gaps = tuple(getattr(artifacts, "gaps", ()) or ())
    all_artifacts = tuple(artifacts)
    artifacts = tuple(artifact for artifact in all_artifacts
                      if not task_id or str(getattr(artifact, "task_id", "") or "") == task_id)
    anchors = anchors_from_artifacts(artifacts, task_id=task_id)
    resolution = resolve_discourse_references(request, anchors, focus=focus)
    selected_ids = {
        item.anchor.artifact_id for item in resolution.resolved if item.anchor.artifact_id
    }
    for raw in focus or ():
        anchor = raw if isinstance(raw, DiscourseAnchor) else DiscourseAnchor.from_dict(raw)
        if anchor is not None and anchor.artifact_id:
            selected_ids.add(anchor.artifact_id)
    by_id = {str(getattr(artifact, "id", "") or ""): artifact for artifact in artifacts}
    prior_texts = [str(text) for text in recent_assistant if str(text or "").strip()]
    for artifact_id in selected_ids:
        artifact = by_id.get(artifact_id)
        if artifact is None:
            continue
        prior = _artifact_text(artifact)
        if prior:
            prior_texts.append(prior)
    snapshot_execution, snapshot_quality = _snapshot_queries(previous_evidence_snapshot)
    inherited_query = snapshot_execution or (
        previous_evidence_query if isinstance(previous_evidence_query, EvidenceQuery)
        else EvidenceQuery.from_dict(previous_evidence_query or {})
    )
    inherited_quality = snapshot_quality or (
        previous_quality_evidence_query
        if isinstance(previous_quality_evidence_query, QualityEvidenceQuery)
        else QualityEvidenceQuery.from_dict(previous_quality_evidence_query or {})
    )
    contract = analyze_turn(
        request, prior_texts=prior_texts, pending_proposal=pending_proposal,
        referents=resolution.resolved, previous_evidence_query=inherited_query,
        previous_quality_evidence_query=inherited_quality,
    )
    projections: tuple[dict, ...] = ()
    snapshot_basis: dict | None = None
    continuation = bool(getattr(contract, "evidence_continuation", False))
    # A corrupt record may be one of the frozen source/domain turns. Never re-materialize an exact adjacent
    # proof over only the readable survivors.
    snapshot_valid = continuation and not artifact_gaps and _valid_adjacent_snapshot(
        previous_evidence_snapshot, artifacts=all_artifacts, task_id=task_id,
        execution_query=contract.evidence_query,
        quality_query=contract.quality_evidence_query,
        current_generation=current_generation,
    )
    if continuation:
        source_turn_id = str((previous_evidence_snapshot or {}).get("source_turn_id") or "")
        materialized = _materialize_frozen_snapshot(
            previous_evidence_snapshot or {}, artifacts=all_artifacts,
            execution_query=contract.evidence_query,
            quality_query=contract.quality_evidence_query,
        ) if snapshot_valid else None
        if materialized is not None:
            frozen_refs, projections = materialized
            snapshot_basis = dict((previous_evidence_snapshot or {}).get("basis") or {})
            contract = replace(contract, referents=(*contract.referents, *frozen_refs, {
                "kind": "evidence_snapshot", "status": "frozen",
                "source_turn_id": source_turn_id,
            }))
            # The assessment response itself is the claim under verification. Retain that exact artifact, while
            # immutable underlying evidence stays addressable through the frozen source handles in the basis.
            if source_turn_id:
                selected_ids.add(source_turn_id)
        else:
            if contract.evidence_query is not None:
                contract = replace(contract, referents=(*contract.referents, {
                    "kind": "execution_receipt_absence", "task_id": task_id,
                    "query": contract.evidence_query.to_dict(),
                    "reason": "frozen adjacent evidence snapshot unavailable or invalid",
                }, {
                    "kind": "execution_receipt_coverage", "coverage": "unavailable",
                    "candidate_turn_artifacts": 0, "receipt_bearing": 0,
                    "missing_receipt_count": 0, "scope": contract.evidence_query.scope,
                    "candidate_set_sha256": _digest_ids(()), "missing_set_sha256": _digest_ids(()),
                    "missing_receipt_sample": [], "corrupt_artifact_count": len(artifact_gaps),
                    "corrupt_artifact_sample": list(_artifact_gap_ids(artifact_gaps)[:3]),
                    "corrupt_set_sha256": _digest_ids(_artifact_gap_ids(artifact_gaps)),
                    "ambiguous_order_count": 0, "ambiguous_order_sample": [],
                    "ambiguous_order_set_sha256": _digest_ids(()),
                    "source_index_handle": "artifacts/index.md",
                }))
            if contract.quality_evidence_query is not None:
                projections = ({
                    "kind": "quality_exchange_coverage",
                    "query": contract.quality_evidence_query.to_dict(),
                    "coverage": "unavailable", "candidate_turn_artifacts": 0,
                    "complete_exchange_pairs": 0, "missing_exchange_count": 0,
                    "partial_response_pairs": 0,
                    "missing_exchange_sample": [], "candidate_set_sha256": _digest_ids(()),
                    "grounding_artifact_count": 0, "missing_grounding_artifact_count": 0,
                    "missing_grounding_artifact_sample": [],
                    "corrupt_artifact_count": len(artifact_gaps),
                    "corrupt_artifact_sample": list(_artifact_gap_ids(artifact_gaps)[:3]),
                    "corrupt_set_sha256": _digest_ids(_artifact_gap_ids(artifact_gaps)),
                    "ambiguous_order_count": 0, "ambiguous_order_sample": [],
                    "ambiguous_order_set_sha256": _digest_ids(()),
                    "grounding_set_sha256": hashlib.sha256(b"").hexdigest(),
                    "source_set_sha256": hashlib.sha256(b"").hexdigest(),
                    "source_index_handle": "artifacts/index.md",
                    "reason": "frozen adjacent evidence snapshot unavailable or invalid",
                },)
            contract = replace(contract, referents=(*contract.referents, {
                "kind": "evidence_snapshot", "status": "unavailable",
                "source_turn_id": source_turn_id,
            }))
    elif "execution_receipt" in contract.source_needs:
        evidence_query = contract.evidence_query or EvidenceQuery(source="execution_receipt")
        evidence_artifacts = artifacts
        if evidence_query.scope == "session" and session_id:
            evidence_artifacts = tuple(
                artifact for artifact in all_artifacts
                if str(getattr(artifact, "session_id", "") or "") == session_id
            )
        receipt_refs = execution_receipts_from_artifacts(evidence_query, evidence_artifacts)
        aggregate = aggregate_execution_receipts_from_artifacts(evidence_query, evidence_artifacts)
        coverage = execution_receipt_coverage(
            evidence_artifacts, evidence_query, gaps=artifact_gaps,
        )
        scoped_execution_artifacts = _artifacts_in_query_scope(evidence_query, evidence_artifacts)
        if aggregate is not None:
            contract = replace(contract, referents=(
                *contract.referents,
                aggregate,
                *receipt_refs,
                coverage,
            ))
            selected_ids.update(
                str(ref.get("artifact_id") or "") for ref in receipt_refs if ref.get("artifact_id")
            )
        else:
            # No receipt-bearing turn exists in the selected scope. This is unavailable source evidence,
            # distinct from a complete receipt scan whose predicate matched zero operations.
            contract = replace(contract, referents=(
                *contract.referents,
                {
                    "kind": "execution_receipt_absence", "task_id": task_id,
                    "query": evidence_query.to_dict(),
                },
                coverage,
            ))
        execution_refs = tuple(
            ref for ref in contract.referents
            if isinstance(ref, Mapping)
            and str(ref.get("kind") or "") in _EXECUTION_EVIDENCE_KINDS
        )
        snapshot_basis = {
            "execution_artifact_ids": [
                str(getattr(item, "id", "") or "") for item in scoped_execution_artifacts
                if str(getattr(item, "kind", "") or "") == "turn"
            ],
            "quality_artifact_ids": [],
            "quality_grounding_artifact_ids": [],
            "execution_signature": _execution_projection_signature(execution_refs),
            "quality_signature": {},
        }
    if not continuation and contract.quality_evidence_query is not None:
        quality_artifacts = artifacts
        if contract.quality_evidence_query.scope == "session" and session_id:
            quality_artifacts = tuple(
                artifact for artifact in all_artifacts
                if str(getattr(artifact, "session_id", "") or "") == session_id
            )
        projections = quality_exchanges_from_artifacts(
            contract.quality_evidence_query, quality_artifacts, grounding_artifacts=all_artifacts,
            gaps=artifact_gaps,
        )
        if snapshot_basis is None:
            snapshot_basis = {
                "execution_artifact_ids": [], "quality_artifact_ids": [],
                "quality_grounding_artifact_ids": [],
                "execution_signature": {}, "quality_signature": {},
            }
        scoped_quality_artifacts = _quality_scope_artifacts(
            contract.quality_evidence_query, quality_artifacts,
        )
        snapshot_basis["quality_artifact_ids"] = [
            str(getattr(item, "id", "") or "") for item in scoped_quality_artifacts
        ]
        snapshot_basis["quality_grounding_artifact_ids"] = list(
            _quality_grounding_artifact_ids(scoped_quality_artifacts)
        )
        snapshot_basis["quality_signature"] = _quality_projection_signature(projections)
    target = (
        repair.replacement if repair is not None else
        explicit_subject if explicit_subject is not None else
        replace(inherited_subject, source="focus", source_span=None)
        if inherited_subject is not None else None
    )
    contract = replace(
        contract,
        target=target,
        focus_repairs=(repair,) if repair is not None else (),
    )
    selected_option = _selected_pending_option(request, pending_proposal)
    if pending_proposal and (contract.effect_authority == "continuation" or selected_option is not None):
        contract = replace(contract, referents=(*contract.referents, {
            "kind": "pending_proposal", **dict(pending_proposal),
            **({"selected_option": selected_option} if selected_option else {}),
        }))
    if resolution.ambiguous:
        contract = replace(
            contract, effect_authority="uncertain",
            requested_modes=(*contract.requested_modes, "clarify_reference"),
        )
    if resolution.grounding != "none" and contract.grounding == "none":
        contract = replace(contract, grounding=resolution.grounding)
    elif resolution.grounding != "none" and resolution.grounding != contract.grounding:
        contract = replace(contract, grounding="both")
    # A stable task subject survives ordinary turns; a temporary explicit self-reference does not erase a
    # different project focus. An explicit named project or an ``I mean …`` repair replaces it immediately.
    primary_subject = repair.replacement if repair is not None else explicit_subject
    if (primary_subject is not None and primary_subject.kind == "agent"
            and inherited_subject is not None and inherited_subject.kind != "agent"):
        primary_subject = inherited_subject
    if primary_subject is None:
        primary_subject = inherited_subject
    new_focus = tuple(item.anchor.to_dict() for item in resolution.resolved)
    if primary_subject is not None:
        new_focus = (*new_focus, {"kind": "subject_focus", "entity": primary_subject.to_dict()})
    return AdmissionPreview(
        admission=contract,
        focus=new_focus,
        referenced_artifact_ids=tuple(sorted(artifact_id for artifact_id in selected_ids if artifact_id)),
        projections=projections,
        snapshot_basis=snapshot_basis,
        ambiguous=resolution.ambiguous,
    )


__all__ = [
    "AdmissionPreview", "DiscourseAnchor", "ResolvedAnchor", "ResolutionResult", "TurnInterpretation",
    "anchors_from_artifacts",
    "aggregate_execution_receipts",
    "aggregate_execution_receipts_from_artifacts",
    "execution_receipt_coverage",
    "execution_receipts_from_artifacts",
    "make_evidence_snapshot", "quality_exchanges_from_artifacts",
    "extract_addressable_anchors", "extract_pending_proposal", "render_resolved_references",
    "interpret_turn", "resolve_discourse_references",
]
