"""Typed user-intent state.

The active intent ledger is semantic state, not a copy of the conversation.  The
current request is kept verbatim for the running turn; earlier clauses stay
resident only when they were deliberately recorded as standing obligations.

This module is deliberately pure: it owns no rendering, tools, or persistence.
Those layers serialize/project these records without becoming a second mutable
authority.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal


IntentStatus = Literal[
    "active",
    "provisionally_satisfied",
    "satisfied",
    "superseded",
    "deferred",
]
IntentAuthority = Literal["user", "task", "legacy"]
IntentKind = Literal["constraint", "correction"]


# High-precision host capture for clauses whose language explicitly says they remain binding. This is not
# a general intent classifier: ordinary follow-ups stay in current_request/recent continuity, while durable
# constraints survive even if a model forgets to call require(). Exact source text is retained verbatim.
_EXPLICIT_CONSTRAINT = re.compile(
    r"(?:\b(?:must(?:n't|\s+not)?|never|do\s+not|don't|cannot|can't|preserve|required?|ensure|"
    r"make\s+sure|exactly)\b"
    r"|\bonly\s+(?:change|modify|edit|touch|use|return|output|write|call|support|include|exclude)\b"
    r"|\bwithout\s+(?:changing|modifying|editing|removing|adding|breaking)\b"
    r"|\bleave\b.{0,80}\b(?:unchanged|alone|intact)\b"
    r"|\bkeep\b.{0,80}\b(?:unchanged|stable|compatible|intact|the\s+same)\b"
    r"|^\s*(?:please\s+)?(?:use|target)\b)",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(r"(?:\r?\n+|;+\s*|(?<=[.!?])\s+)")
_CLAUSE_PREFIX = re.compile(r"(?:[-*•]\s+|\d+[.)]\s+)")
_EXPLICIT_CORRECTION = re.compile(
    r"\b(?:correction|correcting|instead|actually|no\s+longer|change\s+of\s+plan|replace)\b",
    re.IGNORECASE,
)
_BARE_CORRECTION_HEADER = re.compile(
    r"^\s*(?:correction|correcting|change\s+of\s+plan|instead|actually|replace)\s*:?[\s-]*$",
    re.IGNORECASE,
)
_STRONG_CORRECTION = re.compile(
    r"\b(?:correction|correcting|instead|no\s+longer|change\s+of\s+plan|replace)\b",
    re.IGNORECASE,
)
_DIRECTIVE_PREFIX = re.compile(
    r"^\s*(?:(?:actually|correction|correcting)\s*:?,?\s*)?(?:please\s+)?"
    r"(?:use|target|return|switch|change|make|build|create|add|remove|write|output|implement|fix|"
    r"refactor|modif(?:y|ying))\b",
    re.IGNORECASE,
)
_ACTUALLY_PREFIX = re.compile(r"^\s*actually\b", re.IGNORECASE)
_REVERSAL_LANGUAGE = re.compile(
    r"\b(?:allowed|permitted|forbidden|breaking\s+changes?|can\s+(?:change|modify|break)|"
    r"may\s+(?:change|modify|break))\b",
    re.IGNORECASE,
)
_ADDITIVE_LANGUAGE = re.compile(r"\b(?:also|too|in\s+addition|as\s+well)\b", re.IGNORECASE)
_FENCE_START = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\r\n]*$")
_LAZY_QUOTE_INTERRUPT = re.compile(
    r"^ {0,3}(?:[-+*]\s+|\d{1,9}[.)]\s+|#{1,6}(?:\s+|$)|(?:[*_-]\s*){3,}$|<[^>]+>)"
)
_PLAIN_QUOTED_DATA = (
    re.compile(r'"(?:\\.|[^"\\\r\n])*"'),
    re.compile(r"“[^”\r\n]*”"),
    re.compile(r"‘[^’\r\n]*’"),
)
_QUESTION_PREFIX = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|should|would|could|can|why|how|what|when|where|who)\b",
    re.IGNORECASE,
)
_POLITE_DIRECTIVE_QUESTION = re.compile(
    r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:ensure|make\s+sure|keep|preserve|"
    r"never|do\s+not|don't|use|target|return|implement|fix|add|remove|write|change|modify)\b",
    re.IGNORECASE,
)

# Linkage ignores grammatical/modal scaffolding, but deliberately does not stem words or compare embeddings.
# The fallback below therefore remains deterministic and conservative: an explicit correction must retain at
# least two content-bearing words and cover most of both the old clause and its captured replacement.
_LINKAGE_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "we", "us", "our", "you", "your", "it", "its",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "to", "of", "for", "in", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "if", "then",
    "must", "should", "shall", "may", "might", "can", "could", "would",
    "not", "no", "never", "longer",
    "actually", "correction", "correcting", "instead", "change", "plan", "replace",
    "require", "required", "requires",
    "preserve", "keep", "ensure", "make", "sure", "leave", "only", "use", "target",
})


def _phrase_tokens(text: str) -> tuple[str, ...]:
    """Punctuation-insensitive exact-word projection used only for explicit correction linkage."""
    return tuple(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def _contains_phrase(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[index:index + len(needle)] == needle
               for index in range(len(haystack) - len(needle) + 1))


def _significant_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token for token in _phrase_tokens(text)
        if len(token) >= 3 and token not in _LINKAGE_STOPWORDS
    )


def _concrete_values(text: str) -> frozenset[str]:
    """Concrete numeric/version values that make a one-subject replacement unambiguous."""
    return frozenset(re.findall(r"\d+(?:\.\d+)+|\d+", str(text or "").casefold()))


def _directive_action(text: str) -> str:
    """Normalize only an explicit leading action verb; this is not semantic/fuzzy matching."""
    cleaned = re.sub(
        r"^\s*(?:(?:actually|correction|correcting)\s*:?,?\s*)?"
        r"(?:(?:you\s+)?(?:must|should|shall)\s+)?(?:do\s+not\s+|never\s+)?",
        "", str(text or ""), flags=re.IGNORECASE,
    )
    match = re.match(
        r"(use|target|return|switch|change|make|build|create|add|remove|write|output|implement|fix|"
        r"refactor|modif(?:y|ying))\b",
        cleaned, re.IGNORECASE,
    )
    if match is None:
        return ""
    action = match.group(1).casefold()
    return "modify" if action in {"modify", "modifying"} else action


def _correction_match_score(old_text: str, replacement_text: str) -> int:
    """Rank a deterministic correction link; callers reject ties instead of guessing.

    Exact word-sequence containment remains the strongest path. The fallback is set identity after removing
    grammatical/correction scaffolding, a shared subject plus changed concrete value, or a parallel leading
    action with a shared subject. This is called only after the explicit-correction cue gate.
    """
    if _ADDITIVE_LANGUAGE.search(replacement_text):
        return 0
    old_words = _phrase_tokens(old_text)
    replacement_words = _phrase_tokens(replacement_text)
    if _contains_phrase(replacement_words, old_words):
        return 1000
    old_significant = _significant_tokens(old_text)
    replacement_significant = _significant_tokens(replacement_text)
    if len(old_significant) >= 2 and old_significant == replacement_significant:
        return 900
    old_values = _concrete_values(old_text)
    replacement_values = _concrete_values(replacement_text)
    if (
        len(old_significant) == 1
        and old_significant == replacement_significant
        and bool(old_values and replacement_values and old_values != replacement_values)
    ):
        return 850
    old_action = _directive_action(old_text)
    replacement_action = _directive_action(replacement_text)
    if old_action and old_action == replacement_action:
        old_subject = set(old_significant)
        replacement_subject = set(replacement_significant)
        for action_token in ({"modify", "modifying"} if old_action == "modify" else {old_action}):
            old_subject.discard(action_token)
            replacement_subject.discard(action_token)
        shared = old_subject.intersection(replacement_subject)
        return 600 + len(shared) if shared else 0
    return 0


def _match_key(text: str) -> str:
    """Compatibility matcher for model-supplied requirement text.

    The first spelling is retained verbatim.  Matching remains whitespace- and
    case-insensitive so the existing requirement_done/drop tool contract does
    not become brittle during migration.
    """
    return " ".join(str(text or "").split()).casefold()


def _mask_quoted_data(text: str) -> str:
    """Same-length instruction scan with Markdown data regions blanked.

    Source ranges still index the untouched request. Fenced code, blockquotes, and inline code are evidence
    supplied by the user, not automatically promoted into higher-authority standing directives.
    """
    source = str(text or "")
    masked = list(source)
    fence_char = ""
    fence_size = 0
    lazy_quote = False

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            if masked[index] not in "\r\n":
                masked[index] = " "

    offset = 0
    for line in source.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        end = offset + len(line)
        if fence_char:
            blank(offset, end)
            # CommonMark closing fences use the same character, at least the opening length, up to three
            # leading spaces, and no info string. `````not-a-close`` therefore remains fenced data.
            if re.fullmatch(rf" {{0,3}}{re.escape(fence_char)}{{{fence_size},}}[ \t]*", body):
                fence_char = ""; fence_size = 0
            offset = end
            continue
        opened = _FENCE_START.fullmatch(body)
        if opened is not None:
            marker = opened.group(1)
            fence_char, fence_size = marker[0], len(marker)
            blank(offset, end)
            lazy_quote = False
            offset = end
            continue

        stripped = body.lstrip(" ")
        indent = len(body) - len(stripped)
        is_quote = indent <= 3 and stripped.startswith(">")
        if is_quote:
            blank(offset, end)
            lazy_quote = bool(stripped[1:].strip())
            offset = end
            continue
        if lazy_quote:
            if not body.strip():
                lazy_quote = False
            elif _LAZY_QUOTE_INTERRUPT.match(body):
                # Lists/headings/thematic/HTML block starts interrupt a CommonMark lazy paragraph. They are
                # outside the quote and may carry a real standing directive, so resume normal scanning.
                lazy_quote = False
            else:
                # A blockquote paragraph may continue lazily on following nonblank lines without another `>`.
                blank(offset, end)
                offset = end
                continue
        if body.startswith("    ") or body.startswith("\t"):
            blank(offset, end)  # indented Markdown code block
        offset = end

    # Code spans may legally cross line boundaries. Match a closing run of exactly the opening length;
    # a longer adjacent run is a different delimiter. Fenced/quoted ranges are already spaces here.
    scan = "".join(masked)
    index = 0
    while index < len(scan):
        if scan[index] != "`":
            index += 1
            continue
        opened = index
        while index < len(scan) and scan[index] == "`":
            index += 1
        width = index - opened
        cursor = index
        closed = -1
        while cursor < len(scan):
            cursor = scan.find("`", cursor)
            if cursor < 0:
                break
            finish = cursor
            while finish < len(scan) and scan[finish] == "`":
                finish += 1
            if finish - cursor == width:
                closed = finish
                break
            cursor = finish
        if closed < 0:
            continue
        blank(opened, closed)
        scan = "".join(masked)
        index = closed
    # Ordinary quoted strings are also attributed data. Mask only the payload; an enclosing directive such
    # as `Return exactly "OK"` remains capturable from the unquoted action words and retains exact source text.
    scan = "".join(masked)
    for pattern in _PLAIN_QUOTED_DATA:
        for match in pattern.finditer(scan):
            blank(match.start(), match.end())
        scan = "".join(masked)
    return "".join(masked)


@dataclass(frozen=True)
class IntentEntry:
    id: str
    verbatim_clause: str
    source_artifact: str | None = None
    source_range: tuple[int, int] | None = None
    authority: IntentAuthority = "task"
    kind: IntentKind = "constraint"
    status: IntentStatus = "active"
    superseded_by: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "verbatim_clause": self.verbatim_clause,
            "source_artifact": self.source_artifact,
            "source_range": list(self.source_range) if self.source_range is not None else None,
            "authority": self.authority,
            "kind": self.kind,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "IntentEntry | None":
        if not isinstance(value, Mapping):
            return None
        clause = value.get("verbatim_clause")
        if not isinstance(clause, str) or not clause.strip():
            return None
        raw_range = value.get("source_range")
        source_range = None
        if (isinstance(raw_range, (list, tuple)) and len(raw_range) == 2
                and all(isinstance(n, int) for n in raw_range)):
            source_range = (raw_range[0], raw_range[1])
        authority = value.get("authority")
        if authority not in ("user", "task", "legacy"):
            authority = "legacy"
        kind = value.get("kind")
        if kind not in ("constraint", "correction"):
            kind = "constraint"
        status = value.get("status")
        if status not in ("active", "provisionally_satisfied", "satisfied", "superseded", "deferred"):
            status = "active"
        refs = value.get("evidence_refs")
        return cls(
            id=str(value.get("id") or ""),
            verbatim_clause=clause,
            source_artifact=(str(value["source_artifact"])
                             if value.get("source_artifact") is not None else None),
            source_range=source_range,
            authority=authority,
            kind=kind,
            status=status,
            superseded_by=(str(value["superseded_by"])
                           if value.get("superseded_by") is not None else None),
            evidence_refs=tuple(str(r) for r in refs) if isinstance(refs, (list, tuple)) else (),
        )


@dataclass
class IntentState:
    current_request: str = ""
    current_source: str | None = None
    entries: list[IntentEntry] = field(default_factory=list)
    next_id: int = 1

    def reset(self, request: str = "", *, source_artifact: str | None = None) -> None:
        self.current_request = str(request or "")
        self.current_source = source_artifact
        self.entries = []
        self.next_id = 1

    def begin_turn(self, request: str, *, source_artifact: str | None = None) -> None:
        """Install the one authoritative, verbatim request for this turn."""
        self.current_request = str(request or "")
        self.current_source = source_artifact
        prior = tuple(self.resident_entries())
        captured = self.capture_explicit_constraints(prior)
        self.reconcile_explicit_corrections(prior, captured)

    def reconcile_explicit_corrections(
        self, prior: Iterable[IntentEntry], captured: Iterable[IntentEntry],
    ) -> tuple[IntentEntry, ...]:
        """Link a high-confidence user correction to the exact older clause it names.

        The host does not guess semantic similarity. Supersession fires only when the current request uses
        explicit correction language and a newly captured exact constraint either repeats the older clause's
        word sequence or conservatively overlaps its significant words. More implicit rewrites remain
        available through ``supersede_requirement``.
        """
        if _EXPLICIT_CORRECTION.search(_mask_quoted_data(self.current_request)) is None:
            return ()
        # A correction cue elsewhere in the message (especially inside quoted evidence) cannot lend
        # supersession authority to an unrelated directive. Only the exact captured clause whose own masked
        # text was classified as a correction may retire an older user obligation.
        replacements = tuple(entry for entry in captured if entry.kind == "correction")
        candidates = [entry for entry in prior
                      if entry.status in ("active", "provisionally_satisfied")]
        changed = []
        used: set[str] = set()
        for replacement in replacements:
            scored = [
                (_correction_match_score(old.verbatim_clause, replacement.verbatim_clause), old)
                for old in candidates if old.id != replacement.id and old.id not in used
            ]
            best = max((score for score, _old in scored), default=0)
            matches = [old for score, old in scored if score == best and score > 0]
            # One replacement may retire at most one uniquely best older clause. A shared action verb alone
            # cannot delete several independent requirements; ambiguity keeps both until an explicit tool/user
            # transition names the target.
            if len(matches) != 1:
                continue
            old = matches[0]
            # Once linked, the replacement owns the superseded obligation and becomes a standing constraint.
            # Unlinked correction text stays a distinct clarification rather than being guessed into one.
            replacement = self._replace(replacement, kind="constraint")
            self._replace(old, status="superseded", superseded_by=replacement.id)
            used.add(old.id)
            changed.append(replacement)
        return tuple(changed)

    def capture_explicit_constraints(
        self, prior: Iterable[IntentEntry] = (),
    ) -> tuple[IntentEntry, ...]:
        """Promote explicit durable clauses without copying the whole historical message."""
        text = self.current_request
        scan = _mask_quoted_data(text)
        prior_significant = tuple(
            _significant_tokens(entry.verbatim_clause) for entry in prior
            if entry.status in ("active", "provisionally_satisfied")
        )
        captured = []
        pending_correction_header = False
        start = 0
        spans = []
        for boundary in _CLAUSE_BOUNDARY.finditer(scan):
            spans.append((start, boundary.start()))
            start = boundary.end()
        spans.append((start, len(scan)))
        for raw_start, raw_end in spans:
            left = raw_start
            while left < raw_end and scan[left].isspace():
                left += 1
            prefix = _CLAUSE_PREFIX.match(scan, left, raw_end)
            if prefix is not None:
                left = prefix.end()
            right = raw_end
            while right > left and scan[right - 1].isspace():
                right -= 1
            clause = text[left:right]
            scan_clause = scan[left:right]
            if _BARE_CORRECTION_HEADER.fullmatch(scan_clause):
                # A syntactic header scopes exactly the immediately following clause; it is not itself a
                # durable obligation. Quoted headers were blanked before this point and cannot lend authority.
                pending_correction_header = True
                continue
            scoped_correction = pending_correction_header
            if scan_clause.strip():
                pending_correction_header = False
            if (scan_clause.rstrip().endswith("?") and _QUESTION_PREFIX.match(scan_clause)
                    and not _POLITE_DIRECTIVE_QUESTION.match(scan_clause)):
                # Modal words inside a question describe uncertainty, not a durable instruction. Preserve
                # polite `could you ensure ...?` requests, but do not turn `Do we need to never ...?` into law.
                continue
            clause_significant = _significant_tokens(scan_clause)
            related_actually = (
                _ACTUALLY_PREFIX.search(scan_clause) is not None
                and (
                    bool(_concrete_values(scan_clause))
                    or any(clause_significant.intersection(old) for old in prior_significant)
                    or _REVERSAL_LANGUAGE.search(scan_clause) is not None
                )
            )
            if not clause or not (
                _EXPLICIT_CONSTRAINT.search(scan_clause)
                or _STRONG_CORRECTION.search(scan_clause)
                or _DIRECTIVE_PREFIX.search(scan_clause)
                or related_actually
            ):
                continue
            entry = self.add_exact(
                clause, source_artifact=self.current_source,
                source_range=(left, right), authority="user",
                kind=("correction" if scoped_correction or _EXPLICIT_CORRECTION.search(scan_clause)
                      else "constraint"),
            )
            if entry is not None:
                captured.append(entry)
        return tuple(captured)

    def seal(self) -> None:
        """Intent is task-scoped; explicit transitions, not elapsed turns, retire it."""
        return None

    def _mint_id(self) -> str:
        live = {entry.id for entry in self.entries}
        while True:
            value = f"intent-{self.next_id}"
            self.next_id += 1
            if value not in live:
                return value

    def find(self, text: str, *, statuses: Iterable[str] | None = None) -> IntentEntry | None:
        key = _match_key(text)
        allowed = set(statuses) if statuses is not None else None
        for entry in self.entries:
            if _match_key(entry.verbatim_clause) == key and (allowed is None or entry.status in allowed):
                return entry
        return None

    def add_exact(
        self,
        clause: str,
        *,
        source_artifact: str | None = None,
        source_range: tuple[int, int] | None = None,
        authority: IntentAuthority = "task",
        kind: IntentKind = "constraint",
    ) -> IntentEntry | None:
        """Add a standing clause without a semantic count or character cap."""
        text = str(clause or "").strip()
        if not text:
            return None
        hit = self.find(text, statuses=("active", "provisionally_satisfied", "satisfied"))
        if hit is not None:
            # Session routing may see the request before its pending artifact id is allocated. When the
            # journal-backed record arrives, enrich the same authoritative entry rather than duplicating it.
            if (source_artifact is not None and hit.source_artifact is None
                    and hit.authority == authority):
                return self._replace(hit, source_artifact=source_artifact,
                                     source_range=source_range or hit.source_range,
                                     kind=(kind if hit.kind == "correction" else hit.kind))
            return hit
        entry = IntentEntry(
            id=self._mint_id(),
            verbatim_clause=text,
            source_artifact=source_artifact,
            source_range=source_range,
            authority=authority,
            kind=kind,
        )
        self.entries.append(entry)
        return entry

    def add_from_current_request(self, clause: str) -> IntentEntry | None:
        """Record a tool-supplied clause, preserving whether it is an exact user quote.

        A model-authored paraphrase remains useful task state, but it must not be
        mislabeled as user-authored intent.  Exact substring membership supplies
        the source range without an extra model call.
        """
        text = str(clause or "").strip()
        if not text:
            return None
        start = self.current_request.find(text)
        return self.add_exact(
            text,
            source_artifact=self.current_source,
            source_range=(start, start + len(text)) if start >= 0 else None,
            authority="user" if start >= 0 else "task",
        )

    def _replace(self, old: IntentEntry, **changes) -> IntentEntry:
        new = replace(old, **changes)
        self.entries[self.entries.index(old)] = new
        return new

    def mark_provisional(self, text: str, *, evidence_refs: Iterable[str] = ()) -> IntentEntry | None:
        hit = self.find(text, statuses=("active", "provisionally_satisfied"))
        if hit is None:
            return None
        refs = tuple(str(r) for r in evidence_refs if r)
        if not refs:
            return None  # model assertion alone is not verification evidence
        return self._replace(hit, status="provisionally_satisfied", evidence_refs=refs)

    def satisfy(self, text: str, *, evidence_refs: Iterable[str] = ()) -> IntentEntry | None:
        """Finalize satisfaction; callers must enforce the user/task-boundary policy."""
        hit = self.find(text, statuses=("active", "provisionally_satisfied"))
        if hit is None:
            return None
        refs = tuple(str(r) for r in evidence_refs if r) or hit.evidence_refs
        return self._replace(hit, status="satisfied", evidence_refs=refs)

    def defer_model_entry(self, text: str) -> IntentEntry | None:
        """Allow model-owned task state to leave the active set.

        User-authored entries deliberately cannot take this transition.  A model
        calling drop_requirement is not evidence that the user retracted it.
        """
        hit = self.find(text, statuses=("active", "provisionally_satisfied"))
        if hit is None or hit.authority == "user":
            return None
        return self._replace(hit, status="deferred")

    def supersede_from_user(
        self,
        old_text: str,
        new_clause: str,
        *,
        source_artifact: str | None = None,
        source_range: tuple[int, int] | None = None,
    ) -> IntentEntry | None:
        old = self.find(old_text, statuses=("active", "provisionally_satisfied", "superseded"))
        if old is None:
            return None
        # begin_turn may already have retained a wrapper such as ``Correction: use API v2 instead`` so the
        # correction cannot disappear before the model calls this transition. Canonicalize that same entry
        # to the exact replacement substring instead of creating a redundant second authority record.
        wrapper = next((entry for entry in self.resident_entries()
                        if entry.authority == "user" and entry.id != old.id
                        and new_clause in entry.verbatim_clause
                        and (source_artifact is None or entry.source_artifact in (None, source_artifact))), None)
        if wrapper is None and old.status == "superseded" and old.superseded_by:
            wrapper = next((entry for entry in self.entries
                            if entry.id == old.superseded_by and new_clause in entry.verbatim_clause), None)
        if wrapper is not None:
            new = self._replace(
                wrapper, verbatim_clause=new_clause,
                source_artifact=source_artifact or wrapper.source_artifact,
                source_range=source_range or wrapper.source_range,
                kind="constraint",
            )
        else:
            new = self.add_exact(
                new_clause,
                source_artifact=source_artifact,
                source_range=source_range,
                authority="user",
            )
        if new is None:
            return None
        if old.status != "superseded" or old.superseded_by != new.id:
            self._replace(old, status="superseded", superseded_by=new.id)
        return new

    def resident_entries(self) -> list[IntentEntry]:
        return [entry for entry in self.entries
                if entry.status in ("active", "provisionally_satisfied")]

    def open_entries(self) -> list[IntentEntry]:
        return [entry for entry in self.entries if entry.status == "active"]

    def as_legacy_requirements(self) -> list[dict]:
        """Read-only compatibility projection for old call sites/checkpoints."""
        return [
            {
                "text": entry.verbatim_clause,
                "done": entry.status in ("provisionally_satisfied", "satisfied"),
            }
            for entry in self.resident_entries()
            if entry.kind == "constraint"
        ]

    def load_legacy_requirements(self, requirements: Iterable[Mapping]) -> None:
        """Replace entries from a v1 requirement list without upgrading authority."""
        self.entries = []
        self.next_id = 1
        for item in requirements or ():
            if not isinstance(item, Mapping):
                continue
            entry = self.add_exact(str(item.get("text") or ""), authority="legacy")
            if entry is not None and item.get("done"):
                self._replace(entry, status="provisionally_satisfied")

    def to_records(self) -> list[dict]:
        return [entry.to_dict() for entry in self.entries]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping],
        *,
        current_request: str = "",
        current_source: str | None = None,
        next_id: int = 1,
    ) -> "IntentState":
        state = cls(current_request=str(current_request or ""), current_source=current_source)
        for index, raw in enumerate(records or ()):
            entry = IntentEntry.from_dict(raw)
            if entry is None:
                raise ValueError(f"invalid intent record at index {index}")
            if not entry.id:
                raise ValueError(f"intent record at index {index} has no id")
            if any(old.id == entry.id for old in state.entries):
                # A duplicate source id makes supersession references inherently ambiguous.  Guessing a new
                # id here silently rewires authority; authoritative v2 state must fail visibly instead.
                raise ValueError(f"duplicate intent id {entry.id!r}")
            state.entries.append(entry)

        by_id = {entry.id: entry for entry in state.entries}
        for entry in state.entries:
            target = entry.superseded_by
            if entry.status == "superseded":
                if not target:
                    raise ValueError(f"superseded intent {entry.id!r} has no replacement link")
                if target == entry.id:
                    raise ValueError(f"intent {entry.id!r} supersedes itself")
                if target not in by_id:
                    raise ValueError(
                        f"intent {entry.id!r} points to missing replacement {target!r}"
                    )
            elif target is not None:
                raise ValueError(
                    f"non-superseded intent {entry.id!r} carries replacement link {target!r}"
                )

        # Chained corrections are valid; cycles are not.  Validate after every id is known so record order
        # cannot change the result.
        for entry in state.entries:
            seen = {entry.id}
            cursor = entry
            while cursor.superseded_by:
                target = cursor.superseded_by
                if target in seen:
                    raise ValueError(f"cyclic intent supersession involving {target!r}")
                seen.add(target)
                cursor = by_id[target]
        state.next_id = max(int(next_id or 1), state.next_id)
        # Avoid reusing numeric ids from a restored state even if next_id was absent.
        for entry in state.entries:
            if entry.id.startswith("intent-") and entry.id[7:].isdigit():
                state.next_id = max(state.next_id, int(entry.id[7:]) + 1)
        return state
