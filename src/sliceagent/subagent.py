"""Subagents — bounded delegation on the slice architecture.

A large, decomposable task can be split: the parent spawns a CHILD agent for a
sub-task; the child runs its own loop with a FRESH slice, does the work in the SAME
workspace, and returns ONLY a compact summary. The parent's slice never sees the
child's transcript — just the summary — so parent context is bounded by that summary,
not by the child's raw work-volume. That's the slice thesis applied recursively.

Exposed as ONE tool (`spawn_agent`, agent=<kind>) via a ToolHost wrapper, so the loop is
unchanged: from the parent loop's view it's one tool call that returns a summary string.
(The former `spawn_explore` / `spawn_subagent` were just agent="explorer" / "general";
collapsed after measuring parallel-fan-out parity — run() still recognises the old names.)
A named spawn HIRES a standing specialist; an unnamed one is a one-shot temp. The child is
depth-capped (a child can't spawn grandchildren by default) and runs under the same
permission policy. Tool execution and reads delegate to the wrapped (real) ToolHost, so
parent and child share one workspace and one sandbox.
"""
from __future__ import annotations

import copy
import hashlib
import os
import posixpath
import re
import threading

from .access import AllAccess, ReadAllAccess
from .agents import BUILTIN_AGENTS, READ_ONLY_TOOLS, SUBAGENT_EXCLUDED_TOOLS, AgentSpec  # named-agent registry
from .context import ResourceKind, ResourceRef, reserved_resource_ref
from .events import AssistantText, ToolResult, ToolStarted
from .execution import CHILD_TOKEN_BUDGET_ARG
from .safety import redact_text
from .subagent_contract import (SubagentArtifact, SubagentBrief, SubagentClaim, SubagentObservation,
                                exact_intent_clauses)
from .text_utils import one_line

# INSTANCE identity — an optional short name the parent gives ONE delegation ("auth-explorer"). Distinct
# from the KIND (the AgentSpec): the kind is the job description, the name is the employee. A named seal
# is addressable as subagents/<name>.md (latest job by that name) in the roster manifest, so the parent
# can refer to work by WHO did it, not just an ordinal. Validation is strict: it becomes a virtual-FS
# leaf, so path chars are out, and it must not shadow the canonical handles (sub-N) or index.md.
_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")
_RESERVED_NAME = re.compile(r"^(sub-\d+|index|history|subagents|roster)$", re.IGNORECASE)

_NAME_PARAM = {
    "type": "string",
    "description": ("optional stable identity for this delegation (e.g. 'auth-explorer'); names its sealed "
                    "report in the DELEGATED WORK roster (subagents/<name>.md). Re-using a name later means "
                    "'the same specialist' — its latest report lives at that address."),
}


def _valid_instance_name(name: str) -> bool:
    return bool(_VALID_NAME.match(name)) and not _RESERVED_NAME.match(name)


# CAPABILITY GRANTS — the governed handle channel (v3.5). The parent wires child A's output to child B by
# granting B the EXACT address of A's sealed artifact; the payload flows archive→B without transiting the
# parent's context (parent cost = O(edges), not O(payloads)). Rules that keep "children couple only through
# seals" true: exact file handles only (never a dir or index.md), spawn-time existence validation + a hard
# cap (the kernel can say no), and NO transitive propagation — only the PARENT mints grants.
_MAX_GRANTS = 16
_GRANT_SUB = re.compile(r"^sub-\d+\.md$")

_GRANTS_PARAM = {
    "type": "array", "items": {"type": "string"},
    "description": ("optional: EXACT sealed-report handles this child may read as INPUT (e.g. "
                    "[\"subagents/sub-1.md\", \"subagents/auth-explorer.md\"]) — hand a sibling's full "
                    "report to the child without pasting it into the task."),
}

_SCOPE_PARAM = {
    "type": "array", "items": {"type": "string"},
    "description": "optional concrete areas/files/questions that are in scope for this delegation",
}
_EXCLUSIONS_PARAM = {
    "type": "array", "items": {"type": "string"},
    "description": "optional explicit exclusions; the child reports rather than crossing them",
}
_REPORT_SHAPE_PARAM = {
    "type": "string",
    "description": "optional exact shape/content required from the child's sealed report",
}
_DRIFT_POLICY_PARAM = {
    "type": "string", "enum": ["report", "fail", "ignore"],
    "description": "how the parent should treat later drift in files fingerprinted by the child",
}


def _norm_vpath(path) -> str:
    """CANONICAL virtual-namespace path ('./subagents\\sub-1.md/' -> 'subagents/sub-1.md'). posixpath.normpath
    collapses '..' and '.' SEGMENTS — load-bearing for every prefix-based guard downstream: without it,
    'roster/<own>/../other/job-1.md' passes an own-namespace prefix check and the mounted FS then normalizes
    it into ANOTHER specialist's file (guard and FS must normalize identically, or the gap between them is
    a traversal)."""
    p = (path or "").strip().replace("\\", "/") if isinstance(path, str) else ""
    if not p:
        return ""
    p = posixpath.normpath(p)
    return "" if p == "." else p.rstrip("/")


def _task_mentions_exact_target(task: str, target: str) -> bool:
    """Match a typed target in free task prose without basename-substring or sentence-punctuation errors."""
    surface = str(task or "").replace("\\", "/")
    needle = str(target or "").strip().replace("\\", "/")
    if not needle:
        return False
    # Left boundary rejects `a.py` inside `data.py` while still allowing an absolute `/workspace/a.py` suffix.
    # The right boundary accepts ordinary punctuation, including a sentence-final extra period (`app.py.`), but
    # rejects path continuations such as `app.py.bak`, `app.py/child`, and identifier suffixes.
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.-])" + re.escape(needle)
        + r"(?=$|[^A-Za-z0-9_./-]|\.(?=$|[^A-Za-z0-9_./-]))"
    )
    return pattern.search(surface) is not None


# NOTE: the former spawn_explore / spawn_subagent tool schemas are GONE — spawn_agent (built per-host in
# SubagentHost._agent_schema) subsumes both (they were just agent="explorer" / agent="general"); measured
# parity on parallel fan-out (evals/eval_spawn_breadth_ab.py). run() still RECOGNISES those two names for
# back-compat (an old cached prompt or a stale caller), routing them to the explorer / general kinds.

# Tools a READ-ONLY child may see. NO run_command/execute_code: the policy layer can't
# guarantee a side-effect-free shell, so they are deferred (plan sec 6 defer). spawn_subagent
# is absent by construction — a read-only child cannot recurse into a writable one.
_READ_ONLY_TOOLS = frozenset(READ_ONLY_TOOLS)   # the explorer allowlist — single source of truth in agents.py

# An EXPLORER's whole job is read-N-files-then-summarize over a SHORT, bounded turn: every file it
# reads is relevant to its one summary, so the working-set eviction (READ_BUDGET) has NO benefit and
# actively BREAKS it — evicted files get re-read (refault), which the anti-loop guard flags as
# no-progress, and the child goes "stuck" before it can summarize. So an explorer keeps its whole
# exploration resident: a generous, still-bounded read budget. (The parent only ever gets the child's
# summary, so this never reaches the parent slice — the moat is unaffected.)
EXPLORER_READ_BUDGET = 64

# EXPLORER PROFILE — explorers navigate/read, but they also make evidence-sensitive judgments. Those
# judgments are exactly where cheap/minimal reasoning amplified a child's speculation into the parent's
# asserted fact, so the truthful default is the provider's full profile. Latency-sensitive users can still
# opt into "fast" explicitly. A per-profile setting is applied via a per-child LLM view so parallel siblings
# never mutate the shared parent client.
EXPLORER_REASONING = (os.environ.get("AGENT_EXPLORER_REASONING") or "full").lower()


def _redact_archive_value(value):
    """Redact both values and structural keys before canonical child persistence."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {redact_text(str(key)): _redact_archive_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_archive_value(child) for child in value]
    return value


def _profile_llm(llm, reasoning):
    """The llm VIEW for a CHILD: ALWAYS a SHALLOW COPY (shares the thread-safe client + immutable config),
    never the parent object. Two isolations the shared object lacked (external review S7): (1) a child's
    model/_fellback mutation on context-overflow must not silently switch the PARENT's model — a copy makes
    those attributes child-local; (2) the parent's streaming delta sink is DISCONNECTED so child deltas never
    reach the parent UI (a child's deliverable is its sealed summary, not its token stream). Reasoning is
    applied when given/differing. Copy is cheap; correctness beats the old no-op-when-matching shortcut."""
    view = copy.copy(llm)
    if reasoning:
        view.reasoning = reasoning
    view._on_delta = None   # child streaming stays OFF the parent's UI sink (summary-only seal, isolated state)
    return view


def read_only_schemas(schemas) -> list[dict]:
    """Filter a schema list down to the read-only allowlist (drops edit/shell/spawn tools)."""
    return [s for s in schemas
            if s.get("function", {}).get("name") in _READ_ONLY_TOOLS]


class _CaptureLast:
    """Sink that remembers the child's last assistant text (its own final summary)."""
    def __init__(self):
        self.text = ""

    def __call__(self, event):
        if isinstance(event, AssistantText) and event.content:
            self.text = event.content


_TRACE_MAX_LINES = 200   # bounded action trace per seal (a line is tiny; 200 covers any real child turn)

# Evidence capsule v1: returned observation VIEWS, never a child transcript or full journal.  The caps are
# deliberately small because several children may be grounded in one parent audit.  A large/read-windowed result
# remains useful for its retained bytes but is explicitly marked truncated, so it cannot prove omitted content.
_OBSERVATION_TOOLS = frozenset({"read_file", "list_files", "grep", "glob"})
_OBSERVATION_ARGS = {
    "read_file": ("path", "offset", "limit"),
    "list_files": ("path",),
    "grep": ("pattern", "path", "glob", "type", "output_mode", "context", "offset", "limit"),
    "glob": ("pattern", "path", "limit"),
}
_OBSERVATION_PER_VIEW_BYTES = 8 * 1024
_OBSERVATION_TOTAL_BYTES = 16 * 1024
_OBSERVATION_MAX_COUNT = 8
_OBSERVATION_TRUNCATION_MARKER = "\n…[sealed observation view truncated by capsule budget]…\n"


class _TraceSink:
    """W6': the child's ACTION TRACE, sealed into the artifact — one bounded line per tool result. This is
    the 'what did you actually DO?' grounding a later rehydration needs (a report states conclusions; the
    trace shows the path), without retaining any transcript: lines are locator-grade (tool + primary arg),
    not payloads."""
    def __init__(self):
        self.lines: list[str] = []
        self.dropped = 0

    def __call__(self, event):
        if isinstance(event, ToolResult):
            if len(self.lines) >= _TRACE_MAX_LINES:
                self.dropped += 1
                return
            mark = " ✗" if getattr(event, "failing", False) else ""
            self.lines.append(one_line(f"{event.name} {_primary_arg(event.args)}".strip(), 160) + mark)

    def text(self) -> str:
        t = "\n".join(self.lines)
        if self.dropped:
            t += f"\n(+{self.dropped} more action(s) not recorded)"
        return t


def _bounded_observation_view(text: str, budget: int) -> tuple[str, bool]:
    """Return a UTF-8-safe head/tail view whose encoded size never exceeds ``budget``."""
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text, False
    marker = _OBSERVATION_TRUNCATION_MARKER.encode("utf-8")
    if budget <= len(marker):
        return "", True
    remaining = budget - len(marker)
    head_n = remaining // 2
    tail_n = remaining - head_n
    head = encoded[:head_n].decode("utf-8", "ignore")
    tail = encoded[-tail_n:].decode("utf-8", "ignore")
    view = head + _OBSERVATION_TRUNCATION_MARKER + tail
    return view, True


def _tool_view_is_truncated(tool: str, view: str) -> bool:
    """Recognize the built-in tools' own explicit paging/truncation markers."""
    lowered = view.casefold()
    return bool(
        "[truncated;" in lowered
        or "paged out" in lowered
        or (tool == "read_file" and re.search(r"\+\d+\s+more", lowered))
    )


class _ObservationSink:
    """Seal a bounded typed capsule of successful workspace observations made by a child."""

    def __init__(self, resource_ref=None, canonicalize=None):
        self.items: list[SubagentObservation] = []
        self.used_bytes = 0
        self.dropped = 0
        self._seen: set[tuple] = set()
        # The default host knows whether a reserved-looking path is a virtual archive view or a real
        # workspace shadow.  Preserve that routing decision here instead of reclassifying by spelling.
        self._resource_ref = resource_ref
        self._canonicalize = canonicalize

    @staticmethod
    def _selected_args(tool: str, args: object) -> dict:
        args = args if isinstance(args, dict) else {}
        selected = {}
        for key in _OBSERVATION_ARGS.get(tool, ()):
            value = args.get(key)
            if value is None or isinstance(value, (int, float, bool)):
                if value is not None:
                    selected[key] = value
            elif isinstance(value, str):
                selected[key] = redact_text(value)
        return selected

    def __call__(self, event):
        if not isinstance(event, ToolResult) or event.name not in _OBSERVATION_TOOLS or event.failing:
            return
        status = str(
            event.status
            or getattr(getattr(getattr(event, "outcome", None), "status", None), "value", "")
            or "succeeded"
        ).casefold()
        ref, _, private = _classified_read_target(
            event.args, self._resource_ref, canonicalize=self._canonicalize, event=event,
        )
        if status != "succeeded" or ref.virtual or private:
            return
        raw = str(event.output or "")
        raw_bytes = raw.encode("utf-8")
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        args = self._selected_args(event.name, event.args)
        key = (event.name, tuple(args.items()), raw_sha256)
        if key in self._seen:
            return
        self._seen.add(key)
        if len(self.items) >= _OBSERVATION_MAX_COUNT:
            self.dropped += 1
            return
        remaining = _OBSERVATION_TOTAL_BYTES - self.used_bytes
        budget = min(_OBSERVATION_PER_VIEW_BYTES, remaining)
        if budget <= len(_OBSERVATION_TRUNCATION_MARKER.encode("utf-8")):
            self.dropped += 1
            return
        redacted_view = redact_text(raw)
        view, capsule_truncated = _bounded_observation_view(redacted_view, budget)
        view_bytes = view.encode("utf-8")
        observation = SubagentObservation(
            tool=event.name,
            args=args,
            status="succeeded",
            view=view,
            raw_sha256=raw_sha256,
            view_sha256=hashlib.sha256(view_bytes).hexdigest(),
            raw_bytes=len(raw_bytes),
            view_bytes=len(view_bytes),
            redacted=redacted_view != raw,
            truncated=capsule_truncated or _tool_view_is_truncated(event.name, raw),
        )
        self.items.append(observation)
        self.used_bytes += observation.view_bytes

    @property
    def observations(self) -> tuple[SubagentObservation, ...]:
        return tuple(self.items)

    @property
    def gaps(self) -> tuple[str, ...]:
        if not self.dropped:
            return ()
        return (f"{self.dropped} read-only child observation(s) omitted by the sealed capsule budget",)


def _primary_observation(
    observations: tuple[SubagentObservation, ...], brief: SubagentBrief,
) -> SubagentObservation | None:
    """Choose the child's best scoped primary view without treating its report as evidence."""
    if not observations:
        return None
    scoped = {str(item).replace("\\", "/").rstrip("/") for item in brief.scope}

    def rank(observation: SubagentObservation) -> tuple[int, int]:
        path = str(observation.args.get("path") or "").replace("\\", "/").rstrip("/")
        exact_scope = bool(path and any(path == item or path.endswith("/" + item) for item in scoped))
        return (0 if observation.tool == "read_file" and exact_scope else
                1 if observation.tool == "read_file" else 2,
                1 if observation.truncated else 0)

    return min(observations, key=rank)


def _parent_observation_excerpt(
    observations: tuple[SubagentObservation, ...], brief: SubagentBrief, *, limit: int = 520,
) -> str:
    """Return one bounded primary-source view beside the child's interpretive digest.

    The immutable artifact remains the refinement map. This tiny excerpt closes the immediate provenance gap:
    without it the parent sees only child prose at synthesis time, so a later observation capsule can diagnose
    laundering but cannot prevent it. Prefer a scoped file read, and keep redaction/truncation explicit.
    """
    chosen = _primary_observation(observations, brief)
    if chosen is None:
        return "primary observation: unavailable — treat the child report as unverified inference"
    path = str(chosen.args.get("path") or "(workspace)")
    flags = []
    if chosen.redacted:
        flags.append("redacted")
    if chosen.truncated:
        flags.append("truncated")
    suffix = f"; {', '.join(flags)}" if flags else ""
    shown = chosen.view
    presentation_cut = len(shown) > limit
    if presentation_cut:
        head = max(1, (limit * 2) // 3)
        tail = max(1, limit - head)
        omitted = len(shown) - head - tail
        shown = (
            shown[:head]
            + f"\n…[primary presentation omitted {omitted} chars; open sealed report for the middle]…\n"
            + shown[-tail:]
        )
        suffix += (
            f"; presentation chars=0:{head} and {len(chosen.view) - tail}:{len(chosen.view)}"
        )
    coverage = "complete retained view" if not presentation_cut else "presentation-truncated retained view"
    return (
        f"primary observation [obs:{chosen.view_sha256[:12]}; {coverage}{suffix}] {path}:\n"
        f"{shown}"
    )


def _parent_report_excerpt(report: str, *, limit: int = 800) -> str:
    """Bound presentation without pretending a cut child interpretation is complete."""
    value = redact_text(report or "")
    if len(value) <= limit:
        return "child report (complete interpretation; preserve its qualifiers):\n" + (value or "(empty)")
    head = max(1, (limit * 2) // 3)
    tail = max(1, limit - head)
    omitted = len(value) - head - tail
    return (
        f"child report excerpt (interpretation; chars=0:{head} and {len(value) - tail}:{len(value)}; "
        f"presentation-truncated; omitted={omitted} — do not infer omitted claims):\n"
        + value[:head]
        + f"\n…[presentation omitted {omitted} chars; open sealed report for the middle]…\n"
        + value[-tail:]
    )


_CLAIM_SECTION = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*|__)?(?:top\s+bug|single\s+most\s+impactful|"
    r"findings?(?:\s+with\s+evidence)?|observed\s+issue)(?:\*\*|__)?\s*:?[\s-]*$",
    re.IGNORECASE,
)
_CLAIM_LABEL = re.compile(
    r"^(?:#{1,6}\s*)?(?:[-*+]\s*)?(?:\*\*|__|`{1,3})?"
    r"(?P<label>top\s+claim(?:\s*\([^)]*\))?|"
    r"bug(?:\s*\([^)]*\)|\s*#[^:]*)?|"
    r"observed\s+issue|finding|inference|conditional\s+consequence)(?:\*\*|__|`{1,3})?\s*:\s*",
    re.IGNORECASE,
)
_CLAIM_SKIP = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*|__)?(?:status|scope(?:\s+covered)?|files?\s+examined|"
    r"gaps?(?:\s*/\s*uncertainty)?|uncertainty|conflicts?)(?:\*\*|__)?\s*:",
    re.IGNORECASE,
)
_CLAIM_CONDITIONAL = re.compile(r"\b(?:if|unless|when|could|may|might|conditional)\b", re.IGNORECASE)


def _claim_section_priority(exact: str) -> int:
    """Recognize standalone report headings whose following line carries the actual claim."""
    value = re.sub(r"^#{1,6}\s*", "", str(exact or "").strip())
    # Markdown often puts the colon inside the closing emphasis: `**Top claim:**`.
    value = value.strip(" \t*_`")
    value = value.rstrip(":").strip(" \t*_`")
    if re.fullmatch(r"top\s+claim(?:\s*\([^)]*\))?", value, re.IGNORECASE):
        return 240
    if re.fullmatch(
            r"(?:top\s+bug(?:\s*\([^)]*\))?|single\s+most\s+impactful(?:\s+bug)?|"
            r"findings?(?:\s+with\s+evidence)?|observed\s+issue)", value, re.IGNORECASE):
        return 80
    return 0


def _claim_text(exact: str, *, table_issue: str = "") -> str:
    """Produce a one-line display from an exact report span without adding semantics."""
    value = table_issue or exact
    value = re.sub(r"^\s*(?:[-*+]\s+|#{1,6}\s+)", "", value.strip())
    value = value.replace("**", "").replace("__", "")
    value = re.sub(r"\s+", " ", value).strip(" |")
    return value if len(value) <= 520 else value[:519].rstrip() + "…"


def _extract_report_claims(
    report: str, observations: tuple[SubagentObservation, ...], brief: SubagentBrief,
) -> tuple[SubagentClaim, ...]:
    """Index one bounded, exact child-report claim for deterministic fan-in.

    This deliberately does *not* decide whether the claim follows from source bytes. ``report_exact`` binds the
    ledger entry to the sealed report; the parent may attribute it as testimony or independently verify it. The
    extraction only identifies the report's own top-finding line, so it cannot promote child prose to observation.
    """
    candidates: list[tuple[int, int, str, str]] = []
    claim_section_score = 0
    fenced = False
    for index, raw in enumerate(str(report or "").splitlines()):
        exact = raw.strip()
        if exact.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not exact or exact in {"---", "***"}:
            continue
        section_priority = _claim_section_priority(exact)
        if section_priority or _CLAIM_SECTION.match(exact):
            claim_section_score = section_priority or 80
            continue
        if _CLAIM_SKIP.match(exact):
            claim_section_score = 0
            continue

        table_issue = ""
        score = 0
        if exact.startswith("|") and exact.endswith("|"):
            cells = [cell.strip().replace("**", "").replace("__", "")
                     for cell in exact.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", cell or "-") for cell in cells):
                continue
            if len(cells) >= 3 and (cells[0].isdigit() or re.search(r"\b(?:critical|high|medium|low)\b",
                                                                    exact, re.IGNORECASE)):
                table_issue = cells[-1]
                score = 125

        label = _CLAIM_LABEL.match(exact)
        if label:
            kind = label.group("label").casefold()
            score = max(score, 200 if kind == "top claim"
                        else 130 if kind.startswith("bug") or kind in {"finding", "observed issue"}
                        else 115 if kind == "inference" else 105)
        elif claim_section_score:
            score = max(score, claim_section_score)
        if re.search(r"\b(?:bug|vulnerab|failure|incorrect|undefined|unhandled|swallow|mask|break|leak)\b",
                     exact, re.IGNORECASE):
            score += 12
        if score:
            # Preserve the entire physical report line as the source span. Display cleanup is a separate field;
            # deterministic publication uses this exact value, never the cleaned derivative.
            candidates.append((score, -index, raw, table_issue))
            claim_section_score = 0

    if not candidates:
        return ()
    _, _, report_exact, table_issue = max(candidates)
    if len(report_exact.encode("utf-8")) > 1200:
        # Never cut an exact claim span: a qualifier may live in the omitted tail. Fall back to source review.
        return ()
    text = _claim_text(report_exact, table_issue=table_issue)
    if not text:
        return ()
    primary = _primary_observation(observations, brief)
    refs = (primary.view_sha256,) if primary is not None else ()
    modality = "conditional" if _CLAIM_CONDITIONAL.search(report_exact) else "inference"
    return (SubagentClaim(
        text=text, report_exact=report_exact, modality=modality, observation_refs=refs,
    ),)


def _primary_arg(args) -> str:
    """The one informative arg for a compact activity line (path/command/pattern/…), whitespace-collapsed."""
    if not isinstance(args, dict):
        return ""
    for k in ("path", "command", "pattern", "name", "ref", "goal", "task"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())[:50]
    return ""


_CHILD_PRIVATE_RESOURCE_KINDS = frozenset({
    ResourceKind.ARTIFACT, ResourceKind.HISTORY, ResourceKind.SUBAGENT, ResourceKind.ROSTER,
})


def _classified_read_target(
    args, resource_ref=None, *, canonicalize=None, event=None,
) -> tuple[ResourceRef, str, bool]:
    """Return the host-routed resource, its canonical handle, and private-host-dir status.

    Namespace spelling alone is not authoritative: ``/workspace/history/turn-1.md`` can be the same virtual
    handle as ``history/turn-1.md``, while a real ``history/`` or ``artifacts/`` project path shadows that
    mount and must remain readable.  Prefer the typed effect captured at execution time, then the host's
    canonical ``resource_ref`` seam.  The lexical fallback exists only for minimal legacy/test hosts.
    """
    path = args.get("path") if isinstance(args, dict) else ""
    ref = None
    if event is not None:
        for effect in (getattr(getattr(event, "outcome", None), "effects", ()) or ()):
            if getattr(effect, "kind", "") != "resource_observed":
                continue
            payload = getattr(effect, "payload", {}) or {}
            try:
                ref = ResourceRef(ResourceKind(str(payload.get("resource_kind") or "")),
                                  str(payload.get("handle") or "."))
            except (TypeError, ValueError):
                continue
            break

    if ref is None and callable(resource_ref):
        try:
            candidate = resource_ref(str(path or ""))
            if isinstance(candidate, ResourceRef):
                ref = candidate
        except Exception:  # noqa: BLE001 — a classifier failure must fail closed via the lexical fallback
            ref = None
    if ref is None:
        ref = reserved_resource_ref(_norm_vpath(path))
    canonical_source = ref.handle if ref.virtual else path
    if not ref.virtual and callable(canonicalize):
        try:
            canonical_source = canonicalize(str(path or ""))
        except Exception:  # noqa: BLE001 — lexical fallback still protects the ordinary relative spelling
            pass
    canonical = _norm_vpath(canonical_source)
    # `.sliceagent/` is a physical host-private store, not a virtual archive kind. Keep the existing
    # default-deny for both its relative spelling and the absolute spelling canonicalized by the host.
    private = canonical == ".sliceagent" or canonical.startswith(".sliceagent/")
    return ref, canonical, private


def _targets_reserved_ns(args, resource_ref=None) -> bool:
    """True only for the resource the host actually routes as a parent-private view."""
    ref, _, private = _classified_read_target(args, resource_ref)
    return ref.kind in _CHILD_PRIVATE_RESOURCE_KINDS or private


# NO hire cap — a dormant specialist is just files on disk and a wake reads only its own files (flat in
# roster size), so the roster isn't a scarce resource to ration. The per-turn cost is bounded on the
# manifest side instead (hippocampus.roster_recent parses only the top-K), so the store can grow freely.
_CAREER_MANIFEST_K = 5   # wake-seed career lines (one-liners + handles; full jobs stay paged out)


def _render_wake_block(profile: dict, jobs: list, name: str) -> str:
    """The WAKE seed: identity + bounded career manifest + the abstention self-model. FLAT by construction —
    lessons ≤ K (curated), career = last K one-liners with handles; the full jobs stay paged out in
    roster/<name>/ (the specialist may read its OWN files). The abstention line is #114 one level down:
    a persona + 'memories' is the maximal confabulation trap, so the seed says exactly what the memories
    are (sealed reports) and what to do beyond them (say so)."""
    lines = [f"YOUR STANDING IDENTITY — you are {name!r}, a standing {profile.get('kind', '?')} specialist "
             f"(hired {(profile.get('created') or '?')[:10]}; {profile.get('jobs', 0)} completed job(s), "
             f"last active {(profile.get('last_active') or '?')[:10]}).",
             "Your memories are ONLY what your sealed reports say. If this task needs detail they don't "
             "contain, say so in your report rather than reconstructing it. The workspace may have changed "
             "since your last job — re-read files; never trust quoted content from an old report over the "
             "file on disk."]
    lessons = [L for L in (profile.get("lessons") or []) if isinstance(L, dict) and L.get("text")]
    if lessons:
        lines.append("LESSONS from your past jobs (advisory priors — they may be stale or wrong; ignore one "
                     "when the evidence disagrees):")
        lines += [f"- {L['text']}  ({L.get('job', '?')}, {(L.get('ts') or '')[:10]})" for L in lessons]
    if jobs:
        lines.append(f'YOUR CAREER (own sealed reports — read one in full: '
                     f'read_file("roster/{name}/job-<N>.md")):')
        for r in jobs[-_CAREER_MANIFEST_K:]:
            a = r.get("artifact") or {}
            lines.append(f"- {r.get('id')} · {a.get('status', '?')} · {(r.get('ts') or '')[:10]} — "
                         f"{one_line(a.get('report') or a.get('task', ''), 90)}")
        if len(jobs) > _CAREER_MANIFEST_K:
            lines.append(f"(+{len(jobs) - _CAREER_MANIFEST_K} earlier job(s) — "
                         f'read_file("roster/{name}/profile.md") for the full career)')
    return "\n".join(lines)


def _nested_sink(notify, depth: int):
    """Surface a child agent's progress as ONE DYNAMIC line: each tool call updates a single
    status line with the current action + a running count, instead of printing a line per call. The renderer
    (RichSink.subagent_notify) overwrites in place; the child's final summary returns via the spawn tool's
    result, so there's no per-assistant-text spam here."""
    pad = "    " * depth
    state = {"n": 0}
    def sink(event):
        if isinstance(event, ToolStarted):
            state["n"] += 1
            notify(f"{pad}↳ {event.name} {_primary_arg(event.args)} · {state['n']} calls".rstrip())
    return sink


def run_subagent(task: str, *, tools, llm, retriever, memory, policy,
                 max_steps: int = 20, depth: int = 1, notify=None,
                 read_only: bool = False, spec: AgentSpec | None = None, session_id: str = "",
                 name: str = "", grants: tuple = (), identity_block: str = "",
                 brief: SubagentBrief | None = None, workspace_id: str = "", task_id: str = "",
                 parent_id: str = "", artifact_store=None, artifact_id: str = "",
                 artifact_ref_sink=None, token_budget: int | None = None,
                 launch_ordinal: int = 0) -> str:
    """Run a child agent of a given KIND (`spec`) on `task` with a fresh slice; return a bounded summary.
    The child's events stay on its OWN dispatcher — they never touch the parent's slice (the bounded-
    context guarantee); only the summary crosses back.

    `spec` is the named AgentSpec (tools allowlist + reasoning + system-prompt layer). Back-compat: when
    `spec` is None it is derived from `read_only` (the built-in explorer vs general). A read-only spec runs
    as an EXPLORER — its tool host exposes only the read-only allowlist, so it cannot mutate the workspace."""
    from .events import make_dispatcher
    from .guardrails import ToolCallGuardrailConfig
    from .hooks import BudgetHook, CompositeHooks, GuardrailHook, PermissionHook
    from .loop import run_turn
    from .pfc import Slice, slice_sink
    from .seed import make_build_slice

    child_journal = None
    if artifact_store is not None and artifact_id:
        from .persistence import PendingTurnJournal
        child_journal = PendingTurnJournal.begin(
            artifact_store.root, artifact_id=artifact_id,
            workspace_id=workspace_id or "workspace-unknown",
            session_id=session_id or "session-ephemeral", task_id=task_id or "task-unknown",
            base_generation=0, user_request=redact_text(task),
        )

    def journal_sink(event):
        if child_journal is None:
            return
        def safe(value):
            if isinstance(value, str):
                return redact_text(value)
            if isinstance(value, dict):
                return {redact_text(str(key)): safe(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(child) for child in value]
            return value

        if isinstance(event, ToolStarted) and event.invocation is not None:
            child_journal.record_invocation(
                event.invocation.id, name=event.name, args=safe(dict(event.args or {})),
            )
        elif isinstance(event, ToolResult) and event.outcome is not None:
            outcome = event.outcome
            child_journal.record_invocation(
                outcome.invocation.id, name=outcome.invocation.name,
                args=safe(dict(outcome.invocation.args)),
            )
            child_journal.record_outcome(outcome.invocation.id, safe({
                "status": outcome.status.value, "text": outcome.text,
                "effects": [{"id": effect.id, "kind": effect.kind,
                             "payload": dict(effect.payload)} for effect in outcome.effects],
            }))

    if spec is None:
        spec = BUILTIN_AGENTS["explorer" if read_only else "general"]
    read_only = spec.read_only   # the kind decides; everything below keys off the SPEC

    # The brief is the ONLY parent→child semantic channel. Direct legacy callers get the same objective
    # and grants projected into the typed shape; SubagentHost supplies intent/scope/source provenance.
    brief = brief or SubagentBrief.create(task, canonical_refs=grants)
    if brief.objective != task:
        raise ValueError("subagent task and typed brief objective disagree")
    grants = tuple(brief.canonical_refs)
    child_task = brief.render()

    child_state = Slice()
    child_state.reset(child_task)
    if read_only:
        # explorer: keep the whole exploration resident (no eviction churn → no "stuck") AND don't let the
        # read-only convergence nudge cut the review short before the key files are read — see
        # EXPLORER_READ_BUDGET + Slice.explore_mode. max_steps bounds the explorer.
        child_state.read_budget = child_state.read_ceiling = EXPLORER_READ_BUDGET
        child_state.explore_mode = True
    # per-kind reasoning via a per-child llm view (no mutation). The explorer kind honors the documented
    # AGENT_EXPLORER_REASONING knob (EXPLORER_REASONING) instead of its hard-wired "fast", so the env var works.
    child_reasoning = EXPLORER_REASONING if spec.name == "explorer" else spec.reasoning
    child_llm = _profile_llm(llm, child_reasoning)
    # A WOKEN specialist gets its identity block (career + lessons + abstention self-model) as an extra
    # system layer under the kind prompt — the kind prompt stays IMMUTABLE; the identity is data.
    system_extra = spec.system_prompt + ("\n\n" + identity_block if identity_block else "")
    if name:
        # W5' seal-time reflection — the proven trailing-marker pattern (VERDICT:). One optional line;
        # curation (dedupe/cap/provenance) happens at the archive, not here.
        system_extra += ("\n\nIf this job taught you something a future you should know (a pitfall, a "
                         "convention, where the bodies are buried), end your summary with ONE line: "
                         '"LESSON: <the lesson>". Only a real lesson — most jobs have none.')
    build = make_build_slice(child_state, tools, retriever, memory, child_task, system_extra=system_extra)

    cap = _CaptureLast()
    trace = _TraceSink()
    observation_sink = _ObservationSink(
        getattr(tools, "resource_ref", None), getattr(tools, "_archive_handle", None),
    )
    reducer = slice_sink(child_state)
    sinks = [cap, trace, observation_sink]
    if notify is not None:
        sinks.append(_nested_sink(notify, depth))
    required = (journal_sink, reducer) if child_journal is not None else (reducer,)
    child_dispatch = make_dispatcher(*sinks, required=required)

    _child_hooks = [PermissionHook(policy)] if policy is not None else []
    if token_budget is not None:
        _child_hooks.append(BudgetHook(max(0, int(token_budget))))
    # An EXPLORER does read-only investigation: a repeated read/list/grep is at most inefficient, never the
    # write-loop disaster the anti-loop HARD-BLOCK guards against — and max_steps already bounds it. So relax
    # the READ axes (no-progress / result-repeat) for explorers while KEEPING exact-failure (a repeated
    # FAILING call is still a real loop). Stops review children going "stuck" on legitimate reads.
    _guard_cfg = (ToolCallGuardrailConfig(no_progress_block_after=10**6, result_repeat_block_after=10**6)
                  if read_only else None)
    _child_hooks.append(GuardrailHook(_guard_cfg))
    hooks = CompositeHooks(*_child_hooks)
    result = run_turn(build_slice=build, llm=child_llm, tools=tools, dispatch=child_dispatch,
                      hooks=hooks, max_steps=max_steps, turn_id=artifact_id)
    # Child usage is sealed into the artifact and returned as a typed ToolEffect. The parent loop consumes
    # that effect exactly once into TurnOutcome, StepEnd metrics, and the same task budget.
    _child_usage = dict(getattr(result, "usage", None) or {})
    core_handle = ""
    claim_projection: list[dict] = []
    scope_projection: list[str] = []
    target_projection = ""

    def child_result(text: str, *, ok: bool, outcome_status: str | None = None):
        from .execution import ToolEffect, Usage
        from .registry import ToolText
        usage = Usage.from_value(_child_usage)
        identity = artifact_id or ("ephemeral-" + hashlib.sha256(
            f"{workspace_id}|{task_id}|{parent_id}|{depth}|{name}|{task}".encode("utf-8", "replace")
        ).hexdigest()[:24])
        effects = [ToolEffect(f"{identity}:model-usage", "model_usage", usage.as_dict())]
        if core_handle:
            # The bounded prose result is presentation; this typed relationship is the exact refinement
            # map from the parent invocation to the immutable child report that actually sealed.
            effects.append(ToolEffect(
                f"{identity}:child-artifact", "child_artifact", {
                    "artifact_id": core_handle,
                    "kind": spec.name,
                    "name": name,
                    "launch_ordinal": launch_ordinal,
                    "status": status,
                    "scope": scope_projection,
                    "delegation_target": target_projection,
                    "claims": claim_projection,
                },
            ))
        return ToolText(
            text, ok=ok, status=outcome_status,
            effects=tuple(effects),
        )

    _af = list(child_state.active_files)   # BOUND the resident head: a child that read 100 files must not
    files = (", ".join(_af[:20]) + (f" +{len(_af) - 20} more" if len(_af) > 20 else "")) or "(none)"
    # A READ-ONLY explorer's deliverable is its summary; so is a verifier's verdict (summary_is_deliverable),
    # whose LAST check is often a deliberate failing repro. A lingering last_error must NOT flag those as "did
    # not finish cleanly". Only a genuinely WRITABLE worker's last_error matters (it may have left the task
    # broken). end_turn means it produced a final summary either way.
    summary_is_deliverable = read_only or getattr(spec, "summary_is_deliverable", False)
    success = result.stop_reason == "end_turn" and (summary_is_deliverable or not child_state.last_error)
    status = "ok" if success else result.stop_reason
    kind_label = {"explorer": "explore", "general": "subagent"}.get(spec.name, spec.name)  # named-kind label
    label = f"{name} ({kind_label})" if name else kind_label   # instance identity first, kind in parens

    # SEAL the child's work as a structured artifact and ARCHIVE it. The parent gets a bounded digest + a
    # recall handle; the FULL report lives at subagents/<id>.md — paged in on demand, out again next seal — so
    # the parent's context tracks the child's digest, not its raw work-volume (the moat, one level up). No detail is
    # lost: the digest is a coarse-graining, the handle is its refinement map.
    # `name` is the INSTANCE identity (who); `brief` is the VERBATIM ask (what they were told) — provenance:
    # whoever later reads this report can see the question alongside the answer, so a narrowly-briefed child
    # is never silently cited for broad claims.
    # W5': lift the optional trailing "LESSON: ..." reflection out of the report into a typed field
    # (the line stays in the report verbatim — the seal is honest; this is indexing, not editing).
    _lm = re.findall(r"^LESSON:\s*(.+)$", cap.text or "", re.MULTILINE)
    try:
        workspace_root = tools.root() if hasattr(tools, "root") else os.getcwd()
    except Exception:  # noqa: BLE001 — a missing root becomes visible uncertainty, not a lost child result
        workspace_root = os.getcwd()
    child_files = list(dict.fromkeys([*child_state.active_files, *sorted(child_state.edited_files)]))
    evidence_refs = tuple(dict.fromkeys([
        *grants, *(clause.source_artifact for clause in brief.intent_clauses if clause.source_artifact),
    ]))
    gaps = (
        (() if success else (child_state.last_error or f"child stopped with {result.stop_reason}",))
        + observation_sink.gaps
    )
    uncertainty = (() if cap.text else ("child produced no final report text",))
    claims = _extract_report_claims(cap.text or "", observation_sink.observations, brief)
    typed_artifact = SubagentArtifact.create(
        kind=spec.name, name=name, workspace_id=workspace_id or os.path.realpath(workspace_root),
        session_id=session_id or "session-ephemeral", task_id=task_id or "task-unknown",
        parent_id=parent_id, launch_ordinal=launch_ordinal, brief=brief, status=status,
        coverage=f"{len(child_files)} file(s) examined or changed; stop={result.stop_reason}",
        report=cap.text or "", findings=tuple(child_state.findings),
        evidence_refs=evidence_refs, observations=observation_sink.observations, claims=claims,
        files=child_files, workspace_root=workspace_root,
        change_set=tuple(sorted(child_state.edited_files)), gaps=gaps, uncertainty=uncertainty,
        error=(child_state.last_error or ("" if success else str(result.stop_reason))),
        steps=result.steps, usage=_child_usage,
        trace=trace.text(),   # W6': locator-grade path; detailed payload stays outside parent context
        lesson=one_line(_lm[-1], 200) if _lm else "",
    )
    artifact = typed_artifact.to_record()
    core_archive_error = ""
    if artifact_store is not None and artifact_id:
        try:
            from datetime import datetime, timezone
            from .persistence import Artifact
            safe_artifact = _redact_archive_value(artifact)
            # Redaction transforms both report and indexed spans. Re-parse the exact bytes that will be stored so
            # report membership, reference closure, schema bounds, and hashes cannot diverge at this last seam.
            safe_typed_artifact = SubagentArtifact.from_record(safe_artifact)
            claim_projection = [claim.to_dict() for claim in safe_typed_artifact.claims]
            scope_projection = list(safe_typed_artifact.brief.scope)
            target_projection = safe_typed_artifact.brief.delegation_target
            core = Artifact(
                id=artifact_id, kind="subagent", workspace_id=typed_artifact.workspace_id,
                session_id=typed_artifact.session_id, task_id=typed_artifact.task_id,
                parent_id=typed_artifact.parent_id,
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                status=typed_artifact.status,
                title=redact_text(typed_artifact.name or typed_artifact.brief.objective),
                brief=_redact_archive_value(typed_artifact.brief.to_dict()),
                summary=redact_text(one_line(typed_artifact.report, 300)),
                structured_body=safe_artifact,
                files=tuple(redact_text(path) for path in typed_artifact.files),
                refs=tuple(redact_text(ref) for ref in typed_artifact.evidence_refs
                           if not ref.startswith(("history/", "subagents/"))),
                uncertainty=tuple(redact_text(item) for item in typed_artifact.uncertainty),
                error=redact_text(typed_artifact.error),
            )
            artifact_store.put(core)
            # Publish the downward parent dependency before closing the child journal or returning success.
            # If this handoff fails, the child stays recoverable but is not accepted as parent state.
            if artifact_ref_sink is not None:
                artifact_ref_sink(core.id)
            if child_journal is not None:
                child_journal.mark_artifact_written(core)
                child_journal.mark_sealed()
                child_journal.cleanup()
            core_handle = core.id
        except Exception as exc:  # noqa: BLE001 — a missing canonical seal invalidates child acceptance
            core_archive_error = f"{type(exc).__name__}: {exc}"
    if artifact_store is not None and not core_handle:
        why = core_archive_error or "canonical artifact identity was not allocated"
        return child_result(
            "Error: subagent result is indeterminate: its canonical local report could not be sealed "
            f"({why}). No child finding was accepted; rerun or repair the local artifact store.",
            ok=False, outcome_status="indeterminate")
    handle, archive_error = "", ""
    if memory is not None and session_id:
        try:
            handle = memory.append_subagent_artifact(session_id, artifact)
        except Exception as exc:  # noqa: BLE001 — convert durable-seal failure into an honest tool result
            archive_error = f"{type(exc).__name__}: {exc}"
    durable_archive_required = bool(memory is not None and getattr(memory, "is_durable", False))
    if handle:   # W6': additive FTS5 mirror → search_history finds delegated work by CONTENT (never
        try:
            memory.index_subagent_artifact(session_id, handle, artifact)   # derived index; artifact is authority
        except Exception:  # noqa: BLE001 — a rebuildable search mirror cannot invalidate the durable seal
            pass
    if handle and name and memory is not None:   # career duplicates only a successfully sealed job
        try:
            memory.roster_append_job(name, artifact)   # extension view; canonical session artifact is authority
        except Exception:  # noqa: BLE001 — roster indexing cannot invalidate an already durable child seal
            pass

    head = f"[{label} {status} · {result.steps} steps · files: {files}]"
    durable_handle = core_handle or handle
    if durable_archive_required and not durable_handle:
        why = core_archive_error or archive_error or (
            "missing session id" if not session_id else "artifact store returned no handle")
        return child_result(
            "Error: subagent result is indeterminate: its durable report could not be sealed "
            f"({why}). No child finding was accepted; rerun or repair the local artifact store.",
            ok=False, outcome_status="indeterminate")
    if durable_handle:   # archived → bounded digest + recall handle (the refinable seal)
        body = _parent_report_excerpt(cap.text)
        # ALWAYS hand back the CANONICAL immutable id (sub-N.md), never the subagents/<name>.md alias: the
        # alias retargets to the LATEST job for that name, so a later same-name job would silently make an
        # earlier tool result / grant open a DIFFERENT report (external review S11). The <name>.md alias
        # stays resolvable in SubagentFS as a convenience; the sealed handle the parent stores is immutable.
        target = f"artifacts/{core_handle}.md" if core_handle else f"subagents/{handle}.md"
        primary = _parent_observation_excerpt(observation_sink.observations, brief)
        summary = (
            f"{head}\n{body}\n"
            f"{primary}\n→ full report: read_file(\"{target}\")"
        )
    else:        # no durable archive (eval/headless) → inline, back-compat with the pre-artifact behavior
        summary = (
            f"{head}\n{_parent_report_excerpt(cap.text)}\n"
            f"{_parent_observation_excerpt(observation_sink.observations, brief)}"
        )
    if not success:
        if child_state.last_error:
            summary += " | unresolved: " + one_line(child_state.last_error, 160)
        return child_result(
            "Error: subagent did not finish cleanly: " + summary, ok=False,
            outcome_status="indeterminate" if result.stop_reason == "indeterminate" else None,
        )
    return child_result(summary, ok=True)


class SubagentHost:
    """ToolHost wrapper that adds the `spawn_agent` delegation tool. Delegates every real tool (and
    read_text/accesses) to the wrapped host, so parent and child share one workspace."""

    def __init__(self, inner, *, llm, retriever, memory, policy,
                 max_depth: int = 1, max_steps: int = 20, depth: int = 0, notify=None,
                 read_only: bool = False, spec: AgentSpec | None = None, agents=None, session_id: str = "",
                 grants: frozenset = frozenset(), instance_name: str = "",
                 intent_provider=None, task_id_fn=None, parent_id_fn=None, workspace_id: str = "",
                 artifact_store=None, artifact_ref_sink=None, core_mode: bool = False):
        self.inner = inner
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.policy = policy
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.depth = depth
        self.notify = notify
        # spec set on a CHILD host restricts its tools to that kind's allowlist; None = a PARENT host that
        # offers the spawn tools. read_only is a back-compat alias for spec=explorer.
        self.spec = spec or (BUILTIN_AGENTS["explorer"] if read_only else None)
        self.read_only = self.spec.read_only if self.spec is not None else False
        self.agents = agents or BUILTIN_AGENTS
        self.session_id = session_id   # the PARENT session — children archive artifacts under it (recall handle)
        # W2: exact sealed-report handles THIS child may read (parent-minted; empty for temps/parents).
        # A grant is a pointer to a SEAL, so the coupling law ("children couple only through seals") holds.
        self.grants = frozenset(grants)
        # W4': THIS child's standing identity (empty for temps/parents) — unlocks reads of its OWN
        # roster/<name>/ files only (self-memory is not a third channel; siblings stay denied).
        self.instance_name = instance_name
        # Typed brief provenance. The provider receives the delegated objective and returns ONLY the exact
        # relevant IntentEntry selection (or an IntentState); no parent transcript crosses this seam.
        self.intent_provider = intent_provider
        self.task_id_fn = task_id_fn
        self.parent_id_fn = parent_id_fn
        self.workspace_id = workspace_id
        self.artifact_store = artifact_store
        self.artifact_ref_sink = artifact_ref_sink
        self.core_mode = bool(core_mode)
        self._artifact_seq: dict[tuple[str, str], int] = {}
        self._artifact_lock = threading.Lock()

    def _next_artifact_identity(self, *, task_id: str, parent_id: str) -> tuple[str, int]:
        """Allocate launch identity before child work starts.

        The ordinal is meaningful even in legacy/non-core mode.  The old ``sub-N`` handle is assigned only
        when a report finishes archiving, so concurrent completion order must never be mistaken for launch
        order in later discourse resolution.
        """
        key = (task_id, parent_id)
        with self._artifact_lock:
            sequence = self._artifact_seq.get(key, 0) + 1
            self._artifact_seq[key] = sequence
        if self.artifact_store is None:
            return "", sequence
        from .persistence import deterministic_artifact_id
        artifact_id = deterministic_artifact_id(
            kind="subagent", workspace_id=self.workspace_id or "workspace-unknown",
            session_id=self.session_id or "session-ephemeral", task_id=task_id,
            logical_id=f"{parent_id or 'parent-none'}:{sequence}",
        )
        return artifact_id, sequence

    def __getattr__(self, name):
        # FAITHFUL ToolHost projection: any host attribute NOT explicitly overridden above
        # (root, add_root, registry, on_ask_user, …) delegates to the wrapped host, so parent and
        # child share ONE host surface. Without this, root() was silently missing → make_build_slice
        # got cwd="" → the slice's WORKING DIRECTORY / cwd / WORKSPACE / git ENVIRONMENT tier vanished
        # whenever subagents were enabled (the agent then can't see its own folder). Kills the whole
        # "wrapper forgot to forward a host method" class, not just root().
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "inner"), name)

    def schemas(self) -> list[dict]:
        s = list(self.inner.schemas())
        if self.spec is not None:
            # CHILD host: never expose ask_user (a subagent must not stall on the END-USER — ambiguity is the
            # parent's job; it returns a summary instead). Then restrict to the kind's allowlist if it has one.
            s = [x for x in s if x.get("function", {}).get("name") not in SUBAGENT_EXCLUDED_TOOLS]
            if self.spec.tools is not None:
                allow = set(self.spec.tools) - {"search_history"}   # child: search_history leaks the parent session
                s = [x for x in s if x.get("function", {}).get("name") in allow]
                # ONE spawn tool now (spawn_agent subsumes the old spawn_explore/spawn_subagent aliases —
                # measured parity on parallel fan-out). Offer it if the kind's allowlist permits delegation.
                if self.depth < self.max_depth and {"spawn_agent", "spawn_explore", "spawn_subagent"} & allow:
                    s.append(self._agent_schema())
                return s
        if (self.depth < self.max_depth
                and (not self.core_mode or "explorer" in self.agents)):
            # parent (or a general child) — offer delegation while depth remains. Core mode has exactly one
            # truthful capability; if that spec is absent, advertise no spawn tool rather than an empty lie.
            s.append(self._agent_schema())
        return s

    def _agent_schema(self) -> dict:
        """The ONE delegation tool — `spawn_agent`. It subsumes the former spawn_explore / spawn_subagent
        (each was just this with agent='explorer' / 'general'); measured parity on parallel fan-out, so the
        breadth nudge lives here in the description, not in a dedicated verb. The two orthogonal dials —
        KIND (agent=) and IDENTITY (name=) — are spelled out so the model has the right mental model."""
        available = ({"explorer": self.agents["explorer"]}
                     if self.core_mode and "explorer" in self.agents else dict(self.agents))
        kinds = "; ".join(f"{n} ({sp.description})" for n, sp in available.items())
        properties = {
            "agent": {"type": "string", "enum": list(available),
                      "description": "the KIND to run (one of the live values in this schema)"},
            "task": {"type": "string", "description": "the self-contained sub-task for that agent"},
            "scope": _SCOPE_PARAM, "exclusions": _EXCLUSIONS_PARAM,
            "report_shape": _REPORT_SHAPE_PARAM, "drift_policy": _DRIFT_POLICY_PARAM,
        }
        if not self.core_mode:
            properties.update({"name": _NAME_PARAM, "grants": _GRANTS_PARAM})
        return {"type": "function", "function": {
            "name": "spawn_agent",
            "description": (
                "Delegate a self-contained sub-task to a child agent that runs in its OWN bounded context and "
                "returns ONLY a short summary (its reads never enter your context). Two dials:\n"
                "• agent = which KIND — " + kinds + ". For BREADTH (review/understand a repo, find a bug, "
                "audit several modules) emit MULTIPLE spawn_agent(agent=\"explorer\", …) calls in ONE response "
                "— explorers are read-only and run in PARALLEL; one per area/module/question, then synthesize "
                "their summaries. Stay single-agent for one tightly-coupled change you're editing yourself.\n"
                + ("Core mode exposes one-shot read-only children only."
                   if self.core_mode else
                "• name = OPTIONAL identity. OMIT it → a one-shot TEMP (used once, then only its sealed report "
                "remains). PASS one → HIRE a STANDING specialist that persists across sessions, accumulates "
                "lessons, and can be WOKEN by re-using the same name later (see STANDING SPECIALISTS). Hire "
                "when this is an area you'll revisit; use a temp for a one-off.")),
            "parameters": {"type": "object", "properties": properties,
                           "required": ["agent", "task"]}}}

    def _brief(self, task: str, args: dict, canonical_refs: frozenset) -> SubagentBrief:
        intent_entries = ()
        scope = tuple(args.get("scope") or ())
        delegation_target = ""
        if self.intent_provider is not None:
            # A configured provenance seam failing must block delegation; silently dropping binding
            # constraints would create a deceptively successful but under-scoped child report.
            intent_state = (self.intent_provider(task) if callable(self.intent_provider)
                            else self.intent_provider)
            intent_entries = exact_intent_clauses(intent_state)
            contract = getattr(intent_state, "turn_contract", None)
            requirement = getattr(contract, "delegation_requirement", None)
            if requirement is not None:
                # Fan-out count/targets/reduce shape are PARENT orchestration invariants. Replicating the whole
                # current request into each scoped child tells every explorer to perform the parent fan-out and
                # conflicts with its one-file objective. Keep independent standing constraints; remove only the
                # current request owned by the typed delegation requirement.
                current_request = str(getattr(intent_state, "current_request", "") or "")
                intent_entries = tuple(
                    clause for clause in intent_entries if clause.verbatim_clause != current_request
                )
                raw_targets = (requirement.get("targets", ()) if isinstance(requirement, dict)
                               else getattr(requirement, "targets", ())) or ()
                scope_keys = {_norm_vpath(item) for item in scope if isinstance(item, str)}
                task_surface = str(task or "").replace("\\", "/")
                matches = []
                for raw_target in raw_targets:
                    target = str(raw_target or "").strip()
                    if not target:
                        continue
                    if _task_mentions_exact_target(task_surface, target) \
                            or _norm_vpath(target) in scope_keys:
                        matches.append(target)
                if len(matches) == 1:
                    delegation_target = matches[0]
                    if not scope:
                        scope = (delegation_target,)
        return SubagentBrief.create(
            task, intent_entries=intent_entries,
            scope=scope, delegation_target=delegation_target,
            exclusions=args.get("exclusions") or (),
            report_shape=(args.get("report_shape") or None),
            canonical_refs=tuple(sorted(canonical_refs)),
            drift_policy=args.get("drift_policy") or "report",
        )

    @staticmethod
    def _context_id(provider, fallback: str) -> str:
        if provider is None:
            return fallback
        value = provider() if callable(provider) else provider
        return str(value or fallback)

    def _validate_grants(self, raw):
        """Spawn-time grant validation (kernel says no, loudly): (err, frozenset). Rules — parent-minted only
        (NO transitive propagation: a child cannot re-grant, so a handle's reach is one hop), exact file
        handles only, must resolve to an EXISTING sealed artifact right now, hard cap."""
        if not raw:
            return "", frozenset()
        if self.spec is not None:
            return ("Error: a subagent cannot re-grant sealed-report handles to its own children — grants "
                    "are minted by the parent only. Ask for what you need in your report instead.", frozenset())
        if not isinstance(raw, (list, tuple)):
            return "Error: 'grants' must be a list of sealed-report handles like [\"subagents/sub-1.md\"]", frozenset()
        if len(raw) > _MAX_GRANTS:
            return f"Error: too many grants ({len(raw)} > {_MAX_GRANTS}) — grant only the reports this child needs", frozenset()
        arts = (self.memory.read_subagent_artifacts(self.session_id)
                if (self.session_id and self.memory is not None) else [])
        ids = {str(r.get("id")) for r in arts if r.get("id")}
        # A mutable identity alias is accepted at the public edge for convenience, then resolved ONCE to
        # the latest immutable job. Neither the child brief nor the resulting artifact retains the alias.
        names: dict[str, str] = {}
        for record in arts:
            identity = (record.get("artifact") or {}).get("name")
            handle = record.get("id")
            if identity and handle:
                names[str(identity)] = str(handle)
        out = set()
        for g in raw:
            p = _norm_vpath(g)
            if p and "/" not in p:                       # accept a bare leaf ("sub-1.md") for convenience
                p = "subagents/" + p
            leaf = p[len("subagents/"):] if p.startswith("subagents/") else ""
            stem = leaf[:-3] if leaf.endswith(".md") else ""
            canonical = ""
            if "/" not in leaf and leaf:
                if _GRANT_SUB.match(leaf) and stem in ids:
                    canonical = stem
                elif _valid_instance_name(stem) and stem in names:
                    canonical = names[stem]
            if not canonical:
                return (f"Error: cannot grant {g!r} — grants must be EXACT existing sealed-report handles "
                        f'(e.g. "subagents/sub-1.md" or "subagents/<name>.md"; never a directory or '
                        f'index.md). See read_file("subagents/index.md") for what exists.', frozenset())
            out.add(f"subagents/{canonical}.md")
        return "", frozenset(out)

    def accesses(self, name: str, args: dict) -> list:
        if name == "spawn_explore":
            # read-only child (no edit/shell/spawn): parallelizes with OTHER explorers, serializes vs any
            # writer — so a broad task can fan out N explorers concurrently (the real swarm).
            return [ReadAllAccess()]
        if name == "spawn_subagent":
            return [AllAccess()]  # WRITABLE nested work → globally exclusive (two writers in one workspace serialize)
        if name == "spawn_agent":   # a read-only kind parallelizes (swarm); a writable kind serializes
            sp = self.agents.get(args.get("agent", ""))
            return [ReadAllAccess()] if (sp is not None and sp.read_only) else [AllAccess()]
        return self.inner.accesses(name, args)

    def resolve_intent_effect(self, name: str, args: dict):
        """Resolve wrapper-owned delegation tools for the per-turn authority gate.

        Delegation schemas are projected by this wrapper rather than registered on the inner host. A
        read-only child is observational fan-out; a writable child can change the workspace and therefore
        requires explicit effect authority.
        """
        from .registry import ToolIntentEffect
        if name == "spawn_explore":
            return ToolIntentEffect.OBSERVE
        if name == "spawn_subagent":
            return ToolIntentEffect.EXTERNAL
        if name == "spawn_agent":
            spec = self.agents.get(str((args or {}).get("agent") or ""))
            if spec is None:
                return ToolIntentEffect.UNKNOWN
            return ToolIntentEffect.OBSERVE if spec.read_only else ToolIntentEffect.EXTERNAL
        resolver = getattr(self.inner, "resolve_intent_effect", None)
        if callable(resolver):
            return resolver(name, args or {})
        registry = getattr(self.inner, "registry", None)
        if registry is None:
            return ToolIntentEffect.UNKNOWN
        return registry.resolve_intent_effect(name, args or {})

    def read_text(self, path: str) -> str:
        return self.inner.read_text(path)

    def run(self, name: str, args: dict) -> str:
        # Scheduler-owned metadata is not a model argument. Consume it at the host edge before any
        # public validation, and never pass it to briefs, policies, artifacts, or nested tool calls.
        token_budget = args.get(CHILD_TOKEN_BUDGET_ARG)
        args = {key: value for key, value in args.items() if key != CHILD_TOKEN_BUDGET_ARG}
        if self.spec is not None and name in SUBAGENT_EXCLUDED_TOOLS:
            # defense-in-depth: even if the model calls a tool it was not offered, a CHILD can't ask the
            # end-user — return a directive instead of blocking on input (which would stall the parent).
            return ("Error: a subagent cannot ask the user. Decide on a reasonable assumption, proceed, and "
                    "state the assumption in your summary; the parent will handle any real ambiguity.")
        # #42/#43: ENFORCE the kind's allowlist at RUNTIME, not just in schemas() (which only HIDES tools).
        # Without this a child that emits an out-of-kind tool anyway slips through to inner.run — and a
        # read-only EXPLORER could call spawn_subagent to escalate into a WRITABLE child. spawn_* are not in
        # the read-only allowlist, so this also blocks that escalation. (A general child has tools=None → skip.)
        if self.spec is not None and self.spec.tools is not None and name not in self.spec.tools:
            return f"Error: tool {name!r} is not available to the {getattr(self.spec, 'name', 'sub')!r} agent"
        # ISOLATION: a CHILD must not read the PARENT's trajectory (history/) or its siblings' sealed artifacts
        # (subagents/) — reserved virtual namespaces on the SHARED base host. Blocking keeps the ONLY
        # child↔parent coupling the two seals (brief down, artifact up); a child needing more context says so
        # in its report rather than paging the parent's.
        # search_history is bound to the PARENT session (its FTS5 this-session mode returns previews of the
        # parent's own turns) → same trajectory leak as reading history/. A child works from its brief, not the
        # parent's memory, so block it too (and it's dropped from the child's schemas below).
        canonical_path = _norm_vpath(args.get("path") if isinstance(args, dict) else "")
        private_read = False
        if name in ("read_file", "list_files", "grep", "glob"):
            read_ref, canonical_path, private_path = _classified_read_target(
                args, getattr(self.inner, "resource_ref", None),
                canonicalize=getattr(self.inner, "_archive_handle", None),
            )
            private_read = read_ref.kind in _CHILD_PRIVATE_RESOURCE_KINDS or private_path
        if self.spec is not None and (name == "search_history" or private_read):
            # W2 carve-out: an EXACT granted handle passes through (read_file/grep on that one file only —
            # never list_files, never a directory, never index.md; those can't be granted). W4' carve-out:
            # a standing specialist may read ITS OWN roster/<name>/ files (career, lessons, profile) —
            # self-memory, not a channel. Everything else in the reserved namespaces stays default-deny.
            p = canonical_path
            if name in ("read_file", "grep") and p in self.grants:
                return self.inner.run(name, args)
            if self.instance_name and (p == f"roster/{self.instance_name}"
                                       or p.startswith(f"roster/{self.instance_name}/")):
                return self.inner.run(name, args)
            hint = (" Your granted input reports: " + ", ".join(sorted(self.grants)) + "."
                    if self.grants else "")
            own = (f" Your own past work is under roster/{self.instance_name}/."
                   if self.instance_name else "")
            return ("Error: artifacts/, history/, subagents/ and roster/ (and search_history over them) are the "
                    "parent's private namespaces — a subagent works only from its own task/brief."
                    + hint + own + " If you lack context, say so in your report.")
        if name not in ("spawn_subagent", "spawn_explore", "spawn_agent"):
            return self.inner.run(name, args)
        if self.depth >= self.max_depth:
            return "Error: subagent depth limit reached"
        task = (args.get("task") or "").strip()   # #59: missing/empty 'task' → clear error, not a KeyError
        if not task:
            return "Error: spawn requires a non-empty 'task' describing the self-contained sub-task"
        child_name = (args.get("name") or "").strip()
        if self.core_mode and (child_name or args.get("grants")):
            return "Error: core delegation is one-shot and does not expose names, careers, or artifact grants"
        if child_name and not _valid_instance_name(child_name):
            return ("Error: invalid subagent name %r — use a short slug (letters/digits/-/_, starts with a "
                    "letter, ≤40 chars; 'sub-N'/'index' are reserved), e.g. 'auth-explorer'." % child_name)
        err, child_grants = self._validate_grants(args.get("grants"))
        if err:
            return err
        if name == "spawn_agent":
            spec = self.agents.get(args.get("agent", ""))
            if spec is None:
                return ("Error: unknown agent %r. Available: %s"
                        % (args.get("agent", ""), ", ".join(self.agents)))
        else:   # back-compat built-in tools → their specs
            spec = BUILTIN_AGENTS["explorer" if name == "spawn_explore" else "general"]
        if self.core_mode and (name != "spawn_agent" or spec.name != "explorer"):
            return ("Error: core delegation exposes only spawn_agent(agent='explorer'); enable "
                    "AGENT_ADVANCED_AGENTS for writable or legacy delegation")
        try:
            child_brief = self._brief(task, args, child_grants)
        except Exception as exc:  # noqa: BLE001 — never delegate after silently losing/warping constraints
            return f"Error: invalid subagent brief: {type(exc).__name__}: {exc}"

        # W4' — HIRE ONCE, WAKE MANY. A NAMED spawn resolves against the durable roster:
        #   roster hit  → WAKE: same kind required; the child is seeded with its identity block
        #                 (career manifest + lessons + abstention self-model), all bounded.
        #   miss        → HIRE: mint the standing identity (cap-gated — the kernel can say no).
        # Without a durable vault (NullMemory) hire returns {} and the named child runs as a temp.
        identity_block, hired = "", False
        if child_name and self.memory is not None:
            profile = self.memory.roster_get(child_name)
            if not profile:
                # ATOMIC get-or-create (no cap — the roster is unbounded). Under a concurrent same-name race
                # the loser gets the WINNER's profile back, so the kind-stability check below runs against the
                # authoritative identity, never a phantom the caller thought it created.
                profile = self.memory.roster_hire(child_name, spec.name)
                if profile:
                    hired = bool(profile.pop("_created", False))   # ONLY the creating caller announces the hire
                # else: {} from a memory with NO durable roster (NullMemory) or a transient write failure →
                # run as a session TEMP (the name still labels this seal; no standing identity accrues).
            if profile:
                if profile.get("kind") != spec.name:   # identity is kind-stable; waking as another kind lies
                    return (f"Error: {child_name!r} is a standing {profile.get('kind')!r} specialist — wake "
                            f"it with spawn_agent(agent={profile.get('kind')!r}, name={child_name!r}, ...) "
                            f"or pick a new name for a {spec.name!r}.")
                if not hired:   # an EXISTING specialist → seed with its career; a fresh hire has none yet
                    identity_block = _render_wake_block(profile, self.memory.roster_read_jobs(child_name),
                                                        child_name)

        try:
            _root = self.inner.root() if hasattr(self.inner, "root") else os.getcwd()
        except Exception:  # noqa: BLE001
            _root = os.getcwd()
        _task_id = self._context_id(self.task_id_fn, "task-unknown")
        _parent_id = self._context_id(self.parent_id_fn, "")
        _artifact_id, _launch_ordinal = self._next_artifact_identity(
            task_id=_task_id, parent_id=_parent_id,
        )

        child_tools = SubagentHost(
            self.inner, llm=self.llm, retriever=self.retriever, memory=self.memory,
            policy=self.policy, max_depth=self.max_depth, max_steps=self.max_steps,
            depth=self.depth + 1, notify=self.notify, spec=spec, agents=self.agents,
            session_id=self.session_id,   # nested children archive under the SAME parent session
            grants=child_grants,          # W2: one hop only — this child's grants never propagate further
            instance_name=child_name,     # W4': unlocks the child's OWN roster/<name>/ files (self-memory)
            intent_provider=self.intent_provider, task_id_fn=self.task_id_fn,
            # Nested children attach to this immediate child's immutable artifact, not the top-level turn.
            # That also makes sibling nested ID namespaces disjoint even though each child host starts at 1.
            parent_id_fn=((lambda artifact_id=_artifact_id: artifact_id)
                          if _artifact_id else self.parent_id_fn),
            workspace_id=self.workspace_id,
            artifact_store=self.artifact_store, artifact_ref_sink=self.artifact_ref_sink,
            core_mode=self.core_mode,
        )
        try:
            out = run_subagent(
                task, tools=child_tools, llm=self.llm, retriever=self.retriever,
                memory=self.memory, policy=self.policy, max_steps=self.max_steps,
                depth=self.depth + 1, notify=self.notify, spec=spec, session_id=self.session_id,
                name=child_name, grants=tuple(child_grants), identity_block=identity_block,
                brief=child_brief, workspace_id=self.workspace_id or os.path.realpath(_root),
                task_id=_task_id, parent_id=_parent_id,
                artifact_store=self.artifact_store,
                artifact_id=_artifact_id,
                artifact_ref_sink=self.artifact_ref_sink,
                token_budget=(max(0, int(token_budget)) if token_budget is not None else None),
                launch_ordinal=_launch_ordinal,
            )
            # announce the lifecycle event (visibility: an unadvertised wake channel stays dead) — but NOT
            # onto a failed child's "Error: ..." return, where it would garble the parent's error tier (the
            # hire is real regardless; it just isn't news worth mixing into an error line).
            if hired and not out.startswith("Error:"):
                suffix = f' | hired standing specialist {child_name!r} — re-use name="{child_name}" to wake it later'
                if hasattr(out, "effects"):
                    from .registry import ToolText
                    out = ToolText(str(out) + suffix, status=out.status, effects=out.effects)
                else:
                    out += suffix
            return out
        except Exception as e:  # a child failure must not crash the parent
            return f"Error: subagent crashed: {e}"
