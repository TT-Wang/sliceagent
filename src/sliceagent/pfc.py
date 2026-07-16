"""PFC — the Active Memory Slice's own carried state (the working-memory brain region).

NORTH STAR — the slice is a CACHE, not a log. Every model call is a pure function
f(selector, store): the durable stores (disk, code graph, episode cache) are the only
authority; the slice is a small typed SELECTOR over them, reconstructed ONCE PER TURN as the
SEED (see seed.py). Within the turn, working memory ACCUMULATES as native assistant/tool
messages — no per-step rebuild, no within-turn eviction; the bound is the TURN-BOUNDARY seal
(the next turn starts from a fresh seed + recall). The single invariant "cache not log" IMPLIES
the moat (a cache keeps no history), task-agnosticism (a cache doesn't know what it caches), and
LLM-agnosticism (the cache contract sits below the model).

IN BRAIN TERMS (a naming aid — see pagetable.py for the fuller legend): the Slice's own carried
state (findings, conversation ring, plan — see seal() below) is PREFRONTAL CORTEX /
working memory: bounded, actively maintained, free, lost on reset. This module owns exactly
that region: the `Slice` dataclass, its lifecycle (reset/seal), and the functions that MUTATE
it in place (touch_file, add_skill, record_user, consolidate_checkpoint, slice_sink). The
reconstruction seam that READS durable stores to build a turn's SEED lives in seed.py; the
stable SYSTEM prompt text lives in prompt.py.

PROVENANCE (Invariant 1): a finding is tagged by where it came from, and generic model prose is never
promoted into evidence. Each explicit finding carries a `source`: a direct tool result is "observed"; the
`note` arg on a non-failing call is "tool-note"; legacy child projections may be "delegated" testimony; an
unsupported tool note is a "claim". New child reports remain ordinary tool results instead of being copied
into PFC findings. Assistant replies remain verbatim only in bounded continuity and immutable turn artifacts.
Load-bearing conclusions therefore cross turns through typed evidence rather than a shadow transcript.
"""
from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from .active_work import WorkGraph
from .intent import IntentState
from .regions import MAX_CONVERSATION
from .slice_state import (ContinuityState, EvidenceState, TaskProgress,
                          TurnRuntime, WorkingSet)
from .swap import READ_BUDGET, READ_BUDGET_MAX, _DEFAULT_SWAP
from .text_utils import one_line

# literal paths the model touches via execute_code helpers — so code-as-action reads/edits
# still populate the OPEN FILES working set (they run in the sandbox, bypassing the ToolHost)
_CODE_PATH_RE = re.compile(
    r"\b(?:read_file|write_file|append_file|str_replace)\(\s*['\"]([^'\"]+)['\"]"
)
# the subset that MUTATES a file (vs read_file) — so code-as-action edits join the protected change set
_CODE_EDIT_PATH_RE = re.compile(
    r"\b(?:write_file|append_file|str_replace)\(\s*['\"]([^'\"]+)['\"]"
)
_VERIFY_TARGET = re.compile(
    r"^(?:run[-_]?)?(?:tests?|checks?|lint|verify|build|type[-_]?checks?)(?:[-_:].*)?$",
    re.I,
)
_VERIFY_SCRIPT = re.compile(
    r"^(?:run[-_]?)?(?:tests?|checks?|lint|verify|build|type[-_]?checks?)(?:[-_.].*)?"
    r"(?:\.sh|\.py|\.js|\.mjs|\.cjs|\.rb|\.ps1)?$",
    re.I,
)
_REPORT_FAMILY_PATTERNS = {
    "test": re.compile(r"\b(?:tests?|test\s+suite|pytest|unittest|vitest|jest|ctest|suite\s+(?:fails?|failed))\b", re.I),
    "lint": re.compile(r"\b(?:lint(?:er|ing)?|ruff|eslint)\b", re.I),
    "type": re.compile(r"\b(?:type[- ]?checks?|typechecking|static\s+typing|mypy|tsc)\b", re.I),
    "build": re.compile(r"\b(?:build|compile|compilation)\b", re.I),
    "generic": re.compile(r"\b(?:checks?|verify|verification|command\s+(?:fails?|failed))\b", re.I),
}
_NON_EXECUTING_VERIFY_FLAGS = {
    "--help", "--version",
    "--collect-only", "--co", "--setup-plan", "--fixtures", "--markers",
    "--setup-only", "--fixtures-per-test", "--showconfig", "--show-config",
    "--show-settings", "--show-files", "--print-config", "--env-info",
    "--list", "--list-tests", "--listtests", "--listenvs", "--list-sessions",
    "--show-only",
    "--if-present", "--dry-run", "--just-print", "--no-run", "--passwithnotests",
    "--exit-zero", "--nocheck", "--fail-never",
}
# The typed SliceReducer imports these small source-analysis helpers.


def paths_in_code(code: str) -> list[str]:
    return _CODE_PATH_RE.findall(code or "")


def edited_paths_in_code(code: str) -> list[str]:
    return _CODE_EDIT_PATH_RE.findall(code or "")


def _label_verification_family(label: str) -> str | None:
    label = str(label or "").rsplit("/", 1)[-1].casefold()
    label = re.sub(r"^(?:run[-_]?)", "", label)
    if re.match(r"^tests?(?:[-_.:]|$)", label):
        return "test"
    if re.match(r"^lint(?:[-_.:]|$)", label):
        return "lint"
    if re.match(r"^type[-_]?checks?(?:[-_.:]|$)", label):
        return "type"
    if re.match(r"^build(?:[-_.:]|$)", label):
        return "build"
    if re.match(r"^(?:checks?|verify)(?:[-_.:]|$)", label):
        return "generic"
    return None


def _is_nonexecuting_verify_flag(token: str) -> bool:
    token = str(token or "")
    lowered = token.casefold()
    if lowered.startswith("--") and lowered.split("=", 1)[0] in _NON_EXECUTING_VERIFY_FLAGS:
        return True
    # Pytest accepts combined short options: ``-qh`` and ``-qV`` both exit 0 after displaying
    # help/version without running a test. Lowercase ``-v`` remains ordinary verbosity.
    return token.startswith("-") and not token.startswith("--") \
        and any(flag in token[1:] for flag in ("h", "V"))


def _verification_families_argv(tokens: list[str]) -> set[str]:
    tokens = list(tokens)
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens.pop(0)
    if not tokens:
        return set()
    if tokens[0] == "env":
        if any(_is_nonexecuting_verify_flag(token) for token in tokens[1:]):
            return set()
        tokens.pop(0)
        while tokens and (tokens[0].startswith("-")
                          or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0])):
            tokens.pop(0)
    while len(tokens) >= 2 and (
        (tokens[0] in {"uv", "poetry", "pipenv"} and tokens[1] == "run")
        or (tokens[0] in {"pnpm", "yarn"} and tokens[1] == "exec")
    ):
        tokens = tokens[2:]
    if tokens and tokens[0] in {"npx", "bunx"}:
        tokens = tokens[1:]
    if not tokens:
        return set()
    program = tokens[0].rsplit("/", 1)[-1].casefold()
    rest = tokens[1:]
    # A zero exit from these modes proves only that discovery/help/configuration succeeded.  It does not
    # prove that the named test, lint, or build actually ran, so it must not close a user-reported defect.
    if any(_is_nonexecuting_verify_flag(token) for token in rest):
        return set()
    if program in {"bash", "sh", "zsh"}:
        index = 0
        while index < len(rest):
            token = rest[index]
            if token == "--":
                index += 1
                break
            if not token.startswith("-") or token == "-":
                break
            if token.startswith("--"):
                if token not in {"--noprofile", "--norc", "--posix", "--restricted",
                                 "--verbose", "--login"}:
                    return set()
                index += 1
                continue
            flags = token[1:]
            if "n" in flags:
                return set()
            if "c" in flags:
                return (_command_verification_families(rest[index + 1])
                        if index + 1 < len(rest) else set())
            if any(flag not in "abefhklmptuvxBCEHPT" for flag in flags):
                return set()
            index += 1
        positional = rest[index] if index < len(rest) else ""
        if positional and _VERIFY_SCRIPT.fullmatch(positional.rsplit("/", 1)[-1]):
            family = _label_verification_family(positional)
            return {family} if family else set()
        return set()
    if program in {"pytest", "py.test", "unittest", "vitest", "jest", "tox", "nox", "ctest"}:
        if program in {"tox", "nox"} and any(
                token == "-l" or (token.startswith("-") and not token.startswith("--")
                                    and "l" in token[1:]) for token in rest):
            return set()
        if program == "ctest" and any(
                token == "-N" or (token.startswith("-") and not token.startswith("--")
                                    and "N" in token[1:]) for token in rest):
            return set()
        return {"test"}
    if program in {"ruff", "eslint"}:
        return {"lint"}
    if program in {"mypy", "tsc"}:
        return {"type"}
    if program in {"python", "python3", "pypy", "pypy3"}:
        if "-c" in rest:
            return set()
        if "-m" in rest:
            index = rest.index("-m")
            return ({"test"} if index + 1 < len(rest)
                    and rest[index + 1] in {"pytest", "unittest"} else set())
        script = next((token for token in rest if not token.startswith("-")), "")
        if script and _VERIFY_SCRIPT.fullmatch(script.rsplit("/", 1)[-1]):
            family = _label_verification_family(script)
            return {family} if family else set()
        return set()
    if program in {"npm", "pnpm", "yarn"}:
        targets = [token for token in rest if token not in {"run", "--"} and not token.startswith("-")]
        if targets and _VERIFY_TARGET.fullmatch(targets[0]):
            family = _label_verification_family(targets[0])
            return {family} if family else set()
        return set()
    if program in {"make", "just", "task"}:
        if any(token in {"-q", "--question", "--recon", "--ignore-errors"} or (
                token.startswith("-") and not token.startswith("--")
                and any(flag in token[1:] for flag in ("n", "i"))
        ) for token in rest):
            return set()
        return {
            family for token in rest if not token.startswith("-")
            if _VERIFY_TARGET.fullmatch(token)
            for family in (_label_verification_family(token),) if family
        }
    if program in {"cargo", "go", "dotnet", "gradle", "gradlew", "mvn", "mvnw"}:
        lowered_rest = [token.casefold() for token in rest]
        if program == "go" and any(
                token == "-list" or token.startswith("-list=") for token in lowered_rest):
            return set()
        if program in {"mvn", "mvnw"} and any(
                token == "-dskiptests" or token.startswith("-dskiptests=")
                or token == "-dmaven.test.skip" or token.startswith("-dmaven.test.skip=")
                or token == "-fn"
                for token in lowered_rest):
            return set()
        if program in {"gradle", "gradlew"} and any(
                token in {"-x", "--exclude-task"} and index + 1 < len(lowered_rest)
                and lowered_rest[index + 1] == "test"
                for index, token in enumerate(lowered_rest)):
            return set()
        families = set()
        for token in rest:
            target = token.casefold()
            if target == "test":
                families.add("test")
            elif target == "build":
                families.add("build")
            elif target == "check":
                families.add("build" if program in {"cargo", "go", "dotnet"} else "generic")
            elif target == "verify":
                families.add("generic")
        return families
    if _VERIFY_SCRIPT.fullmatch(program) and ("/" in tokens[0] or "." in program):
        family = _label_verification_family(program)
        return {family} if family else set()
    return set()


def _command_verification_families(command: str) -> set[str]:
    command = str(command or "")
    # Without a shell-status receipt, only ``&&`` composes soundly: success proves every preceding command.
    # ``|| true``, pipelines, sequential commands, newlines, and background jobs can hide a failed test.
    if re.search(r"\|\||(?<!\|)\|(?!\|)|;|\r|\n|(?<!&)&(?!&)", command):
        return set()
    families = set()
    for segment in command.split("&&"):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        families.update(_verification_families_argv(tokens))
    return families


def _verification_families_call(name: str, args: Mapping) -> set[str]:
    if name == "run_command":
        return _command_verification_families(str((args or {}).get("command") or ""))
    return set()


def _report_verification_families(report: str) -> set[str]:
    return {
        family for family, pattern in _REPORT_FAMILY_PATTERNS.items()
        if pattern.search(str(report or ""))
    }


def _verification_matches_report(report: str, observed: set[str]) -> bool:
    required = _report_verification_families(report)
    specific = required - {"generic"}
    if specific:
        return specific.issubset(observed)
    return "generic" in required and bool(observed)


@dataclass
class Slice:
    # The one semantic owner for what remains to be handled.  It stores source/evidence/resource/output
    # locators and lifecycle, never a transcript or an inferred rewrite of the user's request.
    active_work: WorkGraph = field(default_factory=WorkGraph)
    intent: IntentState = field(default_factory=IntentState)
    task: TaskProgress = field(default_factory=TaskProgress)
    evidence: EvidenceState = field(default_factory=EvidenceState)
    work: WorkingSet = field(default_factory=lambda: WorkingSet(
        read_budget=READ_BUDGET, read_ceiling=READ_BUDGET_MAX))
    continuity: ContinuityState = field(default_factory=ContinuityState)
    runtime: TurnRuntime = field(default_factory=TurnRuntime)

    # ONE compatibility map, not a second state model. All legacy mutable attributes resolve directly to
    # their authoritative region object, so append/update/in-place mutations keep working during migration.
    _ALIASES: ClassVar[dict[str, tuple[str, str]]] = {
        "goal": ("task", "goal"), "plan": ("task", "plan"),
        "action_log": ("task", "action_log"), "world": ("task", "world"),
        "progress_signals": ("task", "progress_signals"),
        "findings": ("evidence", "findings"), "finding_source": ("evidence", "finding_source"),
        "last_error": ("evidence", "last_error"), "open_report": ("evidence", "open_report"),
        "reconciliation_required": ("evidence", "reconciliation_required"),
        "reconciliation_targets": ("evidence", "reconciliation_targets"),
        "active_files": ("work", "active_files"), "active_skills": ("work", "active_skills"),
        "edit_anchor": ("work", "edit_anchor"), "edited_files": ("work", "edited_files"),
        "ghosts": ("work", "ghosts"), "protected_deps": ("work", "protected_deps"),
        "pre_defs": ("work", "pre_defs"), "stale_deps": ("work", "stale_deps"),
        "io": ("work", "io"), "hot": ("work", "hot"),
        "read_budget": ("work", "read_budget"), "read_ceiling": ("work", "read_ceiling"),
        "conversation": ("continuity", "conversation"), "turns": ("continuity", "turns"),
        "since_edit": ("runtime", "since_edit"), "turn_actions": ("runtime", "turn_actions"),
        "explore_mode": ("runtime", "explore_mode"),
    }

    def __getattr__(self, name):
        alias = self._ALIASES.get(name)
        if alias is None:
            raise AttributeError(name)
        region, attr = alias
        return getattr(object.__getattribute__(self, region), attr)

    def __setattr__(self, name, value) -> None:
        alias = self._ALIASES.get(name)
        if alias is None:
            object.__setattr__(self, name, value)
            return
        region, attr = alias
        try:
            owner = object.__getattribute__(self, region)
        except AttributeError:
            object.__setattr__(self, name, value)
        else:
            if name == "last_error" and hasattr(owner, "last_error_identity"):
                owner.last_error_identity = ""
            setattr(owner, attr, value)

    @property
    def requirements(self) -> list[dict]:
        """Legacy read projection over typed intent (not a second mutable authority)."""
        return self.intent.as_legacy_requirements()

    @requirements.setter
    def requirements(self, value) -> None:
        # Supports old tests/checkpoint adapters that assign v1 [{text,done}] rows. Runtime mutations use
        # IntentState methods so appending to this projected list is intentionally not supported.
        self.intent.load_legacy_requirements(value or [])

    def reset(self, goal: str) -> None:
        self.active_work = WorkGraph()
        self.intent.reset(goal)
        self.task.reset(goal)
        self.evidence.reset()
        self.work.reset(read_budget=READ_BUDGET, read_ceiling=READ_BUDGET_MAX)
        self.continuity.reset()
        self.runtime.reset()

    def seal(self) -> None:
        """Delegate the turn boundary to the six semantic owners."""
        self.intent.seal()
        self.task.seal()
        self.evidence.seal()
        self.work.seal()
        self.continuity.seal()
        self.runtime.seal()

def touch_file(s: Slice, path: str, edited: bool = False) -> None:
    """Shim → SwapManager.load (swap.py owns the file load→evict→ghost lifecycle). Signature unchanged."""
    _DEFAULT_SWAP.load(s, path, edited=edited)


def add_skill(s: Slice, name: str, body: str) -> None:
    """Shim → SwapManager.load_skill (swap.py owns skill load/evict + ghosts). Signature unchanged."""
    _DEFAULT_SWAP.load_skill(s, name, body)


def _active(state):
    """Resolve the current Slice from a Slice or a Session (host-side topic manager)."""
    return state.active() if hasattr(state, "active") else state


def record_user(s: Slice, message: str, *, source_artifact: str | None = None,
                source_event_id: str | None = None, logical_id: str | None = None,
                workspace_epoch: int = 0, source_text: str | None = None, contract=None) -> None:
    """Append the user's message to the short-range CONVERSATION ring and count the turn. The host
    calls this once per user message; slice_sink fills the assistant side as the turn produces text.
    Bounded ring — older exchanges live in the durable cache, paged in on demand (not kept here)."""
    first_task_request = s.turns == 0 and not s.task.goal_source
    s.turns += 1
    s.turn_actions = 0   # new user turn → reset the per-turn exploration budget (drives the explore-nudge)
    # A deliverable belongs to one exact logical request. Preserve it across workspace segments (record_user is
    # not repeated there), but retire it before admitting an unrelated follow-up in the same task topic.
    requirement = s.task.deliverable_requirement
    if requirement is not None and logical_id and requirement.logical_id != logical_id:
        s.task.deliverable_requirement = None
    # ONE authoritative verbatim request for the active turn. Persistent clauses are promoted separately
    # into intent.entries; the raw full message is archived by the turn sink rather than accumulated here.
    s.intent.begin_turn(message, source_artifact=source_artifact, contract=contract)
    # Admission is deliberately mechanical: one exact source event becomes one request root.  The
    # caller binds to the application ledger's canonical (possibly persistence-redacted) bytes.  Older
    # embedding hosts that have no application ledger retain the legacy Slice path until they opt into this
    # explicit seam; silently treating a workspace-local artifact as a global event would break transitions.
    event_id = str(source_event_id or "").strip()
    canonical_source = str(message if source_text is None else source_text)
    if event_id and canonical_source:
        s.active_work = s.active_work.open_request(
            event_id,
            canonical_source,
            workspace_epoch=int(workspace_epoch),
            logical_id=str(logical_id or event_id),
        )
    if first_task_request and source_artifact:
        s.task.goal_source = source_artifact
    # RECENT CONVERSATION ring — VERBATIM (including whitespace, NOT truncated): the last few turns are the
    # active loop's antecedents, so a deictic follow-up ("go with your recommendation", "save this") resolves
    # against the real text, not a lossy gist. Count-bounded by MAX_CONVERSATION; older turns page out to history/.
    s.conversation.append({
        "user": str(message or ""), "assistant": "", "artifact_id": source_artifact or "",
    })
    s.conversation = s.conversation[-MAX_CONVERSATION:]


def consolidate_checkpoint(s: "Slice", *, compact: bool = True) -> str:
    """F1 — the CHECKPOINT: a deterministic, BOUNDED re-projection of the carried task state into ONE dense
    'state of play' snapshot (intent · decisions · change-set · open/next, plus a findings digest in full
    mode). Pure (no LLM) — built from the durable tiers seal() already carries, so it adds LEGIBILITY +
    a single resume/rebuild artifact, never new state. `compact=True` is the steady-state slice tier (no
    findings re-list — those have their own tier); `compact=False` is the FULL artifact for the overflow
    REBUILD (where the detailed tiers are gone, so the snapshot must stand alone). Self-suppresses when
    there is nothing to report (a fresh greeting → no bytes)."""
    from .finding_types import RULED_OUT, classify_finding  # typed decisions read sharper in the snapshot
    lines: list[str] = []
    goal = (s.intent.current_request or s.goal or "").strip()
    if goal:
        lines.append(f"intent: {one_line(goal, 240)}")
    open_reqs = [r.get("text", "") for r in s.requirements if isinstance(r, dict) and not r.get("done")]
    if open_reqs:
        lines.append("requirements: " + " · ".join(one_line(t, 80) for t in open_reqs[:5]))
    decisions = [f for f in s.findings if classify_finding(f) in ("decision", RULED_OUT)]
    if decisions:
        lines.append("decisions:")
        lines += [f"  - {one_line(d, 160)}" for d in decisions[-4:]]
    if s.edited_files:
        ch = sorted(s.edited_files)
        lines.append("change-set: " + ", ".join(ch[:8]) + (f" (+{len(ch) - 8})" if len(ch) > 8 else ""))
    if not compact:                                  # FULL artifact: include the non-decision findings digest
        facts = [f for f in s.findings if f not in decisions]
        if facts:
            lines.append("findings:")
            lines += [f"  - {one_line(f, 160)}" for f in facts[-8:]]
    if s.open_report:
        lines.append(f"open: {one_line(s.open_report, 200)}")
    if s.reconciliation_required:
        lines.append(f"execution-uncertainty: {one_line(s.reconciliation_required, 240)}")
    return "\n".join(lines)


def slice_sink(state):
    """Return the single typed reducer for the active Slice."""

    from .slice_reducer import SliceReducer

    return SliceReducer(state)
