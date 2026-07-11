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
`note` arg on a non-failing call is "tool-note"; child fan-in is "delegated" testimony; an unsupported tool
note is a "claim". Assistant replies remain verbatim only in bounded continuity and immutable turn artifacts.
Load-bearing conclusions therefore cross turns through typed evidence rather than a shadow transcript.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from .events import (AssistantText, Event, StepBegin, StepEnd, ToolResult,
                     ToolStarted, TurnEnd, TurnInterrupted)
from .intent import IntentState
from .execution import reconciliation_targets
from .regions import (
    MAX_CONVERSATION,
    MAX_PLAN_CHARS,
    MAX_PLAN_ITEMS,
    record_action,
    record_note,
)
from .slice_state import (ContinuityState, EvidenceState, TaskProgress,
                          TurnRuntime, WorkingSet)
from .subagent_contract import SubagentClaim
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
# code_ops (the anti-loop tally's view THROUGH execute_code) lives in regions.py with action_sig/
# record_action. paths_in_code/edited_paths_in_code stay here (slice_sink is their only caller).


def paths_in_code(code: str) -> list[str]:
    return _CODE_PATH_RE.findall(code or "")


def edited_paths_in_code(code: str) -> list[str]:
    return _CODE_EDIT_PATH_RE.findall(code or "")


@dataclass
class Slice:
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
                contract=None) -> None:
    """Append the user's message to the short-range CONVERSATION ring and count the turn. The host
    calls this once per user message; slice_sink fills the assistant side as the turn produces text.
    Bounded ring — older exchanges live in the durable cache, paged in on demand (not kept here)."""
    first_task_request = s.turns == 0 and not s.task.goal_source
    s.turns += 1
    s.turn_actions = 0   # new user turn → reset the per-turn exploration budget (drives the explore-nudge)
    # ONE authoritative verbatim request for the active turn. Persistent clauses are promoted separately
    # into intent.entries; the raw full message is archived by the turn sink rather than accumulated here.
    s.intent.begin_turn(message, source_artifact=source_artifact, contract=contract)
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
        lines.append(f"reconciliation-required: {one_line(s.reconciliation_required, 240)}")
    return "\n".join(lines)


def slice_sink(state):
    """Event sink that folds tool results back into the tiers (keeps the loop decoupled). `state`
    is a Slice or a Session — events fold into the CURRENT active slice (so a topic switch redirects
    subsequent folding)."""
    def sink(event: Event) -> None:
        # Host/presentation lifecycle events share the dispatcher but do not reduce task state. Reject them
        # before resolving an active Slice so a missing task can still report a commit failure to observers.
        if not isinstance(event, (
            StepBegin, StepEnd, TurnEnd, TurnInterrupted, ToolStarted, AssistantText, ToolResult,
        )):
            return
        s = _active(state)
        if isinstance(event, StepBegin):
            s.runtime.step = event.step
            return
        if isinstance(event, StepEnd):
            for key, value in (event.usage or {}).items():
                if isinstance(value, (int, float)):
                    s.runtime.usage[key] = s.runtime.usage.get(key, 0) + value
            return
        if isinstance(event, TurnEnd):
            # A clean model completion is not user acceptance and does not close the topic.  It only demotes
            # the original objective once no explicit unresolved state says it must remain an outstanding
            # instruction.  Binding intent remains independently resident; a later continue/failure can
            # reactivate the objective exactly.
            open_plan = any(
                not isinstance(item, dict) or item.get("status") != "done"
                for item in (s.plan or ())
            )
            unresolved = bool(
                s.last_error or s.open_report or s.reconciliation_required
                or s.intent.open_entries() or open_plan
            )
            if event.stop_reason == "end_turn" and not unresolved:
                s.task.mark_objective_provisional()
            else:
                s.task.activate_objective()
            return
        if isinstance(event, TurnInterrupted):
            s.task.activate_objective()
            if event.reason == "indeterminate" and not s.reconciliation_required:
                detail = one_line(event.message or "an operation may still have side effects", 400)
                s.reconciliation_required = detail
                s.reconciliation_targets = ["workspace:*", "opaque:interrupted-turn"]
            return
        if isinstance(event, ToolStarted):
            call_id = getattr(getattr(event, "invocation", None), "id", "")
            call = next((item for item in reversed(s.runtime.recent_calls)
                         if call_id and item.get("id") == call_id), None)
            if call is None:
                s.runtime.recent_calls.append({
                    "id": call_id, "name": event.name, "args": dict(event.args or {}),
                    "status": "running", "step": s.runtime.step,
                })
            elif str(call.get("status") or "") == "running":
                # A replayed start is the same lifecycle edge, not a second logical request. Never downgrade
                # a terminal row if a recovery host delivers an old start after its settled outcome.
                call.update(name=event.name, args=dict(event.args or {}), step=s.runtime.step)
            return
        # I1 PROVENANCE — generic assistant prose is continuity, not evidence. The bounded recent ring
        # resolves short-range references; the episode/artifact sink archives the full reply. Durable
        # findings enter only through explicit typed tool evidence below.
        if isinstance(event, AssistantText):
            if s.conversation and (event.content or "").strip():
                # fill the assistant side of the in-progress exchange — the LAST AssistantText of the
                # turn wins, so this ends up holding the final reply shown to the user (continuity).
                full = event.content
                # VERBATIM (including whitespace, NOT truncated): the last MAX_CONVERSATION turns keep their
                # FULL reply so a next-turn back-reference ("go with your recommendation") resolves against the
                # real conclusion — which usually sits at the TAIL, exactly what a head-gist used to sever. The
                # bound is the turn COUNT, not bytes; older turns page out to history/ (recall pages them back).
                s.conversation[-1]["assistant"] = str(full)
                if event.final:
                    # A terse next-turn assent ("yes" / "go ahead") gains action authority only from an
                    # explicit, immediately preceding offer.  This small continuity object is not a transcript
                    # and is replaced/cleared by every terminal assistant response.
                    from .discourse import extract_pending_proposal
                    s.continuity.pending_proposal = extract_pending_proposal(full)
            return
        if isinstance(event, ToolResult):
            # Reject an ambiguous partial effect replay before changing even the ephemeral runtime ledger.
            # Complete effect replay is safe: semantic reduction is skipped below, while a distinct logical
            # invocation ID still receives its own terminal accounting row.
            _effect_ids = tuple(
                effect.id for effect in (getattr(getattr(event, "outcome", None), "effects", ()) or ())
                if getattr(effect, "id", "")
            )
            _seen_effect_ids = s.runtime.applied_effect_ids.intersection(_effect_ids)
            if _seen_effect_ids and len(_seen_effect_ids) != len(set(_effect_ids)):
                raise RuntimeError("partially replayed tool outcome cannot be reduced safely")
            effects_replayed = bool(_effect_ids and len(_seen_effect_ids) == len(set(_effect_ids)))

            # Account for the LOGICAL invocation before reducing its semantic effects. A provider may issue
            # the same call twice and receive the same cached/idempotent effect for both invocation IDs. The
            # world-state mutation must still apply exactly once, but completion invariants (for example,
            # "spawn exactly 3 children") must see both requests and outcomes. Exact host replay of the same
            # invocation ID remains one runtime row because the lookup below updates it in place.
            outcome_invocation = getattr(getattr(event, "outcome", None), "invocation", None)
            call_id = event.invocation_id or getattr(outcome_invocation, "id", "")
            call = next((item for item in reversed(s.runtime.recent_calls)
                         if call_id and item.get("id") == call_id), None)
            prior_status = str(call.get("status") or "") if call is not None else ""
            accept_terminal_metadata = not prior_status or prior_status == "running"
            if call is None:
                call = {
                    "id": call_id,
                    "name": event.name,
                    "args": dict(event.args or {}),
                    "step": s.runtime.step,
                }
                s.runtime.recent_calls.append(call)
            call["status"] = event.status or ("failed" if event.failing else "succeeded")
            newly_settled = not prior_status or prior_status == "running"

            # Preserve the bounded child-claim ledger on the logical invocation that produced it. The effect says
            # only what occurred verbatim in the sealed child report; it is runtime fan-in metadata, never a
            # workspace observation. Normalize and cap defensively because effect payloads can also come from
            # third-party tool hosts.
            for effect in (() if not accept_terminal_metadata or event.name not in {
                    "spawn_agent", "spawn_explore", "spawn_subagent",
            } else
                           (getattr(getattr(event, "outcome", None), "effects", ()) or ())):
                if getattr(effect, "kind", "") != "child_artifact":
                    continue
                payload = getattr(effect, "payload", {}) or {}
                artifact_id = str(payload.get("artifact_id") or "")
                if artifact_id:
                    call["child_artifact_id"] = artifact_id[:200]
                target = payload.get("delegation_target")
                if isinstance(target, str) and target.strip() and len(target.encode("utf-8")) <= 300:
                    call["child_target"] = target.strip()
                raw_scope = payload.get("scope") or ()
                if isinstance(raw_scope, (list, tuple)) and len(raw_scope) <= 16 and all(
                        isinstance(item, str) and item.strip() and len(item.encode("utf-8")) <= 300
                        for item in raw_scope):
                    call["child_scope"] = list(dict.fromkeys(item.strip() for item in raw_scope))
                normalized_claims = []
                raw_claims = payload.get("claims") or ()
                valid_claims = isinstance(raw_claims, (list, tuple)) and len(raw_claims) <= 3
                if valid_claims:
                    for row in raw_claims:
                        if (not isinstance(row, Mapping)
                                or not isinstance(row.get("text"), str)
                                or not isinstance(row.get("report_exact"), str)):
                            valid_claims = False
                            break
                        try:
                            normalized_claims.append(SubagentClaim.from_dict(row).to_dict())
                        except (TypeError, ValueError):
                            valid_claims = False
                            break
                if not valid_claims:
                    normalized_claims = []
                if normalized_claims:
                    call["child_claims"] = normalized_claims
                break

            # Canonical typed outcomes may be delivered again by a replaying host. Stable effect IDs make
            # reduction exactly-once within the active turn; legacy events without typed effects preserve
            # their historical behavior. A partially repeated effect set is inconsistent and must fail the
            # required reducer rather than applying an ambiguous subset. Logical accounting above is not an
            # effect: it intentionally records distinct invocation IDs even when their effects are identical.
            if newly_settled and call["status"] == "indeterminate":
                detail = (
                    f"{event.name} ({call_id or 'unknown invocation'}) returned an indeterminate outcome; "
                    "re-observe the affected workspace/process state before any further side effect"
                )
                if detail not in s.reconciliation_required:
                    s.reconciliation_required = " | ".join(
                        item for item in (s.reconciliation_required, detail) if item
                    )
                s.reconciliation_targets = list(dict.fromkeys((
                    *s.reconciliation_targets, *reconciliation_targets(event.name, event.args),
                )))
            if newly_settled and event.failing:
                s.runtime.blocked_calls += 1
            if effects_replayed or not event.apply_effects:
                return
            # the model's distilled conclusion rides on the tool call (the note arg) — fold it into
            # the FINDINGS tier so a reasoning model reuses it instead of re-deriving next turn. A note
            # on a NON-FAILING call is backed by a real tool result (source "tool-note"); a note on a
            # FAILING call has no observation behind it → "claim" (rendered as unverified). record_note
            # further downgrades any "done"-style note to "claim" unless an observation backs it.
            new_finding = record_note(s, event.args.get("note", ""),
                                      source="tool-note" if not event.failing else "claim")
            # WORLD MODEL — fold world_set/world_clear into the durable scratchpad (the note→findings seam,
            # but structured key→value). The tool handler only confirms; the STATE lives here so it renders
            # into each turn's seed, survives the seal, and clears on reset.
            if event.name == "reconcile_execution" and not event.failing:
                resolution = one_line(str(event.args.get("resolution") or "state re-observed"), 300)
                s.reconciliation_required = ""
                s.reconciliation_targets = []
                s.task.add_progress("reconciliation", resolution)
                s.last_error = ""
            elif event.name == "world_set" and not event.failing:
                _k = str(event.args.get("key", "")).strip()
                if _k:
                    s.world[_k] = str(event.args.get("value", ""))
            elif event.name == "world_clear" and not event.failing:
                _k = str(event.args.get("key", "")).strip()
                if _k:
                    s.world.pop(_k, None)
                else:
                    s.world.clear()
            # STANDING INTENT — fold the legacy require/requirement_done/drop_requirement tool names into
            # the typed ledger. The clause itself is never count- or character-capped. An exact substring
            # of CURRENT REQUEST gets user authority + a source range; a model paraphrase remains lower-
            # authority task state. `done` is provisional (model prose cannot finalize user acceptance),
            # and a model-issued drop cannot retire a user-authored clause.
            elif event.name in ("require", "requirement_done", "drop_requirement",
                                "supersede_requirement") and not event.failing:
                _t = str(event.args.get("text", "")).strip()
                if event.name == "supersede_requirement":
                    old = str(event.args.get("old_text", "")).strip()
                    new = str(event.args.get("new_text", "")).strip()
                    start = s.intent.current_request.find(new) if new else -1
                    if old and new and start >= 0:
                        s.intent.supersede_from_user(
                            old, new, source_artifact=s.intent.current_source,
                            source_range=(start, start + len(new)),
                        )
                elif _t:
                    if event.name == "require":
                        s.intent.add_from_current_request(_t)
                    elif event.name == "requirement_done":
                        excluded = {"require", "requirement_done", "drop_requirement",
                                    "supersede_requirement", "update_plan", "world_set", "world_clear"}
                        prior = next((call for call in reversed(s.runtime.recent_calls[:-1])
                                      if call.get("status") == "succeeded"
                                      and call.get("name") not in excluded and call.get("id")), None)
                        if prior is not None:
                            s.intent.mark_provisional(_t, evidence_refs=(f"invocation:{prior['id']}",))
                    else:                                             # drop_requirement
                        s.intent.defer_model_entry(_t)
            # PLAN (TodoWrite) — fold update_plan: the model sends the FULL ordered list each call, so this
            # REPLACES s.plan (validated + bounded). Distinct from requirements (criteria); this is the step
            # sequence + live progress. Replace-all keeps it simple and always consistent with the model's view.
            elif event.name == "update_plan" and not event.failing:
                _new = []
                for _it in (event.args.get("steps") or [])[:MAX_PLAN_ITEMS]:
                    if not isinstance(_it, dict):
                        continue
                    _step = " ".join(str(_it.get("step", "")).split())[:MAX_PLAN_CHARS]
                    _st = str(_it.get("status", "pending")).strip().lower()
                    if _st not in ("pending", "in_progress", "done"):
                        _st = "pending"
                    if _step:
                        _new.append({"step": _step, "status": _st})
                s.plan = _new
            # FAN-IN: this output deliberately mixes a child interpretation, a bounded primary excerpt, and an
            # immutable handle.  A successful spawn proves that the testimony was returned and sealed; it does
            # NOT make every sentence an observed workspace fact. Carry it under the delegated trust tag so it
            # survives the turn-boundary seal without provenance laundering (the full report remains recallable
            # through its handle; the parent still never receives the child transcript).
            if event.name in ("spawn_subagent", "spawn_explore", "spawn_agent") and not event.failing and event.output:
                new_finding = record_note(s, event.output, source="delegated") or new_finding
            did_edit = False
            if event.name == "skill" and not event.failing:
                # a loaded skill's body must enter the ACTIVE SKILL tier or it vanishes
                # next turn (no transcript). The skill tool returns the body as its output.
                add_skill(s, event.args.get("name", ""), event.output)
            # list_files/grep/glob "path" is a DIRECTORY scope, not a working-set FILE — don't pin it (else
            # build_artifacts read_text(dir) → IsADirectoryError → a bogus OPEN FILES entry every turn).
            if event.args.get("path") and event.name not in ("list_files", "grep", "glob"):
                did_edit = event.name in ("edit_file", "append_to_file", "str_replace") and not event.failing
                resource_effect = next((
                    effect for effect in (
                        getattr(getattr(event, "outcome", None), "effects", ()) or ()
                    )
                    if getattr(effect, "kind", "") == "resource_observed"
                ), None)
                resource_kind = str(
                    getattr(resource_effect, "payload", {}).get("resource_kind", "")
                    if resource_effect is not None else ""
                )
                virtual_read = (
                    event.name == "read_file" and resource_kind in {
                        "artifact", "history", "subagent", "roster",
                    }
                )
                # WS1 — gate membership on SUCCESS. A read/edit that FAILED (e.g. _resolve raised
                # "path escapes workspace") must NOT be pinned into the working set, or OPEN FILES
                # re-renders the unreachable/missing path every rebuild and poisons the slice
                # (the read-blindness loop). Successful PHYSICAL reads and edits still join the set.
                # Archive reads remain typed virtual observations in the canonical ToolOutcome; pinning
                # their handles here would make seed.py physically re-read them and lie "not created yet".
                if not event.failing and not virtual_read:
                    touch_file(s, event.args["path"], edited=did_edit)
                # remember the agent's edit target so a HUGE file's region follows it (and a
                # failed str_replace re-aims the region to where the agent meant to edit)
                if event.name == "str_replace" and event.args.get("old_string"):
                    anchor = next((ln.strip() for ln in event.args["old_string"].splitlines() if ln.strip()), "")
                    if anchor:
                        s.edit_anchor[event.args["path"]] = anchor[:80]
            elif event.name == "execute_code":
                code = event.args.get("code", "")
                mutated = set(edited_paths_in_code(code))
                did_edit = bool(mutated) and not event.failing
                # WS1 parity with the file-tool branch above: gate working-set/change-set membership on
                # SUCCESS. A FAILED execute_code (e.g. NameError before write_file ran) must NOT pin the
                # files it MENTIONS — that poisons edited_files with phantom writes (false "done") and
                # active_files with phantom reads (read-blindness), and survives the seal.
                if not event.failing:
                    for p in paths_in_code(code):
                        touch_file(s, p, edited=(p in mutated))  # code-as-action edits join the change set
            record_action(s, event.name, event.args, event.output, failing=event.failing)
            # Cross-turn progress keeps only coalesced semantic signals, never the raw invocation/output.
            # Detailed call history belongs to TurnRuntime/the sealed artifact and resets at the boundary.
            if event.failing:
                s.task.add_progress("blocked", f"{event.name} failed")
            if did_edit:
                s.task.add_progress("edit", str(event.args.get("path") or event.name))
            if new_finding:
                s.task.add_progress("evidence", f"new evidence from {event.name}")
            # convergence tracking: a real edit OR a genuinely-new finding resets the spin counter —
            # actively LEARNING (recording new facts) is progress, not spinning (review #5). Only a call
            # that neither edits nor learns advances the convergence/no-progress counter.
            s.since_edit = 0 if (did_edit or new_finding) else s.since_edit + 1
            s.runtime.applied_effect_ids.update(_effect_ids)
    return sink
