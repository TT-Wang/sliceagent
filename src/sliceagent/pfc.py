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

PROVENANCE (Invariant 1): a finding is tagged by where it came from, and model prose is never an
established FACT. Each finding carries a `source` (observed > tool-note > claim): a tool result is
"observed"; the `note` arg on a non-failing call is "tool-note"; the model's free reasoning and any
"done"-style claim are "claim". ALL are folded forward so a reasoning model reuses them instead of
re-deriving (the costly Markov trap) — but rendered as the model's OWN notes to VERIFY against OPEN
FILES, never as "do not re-derive". Pure intent/narration ("Let me…") is dropped. This keeps the
"already done" ratchet dead (a claim is not a fact) WITHOUT starving carry-forward — dropping prose
entirely ~2x'd steps on normal tasks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .events import AssistantText, Event, ToolResult
from .regions import (
    MAX_CONVERSATION,
    MAX_FINDINGS,
    MAX_PLAN_CHARS,
    MAX_PLAN_ITEMS,
    MAX_REQUIREMENTS,
    MAX_REQ_CHARS,
    record_action,
    record_note,
)
from .swap import READ_BUDGET, READ_BUDGET_MAX, _DEFAULT_SWAP
from .text_utils import normalize_ws, one_line

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
    goal: str = ""
    # DURABLE TASK SPEC — the original defining request for this topic (the first user message), kept
    # WHOLE and resident for the life of the topic. Standing requirements live here (an exact function
    # name/signature, output format, "use British spelling", "don't use lib X", an invariant) — they are
    # STANDING REQUIREMENTS — the live contract that must hold when the task is DONE: a model-CURATED set
    # of constraints (an exact name/signature, an output format, a stated rule, an added requirement), NOT
    # the frozen first message. The model maintains it in-band via require / requirement_done /
    # drop_requirement (folded by slice_sink, the world_set seam — zero extra LLM call). CARRIED across
    # turns (the seal never touches it; continue_topic moves `goal`, not this), wiped only by reset/new_topic.
    # EMPTY by default → a greeting/question has NO contract and the region self-suppresses, so a trivial
    # first message can never become a binding spec. bound-is-relevance: only what must hold at the end.
    requirements: list[dict] = field(default_factory=list)  # [{"text": str, "done": bool}], insertion order
    # PLAN (TodoWrite) — the model's ORDERED execution steps with live status. Distinct from requirements
    # (acceptance criteria): this is the step sequence + progress. Replace-all via the update_plan tool
    # (folded by slice_sink). Carried by seal() (continuity), wiped by reset(). Bounded (MAX_PLAN_ITEMS).
    plan: list[dict] = field(default_factory=list)  # [{"step": str, "status": pending|in_progress|done}]
    action_log: dict[str, dict] = field(default_factory=dict)
    active_files: list[str] = field(default_factory=list)
    last_error: str = ""
    active_skills: list[dict] = field(default_factory=list)  # [{name, body}] loaded SKILLs
    edit_anchor: dict[str, str] = field(default_factory=dict)  # path -> last edit-target text (huge-file focus)
    findings: list[str] = field(default_factory=list)  # distilled conclusions carried across turns
    # I1 PROVENANCE — source tag per finding (text -> "observed" | "tool-note" | "claim"). Parallel
    # to `findings` (kept a plain list[str] so it stays JSON-serializable for taskstate/memory and
    # readable by discovery_query). Bounded with the findings ring; pruned to live keys only.
    finding_source: dict = field(default_factory=dict)
    # AGENT WORLD MODEL — a durable, agent-MAINTAINED key→value scratchpad for NON-code task state the
    # model must carry across many steps: an explored maze map, a text-adventure's rooms+inventory, a
    # system inventory (processes/ports/services), a running plan. Written via the world_set tool (folded
    # in by slice_sink, the same note→findings seam); READ from the rendered WORLD MODEL region, built into
    # each turn's SEED (no world_get needed) — within the SAME turn a just-set value lives only in the
    # model's own world_set call above until the next seed re-renders it. Unbounded (bound = the seal); SURVIVES
    # the seal (distilled task state); cleared only by reset (a new task). This generalizes the slice
    # beyond source files — where its multi-step memory wins on non-code tasks (maze/zork) actually lives.
    world: dict = field(default_factory=dict)
    edited_files: set = field(default_factory=set)  # the change set — protected from eviction
    since_edit: int = 0  # tool calls since the last successful edit — drives the EDIT convergence check
    turn_actions: int = 0  # tool calls THIS user turn — finding-INDEPENDENT (unlike since_edit, which resets
    # on every new finding); drives the explore-nudge so a read-heavy Q&A that records notes still converges
    # I3 — OPEN USER REPORT. The user's most-recent FAILURE REPORT ("it can't play", "cd: no such
    # file"), captured verbatim as a BLOCKER the model must verify against the real artifact before
    # claiming done. A snapshot agent loses the dialectic — the user pushing back on a "done" claim —
    # so the report is a durable tier. ONE string (inherently bounded); survives continue_topic (a new
    # directive does NOT mean the user retracted the report); cleared only by a real topic reset or a
    # NEWER report. NOT a transcript: a single most-recent line, capped.
    open_report: str = ""
    # CONTINUITY (short-range): a bounded ring of the last few user<->assistant exchanges so the slice
    # carries the immediate conversational thread (a snapshot agent otherwise loses "what we just said").
    # Older turns are NOT here — they live in the durable episodic cache, paged in ON DEMAND by reading
    # the history/ turn files (the decompression path). `turns` counts user turns this topic (for the "+N older"
    # pointer). Bounded => growth stays decoupled from conversation length (the moat).
    conversation: list[dict] = field(default_factory=list)  # [{user, assistant}], last MAX_CONVERSATION
    # AUTHORITATIVE INTENT (full-range): every user message this topic, VERBATIM + uncapped. Unlike the
    # short-range `conversation` ring (bounded to MAX_CONVERSATION, per-message gisted), a user's stated
    # instruction/constraint is IRREDUCIBLE ground truth — it cannot be re-derived from disk or from an
    # archived turn's gist. So it is the ONE part of the conversation that is never compacted: kept in
    # full, forever. (T1 buried-detail fix: a constraint stated once early must survive to a later turn.
    # Verbatim = no lossy extraction step — the exact step that dropped `40000` and substituted `65536`.
    # These bytes never change turn-to-turn, so as a STABLE region they EXTEND the cacheable prefix.)
    user_log: list[dict] = field(default_factory=list)  # [{turn, text}] verbatim, uncapped
    turns: int = 0
    # GHOST INDEX: bounded recovery POINTERS (references, not content) to things recently paged OUT of
    # the slice — an evicted read, a dropped skill. Turns "omission is unrecoverable" into a one-call
    # fetch. [{kind, ref}], bounded by MAX_GHOSTS; an item leaves the moment it's back in the slice.
    ghosts: list[dict] = field(default_factory=list)
    # CO-RESIDENCY: read-only files that are DEPENDENCIES (contracts/callers) of the change set,
    # recomputed each turn from the code graph (make_build_slice). Protected from eviction so the
    # files an edit must stay consistent with don't page out from under it. Bounded (DEP_CEILING);
    # a plain set of relpaths (serializable, like edited_files). Empty without a dep graph.
    protected_deps: set = field(default_factory=set)
    # CHANGE-SET CLOSURE state (symbol-aware): pre_defs snapshots each file's def-names BEFORE it is
    # edited, so prefetch can compute what an edit REMOVED (pre - current) and flag dependents whose
    # CURRENT tokens still reference a removed name — a precise dangling-call-site signal. INTERNAL
    # host state (def-name sets from the durable code graph), never rendered into the slice and never a
    # conversation transcript; scoped to the files touched this session (one small set per such file).
    pre_defs: dict = field(default_factory=dict)
    stale_deps: set = field(default_factory=set)
    # KERNEL-INTERNAL self-tuning state — NOT rendered into the slice the model sees, NOT serialized
    # (taskstate ignores it), transient. Pure mechanism, like Linux vmstat + mm/workingset refault
    # tracking. `io` = per-session page hit/miss/refault/evict counters (makes the moat MEASURED, not
    # asserted; the only legit input to any future budget auto-sizing). `hot` = files the kernel granted
    # ITSELF a brief reclaim-protection on a refault (path -> TTL steps), bounded by HOT_CEILING — the
    # automatic self-tuning loop (no model involvement), the validated automatic-beats-active-asker path.
    io: dict = field(default_factory=lambda: {"hit": 0, "miss": 0, "refault": 0, "evict": 0})
    hot: dict = field(default_factory=dict)
    # ADAPTIVE working-set budget (the "bounded = Markov current-state, not a fixed ceiling" reframe). The
    # resident exploratory-read budget is no longer the constant READ_BUDGET: it starts at that FLOOR and the
    # kernel GROWS it on refault thrash (SwapManager._grow) up to read_ceiling. Transient session state (like
    # io/hot) — NOT serialized, reset per task. Growth is task/refault-driven + window-bounded, never history-
    # proportional, so the moat holds. read_ceiling is the per-slice disaster ceiling (genuine breadth goes to
    # the swarm, not to inflating this one slice).
    read_budget: int = READ_BUDGET
    read_ceiling: int = READ_BUDGET_MAX
    # EXPLORER mode: a read-only delegated explorer's whole job IS thorough read-only investigation, so
    # the read-only convergence nudge ("you've explored N times, ANSWER now") must NOT fire on it — that
    # nudge is for the TOP-LEVEL agent over-exploring instead of answering the user. max_steps bounds the
    # explorer. Transient, set by run_subagent for read-only children. (Sibling of the EXPLORER_READ_BUDGET fix.)
    explore_mode: bool = False

    def reset(self, goal: str) -> None:
        self.goal = goal
        self.requirements = []   # a brand-new task starts with an EMPTY contract (model curates it in-band)
        self.plan = []           # a brand-new task starts with an empty plan (kept by seal() within a task)
        self.action_log = {}
        self.active_files = []
        self.last_error = ""
        self.active_skills = []
        self.edit_anchor = {}
        self.findings = []
        self.finding_source = {}
        self.world = {}                  # agent world model → wiped on a brand-new task (kept by seal())
        self.edited_files = set()
        self.since_edit = 0
        self.turn_actions = 0
        self.open_report = ""
        self.conversation = []
        self.user_log = []
        self.turns = 0
        self.ghosts = []
        self.protected_deps = set()
        self.pre_defs = {}
        self.stale_deps = set()
        self.io = {"hit": 0, "miss": 0, "refault": 0, "evict": 0}
        self.hot = {}
        self.read_budget = READ_BUDGET   # back to the lean floor each task; grows on refault within the task
        self.read_ceiling = READ_BUDGET_MAX
        self.explore_mode = False

    def seal(self) -> None:
        """SEAL the loop at a TURN boundary — "bounded = seal the within-loop info" (the moat is the seal
        BETWEEN loops, never a cut WITHIN one). The finished loop's COMPLETE info was archived to the
        durable cache on TurnEnd; here the NEXT loop starts FRESH so per-turn cost stays flat across a
        long multi-turn session instead of growing like a transcript.

        CARRY (distilled, durable continuity): findings + their sources, the in-progress edited change-set
        (+ its edit anchors), conversation ring, goal, the OPEN USER REPORT blocker, deliberate pins, and
        the demoted anti-loop tally — all kept by simply not touching them.
        SEAL (archived + recall-on-demand via recall_history): the RAW within-loop trajectory — recent
        steps, the intra-turn step cache, exploratory (non-edited) reads, and the prior loop's transient
        kernel state. None of it is lost: it's in the durable cache, one recall away if the next loop needs
        it. Distinct from reset() (a brand-new task wipes everything); seal() preserves the distilled carry."""
        self.ghosts = []                  # recovery pointers for the prior loop's evictions → moot now
        self.hot = {}                     # prior loop's kernel soft-pins → reset
        self.turn_actions = 0             # fresh action epoch for the new loop
        self.read_budget = READ_BUDGET    # back to the lean floor; re-grows on refault within the new loop
        # BOUND THE CARRY AT THE SEAL (not within the loop): the next loop starts from the most-recent
        # MAX_FINDINGS distilled facts; older ones are in the durable episodic cache (archived at TurnEnd,
        # recallable). This is the loop-boundary bound the moat prescribes — without it findings carried
        # unbounded across a long session (transcript-style growth). pre_defs is mostly transient pre-edit
        # state (re-derived by prefetch next loop) — but prefetch only snapshots NON-edited files, so the
        # pre-edit baseline for an EDITED file would be lost here and the change-set-closure (stale_deps)
        # could never detect a removed symbol's dangling callers next turn. So KEEP pre_defs for the carried
        # change-set (bounded by it), drop the exploratory rest. (world is intentionally durable: kept).
        if len(self.findings) > MAX_FINDINGS:
            self.findings = self.findings[-MAX_FINDINGS:]
            live = set(self.findings)
            self.finding_source = {k: v for k, v in self.finding_source.items() if k in live}
        self.pre_defs = {p: d for p, d in self.pre_defs.items() if p in self.edited_files}
        # CARRY the in-progress change-set resident; SEAL exploratory reads (re-readable / recallable).
        self.active_files = [p for p in self.active_files if p in self.edited_files]
        # Keep edited_files ⊆ active_files coherent: drop any phantom edit that is no longer resident, so a
        # restored/desynced state can't feed ghost files into prefetch's change-set closure next turn.
        self.edited_files = type(self.edited_files)(p for p in self.edited_files if p in self.active_files)
        self.edit_anchor = {p: a for p, a in self.edit_anchor.items() if p in self.edited_files}
        self.protected_deps = set()       # re-derived from the carried change-set by prefetch on next build
        self.stale_deps = set()


# ── SLICE FIELD LIFECYCLE — single source of truth ──────────────────────────────────────────────────
# reset() (the TASK boundary) uniformly wipes EVERY field to its default. seal() (the TURN boundary) does
# NOT: it CARRIES the distilled durable state, RESETS transient per-loop kernel state, and applies CUSTOM
# bounding (cap/filter) to a few. Historically each field's seal behavior was encoded by OMISSION (a field
# survives iff seal() happens not to touch it) — so adding a transient field and forgetting to reset it here
# silently CARRIED it, accumulating across turns and breaking the flat-peak moat; forgetting a field in
# reset() leaked it across unrelated tasks. This table makes the choice EXPLICIT, and test_slice_lifecycle
# enforces: (a) every Slice field is classified here, (b) reset() wipes all fields, (c) seal() resets every
# 'reset' field and preserves every 'carry' field. Add a Slice field → classify it here or the suite fails.
_SLICE_SEAL_POLICY: dict[str, str] = {
    # CARRY — distilled/durable across a turn (kept by seal by NOT touching it)
    "goal": "carry", "requirements": "carry", "plan": "carry", "action_log": "carry",
    "last_error": "carry", "active_skills": "carry", "world": "carry", "since_edit": "carry",
    "open_report": "carry", "conversation": "carry", "user_log": "carry", "turns": "carry",
    "io": "carry", "read_ceiling": "carry", "explore_mode": "carry",
    # RESET — transient per-loop kernel state → back to default at the seal
    "turn_actions": "reset", "ghosts": "reset", "protected_deps": "reset", "stale_deps": "reset",
    "hot": "reset", "read_budget": "reset",
    # CUSTOM — bounded/filtered by seal (findings cap, change-set filter); behavior has its own tests
    "findings": "custom", "finding_source": "custom", "active_files": "custom",
    "edited_files": "custom", "edit_anchor": "custom", "pre_defs": "custom",
}


def touch_file(s: Slice, path: str, edited: bool = False) -> None:
    """Shim → SwapManager.load (swap.py owns the file load→evict→ghost lifecycle). Signature unchanged."""
    _DEFAULT_SWAP.load(s, path, edited=edited)


def add_skill(s: Slice, name: str, body: str) -> None:
    """Shim → SwapManager.load_skill (swap.py owns skill load/evict + ghosts). Signature unchanged."""
    _DEFAULT_SWAP.load_skill(s, name, body)


def _active(state):
    """Resolve the current Slice from a Slice or a Session (host-side topic manager)."""
    return state.active() if hasattr(state, "active") else state


def record_user(s: Slice, message: str) -> None:
    """Append the user's message to the short-range CONVERSATION ring and count the turn. The host
    calls this once per user message; slice_sink fills the assistant side as the turn produces text.
    Bounded ring — older exchanges live in the durable cache, paged in on demand (not kept here)."""
    s.turns += 1
    s.turn_actions = 0   # new user turn → reset the per-turn exploration budget (drives the explore-nudge)
    # VERBATIM, uncapped — the authoritative user-intent record (see user_log field note). Full message,
    # NOT one_line'd: the whole point is that a precise constraint stated here is never truncated/dropped.
    s.user_log.append({"turn": s.turns, "text": message})
    # RECENT CONVERSATION ring — VERBATIM (whitespace-normalized, NOT truncated): the last few turns are the
    # active loop's antecedents, so a deictic follow-up ("go with your recommendation", "save this") resolves
    # against the real text, not a lossy gist. Count-bounded by MAX_CONVERSATION; older turns page out to history/.
    s.conversation.append({"user": normalize_ws(message), "assistant": ""})
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
    goal = (s.goal or "").strip()
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
    return "\n".join(lines)


def slice_sink(state):
    """Event sink that folds tool results back into the tiers (keeps the loop decoupled). `state`
    is a Slice or a Session — events fold into the CURRENT active slice (so a topic switch redirects
    subsequent folding)."""
    def sink(event: Event) -> None:
        s = _active(state)
        # I1 PROVENANCE (root-cause revision) — fold the model's reasoning forward as an UNVERIFIED
        # CLAIM, never as an established fact. Dropping assistant text ENTIRELY (the first I1 cut)
        # starved the anti-re-derivation tier and ~2x'd steps on normal tasks; the defect was narration
        # becoming FACT, not carry-forward itself. record_note drops pure narration (_NARRATION_RE),
        # downgrades "done"-style claims, and bounds+dedups the ring — so this restores reasoning-reuse
        # WITHOUT reviving the "already done" ratchet: a claim renders as "verify against OPEN FILES",
        # and OPEN FILES stays the only ground truth. (The episode cache keeps assistant text losslessly.)
        if isinstance(event, AssistantText):
            record_note(s, event.content, source="claim")
            if s.conversation and (event.content or "").strip():
                # fill the assistant side of the in-progress exchange — the LAST AssistantText of the
                # turn wins, so this ends up holding the final reply shown to the user (continuity).
                full = event.content
                # VERBATIM (whitespace-normalized, NOT truncated): the last MAX_CONVERSATION turns keep their
                # FULL reply so a next-turn back-reference ("go with your recommendation") resolves against the
                # real conclusion — which usually sits at the TAIL, exactly what a head-gist used to sever. The
                # bound is the turn COUNT, not bytes; older turns page out to history/ (recall pages them back).
                s.conversation[-1]["assistant"] = normalize_ws(full)
            return
        if isinstance(event, ToolResult):
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
            if event.name == "world_set" and not event.failing:
                _k = str(event.args.get("key", "")).strip()
                if _k:
                    s.world[_k] = str(event.args.get("value", ""))
            elif event.name == "world_clear" and not event.failing:
                _k = str(event.args.get("key", "")).strip()
                if _k:
                    s.world.pop(_k, None)
                else:
                    s.world.clear()
            # STANDING REQUIREMENTS — fold require/requirement_done/drop_requirement into the carried
            # contract (same seam; the handler only confirms). Text-matched + idempotent so a re-emit is a
            # no-op (byte-stable prefix); append-only + status-flip-in-place so a change never reorders
            # existing lines; no match on done/drop = no-op (never silently corrupt the contract).
            elif event.name in ("require", "requirement_done", "drop_requirement") and not event.failing:
                _t = " ".join(str(event.args.get("text", "")).split())[:MAX_REQ_CHARS]
                if _t:
                    _hit = next((r for r in s.requirements if isinstance(r, dict) and r.get("text", "").lower() == _t.lower()), None)
                    if event.name == "require":
                        if _hit is None:
                            s.requirements.append({"text": _t, "done": False})
                            del s.requirements[:-MAX_REQUIREMENTS]   # bound: keep the most recent N
                    elif event.name == "requirement_done":
                        if _hit:
                            _hit["done"] = True
                    elif _hit:                                        # drop_requirement
                        s.requirements.remove(_hit)
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
            # FAN-IN: a subagent/explorer reports its result as the tool OUTPUT (not the note arg). Fold that
            # distilled summary into the carried FINDINGS tier (observed) so it survives the turn-boundary seal —
            # the parent reconciles summaries, never the children's transcripts (the swarm's no-bloat guarantee).
            if event.name in ("spawn_subagent", "spawn_explore", "spawn_agent") and not event.failing and event.output:
                new_finding = record_note(s, event.output, source="observed") or new_finding
            did_edit = False
            if event.name == "skill" and not event.failing:
                # a loaded skill's body must enter the ACTIVE SKILL tier or it vanishes
                # next turn (no transcript). The skill tool returns the body as its output.
                add_skill(s, event.args.get("name", ""), event.output)
            # list_files/grep/glob "path" is a DIRECTORY scope, not a working-set FILE — don't pin it (else
            # build_artifacts read_text(dir) → IsADirectoryError → a bogus OPEN FILES entry every turn).
            if event.args.get("path") and event.name not in ("list_files", "grep", "glob"):
                did_edit = event.name in ("edit_file", "append_to_file", "str_replace") and not event.failing
                # WS1 — gate membership on SUCCESS. A read/edit that FAILED (e.g. _resolve raised
                # "path escapes workspace") must NOT be pinned into the working set, or OPEN FILES
                # re-renders the unreachable/missing path every rebuild and poisons the slice
                # (the read-blindness loop). Successful reads and edits still join the set.
                if not event.failing:
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
            # convergence tracking: a real edit OR a genuinely-new finding resets the spin counter —
            # actively LEARNING (recording new facts) is progress, not spinning (review #5). Only a call
            # that neither edits nor learns advances the convergence/no-progress counter.
            s.since_edit = 0 if (did_edit or new_finding) else s.since_edit + 1
    return sink
