"""Session — host-side topic manager (MEMORY-SPEC step 3, mechanical core).

Holds one bounded Slice per topic; switching PARKS the current and activates another. Within a
session, parked topic-slices stay in memory (lossless — switching back returns the same Slice);
a durable checkpoint is ALSO written on park when the memory is durable, so a topic can be resumed
in a future session (distilled from the vault, files re-read live). A topic not in the live set is
resumed via memory.load_task.

This is the host layer — it never touches the loop/slice core. The remaining half of step 3 (the
model-facing new_topic/switch_topic TOOLS and the OTHER OPEN THREADS render tier) layers on top of
this. NullMemory works fine: the in-memory topic dict gives full within-session switching; only
cross-session resume needs a durable vault.
"""
from __future__ import annotations

import json
import os
import re
import uuid

from .interfaces import TaskRef
from .pfc import Slice
from .regions import capture_user_report, report_retracted
from .text_utils import is_chitchat, one_line
from .taskstate import slice_to_task_state, task_state_to_slice


_CONTINUATION_ONLY = re.compile(
    r"^\s*(?:please\s+)?(?:continue|keep\s+going|go\s+on|proceed|try\s+again|retry|"
    r"another\s+approach|try\s+(?:a\s+)?different\s+approach|pick\s+up(?:\s+where\s+you\s+left\s+off)?)"
    r"[.!?]*\s*$",
    re.IGNORECASE,
)


def _mint_task_id() -> str:
    return "t-" + uuid.uuid4().hex[:8]


class Session:
    def __init__(self, memory, session_id: str | None = None):
        self.memory = memory
        self.session_id = session_id or ("s-" + uuid.uuid4().hex[:12])
        self.tasks: dict[str, Slice] = {}     # task_id -> live bounded slice (in-session)
        self.active_id: str | None = None
        self.turn_task_id: str | None = None  # immutable task binding while one model/tool turn runs
        # Monotonic only within this live session. Evidence snapshots use it to prove conversational adjacency
        # across task switches; it is intentionally not durable because snapshots are not durable either.
        self.turn_generation: int = 0

    def active(self) -> Slice:
        return self.tasks[self.active_id]

    def _require_reconciled_boundary(self) -> None:
        if self.active_id is None or self.active_id not in self.tasks:
            return
        detail = getattr(self.tasks[self.active_id], "reconciliation_required", "")
        if detail:
            raise RuntimeError(
                "cannot change tasks while an earlier operation is indeterminate; continue this task, "
                "re-observe the live state, and call reconcile_execution first"
            )

    def _park(self, status: str = "parked") -> None:
        """Durably checkpoint the active topic (for cross-session resume); a no-op under NullMemory.
        The live Slice stays in self.tasks regardless, so within-session switching is lossless."""
        # M6: guard the dict access — if active_id is set but its Slice isn't in self.tasks (shouldn't
        # happen), do NOT raise KeyError here: switch_topic's caller `_switch` has `except KeyError` and would
        # mislabel a park failure as "no such topic" for the DIFFERENT topic being switched to. Nothing to save.
        if self.active_id is None or self.active_id not in self.tasks:
            return
        if getattr(self.memory, "is_durable", False):
            self.memory.checkpoint_task(slice_to_task_state(
                self.tasks[self.active_id], self.active_id,
                session_id=self.session_id, status=status))

    def prepare_new_topic(self, goal: str) -> tuple[str, Slice]:
        """Prepare a fresh topic without publishing it as active.

        Routing/orientation can inspect the returned Slice, the host can begin its durable turn journal,
        and only then call :meth:`activate_prepared_topic`. Preparation itself does not park or publish
        anything; the admission journal must exist before durable/session state changes.
        """
        self._require_reconciled_boundary()
        tid = _mint_task_id()
        s = Slice()
        s.reset(goal)
        return tid, s

    def prepare_switch_topic(self, task_id: str) -> tuple[str, Slice]:
        """Load a topic for admission without changing ``active_id``."""
        self._require_reconciled_boundary()
        if task_id in self.tasks:
            return task_id, self.tasks[task_id]
        ts = self.memory.load_task(task_id)
        if ts is None:
            raise KeyError(f"unknown topic {task_id}")
        return task_id, task_state_to_slice(ts)

    def activate_prepared_topic(self, task_id: str, state: Slice) -> Slice:
        """Publish a prevalidated topic after the turn journal exists."""
        self._park()
        self.tasks[str(task_id)] = state
        self.active_id = str(task_id)
        return state

    def new_topic(self, goal: str) -> str:
        """Park the current topic and start a fresh one. Returns the new task_id."""
        tid, s = self.prepare_new_topic(goal)
        self.activate_prepared_topic(tid, s)
        return tid

    def switch_topic(self, task_id: str) -> Slice:
        """Park the current topic and activate another — from the live set if present, else resumed
        from the durable vault (distilled). Raises KeyError if neither has it."""
        task_id, state = self.prepare_switch_topic(task_id)
        return self.activate_prepared_topic(task_id, state)

    def open_threads(self, *, include_active: bool = False) -> list[TaskRef]:
        """The OTHER OPEN THREADS source: live topics (parked by default; the active one optional)."""
        out: list[TaskRef] = []
        for tid, s in self.tasks.items():
            if not include_active and tid == self.active_id:
                continue
            out.append(TaskRef(task_id=tid, title=one_line(s.goal, 60),
                               status="active" if tid == self.active_id else "parked"))
        return out

    def continue_topic(
        self, message: str, *, resume: bool = False, admission=None, contract=None,
        install_intent: bool = True,
    ) -> Slice:
        """Continue the active topic with a NEW directive: set the goal, start a fresh action epoch
        but KEEP the durable context — findings and the working set — so the follow-up builds on
        what's already done.

        I3 WS2 — DEMOTE the anti-loop epoch, don't wipe it. Clearing action_log every directive let a
        completed sub-step re-run with no REPEATED warning (turn 14 ran the same command twice). Instead
        we mark prior entries non-failing (a NEW directive shouldn't carry a stale failure) but KEEP
        their counts, so a genuinely-repeated command still trips REPEATED-with-no-progress. since_edit
        is still cleared (a fresh convergence epoch).

        I3 OPEN USER REPORT — if this follow-up looks like a FAILURE REPORT, capture it as a blocker;
        it is NOT cleared by continue_topic (a new directive does not mean the user retracted the
        report — only verifying the fix, or a real topic change, clears it)."""
        s = self.active()
        # The completed turn was already sealed at its actual terminal boundary by the runtime. Do not seal
        # again here: a second seal would reset the new TurnRuntime epoch and apply working-set contraction
        # twice before the follow-up starts.
        # ``goal`` is the stable task objective. A follow-up—resume cue or ordinary continuation—changes
        # only current_request; otherwise "yes, do that" destroys the objective in the next checkpoint.
        # The topic label/objective and the CURRENT REQUEST are distinct. A resume cue should not rename the
        # parked task, but it is still the user's authoritative request for this turn and must be rendered.
        # Direct Session callers may still request the historical eager installation.  The production host
        # passes ``install_intent=False``: it journals the immutable TurnAdmission first, then record_user()
        # installs it exactly once together with the verbatim conversation entry.
        if admission is not None and contract is not None and admission != contract:
            raise ValueError("pass one TurnAdmission, not competing admission/contract values")
        if install_intent:
            s.intent.begin_turn(message, admission=admission or contract)
        return apply_turn_continuation(s, message, resume=resume, admission=admission or contract)


def apply_turn_continuation(
    s: Slice, message: str, *, resume: bool = False, admission=None,
) -> Slice:
    """Apply the non-intent continuation reducer in live admission and crash recovery."""
    resuming_unresolved_work = bool(resume or _CONTINUATION_ONLY.fullmatch(str(message or "")))
    if not resuming_unresolved_work:
        # A substantive new directive starts a fresh blocker focus. A bare resume cue does not: the
        # unresolved error and its failing action row are the state the user asked us to continue from.
        s.last_error = ""
        # demote (don't clear): keep counts, drop the failing flag — see WS2 above
        for _sig, action in s.action_log.items():
            action["failing"] = False
    s.since_edit = 0
    # A sealed-history question or self-audit may contain words such as "failed" or "went wrong" without
    # asserting that anything is broken. The typed admission distinguishes that inquiry from a live user
    # report; lexical capture remains the conservative fallback for legacy/direct callers.
    evidence_query = getattr(admission, "evidence_query", None)
    sealed_inquiry = bool(
        admission is not None
        and getattr(admission, "grounding", "none") == "sealed_past"
        and (
            "audit" in (getattr(admission, "requested_modes", ()) or ())
            or evidence_query is not None
        )
    )
    failure_report = False if sealed_inquiry else capture_user_report(s, message)
    # A stale OPEN USER REPORT is a blocker on the PRIOR concern. An explicit move-on cue ("anyways do X",
    # "forget that", "new topic") abandons it — clear it so it can't hijack the fresh directive (the design
    # clears the blocker on a real topic change; the LLM router, biased to 'continue', may miss the switch).
    # Only when THIS turn is not itself a fresh report and is not a bare resume of the reported work.
    if s.open_report and not failure_report and not resuming_unresolved_work and report_retracted(message):
        s.open_report = ""
    # A true failure report rides forward as an OPEN USER REPORT blocker.
    if resuming_unresolved_work or failure_report:
        # Topic identity and execution authority are separate: a completed objective normally becomes
        # background, but an explicit resume or user push-back makes the exact original outstanding again.
        s.task.activate_objective()
    return s


def route_topic(llm, message: str, session: "Session") -> tuple[str, str]:
    """Classify a new user message against the session: ('continue'|'new'|'resume', task_id). ONE
    cheap LLM call, biased to 'continue', safe defaults on any parse failure. No topic is mutated
    here — the host applies the result — so there are no junk topics. (Provider-agnostic: uses the
    LLMClient contract.)"""
    if session.active_id is None:
        return ("new", "")
    if is_chitchat(session.active().goal) and not is_chitchat(message):
        return ("new", "")
    threads = session.open_threads(include_active=False)
    parked = "\n".join(f"- {t.task_id}: {t.title}" for t in threads) or "(none)"
    sys_msg = (
        "You route a user's new message in a coding session into ONE action. Reply with ONLY a JSON "
        'object: {"action":"continue|new|resume","task_id":"<a parked id, or empty>"}. '
        "continue = it continues or refines the ACTIVE task. new = a different, unrelated task. "
        "resume = it asks to return to one of the PARKED topics (give that topic's id). "
        "Bias to 'continue' when unsure.")
    usr = (f"ACTIVE TASK: {session.active().goal or '(none)'}\nPARKED TOPICS:\n{parked}\n"
           f"NEW MESSAGE: {message}")
    try:
        from .model_runner import complete_model_call
        resp = complete_model_call(
            llm, [{"role": "system", "content": sys_msg},
                  {"role": "user", "content": usr}], [],
        )
        m = re.search(r"\{.*\}", resp.content or "", re.S)
        d = json.loads(m.group(0)) if m else {}
        action = d.get("action", "continue")
        if action == "resume":
            tid = d.get("task_id", "")
            return ("resume", tid) if any(t.task_id == tid for t in threads) else ("continue", "")
        if action in ("new", "continue"):
            return (action, "")
    except Exception:
        pass
    return ("continue", "")


_RESUME_CUES = ("go back", "going back", "back to", "return to", "returning to", "resume",
                "switch back", "switch to", "revisit", "pick up where", "pick back up")
_EXPLICIT_NEW_TASK = re.compile(
    r"^\s*(?:new\s+(?:task|topic)|start\s+(?:a\s+)?new\s+(?:task|topic))\s*[:—-]\s*\S",
    re.IGNORECASE,
)


def route_topic_lexical(message: str, session: "Session") -> tuple[str, str]:
    """Routing WITHOUT an LLM round-trip (the moat-aligned default): the dominant 'continue' case pays
    nothing and ambiguous messages continue. The host acts only on explicit boundary language: a
    ``New task: ...`` prefix, a parked task id, or a resume cue plus title-keyword match. Everything else
    continues. Same signature/return contract as route_topic, so it is a drop-in."""
    if session.active_id is None:
        return ("new", "")
    if is_chitchat(session.active().goal) and not is_chitchat(message):
        return ("new", "")
    if _EXPLICIT_NEW_TASK.search(message):
        return ("new", "")
    threads = session.open_threads(include_active=False)
    if threads:
        msg_l = message.lower()
        for t in threads:                                  # explicit parked id mentioned → resume it
            if t.task_id.lower() in msg_l:
                return ("resume", t.task_id)
        if any(cue in msg_l for cue in _RESUME_CUES):      # resume cue → best title-keyword overlap
            msg_words = set(re.findall(r"[a-z0-9]+", msg_l))
            best, score = "", 0
            for t in threads:
                kw = {w for w in re.findall(r"[a-z0-9]+", (t.title or "").lower()) if len(w) > 3}
                overlap = len(kw & msg_words)
                if overlap > score:
                    best, score = t.task_id, overlap
            if best and score >= 1:
                return ("resume", best)
    return ("continue", "")


def route(llm, message: str, session: "Session") -> tuple[str, str]:
    """The configured topic router. DEFAULT = lexical (zero LLM round-trips — moat-aligned: a follow-up
    no longer pays a per-message provider call before the turn even starts). Set AGENT_ROUTER=llm to
    restore the classifier (tighter automatic 'new'-task detection, at one round-trip per follow-up).
    Explicit ``New task:`` boundary language is deterministic; ambiguous unrelatedness still follows the
    fail-safe continuation law. Single call site for both UI paths."""
    if os.environ.get("AGENT_ROUTER", "lexical").strip().lower() == "llm":
        return route_topic(llm, message, session)
    return route_topic_lexical(message, session)


def make_topic_tools(session: "Session"):
    """Model-facing tools so the agent can route topics itself. Default behaviour is CONTINUE (no
    call); a switch/new is an explicit, recoverable action. Returns ToolEntry list for the registry."""
    from .registry import ToolEntry

    def _new(args: dict) -> str:
        if session.turn_task_id is not None:
            return ("Error: topic changes are turn-boundary operations. Finish this turn, then start the "
                    "new topic from the next user request or host command.")
        tid = session.new_topic(args["goal"])
        return f"Started new topic [{tid}]: {one_line(args['goal'], 80)}. Previous topic parked (resumable)."

    def _switch(args: dict) -> str:
        if session.turn_task_id is not None:
            return ("Error: topic changes are turn-boundary operations. Finish this turn, then switch from "
                    "the next user request or host command.")
        try:
            s = session.switch_topic(args["task_id"])
        except KeyError:
            return (f"Error: no open topic {args.get('task_id')!r}. Pick a task_id from "
                    "OTHER OPEN THREADS.")
        return f"Switched to topic [{args['task_id']}]: {one_line(s.goal, 80)} (its state is restored)."

    new_schema = {"type": "function", "function": {
        "name": "new_topic",
        "description": ("Start a NEW, unrelated task as its own topic — parks the current one (you can "
                        "return to it later via switch_topic). Use ONLY when the request is a different "
                        "task, not a continuation of the current one."),
        "parameters": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}}}
    switch_schema = {"type": "function", "function": {
        "name": "switch_topic",
        "description": ("Resume a PARKED topic listed in OTHER OPEN THREADS — restores its state. Use "
                        "only to return to earlier work, not to start something new."),
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}},
                       "required": ["task_id"]}}}
    return [ToolEntry(name="new_topic", schema=new_schema, handler=_new, source="builtin"),
            ToolEntry(name="switch_topic", schema=switch_schema, handler=_switch, source="builtin")]
