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
from .slice import Slice, capture_user_report, one_line
from .taskstate import slice_to_task_state, task_state_to_slice


def _mint_task_id() -> str:
    return "t-" + uuid.uuid4().hex[:8]


class Session:
    def __init__(self, memory, session_id: str | None = None):
        self.memory = memory
        self.session_id = session_id or ("s-" + uuid.uuid4().hex[:12])
        self.tasks: dict[str, Slice] = {}     # task_id -> live bounded slice (in-session)
        self.active_id: str | None = None

    def active(self) -> Slice:
        return self.tasks[self.active_id]

    def _park(self, status: str = "parked") -> None:
        """Durably checkpoint the active topic (for cross-session resume); a no-op under NullMemory.
        The live Slice stays in self.tasks regardless, so within-session switching is lossless."""
        if self.active_id is None:
            return
        if getattr(self.memory, "is_durable", False):
            self.memory.checkpoint_task(slice_to_task_state(
                self.tasks[self.active_id], self.active_id,
                session_id=self.session_id, status=status))

    def new_topic(self, goal: str) -> str:
        """Park the current topic and start a fresh one. Returns the new task_id."""
        self._park()
        tid = _mint_task_id()
        s = Slice()
        s.reset(goal)
        self.tasks[tid] = s
        self.active_id = tid
        return tid

    def switch_topic(self, task_id: str) -> Slice:
        """Park the current topic and activate another — from the live set if present, else resumed
        from the durable vault (distilled). Raises KeyError if neither has it."""
        self._park()
        if task_id not in self.tasks:
            ts = self.memory.load_task(task_id)
            if ts is None:
                raise KeyError(f"unknown topic {task_id}")
            self.tasks[task_id] = task_state_to_slice(ts)   # cross-session: distilled + since_edit=0
        self.active_id = task_id
        return self.tasks[task_id]

    def open_threads(self, *, include_active: bool = False) -> list[TaskRef]:
        """The OTHER OPEN THREADS source: live topics (parked by default; the active one optional)."""
        out: list[TaskRef] = []
        for tid, s in self.tasks.items():
            if not include_active and tid == self.active_id:
                continue
            out.append(TaskRef(task_id=tid, title=one_line(s.goal, 60),
                               status="active" if tid == self.active_id else "parked"))
        return out

    def continue_topic(self, message: str, *, resume: bool = False) -> Slice:
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
        # SEAL the prior loop at this turn boundary: the finished loop was archived on TurnEnd; start the
        # next loop fresh — CARRY the distilled context (findings + edited change-set + conversation), SEAL
        # the raw trajectory (recent/step-cache/exploratory reads → recall-on-demand). This is what keeps
        # per-turn cost flat across a long session (the moat) while within-loop info stays complete.
        s.seal()
        if not resume:
            s.goal = message   # a RESUME cue ("go back to the auth task") must NOT replace the topic's defining goal
        s.last_error = ""
        # demote (don't clear): keep counts, drop the failing flag — see WS2 above
        for sig, a in s.action_log.items():
            a["failing"] = False
        s.since_edit = 0
        s.reviewed = []          # new directive → the history ratchet resets (re-allow fresh lookbacks)
        capture_user_report(s, message)   # a failure report rides forward as an OPEN USER REPORT blocker
        return s


def route_topic(llm, message: str, session: "Session") -> tuple[str, str]:
    """Classify a new user message against the session: ('continue'|'new'|'resume', task_id). ONE
    cheap LLM call, biased to 'continue', safe defaults on any parse failure. No topic is mutated
    here — the host applies the result — so there are no junk topics. (Provider-agnostic: uses the
    LLMClient contract.)"""
    if session.active_id is None:
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
        resp = llm.complete([{"role": "system", "content": sys_msg},
                             {"role": "user", "content": usr}], [])
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


def route_topic_lexical(message: str, session: "Session") -> tuple[str, str]:
    """Routing WITHOUT an LLM round-trip (the moat-aligned default): the dominant 'continue' case pays
    nothing, and the host never *guesses* 'new' — a genuinely new task is the agent's call via its own
    new_topic tool (make_topic_tools), which is recoverable and parks the old topic. The host only acts
    on an UNAMBIGUOUS resume signal: an explicit parked task_id in the message, or a resume cue
    ('go back to…') plus a title-keyword match. Everything else → continue. Same signature/return contract
    as route_topic, so it's a drop-in. See memory route-topic-hidden-llm-call for why this exists."""
    if session.active_id is None:
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
    Measured (evals/route_accuracy.py): lexical == llm on continue+resume (15/15); they differ only on
    'new', which lexical defers to the agent's new_topic tool. Single call site for both UI paths."""
    if os.environ.get("AGENT_ROUTER", "lexical").strip().lower() == "llm":
        return route_topic(llm, message, session)
    return route_topic_lexical(message, session)


def make_topic_tools(session: "Session"):
    """Model-facing tools so the agent can route topics itself. Default behaviour is CONTINUE (no
    call); a switch/new is an explicit, recoverable action. Returns ToolEntry list for the registry."""
    from .registry import ToolEntry

    def _new(args: dict) -> str:
        tid = session.new_topic(args["goal"])
        return f"Started new topic [{tid}]: {one_line(args['goal'], 80)}. Previous topic parked (resumable)."

    def _switch(args: dict) -> str:
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
