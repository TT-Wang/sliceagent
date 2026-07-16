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
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

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


MAX_WORKSPACE_TRANSITIONS = 4


def _mint_task_id() -> str:
    return "t-" + uuid.uuid4().hex[:8]


@dataclass
class LogicalTurn:
    """One exact user request spanning one or more workspace-bound runtime segments.

    A workspace handoff is a transport boundary, not another user turn.  Keeping this identity on the
    application-owned :class:`Session` prevents the target workspace from routing a synthetic ``go`` request,
    incrementing conversational turn counts, or silently attaching the continuation to a different task.
    A bounded edge history permits real multi-project workflows while rejecting an A→B edge that the model
    already traversed in this request. ``admission`` is deliberately runtime-only: the durable segment journal
    owns its serialised projection.
    """

    id: str
    task_id: str
    request: str
    source_artifact_id: str
    source_event_id: str = ""
    source_workspace: str = ""
    segment_index: int = 0
    workspace_epoch: int = 0
    workspace_switches: int = 0
    workspace_history: tuple[str, ...] = ()
    workspace_edges: tuple[tuple[str, str], ...] = ()
    admission: Any = field(default=None, repr=False, compare=False)

    @property
    def segment_id(self) -> str:
        return f"{self.id}:segment:{self.segment_index}"


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
        # Workspace truth changes independently from the application/session identity.  Epoch 0 is the launch
        # workspace; every successfully published handoff increments it exactly once.
        self.workspace_epoch: int = 0
        self.logical_turn: LogicalTurn | None = None

    def active(self) -> Slice:
        return self.tasks[self.active_id]

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
        tid = _mint_task_id()
        s = Slice()
        s.reset(goal)
        return tid, s

    def prepare_switch_topic(self, task_id: str) -> tuple[str, Slice]:
        """Load a topic for admission without changing ``active_id``."""
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
        """Continue the active topic with a new directive while keeping its durable task context."""
        s = self.active()
        # The completed turn was already sealed at its actual terminal boundary by the runtime. Do not seal
        # again here: a second seal would reset the new TurnRuntime epoch and contract the working set twice.
        if admission is not None and contract is not None and admission != contract:
            raise ValueError("pass one TurnAdmission, not competing admission/contract values")
        if install_intent:
            s.intent.begin_turn(message, admission=admission or contract)
        return apply_turn_continuation(s, message, resume=resume, admission=admission or contract)

    def start_logical_turn(
        self, *, logical_id: str, task_id: str, request: str, source_artifact_id: str,
        source_event_id: str = "", admission=None, source_workspace: str = "",
    ) -> LogicalTurn:
        """Bind an admitted user request to its stable task before model/tool execution begins."""
        if self.logical_turn is not None:
            raise RuntimeError(f"logical turn {self.logical_turn.id!r} is already active")
        if self.active_id != str(task_id):
            raise RuntimeError("logical turn task does not match the active task")
        logical = LogicalTurn(
            id=str(logical_id), task_id=str(task_id), request=str(request),
            source_artifact_id=str(source_artifact_id), source_event_id=str(source_event_id),
            source_workspace=os.path.realpath(source_workspace)
            if source_workspace else "", workspace_epoch=self.workspace_epoch,
            workspace_history=((os.path.realpath(source_workspace),) if source_workspace else ()),
            admission=admission,
        )
        self.logical_turn = logical
        return logical

    def begin_workspace_segment(
        self, *, source_artifact_id: str, admission=None, workspace_path: str = "",
    ) -> LogicalTurn:
        """Resume the current exact request after a workspace publication, without admitting a new user turn."""
        logical = self.logical_turn
        if logical is None:
            raise RuntimeError("no logical turn is available for workspace continuation")
        if logical.workspace_switches >= MAX_WORKSPACE_TRANSITIONS:
            raise RuntimeError(
                f"a logical turn may cross at most {MAX_WORKSPACE_TRANSITIONS} workspace boundaries",
            )
        if self.active_id != logical.task_id or logical.task_id not in self.tasks:
            raise RuntimeError("workspace continuation lost its task identity")
        state = self.tasks[logical.task_id]
        # Re-install the same exact source and contract for this runtime segment only.  In particular, do not call
        # record_user/continue_topic: they would increment turns, append a duplicate conversation row, and apply
        # lexical continuation semantics to a message the user never sent twice.
        next_admission = admission if admission is not None else logical.admission
        target = os.path.realpath(workspace_path) if workspace_path else ""
        source = logical.workspace_history[-1] if logical.workspace_history else logical.source_workspace
        edge = (source, target) if source and target else None
        if edge is not None and edge in logical.workspace_edges:
            raise RuntimeError(f"workspace continuation would repeat transition {source} -> {target}")
        state.intent.begin_turn(
            logical.request, source_artifact=str(source_artifact_id), admission=next_admission,
        )
        # Publish the segment counters only after contract validation succeeds, so a malformed continuation
        # cannot consume the one-switch allowance while leaving the old segment active.
        logical.segment_index += 1
        logical.workspace_switches += 1
        logical.workspace_epoch = self.workspace_epoch
        if target:
            logical.workspace_history = (*logical.workspace_history, target)
        if edge is not None:
            logical.workspace_edges = (*logical.workspace_edges, edge)
        logical.source_artifact_id = str(source_artifact_id)
        logical.admission = next_admission
        state.task.activate_objective()
        state.runtime.reset()
        self.turn_task_id = logical.task_id
        return logical

    def finish_logical_turn(self) -> LogicalTurn | None:
        logical, self.logical_turn = self.logical_turn, None
        return logical


class SessionBinding:
    """Stable application-owned session identity over a replaceable workspace view.

    Workspace resource factories capture a binding, not a one-off concrete Session. During an atomic handoff
    the binding is redirected to the merged target view, so topic tools, subagents, episodic sinks, and the CLI
    all observe the same session without rebuilding the model/UI connection.
    """

    def __init__(self, target: Session):
        object.__setattr__(self, "_target", target)

    @property
    def target(self) -> Session:
        return object.__getattribute__(self, "_target")

    def bind(self, target: Session) -> None:
        if target.session_id != self.target.session_id:
            raise ValueError("cannot rebind an application session to a different session_id")
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __setattr__(self, name, value) -> None:
        if name == "_target":
            object.__setattr__(self, name, value)
        else:
            setattr(self.target, name, value)


def _workspace_rebased_slice(source: Slice) -> Slice:
    """Carry language/task intent into a new workspace while dropping old physical-world claims."""
    state = deepcopy(source)
    # File residency, revisions, shell facts, action tallies, and observed findings belong to the old root.
    state.work.reset(read_budget=source.work.read_budget, read_ceiling=source.work.read_ceiling)
    state.findings = []
    state.finding_source = {}
    state.last_error = ""
    state.reconciliation_required = ""
    state.reconciliation_targets = []
    state.task.action_log = {}
    state.task.world = {}
    state.task.progress_signals = []
    state.task.goal_source = ""
    state.plan = []
    # Keep exact user-authored clauses, but detach store-local artifact/invocation handles. The next target
    # turn receives its own local source artifact before any new checkpoint can publish.
    state.intent.current_source = None
    state.intent.entries = [replace(
        entry, source_artifact=None, source_range=None, evidence_refs=(),
    ) for entry in state.intent.entries]
    state.intent.seal()
    state.conversation = [
        {**dict(exchange), "artifact_id": ""} for exchange in state.conversation
    ]
    state.continuity.discourse_focus = [
        dict(anchor) for anchor in state.continuity.discourse_focus
        if isinstance(anchor, dict) and anchor.get("kind") == "subject_focus"
    ]
    if isinstance(state.continuity.pending_proposal, dict):
        state.continuity.pending_proposal = {
            key: value for key, value in state.continuity.pending_proposal.items()
            if key not in {"artifact_id", "source_artifact"}
        }
    state.continuity.previous_evidence_snapshot = None
    state.runtime.reset()
    return state


def rebase_session_for_workspace(current: Session, restored: Session) -> Session:
    """Merge target checkpoints with the live app conversation under one stable session identity."""
    if current.session_id != restored.session_id:
        raise ValueError("workspace session views do not share the application session_id")
    merged = Session(current.memory, current.session_id)
    # Target-local authoritative checkpoints win task-ID collisions. Other open topics remain available as
    # intent/conversation shells and acquire target-local provenance on their next admitted turn.
    merged.tasks = dict(restored.tasks)
    for task_id, state in current.tasks.items():
        if task_id in merged.tasks:
            # On A→B→A the target can have an older checkpoint for any still-open topic. Keep its valid A-local
            # physical projection, but overlay app-owned conversation/intent accumulated in B—even when the
            # user switched away from that topic before navigating back.
            carried = _workspace_rebased_slice(state)
            target = deepcopy(merged.tasks[task_id])
            target.active_work = carried.active_work
            target.intent = carried.intent
            target.continuity = carried.continuity
            target.task.goal = carried.task.goal
            target.task.objective_status = carried.task.objective_status
            target.task.goal_source = ""
            target.task.plan = carried.task.plan
            target.task.deliverable_requirement = carried.task.deliverable_requirement
            target.open_report = carried.open_report
            target.runtime.reset()
            merged.tasks[task_id] = target
        elif task_id not in merged.tasks:
            merged.tasks[task_id] = _workspace_rebased_slice(state)
    merged.active_id = (
        current.active_id if current.active_id in merged.tasks else restored.active_id
    )
    merged.turn_task_id = None
    merged.turn_generation = max(current.turn_generation, restored.turn_generation)
    merged.workspace_epoch = current.workspace_epoch + 1
    merged.logical_turn = deepcopy(current.logical_turn)
    return merged


def apply_turn_continuation(
    s: Slice, message: str, *, resume: bool = False, admission=None,
) -> Slice:
    """Apply the non-intent continuation reducer in live admission and crash recovery."""
    if s.active_work.items:
        # The source-linked graph and exact current request now own semantic continuation.  Do not run the
        # legacy failure-report/retraction/resume grammar or mutate host-authored blocker prose; those fields
        # remain checkpoint adapters only and the Active Work compiler does not render them.
        s.since_edit = 0
        return s
    resuming_unresolved_work = bool(resume or _CONTINUATION_ONLY.fullmatch(str(message or "")))
    if not resuming_unresolved_work:
        # A substantive new directive starts a fresh blocker focus. A bare resume cue does not: the
        # unresolved error and its failing action row are the state the user asked us to continue from.
        s.last_error = ""
        # demote (don't clear): keep counts, drop the failing flag — see WS2 above
        for _sig, action in s.action_log.items():
            action["failing"] = False
            action.pop("failure_identity", None)
            action.pop("failure_last", None)
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
    from .execution import ToolStatus
    from .registry import ToolEntry, ToolText

    def _new(args: dict) -> str:
        if session.turn_task_id is not None:
            return ToolText(
                "topic changes are turn-boundary operations. Finish this turn, then start the new topic "
                "from the next user request or host command.",
                status=ToolStatus.STEERED,
            )
        tid = session.new_topic(args["goal"])
        return f"Started new topic [{tid}]: {one_line(args['goal'], 80)}. Previous topic parked (resumable)."

    def _switch(args: dict) -> str:
        if session.turn_task_id is not None:
            return ToolText(
                "topic changes are turn-boundary operations. Finish this turn, then switch from the next "
                "user request or host command.",
                status=ToolStatus.STEERED,
            )
        try:
            s = session.switch_topic(args["task_id"])
        except KeyError:
            return ToolText(
                f"no open topic {args.get('task_id')!r}. Pick a task_id from OTHER OPEN THREADS.",
                status=ToolStatus.STEERED,
            )
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
