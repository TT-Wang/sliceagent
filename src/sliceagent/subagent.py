"""Subagents — bounded delegation on the slice architecture.

A large, decomposable task can be split: the parent spawns a CHILD agent for a
sub-task; the child runs its own loop with a FRESH slice, does the work in the SAME
workspace, and returns ONLY a compact summary. The parent's slice never sees the
child's transcript — just the summary — so parent context stays bounded no matter how
much work the child did. That's the slice thesis applied recursively.

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
import os
import posixpath
import re

from .access import AllAccess, ReadAllAccess
from .agents import BUILTIN_AGENTS, READ_ONLY_TOOLS, SUBAGENT_EXCLUDED_TOOLS, AgentSpec  # named-agent registry
from .events import AssistantText, ToolResult, ToolStarted
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

# EXPLORER PROFILE — reasoning intent for read-only explorer children. They NAVIGATE/READ (find files,
# trace usages, summarize), which the model does well at low reasoning effort; running them at the parent's
# (often "full") setting just burns wall-clock. Default "fast"; override for an A/B (set to "full" to match
# the parent). A per-PROFILE subagent reasoning setting, applied via a per-child llm VIEW so the
# shared parent llm is never mutated and parallel siblings never race on it.
EXPLORER_REASONING = (os.environ.get("AGENT_EXPLORER_REASONING") or "fast").lower()


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


def _primary_arg(args) -> str:
    """The one informative arg for a compact activity line (path/command/pattern/…), whitespace-collapsed."""
    if not isinstance(args, dict):
        return ""
    for k in ("path", "command", "pattern", "name", "ref", "goal", "task"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())[:50]
    return ""


def _targets_reserved_ns(args) -> bool:
    """True if a read tool's path targets the PARENT-only virtual namespaces (subagents/, history/ or
    roster/) — a child shares the base host, so without this it could page the parent's trajectory, a
    sibling's sealed artifact, or another specialist's career (a third channel the design forbids:
    children couple ONLY through the two seals — or an EXPLICIT grant / their OWN roster files, both
    checked by the caller: a grant is a pointer to a seal and self-memory is not a channel)."""
    p = _norm_vpath(args.get("path") if isinstance(args, dict) else "")
    # `.sliceagent/` is the host's PRIVATE dir — paged-out blobs (.sliceagent/blobs/, the L1→L2 store for big
    # tool results), config, agents. A child shares the base host, so without blocking it a child could
    # read_file/list a parent-created blob = an undocumented parent→child output channel (external review S8).
    return (p in ("subagents", "history", "roster", ".sliceagent")
            or p.startswith(("subagents/", "history/", "roster/", ".sliceagent/")))


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
                 name: str = "", grants: tuple = (), identity_block: str = "", budget_sink=None) -> str:
    """Run a child agent of a given KIND (`spec`) on `task` with a fresh slice; return a bounded summary.
    The child's events stay on its OWN dispatcher — they never touch the parent's slice (the bounded-
    context guarantee); only the summary crosses back.

    `spec` is the named AgentSpec (tools allowlist + reasoning + system-prompt layer). Back-compat: when
    `spec` is None it is derived from `read_only` (the built-in explorer vs general). A read-only spec runs
    as an EXPLORER — its tool host exposes only the read-only allowlist, so it cannot mutate the workspace."""
    from .events import make_dispatcher
    from .guardrails import ToolCallGuardrailConfig
    from .hooks import CompositeHooks, GuardrailHook, PermissionHook
    from .loop import run_turn
    from .pfc import Slice, slice_sink
    from .seed import make_build_slice

    if spec is None:
        spec = BUILTIN_AGENTS["explorer" if read_only else "general"]
    read_only = spec.read_only   # the kind decides; everything below keys off the SPEC

    # A grant the child can't SEE is a grant it never uses (the visible-manifest lesson, applied down a
    # level): granted input reports are advertised in the child's own task text with the exact call.
    child_task = task
    if grants:
        child_task = task + "\n\nINPUT REPORTS — sealed sibling work you may (and should) read:\n" + \
            "\n".join(f'- read_file("{g}")' for g in sorted(grants))

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
    sinks = [slice_sink(child_state), cap, trace]
    if notify is not None:
        sinks.append(_nested_sink(notify, depth))
    child_dispatch = make_dispatcher(*sinks)

    _child_hooks = [PermissionHook(policy)] if policy is not None else []
    # An EXPLORER does read-only investigation: a repeated read/list/grep is at most inefficient, never the
    # write-loop disaster the anti-loop HARD-BLOCK guards against — and max_steps already bounds it. So relax
    # the READ axes (no-progress / result-repeat) for explorers while KEEPING exact-failure (a repeated
    # FAILING call is still a real loop). Stops review children going "stuck" on legitimate reads.
    _guard_cfg = (ToolCallGuardrailConfig(no_progress_block_after=10**6, result_repeat_block_after=10**6)
                  if read_only else None)
    _child_hooks.append(GuardrailHook(_guard_cfg))
    hooks = CompositeHooks(*_child_hooks)
    result = run_turn(build_slice=build, llm=child_llm, tools=tools, dispatch=child_dispatch,
                      hooks=hooks, max_steps=max_steps)
    # S5: a child's tokens were invisible to the parent's budget — a fan-out of N children could blow the
    # per-turn cap unseen. Charge the child's usage back to the parent budget so the parent turn stops once
    # the TOTAL (parent + all children) crosses the ceiling. usage is also sealed into the artifact below.
    _child_usage = dict(getattr(result, "usage", None) or {})
    if budget_sink is not None:
        try:
            budget_sink(int(_child_usage.get("prompt_tokens", 0)) + int(_child_usage.get("completion_tokens", 0)))
        except Exception:  # noqa: BLE001 — budget accounting must never crash a returned child result
            pass

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
    # the parent's context stays flat no matter how much the child did (the moat, one level up). No detail is
    # lost: the digest is a coarse-graining, the handle is its refinement map.
    # `name` is the INSTANCE identity (who); `brief` is the VERBATIM ask (what they were told) — provenance:
    # whoever later reads this report can see the question alongside the answer, so a narrowly-briefed child
    # is never silently cited for broad claims.
    # W5': lift the optional trailing "LESSON: ..." reflection out of the report into a typed field
    # (the line stays in the report verbatim — the seal is honest; this is indexing, not editing).
    _lm = re.findall(r"^LESSON:\s*(.+)$", cap.text or "", re.MULTILINE)
    artifact = {
        "kind": spec.name, "name": name, "task": task, "brief": {"task": task, "grants": sorted(grants)},
        "lesson": one_line(_lm[-1], 200) if _lm else "",
        "status": status, "steps": result.steps, "usage": _child_usage,   # S5: child cost, sealed + billed up
        "report": cap.text or "", "findings": list(child_state.findings),
        "change_set": sorted(child_state.edited_files), "files": list(child_state.active_files),
        "trace": trace.text(),   # W6': the action path, bounded — full-detail grounding for rehydration
        "coverage": f"{len(child_state.active_files)} file(s) touched; stop={result.stop_reason}",
        # refs = the sealed inputs this work was built ON (its granted reports) — the seal's refinement map
        # back to its sources, so a synthesis is drillable to what it merged (invariant: every seal ships
        # its refinement handle, in BOTH directions).
        "refs": sorted(grants),
    }
    handle = memory.append_subagent_artifact(session_id, artifact) if (memory is not None and session_id) else ""
    if handle:   # W6': additive FTS5 mirror → search_history finds delegated work by CONTENT (never
        memory.index_subagent_artifact(session_id, handle, artifact)   # written to the turn timeline)
    if name and memory is not None:   # a STANDING specialist also accumulates the job in its durable career
        memory.roster_append_job(name, artifact)   # (no-op for temps: roster_append_job needs a profile)

    head = f"[{label} {status} · {result.steps} steps · files: {files}]"
    if handle:   # archived → bounded digest + recall handle (the refinable seal)
        body = one_line(cap.text, 300) if cap.text else "(no summary produced)"
        # ALWAYS hand back the CANONICAL immutable id (sub-N.md), never the subagents/<name>.md alias: the
        # alias retargets to the LATEST job for that name, so a later same-name job would silently make an
        # earlier tool result / grant open a DIFFERENT report (external review S11). The <name>.md alias
        # stays resolvable in SubagentFS as a convenience; the sealed handle the parent stores is immutable.
        summary = f'{head} {body} → full report: read_file("subagents/{handle}.md")'
    else:        # no durable archive (eval/headless) → inline, back-compat with the pre-artifact behavior
        summary = head + (" " + one_line(cap.text, 400) if cap.text else "")
    if not success:
        if child_state.last_error:
            summary += " | unresolved: " + one_line(child_state.last_error, 160)
        return "Error: subagent did not finish cleanly: " + summary  # surfaces in parent's error tier
    return summary


class SubagentHost:
    """ToolHost wrapper that adds the `spawn_agent` delegation tool. Delegates every real tool (and
    read_text/accesses) to the wrapped host, so parent and child share one workspace."""

    def __init__(self, inner, *, llm, retriever, memory, policy,
                 max_depth: int = 1, max_steps: int = 20, depth: int = 0, notify=None,
                 read_only: bool = False, spec: AgentSpec | None = None, agents=None, session_id: str = "",
                 grants: frozenset = frozenset(), instance_name: str = "", budget_sink=None):
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
        # S5: a callable(int) that charges child tokens back to the parent's per-turn budget (set by the CLI
        # after the budget hook exists). Propagated to nested children so ALL delegated cost bills upward.
        self.budget_sink = budget_sink

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
        if self.depth < self.max_depth:  # parent (or a general child) — offer delegation while depth remains
            s.append(self._agent_schema())
        return s

    def _agent_schema(self) -> dict:
        """The ONE delegation tool — `spawn_agent`. It subsumes the former spawn_explore / spawn_subagent
        (each was just this with agent='explorer' / 'general'); measured parity on parallel fan-out, so the
        breadth nudge lives here in the description, not in a dedicated verb. The two orthogonal dials —
        KIND (agent=) and IDENTITY (name=) — are spelled out so the model has the right mental model."""
        kinds = "; ".join(f"{n} ({sp.description})" for n, sp in self.agents.items())
        return {"type": "function", "function": {
            "name": "spawn_agent",
            "description": (
                "Delegate a self-contained sub-task to a child agent that runs in its OWN bounded context and "
                "returns ONLY a short summary (its reads never enter your context). Two dials:\n"
                "• agent = which KIND — " + kinds + ". For BREADTH (review/understand a repo, find a bug, "
                "audit several modules) emit MULTIPLE spawn_agent(agent=\"explorer\", …) calls in ONE response "
                "— explorers are read-only and run in PARALLEL; one per area/module/question, then synthesize "
                "their summaries. Stay single-agent for one tightly-coupled change you're editing yourself.\n"
                "• name = OPTIONAL identity. OMIT it → a one-shot TEMP (used once, then only its sealed report "
                "remains). PASS one → HIRE a STANDING specialist that persists across sessions, accumulates "
                "lessons, and can be WOKEN by re-using the same name later (see STANDING SPECIALISTS). Hire "
                "when this is an area you'll revisit; use a temp for a one-off."),
            "parameters": {"type": "object", "properties": {
                "agent": {"type": "string", "description": "the KIND to run (a name from the list above)"},
                "task": {"type": "string", "description": "the self-contained sub-task for that agent"},
                "name": _NAME_PARAM, "grants": _GRANTS_PARAM,
            }, "required": ["agent", "task"]}}}

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
        ids = {r.get("id") for r in arts}
        names = {(r.get("artifact") or {}).get("name") for r in arts} - {"", None}
        out = set()
        for g in raw:
            p = _norm_vpath(g)
            if p and "/" not in p:                       # accept a bare leaf ("sub-1.md") for convenience
                p = "subagents/" + p
            leaf = p[len("subagents/"):] if p.startswith("subagents/") else ""
            stem = leaf[:-3] if leaf.endswith(".md") else ""
            ok = ("/" not in leaf) and leaf and (
                (_GRANT_SUB.match(leaf) and stem in ids)                          # exact per-job handle
                or (_valid_instance_name(stem) and stem in names))                # name alias (latest job)
            if not ok:
                return (f"Error: cannot grant {g!r} — grants must be EXACT existing sealed-report handles "
                        f'(e.g. "subagents/sub-1.md" or "subagents/<name>.md"; never a directory or '
                        f'index.md). See read_file("subagents/index.md") for what exists.', frozenset())
            out.add(p)
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

    def read_text(self, path: str) -> str:
        return self.inner.read_text(path)

    def run(self, name: str, args: dict) -> str:
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
        if self.spec is not None and (name == "search_history"
                                      or (name in ("read_file", "list_files", "grep") and _targets_reserved_ns(args))):
            # W2 carve-out: an EXACT granted handle passes through (read_file/grep on that one file only —
            # never list_files, never a directory, never index.md; those can't be granted). W4' carve-out:
            # a standing specialist may read ITS OWN roster/<name>/ files (career, lessons, profile) —
            # self-memory, not a channel. Everything else in the reserved namespaces stays default-deny.
            p = _norm_vpath(args.get("path") if isinstance(args, dict) else "")
            if name in ("read_file", "grep") and p in self.grants:
                return self.inner.run(name, args)
            if self.instance_name and (p == f"roster/{self.instance_name}"
                                       or p.startswith(f"roster/{self.instance_name}/")):
                return self.inner.run(name, args)
            hint = (" Your granted input reports: " + ", ".join(sorted(self.grants)) + "."
                    if self.grants else "")
            own = (f" Your own past work is under roster/{self.instance_name}/."
                   if self.instance_name else "")
            return ("Error: history/, subagents/ and roster/ (and search_history over them) are the "
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

        child_tools = SubagentHost(
            self.inner, llm=self.llm, retriever=self.retriever, memory=self.memory,
            policy=self.policy, max_depth=self.max_depth, max_steps=self.max_steps,
            depth=self.depth + 1, notify=self.notify, spec=spec, agents=self.agents,
            session_id=self.session_id,   # nested children archive under the SAME parent session
            grants=child_grants,          # W2: one hop only — this child's grants never propagate further
            instance_name=child_name,     # W4': unlocks the child's OWN roster/<name>/ files (self-memory)
            budget_sink=self.budget_sink, # S5: nested child cost bills up to the same per-turn budget
        )
        try:
            out = run_subagent(
                task, tools=child_tools, llm=self.llm, retriever=self.retriever,
                memory=self.memory, policy=self.policy, max_steps=self.max_steps,
                depth=self.depth + 1, notify=self.notify, spec=spec, session_id=self.session_id,
                name=child_name, grants=tuple(child_grants), identity_block=identity_block,
                budget_sink=self.budget_sink,
            )
            # announce the lifecycle event (visibility: an unadvertised wake channel stays dead) — but NOT
            # onto a failed child's "Error: ..." return, where it would garble the parent's error tier (the
            # hire is real regardless; it just isn't news worth mixing into an error line).
            if hired and not out.startswith("Error:"):
                out += f' | hired standing specialist {child_name!r} — re-use name="{child_name}" to wake it later'
            return out
        except Exception as e:  # a child failure must not crash the parent
            return f"Error: subagent crashed: {e}"
