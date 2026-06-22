"""Subagents — bounded delegation (Kimi/Hermes pattern, on the slice architecture).

A large, decomposable task can be split: the parent spawns a CHILD agent for a
sub-task; the child runs its own loop with a FRESH slice, does the work in the SAME
workspace, and returns ONLY a compact summary. The parent's slice never sees the
child's transcript — just the summary — so parent context stays bounded no matter how
much work the child did. That's the slice thesis applied recursively.

Exposed as a tool (`spawn_subagent`) via a ToolHost wrapper, so the loop is unchanged:
from the parent loop's view it's one tool call that returns a summary string. The child
is depth-capped (a child can't spawn grandchildren by default) and runs under the same
permission policy. Tool execution and reads delegate to the wrapped (real) ToolHost,
so parent and child share one workspace and one sandbox.
"""
from __future__ import annotations

import copy
import json
import os

from .access import AllAccess, ReadAllAccess
from .agents import BUILTIN_AGENTS, READ_ONLY_TOOLS, AgentSpec  # named-agent registry (file-defined kinds)
from .events import AssistantText, ToolStarted
from .slice import one_line

_SUBAGENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "spawn_subagent",
        "description": (
            "Delegate a self-contained WRITABLE sub-task to a child agent (full tools). It works in "
            "the SAME workspace and returns only a SHORT summary (not its full transcript), so your own "
            "context stays small. Use for a large, decomposable piece of work you want carried out "
            "end-to-end; give a complete, standalone description (the child sees none of your context). "
            "For pure investigation prefer spawn_explore; for one tightly-coupled change you are "
            "actively editing yourself, stay single-agent."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
    },
}

_EXPLORE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "spawn_explore",
        "description": (
            "Delegate a READ-ONLY investigation to a child agent that reads/lists/searches (grep/glob)/"
            "recalls in the SAME workspace and returns only a SHORT summary — its file reads never enter "
            "your context. USE PROACTIVELY FOR BREADTH: whenever answering would require reading more "
            "than a couple of files, or the task spans several areas (e.g. 'review the repo', 'where/how "
            "is X handled', 'find the bug'), emit SEVERAL spawn_explore calls in ONE response (one per "
            "area or question) — they run in PARALLEL — instead of reading everything yourself, then "
            "synthesize their summaries. The child cannot edit, run commands, or spawn its own children."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
    },
}

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
# the parent). Borrowed from Kimi Code's per-PROFILE subagent config. Applied via a per-child llm VIEW so the
# shared parent llm is never mutated and parallel siblings never race on it.
EXPLORER_REASONING = (os.environ.get("AGENT_EXPLORER_REASONING") or "fast").lower()


def _profile_llm(llm, reasoning):
    """The llm VIEW for a child running at a given reasoning intent ("fast"/"full"): a SHALLOW COPY with
    `reasoning` overridden (shares the thread-safe openai client + all config; never mutates the parent or
    races a sibling). No-op — returns the parent llm — when `reasoning` is falsy (inherit) or already matches."""
    if not reasoning or getattr(llm, "reasoning", None) == reasoning:
        return llm
    view = copy.copy(llm)
    view.reasoning = reasoning
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


def _nested_sink(notify, depth: int):
    pad = "    " * depth
    def sink(event):
        if isinstance(event, ToolStarted):
            notify(f"{pad}  ↳ {event.name}({json.dumps(event.args, ensure_ascii=False)[:60]})")
        elif isinstance(event, AssistantText) and event.content:
            notify(f"{pad}  ↳ {event.content[:100]}")
    return sink


def run_subagent(task: str, *, tools, llm, retriever, memory, policy,
                 max_steps: int = 20, depth: int = 1, notify=None,
                 read_only: bool = False, spec: AgentSpec | None = None) -> str:
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
    from .slice import Slice, make_build_slice, slice_sink

    if spec is None:
        spec = BUILTIN_AGENTS["explorer" if read_only else "general"]
    read_only = spec.read_only   # the kind decides; everything below keys off the SPEC

    child_state = Slice()
    child_state.reset(task)
    if read_only:
        # explorer: keep the whole exploration resident (no eviction churn → no "stuck") AND don't let the
        # read-only convergence nudge cut the review short before the key files are read — see
        # EXPLORER_READ_BUDGET + Slice.explore_mode. max_steps bounds the explorer.
        child_state.read_budget = child_state.read_ceiling = EXPLORER_READ_BUDGET
        child_state.explore_mode = True
    child_llm = _profile_llm(llm, spec.reasoning)   # per-kind reasoning via a per-child llm view (no mutation)
    build = make_build_slice(child_state, tools, retriever, memory, task, system_extra=spec.system_prompt)

    cap = _CaptureLast()
    sinks = [slice_sink(child_state), cap]
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

    files = ", ".join(child_state.active_files) or "(none)"
    # A READ-ONLY explorer's deliverable is its summary; a stray failed read (lingering last_error) must NOT
    # flag the whole review as "did not finish cleanly". Only a WRITABLE child's last_error matters (it may
    # have left the task broken). end_turn means it produced a final summary either way.
    success = result.stop_reason == "end_turn" and (read_only or not child_state.last_error)
    status = "ok" if success else result.stop_reason
    label = {"explorer": "explore", "general": "subagent"}.get(spec.name, spec.name)  # named-kind label
    summary = f"[{label} {status} · {result.steps} steps · files: {files}]"
    if cap.text:
        summary += " " + one_line(cap.text, 400)
    if not success:
        if child_state.last_error:
            summary += " | unresolved: " + one_line(child_state.last_error, 160)
        return "Error: subagent did not finish cleanly: " + summary  # surfaces in parent's error tier
    return summary


class SubagentHost:
    """ToolHost wrapper that adds `spawn_subagent`. Delegates every real tool (and
    read_text/accesses) to the wrapped host, so parent and child share one workspace."""

    def __init__(self, inner, *, llm, retriever, memory, policy,
                 max_depth: int = 1, max_steps: int = 20, depth: int = 0, notify=None,
                 read_only: bool = False, spec: AgentSpec | None = None, agents=None):
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
        if self.spec is not None and self.spec.tools is not None:
            # a restricted CHILD (explorer, or any custom kind with an allowlist) sees ONLY its tools —
            # no edit/shell/spawn beyond the allowlist (an explorer's list has none → cannot recurse).
            allow = set(self.spec.tools)
            return [x for x in s if x.get("function", {}).get("name") in allow]
        if self.depth < self.max_depth:  # parent (or a general child) — offer delegation while depth remains
            s.append(_SUBAGENT_SCHEMA)
            s.append(_EXPLORE_SCHEMA)
            s.append(self._agent_schema())
        return s

    def _agent_schema(self) -> dict:
        """The generic `spawn_agent` tool — delegate to a NAMED agent kind from the registry (Kimi-style:
        one Agent tool + a pluggable roster). The description enumerates the available kinds by name."""
        roster = "; ".join(f"{n}: {sp.description}" for n, sp in self.agents.items())
        return {"type": "function", "function": {
            "name": "spawn_agent",
            "description": ("Delegate a self-contained sub-task to a NAMED agent kind that runs in its OWN "
                            "bounded context and returns ONLY a summary. Available kinds — " + roster),
            "parameters": {"type": "object", "properties": {
                "agent": {"type": "string", "description": "the agent kind to run (a name from the list)"},
                "task": {"type": "string", "description": "the self-contained sub-task for that agent"},
            }, "required": ["agent", "task"]}}}

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
        if name not in ("spawn_subagent", "spawn_explore", "spawn_agent"):
            return self.inner.run(name, args)
        if self.depth >= self.max_depth:
            return "Error: subagent depth limit reached"
        if name == "spawn_agent":
            spec = self.agents.get(args.get("agent", ""))
            if spec is None:
                return ("Error: unknown agent %r. Available: %s"
                        % (args.get("agent", ""), ", ".join(self.agents)))
        else:   # back-compat built-in tools → their specs
            spec = BUILTIN_AGENTS["explorer" if name == "spawn_explore" else "general"]
        child_tools = SubagentHost(
            self.inner, llm=self.llm, retriever=self.retriever, memory=self.memory,
            policy=self.policy, max_depth=self.max_depth, max_steps=self.max_steps,
            depth=self.depth + 1, notify=self.notify, spec=spec, agents=self.agents,
        )
        try:
            return run_subagent(
                args["task"], tools=child_tools, llm=self.llm, retriever=self.retriever,
                memory=self.memory, policy=self.policy, max_steps=self.max_steps,
                depth=self.depth + 1, notify=self.notify, spec=spec,
            )
        except Exception as e:  # a child failure must not crash the parent
            return f"Error: subagent crashed: {e}"
