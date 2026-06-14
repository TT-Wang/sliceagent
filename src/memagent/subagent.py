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

import json

from .access import AllAccess
from .events import AssistantText, ToolStarted
from .slice import one_line

_SUBAGENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "spawn_subagent",
        "description": (
            "Delegate a self-contained SUB-TASK to a child agent. It works in the SAME "
            "workspace and returns only a SHORT summary (not its full transcript), so your "
            "own context stays small. Use for large decomposable work; give a clear, "
            "complete, standalone sub-task description (the child sees none of your context)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
    },
}


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
                 max_steps: int = 20, depth: int = 1, notify=None) -> str:
    """Run a child agent on `task` with a fresh slice; return a bounded summary string.
    The child's events stay on its OWN dispatcher — they never touch the parent's slice
    (that's the bounded-context guarantee); only the returned summary crosses back."""
    from .events import make_dispatcher
    from .hooks import CompositeHooks, PermissionHook
    from .loop import run_turn
    from .slice import Slice, make_build_slice, slice_sink

    child_state = Slice()
    child_state.reset(task)
    build = make_build_slice(child_state, tools, retriever, memory, task)

    cap = _CaptureLast()
    sinks = [slice_sink(child_state), cap]
    if notify is not None:
        sinks.append(_nested_sink(notify, depth))
    child_dispatch = make_dispatcher(*sinks)

    hooks = CompositeHooks(PermissionHook(policy)) if policy is not None else None
    result = run_turn(build_slice=build, llm=llm, tools=tools, dispatch=child_dispatch,
                      hooks=hooks, max_steps=max_steps)

    files = ", ".join(child_state.active_files) or "(none)"
    success = result.stop_reason == "end_turn" and not child_state.last_error
    status = "ok" if success else result.stop_reason
    summary = f"[subagent {status} · {result.steps} steps · files: {files}]"
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
                 max_depth: int = 1, max_steps: int = 20, depth: int = 0, notify=None):
        self.inner = inner
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.policy = policy
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.depth = depth
        self.notify = notify

    def schemas(self) -> list[dict]:
        s = list(self.inner.schemas())
        if self.depth < self.max_depth:  # only offer delegation while there's depth left
            s.append(_SUBAGENT_SCHEMA)
        return s

    def accesses(self, name: str, args: dict) -> list:
        if name == "spawn_subagent":
            return [AllAccess()]  # arbitrary nested work → globally exclusive
        return self.inner.accesses(name, args)

    def read_text(self, path: str) -> str:
        return self.inner.read_text(path)

    def run(self, name: str, args: dict) -> str:
        if name != "spawn_subagent":
            return self.inner.run(name, args)
        if self.depth >= self.max_depth:
            return "Error: subagent depth limit reached"
        child_tools = SubagentHost(
            self.inner, llm=self.llm, retriever=self.retriever, memory=self.memory,
            policy=self.policy, max_depth=self.max_depth, max_steps=self.max_steps,
            depth=self.depth + 1, notify=self.notify,
        )
        try:
            return run_subagent(
                args["task"], tools=child_tools, llm=self.llm, retriever=self.retriever,
                memory=self.memory, policy=self.policy, max_steps=self.max_steps,
                depth=self.depth + 1, notify=self.notify,
            )
        except Exception as e:  # a child failure must not crash the parent
            return f"Error: subagent crashed: {e}"
