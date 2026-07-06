"""BREADTH end-to-end: each periphery capability driven through the REAL run_turn loop (scripted LLM →
tool dispatch → slice_sink fold → NEXT-turn render). Proves the wiring works end-to-end, not just per-unit.
Deterministic, no network, no model. Run: PYTHONPATH=src python tests/test_breadth_e2e.py

Capabilities covered here: plan, world model, standing requirements, skills (load + $ARGUMENTS),
plugins (a 3rd-party-registered tool runs through the loop). Subagents/swarm, MCP, and recall are covered
end-to-end by test_readonly_subagent / test_mcp_output_cap / test_recall_search + test_history.
"""
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import ToolResult, make_dispatcher           # noqa: E402
from sliceagent.hooks import CompositeHooks                         # noqa: E402
from sliceagent.loop import run_turn                                # noqa: E402
from sliceagent.memory import NullMemory                            # noqa: E402
from sliceagent.registry import ToolEntry                           # noqa: E402
from sliceagent.retriever import NullRetriever                      # noqa: E402
from sliceagent.skills import make_skill_manager, make_skill_tool   # noqa: E402
from sliceagent.pfc import Slice, slice_sink  # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402
from sliceagent.tools import LocalToolHost                          # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _TC:
    def __init__(self, name, args):
        self.name, self.args = name, args


def _resp(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls or [],
                           finish_reason="tool_calls" if calls else "stop",
                           usage={"prompt_tokens": 1, "completion_tokens": 1})


class ScriptedLLM:
    def __init__(self, *responses):
        self.q = list(responses)

    def complete(self, messages, schemas):
        return self.q.pop(0) if self.q else _resp("done")


def _run(state, tools, *responses):
    """Drive ONE real turn with the scripted tool calls; return (next-slice user render, events)."""
    events = []
    dispatch = make_dispatcher(slice_sink(state), events.append)
    build = make_build_slice(state, tools, NullRetriever(), NullMemory(), state.goal)
    run_turn(build_slice=build, llm=ScriptedLLM(*responses), tools=tools, dispatch=dispatch,
             hooks=CompositeHooks(), max_steps=10)
    user = build()[1]["content"]      # the NEXT turn's rendered slice — proves the effect carried
    return user, events


def _host():
    return LocalToolHost(tempfile.mkdtemp(prefix="be2e-"))


@check
def plan_flows_end_to_end():
    s = Slice(); s.reset("a multi-step task")
    user, _ = _run(s, _host(), _resp(calls=[_TC("update_plan", {"steps": [
        {"step": "write the code", "status": "in_progress"},
        {"step": "add tests", "status": "pending"}]})]))
    assert "# PLAN" in user and "write the code" in user and "[~]" in user, user[:300]


@check
def world_model_flows_end_to_end():
    s = Slice(); s.reset("explore the maze")
    user, _ = _run(s, _host(), _resp(calls=[_TC("world_set", {"key": "exit", "value": "north of room A"})]))
    assert "# WORLD MODEL" in user and "exit" in user and "north of room A" in user, user[:300]


@check
def requirements_flow_end_to_end():
    s = Slice(); s.reset("keep the API stable")
    user, _ = _run(s, _host(), _resp(calls=[_TC("require", {"text": "public signatures must not change"})]))
    assert "# STANDING REQUIREMENTS" in user and "public signatures must not change" in user, user[:300]


@check
def skill_loads_with_arguments_end_to_end():
    d = tempfile.mkdtemp(prefix="be2e-")
    with open(os.path.join(d, "deploy.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: deploy\ndescription: ship it\n---\nStep 1: build $1\nStep 2: push to prod\n")
    tools = LocalToolHost(d)
    tools.registry.register(make_skill_tool(make_skill_manager([d])))   # plug the skill tool in (as cli does)
    s = Slice(); s.reset("deploy the service")
    user, _ = _run(s, tools, _resp(calls=[_TC("skill", {"name": "deploy", "arguments": "myapp"})]))
    assert "ACTIVE SKILL" in user and "Step 1: build myapp" in user, user[:400]   # $1 expanded, body resident


@check
def plugin_tool_runs_through_the_loop():
    d = tempfile.mkdtemp(prefix="be2e-")
    tools = LocalToolHost(d)
    marker = os.path.join(d, "plugin_ran.txt")

    def _handler(args):
        with open(marker, "w", encoding="utf-8") as f:
            f.write(args.get("x", ""))
        return f"plugin ran with {args.get('x')}"

    tools.registry.register(ToolEntry(
        name="my_plugin", handler=_handler, source="plugin",
        schema={"type": "function", "function": {"name": "my_plugin", "parameters": {
            "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}))
    # the plugin tool is advertised to the model…
    assert "my_plugin" in [sc["function"]["name"] for sc in tools.schemas()]
    s = Slice(); s.reset("use the plugin")
    _, events = _run(s, tools, _resp(calls=[_TC("my_plugin", {"x": "hello"})]))
    assert os.path.exists(marker), "plugin tool did not execute through the loop"
    trs = [e for e in events if isinstance(e, ToolResult) and e.name == "my_plugin"]
    assert trs and "plugin ran with hello" in trs[0].output, "plugin result did not flow back"


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
