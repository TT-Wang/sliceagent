"""run_turn integration: max_steps budget guidance + denial wording. (These W1 cases were carried over
from the former test_overflow_rebuild.py; the rebuild-path / tighten-ladder tests there were retired
together with the rebuild loop mode.) No model, no pytest.
Run: PYTHONPATH=src python tests/test_turn_budget.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import ToolResult, TurnInterrupted              # noqa: E402
from memagent.guidance import BUDGET_EXHAUSTED, DENIAL_NO_PROMPT, DENIAL_USER  # noqa: E402
from memagent.hooks import PermissionHook, ToolDecision, Hooks       # noqa: E402
from memagent.interfaces import Snippet                              # noqa: E402
from memagent.loop import run_tool_batch, run_turn                   # noqa: E402
from memagent.pfc import Slice, slice_sink  # noqa: E402
from memagent.seed import make_build_slice  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Resp:
    def __init__(self, *, content="done", tool_calls=None, finish_reason="stop", usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {"prompt_tokens": 1, "completion_tokens": 1}


class _Retriever:
    def retrieve(self, query, k=6):
        if k <= 0:
            return []
        return [Snippet(path="related.py", text="def helper():\n    return 1", score=1.0)]


class _TC:
    name = "noop"
    args: dict = {}


class _BudgetLLM:
    """Always returns a tool-call response so run_turn keeps stepping until max_steps."""
    def complete(self, messages, schemas):
        return _Resp(content="", tool_calls=[_TC()], finish_reason="tool_use")


class _NoopTools:
    def schemas(self):
        return []
    def accesses(self, name, args):
        return []
    def run(self, name, args):
        return "ok"


def _collect(events):
    def dispatch(e):
        events.append(e)
    return dispatch


@check
def run_turn_max_steps_dispatches_budget_guidance():
    s = Slice(); s.reset("loop forever")
    tools = _NoopTools()
    build = make_build_slice(s, tools, _Retriever(), None, "loop forever")
    events = []
    result = run_turn(build_slice=build, llm=_BudgetLLM(), tools=tools,
                      dispatch=_collect(events), hooks=Hooks(), max_steps=2)
    assert result.stop_reason == "max_steps"
    interrupts = [e for e in events if isinstance(e, TurnInterrupted)]
    assert len(interrupts) == 1
    assert interrupts[0].reason == "max_steps"
    assert interrupts[0].message == BUDGET_EXHAUSTED("max_steps")


@check
def denied_tool_surfaces_denial_wording_into_last_error():
    # PermissionHook with an `ask` policy and no prompt -> DENIAL_NO_PROMPT.
    s = Slice(); s.reset("t")

    def ask_policy(name, args):
        return ToolDecision(False, "needs approval", ask=True)

    hook = PermissionHook(ask_policy, on_ask=None)
    events = []
    sink = slice_sink(s)
    run_tool_batch([_TC()], _NoopTools(), lambda e: (events.append(e), sink(e)), hook)
    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 1
    out = results[0].output
    assert out.startswith("Error: blocked by policy:")
    assert DENIAL_NO_PROMPT in out
    assert s.last_error.startswith("Error: blocked by policy:")
    assert "approval channel" in s.last_error  # phrase unique to DENIAL_NO_PROMPT


@check
def denied_by_user_uses_denial_user_wording():
    def ask_policy(name, args):
        return ToolDecision(False, "needs approval", ask=True)

    hook = PermissionHook(ask_policy, on_ask=lambda n, a, r: "no")
    d = hook.authorize_tool("edit_file", {"path": "x"})
    assert d.allow is False
    assert d.reason == DENIAL_USER


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
