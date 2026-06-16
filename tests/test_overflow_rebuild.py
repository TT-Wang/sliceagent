"""ITEM 3 — context-overflow rebuild loop (W1 run_step) + W1 wiring integration cases.
No model, no pytest. Run: python tests/test_overflow_rebuild.py

Covers:
  - build.rebuild_tighter() True while it can tighten, False at the floor (W2 contract).
  - the tighter build is SMALLER (RELATED CODE drops out at the floor).
  - run_step: an LLM raising ContextOverflow once then succeeding rebuilds ONCE, the
    step still completes, step_num is unchanged, SliceTightened is dispatched, and the
    messages stay [system, user] (length 2 — no transcript growth).
  - always-overflow re-raises after the floor (bounded — no infinite same-step retry).
  - W1 integration (deferred from test_guidance): run_turn max_steps dispatches
    TurnInterrupted(reason='max_steps', message=BUDGET_EXHAUSTED('max_steps')); a denied
    tool surfaces 'Error: blocked by policy:' carrying DENIAL_* into slice.last_error.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import (                                  # noqa: E402
    SliceBuilt,
    SliceTightened,
    ToolResult,
    TurnInterrupted,
)
from memagent.guidance import BUDGET_EXHAUSTED, DENIAL_NO_PROMPT, DENIAL_USER  # noqa: E402
from memagent.context_overflow import ContextOverflow         # noqa: E402
from memagent.hooks import ALLOW, Hooks, PermissionHook, ToolDecision  # noqa: E402
from memagent.interfaces import Snippet                        # noqa: E402
from memagent.loop import run_step, run_tool_batch, run_turn   # noqa: E402
from memagent.slice import Slice, make_build_slice, record_action, slice_sink  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# --- fakes -------------------------------------------------------------------

class _Tools:
    """Minimal ToolHost stub for make_build_slice (root + read_text + schemas)."""
    def __init__(self, root="/tmp/does-not-matter"):
        self._root = root
        self.files = {}
    def root(self):
        return self._root
    def read_text(self, path):
        if path in self.files:
            return self.files[path]
        raise FileNotFoundError(path)
    def schemas(self):
        return []
    def accesses(self, name, args):
        return []
    def run(self, name, args):
        return "ok"


class _Retriever:
    """Returns one snippet so RELATED CODE renders at level 0 (discovery_k>0)."""
    def retrieve(self, query, k=6):
        if k <= 0:
            return []
        return [Snippet(path="related.py", text="def helper():\n    return 1", score=1.0)]


class _Resp:
    """An LLMClient .complete() response (duck-typed: finish_reason/tool_calls/content/usage)."""
    def __init__(self, *, content="done", tool_calls=None, finish_reason="stop", usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {"prompt_tokens": 1, "completion_tokens": 1}


class _OverflowThenOk:
    """First complete() raises ContextOverflow; subsequent calls succeed. Tracks call count."""
    def __init__(self, fail_times=1):
        self.fail_times = fail_times
        self.calls = 0
    def complete(self, messages, schemas):
        self.calls += 1
        # assert the moat invariant on EVERY call: [system, user], length 2.
        assert len(messages) == 2, f"messages must stay length 2, got {len(messages)}"
        if self.calls <= self.fail_times:
            raise ContextOverflow(RuntimeError("maximum context length exceeded"))
        return _Resp()


class _AlwaysOverflow:
    def __init__(self):
        self.calls = 0
    def complete(self, messages, schemas):
        self.calls += 1
        assert len(messages) == 2
        raise ContextOverflow(RuntimeError("context_length_exceeded"))


def _collect(events):
    def dispatch(e):
        events.append(e)
    return dispatch


def _real_build():
    s = Slice(); s.reset("fix the parser")
    s.active_files = ["parser.py"]
    s.findings = ["root cause is the lexer", "the fix is in tokenize()", "edge case empty input"]
    tools = _Tools()
    tools.files["parser.py"] = "\n".join(f"line {i}" for i in range(1, 60))
    return make_build_slice(s, tools, _Retriever(), None, "fix the parser")


# --- rebuild_tighter contract (W2, exercised here) ---------------------------

@check
def rebuild_tighter_true_then_false_at_floor():
    build = _real_build()
    # 3 tier levels (0,1,2): two tightenings succeed, the third (at floor) returns False.
    assert build.rebuild_tighter() is True      # 0 -> 1
    assert build.rebuild_tighter() is True       # 1 -> 2 (floor)
    assert build.rebuild_tighter() is False      # already at floor — give up


@check
def tighter_build_is_smaller_related_code_absent_at_floor():
    build = _real_build()
    base = build()[1]["content"]
    assert "# RELATED CODE" in base                     # discovery present at level 0
    build.rebuild_tighter()                              # -> level 1 (still has discovery, smaller)
    mid = build()[1]["content"]
    build.rebuild_tighter()                              # -> level 2 (floor: discovery_k=0)
    floor = build()[1]["content"]
    assert "# RELATED CODE" not in floor                # discovery dropped at the floor
    assert len(floor) < len(base)                        # the floor slice is strictly smaller


# --- run_step overflow loop --------------------------------------------------

@check
def run_step_rebuilds_once_then_completes():
    build = _real_build()
    llm = _OverflowThenOk(fail_times=1)
    tools = _Tools()
    events = []
    outcome = run_step(step_num=3, build_slice=build, llm=llm, tools=tools,
                       dispatch=_collect(events), hooks=Hooks())
    # the step completed despite the first overflow
    assert outcome.stop_reason == "end_turn"
    # the llm was called twice (overflow, then success) — rebuilt exactly once
    assert llm.calls == 2
    tightened = [e for e in events if isinstance(e, SliceTightened)]
    assert len(tightened) == 1 and tightened[0].level == 1
    # step_num unchanged: only StepBegin(3) is dispatched (no step renumbering)
    from memagent.events import StepBegin
    begins = [e for e in events if isinstance(e, StepBegin)]
    assert len(begins) == 1 and begins[0].step == 3
    # SliceBuilt dispatched for the initial build AND the rebuild (length 2 each)
    builts = [e for e in events if isinstance(e, SliceBuilt)]
    assert len(builts) == 2
    assert all(b.messages is not None and len(b.messages) == 2 for b in builts)


@check
def run_step_always_overflow_reraises_after_floor():
    build = _real_build()
    llm = _AlwaysOverflow()
    tools = _Tools()
    events = []
    raised = False
    try:
        run_step(step_num=1, build_slice=build, llm=llm, tools=tools,
                 dispatch=_collect(events), hooks=Hooks())
    except ContextOverflow:
        raised = True
    assert raised, "an unfixable overflow must re-raise (no infinite loop)"
    # bounded: initial + 2 tightenings (floor reached) -> 3 llm calls, then re-raise
    assert llm.calls == 3, f"expected 3 bounded attempts, got {llm.calls}"
    tightened = [e for e in events if isinstance(e, SliceTightened)]
    assert len(tightened) == 2                            # tightened twice, then gave up


@check
def messages_stay_length_two_across_rebuild():
    # the _OverflowThenOk fake asserts len==2 on every complete() call; reaching here
    # with calls==2 proves both the initial and rebuilt messages were [system, user].
    build = _real_build()
    llm = _OverflowThenOk(fail_times=1)
    run_step(step_num=1, build_slice=build, llm=llm, tools=_Tools(),
             dispatch=lambda e: None, hooks=Hooks())
    assert llm.calls == 2


# --- W1 integration: budget guidance + denial wording ------------------------

class _BudgetLLM:
    """Always returns a non-tool 'end_turn' response so run_turn keeps issuing steps
    until max_steps — exercising the budget-ceiling branch."""
    def complete(self, messages, schemas):
        return _Resp(content="", tool_calls=[_TC()], finish_reason="tool_use")


class _TC:
    name = "noop"
    args: dict = {}


class _NoopTools:
    def schemas(self):
        return []
    def accesses(self, name, args):
        return []
    def run(self, name, args):
        return "ok"


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


def _deny(reason):
    def policy(name, args):
        return ToolDecision(False, reason)
    return policy


@check
def denied_tool_surfaces_denial_wording_into_last_error():
    # PermissionHook with an `ask` policy and no prompt -> DENIAL_NO_PROMPT.
    s = Slice(); s.reset("t")

    def ask_policy(name, args):
        return ToolDecision(False, "needs approval", ask=True)

    hook = PermissionHook(ask_policy, on_ask=None)
    events = []
    dispatch = _collect(events)
    sink = slice_sink(s)
    # run a tool batch through the real loop path -> blocked -> ToolResult -> slice_sink
    tc = _TC()
    run_tool_batch([tc], _NoopTools(), lambda e: (events.append(e), sink(e)), hook)
    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(results) == 1
    out = results[0].output
    assert out.startswith("Error: blocked by policy:")
    assert DENIAL_NO_PROMPT in out
    # and it landed in the durable CURRENT ERROR tier
    assert s.last_error.startswith("Error: blocked by policy:")
    assert "approval channel" in s.last_error  # a phrase unique to DENIAL_NO_PROMPT


@check
def denied_by_user_uses_denial_user_wording():
    # an interactive prompt that says "no" -> DENIAL_USER.
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
