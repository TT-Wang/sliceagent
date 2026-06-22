"""run_turn overflow + interrupt paths (the rarely-fired branches a happy-path suite never reaches).
Covers three deficiencies found by adversarial review of the collapsed accumulate loop:
  A. overflow compaction must drop a WHOLE exchange (assistant + ALL its tool replies), not a fixed
     2-window — else parallel tool calls leave orphaned `tool` messages (invalid → provider 400).
  B. if the SEED itself overflows (nothing left to compact), fail SOFT (TurnInterrupted 'overflow'),
     never raise uncaught and crash the session.
  C. ctrl-C during the TOOL phase (a hung run_command) aborts cleanly, not just during llm.complete.
No model, no pytest. Run: PYTHONPATH=src python tests/test_loop_overflow.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.context_overflow import ContextOverflow                # noqa: E402
from memagent.events import TurnInterrupted                          # noqa: E402
from memagent.hooks import BudgetHook, Hooks, OracleHook             # noqa: E402
from memagent.interfaces import Snippet                              # noqa: E402
from memagent.loop import run_turn                                   # noqa: E402
from memagent.slice import Slice, make_build_slice                   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Resp:
    def __init__(self, *, content="", tool_calls=None, finish_reason="tool_use", usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {"prompt_tokens": 1, "completion_tokens": 1}


class _TC:
    def __init__(self, name, args, id):
        self.name = name; self.args = args; self.id = id


class _Retriever:
    def retrieve(self, query, k=6):
        return [] if k <= 0 else [Snippet(path="r.py", text="x", score=1.0)]


class _Tools:
    def schemas(self):
        return []
    def accesses(self, name, args):
        return []          # no conflicts → parallel calls run concurrently
    def run(self, name, args):
        return "ok"


class _KbdTools(_Tools):
    def run(self, name, args):
        raise KeyboardInterrupt()   # simulate ctrl-C during tool execution


class _ScriptLLM:
    """Returns scripted responses per complete(); 'OVERFLOW' raises ContextOverflow. Records the
    EXACT messages it was handed each call so we can assert the post-compaction sequence is valid."""
    def __init__(self, script):
        self.script = script; self.i = 0; self.seen = []
    def complete(self, messages, schemas):
        self.seen.append([dict(m) for m in messages])
        r = self.script[min(self.i, len(self.script) - 1)]; self.i += 1
        if r == "OVERFLOW":
            raise ContextOverflow(RuntimeError("context_length_exceeded"))
        return r


def _valid_tool_sequence(msgs) -> bool:
    """Every `tool` message must reference a tool_call_id DECLARED by a preceding assistant message."""
    declared = set()
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                declared.add(tc["id"])
        elif m.get("role") == "tool":
            if m.get("tool_call_id") not in declared:
                return False
    return True


def _build():
    s = Slice(); s.reset("t")
    return make_build_slice(s, _Tools(), _Retriever(), None, "t")


@check
def overflow_compacts_whole_exchange_not_fixed_window():
    # step1: 3 PARALLEL tool calls; step2: 1 tool call; step3: overflow → must drop the WHOLE 3-call
    # exchange (not a 2-window that would orphan two tool messages). Then it completes.
    llm = _ScriptLLM([
        _Resp(tool_calls=[_TC("noop", {}, "c1"), _TC("noop", {}, "c2"), _TC("noop", {}, "c3")]),
        _Resp(tool_calls=[_TC("noop", {}, "c4")]),
        "OVERFLOW",
        _Resp(content="done", finish_reason="stop"),
    ])
    events = []
    res = run_turn(build_slice=_build(), llm=llm, tools=_Tools(),
                   dispatch=lambda e: events.append(e), hooks=Hooks(), max_steps=10)
    assert res.stop_reason == "end_turn", res.stop_reason
    post = llm.seen[-1]   # messages handed to the final (post-compaction) complete()
    assert _valid_tool_sequence(post), f"orphaned tool message after compaction: {[m.get('role') for m in post]}"
    # the oldest (3-call) exchange was dropped wholesale → no leading orphan tool messages
    seed_then = [m.get("role") for m in post][2:]   # seed is [system, user]
    assert "tool" not in seed_then[:1], f"first post-seed message is an orphan tool: {seed_then}"


@check
def seed_overflow_fails_soft_not_crash():
    # the seed itself always overflows (nothing accumulated to compact) → graceful, not an exception.
    llm = _ScriptLLM(["OVERFLOW"])
    events = []
    res = run_turn(build_slice=_build(), llm=llm, tools=_Tools(),
                   dispatch=lambda e: events.append(e), hooks=Hooks(), max_steps=10)
    assert res.stop_reason == "overflow", res.stop_reason
    interrupts = [e for e in events if isinstance(e, TurnInterrupted)]
    assert interrupts and interrupts[0].reason == "overflow", interrupts


@check
def ctrl_c_during_tool_aborts_gracefully():
    llm = _ScriptLLM([_Resp(tool_calls=[_TC("noop", {}, "c1")]), _Resp(content="done", finish_reason="stop")])
    events = []
    res = run_turn(build_slice=_build(), llm=llm, tools=_KbdTools(),
                   dispatch=lambda e: events.append(e), hooks=Hooks(), max_steps=10)
    assert res.stop_reason == "aborted", res.stop_reason
    assert any(isinstance(e, TurnInterrupted) and e.reason == "aborted" for e in events)


class _BudgetLLM:
    def complete(self, messages, schemas):
        return _Resp(tool_calls=[_TC("noop", {}, "c1")], usage={"prompt_tokens": 10, "completion_tokens": 5})


@check
def budget_stop_parks_not_done():
    # F: a token-budget stop must PARK (a non-end_turn reason → caller records 'parked'), NEVER 'done'.
    events = []
    res = run_turn(build_slice=_build(), llm=_BudgetLLM(), tools=_Tools(),
                   dispatch=lambda e: events.append(e), hooks=BudgetHook(15), max_steps=20)
    assert res.stop_reason != "end_turn", f"budget stop masqueraded as done: {res.stop_reason}"
    assert res.stop_reason == "token_budget", res.stop_reason
    assert any(isinstance(e, TurnInterrupted) and e.reason == "token_budget" for e in events)


class _Oracle:
    def __init__(self, results):
        self.results = results; self.i = 0
    def verify(self):
        r = self.results[min(self.i, len(self.results) - 1)]; self.i += 1
        return r


@check
def oracle_feedback_rides_the_message_channel():
    # N: when verify() fails, the failure DETAIL must reach the model via the message channel (the seed
    # never re-renders mid-turn, so last_error is invisible). The model must SEE "TESTS FAILED: foo".
    llm = _ScriptLLM([_Resp(content="done", finish_reason="stop")])   # always "ends"; oracle drives retry
    hooks = OracleHook(_Oracle([(False, "TESTS FAILED: foo"), (True, "")]), lambda out: None)
    res = run_turn(build_slice=_build(), llm=llm, tools=_Tools(),
                   dispatch=lambda e: None, hooks=hooks, max_steps=10)
    assert res.stop_reason == "end_turn", res.stop_reason
    # call 2 (the retry) must have been handed a message containing the failure detail
    retry_msgs = llm.seen[1]
    assert any("TESTS FAILED: foo" in (m.get("content") or "") for m in retry_msgs), \
        "Oracle failure detail never reached the model (verify-loop dead)"


class _CountLLM:
    def __init__(self):
        self.calls = 0
    def complete(self, messages, schemas):
        self.calls += 1
        return _Resp(tool_calls=[_TC("noop", {}, f"c{self.calls}")],
                     usage={"prompt_tokens": 10, "completion_tokens": 5})


@check
def closeout_tokens_are_accounted():
    # G: the closeout's extra completion must be counted in total (and fed to the budget). max_steps=2 →
    # 2 step calls + 1 closeout call = 3 × 10 prompt tokens = 30. Without the fix the closeout is invisible.
    llm = _CountLLM()
    res = run_turn(build_slice=_build(), llm=llm, tools=_Tools(),
                   dispatch=lambda e: None, hooks=Hooks(), max_steps=2)
    assert llm.calls == 3, f"expected 2 steps + 1 closeout call, got {llm.calls}"
    assert res.usage["prompt_tokens"] == 30, f"closeout tokens not accounted: {res.usage}"


class _ErrLLM:
    def is_retryable(self, e):
        return False   # force with_retry to re-raise immediately (no backoff sleep in the test)
    def complete(self, messages, schemas):
        raise RuntimeError("boom: simulated non-retryable provider error")


@check
def unexpected_llm_error_parks_not_crashes():
    # Q: a non-retryable llm error past with_retry must route through _park (reason 'error'), never escape
    # run_turn uncaught (which would kill the session with no TurnInterrupted).
    events = []
    res = run_turn(build_slice=_build(), llm=_ErrLLM(), tools=_Tools(),
                   dispatch=lambda e: events.append(e), hooks=Hooks(), max_steps=10)
    assert res.stop_reason == "error", res.stop_reason
    assert any(isinstance(e, TurnInterrupted) and e.reason == "error" for e in events)


@check
def throwing_build_slice_parks_not_crashes():
    # R: a build_slice (memory/retriever/probe) that throws BEFORE the loop must park, not crash.
    def boom():
        raise RuntimeError("retriever exploded during seed build")
    events = []
    res = run_turn(build_slice=boom, llm=_ScriptLLM([_Resp(content="x", finish_reason="stop")]),
                   tools=_Tools(), dispatch=lambda e: events.append(e), hooks=Hooks(), max_steps=10)
    assert res.stop_reason == "error", res.stop_reason
    assert any(isinstance(e, TurnInterrupted) and e.reason == "error" for e in events)


@check
def selfcheck_forces_one_verification_pass_then_accepts():
    # SelfCheckHook: first 'done' -> forced verification feedback (rides messages); second 'done' accepted.
    from memagent.hooks import SelfCheckHook
    llm = _ScriptLLM([_Resp(content="done", finish_reason="stop"),
                      _Resp(content="verified, done", finish_reason="stop")])
    res = run_turn(build_slice=_build(), llm=llm, tools=_Tools(),
                   dispatch=lambda e: None, hooks=SelfCheckHook(), max_steps=10)
    assert res.stop_reason == "end_turn", res.stop_reason
    assert len(llm.seen) == 2, f"expected done->self-check->done (2 calls), got {len(llm.seen)}"
    assert any("definition-of-done" in (m.get("content") or "") for m in llm.seen[1]), \
        "self-check feedback was not delivered to the model"


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
