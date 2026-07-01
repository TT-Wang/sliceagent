"""The missing 'come back and ask' capability: the ask_user tool + the anti-spin floor that hands
control back to the user after repeated guardrail blocks. No model, no pytest.
Run: PYTHONPATH=src python tests/test_ask_user.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import TurnInterrupted, make_dispatcher  # noqa: E402
from memagent.hooks import CompositeHooks, GuardrailHook       # noqa: E402
from memagent.loop import STUCK_BLOCK_BUDGET, run_turn         # noqa: E402
from memagent.memory import NullMemory                         # noqa: E402
from memagent.retriever import NullRetriever                   # noqa: E402
from memagent.pfc import Slice, slice_sink  # noqa: E402
from memagent.seed import make_build_slice  # noqa: E402
from memagent.tools import LocalToolHost                       # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── ask_user tool (the capability) ───────────────────────────────────────────
@check
def ask_user_returns_the_users_answer():
    host = LocalToolHost("/tmp")
    seen = {}
    host.on_ask_user = lambda q, opts: (seen.update(q=q, opts=opts) or "blue")
    out = host.run("ask_user", {"question": "which color?", "options": ["red", "blue"]})
    assert out == "User answered: blue", out
    assert seen["q"] == "which color?" and seen["opts"] == ["red", "blue"], seen


@check
def ask_user_default_is_non_interactive_and_never_hangs():
    out = LocalToolHost("/tmp").run("ask_user", {"question": "anything?"})
    assert "no interactive user" in out, out


@check
def ask_user_requires_a_question():
    assert "requires" in LocalToolHost("/tmp").run("ask_user", {"question": "  "})


@check
def ask_user_is_advertised_in_the_schema():
    names = [s["function"]["name"] for s in LocalToolHost("/tmp").schemas()]
    assert "ask_user" in names, names


# ── anti-spin floor: repeated blocks → hand back to the user ──────────────────
class _TC:
    def __init__(self, name, args):
        self.name, self.args = name, args


class _Resp:
    def __init__(self, tool_calls):
        self.content = "trying again"
        self.tool_calls = tool_calls
        self.finish_reason = "tool_calls"
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1}


class _SpinLLM:
    """Always emits the SAME failing tool call — a model stuck in a loop. (No len(messages)==2 assert:
    the default loop_mode is now 'accumulate', where working memory grows within the loop; this test
    exercises the mode-agnostic STUCK floor, not the rebuild-only [system,user] shape.)"""
    def complete(self, messages, schemas):
        return _Resp([_TC("read_file", {"path": "does-not-exist.py"})])


class _FailTools:
    def schemas(self): return []
    def accesses(self, n, a): return []
    def run(self, n, a): return "Error: nope"      # every call fails identically
    def root(self): return "/tmp"
    def read_text(self, p): raise FileNotFoundError(p)


@check
def repeated_blocks_stop_the_turn_and_hand_back():
    state = Slice(); state.reset("do the impossible thing")
    tools = _FailTools()
    events = []
    dispatch = make_dispatcher(slice_sink(state), events.append)
    build = make_build_slice(state, tools, NullRetriever(), NullMemory(), state.goal)
    hooks = CompositeHooks(GuardrailHook())
    res = run_turn(build_slice=build, llm=_SpinLLM(), tools=tools, dispatch=dispatch, hooks=hooks, max_steps=40)
    assert res.stop_reason == "stuck", res.stop_reason          # NOT max_steps — the floor fired first
    assert res.steps < 40, f"floor must fire well before max_steps, got {res.steps}"
    ti = [e for e in events if isinstance(e, TurnInterrupted) and e.reason == "stuck"]
    assert ti and "ask_user" in (ti[0].message or ""), "stuck message should point at ask_user"


@check
def stuck_budget_is_bounded_and_small():
    assert 1 <= STUCK_BLOCK_BUDGET <= 5


# ── Fix 5b: closeout never ends silently, offers ask_user, falls back deterministically ──
import types  # noqa: E402
from memagent.events import AssistantText  # noqa: E402
from memagent.loop import _final_answer    # noqa: E402


class _ToolsWithAsk:
    def schemas(self): return [{"function": {"name": "ask_user"}}, {"function": {"name": "read_file"}}]


def _collect():
    out = []
    return out, (lambda e: out.append(e.content) if isinstance(e, AssistantText) else None)


@check
def closeout_emits_model_summary_when_present():
    class _LLM:
        def complete(self, m, t): return types.SimpleNamespace(
            content="Done: fixed the parser; verified import compiles; nothing remains.", tool_calls=[])
    out, disp = _collect()
    _final_answer(_LLM(), [{"role": "user", "content": "x"}], _ToolsWithAsk(), disp, "max_steps")
    assert out and "fixed the parser" in out[0]


@check
def closeout_never_silent_on_empty_completion():
    class _LLM:
        def complete(self, m, t): return types.SimpleNamespace(content="", tool_calls=[])
    out, disp = _collect()
    _final_answer(_LLM(), [{"role": "user", "content": "x"}], _ToolsWithAsk(), disp, "max_steps")
    assert out and len(out[0]) > 20, "must emit a non-empty deterministic fallback, never silence"


@check
def closeout_offers_ask_and_surfaces_the_question():
    seen = {}
    class _LLM:
        def complete(self, m, tools):
            seen["tools"] = tools
            tc = types.SimpleNamespace(name="ask_user", args={"question": "Which module did you mean?"})
            return types.SimpleNamespace(content="", tool_calls=[tc])
    out, disp = _collect()
    _final_answer(_LLM(), [{"role": "user", "content": "x"}], _ToolsWithAsk(), disp, "stuck")
    assert seen["tools"] and seen["tools"][0]["function"]["name"] == "ask_user", "ask_user must be offered"
    assert out == ["Which module did you mean?"], out


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
