"""Two interactive-robustness fixes: provider-aware proxy choice, and ctrl-c aborting a turn even
while the LLM is 'thinking' (a blocking call). No model, no pytest.
Run: PYTHONPATH=src python tests/test_proxy_interrupt.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.llm import _CLASHX, _choose_proxy                      # noqa: E402
from memagent.events import TurnInterrupted, make_dispatcher         # noqa: E402
from memagent.hooks import CompositeHooks                            # noqa: E402
from memagent.loop import run_turn                                   # noqa: E402
from memagent.memory import NullMemory                               # noqa: E402
from memagent.retriever import NullRetriever                         # noqa: E402
from memagent.slice import Slice, make_build_slice, slice_sink       # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── provider-aware proxy ─────────────────────────────────────────────────────
@check
def openai_endpoint_uses_clashx_only_when_a_local_proxy_is_up():
    # The default is DIRECT for everyone; a foreign endpoint falls back to the local ClashX proxy ONLY if one
    # is actually listening (so non-CN first runs don't fail on a refused 127.0.0.1:7890). Mock the probe so
    # this is deterministic regardless of whether the CI/dev box happens to have a proxy running.
    import memagent.llm as _llm
    orig = _llm._local_proxy_listening
    try:
        _llm._local_proxy_listening = lambda url: True
        assert _choose_proxy("https://api.openai.com/v1", None) == _CLASHX   # proxy present → use it (CN behind GFW)
        _llm._local_proxy_listening = lambda url: False
        assert _choose_proxy("https://api.openai.com/v1", None) == "none"    # no proxy → DIRECT (non-CN default)
    finally:
        _llm._local_proxy_listening = orig


@check
def cn_providers_go_direct_by_default():
    assert _choose_proxy("https://api.deepseek.com", None) == "none"
    assert _choose_proxy("https://api.moonshot.cn/v1", None) == "none"
    assert _choose_proxy("http://127.0.0.1:11434/v1", None) == "none"   # local model


@check
def explicit_proxy_always_wins():
    assert _choose_proxy("https://api.openai.com/v1", "none") == "none"          # user forces direct
    assert _choose_proxy("https://api.deepseek.com", "http://x:1") == "http://x:1"  # user forces proxy


# ── ctrl-c during a 'thinking' (blocking) call aborts the turn ───────────────
class _CtrlCLLM:
    """Simulates ctrl-c landing inside the blocking llm.complete() (the 'thinking' phase)."""
    def complete(self, messages, schemas):
        raise KeyboardInterrupt()


class _Tools:
    def schemas(self): return []
    def accesses(self, n, a): return []
    def run(self, n, a): return "ok"
    def root(self): return "/tmp"
    def read_text(self, p): raise FileNotFoundError(p)


@check
def ctrl_c_while_thinking_aborts_the_turn_cleanly():
    s = Slice(); s.reset("do a long thing")
    tools = _Tools()
    events = []
    dispatch = make_dispatcher(slice_sink(s), events.append)
    build = make_build_slice(s, tools, NullRetriever(), NullMemory(), s.goal)
    res = run_turn(build_slice=build, llm=_CtrlCLLM(), tools=tools, dispatch=dispatch,
                   hooks=CompositeHooks(), max_steps=10)
    assert res.stop_reason == "aborted", res.stop_reason
    assert any(isinstance(e, TurnInterrupted) and e.reason == "aborted" for e in events)


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
