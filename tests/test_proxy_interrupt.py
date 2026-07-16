"""Two interactive-robustness fixes: provider-aware proxy choice, and ctrl-c aborting a turn even
while the LLM is 'thinking' (a blocking call). No model, no pytest.
Run: PYTHONPATH=src python tests/test_proxy_interrupt.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.llm import OpenAILLM, _choose_proxy, _proxy_route_for_display  # noqa: E402
from sliceagent.events import TurnInterrupted, make_dispatcher         # noqa: E402
from sliceagent.hooks import CompositeHooks                            # noqa: E402
from sliceagent.loop import run_turn                                   # noqa: E402
from sliceagent.memory import NullMemory                               # noqa: E402
from sliceagent.retriever import NullRetriever                         # noqa: E402
from sliceagent.pfc import Slice, slice_sink  # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── no proxy by default; an explicit setting wins ───────────────────────────
@check
def no_proxy_by_default_for_any_endpoint():
    # Default is a DIRECT connection for every endpoint — there is no auto-proxy.
    for base in ("https://api.openai.com/v1", "https://api.deepseek.com",
                 "https://api.moonshot.cn/v1", "http://127.0.0.1:11434/v1", None):
        assert _choose_proxy(base, None) == "none", base


@check
def explicit_proxy_always_wins():
    assert _choose_proxy("https://api.openai.com/v1", "none") == "none"             # user forces direct
    assert _choose_proxy("https://api.openai.com/v1", "off") == "none"             # 'off' alias → direct
    assert _choose_proxy("https://api.deepseek.com", "http://x:1") == "http://x:1"  # user forces a proxy URL


@check
def public_proxy_route_omits_every_credential_bearing_component():
    raw = "http://alice:p%40ss@proxy.example:8443/private?token=secret#fragment"
    assert _proxy_route_for_display(raw) == "http://proxy.example:8443"
    assert _proxy_route_for_display("socks5://u:p@[2001:db8::1]:1080") == "socks5://[2001:db8::1]:1080"
    assert _proxy_route_for_display("http://[broken") == "configured proxy"


@check
def startup_and_live_model_status_keep_raw_proxy_private():
    """Both startup and /model render ``proxy_used``; transport alone retains the authenticated URL."""
    keys = ("AGENT_PROXY", "HTTPS_PROXY", "HTTP_PROXY")
    saved = {key: os.environ.pop(key, None) for key in keys}
    llm = None
    startup_proxy = "http://startup-user:startup-secret@proxy-one.invalid:8080"
    switched_proxy = "http://switch-user:switch-secret@proxy-two.invalid:9090"
    try:
        llm = OpenAILLM(
            model="proxy-status-test", api_key="test-key",
            base_url="http://provider-one.invalid/v1", proxy=startup_proxy, timeout=0.01,
        )
        assert llm.proxy_used == "http://proxy-one.invalid:8080"
        assert llm._transport_spec[2] == startup_proxy
        assert "startup-user" not in llm.proxy_used and "startup-secret" not in llm.proxy_used

        os.environ["AGENT_PROXY"] = switched_proxy
        llm.switch(api_key="test-key-2", base_url="http://provider-two.invalid/v1")
        assert llm.proxy_used == "http://proxy-two.invalid:9090"
        assert llm._transport_spec[2] == switched_proxy
        assert "switch-user" not in llm.proxy_used and "switch-secret" not in llm.proxy_used
    finally:
        if llm is not None:
            llm.client.close()
        for key in keys:
            if saved[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved[key]


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
