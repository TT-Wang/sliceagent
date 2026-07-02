"""The clear config journey (user-designed): /config manages providers in-session, and /model shows
ONLY configured providers' models — picking one switches model + endpoint + key together.
Offline: stub objects, no network, no pytest. Run: PYTHONPATH=src python tests/test_config_journey.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.config import Config                      # noqa: E402
from sliceagent.llm import OpenAILLM                      # noqa: E402
from sliceagent.tui import _SLASH, _model_candidates, select_model_reasoning  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Llm:
    model = "deepseek-chat"
    _base_url = "https://api.deepseek.com/v1"
    reasoning = "full"


_CFG = Config({"providers": {
    "deepseek": {"api_key": "k1", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "openrouter": {"api_key": "k2", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-5.5"},
    "keyless": {"model": "phantom-model"},               # no api_key → NOT usable → must not appear
}})


@check
def model_menu_lists_only_configured_providers():
    cands = _model_candidates(_Llm(), _CFG)
    pids = {pid for _, _, pid in cands if pid}
    models = [m for m, _, _ in cands]
    assert pids == {"deepseek", "openrouter"}, f"only keyed providers may appear, got {pids}"
    assert "phantom-model" not in models, "a provider without a key must not contribute models"
    assert "deepseek-chat" in models and "openai/gpt-5.5" in models
    assert "anthropic/claude-sonnet-5" in models, "configured providers contribute their suggestions too"


@check
def every_candidate_carries_its_provider_for_endpoint_switch():
    for m, _grp, pid in _model_candidates(_Llm(), _CFG):
        assert pid in ("deepseek", "openrouter"), f"{m} must map to a configured provider, got {pid!r}"


@check
def no_providers_configured_falls_back_to_known_set():
    cands = _model_candidates(_Llm(), Config({}))
    assert cands, "env-only setups still need a menu"
    assert all(pid is None for _, _, pid in cands), "fallback entries have no provider to rebind to"
    assert any(m == "deepseek-chat" for m, _, _ in cands)


@check
def switch_rebinds_endpoint_and_key():
    saved = {k: os.environ.pop(k, None) for k in ("AGENT_PROXY", "HTTPS_PROXY", "HTTP_PROXY")}
    try:
        llm = OpenAILLM(model="deepseek-chat", api_key="k1", base_url="https://api.deepseek.com/v1")
        assert llm._base_url == "https://api.deepseek.com/v1"
        llm.switch(model="openai/gpt-5.5", base_url="https://openrouter.ai/api/v1", api_key="k2")
        assert llm.model == "openai/gpt-5.5"
        assert llm._base_url == "https://openrouter.ai/api/v1"
        assert str(llm.client.base_url).startswith("https://openrouter.ai/api/v1"), \
            "the OpenAI client itself must be rebuilt against the new endpoint"
        assert llm.client.api_key == "k2", "the key must follow the provider"
        # base_url="" → SDK default endpoint (OpenAI)
        llm.switch(model="gpt-5.5", base_url="", api_key="k3")
        assert llm._base_url == "" and "openrouter" not in str(llm.client.base_url)
        # model-only switch keeps the endpoint (the classic /model <name> path)
        llm.switch(model="gpt-5")
        assert llm.model == "gpt-5" and llm.client.api_key == "k3"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@check
def selector_returns_the_provider_triple():
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    with create_pipe_input() as pinp:
        pinp.send_text("\r\r")                     # Enter (model) + Enter (reasoning)
        got = select_model_reasoning(_Llm(), _CFG, pt_input=pinp, pt_output=DummyOutput())
    assert got is not None and len(got) == 3, f"expected (model, reasoning, pid), got {got!r}"
    model, _reasoning, pid = got
    assert pid in ("deepseek", "openrouter") and isinstance(model, str)


@check
def config_command_is_in_the_slash_palette():
    assert "/config" in _SLASH, "/config must be discoverable in the palette"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
