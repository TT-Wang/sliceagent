"""The five-door provider lineup (OpenRouter + OpenAI/Anthropic/DeepSeek/Moonshot + custom) and the
OpenRouter/Anthropic adapter quirks. All offline — stub objects, no network, no pytest.
Run: PYTHONPATH=src python tests/test_provider_lineup.py
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.llm import OpenAILLM, _usage_dict           # noqa: E402
from sliceagent.onboarding import MODEL_SUGGESTIONS, PROVIDERS  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _stub(model: str, base: str, reasoning: str = "full") -> OpenAILLM:
    obj = OpenAILLM.__new__(OpenAILLM)   # bypass __init__ (no network)
    obj.model = model
    obj._base_url = base
    obj.reasoning = reasoning
    return obj


@check
def lineup_is_the_agreed_five_doors_plus_custom():
    ids = [PROVIDERS[k][0] for k in sorted(PROVIDERS)]
    assert ids == ["openrouter", "openai", "anthropic", "deepseek", "moonshot", "custom"], ids
    for _pid, _label, base, model in PROVIDERS.values():
        if _pid != "custom":
            assert model, f"{_pid} preset needs a default model"
    assert PROVIDERS["3"][2] == "https://api.anthropic.com/v1"      # Claude via the OpenAI-compat endpoint
    assert all(PROVIDERS[k][0] in MODEL_SUGGESTIONS for k in PROVIDERS), "every preset needs a model menu"


@check
def openrouter_reasoning_maps_to_the_unified_object():
    llm = _stub("anthropic/claude-sonnet-5", "https://openrouter.ai/api/v1", reasoning="fast")
    assert llm._reasoning_kwargs() == {"extra_body": {"reasoning": {"effort": "low"}}}
    llm.reasoning = "high"
    assert llm._reasoning_kwargs() == {"extra_body": {"reasoning": {"effort": "high"}}}
    llm.reasoning = "full"   # default → provider default, no param at all (never force spend)
    assert llm._reasoning_kwargs() == {}


@check
def openrouter_never_sends_raw_reasoning_effort():
    # the raw chat param is exactly what 400s/DROPS silently upstream — the unified object replaces it
    llm = _stub("openai/gpt-5.5", "https://openrouter.ai/api/v1", reasoning="max")
    kw = llm._reasoning_kwargs()
    assert "reasoning_effort" not in kw and kw["extra_body"]["reasoning"]["effort"] == "high"


@check
def anthropic_direct_stays_on_provider_defaults():
    # claude via api.anthropic.com: no reasoning_effort (capability=False), no prompt_cache_key routing
    llm = _stub("claude-sonnet-5", "https://api.anthropic.com/v1", reasoning="high")
    assert llm._reasoning_kwargs() == {}, "anthropic family must not receive reasoning_effort"
    assert llm._cache_routing_kwargs() == {}, "OpenAI cache routing keys must not go to Anthropic"


@check
def usage_dict_parses_openrouter_cost():
    raw = NS(prompt_tokens=100, completion_tokens=10, prompt_tokens_details=None,
             cache_creation_input_tokens=0, cost=0.0042)
    u = _usage_dict(raw)
    assert u["cost_usd"] == 0.0042 and u["prompt_tokens"] == 100
    raw_no_cost = NS(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None,
                     cache_creation_input_tokens=0)
    assert "cost_usd" not in _usage_dict(raw_no_cost), "absent cost must not fabricate a field"


@check
def openrouter_tools_requests_pin_param_honoring_hosts():
    llm = _stub("openai/gpt-5.5", "https://openrouter.ai/api/v1", reasoning="fast")
    kwargs = {"extra_body": {}}
    llm._merge_kwargs(kwargs, llm._reasoning_kwargs())
    llm._merge_kwargs(kwargs, {"extra_body": {"provider": {"require_parameters": True}}})
    eb = kwargs["extra_body"]
    assert eb["reasoning"]["effort"] == "low" and eb["provider"]["require_parameters"] is True, \
        "both quirks must coexist in one extra_body (deep merge)"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
