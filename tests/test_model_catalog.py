"""Model capability catalog (Kimi modelCatalog): family + wire quirks (tokens-param rename, reasoning_effort
support). The single source of truth llm.py consults. No model, no pytest.
Run: PYTHONPATH=src python tests/test_model_catalog.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.model_catalog import capability  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def openai_reasoning_models():
    for m in ("gpt-5.5", "gpt-5", "o1", "o3-mini", "o4"):
        c = capability(m)
        assert c.tokens_param == "max_completion_tokens", m
        assert c.supports_reasoning_effort is True, m


@check
def cn_providers_use_plain_max_tokens_no_reasoning_effort():
    for m, b in [("kimi-k2.7-code", "https://api.moonshot.cn/v1"),
                 ("deepseek-chat", "https://api.deepseek.com"),
                 ("claude-sonnet", "https://api.anthropic.com")]:
        c = capability(m, b)
        assert c.tokens_param == "max_tokens", (m, c)
        assert c.supports_reasoning_effort is False, (m, c)


@check
def unknown_is_safe_default():
    c = capability("some-future-model")
    assert c.family == "unknown" and c.tokens_param == "max_tokens" and c.supports_reasoning_effort is False


@check
def base_url_routes_when_model_name_is_generic():
    assert capability("default", "https://api.moonshot.cn/v1").family == "moonshot"
    assert capability("x", "https://api.deepseek.com").family == "deepseek"


@check
def gpt5_named_model_on_a_non_openai_endpoint_does_not_get_the_responses_route():
    # TT's crash: "/model gpt-5.5" while still connected to DeepSeek (switch() only changes the model
    # STRING, never the endpoint) → capability() used to grant reasoning_effort/responses routing by MODEL
    # NAME ALONE, so llm._effort() routed to /v1/responses on an endpoint that doesn't implement it →
    # openai.NotFoundError (404) → "an internal error ended the turn". The pairing must degrade instead.
    c = capability("gpt-5.5", "https://api.deepseek.com/v1")
    assert c.supports_reasoning_effort is False, "must NOT claim OpenAI-only reasoning wire support off-endpoint"
    assert c.tokens_param == "max_tokens", "must not apply the OpenAI max_completion_tokens rename either"
    # real OpenAI (default AND explicit base_url) must be completely unaffected
    for b in ("", "https://api.openai.com/v1"):
        c2 = capability("gpt-5.5", b)
        assert c2.supports_reasoning_effort is True and c2.tokens_param == "max_completion_tokens", (b, c2)


@check
def effort_never_routes_to_responses_api_off_openai():
    # the actual decision point in llm.py: OpenAILLM._effort() must return None for this pairing, so
    # complete() never calls self.client.responses.create() against a server without that route.
    from memagent.llm import OpenAILLM
    llm = OpenAILLM.__new__(OpenAILLM)   # bypass __init__ — _effort() only touches these 3 attrs
    llm.model, llm._base_url, llm.reasoning = "gpt-5.5", "https://api.deepseek.com/v1", "high"
    assert llm._effort() is None, "must fall through to plain chat/completions, never /v1/responses"


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
