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
