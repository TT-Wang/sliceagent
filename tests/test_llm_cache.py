"""Anthropic prompt-cache breakpoint on the stable system prefix (#22 cache-efficiency). The bounded slice
keeps a byte-stable system prefix; on a Claude endpoint we mark it with a `cache_control` breakpoint so
Anthropic serves it from cache on later same-prefix turns. No-op for OpenAI-compatible providers (they
cache automatically / via prompt_cache_key). No network. Run: PYTHONPATH=src python tests/test_llm_cache.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.llm import OpenAILLM  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def marks_system_prefix_with_ephemeral_breakpoint():
    msgs = [{"role": "system", "content": "STABLE PREFIX"}, {"role": "user", "content": "hi"}]
    OpenAILLM._mark_cache_breakpoint(msgs)
    assert msgs[0]["content"] == [{"type": "text", "text": "STABLE PREFIX",
                                   "cache_control": {"type": "ephemeral"}}], msgs[0]
    assert msgs[1]["content"] == "hi", "the volatile user turn must be left untouched"


@check
def mark_is_idempotent():
    msgs = [{"role": "system", "content": "P"}, {"role": "user", "content": "hi"}]
    OpenAILLM._mark_cache_breakpoint(msgs)
    OpenAILLM._mark_cache_breakpoint(msgs)
    c = msgs[0]["content"]
    assert isinstance(c, list) and len(c) == 1 and c[0].get("cache_control") == {"type": "ephemeral"}, c
    assert c[0]["text"] == "P", "must not double-wrap the content"


@check
def no_system_message_is_a_noop():
    msgs = [{"role": "user", "content": "hi"}]
    OpenAILLM._mark_cache_breakpoint(msgs)
    assert msgs == [{"role": "user", "content": "hi"}]


@check
def cache_kwargs_gates_on_provider():
    def _kw(model, base):
        c = OpenAILLM(model=model, api_key="x", base_url=base)
        msgs = [{"role": "system", "content": "PREFIX"}, {"role": "user", "content": "hi"}]
        return c._cache_kwargs(msgs), msgs
    # Claude endpoint → marks the system prefix, adds NO top-level kwarg
    out, msgs = _kw("claude-3-5-sonnet", "https://api.anthropic.com")
    assert out == {}, out
    assert isinstance(msgs[0]["content"], list) and msgs[0]["content"][0].get("cache_control"), msgs
    # DeepSeek / OpenAI-compatible → no breakpoint, messages byte-stable
    out, msgs = _kw("deepseek-chat", "https://api.deepseek.com/v1")
    assert out == {} and msgs[0]["content"] == "PREFIX", (out, msgs)


def main():
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
