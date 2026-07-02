"""W4 — LLM adapter: cache read-back, _cache_kwargs stub, extra_body MERGE guard.

No real API, no openai SDK needed (OpenAILLM imports openai only inside __init__; we bypass
__init__ via object.__new__ and inject a fake client). No pytest.
Run: python tests/test_llm_usage_cache.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.context_overflow import ContextOverflow             # noqa: E402
from sliceagent.llm import OpenAILLM                                 # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── Fakes mirroring the openai response shape complete() reads ───────────────
class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFn(name, arguments)

class _FakeMsg:
    def __init__(self, content="hi", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

class _FakeChoice:
    def __init__(self, msg, finish_reason="stop"):
        self.message = msg
        self.finish_reason = finish_reason

class _FakeDetails:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens

class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=3, cached_tokens=None,
                 with_details=True):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if with_details:
            # When cached_tokens is None we still attach details (cached_tokens=None) to prove
            # the read-back omits the key on a None value, not just on a missing attr.
            self.prompt_tokens_details = _FakeDetails(cached_tokens)
        # with_details=False → no prompt_tokens_details attr at all (older/other providers).

class _FakeResp:
    def __init__(self, usage, msg=None):
        self.choices = [_FakeChoice(msg or _FakeMsg())]
        self.usage = usage

class _FakeCompletions:
    def __init__(self, resp):
        self._resp = resp
        self.captured_kwargs = None
    def create(self, **kwargs):
        self.captured_kwargs = kwargs
        return self._resp

class _FakeChat:
    def __init__(self, resp):
        self.completions = _FakeCompletions(resp)

class _FakeClient:
    def __init__(self, resp):
        self.chat = _FakeChat(resp)


def _llm(*, model="gpt-5.5", base_url="", reasoning="full", resp=None):
    """Build an OpenAILLM WITHOUT running __init__ (which needs the openai SDK + network).
    Only the attrs complete()/_cache_kwargs()/_reasoning_kwargs() touch are set."""
    obj = object.__new__(OpenAILLM)
    obj.model = model
    obj._base_url = base_url
    obj.reasoning = reasoning
    obj.max_tokens = 8192
    obj._hard_timeout = 30  # set by __init__ in real use; _create()/_create_watchdog read it
    obj.client = _FakeClient(resp if resp is not None else _FakeResp(_FakeUsage()))
    return obj


# ── cached_tokens read-back ──────────────────────────────────────────────────
@check
def cached_tokens_copied_when_reported():
    llm = _llm(resp=_FakeResp(_FakeUsage(cached_tokens=7)))
    out = llm.complete([{"role": "user", "content": "x"}], [])
    assert out.usage["cached_tokens"] == 7, out.usage
    # base usage still present
    assert out.usage["prompt_tokens"] == 10 and out.usage["completion_tokens"] == 3


@check
def cached_tokens_omitted_when_none_value():
    # details present but cached_tokens is None → key must be omitted (no crash, no None entry).
    llm = _llm(resp=_FakeResp(_FakeUsage(cached_tokens=None, with_details=True)))
    out = llm.complete([{"role": "user", "content": "x"}], [])
    assert "cached_tokens" not in out.usage, out.usage


@check
def cached_tokens_omitted_when_details_absent():
    # no prompt_tokens_details attr at all → still no crash, key omitted.
    llm = _llm(resp=_FakeResp(_FakeUsage(with_details=False)))
    out = llm.complete([{"role": "user", "content": "x"}], [])
    assert "cached_tokens" not in out.usage, out.usage
    assert out.usage["prompt_tokens"] == 10


# ── _cache_kwargs model-sniff ────────────────────────────────────────────────
@check
def cache_kwargs_empty_for_non_claude_model():
    llm = _llm(model="gpt-5.5", base_url="https://api.openai.com/v1")
    assert llm._cache_kwargs([]) == {}


@check
def cache_kwargs_empty_for_deepseek():
    llm = _llm(model="deepseek-chat", base_url="https://api.deepseek.com")
    assert llm._cache_kwargs([]) == {}


@check
def cache_kwargs_stub_empty_for_claude_model():
    # Anthropic-compatible model is detected (no crash) but the real shape is DEFERRED → {} stub.
    llm = _llm(model="claude-sonnet-4", base_url="")
    assert llm._cache_kwargs([]) == {}


@check
def cache_kwargs_stub_empty_for_anthropic_base_url():
    llm = _llm(model="gpt-5.5", base_url="https://api.anthropic.com/v1")
    assert llm._cache_kwargs([]) == {}


# ── extra_body MERGE guard (the critic's ask) ────────────────────────────────
@check
def extra_body_merge_does_not_overwrite():
    # Simulate BOTH _reasoning_kwargs and _cache_kwargs setting extra_body: the merge helper must
    # keep both nested keys, not clobber. (Guards the future Anthropic cache_control wiring.)
    llm = _llm()
    kwargs = {"model": "m", "extra_body": {"thinking": {"type": "disabled"}}}
    llm._merge_kwargs(kwargs, {"extra_body": {"cache_control": {"type": "ephemeral"}}})
    assert kwargs["extra_body"] == {
        "thinking": {"type": "disabled"},
        "cache_control": {"type": "ephemeral"},
    }, kwargs["extra_body"]


@check
def merge_sets_when_no_prior_extra_body():
    llm = _llm()
    kwargs = {"model": "m"}
    llm._merge_kwargs(kwargs, {"extra_body": {"cache_control": {"type": "ephemeral"}}})
    assert kwargs["extra_body"] == {"cache_control": {"type": "ephemeral"}}


@check
def merge_non_extra_body_key_overwrites_normally():
    llm = _llm()
    kwargs = {"model": "m", "reasoning_effort": "high"}
    llm._merge_kwargs(kwargs, {"reasoning_effort": "low"})
    assert kwargs["reasoning_effort"] == "low"


# ── overflow signal: create() error → ContextOverflow (not retried as backoff) ──
class _OverflowRaisingCompletions:
    def create(self, **kwargs):
        raise ValueError("This model's maximum context length is 128000 tokens")

class _OverflowChat:
    completions = _OverflowRaisingCompletions()

class _OverflowClient:
    chat = _OverflowChat()


class _StatusOverflowError(Exception):
    status_code = 413
    def __str__(self):
        return "request entity too large"

class _StatusRaisingCompletions:
    def create(self, **kwargs):
        raise _StatusOverflowError()

class _StatusChat:
    completions = _StatusRaisingCompletions()

class _StatusClient:
    chat = _StatusChat()


class _PlainError(Exception):
    pass

class _PlainRaisingCompletions:
    def create(self, **kwargs):
        raise _PlainError("rate limit exceeded, please slow down")

class _PlainChat:
    completions = _PlainRaisingCompletions()

class _PlainClient:
    chat = _PlainChat()


@check
def overflow_text_raises_context_overflow():
    llm = _llm()
    llm.client = _OverflowClient()
    raised = None
    try:
        llm.complete([{"role": "user", "content": "x"}], [])
    except ContextOverflow as e:
        raised = e
    assert isinstance(raised, ContextOverflow), "expected ContextOverflow"
    assert raised.original is not None


@check
def overflow_413_status_copied_onto_exception():
    llm = _llm()
    llm.client = _StatusClient()
    raised = None
    try:
        llm.complete([{"role": "user", "content": "x"}], [])
    except ContextOverflow as e:
        raised = e
    assert isinstance(raised, ContextOverflow)
    assert raised.status_code == 413, raised.status_code


@check
def non_overflow_error_reraises_unchanged():
    # A rate-limit (NOT overflow) must propagate as-is so the normal is_retryable backoff applies.
    llm = _llm()
    llm.client = _PlainClient()
    raised = None
    try:
        llm.complete([{"role": "user", "content": "x"}], [])
    except ContextOverflow:
        raised = "overflow"
    except _PlainError:
        raised = "plain"
    assert raised == "plain", "non-overflow error must NOT be wrapped as ContextOverflow"


# ── prefix stability: cache kwargs do not perturb the create() payload for the default provider ──
@check
def default_provider_create_kwargs_have_no_cache_extra_body():
    # The moat: a non-Claude request stays byte-identical (no extra_body injected by _cache_kwargs).
    llm = _llm(model="gpt-5.5", base_url="https://api.openai.com/v1", reasoning="full")
    llm.complete([{"role": "user", "content": "x"}], [])
    captured = llm.client.chat.completions.captured_kwargs
    assert "extra_body" not in captured, captured
    assert captured["model"] == "gpt-5.5"


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
