"""Interactive STREAMING (live delta events): when a delta sink is wired, OpenAILLM streams
the completion and emits content/reasoning deltas LIVE, while assembling the SAME AssistantMessage the
blocking path returns (content, tool-calls parsed, usage incl. cached). No sink → blocking path unchanged
(eval byte-identical). No network, no pytest. Run: PYTHONPATH=src python tests/test_llm_streaming.py
"""
import os
import sys
import threading
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.llm import OpenAILLM  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ── fakes mirroring the openai streaming-chunk shape _stream_assemble reads ───────────────────
def _delta(content=None, tool_calls=None, reasoning=None, finish=None):
    d = NS(content=content, tool_calls=tool_calls, reasoning_content=reasoning, reasoning=None)
    return NS(choices=[NS(delta=d, finish_reason=finish)], usage=None)

def _tcd(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))

def _usage_chunk(prompt, completion, cached=None):
    pd = NS(cached_tokens=cached) if cached is not None else None
    return NS(choices=[], usage=NS(prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=pd))


class _Completions:
    def __init__(self, chunks): self.chunks = chunks
    def create(self, **kw):
        assert kw.get("stream") is True, "streaming path must request stream=True"
        assert kw.get("stream_options", {}).get("include_usage") is True, "must request include_usage"
        return iter(self.chunks)

class _Client:
    def __init__(self, chunks): self.chat = NS(completions=_Completions(chunks))


def _stub(chunks, on_delta):
    obj = OpenAILLM.__new__(OpenAILLM)   # bypass __init__ (no network); set only what complete() touches
    obj._hard_timeout = 30
    obj.client = _Client(chunks)
    obj.model = "gpt-5.5"
    obj.max_tokens = 0
    obj._base_url = ""
    obj.reasoning = "full"
    obj._cache_key = None
    obj._on_delta = on_delta
    return obj


_CHUNKS = [
    _delta(reasoning="let me think"),                                  # reasoning delta (hidden channel)
    _delta(content="Hello "),
    _delta(content="world"),
    _delta(tool_calls=[_tcd(0, id="call_1", name="read_file", args='{"path":')]),  # tool-call split…
    _delta(tool_calls=[_tcd(0, args='"a.py"}')]),                                  # …across two deltas
    _delta(finish="tool_calls"),
    _usage_chunk(10, 5, cached=3),                                     # final include_usage chunk
]


@check
def stream_assembles_content_toolcalls_and_usage():
    seen = []
    llm = _stub(_CHUNKS, lambda kind, text: seen.append((kind, text)))
    msg = llm.complete([{"role": "user", "content": "hi"}], [])
    assert msg.content == "Hello world", repr(msg.content)
    assert len(msg.tool_calls) == 1 and msg.tool_calls[0].name == "read_file"
    assert msg.tool_calls[0].args == {"path": "a.py"}, msg.tool_calls[0].args      # fragments reassembled + parsed
    assert msg.tool_calls[0].id == "call_1"
    assert msg.finish_reason == "tool_calls"
    assert msg.usage["prompt_tokens"] == 10 and msg.usage["completion_tokens"] == 5
    assert msg.usage["cached_tokens"] == 3, msg.usage                              # cache read-back preserved


@check
def content_deltas_are_emitted_live_in_order():
    seen = []
    llm = _stub(_CHUNKS, lambda kind, text: seen.append((kind, text)))
    llm.complete([{"role": "user", "content": "hi"}], [])
    assert [t for k, t in seen if k == "content"] == ["Hello ", "world"], seen
    assert ("reasoning", "let me think") in seen, "reasoning deltas must also be emitted"


@check
def no_sink_uses_blocking_path():
    # _on_delta=None → must NOT hit the streaming client (which asserts stream=True). Use a blocking fake.
    resp = NS(choices=[NS(message=NS(content="hi there", tool_calls=[]), finish_reason="stop")],
              usage=NS(prompt_tokens=4, completion_tokens=2, prompt_tokens_details=None))
    class _BlockingCompletions:
        def create(self, **kw):
            assert "stream" not in kw, "blocking path must NOT request streaming"
            return resp
    llm = _stub([], on_delta=None)
    llm.client = NS(chat=NS(completions=_BlockingCompletions()))
    msg = llm.complete([{"role": "user", "content": "hi"}], [])
    assert msg.content == "hi there" and msg.usage["prompt_tokens"] == 4


@check
def off_main_thread_uses_blocking_path():
    # subagents share the parent llm (sink IS set) but run OFF the main thread via run_scheduled — they must
    # NOT stream (keeps the off-main watchdog deadline + no N-thread spinner race).
    resp = NS(choices=[NS(message=NS(content="child done", tool_calls=[]), finish_reason="stop")],
              usage=NS(prompt_tokens=3, completion_tokens=1, prompt_tokens_details=None))
    class _BlockingCompletions:
        def create(self, **kw):
            assert "stream" not in kw, "off-main run must NOT stream"
            return resp
    llm = _stub(_CHUNKS, on_delta=lambda k, t: None)        # sink wired
    llm.client = NS(chat=NS(completions=_BlockingCompletions()))
    box = {}
    def _run():
        try:
            box["msg"] = llm.complete([{"role": "user", "content": "x"}], [])
        except Exception as e:  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=_run); t.start(); t.join()
    assert "err" not in box, box.get("err")
    assert box["msg"].content == "child done"


@check
def render_error_never_breaks_the_call():
    # a throwing delta sink must not break assembly (the result still comes back).
    def _boom(kind, text):
        raise RuntimeError("render blew up")
    llm = _stub(_CHUNKS, _boom)
    msg = llm.complete([{"role": "user", "content": "hi"}], [])
    assert msg.content == "Hello world", "a render error must be swallowed; assembly continues"


@check
def reasoning_effort_with_tools_400_degrades_and_sticks():
    # gpt-5.5 rejects reasoning_effort + function tools on chat/completions (400). complete() must drop
    # reasoning_effort, retry once, remember (sticky), and never re-send it with tools.
    calls = []
    ok = NS(choices=[NS(message=NS(content="ok", tool_calls=[]), finish_reason="stop")],
            usage=NS(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None))
    class _C:
        def create(self, **kw):
            calls.append("reasoning_effort" in kw)
            if "reasoning_effort" in kw:
                raise RuntimeError("400 - Function tools with reasoning_effort are not supported ... use /v1/responses")
            return ok
    llm = _stub([], on_delta=None)
    llm.model = "gpt-5.5"; llm.reasoning = "fast"   # → _reasoning_kwargs emits reasoning_effort=low
    llm._drop_reasoning_effort = False
    llm.client = NS(chat=NS(completions=_C()))
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    msg = llm.complete([{"role": "user", "content": "x"}], tools)
    assert msg.content == "ok", msg.content
    assert calls == [True, False], f"first WITH reasoning_effort (400) then retry WITHOUT: {calls}"
    assert llm._drop_reasoning_effort is True, "must remember the incompatibility"
    calls.clear()
    llm.complete([{"role": "user", "content": "y"}], tools)
    assert calls == [False], f"sticky → reasoning_effort never sent again with tools: {calls}"


@check
def switch_mutates_model_and_reasoning_live():
    # the /model switch: mutate model + reasoning in place (the loop reuses this llm every turn) and reset
    # the effort+tools degrade memory, so a new model re-evaluates routing.
    llm = _stub([], on_delta=None)
    llm.model = "gpt-5.5"; llm.reasoning = "full"; llm._drop_reasoning_effort = True
    assert llm._effort() is None                       # full → chat path
    llm.switch(model="gpt-5", reasoning="max")
    assert llm.model == "gpt-5" and llm.reasoning == "max"
    assert llm._drop_reasoning_effort is False         # reset for the new model
    assert llm._effort() == "xhigh"                    # max → /v1/responses routing


@check
def explicit_effort_with_tools_routes_to_responses():
    # the FIX: when the SDK has /v1/responses, an explicit effort + tools goes through Responses (which
    # supports the pairing) instead of degrading. Verify it calls responses.create (not chat) with the
    # right reasoning/input/tools and parses the Response.
    seen = {"responses": 0, "chat": 0}
    fake = NS(output_text="hi", output=[], status="completed",
              usage=NS(input_tokens=5, output_tokens=2, input_tokens_details=NS(cached_tokens=0)))
    class _Resp:
        def create(self, **kw):
            seen["responses"] += 1
            assert kw.get("reasoning") == {"effort": "low"}, kw.get("reasoning")
            assert "input" in kw and "tools" in kw, list(kw)
            return fake
    class _Chat:
        def create(self, **kw):
            seen["chat"] += 1
            return NS(choices=[NS(message=NS(content="x", tool_calls=[]), finish_reason="stop")], usage=None)
    llm = _stub([], on_delta=None)
    llm.model = "gpt-5.5"; llm.reasoning = "fast"           # effort=low
    llm.client = NS(responses=_Resp(), chat=NS(completions=_Chat()))
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    msg = llm.complete([{"role": "user", "content": "x"}], tools)
    assert msg.content == "hi", msg.content
    assert seen == {"responses": 1, "chat": 0}, f"effort+tools must use /v1/responses, not chat: {seen}"


@check
def typed_usage_splits_cache_read_from_other():
    # TokenUsage: input split into other / cache-read / cache-creation, output kept.
    from memagent.llm import _usage_dict
    raw = NS(prompt_tokens=100, completion_tokens=20,
             prompt_tokens_details=NS(cached_tokens=30), cache_creation_input_tokens=10)
    u = _usage_dict(raw)
    assert u["output"] == 20 and u["input_cache_read"] == 30 and u["input_cache_creation"] == 10
    assert u["input_other"] == 60, u                     # 100 - 30 - 10
    assert u["prompt_tokens"] == 100 and u["completion_tokens"] == 20 and u["cached_tokens"] == 30  # back-compat
    # Moonshot reports cached_tokens at the TOP level (no prompt_tokens_details)
    u2 = _usage_dict(NS(prompt_tokens=50, completion_tokens=5, cached_tokens=12))
    assert u2["input_cache_read"] == 12 and u2["input_other"] == 38, u2
    # provider that omits cache counters → zeros, no crash, no spurious cached_tokens key
    u3 = _usage_dict(NS(prompt_tokens=8, completion_tokens=2, prompt_tokens_details=None))
    assert u3["input_cache_read"] == 0 and u3["input_other"] == 8 and "cached_tokens" not in u3
    assert _usage_dict(None) is None


@check
def empty_completion_raises_retryable():
    from memagent.errors import EmptyResponseError, classify
    resp = NS(choices=[NS(message=NS(content=None, tool_calls=[]), finish_reason="stop")],
              usage=NS(prompt_tokens=5, completion_tokens=0, prompt_tokens_details=None))
    class _Blocking:
        def create(self, **kw):
            return resp
    llm = _stub([], on_delta=None)
    llm.client = NS(chat=NS(completions=_Blocking()))
    raised = None
    try:
        llm.complete([{"role": "user", "content": "x"}], [])
    except EmptyResponseError as e:
        raised = e
    assert raised is not None, "empty content + no tool calls must raise EmptyResponseError"
    assert llm.is_retryable(raised) is True, "empty response must be retryable"
    c = classify(raised)
    assert c["retryable"] is True and c["kind"] == "empty_response", c
    # content_filter is NOT re-rolled (would just filter again) → no raise
    resp.choices[0].finish_reason = "content_filter"
    llm.complete([{"role": "user", "content": "x"}], [])  # must not raise


@check
def classify_buckets_failures_for_telemetry():
    from memagent.errors import classify
    assert classify(RuntimeError("429 rate limit exceeded"))["kind"] == "rate_limit"
    assert classify(RuntimeError("connection error: econnreset"))["kind"] == "connection"
    e = RuntimeError("server blew up"); e.status_code = 503
    assert classify(e)["kind"] == "server" and classify(e)["retryable"] is True
    a = RuntimeError("forbidden"); a.status_code = 403
    assert classify(a)["kind"] == "auth" and classify(a)["retryable"] is False


@check
def reasoning_intent_maps_to_effort():  # #51
    llm = _stub([], on_delta=None); llm.model = "gpt-5.5"; llm._base_url = ""
    for intent, expect in [("fast", {"reasoning_effort": "low"}), ("full", {}),
                           ("high", {"reasoning_effort": "high"}), ("max", {"reasoning_effort": "xhigh"})]:
        llm.reasoning = intent
        assert llm._reasoning_kwargs() == expect, (intent, llm._reasoning_kwargs())
    # a non-reasoning provider ignores it entirely
    llm.model = "kimi-k2.7-code"; llm._base_url = "https://api.moonshot.cn/v1"; llm.reasoning = "high"
    assert llm._reasoning_kwargs() == {}, llm._reasoning_kwargs()


@check
def watchdog_is_daemon_and_times_out():  # #47
    import time
    llm = _stub([], on_delta=None); llm._hard_timeout = 1; llm._base_url = ""
    from memagent.llm import _import_api_timeout_error
    APITimeoutError = _import_api_timeout_error()

    class _Slow:
        def create(self, **kw):
            time.sleep(6); return "never"
    llm.client = NS(chat=NS(completions=_Slow()))
    t0 = time.monotonic()
    raised = False
    try:
        llm._create_watchdog({})
    except APITimeoutError:
        raised = True
    assert raised and (time.monotonic() - t0) < 3.5, "must abort near the 1s deadline, not wait for the slow call"

    class _Ok:
        def create(self, **kw):
            return "ok"
    llm.client = NS(chat=NS(completions=_Ok()))
    assert llm._create_watchdog({}) == "ok"   # fast call returns normally

    class _Boom:
        def create(self, **kw):
            raise RuntimeError("provider 500")
    llm.client = NS(chat=NS(completions=_Boom()))
    try:
        llm._create_watchdog({}); assert False, "error must propagate"
    except RuntimeError:
        pass


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
