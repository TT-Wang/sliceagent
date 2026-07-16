"""Unified streaming transport: every OpenAILLM call streams independently of UI/thread state, while
assembling the SAME AssistantMessage contract (content, tool calls, usage). The async bridge is cancellable
while blocked on a read and confirms physical closure before retry can proceed. No network, no pytest. Run:
  PYTHONPATH=src python tests/test_llm_streaming.py
"""
import os
import sys
import threading
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.llm import OpenAILLM  # noqa: E402

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


class _FakeHub:
    """Deterministic transport seam: production uses _AsyncTransportHub; parser tests stay network-free."""
    def __init__(self, chunks=(), *, response=None, handler=None):
        self.chunks = list(chunks)
        self.response = response
        self.handler = handler
        self.calls = []

    def run(self, kind, spec, kwargs, **control):
        self.calls.append((kind, dict(kwargs)))
        if self.handler is not None:
            return self.handler(kind, kwargs, control)
        if kind == "chat":
            assert kwargs.get("stream") is True
            assert kwargs.get("stream_options", {}).get("include_usage") is True
        for item in self.chunks:
            control["on_item"](item)
        return self.response


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
    obj._stream_transport_enabled = True
    obj._transport_activity = None
    obj._transport_spec = ("test", "", "none", 60.0)
    obj._transport_hub = _FakeHub(chunks)
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
def no_sink_still_uses_streaming_transport():
    llm = _stub(_CHUNKS, on_delta=None)
    msg = llm.complete([{"role": "user", "content": "hi"}], [])
    assert msg.content == "Hello world" and msg.usage["prompt_tokens"] == 10
    assert llm._transport_hub.calls[0][0] == "chat"


@check
def off_main_thread_uses_streaming_transport_without_a_sink():
    # This is the child path: no delta renderer and a worker thread. It must still request SSE.
    llm = _stub(_CHUNKS, on_delta=None)
    box = {}
    def _run():
        try:
            box["msg"] = llm.complete([{"role": "user", "content": "x"}], [])
        except Exception as e:  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=_run); t.start(); t.join()
    assert "err" not in box, box.get("err")
    assert box["msg"].content == "Hello world"
    assert llm._transport_hub.calls[0][1]["stream"] is True


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
    # gpt-5.5 rejects reasoning_effort + function tools on chat/completions (400). The adapter remembers the
    # downgrade, but the app-owned seam must issue the replacement request as a separately visible attempt.
    from sliceagent.model_runner import complete_model_call

    from sliceagent.events import ApiRetry

    calls, attempts, events = [], [], []
    def transport(_kind, kw, control):
        calls.append("reasoning_effort" in kw)
        if "reasoning_effort" in kw:
            raise RuntimeError("400 - Function tools with reasoning_effort are not supported ... use /v1/responses")
        control["on_item"](_delta(content="ok"))
        control["on_item"](_delta(finish="stop"))
        control["on_item"](_usage_chunk(1, 1))
    llm = _stub([], on_delta=None)
    llm.model = "gpt-5.5"; llm.reasoning = "fast"   # → _reasoning_kwargs emits reasoning_effort=low
    llm._drop_reasoning_effort = False
    llm._transport_hub = _FakeHub(handler=transport)
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    msg = complete_model_call(
        llm, [{"role": "user", "content": "x"}], tools,
        dispatch=events.append,
        on_attempt=lambda attempt, _messages, _report: attempts.append(attempt),
    )
    assert msg.content == "ok", msg.content
    assert calls == [True, False], f"first WITH reasoning_effort (400) then retry WITHOUT: {calls}"
    assert attempts == [1, 2], "each HTTP request must cross the app-owned attempt observer"
    retries = [event for event in events if isinstance(event, ApiRetry)]
    assert len(retries) == 1 and retries[0].delay_s == 0.0
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
    def responses_transport(kind, kw, _control):
        assert kind == "responses"
        seen["responses"] += 1
        assert kw.get("reasoning") == {"effort": "low"}, kw.get("reasoning")
        assert "input" in kw and "tools" in kw, list(kw)
        return fake
    llm._transport_hub = _FakeHub(handler=responses_transport)
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    msg = llm.complete([{"role": "user", "content": "x"}], tools)
    assert msg.content == "hi", msg.content
    assert seen == {"responses": 1, "chat": 0}, f"effort+tools must use /v1/responses, not chat: {seen}"


@check
def typed_usage_splits_cache_read_from_other():
    # TokenUsage: input split into other / cache-read / cache-creation, output kept.
    from sliceagent.llm import _usage_dict
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
    from sliceagent.errors import EmptyResponseError, classify
    llm = _stub([_delta(finish="stop"), _usage_chunk(5, 0)], on_delta=None)
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
    llm._transport_hub = _FakeHub([_delta(finish="content_filter"), _usage_chunk(5, 0)])
    llm.complete([{"role": "user", "content": "x"}], [])  # must not raise


@check
def classify_buckets_failures_for_telemetry():
    from sliceagent.errors import classify
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
    from sliceagent.errors import IndeterminateModelCallError

    class _Slow:
        def create(self, **kw):
            time.sleep(6); return "never"
    llm.client = NS(chat=NS(completions=_Slow()))
    t0 = time.monotonic()
    raised = False
    try:
        llm._create_watchdog({})
    except IndeterminateModelCallError:
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


@check
def transport_activity_is_private_low_rate_and_fail_open():
    events = []
    llm = _stub(_CHUNKS, on_delta=None)
    llm.set_transport_activity(lambda event, detail: events.append((event, detail["transport"])))
    msg = llm.complete([{"role": "user", "content": "x"}], [])
    assert msg.content == "Hello world"
    names = [event for event, _ in events]
    assert names == ["awaiting_model", "first_byte", "reasoning", "writing", "finished"], names
    assert all(transport == "sse" for _, transport in events)
    # An observer is diagnostic only; it may never turn a healthy call into a model error.
    llm.set_transport_activity(lambda *_args: (_ for _ in ()).throw(RuntimeError("observer broke")))
    assert llm.complete([{"role": "user", "content": "y"}], []).content == "Hello world"


@check
def complete_with_control_forwards_cancellation_without_breaking_two_arg_protocol():
    from sliceagent.errors import RetryCancelledError

    seen = []
    llm = _stub(_CHUNKS, on_delta=None)
    def handler(_kind, _kwargs, control):
        seen.append(control["should_cancel"])
        if control["should_cancel"]():
            raise RetryCancelledError("cancelled")
    llm._transport_hub = _FakeHub(handler=handler)
    try:
        llm.complete_with_control(
            [{"role": "user", "content": "x"}], [], should_cancel=lambda: True,
        )
        assert False, "cancel signal must reach the physical transport"
    except RetryCancelledError:
        pass
    assert len(seen) == 1
    # The ordinary protocol remains exactly two arguments for fake/custom LLM compatibility.
    healthy = _stub(_CHUNKS, on_delta=None)
    assert healthy.complete([{"role": "user", "content": "x"}], []).content == "Hello world"


@check
def reasoning_only_broken_stream_is_not_blindly_replayed():
    from sliceagent.errors import IndeterminateModelCallError

    llm = _stub([], on_delta=None)
    def partial(_kind, _kwargs, control):
        control["on_item"](_delta(reasoning="a long hidden derivation"))
        raise ConnectionError("stream reset before answer")
    llm._transport_hub = _FakeHub(handler=partial)
    try:
        llm.complete([{"role": "user", "content": "x"}], [])
        assert False, "semantic generation without a seal must suppress blind automatic replay"
    except IndeterminateModelCallError as error:
        assert "semantic output began" in str(error)


@check
def silent_first_byte_timeout_is_replayed_once_not_three_times():
    from sliceagent.errors import PreFirstByteTimeoutError
    from sliceagent.events import ApiRetry
    from sliceagent.llm import _AsyncTransportHub
    from sliceagent.model_runner import complete_model_call

    calls, attempts, retries = {"n": 0}, [], []

    def silent(_kind, _kwargs, _control):
        calls["n"] += 1
        raise _AsyncTransportHub._timeout_error(("x", "http://local", "none", 60))

    llm = _stub([], on_delta=None)
    llm._transport_hub = _FakeHub(handler=silent)
    try:
        complete_model_call(
            llm, [{"role": "user", "content": "x"}], [],
            dispatch=retries.append,
            on_attempt=lambda attempt, _messages, _report: attempts.append(attempt),
        )
        assert False, "a stream with no first byte must surface after one bounded replay"
    except PreFirstByteTimeoutError:
        pass
    assert calls["n"] == 2 and attempts == [1, 2]
    retry_events = [event for event in retries if isinstance(event, ApiRetry)]
    assert len(retry_events) == 1 and retry_events[0].max_attempts == 2


@check
def semantic_bytes_then_timeout_are_never_replayed():
    import httpx

    from sliceagent.errors import IndeterminateModelCallError
    from sliceagent.model_runner import complete_model_call

    calls, attempts = {"n": 0}, []

    def timed_out_reasoning(_kind, _kwargs, control):
        calls["n"] += 1
        control["on_item"](_delta(reasoning="the provider started expensive reasoning"))
        # Async SSE iteration may expose raw httpx.ReadTimeout rather than SDK APITimeoutError.
        raise httpx.ReadTimeout("idle SSE body", request=httpx.Request("POST", "http://local"))

    llm = _stub([], on_delta=None)
    llm._transport_hub = _FakeHub(handler=timed_out_reasoning)
    try:
        complete_model_call(
            llm, [{"role": "user", "content": "x"}], [],
            on_attempt=lambda attempt, _messages, _report: attempts.append(attempt),
        )
        assert False, "observed reasoning must make timeout replay unsafe"
    except IndeterminateModelCallError as error:
        assert "timed out after semantic output" in str(error)
    assert calls["n"] == 1 and attempts == [1]


@check
def responses_stream_failure_never_hides_a_blocking_second_request():
    calls = []
    llm = _stub([], on_delta=None)
    llm.client = NS(responses=object())
    def fail(kind, _kwargs, _control):
        calls.append(kind)
        raise RuntimeError("stream transport failed")
    llm._transport_hub = _FakeHub(handler=fail)
    try:
        llm._responses_stream({"model": "x"})
        assert False, "the original stream error must return to the app retry seam"
    except RuntimeError as error:
        assert "stream transport failed" in str(error)
    assert calls == ["responses"], "one visible attempt must be one physical request"


@check
def responses_semantic_partial_suppresses_blind_replay():
    from sliceagent.errors import IndeterminateModelCallError

    llm = _stub([], on_delta=None)
    llm.client = NS(responses=object())
    def partial(_kind, _kwargs, control):
        control["on_item"](NS(type="response.reasoning_text.delta", delta="worked for a while"))
        raise ConnectionError("SSE reset")
    llm._transport_hub = _FakeHub(handler=partial)
    try:
        llm._responses_stream({"model": "x"})
        assert False, "semantic Responses output must not trigger an automatic full replay"
    except IndeterminateModelCallError as error:
        assert "semantic output began" in str(error)


@check
def responses_function_call_events_are_semantic_and_suppress_blind_replay():
    from sliceagent.errors import IndeterminateModelCallError

    events = [
        NS(type="response.function_call_arguments.delta", delta='{"path":'),
        NS(type="response.function_call_arguments.done", arguments='{"path":"x.py"}', name="read_file"),
        NS(type="response.output_item.added", item=NS(type="function_call")),
        NS(type="response.output_item.done", item=NS(type="function_call")),
    ]
    for semantic_event in events:
        activity = []
        llm = _stub([], on_delta=None)
        llm.client = NS(responses=object())
        def partial(_kind, _kwargs, control, event=semantic_event):
            control["on_item"](event)
            raise ConnectionError("SSE reset after tool generation")
        llm._transport_hub = _FakeHub(handler=partial)
        try:
            llm._responses_stream(
                {"model": "x"}, activity=lambda event, _detail: activity.append(event),
            )
            assert False, f"{semantic_event.type} must suppress full request replay"
        except IndeterminateModelCallError as error:
            assert "semantic output began" in str(error)
        assert activity == ["awaiting_model", "first_byte", "writing", "failed"], (
            semantic_event.type, activity,
        )


@check
def responses_non_function_output_item_remains_presemantic():
    llm = _stub([], on_delta=None)
    llm.client = NS(responses=object())
    def partial(_kind, _kwargs, control):
        control["on_item"](NS(type="response.output_item.added", item=NS(type="message")))
        raise ConnectionError("SSE reset before semantic bytes")
    llm._transport_hub = _FakeHub(handler=partial)
    try:
        llm._responses_stream({"model": "x"})
        assert False, "raw transport failure must propagate to the app retry seam"
    except ConnectionError as error:
        assert "before semantic" in str(error)


@check
def responses_function_call_success_keeps_normal_tool_and_usage_assembly():
    activity = []
    final = NS(
        output_text="",
        output=[NS(
            type="function_call", name="read_file", arguments='{"path":"x.py"}',
            call_id="call-1", id="item-1",
        )],
        status="completed",
        usage=NS(input_tokens=11, output_tokens=4, input_tokens_details=NS(cached_tokens=3)),
    )
    def success(_kind, _kwargs, control):
        control["on_item"](NS(
            type="response.output_item.added", item=NS(type="function_call"),
        ))
        control["on_item"](NS(
            type="response.function_call_arguments.delta", delta='{"path":"x.py"}',
        ))
        control["on_item"](NS(
            type="response.function_call_arguments.done", arguments='{"path":"x.py"}', name="read_file",
        ))
        control["on_item"](NS(
            type="response.output_item.done", item=NS(type="function_call"),
        ))
        return final

    llm = _stub([], on_delta=None)
    llm.model = "gpt-5.5"; llm.reasoning = "fast"
    llm.client = NS(responses=object(), chat=NS(completions=object()))
    llm._transport_hub = _FakeHub(handler=success)
    message = llm.complete_with_control(
        [{"role": "user", "content": "inspect"}], [],
        transport_activity=lambda event, _detail: activity.append(event),
    )
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].name == "read_file" and message.tool_calls[0].args == {"path": "x.py"}
    assert message.usage["prompt_tokens"] == 11 and message.usage["cached_tokens"] == 3
    assert activity == ["awaiting_model", "first_byte", "writing", "finished"], activity


@check
def async_bridge_reports_capacity_admission_first_byte_and_low_rate_heartbeat():
    import asyncio
    import time

    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    awaiting_heartbeat = threading.Event()
    receiving_heartbeat = threading.Event()

    class TimelineHub(_AsyncTransportHub):
        async def _chat(self, _spec, _kwargs, publish):
            # Synchronize on the bridge's observations instead of assuming its caller thread will be
            # scheduled inside a sub-100ms window on a busy CI host.
            for _ in range(200):
                if awaiting_heartbeat.is_set():
                    break
                await asyncio.sleep(0.005)
            assert awaiting_heartbeat.is_set(), "bridge never reported its pre-byte heartbeat"
            publish(object())
            for _ in range(200):
                if receiving_heartbeat.is_set():
                    break
                await asyncio.sleep(0.005)
            assert receiving_heartbeat.is_set(), "bridge never reported its receiving heartbeat"

    hub, gate = TimelineHub(), _PhysicalCallGate(1)
    occupied = gate.acquire(timeout=1)
    events, items, box = [], [], {}

    def observe(event, detail):
        events.append((event, dict(detail)))
        if event == "stream_heartbeat":
            (receiving_heartbeat if detail["state"] == "receiving" else awaiting_heartbeat).set()
            raise RuntimeError("diagnostic observer failure must stay fail-open")

    def call():
        try:
            box["result"] = hub.run(
                "chat", ("x", "", "none", 60), {}, timeout=2, close_grace=0.2,
                provider_gate=gate, on_item=items.append, on_activity=observe,
                heartbeat_interval=0.04,
            )
        except Exception as error:  # noqa: BLE001
            box["error"] = error

    thread = threading.Thread(target=call); thread.start()
    deadline = time.monotonic() + 1
    while not events and time.monotonic() < deadline:
        time.sleep(0.005)
    assert events and events[0][0] == "provider_queue", events
    occupied.release()
    thread.join(2)
    assert not thread.is_alive() and "error" not in box, box
    names = [name for name, _detail in events]
    assert names[0:2] == ["provider_queue", "provider_admitted"], names
    assert names.count("first_byte") == 1 and len(items) == 1
    first = next(detail for name, detail in events if name == "first_byte")
    admitted = next(detail for name, detail in events if name == "provider_admitted")
    assert admitted["queued"] is True
    assert first["ttfb_ms"] > 0 and first["elapsed_ms"] >= admitted["queue_ms"]
    heartbeats = [detail for name, detail in events if name == "stream_heartbeat"]
    assert {detail["state"] for detail in heartbeats} == {"awaiting_first_byte", "receiving"}, heartbeats
    assert all(set(detail) >= {"state", "elapsed_ms", "idle_ms", "chunks"} for detail in heartbeats)


@check
def openai_adapter_forwards_hub_timing_without_duplicate_first_byte():
    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    class OneResponseHub(_AsyncTransportHub):
        async def _chat(self, _spec, _kwargs, publish):
            publish(_delta(content="ok"))
            publish(_delta(finish="stop"))

    llm = _stub([], on_delta=None)
    llm._transport_hub = OneResponseHub()
    llm._provider_call_gate = _PhysicalCallGate(1)
    events = []
    message = llm.complete_with_control(
        [{"role": "user", "content": "x"}], [],
        transport_activity=lambda event, detail: events.append((event, dict(detail))),
    )
    names = [name for name, _detail in events]
    assert message.content == "ok"
    assert names == ["awaiting_model", "provider_admitted", "first_byte", "writing", "finished"], names
    first = next(detail for name, detail in events if name == "first_byte")
    admitted = next(detail for name, detail in events if name == "provider_admitted")
    assert first["transport"] == admitted["transport"] == "sse"
    assert set(first) >= {"queue_ms", "ttfb_ms", "elapsed_ms"}


@check
def async_bridge_deadline_interrupts_blocked_read_and_confirms_closure():
    import asyncio

    from sliceagent.llm import _AsyncTransportHub, _import_api_timeout_error

    class BlockedStream:
        def __init__(self):
            self.closed = threading.Event()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            self.closed.set()
        def __aiter__(self):
            return self
        async def __anext__(self):
            await asyncio.Event().wait()  # no chunk can wake a caller-side monotonic check

    stream = BlockedStream()
    class Completions:
        async def create(self, **_kwargs):
            return stream
    hub, spec = _AsyncTransportHub(), ("x", "", "none", 60)
    hub._ensure_started()
    hub._clients[spec] = NS(chat=NS(completions=Completions()))
    t0 = __import__("time").monotonic()
    try:
        hub.run("chat", spec, {}, timeout=0.12, close_grace=0.5)
        assert False, "absolute deadline must interrupt the blocked async read"
    except _import_api_timeout_error():
        pass
    assert stream.closed.is_set(), "retry lease releases only after the stream context closes"
    assert __import__("time").monotonic() - t0 < 1.0


@check
def async_bridge_owner_cancel_interrupts_blocked_read_without_overlap():
    import asyncio

    from sliceagent.errors import RetryCancelledError
    from sliceagent.llm import _AsyncTransportHub

    class Blocked(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.entered = threading.Event(); self.closed = threading.Event()
        async def _chat(self, _spec, _kwargs, _publish):
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.closed.set()

    hub, cancel, box = Blocked(), threading.Event(), {}
    def call():
        try:
            hub.run(
                "chat", ("x", "", "none", 60), {}, timeout=30, close_grace=0.5,
                should_cancel=cancel.is_set,
            )
        except Exception as error:  # noqa: BLE001
            box["error"] = error
    thread = threading.Thread(target=call); thread.start()
    assert hub.entered.wait(1)
    cancel.set(); thread.join(2)
    assert not thread.is_alive() and isinstance(box.get("error"), RetryCancelledError), box
    assert hub.closed.wait(0.2), "owner cancellation must close before returning"


@check
def async_bridge_owner_cancel_wins_before_admission_on_a_blocked_loop():
    import time

    from sliceagent.errors import RetryCancelledError
    from sliceagent.llm import _AsyncTransportHub

    class AuditHub(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.provider_starts = 0
        async def _chat(self, _spec, _kwargs, _publish):
            self.provider_starts += 1

    hub = AuditHub(); hub._ensure_started()
    loop_blocked, release_loop = threading.Event(), threading.Event()

    def block_transport_loop():
        loop_blocked.set()
        release_loop.wait(2)

    hub._loop.call_soon_threadsafe(block_transport_loop)
    assert loop_blocked.wait(1)
    cancel, box = threading.Event(), {}

    def call():
        try:
            hub.run(
                "chat", ("x", "", "none", 60), {}, timeout=30, close_grace=0.1,
                should_cancel=cancel.is_set,
            )
        except Exception as error:  # noqa: BLE001
            box["error"] = error

    caller = threading.Thread(target=call); caller.start()
    time.sleep(0.05)  # drive() is queued behind the deterministic loop blocker
    cancel.set(); caller.join(1)
    try:
        assert not caller.is_alive() and isinstance(box.get("error"), RetryCancelledError), box
        assert hub.provider_starts == 0
    finally:
        release_loop.set()
    time.sleep(0.1)
    assert hub.provider_starts == 0, "cancelled pending coroutine opened a provider call after return"


@check
def async_bridge_fallback_timeout_wins_before_admission_on_a_blocked_loop():
    import time

    from sliceagent.errors import TransportStartupError
    from sliceagent.llm import _AsyncTransportHub

    class AuditHub(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.provider_starts = 0
        async def _chat(self, _spec, _kwargs, _publish):
            self.provider_starts += 1

    hub = AuditHub(); hub._ensure_started()
    loop_blocked, release_loop = threading.Event(), threading.Event()

    def block_transport_loop():
        loop_blocked.set()
        release_loop.wait(2)

    hub._loop.call_soon_threadsafe(block_transport_loop)
    assert loop_blocked.wait(1)
    box = {}

    def call():
        try:
            hub.run("chat", ("x", "", "none", 60), {}, timeout=0.05, close_grace=0.05)
        except Exception as error:  # noqa: BLE001
            box["error"] = error

    caller = threading.Thread(target=call); caller.start(); caller.join(1)
    try:
        assert not caller.is_alive() and isinstance(box.get("error"), TransportStartupError), box
        assert hub.provider_starts == 0
    finally:
        release_loop.set()
    time.sleep(0.1)
    assert hub.provider_starts == 0, "timed-out pending coroutine opened a provider call after return"


@check
def externally_cancelled_pending_future_retires_admission_and_releases_capacity():
    import asyncio
    import time

    from sliceagent.errors import RetryCancelledError
    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    class AuditHub(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.provider_starts = 0
        async def _chat(self, _spec, _kwargs, _publish):
            self.provider_starts += 1

    hub = AuditHub(); hub._ensure_started()
    loop_blocked, release_loop = threading.Event(), threading.Event()
    hub._loop.call_soon_threadsafe(lambda: (loop_blocked.set(), release_loop.wait(2)))
    assert loop_blocked.wait(1)
    gate, submitted, box = _PhysicalCallGate(1), threading.Event(), {}
    original_submit = asyncio.run_coroutine_threadsafe

    def capture_submit(coroutine, loop):
        future = original_submit(coroutine, loop)
        box["future"] = future
        submitted.set()
        return future

    asyncio.run_coroutine_threadsafe = capture_submit

    def call():
        try:
            hub.run(
                "chat", ("x", "", "none", 60), {}, timeout=30, close_grace=0.1,
                provider_gate=gate,
            )
        except Exception as error:  # noqa: BLE001
            box["error"] = error

    caller = threading.Thread(target=call); caller.start()
    try:
        assert submitted.wait(1) and gate.active == 1
        box["future"].cancel(); caller.join(1)
        assert not caller.is_alive() and isinstance(box.get("error"), RetryCancelledError), box
        assert gate.active == 0
    finally:
        asyncio.run_coroutine_threadsafe = original_submit
        release_loop.set()
    time.sleep(0.1)
    assert hub.provider_starts == 0, "externally-cancelled future started after its caller returned"


@check
def async_bridge_consumer_failure_cancels_and_closes_the_producer():
    import asyncio

    from sliceagent.llm import _AsyncTransportHub

    class OneThenBlock:
        def __init__(self):
            self.sent = False; self.closed = threading.Event()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            self.closed.set()
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.sent:
                self.sent = True
                return object()
            await asyncio.Event().wait()

    stream = OneThenBlock()
    class Completions:
        async def create(self, **_kwargs):
            return stream
    hub, spec = _AsyncTransportHub(), ("x", "", "none", 60)
    hub._ensure_started()
    hub._clients[spec] = NS(chat=NS(completions=Completions()))
    try:
        hub.run(
            "chat", spec, {}, timeout=30, close_grace=0.5,
            on_item=lambda _item: (_ for _ in ()).throw(RuntimeError("parser stopped")),
        )
        assert False, "consumer failure must propagate after producer cleanup"
    except RuntimeError as error:
        assert "parser stopped" in str(error)
    assert stream.closed.wait(0.2), "consumer failure must not orphan the provider stream"


@check
def consumer_indeterminate_marker_still_cancels_and_closes_the_producer():
    import asyncio

    from sliceagent.errors import IndeterminateModelCallError
    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    class Producer(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.closed = threading.Event()
        async def _chat(self, _spec, _kwargs, publish):
            try:
                publish(object())
                await asyncio.Event().wait()
            finally:
                self.closed.set()

    hub, gate = Producer(), _PhysicalCallGate(1)
    try:
        hub.run(
            "chat", ("x", "", "none", 60), {}, timeout=30, close_grace=0.5,
            provider_gate=gate,
            on_item=lambda _item: (_ for _ in ()).throw(
                IndeterminateModelCallError("consumer rejected assembled state")
            ),
        )
        assert False, "consumer marker must propagate only after producer cleanup"
    except IndeterminateModelCallError as error:
        assert "consumer rejected" in str(error)
    assert hub.closed.wait(0.2) and gate.active == 0


@check
def stream_close_failure_is_indeterminate_and_retains_physical_capacity():
    import asyncio

    from sliceagent.errors import IndeterminateModelCallError
    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    class CloseFails:
        def __init__(self):
            self.entered = threading.Event()
            self.close_attempted = threading.Event()
            self.closed = False
        async def __aenter__(self):
            self.entered.set()
            return self
        async def __aexit__(self, *_args):
            self.close_attempted.set()
            raise ConnectionError("close failed before socket retirement")
        def __aiter__(self):
            return self
        async def __anext__(self):
            await asyncio.Event().wait()

    stream = CloseFails()
    class Completions:
        async def create(self, **_kwargs):
            return stream

    hub, spec, gate = _AsyncTransportHub(), ("x", "", "none", 60), _PhysicalCallGate(1)
    hub._ensure_started()
    hub._clients[spec] = NS(chat=NS(completions=Completions()))
    cancel, box = threading.Event(), {}

    def call():
        try:
            hub.run(
                "chat", spec, {}, timeout=30, close_grace=0.05,
                should_cancel=cancel.is_set, provider_gate=gate,
            )
        except Exception as error:  # noqa: BLE001
            box["error"] = error

    caller = threading.Thread(target=call); caller.start()
    assert stream.entered.wait(1)
    cancel.set(); caller.join(2)
    assert not caller.is_alive() and isinstance(box.get("error"), IndeterminateModelCallError), box
    assert stream.close_attempted.is_set() and not stream.closed
    assert gate.active == 1, "unconfirmed teardown must quarantine its physical provider lease"


@check
def provider_lease_retirement_is_retry_safe_on_both_sides_of_interrupt():
    from sliceagent.llm import _PhysicalCallGate

    for interrupt_after_commit in (False, True):
        gate = _PhysicalCallGate(1)
        lease = gate.acquire(timeout=1)
        original = gate._release
        interrupted = {"done": False}

        def once(value):
            if interrupted["done"]:
                return original(value)
            interrupted["done"] = True
            if interrupt_after_commit:
                original(value)
            raise KeyboardInterrupt("release handoff interrupted")

        gate._release = once
        try:
            lease.release()
            assert False, "the injected handoff interruption must surface"
        except KeyboardInterrupt:
            pass
        finally:
            gate._release = original
        lease.release()
        assert gate.active == 0, (interrupt_after_commit, gate.active)


@check
def provider_gate_rolls_back_interrupt_after_active_set_insert():
    from sliceagent.llm import _PhysicalCallGate

    class InterruptingSet(set):
        def add(self, value):
            super().add(value)
            raise KeyboardInterrupt("interrupted immediately after active-set insertion")

    gate = _PhysicalCallGate(1)
    gate._leases = InterruptingSet()
    lease = gate.new_lease()
    try:
        gate.acquire_lease(lease, timeout=1)
        assert False, "the injected admission interruption must surface"
    except KeyboardInterrupt:
        pass
    assert gate.active == 0


@check
def async_run_preowns_lease_across_acquire_return_handoff():
    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    hub, gate = _AsyncTransportHub(), _PhysicalCallGate(1)
    original = gate.acquire_lease

    def interrupt_after_return(lease, **kwargs):
        original(lease, **kwargs)
        raise KeyboardInterrupt("caller assignment handoff interrupted")

    gate.acquire_lease = interrupt_after_return
    try:
        try:
            hub.run(
                "chat", ("x", "", "none", 60), {}, timeout=1, close_grace=0.05,
                provider_gate=gate,
            )
            assert False, "the injected handoff interruption must surface"
        except KeyboardInterrupt:
            pass
        assert gate.active == 0
    finally:
        gate.acquire_lease = original


@check
def coroutine_submission_interrupt_retires_pre_admission_capacity():
    import asyncio

    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    hub, gate = _AsyncTransportHub(), _PhysicalCallGate(1)
    hub._ensure_started()
    original = asyncio.run_coroutine_threadsafe

    def interrupted(coroutine, _loop):
        coroutine.close()
        raise KeyboardInterrupt("submit handoff interrupted")

    asyncio.run_coroutine_threadsafe = interrupted
    try:
        try:
            hub.run(
                "chat", ("x", "", "none", 60), {}, timeout=1, close_grace=0.05,
                provider_gate=gate,
            )
            assert False, "submission interruption must surface"
        except KeyboardInterrupt:
            pass
        assert gate.active == 0
    finally:
        asyncio.run_coroutine_threadsafe = original


@check
def submitted_coroutine_waits_for_future_ownership_before_provider_admission():
    import asyncio
    import time

    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    class AuditHub(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.provider_starts = 0
        async def _chat(self, _spec, _kwargs, _publish):
            self.provider_starts += 1

    hub, gate = AuditHub(), _PhysicalCallGate(1)
    hub._ensure_started()
    original = asyncio.run_coroutine_threadsafe

    def schedule_then_interrupt(coroutine, loop):
        original(coroutine, loop)
        time.sleep(0.05)  # give an incorrectly unguarded coroutine ample time to enter _chat
        assert hub.provider_starts == 0
        raise KeyboardInterrupt("future return-to-assignment handoff interrupted")

    asyncio.run_coroutine_threadsafe = schedule_then_interrupt
    try:
        try:
            hub.run(
                "chat", ("x", "", "none", 60), {}, timeout=1, close_grace=0.1,
                provider_gate=gate,
            )
            assert False, "the injected Future handoff interruption must surface"
        except KeyboardInterrupt:
            pass
        time.sleep(0.05)
        assert hub.provider_starts == 0 and gate.active == 0
    finally:
        asyncio.run_coroutine_threadsafe = original


@check
def occupied_provider_capacity_fails_once_without_opening_a_request():
    from sliceagent.errors import ProviderCapacityError, with_retry
    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    class AuditHub(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.provider_starts = 0
        async def _chat(self, _spec, _kwargs, _publish):
            self.provider_starts += 1

    hub, gate = AuditHub(), _PhysicalCallGate(1)
    occupied = gate.acquire(timeout=1)
    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        return hub.run(
            "chat", ("x", "", "none", 60), {}, timeout=0.05, close_grace=0.05,
            provider_gate=gate,
        )

    try:
        try:
            with_retry(attempt, max_attempts=3, is_retryable=lambda _error: True)
            assert False, "capacity exhaustion must surface"
        except ProviderCapacityError:
            pass
        assert calls["n"] == 1 and hub.provider_starts == 0
        assert gate.active == 1
    finally:
        occupied.release()


@check
def async_hub_startup_wait_obeys_deadline_and_owner_cancellation():
    import time

    from sliceagent.errors import RetryCancelledError, TransportStartupError
    from sliceagent.llm import _AsyncTransportHub

    class StuckStart(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.entered = threading.Event(); self.release = threading.Event()
        def _new_event_loop(self):
            self.entered.set()
            assert self.release.wait(2), "test failed to retire stuck startup"
            return super()._new_event_loop()

    timed = StuckStart()
    started = time.monotonic()
    try:
        timed._ensure_started(timeout=0.05)
        assert False, "stuck loop construction must not freeze beyond the call deadline"
    except TransportStartupError:
        pass
    assert time.monotonic() - started < 0.3
    timed.release.set()

    cancelled = StuckStart()
    signal = threading.Event()
    box = {}
    def wait_for_cancelled_start():
        try:
            cancelled._ensure_started(timeout=1, should_cancel=signal.is_set)
        except Exception as error:  # noqa: BLE001 - asserted below
            box["result"] = error

    caller = threading.Thread(target=wait_for_cancelled_start)
    caller.start(); assert cancelled.entered.wait(1)
    signal.set(); caller.join(1)
    cancelled.release.set()
    assert not caller.is_alive() and isinstance(box.get("result"), RetryCancelledError), box


@check
def startup_or_scheduling_delay_cannot_start_a_request_after_its_deadline():
    import time

    from sliceagent.errors import TransportStartupError
    from sliceagent.llm import _AsyncTransportHub

    class DelayedStart(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.provider_starts = 0
        def _ensure_started(self, **kwargs):
            time.sleep(0.03)
            return super()._ensure_started(**kwargs)
        async def _chat(self, _spec, _kwargs, _publish):
            self.provider_starts += 1

    hub = DelayedStart()
    try:
        hub.run("chat", ("x", "", "none", 60), {}, timeout=0.005, close_grace=0.01)
        assert False, "expired local preparation must not open a provider request"
    except TransportStartupError:
        pass
    assert hub.provider_starts == 0


@check
def admission_delay_cannot_resurrect_an_expired_request_with_a_positive_clamp():
    import time

    import sliceagent.llm as llm_module
    from sliceagent.errors import TransportStartupError
    from sliceagent.llm import _AsyncTransportHub, _PhysicalCallGate

    original_admission = llm_module._TransportAdmission

    class DelayedAdmission(original_admission):
        def try_admit(self):
            time.sleep(0.03)
            return super().try_admit()

    class AuditHub(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.provider_starts = 0

        async def _chat(self, _spec, _kwargs, _publish):
            self.provider_starts += 1

    hub, gate = AuditHub(), _PhysicalCallGate(1)
    llm_module._TransportAdmission = DelayedAdmission
    try:
        try:
            hub.run(
                # The close grace is deliberately generous: the invariant is that the delayed admission
                # cannot start provider I/O, not that a loaded runner reports that result within 50ms.
                "chat", ("x", "", "none", 60), {}, timeout=0.01, close_grace=0.5,
                provider_gate=gate,
            )
            assert False, "post-admission expiry must surface before provider I/O"
        except TransportStartupError:
            pass
        assert hub.provider_starts == 0 and gate.active == 0
    finally:
        llm_module._TransportAdmission = original_admission


@check
def async_hub_cold_start_admits_exactly_one_loop_generation():
    import time

    from sliceagent.llm import _AsyncTransportHub

    class SlowStart(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.gate = threading.Event(); self.started = threading.Event(); self.count = 0
        def _new_event_loop(self):
            self.count += 1
            self.started.set()
            assert self.gate.wait(2), "test failed to release loop construction"
            return super()._new_event_loop()

    hub, errors = SlowStart(), []
    def start():
        try:
            hub._ensure_started()
        except Exception as error:  # noqa: BLE001 - collected for concurrent assertion
            errors.append(error)
    callers = [threading.Thread(target=start) for _ in range(12)]
    for caller in callers:
        caller.start()
    assert hub.started.wait(1)
    time.sleep(0.05)  # widen the old lock-release-before-ready race deterministically
    assert hub.count == 1, f"cold-start fanout launched {hub.count} transport loops"
    hub.gate.set()
    for caller in callers:
        caller.join(2)
    assert not errors and all(not caller.is_alive() for caller in callers), errors
    assert hub.count == 1 and hub._loop_ready


@check
def async_hub_failed_start_is_reported_then_recovers_on_a_fresh_generation():
    from sliceagent.llm import _AsyncTransportHub

    class FlakyStart(_AsyncTransportHub):
        def __init__(self):
            super().__init__(); self.attempts = 0
        def _new_event_loop(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("loop factory failed")
            return super()._new_event_loop()

    hub = FlakyStart()
    try:
        hub._ensure_started()
        assert False, "startup failure must reach the waiting caller"
    except RuntimeError as error:
        assert "failed during startup" in str(error)
    assert not hub._loop_ready and hub._loop is None
    hub._ensure_started()
    assert hub.attempts == 2 and hub._loop_ready and hub._loop is not None


@check
def model_runner_preserves_legacy_llm_and_forwards_control_to_openai_adapter():
    from sliceagent.interfaces import AssistantMessage
    from sliceagent.model_runner import complete_model_call

    class Legacy:
        model = "unknown"; max_tokens = 0
        def __init__(self): self.calls = 0
        def complete(self, messages, tools):
            self.calls += 1
            return AssistantMessage(content="legacy", tool_calls=[], usage=None, finish_reason="stop")

    legacy = Legacy()
    assert complete_model_call(legacy, [{"role": "user", "content": "x"}], [], retry=False).content == "legacy"
    assert legacy.calls == 1, "two-argument third-party LLMs remain valid"

    activity, controls = [], []
    llm = _stub(_CHUNKS, on_delta=None)
    original = llm._transport_hub
    def capture(kind, spec, kwargs, **control):
        controls.append(control["should_cancel"])
        return original.run(kind, spec, kwargs, **control)
    llm._transport_hub = NS(run=capture)
    cancel = lambda: False
    result = complete_model_call(
        llm, [{"role": "user", "content": "x"}], [], retry=False,
        should_cancel=cancel,
        transport_activity=lambda event, _detail: activity.append(event),
    )
    assert result.content == "Hello world" and controls == [cancel]
    assert activity[0] == "awaiting_model" and activity[-1] == "finished"


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
