"""Official DeepSeek V4 thinking/tool compatibility. No network or pytest.

Run: PYTHONPATH=src python tests/test_deepseek_v4.py
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hooks import Hooks  # noqa: E402
from sliceagent.interfaces import AssistantMessage, ToolCall  # noqa: E402
from sliceagent.llm import OpenAILLM  # noqa: E402
from sliceagent.loop import _assistant_message, run_turn  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _chunk(*, content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    if usage is not None:
        return NS(choices=[], usage=usage)
    delta = NS(content=content, reasoning_content=reasoning, reasoning=None, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=None)


def _tool_delta(*, index=0, call_id=None, name=None, arguments=None):
    return NS(index=index, id=call_id, function=NS(name=name, arguments=arguments))


class _CaptureHub:
    def __init__(self, chunks):
        self.chunks = list(chunks); self.kwargs = []

    def run(self, kind, _spec, kwargs, **control):
        assert kind == "chat"
        self.kwargs.append(dict(kwargs))
        for chunk in self.chunks:
            control["on_item"](chunk)


def _llm(*, model="deepseek-v4-pro", base="https://api.deepseek.com/v1",
         reasoning="full", chunks=()):
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.model = model
    llm._base_url = base
    llm.reasoning = reasoning
    llm.max_tokens = 0
    llm._cache_key = None
    llm._drop_reasoning_effort = False
    llm._hard_timeout = 30
    llm._on_delta = None
    llm._transport_activity = None
    llm._stream_transport_enabled = True
    llm._transport_spec = ("test", base, "none", 60.0)
    llm._transport_hub = _CaptureHub(chunks or [
        _chunk(content="done"), _chunk(finish="stop"),
    ])
    llm.client = NS(chat=NS(completions=object()))
    return llm


@check
def official_thinking_omits_tool_choice_and_preserves_streamed_reasoning():
    usage = NS(prompt_tokens=7, completion_tokens=4, prompt_tokens_details=None)
    llm = _llm(chunks=[
        _chunk(reasoning="first "),
        _chunk(reasoning="reason"),
        _chunk(content="checking"),
        _chunk(tool_calls=[_tool_delta(
            call_id="c1", name="read_file", arguments='{"path":"x.py"}',
        )]),
        _chunk(finish="tool_calls"),
        _chunk(usage=usage),
    ])
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    message = llm.complete([{"role": "user", "content": "inspect"}], tools)
    sent = llm._transport_hub.kwargs[-1]
    assert "tool_choice" not in sent, sent
    assert sent["extra_body"]["thinking"]["type"] == "enabled", sent
    assert message.reasoning_content == "first reason"
    assert message.content == "checking" and message.tool_calls[0].id == "c1"


@check
def official_v4_effort_maps_high_and_max_without_openai_xhigh():
    for profile, expected in (("high", "high"), ("max", "max"), ("xhigh", "max")):
        llm = _llm(reasoning=profile)
        llm.complete([{"role": "user", "content": "x"}], [])
        sent = llm._transport_hub.kwargs[-1]
        assert sent["reasoning_effort"] == expected, (profile, sent)
        assert sent["extra_body"] == {"thinking": {"type": "enabled"}}
        assert "tool_choice" not in sent


@check
def fast_and_legacy_aliases_keep_their_mode_contracts():
    fast = _llm(reasoning="fast")
    fast.complete([{"role": "user", "content": "x"}], [])
    sent = fast._transport_hub.kwargs[-1]
    assert sent["tool_choice"] == "auto"
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}

    reasoner = _llm(model="deepseek-reasoner", reasoning="high")
    reasoner.complete([{"role": "user", "content": "x"}], [])
    sent = reasoner._transport_hub.kwargs[-1]
    assert "tool_choice" not in sent and sent["reasoning_effort"] == "high"
    assert sent["extra_body"] == {"thinking": {"type": "enabled"}}

    chat = _llm(model="deepseek-chat", reasoning="full")
    chat.complete([{"role": "user", "content": "x"}], [])
    sent = chat._transport_hub.kwargs[-1]
    assert sent["tool_choice"] == "auto"
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}


@check
def unofficial_router_keeps_generic_tool_choice_compatibility():
    llm = _llm(base="https://router.example/v1", reasoning="full")
    llm.complete([{"role": "user", "content": "x"}], [])
    sent = llm._transport_hub.kwargs[-1]
    assert sent["tool_choice"] == "auto"
    assert "reasoning_effort" not in sent and "extra_body" not in sent


@check
def blocking_response_preserves_reasoning_content_too():
    llm = _llm()
    llm._stream_transport_enabled = False
    response = NS(
        choices=[NS(
            message=NS(content="", reasoning_content="blocking thought", tool_calls=[NS(
                id="c1", function=NS(name="read_file", arguments='{"path":"x.py"}'),
            )]),
            finish_reason="tool_calls",
        )],
        usage=NS(prompt_tokens=2, completion_tokens=3, prompt_tokens_details=None),
    )
    captured = []
    llm._create = lambda kwargs: captured.append(dict(kwargs)) or response
    message = llm.complete([{"role": "user", "content": "inspect"}], [])
    assert message.reasoning_content == "blocking thought"
    assert "tool_choice" not in captured[-1]


@check
def accumulated_tool_call_message_replays_reasoning_content_and_nonnull_content():
    response = AssistantMessage(
        content=None,
        reasoning_content="must survive",
        tool_calls=[ToolCall("c1", "read_file", {"path": "x.py"})],
        finish_reason="tool_calls",
    )
    message = _assistant_message(response)
    assert message["content"] == "", "DeepSeek requires non-null assistant content"
    assert message["reasoning_content"] == "must survive"
    assert message["tool_calls"][0]["id"] == "c1"


@check
def run_turn_replays_reasoning_on_the_request_after_a_tool_result():
    class ScriptLLM:
        model = "unknown"; max_tokens = 0
        def __init__(self): self.calls = 0; self.seen = []
        def complete(self, messages, _tools):
            self.seen.append([dict(message) for message in messages])
            self.calls += 1
            if self.calls == 1:
                return AssistantMessage(
                    content="", reasoning_content="tool reasoning",
                    tool_calls=[ToolCall("call-1", "read_file", {"path": "x.py"})],
                    usage={}, finish_reason="tool_calls",
                )
            return AssistantMessage(content="done", usage={}, finish_reason="stop")

    class Tools:
        def schemas(self): return []
        def accesses(self, _name, _args): return []
        def run(self, _name, _args): return "file contents"

    llm = ScriptLLM()
    result = run_turn(
        build_slice=lambda: [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        ],
        llm=llm, tools=Tools(), dispatch=lambda _event: None, hooks=Hooks(), max_steps=3,
    )
    assert result.stop_reason == "end_turn", result.stop_reason
    replay = next(message for message in llm.seen[1] if message.get("role") == "assistant")
    assert replay["reasoning_content"] == "tool reasoning", replay
    tool_result = next(message for message in llm.seen[1] if message.get("role") == "tool")
    assert replay["content"] == ""
    assert replay["tool_calls"][0]["id"] == tool_result["tool_call_id"]


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {error!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
