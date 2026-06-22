"""Interactive STREAMING (borrowed Kimi-style live events): when a delta sink is wired, OpenAILLM streams
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
