"""OpenAILLM — the default LLMClient (OpenAI-compatible: OpenAI, Moonshot, …).

Proxy-aware (mirrors the prototype: defaults to a local ClashX proxy; set AGENT_PROXY=none
to go direct). The only module that imports the openai SDK — the core stays openai-free.
"""
from __future__ import annotations

import json
import os

from .interfaces import AssistantMessage, ToolCall


class OpenAILLM:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, proxy: str | None = None, timeout: float = 60.0):
        import httpx
        from openai import OpenAI

        proxy = proxy or os.environ.get("AGENT_PROXY") or os.environ.get("HTTPS_PROXY") \
            or os.environ.get("HTTP_PROXY") or "http://127.0.0.1:7890"
        use_proxy = bool(proxy) and proxy != "none"
        http_client = httpx.Client(proxy=proxy, timeout=timeout) if use_proxy else httpx.Client(timeout=timeout)

        kwargs: dict = {"api_key": api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")}
        if base_url:
            kwargs["base_url"] = base_url
        elif os.environ.get("MOONSHOT_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            kwargs["base_url"] = "https://api.moonshot.cn/v1"

        self.client = OpenAI(http_client=http_client, max_retries=2, **kwargs)
        self.model = model or os.environ.get("AGENT_MODEL") or "gpt-5.5"

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, tool_choice="auto",
        )
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        usage = None
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return AssistantMessage(content=msg.content, tool_calls=calls, usage=usage)
