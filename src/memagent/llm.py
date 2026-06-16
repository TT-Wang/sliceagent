"""OpenAILLM — the default LLMClient over any OpenAI-COMPATIBLE endpoint (OpenAI, Moonshot,
DeepSeek, …). Configured by provider-AGNOSTIC env: LLM_API_KEY + LLM_BASE_URL (+ AGENT_MODEL);
OPENAI_*/MOONSHOT_* are accepted only as a back-compat fallback.

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

        # Provider-AGNOSTIC env: LLM_API_KEY / LLM_BASE_URL are canonical. OPENAI_*/MOONSHOT_* are
        # kept ONLY as a back-compat fallback (the SDK is OpenAI-compatible and many shells already
        # export OPENAI_API_KEY) — the surface the user configures says "LLM", not a provider name.
        kwargs: dict = {"api_key": api_key or os.environ.get("LLM_API_KEY")
                        or os.environ.get("OPENAI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")}
        resolved_base = base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if resolved_base:
            kwargs["base_url"] = resolved_base
        elif os.environ.get("MOONSHOT_API_KEY") and not (
                os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            kwargs["base_url"] = "https://api.moonshot.cn/v1"

        self.client = OpenAI(http_client=http_client, max_retries=2, **kwargs)
        self.model = model or os.environ.get("AGENT_MODEL") or "gpt-5.5"
        self._base_url = kwargs.get("base_url") or ""
        # Provider-AGNOSTIC reasoning intent: "full" (default) keeps the model's reasoning; "fast"
        # minimizes it (wall-clock tracks reasoning tokens, and the slice reconstructs ground-truth
        # STATE each turn, which can substitute for per-step re-derivation). The core/agent never
        # sees this — _reasoning_kwargs() maps it to each provider's own param, here in the adapter
        # (the one place permitted to know provider specifics). AGENT_THINKING=off kept as an alias.
        self.reasoning = (os.environ.get("AGENT_REASONING")
                          or ("fast" if (os.environ.get("AGENT_THINKING") or "").lower() == "off"
                              else "full")).lower()
        # Cap the completion generously. Providers default low (deepseek ~4096); a response that
        # exceeds it truncates mid-edit → the agent retries the broken edit → step/time blowup. A
        # generous explicit cap avoids that. Standard param (provider-agnostic). 0 → leave default.
        self.max_tokens = int(os.environ.get("AGENT_MAX_TOKENS") or 8192)

    def is_retryable(self, error: Exception) -> bool:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
        return isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError))

    def _reasoning_kwargs(self) -> dict:
        """Map the provider-agnostic reasoning intent to the ACTIVE provider's knob; no-op (never
        error) for providers that have none. Keeps the quirk isolated to this adapter."""
        if self.reasoning != "fast":
            return {}
        model, base = self.model.lower(), self._base_url.lower()
        if "deepseek" in model or "deepseek" in base:
            return {"extra_body": {"thinking": {"type": "disabled"}}}  # deepseek: disable thinking
        if model.startswith(("o1", "o3", "o4", "gpt-5")):
            return {"reasoning_effort": "low"}                          # OpenAI reasoning models
        return {}  # unknown / non-reasoning provider → leave at provider default (graceful)

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        kwargs: dict = dict(model=self.model, messages=messages, tools=tools, tool_choice="auto")
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        kwargs.update(self._reasoning_kwargs())
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
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
        return AssistantMessage(
            content=msg.content, tool_calls=calls, usage=usage, finish_reason=choice.finish_reason
        )
