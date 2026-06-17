"""OpenAILLM — the default LLMClient over any OpenAI-COMPATIBLE endpoint (OpenAI, Moonshot,
DeepSeek, …). Configured by provider-AGNOSTIC env: LLM_API_KEY + LLM_BASE_URL (+ AGENT_MODEL);
OPENAI_*/MOONSHOT_* are accepted only as a back-compat fallback.

Proxy-aware (mirrors the prototype: defaults to a local ClashX proxy; set AGENT_PROXY=none
to go direct). The only module that imports the openai SDK — the core stays openai-free.
"""
from __future__ import annotations

import json
import os

from .context_overflow import ContextOverflow, is_context_overflow
from .interfaces import AssistantMessage, ToolCall

_CLASHX = "http://127.0.0.1:7890"  # local ClashX default for foreign endpoints (OpenAI) behind the GFW


def _choose_proxy(resolved_base: str | None, explicit: str | None) -> str:
    """Pick the HTTP proxy for the active provider. An EXPLICIT setting (arg or AGENT_PROXY/HTTPS_PROXY/
    HTTP_PROXY) ALWAYS wins. Otherwise: foreign endpoints (OpenAI/gpt) route through the local ClashX
    proxy, CN-direct providers (deepseek/moonshot) go DIRECT — so picking a model 'just works' without
    juggling AGENT_PROXY. Returns a proxy URL or 'none'. (Environment/provider quirk, isolated here.)"""
    if explicit:
        return explicit
    base_l = (resolved_base or "").lower()
    direct = any(d in base_l for d in ("deepseek", "moonshot", "127.0.0.1", "localhost"))
    return "none" if direct else _CLASHX


class OpenAILLM:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, proxy: str | None = None, timeout: float = 60.0):
        import httpx
        from openai import OpenAI

        # Provider-AGNOSTIC env: LLM_API_KEY / LLM_BASE_URL are canonical. OPENAI_*/MOONSHOT_* are
        # kept ONLY as a back-compat fallback (the SDK is OpenAI-compatible and many shells already
        # export OPENAI_API_KEY) — the surface the user configures says "LLM", not a provider name.
        # Resolve the ENDPOINT first: the proxy choice below depends on which provider it is.
        kwargs: dict = {"api_key": api_key or os.environ.get("LLM_API_KEY")
                        or os.environ.get("OPENAI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")}
        resolved_base = base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not resolved_base and os.environ.get("MOONSHOT_API_KEY") and not (
                os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            resolved_base = "https://api.moonshot.cn/v1"
        if resolved_base:
            kwargs["base_url"] = resolved_base

        # Proxy: an EXPLICIT setting (the arg, or AGENT_PROXY/HTTPS_PROXY/HTTP_PROXY) ALWAYS wins.
        # Otherwise choose per-provider so a model "just works": foreign endpoints (OpenAI/gpt) route
        # through the local ClashX proxy; CN-direct providers (deepseek/moonshot) go DIRECT. This is an
        # environment/provider quirk, isolated to this adapter (llm-agnostic) and overridable.
        proxy = _choose_proxy(resolved_base, proxy or os.environ.get("AGENT_PROXY")
                              or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
        use_proxy = bool(proxy) and proxy != "none"
        http_client = httpx.Client(proxy=proxy, timeout=timeout) if use_proxy else httpx.Client(timeout=timeout)

        # Enforce the request timeout at the SDK layer too. The openai SDK applies its OWN per-request
        # timeout (default ~600s) which OVERRIDES the httpx client's, so without passing it here a
        # stalled/half-open connection hangs ~10 min before max_retries ever fires (observed: a wedged
        # direct connection, timeout never tripping). Passing `timeout` makes a wedged call abort
        # promptly so the retry recovers on a fresh connection — task-agnostic reliability (→ wall time).
        self.client = OpenAI(http_client=http_client, timeout=timeout, max_retries=2, **kwargs)
        # HARD wall-clock backstop for _create() (SIGALRM): a few seconds above the SDK read-timeout so
        # the SDK's own (cleaner) timeout fires first when it can, and SIGALRM only catches the stalls
        # the read-timeout misses (silent mid-response connections).
        self._hard_timeout = max(int(timeout) + 15, 30)
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

    def _on_alarm(self, signum, frame):
        """SIGALRM handler: a request blew the HARD wall-clock deadline → raise a retryable timeout."""
        import httpx
        from openai import APITimeoutError
        raise APITimeoutError(request=httpx.Request("POST", (self._base_url or "http://local") + "/chat/completions"))

    def _create(self, kwargs: dict):
        """Call the SDK with a HARD wall-clock deadline that ALWAYS fires. The httpx/SDK read-timeout
        only bounds the gap BETWEEN bytes, so a connection that goes silent mid-response can hang far
        past `timeout` (observed: a stalled Moonshot read wedging the loop for 10+ min). A SIGALRM
        deadline guarantees the call returns control to the retry path. POSIX + main-thread only;
        falls back to a plain call elsewhere (Windows / worker thread), where the SDK timeout still
        applies. Task-agnostic, provider-agnostic reliability — the backstop the timeout param promised."""
        import signal as _signal
        armed = False
        try:
            prev = _signal.signal(_signal.SIGALRM, self._on_alarm)
            _signal.alarm(self._hard_timeout)
            armed = True
        except (ValueError, AttributeError, OSError):
            armed = False  # not main thread / no SIGALRM → rely on the SDK timeout
        try:
            return self.client.chat.completions.create(**kwargs)
        finally:
            if armed:
                _signal.alarm(0)
                _signal.signal(_signal.SIGALRM, prev)

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

    def _cache_kwargs(self, messages: list[dict]) -> dict:
        """Map prompt-caching intent to the ACTIVE provider's knob; no-op for providers without
        one. Modeled on `_reasoning_kwargs` — the quirk stays isolated to this adapter.

        Only Claude/Anthropic-compatible endpoints support an explicit prompt-cache breakpoint;
        every other provider (the default gpt-5.5 / OpenAI-compatible path) returns {} so the
        request is byte-stable and untouched. For an Anthropic-compatible endpoint we return a
        TODO-stubbed {} for now: the exact `extra_body` cache_control shape is DEFERRED until a
        real Anthropic base_url is wired (the safe half — a byte-stable prefix + cached_tokens
        read-back — is already in place and provider-agnostic).
        """
        model, base = self.model.lower(), self._base_url.lower()
        if "claude" not in model and "anthropic" not in base:
            return {}  # non-Claude provider → no explicit cache breakpoint
        # Anthropic-compatible endpoint: DEFER the real cache_control extra_body shape (see
        # adopt_plan.md sec 6 defer). Stubbed {} keeps the request byte-stable until wired.
        # TODO(anthropic): set extra_body cache_control on the system/stable prefix against a
        #   live Anthropic base_url; MERGE with _reasoning_kwargs' extra_body, do not overwrite.
        return {}

    def _merge_kwargs(self, kwargs: dict, extra: dict) -> None:
        """Fold `extra` into `kwargs`, MERGING `extra_body` instead of overwriting it.

        Both `_reasoning_kwargs` and `_cache_kwargs` may set `extra_body`; a plain
        `kwargs.update(...)` would clobber whichever ran first. Merge the nested dict so both
        provider quirks survive.
        """
        for key, value in extra.items():
            if key == "extra_body" and isinstance(kwargs.get("extra_body"), dict) and isinstance(value, dict):
                kwargs["extra_body"] = {**kwargs["extra_body"], **value}
            else:
                kwargs[key] = value

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        kwargs: dict = dict(model=self.model, messages=messages, tools=tools, tool_choice="auto")
        if self.max_tokens:
            # Provider quirk (isolated here per llm-agnostic): OpenAI gpt-5/o-series renamed this
            # param to max_completion_tokens and REJECT max_tokens with a 400. Pick the right key.
            key = ("max_completion_tokens"
                   if self.model.lower().startswith(("o1", "o3", "o4", "gpt-5"))
                   else "max_tokens")
            kwargs[key] = self.max_tokens
        self._merge_kwargs(kwargs, self._reasoning_kwargs())
        self._merge_kwargs(kwargs, self._cache_kwargs(messages))
        try:
            resp = self._create(kwargs)
        except Exception as e:
            # Context overflow is NOT a backoff case (is_retryable stays unchanged): signal the
            # rebuild loop to TIGHTEN the slice rather than re-send the identical oversized request.
            if is_context_overflow(e):
                raise ContextOverflow(e, status_code=getattr(e, "status_code", None)) from e
            raise
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
            # Cache read-back: provider-agnostic, key omitted when absent (no crash on providers
            # that don't report a cache hit). OpenAI/Anthropic-compatible both nest it here.
            cached = getattr(getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", None)
            if cached is not None:
                usage["cached_tokens"] = cached
        return AssistantMessage(
            content=msg.content, tool_calls=calls, usage=usage, finish_reason=choice.finish_reason
        )
