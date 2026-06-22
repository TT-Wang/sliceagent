"""OpenAILLM — the default LLMClient over any OpenAI-COMPATIBLE endpoint (OpenAI, Moonshot,
DeepSeek, …). Configured by provider-AGNOSTIC env: LLM_API_KEY + LLM_BASE_URL (+ AGENT_MODEL);
OPENAI_*/MOONSHOT_* are accepted only as a back-compat fallback.

Proxy-aware (mirrors the prototype: defaults to a local ClashX proxy; set AGENT_PROXY=none
to go direct). The only module that imports the openai SDK — the core stays openai-free.
"""
from __future__ import annotations

import json
import os
import threading

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


def _usage_dict(raw) -> dict | None:
    """Normalize a provider usage object into a typed token breakdown (borrowed from Kimi kosong
    `TokenUsage`, usage.ts): `output` plus input split into cache-read / cache-creation / other. Keeps
    the legacy prompt_tokens/completion_tokens/cached_tokens keys so existing consumers keep working,
    and adds the typed fields the telemetry layer needs to measure per-turn FRESH-input cost (the moat).
    Provider-agnostic: every field defaults to 0, so a provider that omits a counter never crashes."""
    if not raw:
        return None
    prompt = getattr(raw, "prompt_tokens", 0) or 0
    output = getattr(raw, "completion_tokens", 0) or 0
    details = getattr(raw, "prompt_tokens_details", None)
    # cache READ: OpenAI nests it under prompt_tokens_details; Moonshot/some report it top-level.
    cache_read = (getattr(details, "cached_tokens", None)
                  or getattr(raw, "cached_tokens", None) or 0)
    # cache CREATION: Anthropic-compatible only (absent on OpenAI/Moonshot → 0).
    cache_create = getattr(raw, "cache_creation_input_tokens", 0) or 0
    input_other = max(0, prompt - cache_read - cache_create)
    usage = {
        "prompt_tokens": prompt, "completion_tokens": output,            # legacy / back-compat
        "input_other": input_other, "output": output,                   # typed (Kimi TokenUsage shape)
        "input_cache_read": cache_read, "input_cache_creation": cache_create,
    }
    if cache_read:
        usage["cached_tokens"] = cache_read                              # legacy key (only when present)
    return usage


class OpenAILLM:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, proxy: str | None = None, timeout: float | None = None):
        import httpx
        from openai import OpenAI

        # Request timeout is env-configurable (LLM_TIMEOUT_SEC) — large ACCUMULATED contexts produce a
        # single long non-streaming completion that legitimately exceeds the 60s default over a high-
        # latency proxy, and the hard watchdog would then false-kill a valid slow call (every retry
        # timing out → the turn parks 'error'). Default stays 60 for snappy interactive use.
        if timeout is None:
            try:
                timeout = float(os.environ.get("LLM_TIMEOUT_SEC") or os.environ.get("LLM_TIMEOUT") or 60.0)
            except (TypeError, ValueError):
                timeout = 60.0

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
        # Provider-AGNOSTIC prompt-cache routing key (OpenAI `prompt_cache_key`, accepted/ignored
        # harmlessly elsewhere). A session-stable key keeps every turn's requests on the same cached
        # prefix → higher cache-hit rate at ZERO added prompt tokens. Set via set_cache_key(); the
        # quirk stays isolated to this adapter (llm-agnostic). None → omit the kwarg entirely.
        self._cache_key: str | None = None
        # Optional LIVE token sink for interactive streaming (set by the cli/TUI). When set, complete()
        # STREAMS the completion and emits deltas (kind in {"content","reasoning"}) so a slow turn renders
        # LIVE instead of freezing on one blocking call (borrowed periphery — Kimi-style live events).
        # None → the blocking non-streaming path (eval/headless unchanged; byte-identical assembled result).
        self._on_delta = None
        # Sticky: set True once this provider 400s on reasoning_effort+tools (gpt-5.5 chat/completions);
        # thereafter reasoning_effort is dropped when tools are present (graceful degrade, no re-400).
        self._drop_reasoning_effort = False

    def set_cache_key(self, key: str | None) -> None:
        """Pin a session-scoped prompt-cache routing key (typically the session_id). Cheapest cache
        lever there is: raises cache-hit rate, adds no tokens. Safe to call repeatedly."""
        self._cache_key = key or None

    def set_delta_sink(self, fn) -> None:
        """Wire a live-delta sink for interactive STREAMING: fn(kind: str, text: str), kind in
        {'content','reasoning'}. None restores the blocking non-streaming path. Safe to call repeatedly.
        Pure transport/UX — the slice/loop/moat never see it (the assembled result is identical)."""
        self._on_delta = fn

    def _emit(self, kind: str, text: str) -> None:
        sink = getattr(self, "_on_delta", None)
        if sink and text:
            try:
                sink(kind, text)
            except Exception:  # noqa: BLE001 — a render error must NEVER break the LLM call
                pass

    def is_retryable(self, error: Exception) -> bool:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        from .errors import EmptyResponseError
        return isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError,
                                  InternalServerError, EmptyResponseError))

    def _on_alarm(self, signum, frame):
        """SIGALRM handler: a request blew the HARD wall-clock deadline → raise a retryable timeout."""
        import httpx
        from openai import APITimeoutError
        raise APITimeoutError(request=httpx.Request("POST", (self._base_url or "http://local") + "/chat/completions"))

    def _create(self, kwargs: dict):
        """Call the SDK with a HARD wall-clock deadline that ALWAYS fires, on ANY thread. The httpx/SDK
        read-timeout only bounds the gap BETWEEN bytes, so a connection that goes silent mid-response can
        hang far past `timeout` (observed: a stalled read wedging the loop 10+ min). On the main thread a
        SIGALRM deadline guarantees control returns to the retry path. OFF the main thread — e.g. a
        Terminal-Bench / any host ThreadPoolExecutor worker, where SIGALRM cannot arm — a watchdog thread
        enforces the SAME deadline (the abandoned SDK call is left to die on its socket while control
        returns). Without this, a wedged connection in a worker thread hangs the turn FOREVER, since the
        SDK timeout alone misses silent mid-response stalls. Task/provider-agnostic reliability."""
        import signal as _signal
        try:
            prev = _signal.signal(_signal.SIGALRM, self._on_alarm)
            _signal.alarm(self._hard_timeout)
        except (ValueError, AttributeError, OSError):
            return self._create_watchdog(kwargs)  # not the main thread → enforce the deadline via a thread
        try:
            return self.client.chat.completions.create(**kwargs)
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, prev)

    def _create_watchdog(self, kwargs: dict):
        """Off-main-thread hard deadline: run the SDK call in a 1-shot worker and abandon it if it blows
        the wall-clock budget (raise a RETRYABLE timeout so with_retry can retry, then the loop parks
        gracefully instead of hanging). A fresh executor per call so a wedged call never blocks the next."""
        import concurrent.futures as _f
        import httpx
        from openai import APITimeoutError
        ex = _f.ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-watchdog")
        fut = ex.submit(self.client.chat.completions.create, **kwargs)
        try:
            return fut.result(timeout=self._hard_timeout)
        except _f.TimeoutError:
            raise APITimeoutError(
                request=httpx.Request("POST", (self._base_url or "http://local") + "/chat/completions")
            )
        finally:
            ex.shutdown(wait=False)  # don't block on a possibly-wedged call; let its thread die with the socket

    def _create_streaming(self, kwargs: dict):
        """Interactive STREAMING variant of _create: drain the SSE stream into an assembled response under
        the SAME SIGALRM hard deadline (this path is always main-thread — set only by the cli — so SIGALRM
        arms; if not, fall back to the httpx read-timeout). Returns the same response SHAPE as _create so
        complete() is identical downstream. The deadline wraps the whole drain (the wait is in iteration,
        not in create()), so a stalled stream still aborts instead of hanging."""
        import signal as _signal
        try:
            prev = _signal.signal(_signal.SIGALRM, self._on_alarm)
            _signal.alarm(self._hard_timeout)
        except (ValueError, AttributeError, OSError):
            return self._stream_assemble(kwargs)  # not main thread → rely on the httpx read-timeout
        try:
            return self._stream_assemble(kwargs)
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, prev)

    def _stream_assemble(self, kwargs: dict):
        """Stream the completion, emit content/reasoning deltas live (self._emit), and assemble the pieces
        into a response object with the SAME shape complete() reads from the non-streamed path (choices[0]
        .message.content / .tool_calls[*].function.{name,arguments} / .finish_reason / .usage). So the rest
        of complete() — tool-arg JSON parse, usage dict, cache read-back — is byte-identical to the blocking
        path. include_usage gives the final usage chunk; tool-call deltas are reassembled by index."""
        from types import SimpleNamespace as NS
        skw = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        calls: dict[int, dict] = {}          # index → {id, name, args:[fragments]}
        finish = None
        usage = None
        for chunk in self.client.chat.completions.create(**skw):
            if getattr(chunk, "usage", None):
                usage = chunk.usage           # final include_usage chunk (choices may be empty here)
            for ch in (getattr(chunk, "choices", None) or []):
                if getattr(ch, "finish_reason", None):
                    finish = ch.finish_reason
                d = getattr(ch, "delta", None)
                if d is None:
                    continue
                txt = getattr(d, "content", None)
                if txt:
                    parts.append(txt); self._emit("content", txt)
                rc = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                if rc:
                    self._emit("reasoning", rc)
                for tcd in (getattr(d, "tool_calls", None) or []):
                    slot = calls.setdefault(getattr(tcd, "index", 0), {"id": None, "name": None, "args": []})
                    if getattr(tcd, "id", None):
                        slot["id"] = tcd.id
                    fn = getattr(tcd, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            slot["args"].append(fn.arguments)
        tool_calls = [NS(id=c["id"], function=NS(name=c["name"], arguments="".join(c["args"])))
                      for _, c in sorted(calls.items())]
        message = NS(content=("".join(parts) or None), tool_calls=tool_calls)
        return NS(choices=[NS(message=message, finish_reason=finish)], usage=usage)

    def _reasoning_kwargs(self) -> dict:
        """Map the provider-agnostic reasoning intent to the ACTIVE provider's knob; no-op (never
        error) for providers that have none. Keeps the quirk isolated to this adapter."""
        if self.reasoning != "fast":
            return {}
        from .model_catalog import capability
        model, base = self.model.lower(), self._base_url.lower()
        if "deepseek" in model or "deepseek" in base:
            return {"extra_body": {"thinking": {"type": "disabled"}}}  # deepseek: disable thinking
        if capability(self.model, self._base_url).supports_reasoning_effort:
            return {"reasoning_effort": "low"}                          # OpenAI reasoning models (catalog)
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

    def _cache_routing_kwargs(self) -> dict:
        """Map the session cache-routing hint to the ACTIVE provider; gated like the sibling
        quirk-mappers (`_reasoning_kwargs`/`_cache_kwargs`) so a provider-specific param never
        reaches an endpoint that rejects it.

        `prompt_cache_key` is an OpenAI Chat-Completions field (routes identical-prefix requests to
        the same cache shard for a higher hit rate; 0 added tokens). OpenAI-compatible providers
        (the default Moonshot path, DeepSeek) accept-and-ignore it harmlessly. It is INVALID on an
        Anthropic-compatible endpoint (which caches via explicit cache_control breakpoints — see
        `_cache_kwargs`), so we return {} there to keep that request byte-stable and untouched.
        """
        key = getattr(self, "_cache_key", None)
        if not key:
            return {}
        model, base = self.model.lower(), self._base_url.lower()
        if "claude" in model or "anthropic" in base:
            return {}  # Anthropic uses cache_control, not prompt_cache_key
        return {"prompt_cache_key": key}

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
            # Provider quirk (now sourced from the model catalog — gpt-5/o-series renamed this param to
            # max_completion_tokens and REJECT max_tokens with a 400). One source of truth, not inline.
            from .model_catalog import capability
            kwargs[capability(self.model, self._base_url).tokens_param] = self.max_tokens
        self._merge_kwargs(kwargs, self._cache_routing_kwargs())  # session-stable cache routing (0 added tokens)
        self._merge_kwargs(kwargs, self._reasoning_kwargs())
        self._merge_kwargs(kwargs, self._cache_kwargs(messages))
        # Provider quirk (isolated here, llm-agnostic): some reasoning models (gpt-5.5) reject
        # reasoning_effort TOGETHER with function tools on /v1/chat/completions (400 — "use /v1/responses").
        # Once seen, drop reasoning_effort whenever tools are present so we degrade to default reasoning
        # instead of 400ing every tool-calling turn. (Sticky — set in the except below.)
        if getattr(self, "_drop_reasoning_effort", False) and kwargs.get("tools"):
            kwargs.pop("reasoning_effort", None)
        # STREAM only on the MAIN thread with a live sink wired (the interactive turn). OFF-main runs —
        # parallel subagents/explorers sharing this llm via run_scheduled threads — take the BLOCKING path
        # so they keep the off-main hard-deadline watchdog AND never racily drive the single TUI spinner from
        # N threads. getattr keeps the object-__new__ test stubs working. Same assembled result either way.
        _stream = (getattr(self, "_on_delta", None) is not None
                   and threading.current_thread() is threading.main_thread())
        _creator = self._create_streaming if _stream else self._create
        try:
            resp = _creator(kwargs)
        except Exception as e:
            # Context overflow is NOT a backoff case (is_retryable stays unchanged): signal the
            # rebuild loop to TIGHTEN the slice rather than re-send the identical oversized request.
            if is_context_overflow(e):
                raise ContextOverflow(e, status_code=getattr(e, "status_code", None)) from e
            # reasoning_effort + tools rejected by this model → drop it, remember, retry ONCE (graceful
            # degrade to default reasoning instead of crashing the turn). General; no model name hardcoded.
            if "reasoning_effort" in str(e) and kwargs.pop("reasoning_effort", None) is not None:
                self._drop_reasoning_effort = True
                resp = _creator(kwargs)
            else:
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
        # Degenerate completion — no content AND no tool calls (and not a content-filter stop). Some
        # providers/proxies occasionally emit an empty body; returning it stalls the loop, so raise a
        # RETRYABLE error (Kimi APIEmptyResponseError) and let with_retry re-roll. content_filter is
        # excluded — re-rolling would just filter again.
        if not (msg.content or "").strip() and not calls and choice.finish_reason != "content_filter":
            from .errors import EmptyResponseError
            raise EmptyResponseError(f"empty completion (finish_reason={choice.finish_reason})")
        usage = _usage_dict(resp.usage)
        return AssistantMessage(
            content=msg.content, tool_calls=calls, usage=usage, finish_reason=choice.finish_reason
        )
