"""OpenAILLM — the default LLMClient over any OpenAI-COMPATIBLE endpoint (OpenAI, Moonshot,
DeepSeek, …). Configured by provider-AGNOSTIC env: LLM_API_KEY + LLM_BASE_URL (+ AGENT_MODEL);
OPENAI_*/MOONSHOT_* are accepted only as a back-compat fallback.

Connects directly by default; set AGENT_PROXY (or HTTPS_PROXY/HTTP_PROXY) to route through an HTTP
proxy, or AGENT_PROXY=none to force a direct connection. The only module that imports the openai SDK
— the core stays openai-free.
"""
from __future__ import annotations

import json
import os
import threading

from .context_overflow import ContextOverflow, is_context_overflow
from .interfaces import AssistantMessage, ToolCall


def _import_api_timeout_error():
    """APITimeoutError moved between openai SDK versions; import defensively."""
    try:
        from openai import APITimeoutError
        return APITimeoutError
    except ImportError:
        pass
    try:
        from openai.error import Timeout as APITimeoutError
        return APITimeoutError
    except ImportError:
        pass
    # Fallback: a plain retryable timeout that is_retryable will still classify.
    class _FallbackTimeoutError(Exception):
        pass
    return _FallbackTimeoutError


def _choose_proxy(resolved_base: str | None, explicit: str | None) -> str:
    """Pick the HTTP proxy for the active provider. An EXPLICIT setting (arg or AGENT_PROXY/HTTPS_PROXY/
    HTTP_PROXY) wins; otherwise connect DIRECTLY — no proxy by default. AGENT_PROXY=none/off forces a
    direct connection. Returns a proxy URL or 'none'. (Environment quirk, isolated here.)"""
    if explicit and explicit.strip():            # a whitespace-only env var is NOT an explicit setting
        s = explicit.strip()
        if s.lower() in ("off", "none", "direct", "no", "false", "0"):
            return "none"            # AGENT_PROXY=off → force a DIRECT connection (was treated as a URL → bug)
        return s
    return "none"                    # default: direct connection, no proxy


def _int(x) -> int:
    """Coerce a provider-supplied token counter to int; non-numeric (str/object/None) → 0. Some providers
    report counts as strings or odd objects, and `x or 0` keeps a truthy non-number → arithmetic TypeError."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _usage_dict(raw) -> dict | None:
    """Normalize a provider usage object into a typed token breakdown: `output` plus input split into
    cache-read / cache-creation / other. Keeps
    the legacy prompt_tokens/completion_tokens/cached_tokens keys so existing consumers keep working,
    and adds the typed fields the telemetry layer needs to measure per-turn FRESH-input cost (the moat).
    Provider-agnostic: every field defaults to 0, so a provider that omits a counter never crashes."""
    if not raw:
        return None
    prompt = _int(getattr(raw, "prompt_tokens", 0))
    output = _int(getattr(raw, "completion_tokens", 0))
    details = getattr(raw, "prompt_tokens_details", None)
    # cache READ: OpenAI nests it under prompt_tokens_details; Moonshot/some report it top-level. Use
    # is-None (not truthiness) to choose the source — a legit cached_tokens=0 from details must NOT fall
    # through to raw.cached_tokens (that miscounted a no-cache-hit turn as a top-level value).
    _cr = getattr(details, "cached_tokens", None)
    cache_read = _int(_cr if _cr is not None else getattr(raw, "cached_tokens", None))
    # cache CREATION: Anthropic-compatible only (absent on OpenAI/Moonshot → 0).
    cache_create = _int(getattr(raw, "cache_creation_input_tokens", 0))
    input_other = max(0, prompt - cache_read - cache_create)
    usage = {
        "prompt_tokens": prompt, "completion_tokens": output,            # legacy / back-compat
        "input_other": input_other, "output": output,                   # typed token fields
        "input_cache_read": cache_read, "input_cache_creation": cache_create,
    }
    if cache_read:
        usage["cached_tokens"] = cache_read                              # legacy key (only when present)
    _cost = getattr(raw, "cost", None)                                   # OpenRouter: authoritative $ per call
    if isinstance(_cost, (int, float)) and _cost >= 0:
        usage["cost_usd"] = float(_cost)
    return usage


def _as_text(content) -> str:
    """Flatten a chat `content` (str OR a multimodal parts list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return "" if content is None else str(content)


def _to_responses_content(content):
    """Map a chat message `content` to the Responses-API content shape: a plain string passes through;
    a multimodal parts list is converted (text→input_text, image_url→input_image)."""
    if not isinstance(content, list):
        return content if content is not None else ""
    parts = []
    for p in content:
        if not isinstance(p, dict):
            parts.append({"type": "input_text", "text": str(p)}); continue
        t = p.get("type")
        if t in ("text", "input_text"):
            parts.append({"type": "input_text", "text": p.get("text", "")})
        elif t in ("image_url", "input_image"):
            u = p.get("image_url")
            url = u.get("url") if isinstance(u, dict) else u
            parts.append({"type": "input_image", "image_url": url})
        else:
            parts.append(p)
    return parts


def _to_responses_input(messages: list[dict]) -> list[dict]:
    """Convert chat/completions `messages` → Responses-API `input` items. The Responses API has no
    `tool` role and no nested `tool_calls`: an assistant tool call becomes a flat {type:function_call}
    item and a tool result a {type:function_call_output} item; plain system/user/assistant stay as
    {role, content}. Pure — testable offline."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":                                   # tool result → function_call_output
            out.append({"type": "function_call_output",
                        "call_id": m.get("tool_call_id", ""), "output": _as_text(m.get("content"))})
        elif role == "assistant" and m.get("tool_calls"):    # assistant turn that called tools
            txt = _as_text(m.get("content"))
            if txt:
                out.append({"role": "assistant", "content": txt})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                out.append({"type": "function_call", "call_id": tc.get("id", ""),
                            "name": fn.get("name", ""), "arguments": fn.get("arguments") or "{}"})
        else:                                                # plain text (system/user/assistant)
            out.append({"role": role or "user", "content": _to_responses_content(m.get("content"))})
    return out


def _to_responses_tools(tools: list[dict]) -> list[dict]:
    """chat tool schema {type:function, function:{name,description,parameters}} → Responses flat
    {type:function, name, description, parameters}."""
    out = []
    for t in (tools or []):
        fn = t.get("function") if isinstance(t, dict) else None
        if fn:
            out.append({"type": "function", "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
        elif isinstance(t, dict) and t.get("type") == "function" and "name" in t:
            out.append(t)                                    # already Responses-shaped
    return out


def _responses_usage(u):
    """Adapt a Responses `usage` (input_tokens / output_tokens / input_tokens_details.cached_tokens) to
    the chat-usage attribute names `_usage_dict` reads, so token telemetry/cost is unchanged. None→None."""
    if not u:
        return None
    from types import SimpleNamespace as NS
    det = getattr(u, "input_tokens_details", None)
    cached = (getattr(det, "cached_tokens", 0) if det else 0) or 0
    return NS(prompt_tokens=getattr(u, "input_tokens", 0) or 0,
              completion_tokens=getattr(u, "output_tokens", 0) or 0,
              prompt_tokens_details=NS(cached_tokens=cached),
              cached_tokens=cached, cache_creation_input_tokens=0)


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
                        or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
                        or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
                        or os.environ.get("MOONSHOT_API_KEY")}
        resolved_base = base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not resolved_base and os.environ.get("MOONSHOT_API_KEY") and not (
                os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            resolved_base = "https://api.moonshot.cn/v1"
        if resolved_base:
            kwargs["base_url"] = resolved_base

        # Proxy: an EXPLICIT setting (the arg, or AGENT_PROXY/HTTPS_PROXY/HTTP_PROXY) wins; otherwise connect
        # directly (no proxy by default). Isolated to this adapter (llm-agnostic) and fully overridable.
        proxy = _choose_proxy(resolved_base, proxy or os.environ.get("AGENT_PROXY")
                              or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
        use_proxy = bool(proxy) and proxy != "none"
        self.proxy_used = proxy if use_proxy else "direct"   # exposed so the CLI can announce the route (A4)
        # trust_env=False: the proxy is resolved EXPLICITLY above (AGENT_PROXY/HTTPS_PROXY/HTTP_PROXY →
        # _choose_proxy), so httpx must NOT ALSO auto-read ambient proxy env — otherwise AGENT_PROXY=none
        # ("force direct") still routes through an ambient HTTPS_PROXY (httpx defaults trust_env=True).
        http_client = (httpx.Client(proxy=proxy, timeout=timeout, trust_env=False) if use_proxy
                       else httpx.Client(timeout=timeout, trust_env=False))

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
        self._timeout = timeout                              # kept for live endpoint switches (switch())
        # No built-in default model — the user picks (parallels the CLI's model gate; a silent
        # fallback here would contradict it for library/embedding callers).
        self.model = model or os.environ.get("AGENT_MODEL") or ""
        if not self.model:
            raise ValueError("No model configured. Pass model=... or set AGENT_MODEL "
                             "(interactive setup: `sliceagent init`).")
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
        # per-REQUEST completion cap — its OWN env var, decoupled from AGENT_MAX_TOKENS (which is the
        # per-turn BudgetHook budget; sharing the key made one value drive two quantities orders of
        # magnitude apart). Guarded so a malformed value degrades to the default instead of crashing init.
        try:
            self.max_tokens = int(os.environ.get("AGENT_COMPLETION_TOKENS") or 8192)
        except (TypeError, ValueError):
            self.max_tokens = 8192
        # Provider-AGNOSTIC prompt-cache routing key (OpenAI `prompt_cache_key`, accepted/ignored
        # harmlessly elsewhere). A session-stable key keeps every turn's requests on the same cached
        # prefix → higher cache-hit rate at ZERO added prompt tokens. Set via set_cache_key(); the
        # quirk stays isolated to this adapter (llm-agnostic). None → omit the kwarg entirely.
        self._cache_key: str | None = None
        # Optional LIVE token sink for interactive streaming (set by the cli/TUI). When set, complete()
        # STREAMS the completion and emits deltas (kind in {"content","reasoning"}) so a slow turn renders
        # LIVE instead of freezing on one blocking call.
        # None → the blocking non-streaming path (eval/headless unchanged; byte-identical assembled result).
        self._on_delta = None
        # Sticky: set True once this provider 400s on reasoning_effort+tools (gpt-5.5 chat/completions);
        # thereafter reasoning_effort is dropped when tools are present (graceful degrade, no re-400).
        self._drop_reasoning_effort = False

    def switch(self, *, model: str | None = None, reasoning: str | None = None,
               base_url: str | None = None, api_key: str | None = None) -> None:
        """Live-switch model / reasoning — and, when `base_url`/`api_key` are given, the PROVIDER too:
        the client is rebuilt against the new endpoint (with the proxy re-chosen for it), so /model can
        hop between configured providers in one action. Mutates in place — the loop passes this same llm
        object every turn, so the change applies from the next turn on. base_url semantics: None = keep
        the current endpoint; "" = the SDK's default endpoint (OpenAI). Resets the
        reasoning_effort+tools degrade memory (a different model/provider may support the pairing)."""
        if base_url is not None or api_key:
            import httpx
            from openai import OpenAI
            new_base = self._base_url if base_url is None else base_url
            new_key = api_key or getattr(self.client, "api_key", None)
            proxy = _choose_proxy(new_base, os.environ.get("AGENT_PROXY")
                                  or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
            use_proxy = bool(proxy) and proxy != "none"
            self.proxy_used = proxy if use_proxy else "direct"
            timeout = getattr(self, "_timeout", 60.0)
            http_client = (httpx.Client(proxy=proxy, timeout=timeout, trust_env=False) if use_proxy
                           else httpx.Client(timeout=timeout, trust_env=False))   # explicit proxy only (see __init__)
            ckw: dict = {"api_key": new_key}
            if new_base:
                ckw["base_url"] = new_base
            old_client = getattr(self, "client", None)
            self.client = OpenAI(http_client=http_client, timeout=timeout, max_retries=2, **ckw)
            if old_client is not None:       # close the replaced connection pool — /model hops leaked fds
                try:
                    old_client.close()
                except Exception:  # noqa: BLE001 — cleanup must never break the switch
                    pass
            self._base_url = new_base or ""
            self._drop_reasoning_effort = False
        if model:
            self.model = model
            self._drop_reasoning_effort = False
        if reasoning:
            self.reasoning = reasoning.strip().lower()

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
            except _import_api_timeout_error():
                raise   # the SIGALRM hard-deadline must not be swallowed by the sink wrapper
            except Exception:  # noqa: BLE001 — a render error must NEVER break the LLM call
                pass

    def is_retryable(self, error: Exception) -> bool:
        from .errors import EmptyResponseError
        try:
            from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
            openai_errors = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        except ImportError:
            openai_errors = ()
        try:
            import httpx
            transport = (httpx.TransportError,)   # a raw mid-STREAM drop (SDK doesn't wrap stream-iter errors) must retry like the blocking path
        except ImportError:
            transport = ()
        return isinstance(error, openai_errors + transport + (EmptyResponseError,))

    def _on_alarm(self, signum, frame):
        """SIGALRM handler: a request blew the HARD wall-clock deadline → raise a retryable timeout."""
        APITimeoutError = _import_api_timeout_error()
        try:
            import httpx
            raise APITimeoutError(request=httpx.Request("POST", (self._base_url or "http://local") + "/chat/completions"))
        except TypeError:
            # Older SDKs don't accept `request=` in the constructor.
            raise APITimeoutError("sliceagent hard timeout reached")

    def _create(self, kwargs: dict, caller=None):
        """Call the SDK with a HARD wall-clock deadline that ALWAYS fires, on ANY thread. The httpx/SDK
        read-timeout only bounds the gap BETWEEN bytes, so a connection that goes silent mid-response can
        hang far past `timeout` (observed: a stalled read wedging the loop 10+ min). On the main thread a
        SIGALRM deadline guarantees control returns to the retry path. OFF the main thread — e.g. a
        Terminal-Bench / any host ThreadPoolExecutor worker, where SIGALRM cannot arm — a watchdog thread
        enforces the SAME deadline (the abandoned SDK call is left to die on its socket while control
        returns). Without this, a wedged connection in a worker thread hangs the turn FOREVER, since the
        SDK timeout alone misses silent mid-response stalls. Task/provider-agnostic reliability."""
        caller = caller or (lambda kw: self.client.chat.completions.create(**kw))
        import signal as _signal
        try:
            prev = _signal.signal(_signal.SIGALRM, self._on_alarm)
            _signal.alarm(self._hard_timeout)
        except (ValueError, AttributeError, OSError):
            return self._create_watchdog(kwargs, caller)  # not the main thread → deadline via a thread
        try:
            return caller(kwargs)
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, prev)

    def _create_watchdog(self, kwargs: dict, caller=None):
        """Off-main-thread hard deadline: run the SDK call in a DAEMON worker and abandon it if it blows
        the wall-clock budget (raise a RETRYABLE timeout so with_retry can retry, then the loop parks
        gracefully instead of hanging). #47: a daemon thread (vs a ThreadPoolExecutor, whose worker the
        interpreter joins at exit) means a wedged call can NEVER block process shutdown — it dies with the
        socket whenever the SDK call finally errors on its own timeout. One thread per call; bounded."""
        import threading
        APITimeoutError = _import_api_timeout_error()
        caller = caller or (lambda kw: self.client.chat.completions.create(**kw))
        box: dict = {}

        def _call():
            try:
                box["resp"] = caller(kwargs)
            except BaseException as e:  # noqa: BLE001 — propagate to the caller thread
                box["err"] = e

        t = threading.Thread(target=_call, name="llm-watchdog", daemon=True)
        t.start()
        t.join(self._hard_timeout)
        if t.is_alive():   # blew the deadline — abandon the (daemon) thread, raise a retryable timeout
            try:
                import httpx
                raise APITimeoutError(
                    request=httpx.Request("POST", (self._base_url or "http://local") + "/chat/completions"))
            except TypeError:
                raise APITimeoutError("sliceagent hard timeout reached")
        if "err" in box:
            raise box["err"]
        return box["resp"]

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

        from .model_catalog import capability
        skw = {**kwargs, "stream": True}
        # #49: stream_options is OpenAI-specific — some OpenAI-compatible providers 400 on it. Gate by the
        # catalog flag (default True; set False for a provider that rejects it) so we still get the usage
        # chunk where supported without breaking the others.
        if capability(self.model, self._base_url).supports_stream_options:
            skw["stream_options"] = {"include_usage": True}
        parts: list[str] = []
        calls: dict[int, dict] = {}          # index → {id, name, args:[fragments]}
        finish = None
        usage = None
        _timeout_err = _import_api_timeout_error()   # the SIGALRM hard-deadline exception (must not be swallowed)
        # E3 streaming resilience: a single MALFORMED chunk is skipped (never aborts the whole stream); a
        # mid-stream CONNECTION error re-raises ONLY when nothing was assembled (so with_retry re-rolls) —
        # otherwise we salvage the partial as a truncated stop, which the loop handles cleanly.
        try:
            for chunk in self.client.chat.completions.create(**skw):
                try:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage       # final include_usage chunk (choices may be empty here)
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
                            _ix = getattr(tcd, "index", None)
                            if _ix is None:                       # provider omitted the streaming index
                                _tid = getattr(tcd, "id", None)
                                if _tid is not None:
                                    _ix = _tid                    # a NEW call, announced by its id → its own slot
                                elif calls:
                                    _ix = next(reversed(calls))   # continuation fragment → the OPEN (last) slot,
                                else:                             # NOT len(calls) (that split args into a dead slot)
                                    _ix = 0                       # first fragment before any id/index arrives
                            slot = calls.setdefault(_ix, {"id": None, "name": None, "args": []})
                            if getattr(tcd, "id", None):
                                slot["id"] = tcd.id
                            fn = getattr(tcd, "function", None)
                            if fn is not None:
                                if getattr(fn, "name", None):
                                    slot["name"] = fn.name
                                if getattr(fn, "arguments", None):
                                    slot["args"].append(fn.arguments)
                except _timeout_err:
                    raise   # SIGALRM hard-deadline fired mid-chunk → propagate (one-shot alarm won't re-arm); the outer handler salvages the partial
                except Exception:  # noqa: BLE001 — one bad chunk must not kill the stream
                    continue
        except Exception:  # noqa: BLE001 — stream broke mid-flight
            if not parts and not calls:
                raise                          # nothing salvageable → let with_retry re-roll
            finish = finish or "length"        # partial assembly → treat as a truncated (incomplete) stop
        # Drop any INCOMPLETE tool call (missing id or name) — a mid-stream break before a tool_call's
        # name/id delta arrived would otherwise yield a ToolCall(name=None) that breaks the dispatcher.
        # If this empties content AND tool_calls, complete() raises EmptyResponseError → with_retry re-rolls.
        tool_calls = [NS(id=c["id"], function=NS(name=c["name"], arguments="".join(c["args"])))
                      for _, c in sorted(calls.items(), key=lambda kv: kv[0] if isinstance(kv[0], int) else 0)
                      if c["id"] and c["name"]]   # robust sort: a None/str stream index must not crash assembly
        message = NS(content=("".join(parts) or None), tool_calls=tool_calls)
        return NS(choices=[NS(message=message, finish_reason=finish)], usage=usage)

    def _reasoning_kwargs(self) -> dict:
        """Map the provider-agnostic reasoning intent to the ACTIVE provider's knob; no-op (never error)
        for providers that have none. Keeps the quirk isolated to this adapter. Intents: fast→low,
        high→high, max→xhigh; "full" (default) = the provider's OWN default (deliberately NOT forced-high —
        forcing high would inflate tokens/cost on every turn against the moat; ask for "high"/"max" to
        opt into more reasoning)."""
        from .model_catalog import capability
        r = self.reasoning
        model, base = self.model.lower(), self._base_url.lower()
        if "openrouter" in base:
            # OpenRouter's UNIFIED reasoning object translates effort per upstream vendor (OpenAI
            # reasoning_effort / Anthropic thinking budget / Gemini thinkingLevel) and — unlike raw
            # reasoning_effort on chat/completions — works WITH tools. "full" stays provider-default.
            effort = {"fast": "low", "high": "high", "max": "high"}.get(r)
            return {"extra_body": {"reasoning": {"effort": effort}}} if effort else {}
        if "deepseek" in model or "deepseek" in base:
            return {"extra_body": {"thinking": {"type": "disabled"}}} if r == "fast" else {}
        if not capability(self.model, self._base_url).supports_reasoning_effort:
            return {}  # unknown / non-reasoning provider → leave at provider default (graceful)
        effort = {"fast": "low", "high": "high", "max": "xhigh"}.get(r)   # full/unknown → {} (default)
        return {"reasoning_effort": effort} if effort else {}

    def _cache_kwargs(self, messages: list[dict]) -> dict:
        """Map prompt-caching intent to the ACTIVE provider's knob; no-op for providers without
        one. Modeled on `_reasoning_kwargs` — the quirk stays isolated to this adapter.

        OpenAI-compatible providers (the default gpt-5.5 / Moonshot / DeepSeek path) cache automatically
        or via `prompt_cache_key` (see `_cache_routing_kwargs`), so this returns {} and leaves the request
        byte-stable. Claude/Anthropic-compatible endpoints instead cache via an explicit `cache_control`
        breakpoint on a message content BLOCK — not a top-level kwarg — so for those we mark the SYSTEM
        message (sliceagent's large byte-stable prefix) IN PLACE and return {}. Anthropic then reads the
        whole prefix from cache on every later same-prefix turn — the exact win the bounded slice sets up.
        (Shape per Anthropic's documented content-block cache_control spec; fires ONLY on a Claude endpoint,
        so the default DeepSeek/OpenAI path is byte-for-byte unchanged. Not yet run against a live Anthropic
        endpoint here — unit-tested for shape + gating.)"""
        model, base = self.model.lower(), self._base_url.lower()
        if "claude" not in model and "anthropic" not in base:
            return {}                                        # OpenAI-compatible → automatic / prompt_cache_key
        self._mark_cache_breakpoint(messages)                # Anthropic → content-block breakpoint (mutates in place)
        return {}

    @staticmethod
    def _mark_cache_breakpoint(messages: list[dict]) -> None:
        """Place ONE Anthropic ephemeral `cache_control` breakpoint at the end of the LAST system message
        (sliceagent's largest byte-stable span), converting its string content to a single text content
        block that carries the breakpoint. Anthropic caches everything up to and including the marked block,
        so the whole stable prefix is served from cache on later same-prefix turns. Idempotent (never
        double-marks); a no-op when there is no system message. Anthropic allows up to 4 breakpoints — one
        on the system prefix captures the bulk of the cacheable tokens; tools/older turns can add more later."""
        for m in reversed(messages):
            if m.get("role") != "system":
                continue
            c = m.get("content")
            if isinstance(c, str) and c:
                m["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
            elif isinstance(c, list) and c and isinstance(c[-1], dict) and "cache_control" not in c[-1]:
                c[-1] = {**c[-1], "cache_control": {"type": "ephemeral"}}
            return

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

    def _effort(self) -> str | None:
        """The Responses-API reasoning effort for THIS call ('low'/'high'/'xhigh'), or None when the
        intent is the provider default ('full') or the model has no effort knob. This is the routing key:
        gpt-5.5 REJECTS reasoning_effort + function tools on /v1/chat/completions, so any explicit effort
        goes through /v1/responses (which supports the pairing). Default 'full' → None → chat path."""
        from .model_catalog import capability
        if not capability(self.model, self._base_url).supports_reasoning_effort:
            return None
        return {"fast": "low", "high": "high", "max": "xhigh"}.get(self.reasoning)

    def _complete_responses(self, messages: list[dict], tools: list[dict], effort: str) -> AssistantMessage:
        """The /v1/responses path: lets the gpt-5 family reason at `effort` WITH function tools (the pairing
        chat/completions 400s on). Same AssistantMessage contract, same hard-deadline + live-streaming
        behaviour as the chat path. Isolated provider quirk — the loop/slice/moat never see it."""
        kwargs: dict = {"model": self.model, "input": _to_responses_input(messages),
                        "reasoning": {"effort": effort}}
        rtools = _to_responses_tools(tools)
        if rtools:
            kwargs["tools"] = rtools
            kwargs["tool_choice"] = "auto"
        if self.max_tokens:
            kwargs["max_output_tokens"] = self.max_tokens
        ck = self._cache_routing_kwargs()                    # prompt_cache_key is valid on Responses too
        if ck.get("prompt_cache_key"):
            kwargs["prompt_cache_key"] = ck["prompt_cache_key"]
        _stream = (getattr(self, "_on_delta", None) is not None
                   and threading.current_thread() is threading.main_thread())
        try:
            resp = (self._responses_stream(kwargs) if _stream
                    else self._create(kwargs, caller=lambda kw: self.client.responses.create(**kw)))
        except Exception as e:  # noqa: BLE001
            # route a provider context overflow into the SAME slice-tighten recovery the chat path uses
            # (llm.py chat except) — otherwise an overflow on the responses path crashes the turn instead.
            if is_context_overflow(e):
                raise ContextOverflow(e, status_code=getattr(e, "status_code", None)) from e
            raise
        return self._parse_responses(resp)

    def _responses_stream(self, kwargs: dict):
        """Stream a Responses call, emit content/reasoning deltas live, return the final Response (parsed
        downstream identically to the blocking path). Hard-deadline wrapped; on ANY stream hiccup it falls
        back to a single blocking call (a render path must never kill the turn)."""
        def _drain(kw):
            with self.client.responses.stream(**kw) as stream:
                for ev in stream:
                    try:
                        t = getattr(ev, "type", "")
                        if t == "response.output_text.delta":
                            self._emit("content", getattr(ev, "delta", "") or "")
                        elif t in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                            self._emit("reasoning", getattr(ev, "delta", "") or "")
                    except _import_api_timeout_error():
                        raise   # SIGALRM hard-deadline fired mid-event → propagate (one-shot alarm won't re-arm), mirroring the chat path
                    except Exception:  # noqa: BLE001 — one bad event must not abort the stream
                        continue
                return stream.get_final_response()
        try:
            return self._create(kwargs, caller=_drain)
        except Exception as e:  # noqa: BLE001 — streaming unavailable/broke → blocking call (identical result)
            # but NOT on a deterministic request-level failure (a hard-deadline timeout or a context
            # overflow): re-issuing the SAME request as a blocking call just doubles a guaranteed failure
            # (and overflow must reach _complete_responses' converter to drive recovery). Re-raise those;
            # only fall back for a genuine transport/streaming-unsupported hiccup.
            if isinstance(e, _import_api_timeout_error()) or is_context_overflow(e):
                raise
            return self._create(kwargs, caller=lambda kw: self.client.responses.create(**kw))

    def _parse_responses(self, resp) -> AssistantMessage:
        """Map a Responses Response → AssistantMessage (content / tool_calls / usage / finish_reason)."""
        content = (getattr(resp, "output_text", None) or "").strip() or None
        calls: list[ToolCall] = []
        for item in (getattr(resp, "output", None) or []):
            if getattr(item, "type", None) == "function_call":
                _name = getattr(item, "name", "") or ""
                if not _name:
                    continue                 # malformed function_call (no name) — skip, don't dispatch nameless
                try:
                    args = json.loads(getattr(item, "arguments", "") or "{}")
                except Exception:  # noqa: BLE001
                    args = {}
                calls.append(ToolCall(id=getattr(item, "call_id", "") or getattr(item, "id", ""),
                                      name=_name, args=args))
        status = getattr(resp, "status", None)               # finish_reason from Responses status
        reason = ""
        if status == "incomplete":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", "")
            finish = "length" if reason == "max_output_tokens" else ("content_filter" if reason == "content_filter" else "stop")
        else:
            finish = "tool_calls" if calls else "stop"
        # content_filter is a TERMINAL provider stop, not an empty-response hiccup: exempt it from the raise
        # (mirrors the chat path) so the loop PARKS it instead of re-rolling forever on a filtered completion.
        if not content and not calls and finish != "content_filter":
            from .errors import EmptyResponseError
            raise EmptyResponseError(f"empty responses completion (status={status})")
        return AssistantMessage(content=content, tool_calls=calls,
                                usage=_usage_dict(_responses_usage(getattr(resp, "usage", None))),
                                finish_reason=finish)

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        effort = self._effort()
        if effort and hasattr(self.client, "responses"):   # explicit effort → /v1/responses (chat 400s on
            return self._complete_responses(messages, tools, effort)   # effort+tools). No responses API on
        # an old SDK / a provider that only has chat → fall through; the chat 400→drop below degrades it.
        kwargs: dict = dict(model=self.model, messages=messages, tools=tools, tool_choice="auto")
        if self.max_tokens:
            # Provider quirk (now sourced from the model catalog — gpt-5/o-series renamed this param to
            # max_completion_tokens and REJECT max_tokens with a 400). One source of truth, not inline.
            from .model_catalog import capability
            kwargs[capability(self.model, self._base_url).tokens_param] = self.max_tokens
        self._merge_kwargs(kwargs, self._cache_routing_kwargs())  # session-stable cache routing (0 added tokens)
        self._merge_kwargs(kwargs, self._reasoning_kwargs())
        self._merge_kwargs(kwargs, self._cache_kwargs(messages))
        if "openrouter" in self._base_url.lower() and tools:
            # OpenRouter routes one model slug across MANY hosts; some serve quantizations with broken
            # tool calling, and unsupported params are DROPPED silently by default. require_parameters
            # pins routing to hosts that honor every param we sent — a silent degrade becomes a visible
            # error instead (the exact failure class behind the old reasoning-effort ceiling).
            self._merge_kwargs(kwargs, {"extra_body": {"provider": {"require_parameters": True}}})
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
        if not resp.choices:   # some OpenAI-compatible proxies emit {"choices": []} on filter/transient errors
            from .errors import EmptyResponseError
            raise EmptyResponseError("empty completion (no choices)")   # RETRYABLE → with_retry re-rolls (not a raw IndexError)
        choice = resp.choices[0]
        msg = choice.message
        if msg is None:                      # some proxies emit a choice with no message — retry, don't crash
            from .errors import EmptyResponseError
            raise EmptyResponseError(f"no message in completion (finish_reason={choice.finish_reason})")
        calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            fn = getattr(tc, "function", None)
            if fn is None or not getattr(fn, "name", None):
                continue                        # malformed tool_call (no function/name) — skip, don't crash
            try:
                args = json.loads(fn.arguments)
            except Exception:
                args = {}
            calls.append(ToolCall(id=getattr(tc, "id", "") or "", name=fn.name, args=args))
        # Degenerate completion — no content AND no tool calls (and not a content-filter stop). Some
        # providers/proxies occasionally emit an empty body; returning it stalls the loop, so raise a
        # RETRYABLE error (empty-response) and let with_retry re-roll. content_filter is
        # excluded — re-rolling would just filter again.
        if not (msg.content or "").strip() and not calls and choice.finish_reason != "content_filter":
            from .errors import EmptyResponseError
            raise EmptyResponseError(f"empty completion (finish_reason={choice.finish_reason})")
        usage = _usage_dict(resp.usage)
        return AssistantMessage(
            content=msg.content, tool_calls=calls, usage=usage, finish_reason=choice.finish_reason
        )
