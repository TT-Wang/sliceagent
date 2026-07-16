"""OpenAILLM — the default LLMClient over any OpenAI-COMPATIBLE endpoint (OpenAI, Moonshot,
DeepSeek, …). Configured by provider-AGNOSTIC env: LLM_API_KEY + LLM_BASE_URL (+ AGENT_MODEL);
OPENAI_*/MOONSHOT_* are accepted only as a back-compat fallback.

Connects directly by default; set AGENT_PROXY (or HTTPS_PROXY/HTTP_PROXY) to route through an HTTP
proxy, or AGENT_PROXY=none to force a direct connection. The only module that imports the openai SDK
— the core stays openai-free.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

from .context_overflow import ContextOverflow, is_context_overflow
from .errors import (
    ImmediateRetryError,
    IndeterminateModelCallError,
    PreFirstByteTimeoutError,
    ProviderCapacityError,
    RetryCancelledError,
    TransportStartupError,
)
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


def _is_transport_timeout(error: BaseException) -> bool:
    """Recognize SDK-wrapped and raw async-stream timeout shapes.

    OpenAI-compatible SDKs do not consistently wrap errors raised while iterating an already-open SSE body;
    httpx.ReadTimeout may therefore escape where request creation would raise APITimeoutError.
    """
    if isinstance(error, (_import_api_timeout_error(), TimeoutError)):
        return True
    try:
        import httpx
        return isinstance(error, httpx.TimeoutException)
    except ImportError:
        return False


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


def _proxy_route_for_display(proxy: str | None) -> str:
    """Return a credential-free route label safe for terminal/status output.

    The raw URL remains private transport configuration. Besides userinfo, omit path, query, and fragment:
    proxy services sometimes carry credentials in any of those components. Malformed values get an opaque
    label rather than falling back to the unsafe input.
    """
    if not proxy or str(proxy).strip().lower() in {"none", "off", "direct"}:
        return "direct"
    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(str(proxy).strip())
        scheme = parsed.scheme.lower()
        host = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return "configured proxy"
    if not scheme or not host:
        return "configured proxy"
    # urlsplit removes IPv6 brackets from ``hostname``; restore them so the label stays unambiguous.
    display_host = f"[{host}]" if ":" in host else host
    return f"{scheme}://{display_host}{f':{port}' if port is not None else ''}"


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


def _positive_deadline(raw) -> int | None:
    """Parse an absolute watchdog override in whole seconds; invalid/non-positive means "use policy"."""
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if not (seconds > 0):  # also rejects NaN
        return None
    try:
        return max(1, int(seconds))
    except OverflowError:  # +inf is not a usable alarm duration
        return None


def _default_hard_timeout(
    timeout: float, model: str, base_url: str, completion_tokens: int = 8192,
) -> int:
    """Return a completion-budget-compatible absolute whole-call deadline.

    The SDK timeout still detects an idle socket/read gap.  This independent ceiling bounds a stream that is
    actively producing bytes, so it cannot be shorter than the completion budget it explicitly permits.  The
    conservative 32-token/s floor plus 30 seconds of first-byte/teardown room covers reasoning providers
    without giving one vendor a hidden exception. ``LLM_HARD_TIMEOUT_SEC`` remains the operator override.
    """
    transport_floor = max(int(timeout) + 15, 30)
    try:
        budget = max(0, int(completion_tokens))
    except (TypeError, ValueError):
        budget = 8192
    generation_floor = ((budget + 31) // 32 + 30) if budget else 180
    return max(180, transport_floor, generation_floor)


class _ProviderCapacityTimeout(Exception):
    """No physical provider slot became available before the local admission deadline."""


class _UnconfirmedStreamClose(Exception):
    """A provider stream's explicit close path raised; its physical socket may still be live."""


class _PhysicalCallLease:
    """Idempotent lease whose lifetime is the physical request, not its logical caller."""

    def __init__(self, gate: "_PhysicalCallGate") -> None:
        self._gate = gate

    def release(self) -> None:
        # The gate owns idempotence by this lease's identity.  That makes retirement safe even if an
        # asynchronous KeyboardInterrupt lands before or after the set removal: a retry either performs the
        # missing removal or observes that it already committed. A local boolean cannot distinguish those two.
        self._gate._release(self)


class _PhysicalCallGate:
    """Bound actual provider work, including calls whose logical watchdog already returned.

    Logical worker counts are insufficient here: Python cannot kill an abandoned blocking SDK thread, and an
    async transport can ignore task cancellation.  The worker/stream owns this lease until its physical
    ``finally`` runs, so a later logical attempt cannot exceed the configured provider concurrency ceiling.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._condition = threading.Condition()
        self._leases: set[_PhysicalCallLease] = set()

    @staticmethod
    def _cancelled(should_cancel) -> bool:
        if should_cancel is None:
            return False
        try:
            return bool(should_cancel())
        except Exception:  # noqa: BLE001 - cancellation observation is fail-open
            return False

    def acquire(self, *, timeout: float | None, should_cancel=None) -> _PhysicalCallLease:
        return self.acquire_lease(
            self.new_lease(), timeout=timeout, should_cancel=should_cancel,
        )

    def new_lease(self) -> _PhysicalCallLease:
        """Create an inactive caller-owned lease before the interruptible admission handoff."""
        return _PhysicalCallLease(self)

    def acquire_lease(
        self, lease: _PhysicalCallLease, *, timeout: float | None, should_cancel=None, on_wait=None,
    ) -> _PhysicalCallLease:
        """Admit a lease whose identity is already stored by the caller.

        Production paths use this form so an asynchronous exception after this method returns but before the
        next caller bytecode cannot strand the only reference inside the gate's active set.
        """
        import time

        if not isinstance(lease, _PhysicalCallLease) or lease._gate is not self:
            raise ValueError("provider lease belongs to a different physical call gate")
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        wait_reported = False
        try:
            while True:
                report_wait = False
                wait_state = None
                with self._condition:
                    if lease in self._leases:
                        raise RuntimeError("provider lease is already active")
                    if len(self._leases) < self.capacity:
                        if self._cancelled(should_cancel):
                            raise RetryCancelledError("model call cancelled before provider capacity admission")
                        self._leases.add(lease)
                        return lease
                    if self._cancelled(should_cancel):
                        raise RetryCancelledError("model call cancelled while waiting for provider capacity")
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise _ProviderCapacityTimeout("provider call capacity remained occupied")
                    # Publish the queue transition outside the condition lock: presentation/metrics are
                    # advisory and must never delay retirement of another physical request that needs this
                    # same lock in order to free capacity.
                    if not wait_reported:
                        wait_reported = True
                        report_wait = True
                        wait_state = (len(self._leases), self.capacity)
                    else:
                        self._condition.wait(0.05 if remaining is None else min(0.05, remaining))
                if report_wait:
                    try:
                        if on_wait is not None:
                            on_wait(*wait_state)
                    except Exception:  # noqa: BLE001 - admission observation is fail-open
                        pass
        except BaseException:  # noqa: BLE001 - includes async interruption during the lease handoff
            # Identity-based retirement is safe whether interruption landed just before or just after add().
            lease.release()
            raise

    def _release(self, lease: _PhysicalCallLease) -> None:
        with self._condition:
            if lease not in self._leases:
                return
            self._leases.remove(lease)
            self._condition.notify()

    @property
    def active(self) -> int:
        """Test/diagnostic view of physical calls currently holding provider capacity."""
        with self._condition:
            return len(self._leases)


_PROVIDER_CALL_GATES: dict[tuple[str, bytes], _PhysicalCallGate] = {}
_PROVIDER_CALL_GATES_LOCK = threading.Lock()


def _provider_call_capacity() -> int:
    try:
        capacity = int(os.environ.get("LLM_PROVIDER_MAX_INFLIGHT") or 4)
    except (TypeError, ValueError):
        return 4
    return capacity if capacity > 0 else 4


def _provider_call_gate(spec: tuple) -> _PhysicalCallGate:
    """Return the process-wide gate for one provider/account without retaining its credential in the key."""
    from urllib.parse import urlsplit, urlunsplit

    api_key, base_url, _proxy, _timeout = spec
    credential = hashlib.sha256(str(api_key or "").encode("utf-8", errors="replace")).digest()[:12]
    capacity = _provider_call_capacity()
    raw_url = str(base_url or "https://api.openai.com/v1").strip()
    parsed = urlsplit(raw_url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port is not None and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/")
    if host in {"api.openai.com", "api.deepseek.com", "api.moonshot.cn"} and path in {"", "/v1"}:
        path = "/v1"
    provider = urlunsplit((scheme, host, path, parsed.query, ""))
    key = (provider, credential)
    with _PROVIDER_CALL_GATES_LOCK:
        gate = _PROVIDER_CALL_GATES.get(key)
        if gate is None:
            gate = _PhysicalCallGate(capacity)
            _PROVIDER_CALL_GATES[key] = gate
        return gate


class _TransportAdmission:
    """Linearize async provider admission against cancellation/fallback timeout.

    An Event saying the coroutine has not entered yet is only a stale observation: a blocked event loop may
    enter it after the caller returns.  This gate makes the decision atomic.  Either cancellation changes
    ``pending`` to ``cancelled`` (after which transport can never start), or the coroutine changes it to
    ``admitted`` and every caller must wait for the physical ``closed`` acknowledgement.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "pending"
        self.admitted = threading.Event()
        self.closed = threading.Event()

    def try_admit(self) -> bool:
        with self._lock:
            if self._state != "pending":
                return False
            self._state = "admitted"
            self.admitted.set()
            return True

    def cancel_before_admission(self) -> bool:
        with self._lock:
            if self._state != "pending":
                return False
            self._state = "cancelled"
            # No physical call exists or can subsequently exist; closure is proven by the gate itself.
            self.closed.set()
            return True

    def mark_closed(self) -> None:
        with self._lock:
            self._state = "closed"
            self.closed.set()

    @property
    def was_admitted(self) -> bool:
        return self.admitted.is_set()


class _SubmissionHandoff:
    """Keep a submitted coroutine inert until its synchronous owner stores the returned Future.

    ``run_coroutine_threadsafe`` can schedule the task before its return value reaches the caller's assignment
    bytecode. If that handoff is interrupted, cleanup has no Future to cancel. This one-shot decision is owned
    before submission: confirmation lets the coroutine approach provider admission; cancellation makes it
    retire without opening a request.
    """

    def __init__(self) -> None:
        import concurrent.futures

        self._lock = threading.Lock()
        self._decision = concurrent.futures.Future()

    def _decide(self, value: bool) -> bool:
        with self._lock:
            if self._decision.done():
                return False
            self._decision.set_result(bool(value))
            return True

    def confirm_owner(self) -> bool:
        return self._decide(True)

    def cancel_before_owner(self) -> bool:
        return self._decide(False)

    async def wait(self) -> bool:
        import asyncio

        return bool(await asyncio.wrap_future(self._decision))

    def wait_sync(self) -> bool:
        return bool(self._decision.result())


class _AsyncTransportHub:
    """One cancellable async transport loop shared by synchronous callers.

    ``OpenAILLM.complete`` is intentionally synchronous, while an off-main child still needs a stream whose
    blocked ``anext()`` can be cancelled.  A monotonic check *inside* a normal ``for chunk in stream`` cannot
    run while ``next()`` is blocked; a daemon watchdog around a synchronous request can return, but cannot
    prove that the socket stopped.  This bridge keeps all async clients on one event loop, drains SSE there,
    and lets the calling thread observe chunks through a queue.  ``asyncio.timeout``/task cancellation can
    interrupt the pending async read and the stream context closes the response before an error is returned.

    The sync side also has a bounded close-confirmation path.  If an SDK/transport ignores cancellation, the
    result is ``IndeterminateModelCallError`` (non-retryable) rather than a retry that overlaps a live socket.
    """

    def __init__(self) -> None:
        self._start_lock = threading.Lock()
        self._start_condition = threading.Condition(self._start_lock)
        self._starting = False
        self._start_generation = 0
        self._start_error: BaseException | None = None
        self._loop_ready = False
        self._loop = None
        self._clients: dict[tuple, object] = {}

    def _ensure_started(self, *, timeout: float | None = None, should_cancel=None) -> None:
        """Start exactly one transport loop and wait for its success/failure handshake.

        The start lock cannot be released merely because ``Thread.start`` returned: the loop has not published
        readiness yet, and concurrent first callers would each start another loop. ``_starting`` owns that cold
        generation until ``_loop_main`` reports success or failure through the condition. A failed generation is
        surfaced to its waiters; a later call may safely create a fresh loop/client generation.
        """
        import time

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._start_condition:
            if self._loop_ready and self._loop is not None:
                return
            if not self._starting:
                self._starting = True
                self._start_generation += 1
                self._start_error = None
                generation = self._start_generation
                try:
                    thread = threading.Thread(
                        target=self._loop_main,
                        name="llm-stream-transport",
                        daemon=True,
                    )
                    thread.start()
                except BaseException as error:  # noqa: BLE001 - publish failed admission to every waiter
                    self._starting = False
                    self._start_error = error
                    self._start_condition.notify_all()
                    raise RuntimeError("could not start the LLM stream transport thread") from error
            else:
                generation = self._start_generation

            while self._starting and self._start_generation == generation:
                if self._cancelled(should_cancel):
                    raise RetryCancelledError(
                        "model call cancelled while the local stream transport was starting"
                    )
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TransportStartupError(
                        "the local stream transport did not become ready before the model-call deadline; "
                        "no provider request was started"
                    )
                self._start_condition.wait(
                    0.05 if remaining is None else min(0.05, remaining)
                )
            if self._loop_ready and self._loop is not None:
                return
            error = self._start_error
            raise RuntimeError("LLM stream transport loop failed during startup") from error

    def _new_event_loop(self):
        """Test seam for deterministic startup failure/race coverage."""
        import asyncio

        return asyncio.new_event_loop()

    def _loop_main(self) -> None:
        import asyncio

        loop = None
        try:
            loop = self._new_event_loop()
            asyncio.set_event_loop(loop)
            with self._start_condition:
                # Async clients are loop-affine. Never carry a pool across a recovered loop generation.
                self._clients = {}
                self._loop = loop
                self._loop_ready = True
                self._starting = False
                self._start_error = None
                self._start_condition.notify_all()
            loop.run_forever()
            raise RuntimeError("LLM stream transport loop stopped unexpectedly")
        except BaseException as error:  # noqa: BLE001 - publish startup/runtime failure for safe recovery
            with self._start_condition:
                if self._loop is loop:
                    self._loop = None
                self._loop_ready = False
                self._starting = False
                self._start_error = error
                self._clients = {}
                self._start_condition.notify_all()
        finally:
            if loop is not None and not loop.is_running():
                try:
                    loop.close()
                except Exception:  # noqa: BLE001 - daemon cleanup only
                    pass

    def _client_for(self, spec: tuple):
        client = self._clients.get(spec)
        if client is not None:
            return client

        import httpx
        from openai import AsyncOpenAI

        api_key, base_url, proxy, timeout = spec
        use_proxy = bool(proxy) and proxy != "none"
        http_client = (
            httpx.AsyncClient(proxy=proxy, timeout=timeout, trust_env=False)
            if use_proxy else
            httpx.AsyncClient(timeout=timeout, trust_env=False)
        )
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncOpenAI(
            http_client=http_client,
            timeout=timeout,
            max_retries=0,
            **kwargs,
        )
        self._clients[spec] = client
        return client

    @staticmethod
    async def _drain_context(manager, publish, *, finalizer=None):
        """Drain an async provider stream and distinguish body failure from close failure.

        Catching the body exception *inside* the context lets us observe whether ``__aexit__`` itself
        completed.  Only a successful exit proves socket retirement.  If exit raises, the caller must retain
        its physical provider lease and surface an indeterminate call instead of overlapping a retry.
        """
        stream = await manager.__aenter__()
        body_error = None
        body_traceback = None
        result = None
        try:
            async for item in stream:
                publish(item)
            if finalizer is not None:
                result = await finalizer(stream)
        except BaseException as error:  # noqa: BLE001 - cancellation also requires an explicit close
            body_error = error
            body_traceback = error.__traceback__
        try:
            suppressed = await manager.__aexit__(
                type(body_error) if body_error is not None else None,
                body_error,
                body_traceback,
            )
        except BaseException as close_error:  # noqa: BLE001 - physical closure is now unproven
            raise _UnconfirmedStreamClose(
                "provider stream teardown failed; physical closure was not confirmed"
            ) from close_error
        if body_error is not None and not suppressed:
            raise body_error.with_traceback(body_traceback)
        return result

    async def _chat(self, spec: tuple, kwargs: dict, publish) -> None:
        client = self._client_for(spec)
        stream = await client.chat.completions.create(**kwargs)
        await self._drain_context(stream, publish)

    async def _responses(self, spec: tuple, kwargs: dict, publish):
        client = self._client_for(spec)
        return await self._drain_context(
            client.responses.stream(**kwargs), publish,
            finalizer=lambda stream: stream.get_final_response(),
        )

    @staticmethod
    def _cancelled(should_cancel) -> bool:
        if should_cancel is None:
            return False
        try:
            return bool(should_cancel())
        except Exception:  # noqa: BLE001 - cancellation observation is fail-open
            return False

    def run(
        self,
        kind: str,
        spec: tuple,
        kwargs: dict,
        *,
        timeout: float,
        close_grace: float,
        should_cancel=None,
        on_item=None,
        on_activity=None,
        heartbeat_interval: float = 5.0,
        provider_gate: _PhysicalCallGate | None = None,
    ):
        """Run one physical streaming request and return only after its stream is closed.

        ``on_item`` executes on the synchronous caller thread, which keeps the Rich/TUI sink isolated from
        the transport loop.  Cancellation is polled by that caller while the async task is independently
        blocked in network I/O; cancelling the future therefore interrupts the blocked read rather than
        waiting for another chunk merely to notice a monotonic deadline.
        """
        import asyncio
        import concurrent.futures
        import queue
        import time

        run_started = time.monotonic()
        request_deadline = run_started + max(0.001, float(timeout))
        try:
            heartbeat_every = max(0.05, float(heartbeat_interval))
        except (TypeError, ValueError, OverflowError):
            heartbeat_every = 5.0

        def publish_activity(event: str, **detail) -> None:
            """Publish metadata only; a broken observer cannot perturb transport ownership."""
            if on_activity is None:
                return
            try:
                on_activity(event, detail)
            except Exception:  # noqa: BLE001 - diagnostics/presentation are fail-open
                pass

        if self._cancelled(should_cancel):
            raise RetryCancelledError("model call cancelled before the provider request started")
        self._ensure_started(
            timeout=max(0.0, request_deadline - time.monotonic()),
            should_cancel=should_cancel,
        )
        items: queue.Queue = queue.Queue()
        admission = _TransportAdmission()
        submission = _SubmissionHandoff()
        # The async timeout owns the actual request deadline; this caller-side fallback gives that task one
        # close-confirmation grace before force-cancelling it. Startup and provider-capacity wait consume the
        # same absolute request budget rather than silently resetting it.
        fallback_deadline = request_deadline + max(0.05, close_grace)
        lease = None
        future = None
        provider_wait_started = time.monotonic()
        provider_admitted_at = None
        provider_queued = False
        first_item_at = None
        last_item_at = None
        item_count = 0
        last_heartbeat_at = None

        class _Cutoff(Exception):
            def __init__(self, owner_cancelled: bool):
                super().__init__("owner cancellation" if owner_cancelled else "transport deadline")
                self.owner_cancelled = owner_cancelled

        def close_physical(error: BaseException) -> bool:
            """Retire this owner's admission and prove closure, including every handoff bytecode edge.

            Returns true when no provider request ever started. If admission already won, the physical stream
            must acknowledge closure before the logical caller may return; otherwise uncertainty replaces the
            triggering exception. The lease is idempotent because both the caller and coroutine may observe a
            pre-admission cutoff.
            """
            submission.cancel_before_owner()
            cancelled_before_admission = admission.cancel_before_admission()
            if future is not None:
                future.cancel()
            if cancelled_before_admission:
                if lease is not None:
                    lease.release()
                return True
            if admission.closed.is_set():
                return False
            if not admission.closed.wait(max(0.05, close_grace)):
                # A teardown failure can settle the task without proving socket closure. We deliberately keep
                # its provider lease quarantined, but still observe the task's terminal exception so asyncio
                # does not emit a misleading "Task exception was never retrieved" diagnostic later.
                if future is not None:
                    def observe_terminal(done):
                        try:
                            done.exception()
                        except BaseException:  # noqa: BLE001 - observation only, including cancellation
                            pass
                    future.add_done_callback(observe_terminal)
                raise IndeterminateModelCallError(
                    "the logical model caller stopped but physical connection closure was not confirmed; "
                    "automatic retry was suppressed"
                ) from error
            return False

        # One ownership scope begins before capacity acquisition and ends only after physical closure. This
        # deliberately covers asynchronous exceptions during gate acquire, coroutine construction/submission,
        # helper setup, item delivery, and result assembly—not only the obvious queue-drain loop.
        try:
            if time.monotonic() >= request_deadline:
                raise TransportStartupError(
                    "local stream transport startup consumed the model-call deadline; "
                    "no provider request was started"
                )
            if provider_gate is not None:
                # Store the inactive identity before the interruptible admission call. If an exception lands
                # after acquire_lease returns but before its caller resumes, the outer ownership scope can
                # still retire exactly this lease.
                lease = provider_gate.new_lease()
                def report_capacity_wait(active: int, capacity: int) -> None:
                    nonlocal provider_queued
                    provider_queued = True
                    publish_activity(
                        "provider_queue",
                        queue_ms=max(0, int((time.monotonic() - provider_wait_started) * 1000)),
                        active=active,
                        capacity=capacity,
                    )
                try:
                    provider_gate.acquire_lease(
                        lease,
                        timeout=max(0.0, request_deadline - time.monotonic()),
                        should_cancel=should_cancel,
                        on_wait=report_capacity_wait,
                    )
                except _ProviderCapacityTimeout as error:
                    raise ProviderCapacityError(
                        "provider capacity remained occupied until this call's admission deadline; "
                        "no new request was started"
                    ) from error
            if time.monotonic() >= request_deadline:
                raise ProviderCapacityError(
                    "provider admission consumed the model-call deadline; no request was started"
                )
            provider_admitted_at = time.monotonic()
            last_heartbeat_at = provider_admitted_at
            publish_activity(
                "provider_admitted",
                queue_ms=max(0, int((provider_admitted_at - provider_wait_started) * 1000)),
                queued=provider_queued,
                remaining_ms=max(0, int((request_deadline - provider_admitted_at) * 1000)),
            )

            async def drive():
                if not await submission.wait():
                    admission.cancel_before_admission()
                    if lease is not None:
                        lease.release()
                    return None
                if time.monotonic() >= request_deadline:
                    admission.cancel_before_admission()
                    if lease is not None:
                        lease.release()
                    return TransportStartupError(
                        "local stream scheduling consumed the model-call deadline; "
                        "no provider request was started"
                    )
                if not admission.try_admit():
                    # Cancellation won before this coroutine ran; the caller and coroutine may both release.
                    if lease is not None:
                        lease.release()
                    return None
                closure_confirmed = False
                try:
                    # Admission itself is an interruptible/contended boundary. Recheck after it linearizes;
                    # never turn an already-expired budget into a fresh 1ms provider request.
                    remaining = request_deadline - time.monotonic()
                    if remaining <= 0:
                        admission.mark_closed()
                        if lease is not None:
                            lease.release()
                        return TransportStartupError(
                            "provider admission crossed the model-call deadline; no request was started"
                        )
                    async with asyncio.timeout(remaining):
                        # Capture provider-delivery time on the transport loop. The synchronous caller may be
                        # briefly busy rendering the previous item; TTFT/idle metrics should not include that
                        # local presentation delay.
                        def publish_item(item) -> None:
                            items.put((time.monotonic(), item))
                        if kind == "chat":
                            result = await self._chat(spec, kwargs, publish_item)
                        elif kind == "responses":
                            result = await self._responses(spec, kwargs, publish_item)
                        else:
                            raise ValueError(f"unknown stream transport kind: {kind}")
                    closure_confirmed = True
                    return result
                except _UnconfirmedStreamClose as error:
                    # The stream context's teardown path itself failed.  Do not acknowledge closure or free
                    # provider capacity: a later request could otherwise overlap the still-live socket. Return
                    # the marker as data so a concurrently-cancelled wrapper future cannot leave an unobserved
                    # asyncio Task exception; the synchronous owner still converts it to INDETERMINATE below.
                    return error
                except BaseException:  # request creation/body returned or cancelled after a proven close
                    closure_confirmed = True
                    raise
                finally:
                    if closure_confirmed:
                        admission.mark_closed()
                        if lease is not None:
                            lease.release()

            future = asyncio.run_coroutine_threadsafe(drive(), self._loop)
            # This call is deliberately after the Future assignment. Any exception in the return→assignment
            # gap reaches the outer owner with submission still false, so the queued coroutine cannot admit.
            submission.confirm_owner()

            def consume_queued_item(record) -> None:
                nonlocal first_item_at, last_item_at, item_count, last_heartbeat_at
                observed_at, item = record
                item_count += 1
                last_item_at = observed_at
                if first_item_at is None:
                    first_item_at = observed_at
                    last_heartbeat_at = observed_at
                    publish_activity(
                        "first_byte",
                        queue_ms=max(0, int((provider_admitted_at - provider_wait_started) * 1000)),
                        ttfb_ms=max(0, int((first_item_at - provider_admitted_at) * 1000)),
                        elapsed_ms=max(0, int((first_item_at - run_started) * 1000)),
                    )
                if on_item is not None:
                    on_item(item)

            def publish_heartbeat(now: float) -> None:
                nonlocal last_heartbeat_at
                if provider_admitted_at is None or last_heartbeat_at is None \
                        or now - last_heartbeat_at < heartbeat_every:
                    return
                last_heartbeat_at = now
                reference = last_item_at if last_item_at is not None else provider_admitted_at
                detail = {
                    "state": "receiving" if first_item_at is not None else "awaiting_first_byte",
                    "elapsed_ms": max(0, int((now - run_started) * 1000)),
                    "idle_ms": max(0, int((now - reference) * 1000)),
                    "chunks": item_count,
                }
                if first_item_at is not None:
                    detail["ttfb_ms"] = max(0, int((first_item_at - provider_admitted_at) * 1000))
                publish_activity("stream_heartbeat", **detail)

            while not future.done():
                if self._cancelled(should_cancel):
                    raise _Cutoff(True)
                now = time.monotonic()
                remaining = fallback_deadline - now
                if remaining <= 0:
                    raise _Cutoff(False)
                try:
                    record = items.get(timeout=min(0.05, max(0.001, remaining)))
                except queue.Empty:
                    publish_heartbeat(time.monotonic())
                    continue
                consume_queued_item(record)
                publish_heartbeat(time.monotonic())

            # A coroutine can publish its final item immediately before becoming done. Drain that FIFO before
            # returning so response assembly is byte-equivalent to consuming the async iterator directly.
            while True:
                try:
                    record = items.get_nowait()
                except queue.Empty:
                    break
                consume_queued_item(record)

            try:
                result = future.result()
                if isinstance(result, TransportStartupError):
                    raise result
                if isinstance(result, _UnconfirmedStreamClose):
                    # This deliberately raises IndeterminateModelCallError after the close grace while keeping
                    # the physical lease quarantined.
                    close_physical(result)
                return result
            except concurrent.futures.CancelledError as error:
                never_started = close_physical(error)
                detail = ("before provider admission" if never_started else "after physical closure")
                raise RetryCancelledError(f"model transport task was cancelled {detail}") from error
            except TimeoutError as error:
                # asyncio.timeout surfaces only after drive()'s finally unwound the stream context.
                if not admission.closed.is_set():
                    raise IndeterminateModelCallError(
                        "stream deadline elapsed without confirmed connection closure; "
                        "automatic retry was suppressed"
                    ) from error
                raise self._timeout_error(spec) from error
        except _Cutoff as cutoff:
            never_started = close_physical(cutoff)
            if cutoff.owner_cancelled:
                raise RetryCancelledError("model call cancelled by the owning turn") from cutoff
            if never_started:
                raise TransportStartupError(
                    "the local stream transport missed the model-call deadline before provider admission; "
                    "no request was started"
                ) from cutoff
            raise self._timeout_error(spec) from cutoff
        except BaseException as error:  # noqa: BLE001 - owns parser errors, Ctrl-C, and handoff interruption
            close_physical(error)
            raise

    @staticmethod
    def _timeout_error(spec: tuple):
        APITimeoutError = _import_api_timeout_error()
        _api_key, base_url, _proxy, _timeout = spec
        try:
            import httpx
            return APITimeoutError(
                request=httpx.Request("POST", (base_url or "http://local") + "/chat/completions")
            )
        except TypeError:
            return APITimeoutError("sliceagent stream hard timeout reached")


_ASYNC_TRANSPORT_HUB: _AsyncTransportHub | None = None
_ASYNC_TRANSPORT_LOCK = threading.Lock()


def _async_transport_hub() -> _AsyncTransportHub:
    global _ASYNC_TRANSPORT_HUB
    if _ASYNC_TRANSPORT_HUB is None:
        with _ASYNC_TRANSPORT_LOCK:
            if _ASYNC_TRANSPORT_HUB is None:
                _ASYNC_TRANSPORT_HUB = _AsyncTransportHub()
    return _ASYNC_TRANSPORT_HUB


def _consume_stream_item(consumer, item, timeout_error_type) -> None:
    """Keep malformed provider chunks observational while never swallowing deadline/control failures."""
    try:
        consumer(item)
    except (timeout_error_type, RetryCancelledError, IndeterminateModelCallError):
        raise
    except Exception:  # noqa: BLE001 - one malformed chunk must not abort an otherwise healthy stream
        pass


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
        # Public status only: never let proxy credentials enter startup output, /model, logs, or screenshots.
        # The raw URL is retained below solely by the HTTP client and private transport spec.
        self.proxy_used = _proxy_route_for_display(proxy)
        # trust_env=False: the proxy is resolved EXPLICITLY above (AGENT_PROXY/HTTPS_PROXY/HTTP_PROXY →
        # _choose_proxy), so httpx must NOT ALSO auto-read ambient proxy env — otherwise AGENT_PROXY=none
        # ("force direct") still routes through an ambient HTTPS_PROXY (httpx defaults trust_env=True).
        http_client = (httpx.Client(proxy=proxy, timeout=timeout, trust_env=False) if use_proxy
                       else httpx.Client(timeout=timeout, trust_env=False))

        # SliceAgent owns retries at ``complete_model_call`` so every physical provider attempt has one
        # ModelCallPrepared/ApiRetry lifecycle.  Leaving the SDK default (2) enabled nested three hidden
        # requests inside each visible attempt; in a child watchdog an abandoned SDK retry bundle could also
        # overlap the next app-level attempt.  Keep the SDK timeout, but make this transport strictly one-shot.
        self.client = OpenAI(http_client=http_client, timeout=timeout, max_retries=0, **kwargs)
        self._timeout = timeout                              # kept for live endpoint switches (switch())
        # Immutable connection identity used by the shared async streaming hub. A shallow child LLM view
        # inherits this tuple safely; a live provider switch replaces (never mutates) it on that one view.
        self._transport_spec = (
            kwargs.get("api_key"), kwargs.get("base_url") or "",
            proxy if use_proxy else "none", timeout,
        )
        # Shared by shallow child views and every adapter instance targeting the same provider/account.  A
        # timed-out blocking worker or cancellation-resistant stream keeps its lease until physical closure.
        self._provider_call_gate = _provider_call_gate(self._transport_spec)
        # No built-in default model — the user picks (parallels the CLI's model gate; a silent
        # fallback here would contradict it for library/embedding callers).
        self.model = model or os.environ.get("AGENT_MODEL") or ""
        if not self.model:
            raise ValueError("No model configured. Pass model=... or set AGENT_MODEL "
                             "(interactive setup: `sliceagent init`).")
        self._base_url = kwargs.get("base_url") or ""
        # The SDK timeout above bounds idle/read gaps; this is a separately configurable ABSOLUTE
        # whole-request watchdog. Keep a parsed override on the object so shallow child LLM views inherit
        # the exact policy and live provider switches cannot accidentally discard an operator setting.
        self._hard_timeout_override = _positive_deadline(os.environ.get("LLM_HARD_TIMEOUT_SEC"))
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
        # The absolute stream ceiling and completion cap must be coherent. Derive it only after parsing the
        # cap so an allowed 8k-token reasoning response cannot be killed by a shorter historical wall timer.
        self._refresh_hard_timeout()
        # Provider-AGNOSTIC prompt-cache routing key (OpenAI `prompt_cache_key`, accepted/ignored
        # harmlessly elsewhere). A session-stable key keeps every turn's requests on the same cached
        # prefix → higher cache-hit rate at ZERO added prompt tokens. Set via set_cache_key(); the
        # quirk stays isolated to this adapter (llm-agnostic). None → omit the kwarg entirely.
        self._cache_key: str | None = None
        # Streaming is a TRANSPORT property, independent of rendering. Every call streams and assembles by
        # default, including off-main children with no UI sink; this keeps the provider read active and gives
        # async cancellation a physical stream to close. `_on_delta` remains an optional presentation sink.
        self._stream_transport_enabled = True
        self._on_delta = None
        # Optional low-rate transport lifecycle observer. It is deliberately separate from token rendering so
        # a child can report "reasoning"/"writing" without leaking its private deltas into the parent TUI.
        self._transport_activity = None
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
            self.proxy_used = _proxy_route_for_display(proxy)
            timeout = getattr(self, "_timeout", 60.0)
            http_client = (httpx.Client(proxy=proxy, timeout=timeout, trust_env=False) if use_proxy
                           else httpx.Client(timeout=timeout, trust_env=False))   # explicit proxy only (see __init__)
            ckw: dict = {"api_key": new_key}
            if new_base:
                ckw["base_url"] = new_base
            old_client = getattr(self, "client", None)
            # Keep live provider switches on the same single retry boundary as construction.  Otherwise a
            # /model hop silently restores nested SDK retries and makes child latency provider-history dependent.
            self.client = OpenAI(http_client=http_client, timeout=timeout, max_retries=0, **ckw)
            if old_client is not None:       # close the replaced connection pool — /model hops leaked fds
                try:
                    old_client.close()
                except Exception:  # noqa: BLE001 — cleanup must never break the switch
                    pass
            self._base_url = new_base or ""
            self._transport_spec = (
                new_key, self._base_url, proxy if use_proxy else "none", timeout,
            )
            self._provider_call_gate = _provider_call_gate(self._transport_spec)
            self._drop_reasoning_effort = False
        if model:
            self.model = model
            self._drop_reasoning_effort = False
        if reasoning:
            self.reasoning = reasoning.strip().lower()
        # Recompute after BOTH endpoint and model updates: model-only switches are common, and a shallow
        # child view may switch its own model without mutating the parent's watchdog policy.
        self._refresh_hard_timeout()

    def _refresh_hard_timeout(self) -> None:
        """Apply the explicit deadline, or derive one from transport and completion budgets."""
        override = getattr(self, "_hard_timeout_override", None)
        self._hard_timeout = (override if override is not None else
                              _default_hard_timeout(
                                  getattr(self, "_timeout", 60.0),
                                  getattr(self, "model", ""),
                                  getattr(self, "_base_url", ""),
                                  getattr(self, "max_tokens", 8192),
                              ))

    def set_cache_key(self, key: str | None) -> None:
        """Pin a session-scoped prompt-cache routing key (typically the session_id). Cheapest cache
        lever there is: raises cache-hit rate, adds no tokens. Safe to call repeatedly."""
        self._cache_key = key or None

    def set_delta_sink(self, fn) -> None:
        """Wire a live-delta sink for interactive STREAMING: fn(kind: str, text: str), kind in
        {'content','reasoning'}. None disables rendering, not transport streaming. Safe to call repeatedly.
        Pure UX — the slice/loop/moat never see it (the assembled result is identical)."""
        self._on_delta = fn

    def set_transport_activity(self, fn) -> None:
        """Set a private, fail-open lifecycle observer ``fn(event: str, detail: dict)``.

        Events are metadata-only state transitions: ``awaiting_model``, ``provider_queue``,
        ``provider_admitted``, ``first_byte``, ``stream_heartbeat``, ``reasoning``, ``writing``, ``finished``,
        ``cancelled``, ``timed_out``, ``failed``. Timing fields are integer milliseconds and heartbeat payloads
        contain counts/state only—never token text. Child setup may call this on its shallow LLM view without
        touching the shared SDK client or the parent's rendering sink. An explicit ``transport_activity=``
        passed to :meth:`complete` wins for that request.
        """
        self._transport_activity = fn

    def set_stream_transport(self, enabled: bool) -> None:
        """Enable/disable SSE transport explicitly (default: enabled).

        This is an escape hatch for a demonstrably non-streaming OpenAI-compatible endpoint. Disabling it
        restores the older blocking/watchdog path, including its indeterminate off-main timeout semantics.
        It is not coupled to the presence of a UI delta sink.
        """
        self._stream_transport_enabled = bool(enabled)

    @staticmethod
    def _activity(sink, event: str, **detail) -> None:
        if sink is None:
            return
        try:
            sink(event, detail)
        except Exception:  # noqa: BLE001 - observation must never break a provider call
            pass

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
        return isinstance(error, openai_errors + transport + (
            EmptyResponseError, ImmediateRetryError, PreFirstByteTimeoutError,
        ))

    def _on_alarm(self, signum, frame):
        """SIGALRM handler: a request blew the HARD wall-clock deadline → raise a retryable timeout."""
        APITimeoutError = _import_api_timeout_error()
        try:
            import httpx
            raise APITimeoutError(request=httpx.Request("POST", (self._base_url or "http://local") + "/chat/completions"))
        except TypeError:
            # Older SDKs don't accept `request=` in the constructor.
            raise APITimeoutError("sliceagent hard timeout reached")

    def _arm_hard_alarm(self, seconds: float | None = None):
        """Arm SIGALRM and return ``(signal_module, previous_handler, timer_kind)``, or ``None`` safely.

        Installing the handler and arming the timer are two separate operations. If the second one fails,
        restore the first immediately so a partial platform implementation cannot leave SliceAgent owning the
        process-wide SIGALRM handler despite falling back to the daemon watchdog.
        """
        import signal as _signal

        previous = None
        installed = False
        timer_kind = "alarm"
        try:
            previous = _signal.signal(_signal.SIGALRM, self._on_alarm)
            installed = True
            duration = max(0.001, float(self._hard_timeout if seconds is None else seconds))
            if hasattr(_signal, "setitimer") and hasattr(_signal, "ITIMER_REAL"):
                _signal.setitimer(_signal.ITIMER_REAL, duration)
                timer_kind = "itimer"
            else:
                # alarm() accepts whole seconds only; round upward so the portability fallback never fires
                # before the requested absolute deadline.
                whole_seconds = max(1, int(duration) + (0 if duration.is_integer() else 1))
                _signal.alarm(whole_seconds)
            return _signal, previous, timer_kind
        except (ValueError, AttributeError, OSError):
            if installed:
                try:
                    _signal.signal(_signal.SIGALRM, previous)
                except (ValueError, AttributeError, OSError):
                    pass
            return None

    def _create(self, kwargs: dict, caller=None):
        """Call the SDK with a HARD wall-clock deadline that ALWAYS fires, on ANY thread. The httpx/SDK
        read-timeout only bounds the gap BETWEEN bytes, so a connection that goes silent mid-response can
        hang far past `timeout` (observed: a stalled read wedging the loop 10+ min). On the main thread a
        SIGALRM deadline guarantees control returns to the retry path. OFF the main thread — e.g. a
        Terminal-Bench / any host ThreadPoolExecutor worker, where SIGALRM cannot arm — a watchdog thread
        enforces the SAME deadline (the abandoned SDK call is left to die on its socket while control
        returns). Without this, a wedged connection in a worker thread hangs the turn FOREVER, since the
        SDK timeout alone misses silent mid-response stalls. Task/provider-agnostic reliability."""
        import time

        caller = caller or (lambda kw: self.client.chat.completions.create(**kw))
        deadline = time.monotonic() + self._hard_timeout
        # SIGALRM cannot be installed off-main. Enter the guarded daemon path before acquiring a provider
        # lease, eliminating a needless lease transfer on every legacy/non-streaming child call.
        if threading.current_thread() is not threading.main_thread():
            return self._create_watchdog(kwargs, caller, _deadline=deadline)
        lease = None
        alarm = None
        try:
            gate = getattr(self, "_provider_call_gate", None)
            if gate is not None:
                lease = gate.new_lease()
                try:
                    gate.acquire_lease(
                        lease, timeout=max(0.0, deadline - time.monotonic()),
                    )
                except _ProviderCapacityTimeout as error:
                    raise ProviderCapacityError(
                        "provider capacity remained occupied until this call's admission deadline; "
                        "no new request was started"
                    ) from error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderCapacityError(
                    "provider admission consumed this call's whole deadline; no new request was started"
                )
            # Acquire capacity before arming SIGALRM. A signal can land between a function return and caller
            # assignment; arming first made that tiny handoff window leak an already-incremented provider slot.
            alarm = self._arm_hard_alarm(remaining)
            if alarm is None:
                # A main-thread platform may still lack SIGALRM. No request exists yet: retire this lease and
                # let the guarded worker reacquire under the SAME deadline. That removes the caller→watchdog
                # ownership gap without resetting the wall clock.
                if lease is not None:
                    lease.release()
                    lease = None
                return self._create_watchdog(kwargs, caller, _deadline=deadline)
            return caller(kwargs)
        finally:
            try:
                if alarm is not None:
                    _signal, prev, timer_kind = alarm
                    try:
                        if timer_kind == "itimer":
                            _signal.setitimer(_signal.ITIMER_REAL, 0)
                        else:
                            _signal.alarm(0)
                    finally:
                        _signal.signal(_signal.SIGALRM, prev)
            finally:
                if lease is not None:
                    # The timer is disabled before capacity retirement, and an external interruption during
                    # the first idempotent removal still gets one unconditional retirement attempt.
                    try:
                        lease.release()
                    finally:
                        lease.release()

    def _create_watchdog(self, kwargs: dict, caller=None, *, _lease=None, _deadline: float | None = None):
        """Off-main-thread hard deadline: run the SDK call in a DAEMON worker and abandon it if it blows
        the wall-clock budget. Because Python cannot cancel the abandoned request, that deadline is an
        INDETERMINATE non-retryable result rather than permission to overlap it with another request. #47: a
        daemon thread (vs a ThreadPoolExecutor, whose worker the
        interpreter joins at exit) means a wedged call can NEVER block process shutdown — it dies with the
        socket whenever the SDK call finally errors on its own timeout. One thread per call; bounded."""
        import threading
        import time
        caller = caller or (lambda kw: self.client.chat.completions.create(**kw))
        box: dict = {}
        deadline = (_deadline if _deadline is not None else time.monotonic() + self._hard_timeout)
        lease = _lease
        gate = getattr(self, "_provider_call_gate", None)
        if gate is not None and lease is None:
            lease = gate.new_lease()
            try:
                gate.acquire_lease(
                    lease, timeout=max(0.0, deadline - time.monotonic()),
                )
            except _ProviderCapacityTimeout as error:
                raise ProviderCapacityError(
                    "provider capacity remained occupied until this call's admission deadline; "
                    "no new request was started"
                ) from error

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if lease is not None:
                lease.release()
            raise ProviderCapacityError(
                "provider admission consumed this call's whole deadline; no new request was started"
            )

        handoff = _SubmissionHandoff()
        done = threading.Event()

        def _call():
            if not handoff.wait_sync():
                if lease is not None:
                    lease.release()
                done.set()
                return
            try:
                if time.monotonic() >= deadline:
                    box["err"] = ProviderCapacityError(
                        "watchdog worker admission crossed the model-call deadline; "
                        "no request was started"
                    )
                else:
                    box["resp"] = caller(kwargs)
            except BaseException as e:  # noqa: BLE001 — propagate to the caller thread
                box["err"] = e
            finally:
                # This may run long after the logical watchdog returned. Physical capacity remains occupied
                # until the provider call itself actually exits.
                if lease is not None:
                    lease.release()
                done.set()

        t = threading.Thread(target=_call, name="llm-watchdog", daemon=True)
        try:
            t.start()
            if time.monotonic() >= deadline:
                handoff.cancel_before_owner()
                if lease is not None:
                    lease.release()
                done.wait(0.1)
                raise ProviderCapacityError(
                    "watchdog thread startup consumed the model-call deadline; no request was started"
                )
            handoff.confirm_owner()
        except BaseException:  # noqa: BLE001 - settle thread/lease ownership before propagating
            cancelled_before_owner = handoff.cancel_before_owner()
            if cancelled_before_owner or t.ident is None:
                if lease is not None:
                    lease.release()
            raise
        completed = done.wait(max(0.0, deadline - time.monotonic()))
        if not completed:
            # Python cannot cancel this in-progress SDK request.  A normal retry here would overlap the
            # abandoned socket (and a child's timeout-recovery call could add a third request), multiplying
            # latency and spend.  Surface the indeterminate request instead of pretending it terminated.
            raise IndeterminateModelCallError(
                "provider request exceeded its watchdog deadline and may still be in flight; "
                "automatic retry was suppressed"
            )
        if "err" in box:
            raise box["err"]
        return box["resp"]

    def _stream_close_grace(self) -> float:
        try:
            value = float(os.environ.get("LLM_STREAM_CLOSE_GRACE_SEC") or 2.0)
        except (TypeError, ValueError):
            value = 2.0
        return max(0.05, value)

    def _stream_heartbeat_interval(self) -> float:
        """Return the low-rate transport heartbeat cadence (presentation/metrics only)."""
        try:
            value = float(os.environ.get("LLM_STREAM_HEARTBEAT_SEC") or 5.0)
        except (TypeError, ValueError):
            value = 5.0
        # Keep a bad setting from turning the private observer into a per-poll event firehose.
        return max(0.25, value)

    def _create_streaming(self, kwargs: dict, *, should_cancel=None, activity=None):
        """Drain one chat SSE request through the cancellable async bridge.

        This path is used on every thread and does not depend on a UI sink. The async task owns the stream
        context and absolute timeout, so a blocked network read is cancellable; the sync caller only assembles
        chunks and renders optional deltas. A result/error is returned only after connection closure is proven.
        """
        self._activity(activity, "awaiting_model", transport="sse")
        try:
            response = self._stream_assemble(
                kwargs, should_cancel=should_cancel, activity=activity,
            )
        except RetryCancelledError:
            self._activity(activity, "cancelled", transport="sse")
            raise
        except IndeterminateModelCallError:
            self._activity(activity, "failed", transport="sse", indeterminate=True)
            raise
        except PreFirstByteTimeoutError:
            self._activity(activity, "timed_out", transport="sse", phase="first_byte")
            raise
        except _import_api_timeout_error():
            self._activity(activity, "timed_out", transport="sse")
            raise
        except Exception as error:
            self._activity(activity, "failed", transport="sse", error=type(error).__name__)
            raise
        self._activity(activity, "finished", transport="sse")
        return response

    def _stream_assemble(self, kwargs: dict, *, should_cancel=None, activity=None):
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
        reasoning_parts: list[str] = []
        calls: dict[int, dict] = {}          # index → {id, name, args:[fragments]}
        finish = None
        usage = None
        first_byte = False
        reasoning_seen = False
        writing_seen = False
        _timeout_err = _import_api_timeout_error()
        import time
        stream_started = time.monotonic()

        def observe_transport(event: str, detail: dict) -> None:
            nonlocal first_byte
            # Production hub owns the physical first-item timestamp. Legacy/fake transports do not call this
            # observer, so consume() below retains a deduplicated fallback.
            if event == "first_byte":
                if first_byte:
                    return
                first_byte = True
            self._activity(activity, event, transport="sse", **detail)

        def consume(chunk) -> None:
            nonlocal finish, usage, first_byte, reasoning_seen, writing_seen
            if not first_byte:
                first_byte = True
                self._activity(
                    activity, "first_byte", transport="sse",
                    elapsed_ms=max(0, int((time.monotonic() - stream_started) * 1000)),
                )
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
                    if not writing_seen:
                        writing_seen = True
                        self._activity(activity, "writing", transport="sse")
                    parts.append(txt)
                    self._emit("content", txt)
                rc = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                if rc:
                    reasoning_parts.append(rc)
                    # Activity is a monotonic public phase: an odd provider that emits a late reasoning
                    # fragment after answer text must not rewind the matrix from writing → reasoning.
                    if not reasoning_seen and not writing_seen:
                        reasoning_seen = True
                        self._activity(activity, "reasoning", transport="sse")
                    else:
                        reasoning_seen = True
                    self._emit("reasoning", rc)
                tool_deltas = getattr(d, "tool_calls", None) or []
                if tool_deltas and not writing_seen:
                    writing_seen = True
                    self._activity(activity, "writing", transport="sse")
                for tcd in tool_deltas:
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

        # E3 streaming resilience: a single MALFORMED chunk is skipped (never aborts the whole stream); a
        # mid-stream CONNECTION error re-raises ONLY when nothing was assembled (so with_retry re-rolls) —
        # otherwise we salvage the partial as a truncated stop, which the loop handles cleanly.
        try:
            if not hasattr(self, "_transport_spec"):
                # Backward-compatible seam for legacy object.__new__ adapter tests/embedders. Every real
                # OpenAILLM instance has _transport_spec and therefore uses the cancellable async bridge.
                for chunk in self.client.chat.completions.create(**skw):
                    _consume_stream_item(consume, chunk, _timeout_err)
            else:
                hub = getattr(self, "_transport_hub", None) or _async_transport_hub()
                hub.run(
                    "chat", self._transport_spec, skw,
                    timeout=self._hard_timeout,
                    close_grace=self._stream_close_grace(),
                    should_cancel=should_cancel,
                    on_item=lambda chunk: _consume_stream_item(consume, chunk, _timeout_err),
                    on_activity=observe_transport,
                    heartbeat_interval=self._stream_heartbeat_interval(),
                    provider_gate=getattr(self, "_provider_call_gate", None),
                )
        except Exception as e:  # noqa: BLE001 — stream broke mid-flight
            # Deadlines/cancellation/unknown physical state are never salvageable as a normal truncation.
            if _is_transport_timeout(e):
                if reasoning_seen or calls or parts:
                    raise IndeterminateModelCallError(
                        "model stream timed out after semantic output began; automatic replay was suppressed"
                    ) from e
                if not first_byte:
                    raise PreFirstByteTimeoutError(
                        "provider stream timed out before its first response byte"
                    ) from e
                raise
            if isinstance(e, (RetryCancelledError, IndeterminateModelCallError)):
                raise
            # Reasoning/tool-argument bytes prove that generation began even when there is no safe assistant
            # content to return. Blindly replaying that billed prompt can duplicate a long hidden reasoning run
            # (the exact child retry storm this transport fixes). Suppress automatic replay; the request is
            # physically closed, but its semantic result is indeterminate/incomplete.
            if not parts and (reasoning_seen or calls):
                raise IndeterminateModelCallError(
                    "model stream failed after semantic output began but before a usable response was sealed; "
                    "automatic replay was suppressed"
                ) from e
            if not parts and not calls:
                raise                          # nothing salvageable → let with_retry re-roll
            finish = finish or "length"        # partial assembly → treat as a truncated (incomplete) stop
        # Drop any INCOMPLETE tool call (missing id or name) — a mid-stream break before a tool_call's
        # name/id delta arrived would otherwise yield a ToolCall(name=None) that breaks the dispatcher.
        # If this empties content AND tool_calls, complete() raises EmptyResponseError → with_retry re-rolls.
        tool_calls = [NS(id=c["id"], function=NS(name=c["name"], arguments="".join(c["args"])))
                      for _, c in sorted(calls.items(), key=lambda kv: kv[0] if isinstance(kv[0], int) else 0)
                      if c["id"] and c["name"]]   # robust sort: a None/str stream index must not crash assembly
        message = NS(
            content=("".join(parts) or None),
            reasoning_content=("".join(reasoning_parts) or None),
            tool_calls=tool_calls,
        )
        return NS(choices=[NS(message=message, finish_reason=finish)], usage=usage)

    def _official_deepseek_v4_mode(self) -> str | None:
        """Return ``thinking``/``non_thinking`` only for the official V4 wire contract.

        Official V4 models default to thinking unless the SliceAgent profile is ``fast``. During DeepSeek's
        retirement window the legacy aliases remain mode-specific: ``deepseek-reasoner`` is thinking and
        ``deepseek-chat`` is non-thinking. A model with "deepseek" in its name on a router/custom endpoint is
        intentionally excluded because those providers document different replay/tool-choice behaviour.
        """
        from urllib.parse import urlparse

        base = (getattr(self, "_base_url", "") or "").strip().lower()
        parsed = urlparse(base if "://" in base else f"https://{base}")
        if (parsed.hostname or "").rstrip(".") != "api.deepseek.com":
            return None
        model = (getattr(self, "model", "") or "").strip().lower()
        if model == "deepseek-chat":
            return "non_thinking"
        if model not in {"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner"}:
            return None
        return "non_thinking" if getattr(self, "reasoning", "full") == "fast" else "thinking"

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
        official_v4_mode = self._official_deepseek_v4_mode()
        if official_v4_mode is not None:
            enabled = official_v4_mode == "thinking"
            kwargs: dict = {
                "extra_body": {"thinking": {"type": "enabled" if enabled else "disabled"}},
            }
            # V4's official OpenAI wire accepts exactly high/max. Keep the public SliceAgent profile aliases
            # stable while mapping max directly (not OpenAI's xhigh). "full" leaves DeepSeek's own default.
            if enabled and r in {"high", "max", "xhigh"}:
                kwargs["reasoning_effort"] = "max" if r == "xhigh" else r
            return kwargs
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

    def _complete_responses(
        self,
        messages: list[dict],
        tools: list[dict],
        effort: str,
        *,
        should_cancel=None,
        activity=None,
    ) -> AssistantMessage:
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
        # ``False`` for legacy object.__new__ test doubles that bypass __init__; every real instance sets True.
        _stream = bool(getattr(self, "_stream_transport_enabled", False))
        try:
            resp = (self._responses_stream(
                        kwargs, should_cancel=should_cancel, activity=activity,
                    ) if _stream
                    else self._create(kwargs, caller=lambda kw: self.client.responses.create(**kw)))
        except Exception as e:  # noqa: BLE001
            # route a provider context overflow into the SAME slice-tighten recovery the chat path uses
            # (llm.py chat except) — otherwise an overflow on the responses path crashes the turn instead.
            if is_context_overflow(e):
                raise ContextOverflow(e, status_code=getattr(e, "status_code", None)) from e
            raise
        return self._parse_responses(resp)

    def _responses_stream(self, kwargs: dict, *, should_cancel=None, activity=None):
        """Stream a Responses call, emit content/reasoning deltas live, return the final Response (parsed
        downstream identically to the blocking path). Uses the same async close-confirmed transport as chat;
        there is no hidden blocking fallback request inside a failed visible attempt."""
        first_byte = reasoning_seen = writing_seen = False
        import time
        stream_started = time.monotonic()

        def observe_transport(activity_event: str, detail: dict) -> None:
            nonlocal first_byte
            if activity_event == "first_byte":
                if first_byte:
                    return
                first_byte = True
            self._activity(activity, activity_event, transport="responses_sse", **detail)

        def consume(event) -> None:
            nonlocal first_byte, reasoning_seen, writing_seen
            if not first_byte:
                first_byte = True
                self._activity(
                    activity, "first_byte", transport="responses_sse",
                    elapsed_ms=max(0, int((time.monotonic() - stream_started) * 1000)),
                )
            event_type = getattr(event, "type", "")

            def mark_writing() -> None:
                nonlocal writing_seen
                if not writing_seen:
                    writing_seen = True
                    self._activity(activity, "writing", transport="responses_sse")

            if event_type == "response.output_text.delta":
                mark_writing()
                self._emit("content", getattr(event, "delta", "") or "")
            elif event_type in (
                "response.reasoning_summary_text.delta", "response.reasoning_text.delta",
            ):
                if not reasoning_seen and not writing_seen:
                    reasoning_seen = True
                    self._activity(activity, "reasoning", transport="responses_sse")
                else:
                    reasoning_seen = True
                self._emit("reasoning", getattr(event, "delta", "") or "")
            elif event_type in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }:
                # Function arguments are semantic model output even though they never enter output_text.
                # If the stream breaks after this edge, replaying the prompt can duplicate a billed/effectful
                # tool decision; mark it exactly like answer writing so the failure becomes indeterminate.
                mark_writing()
            elif event_type in {"response.output_item.added", "response.output_item.done"}:
                # The generic output-item lifecycle carries the function-call shell around its argument events.
                # A normal assistant message item is not enough to prove semantic bytes; gate on item.type.
                item = getattr(event, "item", None)
                if getattr(item, "type", "") == "function_call":
                    mark_writing()

        self._activity(activity, "awaiting_model", transport="responses_sse")
        try:
            hub = getattr(self, "_transport_hub", None) or _async_transport_hub()
            response = hub.run(
                "responses", self._transport_spec, kwargs,
                timeout=self._hard_timeout,
                close_grace=self._stream_close_grace(),
                should_cancel=should_cancel,
                on_item=lambda event: _consume_stream_item(
                    consume, event, _import_api_timeout_error(),
                ),
                on_activity=observe_transport,
                heartbeat_interval=self._stream_heartbeat_interval(),
                provider_gate=getattr(self, "_provider_call_gate", None),
            )
        except RetryCancelledError:
            self._activity(activity, "cancelled", transport="responses_sse")
            raise
        except IndeterminateModelCallError:
            self._activity(activity, "failed", transport="responses_sse", indeterminate=True)
            raise
        except PreFirstByteTimeoutError:
            self._activity(activity, "timed_out", transport="responses_sse", phase="first_byte")
            raise
        except Exception as error:
            if not _is_transport_timeout(error):
                if reasoning_seen or writing_seen:
                    # Responses cannot construct its final typed object after a broken stream. Once semantic
                    # bytes arrived, replaying the whole request would duplicate potentially expensive reasoning;
                    # preserve uncertainty and let the owning turn decide how to recover.
                    self._activity(activity, "failed", transport="responses_sse", indeterminate=True)
                    raise IndeterminateModelCallError(
                        "responses stream failed after semantic output began but before a final response was "
                        "sealed; automatic replay was suppressed"
                    ) from error
                self._activity(
                    activity, "failed", transport="responses_sse", error=type(error).__name__,
                )
                raise
            self._activity(activity, "timed_out", transport="responses_sse")
            if reasoning_seen or writing_seen:
                raise IndeterminateModelCallError(
                    "responses stream timed out after semantic output began; automatic replay was suppressed"
                ) from error
            if not first_byte:
                raise PreFirstByteTimeoutError(
                    "provider responses stream timed out before its first response byte"
                ) from error
            raise
        self._activity(activity, "finished", transport="responses_sse")
        return response

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
        """Compatibility entrypoint for the two-argument ``LLMClient`` protocol."""
        return self.complete_with_control(messages, tools)

    def complete_with_control(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        should_cancel=None,
        transport_activity=None,
    ) -> AssistantMessage:
        """Complete with per-request cancellation and private transport activity observation.

        ``model_runner.complete_model_call`` can feature-detect this method while arbitrary two-argument fake
        LLMs remain compatible. The explicit observer wins; otherwise a child-local observer installed with
        :meth:`set_transport_activity` is used. Neither callback is stored or sent to the provider.
        """
        activity = (
            transport_activity if transport_activity is not None
            else getattr(self, "_transport_activity", None)
        )
        effort = self._effort()
        if effort and hasattr(self.client, "responses"):   # explicit effort → /v1/responses (chat 400s on
            return self._complete_responses(
                messages, tools, effort, should_cancel=should_cancel, activity=activity,
            )   # effort+tools). No responses API on
        # an old SDK / a provider that only has chat → fall through; the chat 400→drop below degrades it.
        kwargs: dict = dict(model=self.model, messages=messages, tools=tools)
        # DeepSeek V4's OFFICIAL thinking-mode tool wire rejects tool_choice (despite generic OpenAI-compatible
        # schemas often advertising it). Non-thinking V4, retiring deepseek-chat, routers, and every other
        # provider keep the established explicit "auto" behaviour.
        if self._official_deepseek_v4_mode() != "thinking":
            kwargs["tool_choice"] = "auto"
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
        # Transport streaming is independent of UI/main-thread state. A child view normally has no delta sink,
        # but still drains SSE through the cancellable async bridge and assembles the identical response.
        _stream = bool(getattr(self, "_stream_transport_enabled", False))
        try:
            resp = (
                self._create_streaming(
                    kwargs, should_cancel=should_cancel, activity=activity,
                ) if _stream else self._create(kwargs)
            )
        except Exception as e:
            # Context overflow is NOT a backoff case (is_retryable stays unchanged): signal the
            # rebuild loop to TIGHTEN the slice rather than re-send the identical oversized request.
            if is_context_overflow(e):
                raise ContextOverflow(e, status_code=getattr(e, "status_code", None)) from e
            # Remember the deterministic downgrade, then hand retry ownership back to complete_model_call.
            # Retrying here used to issue a second hidden HTTP request inside one ModelCallPrepared attempt.
            if "reasoning_effort" in str(e) and kwargs.pop("reasoning_effort", None) is not None:
                self._drop_reasoning_effort = True
                raise ImmediateRetryError(
                    "provider rejected reasoning_effort with tools; retrying without that optional field"
                ) from e
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
            content=msg.content,
            tool_calls=calls,
            usage=usage,
            finish_reason=choice.finish_reason,
            reasoning_content=(
                getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            ),
        )
