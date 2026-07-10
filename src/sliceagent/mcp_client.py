"""MCP client — connect external MCP servers, surface their tools in the registry (③.3).

Uses `mcp__server__tool` namespacing + collision handling and an adapter
(MCP Tool → registry ToolEntry) + background-event-loop bridge. The official `mcp` SDK is
async and the agent loop is sync, so we run ONE asyncio loop in a daemon thread and submit
coroutines to it. Each server's connection lives in a SINGLE long-lived task (a worker that
opens the session, lists tools, then serves call requests off a queue) — keeping every
session op in one task sidesteps anyio's "cancel scope in different task" pitfall. A server
that fails to connect degrades to zero tools and never crashes the agent.

This phase supports the stdio transport (declared as [mcp_servers.<name>] with command/args).
HTTP/SSE transports are a later add behind the same McpServer seam.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading

from .registry import ToolEntry, ToolText

_QUALIFY_MAX = 64


def qualify(server: str, tool: str) -> str:
    """mcp__<server>__<tool>, sanitized to [A-Za-z0-9_], hash-truncated if too long."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", f"mcp__{server}__{tool}")
    if len(name) > _QUALIFY_MAX:
        h = hashlib.sha1(name.encode()).hexdigest()[:8]
        name = name[: _QUALIFY_MAX - 9] + "_" + h
    return name


_MCP_SAFETY_CAP = 2_000_000   # last-resort OOM guard if a server streams megabytes. The REAL bounding is
                              # the host page-out applied in the handler: a big MCP result (browser/DB/
                              # Playwright payloads) goes to a blob + head/tail view, not inlined whole.


def _result_to_text(result) -> str:
    parts = []
    for block in (getattr(result, "content", None) or []):
        t = getattr(block, "text", None)
        parts.append(t if t is not None else f"[{getattr(block, 'type', 'content')}]")
    text = "\n".join(parts).strip() or "(no content)"
    if len(text) > _MCP_SAFETY_CAP:           # last-resort OOM guard; page-out (handler) does normal bounding
        text = text[:_MCP_SAFETY_CAP] + f"\n…[truncated {len(text) - _MCP_SAFETY_CAP} chars of MCP output]"
    # isError must propagate as ok=False so the loop's failing-detection + the anti-loop guardrail
    # (repeated_exact_failure) actually see the failure — a plain "Error: …" string gets wrapped ok=True.
    # (MCP results enter as role="tool" messages, which the model already treats as DATA at the protocol
    #  level — unlike web's slice re-injection channels — so an extra wrap_untrusted fence is low-value here.)
    return ToolText(f"Error: {text}", ok=False) if getattr(result, "isError", False) else text


def _mcp_handler(server, tool, page_out):
    """Tool handler that pages a LARGE MCP result OUT to a blob (a head/tail view + a read_file ref) rather
    than inlining the whole payload — browser/DB/Playwright results can be hundreds of KB. The full output
    is preserved on disk and paged back on demand (the moat's L1→L2 page-out), so nothing is lost. With no
    host page_out (eval/headless), returns the raw text (already OOM-capped by _result_to_text)."""
    def _handle(args):
        out = server.call(tool, args)
        status = getattr(out, "status", None)
        ok = getattr(out, "ok", True)   # capture BEFORE page_out — str.translate() (inside page_out's
        # control-char strip) always returns a plain str, silently dropping the ToolText subclass and
        # its .ok flag even on the success path. Re-wrapping with the captured `ok` below (not just on
        # the ok=False branch) is what makes run_tool_batch's `getattr(out, "ok", None)` see an explicit
        # flag instead of None — the None case was falling back to prose-matching ("Error"/"Exit code"
        # prefix), which false-flagged a legitimate success result whose text happened to start that way.
        if page_out:
            try:
                out = page_out(out, label=f"mcp-{tool}")
            except Exception:  # noqa: BLE001 — paging must never fail the tool call
                pass
        return ToolText(str(out), status=status) if status is not None else ToolText(str(out), ok=ok)
    return _handle


def _function_schema(qname: str, tool) -> dict:
    params = getattr(tool, "inputSchema", None)
    if not isinstance(params, dict) or params.get("type") != "object":
        params = {"type": "object", "properties": {}}
    desc = (getattr(tool, "description", None) or f"MCP tool {tool.name}").strip()
    return {"type": "function", "function": {"name": qname, "description": desc[:1024], "parameters": params}}


class McpRuntime:
    """One background asyncio loop in a daemon thread; bridges sync ↔ async."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="mcp-loop", daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro, timeout):
        cf = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return cf.result(timeout)
        except BaseException:
            cf.cancel()   # #60: a timed-out/failed call must stop its coroutine on the loop, not leak it
            raise

    def spawn(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self):
        # #61/#62: cancel every task on the loop BEFORE stopping it, so each _serve()'s
        # `async with stdio_client(...)`/`ClientSession` __aexit__ runs and its child PROCESS is
        # terminated. Stopping the loop with tasks still pending would orphan those subprocesses.
        async def _cancel_all():
            tasks = [t for t in asyncio.all_tasks(self.loop) if t is not asyncio.current_task()]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_cancel_all(), self.loop).result(timeout=5)
        except BaseException:  # noqa: BLE001 — best-effort teardown
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)


class McpServer:
    """One MCP server connection, served by a single long-lived worker task on the runtime loop."""

    def __init__(self, name: str, runtime: McpRuntime):
        self.name = name
        self.runtime = runtime
        self.tools: list = []
        self.error: str | None = None
        self._queue: asyncio.Queue | None = None
        self._ready: asyncio.Event | None = None

    def connect(self, params, timeout: float = 30) -> list:
        self.runtime.submit(self._mk_primitives(), timeout)  # create loop-bound queue/event
        self.runtime.spawn(self._serve(params))              # start the worker (long-lived)
        self.runtime.submit(self._wait_ready(), timeout)     # block until tools listed or error
        if self.error:
            raise RuntimeError(self.error)
        return self.tools

    async def _mk_primitives(self):
        self._queue = asyncio.Queue()
        self._ready = asyncio.Event()

    async def _wait_ready(self):
        await self._ready.wait()

    async def _serve(self, params):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.tools = list((await session.list_tools()).tools)
                    self._ready.set()
                    while True:
                        req = await self._queue.get()
                        if req is None:
                            break
                        tool, args, fut = req
                        try:
                            res = await session.call_tool(tool, args or {})
                            if not fut.done():
                                fut.set_result(res)
                        except Exception as e:  # noqa: BLE001
                            if not fut.done():
                                fut.set_exception(e)
        except Exception as e:  # noqa: BLE001 — connect/transport failure
            self.error = str(e)
            if self._ready is not None and not self._ready.is_set():
                self._ready.set()

    def call(self, tool: str, args: dict, timeout: float = 60) -> str:
        if self.error:
            return ToolText(f"Error: MCP server {self.name!r} unavailable: {self.error}", ok=False)

        async def _do():
            fut = self.runtime.loop.create_future()
            await self._queue.put((tool, args, fut))
            return await fut

        try:
            return _result_to_text(self.runtime.submit(_do(), timeout))
        except Exception as e:  # noqa: BLE001
            # Once the request has crossed into the long-lived server worker, cancelling this waiter does
            # not prove that the remote operation stopped. Report honest uncertainty so the execution
            # kernel blocks dependent barriers and refuses a clean seal.
            return ToolText(
                f"Error: MCP call {self.name}.{tool} outcome indeterminate after transport/timeout "
                f"failure: {type(e).__name__}: {e}",
                status="indeterminate",
            )

    def close(self):
        try:
            self.runtime.submit(self._queue.put(None), 5)
        except Exception:  # noqa: BLE001
            pass


def _params_from_conf(conf: dict):
    from mcp import StdioServerParameters
    env = None
    if conf.get("env"):
        env = {**os.environ, **conf["env"]}
    return StdioServerParameters(command=conf["command"], args=list(conf.get("args", [])),
                                 env=env, cwd=conf.get("cwd"))


def connect_mcp_servers(registry, servers: dict, runtime: McpRuntime | None = None,
                        *, timeout: float = 30, on_log=None, page_out=None):
    """Connect each declared server and register its tools into `registry` (namespaced).
    Returns (connected_servers, runtime). Returns ([], None) when nothing is configured."""
    def log(m):
        if on_log:
            on_log(m)

    if not servers:
        return [], None
    runtime = runtime or McpRuntime()
    connected, total = [], 0
    for name, conf in servers.items():
        if not isinstance(conf, dict) or not conf.get("command"):
            log(f"mcp:{name} skipped (only stdio with a 'command' is supported in this phase)")
            continue
        from .mcp_security import validate_mcp_server_entry   # screen BEFORE spawning (RCE-by-design surface)
        _bad = validate_mcp_server_entry(name, conf)
        if _bad:
            for _b in _bad:
                log(f"mcp:{name} REFUSED (security) — {_b}")
            continue
        server = McpServer(name, runtime)
        try:
            tools = server.connect(_params_from_conf(conf), timeout)
        except Exception as e:  # noqa: BLE001
            log(f"mcp:{name} connect failed: {e}")
            continue
        inc, exc = set(conf.get("include") or []), set(conf.get("exclude") or [])
        n = 0
        for tool in tools:
            if (inc and tool.name not in inc) or tool.name in exc:
                continue
            qname = qualify(name, tool.name)
            if registry.has(qname):
                log(f"mcp:{name} tool {qname} collides with an existing tool, skipped")
                continue
            registry.register(ToolEntry(
                name=qname, schema=_function_schema(qname, tool),
                handler=_mcp_handler(server, tool.name, page_out),
                source="mcp",
            ))
            n += 1
        connected.append(server)
        total += n
        log(f"mcp:{name} connected ({n} tools)")
    return connected, runtime
