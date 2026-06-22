"""MCP client — connect external MCP servers, surface their tools in the registry (③.3).

Borrowed: Kimi's `mcp__server__tool` namespacing + collision handling; Hermes' adapter
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

from .registry import ToolEntry

_QUALIFY_MAX = 64


def qualify(server: str, tool: str) -> str:
    """mcp__<server>__<tool>, sanitized to [A-Za-z0-9_], hash-truncated if too long (Kimi)."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", f"mcp__{server}__{tool}")
    if len(name) > _QUALIFY_MAX:
        h = hashlib.sha1(name.encode()).hexdigest()[:8]
        name = name[: _QUALIFY_MAX - 9] + "_" + h
    return name


_MCP_MAX_OUTPUT = 100_000   # cap one MCP tool result so a runaway payload can't blow up the slice (Kimi)


def _result_to_text(result) -> str:
    parts = []
    for block in (getattr(result, "content", None) or []):
        t = getattr(block, "text", None)
        parts.append(t if t is not None else f"[{getattr(block, 'type', 'content')}]")
    text = "\n".join(parts).strip() or "(no content)"
    if len(text) > _MCP_MAX_OUTPUT:           # bound an oversized result (server returning a huge blob)
        text = text[:_MCP_MAX_OUTPUT] + f"\n…[truncated {len(text) - _MCP_MAX_OUTPUT} chars of MCP output]"
    return f"Error: {text}" if getattr(result, "isError", False) else text


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
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def spawn(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self):
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
            return f"Error: MCP server {self.name!r} unavailable: {self.error}"

        async def _do():
            fut = self.runtime.loop.create_future()
            await self._queue.put((tool, args, fut))
            return await fut

        try:
            return _result_to_text(self.runtime.submit(_do(), timeout))
        except Exception as e:  # noqa: BLE001
            return f"Error: MCP call {self.name}.{tool} failed: {e}"

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
                        *, timeout: float = 30, on_log=None):
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
                handler=(lambda args, s=server, t=tool.name: s.call(t, args)),
                source="mcp",
            ))
            n += 1
        connected.append(server)
        total += n
        log(f"mcp:{name} connected ({n} tools)")
    return connected, runtime
