"""Regression tests for the MCP runtime lifecycle (#60 submit cancels on timeout; #61/#62 shutdown
cancels pending worker tasks so stdio child processes terminate). No real MCP server needed — the
runtime is just an asyncio loop in a daemon thread. Run: PYTHONPATH=src python tests/test_mcp_runtime.py
"""
import asyncio
import concurrent.futures
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.execution import ToolStatus  # noqa: E402
import sliceagent.mcp_client as mcp_client  # noqa: E402
from sliceagent.mcp_client import (McpRuntime, McpServer, _mcp_handler,  # noqa: E402
                                   _params_from_conf)
from sliceagent.registry import ToolRegistry  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def submit_cancels_coroutine_on_timeout():  # #60
    rt = McpRuntime()
    state = {"cancelled": False}

    async def _hang():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    raised = False
    try:
        rt.submit(_hang(), 0.2)
    except (concurrent.futures.TimeoutError, TimeoutError):
        raised = True
    assert raised, "a too-slow submit must surface a timeout"
    time.sleep(0.4)   # let the cancellation propagate on the loop
    assert state["cancelled"], "a timed-out submit must cancel its coroutine, not leak it (#60)"
    rt.shutdown()


@check
def shutdown_cancels_pending_tasks():  # #61/#62
    rt = McpRuntime()
    state = {"cancelled": False}

    async def _worker():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    rt.spawn(_worker())
    time.sleep(0.15)   # let the worker actually start
    rt.shutdown()
    time.sleep(0.3)
    assert state["cancelled"], "shutdown must cancel pending worker tasks so child procs exit (#61/#62)"


@check
def connect_timeout_cancels_the_discarded_server_worker():
    rt = McpRuntime()
    state = {"cancelled": False}

    class SlowServer(McpServer):
        async def _serve(self, _params):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise

    server = SlowServer("slow", rt)
    raised = False
    try:
        try:
            server.connect(object(), timeout=0.05)
        except (concurrent.futures.TimeoutError, TimeoutError):
            raised = True
        time.sleep(0.15)
        assert raised
        assert state["cancelled"], "a timed-out connect must not leave its discarded worker/process alive"

        async def pending_workers():
            return [task for task in asyncio.all_tasks()
                    if task is not asyncio.current_task() and not task.done()]

        assert rt.submit(pending_workers(), 1) == []
    finally:
        rt.shutdown()


@check
def timed_out_queued_mcp_operation_is_indeterminate_even_if_it_finishes_late():
    rt = McpRuntime()
    server = McpServer("late", rt)
    rt.submit(server._mk_primitives(), 1)
    state = {"mutated": False}

    async def _consumer():
        _tool, _args, future = await server._queue.get()
        await asyncio.sleep(0.2)
        state["mutated"] = True
        if not future.done():
            future.set_result(type("Result", (), {"content": [], "isError": False})())

    rt.spawn(_consumer())
    result = server.call("write", {"value": 1}, timeout=0.05)
    assert result.status is ToolStatus.INDETERMINATE, result.status
    assert not state["mutated"], "the timed-out call must return before the queued operation settles"
    proxy = type("Proxy", (), {"call": lambda _self, _tool, _args: result})()
    paged = _mcp_handler(proxy, "write", lambda value, **_kwargs: str(value))({"value": 1})
    assert paged.status is ToolStatus.INDETERMINATE, "paging must preserve typed uncertainty"
    time.sleep(0.3)
    assert state["mutated"], "the late side effect proves FAILED would have been dishonest"
    rt.shutdown()


@check
def custom_mcp_env_never_inherits_process_secrets():
    secret_names = ("LLM_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY")
    previous = {name: os.environ.get(name) for name in secret_names}
    try:
        for name in secret_names:
            os.environ[name] = f"secret-{name}"
        params = _params_from_conf({"command": "demo", "env": {"MODE": "stdio", "PORT": 1234}})
        assert params.env == {"MODE": "stdio", "PORT": "1234"}, params.env
        assert not (set(secret_names) & set(params.env)), params.env
        defaulted = _params_from_conf({"command": "demo"})
        assert defaulted.env is None, "None delegates to the MCP SDK's safe platform allowlist"
        for invalid in ({"command": 123}, {"command": "demo", "args": "--one-string"},
                        {"command": "demo", "args": ["ok", 3]}):
            try:
                _params_from_conf(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid MCP process config was accepted: {invalid!r}")
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@check
def malformed_remote_tool_metadata_is_skipped_without_partial_startup_failure():
    class FakeServer:
        def __init__(self, name, _runtime):
            self.name = name

        def connect(self, _params, _timeout):
            schema = {"type": "object", "properties": {}}
            return [
                type("Tool", (), {"name": "good", "description": "works", "inputSchema": schema})(),
                type("Tool", (), {"name": "bad", "description": 123, "inputSchema": schema})(),
                type("Tool", (), {"name": None, "description": "missing name", "inputSchema": schema})(),
            ]

        def call(self, _tool, _args):
            return "ok"

    original = mcp_client.McpServer
    registry, logs = ToolRegistry(), []
    mcp_client.McpServer = FakeServer
    try:
        connected, runtime = mcp_client.connect_mcp_servers(
            registry, {"demo": {"command": "demo"}}, runtime=object(), on_log=logs.append,
        )
    finally:
        mcp_client.McpServer = original
    assert len(connected) == 1 and runtime is not None
    assert registry.names() == ["mcp__demo__good"]
    assert sum("ignored malformed tool metadata" in item for item in logs) == 2, logs


@check
def invalid_tool_filters_are_rejected_before_spawning_the_server():
    constructed = []

    class FakeServer:
        def __init__(self, name, _runtime):
            constructed.append(name)

    original = mcp_client.McpServer
    mcp_client.McpServer = FakeServer
    logs = []
    try:
        connected, runtime = mcp_client.connect_mcp_servers(
            ToolRegistry(), {"demo": {"command": "demo", "include": 7}},
            runtime=object(), on_log=logs.append,
        )
    finally:
        mcp_client.McpServer = original
    assert connected == [] and runtime is not None
    assert constructed == [], "invalid selection metadata must be rejected before a process can spawn"
    assert any("invalid config" in item for item in logs), logs


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
