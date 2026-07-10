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
from sliceagent.mcp_client import McpRuntime, McpServer, _mcp_handler  # noqa: E402

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
