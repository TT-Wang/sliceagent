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

from memagent.mcp_client import McpRuntime  # noqa: E402

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
