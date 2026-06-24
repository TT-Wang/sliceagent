"""Conflict-aware tool scheduler (ported from Kimi Code's tool-scheduler).

Runs a batch of tool calls with maximum safe concurrency: non-conflicting calls
overlap, conflicting calls serialize, and results are returned in PROVIDER ORDER
(deterministic for the model) regardless of completion order.
"""
from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

from .access import Accesses, conflict

Task = tuple[Accesses, Callable[[], str]]


def run_scheduled(tasks: list[Task], max_workers: int = 8, timeout: float | None = None) -> list[str]:
    """Run a batch with max safe concurrency, results in PROVIDER ORDER. `timeout` (seconds, opt-in) is a
    per-task WALL-CLOCK deadline: a tool that overruns yields a timeout result so the turn proceeds instead
    of hanging on a stuck tool (the orphaned thread is abandoned, not joined — Python can't force-kill a
    thread, so this is a last-resort net above each tool's own subprocess/SIGALRM timeout). timeout=None
    preserves the original behaviour exactly (wait for every task)."""
    n = len(tasks)
    if n == 0:
        return []
    if n == 1 and timeout is None:
        try:
            return [tasks[0][1]()]
        except Exception as e:
            return [f"Error: {e}"]

    results: list[str | None] = [None] * n
    accesses = [t[0] for t in tasks]
    pending = list(range(n))
    running: dict = {}   # future -> index
    started: dict = {}   # future -> monotonic start time (deadline tracking)

    pool = ThreadPoolExecutor(max_workers=min(max_workers, n))
    try:
        while pending or running:
            running_acc = [accesses[i] for i in running.values()]
            selected_acc: list[Accesses] = []
            for idx in list(pending):
                a = accesses[idx]
                # never run two conflicting tasks at once (vs running or vs already-selected this round)
                if any(conflict(a, ra) for ra in running_acc) or any(conflict(a, sa) for sa in selected_acc):
                    continue
                fut = pool.submit(tasks[idx][1])
                running[fut] = idx
                started[fut] = time.monotonic()
                selected_acc.append(a)
                pending.remove(idx)
            if running:
                wait_t = None
                if timeout is not None:
                    # wake no later than the SOONEST deadline = the OLDEST running task (MAX elapsed).
                    # (min elapsed = newest task = furthest-out deadline → would reap the oldest too late.)
                    elapsed_max = max(time.monotonic() - started[f] for f in running)
                    wait_t = max(0.05, timeout - elapsed_max)
                done, _ = wait(list(running.keys()), timeout=wait_t, return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = running.pop(fut)
                    started.pop(fut, None)
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        results[idx] = f"Error: {e}"
                if timeout is not None:    # reap overruns: abandon the wait, let the turn continue
                    now = time.monotonic()
                    for fut in [f for f in running if now - started[f] >= timeout]:
                        idx = running.pop(fut)
                        started.pop(fut, None)
                        results[idx] = (f"Error: tool timed out after {timeout:.0f}s "
                                        "(abandoned; it may still be running in the background)")
        # safety: the loop only exits when pending AND running are empty, so every slot is set — but
        # never let a stray None escape to a caller that expects str.
        return [r if r is not None else "Error: tool produced no result" for r in results]
    finally:
        # never block process teardown on an abandoned/stuck worker (all non-timed-out tasks have
        # already completed by the time we exit the loop, so wait=False is lossless in the normal path).
        pool.shutdown(wait=False)
