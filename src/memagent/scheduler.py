"""Conflict-aware tool scheduler (ported from Kimi Code's tool-scheduler).

Runs a batch of tool calls with maximum safe concurrency: non-conflicting calls
overlap, conflicting calls serialize, and results are returned in PROVIDER ORDER
(deterministic for the model) regardless of completion order.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

from .access import Accesses, conflict

Task = tuple[Accesses, Callable[[], str]]


def run_scheduled(tasks: list[Task], max_workers: int = 8) -> list[str]:
    n = len(tasks)
    if n == 0:
        return []
    if n == 1:
        try:
            return [tasks[0][1]()]
        except Exception as e:
            return [f"Error: {e}"]

    results: list[str | None] = [None] * n
    accesses = [t[0] for t in tasks]
    pending = list(range(n))
    running: dict = {}  # future -> index

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
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
                selected_acc.append(a)
                pending.remove(idx)
            if running:
                done, _ = wait(list(running.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = running.pop(fut)
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        results[idx] = f"Error: {e}"
    return results  # type: ignore[return-value]
