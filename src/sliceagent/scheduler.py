"""Provider-ordered tool execution.

Only consecutive, explicitly pure reads form a parallel wave. Every mutation and
unknown tool is a barrier. This intentionally gives up speculative reordering: a
later read can never overtake an intervening write.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable

from .access import AllAccess, FileAccess, ReadAllAccess
from .execution import (ToolInvocation, ToolOutcome, ToolPurity, ToolStatus,
                        coerce_tool_status)


@dataclass(frozen=True)
class ScheduledTool:
    invocation: ToolInvocation
    purity: ToolPurity
    run: Callable[[], ToolOutcome]
    on_start: Callable[[], None] | None = None
    timeout_safe: bool = True


def _announce(task: ScheduledTool) -> None:
    if task.on_start is None:
        return
    # The dispatcher itself isolates presentation observers. Any exception escaping it therefore came
    # from a required pre-dispatch journal/reducer and must stop execution before the tool can have effects.
    task.on_start()


def _failed(task: ScheduledTool, error: Exception) -> ToolOutcome:
    return ToolOutcome(task.invocation, ToolStatus.FAILED, f"Error: {error}")


def _boundary_error(task: ScheduledTool, error: Exception) -> ToolOutcome:
    """Project an exception at the last execution boundary without inventing settlement.

    A read exception is an ordinary failure. An UNKNOWN/EFFECTFUL call may have changed state before the
    exception crossed this boundary, so only INDETERMINATE is honest and it must close later barriers.
    """
    if task.purity is ToolPurity.PURE_READ:
        return _failed(task, error)
    return ToolOutcome(
        task.invocation, ToolStatus.INDETERMINATE,
        f"Error: {error} (the operation may have applied side effects before raising)",
    )


def _execute(task: ScheduledTool) -> ToolOutcome:
    try:
        result = task.run()
    except Exception as error:  # interrupts still propagate
        return _boundary_error(task, error)
    if not isinstance(result, ToolOutcome):
        return _boundary_error(task, TypeError("scheduled tool did not return ToolOutcome"))
    return result


def _cancelled(task: ScheduledTool, reason: str) -> ToolOutcome:
    return ToolOutcome(task.invocation, ToolStatus.CANCELLED, f"Error: tool not run — {reason}")


def _indeterminate(task: ScheduledTool, timeout: float) -> ToolOutcome:
    return ToolOutcome(
        task.invocation,
        ToolStatus.INDETERMINATE,
        (f"Error: tool timed out after {timeout:g}s (cancellation was not confirmed; "
         "it may still be running and its side effects are indeterminate)"),
    )


def _run_wave(tasks: list[ScheduledTool], *, max_workers: int, timeout: float | None) -> list[ToolOutcome]:
    if not tasks:
        return []
    if len(tasks) == 1 and timeout is None:
        _announce(tasks[0])
        return [_execute(tasks[0])]

    pool = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(tasks))))
    futures = []
    try:
        for task in tasks:
            _announce(task)
            futures.append(pool.submit(_execute, task))
        if timeout is None:
            return [future.result() for future in futures]

        done, _ = wait(futures, timeout=timeout)
        outcomes: list[ToolOutcome] = []
        for task, future in zip(tasks, futures):
            if future in done:
                outcomes.append(future.result())
            elif future.cancel():
                outcomes.append(_cancelled(task, "deadline elapsed before execution started"))
            else:
                # Python cannot kill a running thread. Reporting FAILED/CANCELLED here would assert a stop
                # that the runtime did not prove, so dependent waves must remain blocked.
                outcomes.append(_indeterminate(task, timeout))
        return outcomes
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_ordered(
    tasks: list[ScheduledTool],
    *,
    max_workers: int = 8,
    timeout: float | None = None,
    on_outcomes: Callable[[list[ToolOutcome]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[ToolOutcome]:
    """Run pure-read waves and ordered barriers, preserving provider result order.

    An indeterminate invocation stops all later waves. Calls that have not started receive
    a proven ``cancelled`` outcome so every provider invocation still has one reply.
    """
    outcomes: list[ToolOutcome] = []
    i = 0
    while i < len(tasks):
        if should_cancel is not None and should_cancel():
            cancelled = [_cancelled(task, "turn cancellation requested") for task in tasks[i:]]
            if on_outcomes is not None:
                on_outcomes(cancelled)
            outcomes.extend(cancelled)
            break
        if tasks[i].purity is ToolPurity.PURE_READ:
            end = i + 1
            while end < len(tasks) and tasks[end].purity is ToolPurity.PURE_READ:
                end += 1
            wave = tasks[i:end]
        else:
            end = i + 1
            wave = tasks[i:end]

        # Python threads cannot prove cancellation. Applying an outer deadline to an effectful/unknown
        # handler would let it mutate after the turn checkpoint. Such tools must use their own cancellable
        # subprocess/protocol timeout; the generic thread deadline is safe only for declared pure reads.
        wave_timeout = (timeout if all(
            task.purity is ToolPurity.PURE_READ and task.timeout_safe for task in wave
        ) else None)
        wave_outcomes = _run_wave(wave, max_workers=max_workers, timeout=wave_timeout)
        # Required publication/reduction happens at every barrier. If it fails, no later mutation starts.
        if on_outcomes is not None:
            on_outcomes(wave_outcomes)
        outcomes.extend(wave_outcomes)
        if any(out.status is ToolStatus.INDETERMINATE for out in wave_outcomes):
            reason = "an earlier invocation has unresolved side effects"
            cancelled = [_cancelled(task, reason) for task in tasks[end:]]
            # These calls never receive ToolStarted because they did not execute, but they still require one
            # durable logical outcome/provider reply. Publish them through the same required journal/reducer
            # boundary; otherwise run_tool_batch retains ``None`` holes and the recovery log omits calls.
            if on_outcomes is not None and cancelled:
                on_outcomes(cancelled)
            outcomes.extend(cancelled)
            break
        i = end
    return outcomes


# Legacy surface -----------------------------------------------------------------

Task = tuple[list, Callable[[], str]]


def _purity_from_accesses(accesses: list) -> ToolPurity:
    """Compatibility inference for callers that predate registry purity metadata."""
    for access in accesses:
        if isinstance(access, (AllAccess,)):
            return ToolPurity.UNKNOWN
        if isinstance(access, FileAccess) and access.operation in ("write", "readwrite"):
            return ToolPurity.EFFECTFUL
        if not isinstance(access, (FileAccess, ReadAllAccess)):
            return ToolPurity.UNKNOWN
    return ToolPurity.PURE_READ


def run_scheduled(tasks: list[Task], max_workers: int = 8, timeout: float | None = None) -> list[str]:
    """Backward-compatible string projection over the ordered typed scheduler."""
    scheduled: list[ScheduledTool] = []
    for index, (accesses, fn) in enumerate(tasks):
        invocation = ToolInvocation(f"legacy_{index}", "legacy", {}, index)

        def execute(call=fn, inv=invocation):
            out = call()
            text = "" if out is None else str(out)
            explicit = getattr(out, "status", None)
            if explicit is None:
                explicit = getattr(out, "ok", None)
            return ToolOutcome(inv, coerce_tool_status(explicit), text,
                               tuple(getattr(out, "effects", ()) or ()))

        scheduled.append(ScheduledTool(invocation, _purity_from_accesses(accesses), execute))
    return [out.text for out in run_ordered(scheduled, max_workers=max_workers, timeout=timeout)]


__all__ = ["ScheduledTool", "run_ordered", "run_scheduled"]
