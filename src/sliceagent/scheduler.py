"""Provider-ordered tool execution.

Only consecutive, explicitly pure reads form a parallel wave. Every mutation and
unknown tool is a barrier. This intentionally gives up speculative reordering: a
later read can never overtake an intervening write.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

from .access import AllAccess, FileAccess, ReadAllAccess
from .execution import (ToolEffect, ToolInvocation, ToolOutcome, ToolPurity, ToolStatus,
                        coerce_tool_status)


@dataclass(frozen=True)
class ScheduledTool:
    invocation: ToolInvocation
    purity: ToolPurity
    run: Callable[[], ToolOutcome]
    on_start: Callable[[], None] | None = None
    timeout_safe: bool = True
    prepare: Callable[[], ToolOutcome | None] | None = None
    # Production dispatch uses this lease-aware form. A deadline may expire while a required journal is
    # blocked; the callback can then stop before emitting the next lifecycle edge, and the handler never runs.
    on_start_guarded: Callable[[Callable[[], bool]], None] | None = None
    # Presentation-only admission signal. It never means the handler started and failures are isolated: queue
    # visibility must not become a new execution/journal gate.
    on_queued: Callable[[str], None] | None = None
    # Lifecycle children have a cancellable provider/tool loop behind this scheduler worker. The scheduler
    # signals only tasks that crossed (or may have partially crossed) the start boundary, then waits the
    # task-declared bounded close grace before deciding timed-out-vs-indeterminate truth.
    request_cancel: Callable[[str], None] | None = None
    cancel_grace: float = 0.0


def _announce(task: ScheduledTool, abandoned: Callable[[], bool] | None = None) -> None:
    if task.on_start_guarded is not None:
        task.on_start_guarded(abandoned or (lambda: False))
        return
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
    # ``on_start`` is the last durable boundary before the handler itself. Keeping it inside the worker is
    # what lets a queued future that is successfully cancelled remain honestly "not started". Exceptions
    # still propagate: an unavailable required journal must prevent the handler and the rest of the batch.
    _announce(task)
    return _execute_announced(task)


def _execute_announced(task: ScheduledTool) -> ToolOutcome:
    """Execute a task whose durable start boundary has already been published."""
    try:
        result = task.run()
    except Exception as error:  # interrupts still propagate
        return _boundary_error(task, error)
    if not isinstance(result, ToolOutcome):
        return _boundary_error(task, TypeError("scheduled tool did not return ToolOutcome"))
    return result


def _cancelled(task: ScheduledTool, reason: str) -> ToolOutcome:
    return ToolOutcome(task.invocation, ToolStatus.CANCELLED, f"Not run: {reason}")


def _late_read(task: ScheduledTool, timeout: float | None, *, cancelled: bool) -> ToolOutcome:
    """A declared pure read settled only after its cancellation/deadline cutoff."""
    if cancelled:
        detail = "turn cancellation was requested; it settled afterward, and the late result was discarded"
    else:
        detail = (f"it exceeded its {timeout:g}s deadline, settled during the bounded grace period, "
                  "and the late result was discarded")
    return ToolOutcome(
        task.invocation,
        ToolStatus.FAILED,
        f"Error: read-only tool {detail}",
    )


def _indeterminate_read(task: ScheduledTool, timeout: float | None, *, cancelled: bool) -> ToolOutcome:
    """A declared pure read is still running after both its deadline and bounded grace period."""
    if cancelled:
        detail = "was still running when turn cancellation was requested"
    else:
        detail = f"exceeded its {timeout:g}s deadline and is still running after the bounded grace period"
    return ToolOutcome(
        task.invocation,
        ToolStatus.INDETERMINATE,
        f"Error: read-only tool {detail}; its final outcome is indeterminate",
    )


def _lifecycle_timeout_effects(task: ScheduledTool, settled: ToolOutcome) -> tuple[ToolEffect, ...]:
    """Retain real child effects while refining the late lifecycle result with a typed timeout cause."""
    uncertain = task.purity is not ToolPurity.PURE_READ
    operational_status = "indeterminate" if uncertain else "failed"
    effects: list[ToolEffect] = []
    found_child = False
    for effect in settled.effects:
        if effect.kind != "child_artifact":
            effects.append(effect)
            continue
        found_child = True
        payload = dict(effect.payload or {})
        payload.update({
            "status": operational_status,
            "operational_status": operational_status,
            "stop_reason": "indeterminate" if uncertain else "error",
            "stop_cause": "delegation_timeout",
            "partial": bool(payload.get("partial") or payload.get("artifact_id")),
        })
        effects.append(ToolEffect(effect.id, effect.kind, payload))
    if not found_child:
        effects.append(ToolEffect(
            f"{task.invocation.id}:delegation-timeout",
            "child_artifact",
            {
                "artifact_id": "",
                "kind": str(task.invocation.args.get("agent") or ""),
                "status": operational_status,
                "operational_status": operational_status,
                "source_coverage_status": "not_assessed",
                "stop_reason": "indeterminate" if uncertain else "error",
                "stop_cause": "delegation_timeout",
                "partial": False,
            },
        ))
    return tuple(effects)


def _closed_lifecycle_timeout(
    task: ScheduledTool, settled: ToolOutcome, timeout: float | None,
) -> ToolOutcome:
    """The deadline won, but the child/provider/tool stack proved physical closure during grace."""
    uncertain = task.purity is not ToolPurity.PURE_READ
    suffix = (
        "; the writable child may have applied workspace effects before cancellation"
        if uncertain else ""
    )
    return ToolOutcome(
        task.invocation,
        ToolStatus.INDETERMINATE if uncertain else ToolStatus.FAILED,
        (f"Error: subagent exceeded its {timeout:g}s delegation deadline; cancellation closed the child "
         f"before the bounded grace expired{suffix}"),
        _lifecycle_timeout_effects(task, settled),
    )


def _closed_lifecycle_cancel(task: ScheduledTool, settled: ToolOutcome) -> ToolOutcome:
    """Parent cancellation closed the child; discard late deliverables but retain billed model usage."""
    usage = tuple(effect for effect in settled.effects if effect.kind == "model_usage")
    uncertain = task.purity is not ToolPurity.PURE_READ
    return ToolOutcome(
        task.invocation,
        ToolStatus.INDETERMINATE if uncertain else ToolStatus.CANCELLED,
        (
            "Error: parent turn cancellation closed the writable child before publication; "
            "workspace effects may already have applied"
            if uncertain else
            "Not run to completion: parent turn cancellation closed the child before publication"
        ),
        usage,
    )


def _has_committed_child(settled: ToolOutcome) -> bool:
    """A canonical child effect proves that store+parent-reference publication already committed.

    The child boundary emits this effect only after both durable facts exist.  A later scheduler cutoff may
    suppress optional work, but cannot truthfully relabel or detach that already-accepted result.
    """
    return any(
        effect.kind == "child_artifact" and str((effect.payload or {}).get("artifact_id") or "")
        for effect in settled.effects
    )


# Pure reads run on daemon workers even when no deadline is configured. ThreadPoolExecutor workers are
# non-daemon and Python joins them at interpreter exit, so two stuck FIFOs/network mounts used to freeze both
# Ctrl-C and process shutdown in the executor's ``shutdown(wait=True)`` path. Timeout-safe physical readers
# reserve one global slot until they actually exit, bounding abandoned hangs across concurrent turns.
# Lifecycle reads such as explorer children deliberately do not hold those slots across their own nested read
# waves; their separate max-worker/depth limits provide the bound without introducing a recursive semaphore
# deadlock.
_MAX_TIMEOUT_READERS = 32
_TIMEOUT_READER_SLOTS = threading.BoundedSemaphore(_MAX_TIMEOUT_READERS)
_MAX_LIFECYCLE_READERS = 32
_LIFECYCLE_READER_SLOTS = threading.BoundedSemaphore(_MAX_LIFECYCLE_READERS)
# A child owns a full model loop, not one cheap filesystem read. Launching eight broad explorers at once made
# each provider step contend with seven other full-reasoning requests; a real 7-child audit ended 3 succeeded,
# 2 output-truncated, and 2 exhausted their provider timeout retries. Four keeps useful parallelism while
# bounding per-batch provider pressure. The global lifecycle semaphore remains the cross-turn hard ceiling.
_MAX_PARALLEL_LIFECYCLE_WAVE = 4
# Admit the first useful pair immediately, then ramp additional full model loops. This avoids a synchronized
# prompt/cache/connection burst while retaining four-way throughput once the provider has started responding.
_LIFECYCLE_INITIAL_BURST = 2
_LIFECYCLE_LAUNCH_STAGGER_SECONDS = 0.20
_TIMEOUT_POLL_SECONDS = 0.05
_TIMEOUT_GRACE_SECONDS = 0.10
# A permanently abandoned daemon legitimately keeps its physical slot until it exits.  Capacity exhaustion
# must nevertheless be a bounded, typed *not-started* outcome: without this independent wait ceiling, the
# default no-tool-timeout configuration polls forever after enough cancelled FIFO/network-mount reads.
_READER_SLOT_WAIT_SECONDS = 0.10


@dataclass
class _DaemonReadJob:
    task: ScheduledTool
    launched: bool = False
    announcing: bool = False
    entered: bool = False
    abandoned: bool = False
    done: bool = False
    finished_at: float | None = None
    outcome: ToolOutcome | None = None
    error: BaseException | None = None
    queue_announced: bool = False
    slot_released: bool = False


def _job_result(job: _DaemonReadJob) -> ToolOutcome:
    if job.error is not None:
        raise job.error
    if job.outcome is None:
        raise RuntimeError("daemon read worker settled without an outcome")
    return job.outcome


def _run_read_wave(
    tasks: list[ScheduledTool],
    *,
    max_workers: int,
    timeout: float | None,
    should_cancel: Callable[[], bool] | None,
    on_partial: Callable[[list[ToolOutcome]], None] | None = None,
) -> list[ToolOutcome]:
    """Run pure reads on cancellable daemon workers.

    ``entered`` is synchronized with the durable ``on_start`` callback. At a deadline/cancellation cutoff, a
    launched worker that has not crossed that boundary is marked abandoned and can never run or announce
    later; an entered worker is honestly indeterminate if it cannot settle. This prevents post-settlement
    starts from contaminating a later task/workspace through the live dispatcher routers.
    """
    jobs = [_DaemonReadJob(task) for task in tasks]
    condition = threading.Condition()
    reader_slots = _TIMEOUT_READER_SLOTS
    lifecycle_slots = _LIFECYCLE_READER_SLOTS
    worker_limit = max(1, min(max_workers, len(tasks)))
    lifecycle_wave = any(not task.timeout_safe for task in tasks)
    lifecycle_launch_count = 0
    next_lifecycle_launch_at = 0.0
    next_index = 0
    slot_wait_started: float | None = None
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    cutoff_kind = "deadline"
    late: set[int] = set()
    unstarted: set[int] = set()

    def announce_queued(index: int, reason: str) -> None:
        job = jobs[index]
        if job.queue_announced or job.launched:
            return
        job.queue_announced = True
        callback = job.task.on_queued
        if callback is None:
            return
        try:
            callback(reason)
        except Exception:
            pass  # presentation is never allowed to alter scheduling truth

    def worker(index: int, slot: threading.BoundedSemaphore) -> None:
        job = jobs[index]
        outcome = None
        error = None
        needs_announcement = (
            job.task.on_start is not None or job.task.on_start_guarded is not None
        )
        try:
            with condition:
                # The scheduler owns this transition at cutoff. A launched-but-not-entered thread may wake
                # later, but it cannot cross the start boundary or call the handler after abandonment.
                cancelled = should_cancel is not None and should_cancel()
                expired = deadline is not None and time.monotonic() >= deadline
                if job.abandoned or cancelled or expired:
                    job.abandoned = True
                    job.done = True
                    job.finished_at = time.monotonic()
                    condition.notify_all()
                    return
                if not needs_announcement:
                    # With no journal callback there is no opaque boundary to run outside the lock. Cross the
                    # handler-start edge atomically so a deadline can still prove queued work unstarted.
                    job.entered = True
                else:
                    # Reserve start publication under the scheduler lock, then release it before invoking the
                    # potentially blocking journal. A cutoff during publication marks the job abandoned and
                    # therefore prevents the handler from starting.
                    job.announcing = True
            if needs_announcement:
                try:
                    _announce(job.task, lambda: job.abandoned)
                except BaseException as caught:
                    with condition:
                        job.announcing = False
                        job.error = caught
                        job.finished_at = time.monotonic()
                        job.done = True
                        condition.notify_all()
                    return
                with condition:
                    job.announcing = False
                    cancelled = should_cancel is not None and should_cancel()
                    expired = deadline is not None and time.monotonic() >= deadline
                    if job.abandoned or cancelled or expired:
                        # Publication may have crossed one or more durable sinks, so CANCELLED would be a lie
                        # even though the handler is still provably uncalled. Preserve the partial-start
                        # uncertainty and close later barriers; a guarded callback emits no next lifecycle edge.
                        job.abandoned = True
                        job.outcome = ToolOutcome(
                            job.task.invocation,
                            ToolStatus.INDETERMINATE,
                            "Error: tool start publication crossed the cancellation/deadline boundary; "
                            "the handler did not run, but the start record may be partial",
                        )
                        job.finished_at = time.monotonic()
                        job.done = True
                        condition.notify_all()
                        return
                    job.entered = True
            try:
                outcome = _execute_announced(job.task)
            except BaseException as caught:  # preserve the old Future.result() propagation contract
                error = caught
            # Capture physical settlement immediately. Recording the timestamp only after acquiring
            # ``condition`` can misclassify an on-time reader as late when the scheduler owns the lock.
            finished_at = time.monotonic()
            with condition:
                job.outcome = outcome
                job.error = error
                job.finished_at = finished_at
                job.done = True
                condition.notify_all()
        finally:
            # ``done`` means the handler produced an outcome; ``slot_released`` proves the outer physical
            # lifecycle lease is also free. The scheduler waits for both before reporting a closed timeout.
            slot.release()
            with condition:
                job.slot_released = True
                condition.notify_all()

    def active_count() -> int:
        return sum(job.launched and not job.slot_released for job in jobs)

    def cancel_queued_after_indeterminate() -> bool:
        """Close the unadmitted tail after a lifecycle/read result becomes uncertain.

        A worker can return INDETERMINATE while its underlying provider operation is still physically live.
        Reusing the newly freed logical slot would then exceed the scheduler's real concurrency bound.  Jobs
        already launched are allowed to settle; only the sequential, never-launched tail is cancelled.
        """
        nonlocal next_index
        unresolved = any(
            job.done
            and job.error is None
            and job.outcome is not None
            and job.outcome.status is ToolStatus.INDETERMINATE
            for job in jobs[:next_index]
        )
        if not unresolved or next_index >= len(jobs):
            return False
        reason = (
            "an earlier invocation in this wave has an unresolved outcome; "
            "queued execution was not admitted"
        )
        now = time.monotonic()
        for job in jobs[next_index:]:
            job.abandoned = True
            job.done = True
            job.finished_at = now
            job.outcome = _cancelled(job.task, reason)
        next_index = len(jobs)
        condition.notify_all()
        return True

    def launch(index: int) -> bool:
        nonlocal lifecycle_launch_count, next_lifecycle_launch_at
        job = jobs[index]
        # Explorer-child calls run nested read waves and therefore must not hold a physical-reader slot while
        # waiting for those inner reads. They are still daemonized/cancellable at this outer boundary.
        slot = reader_slots if job.task.timeout_safe else lifecycle_slots
        acquired = False
        thread = None
        try:
            acquired = slot.acquire(blocking=False)
            if not acquired:
                return False
            thread = threading.Thread(
                target=worker,
                args=(index, slot),
                name=f"sliceagent-read-{job.task.invocation.id}",
                daemon=True,
            )
            job.launched = True
            thread.start()
        except BaseException:
            # Once Thread.start() created an OS thread, that worker owns the slot and releases it in finally,
            # even if an asynchronous KeyboardInterrupt lands before start() returns to this frame. Before
            # that handoff, launch() still owns the acquired slot and must roll it back.
            worker_owns_slot = thread is not None and thread.ident is not None
            if not worker_owns_slot:
                job.launched = False
                if acquired:
                    slot.release()
            raise
        if not job.task.timeout_safe:
            lifecycle_launch_count += 1
            if lifecycle_launch_count >= _LIFECYCLE_INITIAL_BURST:
                next_lifecycle_launch_at = (
                    time.monotonic() + max(0.0, _LIFECYCLE_LAUNCH_STAGGER_SECONDS)
                )
        return True

    def establish_cutoff(kind: str, cutoff_at: float) -> None:
        nonlocal cutoff_kind, late, unstarted
        cutoff_kind = kind
        late = {
            index for index, job in enumerate(jobs)
            if (job.entered or job.announcing)
            and (job.finished_at is None or job.finished_at > cutoff_at)
        }
        unstarted = {
            index for index, job in enumerate(jobs)
            if ((job.abandoned and job.outcome is None)
                or (not job.entered and not job.announcing and not job.done))
        }
        for index in unstarted:
            jobs[index].abandoned = True
        for index in late:
            if jobs[index].announcing:
                jobs[index].abandoned = True
            callback = jobs[index].task.request_cancel
            if callback is not None:
                try:
                    callback(kind)
                except Exception:
                    pass  # cancellation notification cannot falsify the observed cutoff
        condition.notify_all()

    with condition:
        try:
            # A lifecycle wave intentionally admits only a bounded number of full model loops. Announce every
            # overflow child immediately instead of letting it appear frozen/nonexistent until a sibling exits.
            for index in range(worker_limit, len(jobs)):
                announce_queued(
                    index,
                    "waiting for agent slot" if not jobs[index].task.timeout_safe else "waiting for read slot",
                )
            cutoff = False
            while True:
                cancelled = should_cancel is not None and should_cancel()
                now = time.monotonic()
                expired = deadline is not None and now >= deadline
                if cancelled or expired:
                    establish_cutoff("cancel" if cancelled else "deadline", now if cancelled else deadline)
                    break

                # Harvest typed uncertainty before reusing a worker slot. In particular, a child model
                # watchdog can abandon its local wait while the provider socket remains live; launching a
                # queued sibling here would turn the four-child ceiling into an unbounded physical fan-out.
                cancel_queued_after_indeterminate()

                while next_index < len(jobs) and active_count() < worker_limit:
                    # A fast worker can free capacity while this launch loop is running. Re-check both cutoff
                    # sources before admitting its queued successor so nothing starts after the boundary.
                    cancelled = should_cancel is not None and should_cancel()
                    now = time.monotonic()
                    expired = deadline is not None and now >= deadline
                    if cancelled or expired:
                        establish_cutoff("cancel" if cancelled else "deadline",
                                         now if cancelled else deadline)
                        cutoff = True
                        break
                    if (not jobs[next_index].task.timeout_safe
                            and lifecycle_launch_count >= _LIFECYCLE_INITIAL_BURST
                            and now < next_lifecycle_launch_at):
                        # Do not sleep while holding scheduler truth. The shared condition poll below remains
                        # cancellation/deadline responsive and wakes early when a child settles.
                        break
                    if not launch(next_index):
                        announce_queued(
                            next_index,
                            "waiting for global agent capacity"
                            if not jobs[next_index].task.timeout_safe else "waiting for global read capacity",
                        )
                        if slot_wait_started is None:
                            slot_wait_started = now
                        elif now - slot_wait_started >= _READER_SLOT_WAIT_SECONDS:
                            # No worker crossed the start boundary. Settle every queued sibling locally so a
                            # permanently occupied global pool cannot freeze this or every later turn.
                            for index, job in enumerate(jobs[next_index:], start=next_index):
                                # Every admitted child needs its own visible pre-start row.  Cancelling the
                                # whole tail while announcing only ``next_index`` made its siblings appear to
                                # materialize for the first time as terminal results.  The callback is
                                # idempotent, so children already announced by the local worker-limit pass are
                                # not duplicated here.
                                announce_queued(
                                    index,
                                    "waiting for global agent capacity"
                                    if not job.task.timeout_safe else "waiting for global read capacity",
                                )
                                job.abandoned = True
                                job.done = True
                                job.finished_at = now
                                job.outcome = _cancelled(
                                    job.task,
                                    ("agent capacity remained unavailable before execution started"
                                     if not job.task.timeout_safe else
                                     "reader capacity remained unavailable before execution started"),
                                )
                            next_index = len(jobs)
                        break
                    slot_wait_started = None
                    next_index += 1

                if cutoff:
                    break
                if next_index == len(jobs) and all(
                        job.done and (not job.launched or job.slot_released) for job in jobs):
                    return [_job_result(job) for job in jobs]

                # A slot held by another concurrent wave may become available before this wave's deadline.
                # Poll rather than immediately fabricating a capacity cancellation.
                now = time.monotonic()
                wait_for = _TIMEOUT_POLL_SECONDS
                if (lifecycle_wave and next_index < len(jobs)
                        and not jobs[next_index].task.timeout_safe
                        and lifecycle_launch_count >= _LIFECYCLE_INITIAL_BURST
                        and now < next_lifecycle_launch_at):
                    wait_for = min(wait_for, max(0.0, next_lifecycle_launch_at - now))
                if deadline is not None:
                    wait_for = min(wait_for, max(0.0, deadline - now))
                condition.wait(timeout=wait_for)

            # Ordinary reads retain the short deadline grace. A lifecycle child advertises enough additional
            # time for its cancellable SSE/tool stack to prove closure. Parent cancellation uses the same lease:
            # returning immediately would recreate the late-provider/occupied-slot bug under a different cause.
            lifecycle_graces = [
                max(0.0, float(jobs[index].task.cancel_grace or 0.0))
                for index in late if jobs[index].task.request_cancel is not None
            ]
            grace = (
                max([_TIMEOUT_GRACE_SECONDS, *lifecycle_graces])
                if cutoff_kind == "deadline" or lifecycle_graces else 0.0
            )
            if grace > 0:
                grace_deadline = time.monotonic() + grace
                while any(
                    not jobs[index].done or not jobs[index].slot_released
                    for index in late
                ):
                    remaining = grace_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    condition.wait(timeout=min(_TIMEOUT_POLL_SECONDS, remaining))
            still_running = {
                index for index in late
                if not jobs[index].done or not jobs[index].slot_released
            }
        except BaseException:
            # SIGINT/KeyboardInterrupt can arrive at any polling point. Freeze the same synchronized start
            # boundary before propagating it: queued/launched workers cannot announce or execute afterward;
            # already-entered lifecycle children must receive the same cancellation lease as an ordinary
            # parent cancel. Otherwise their provider call and global lifecycle slot outlive the sealed turn.
            establish_cutoff("interrupt", time.monotonic())
            # HARVEST before propagating: jobs that already SETTLED WITH A REAL OUTCOME are facts — their
            # side effects (e.g. a child's sealed report) already exist on disk. Re-raising without
            # surfacing them made an Esc during a fan-out erase every finished sibling ("7 reports ready"
            # → "10 state unknown") while one straggler ran: the caller's interrupt path can only
            # synthesize indeterminate for outcomes it never received. Best-effort, lock-held (we own
            # `condition` here); only done+outcome jobs harvest — errored/unfinished keep interrupt semantics.
            if on_partial is not None:
                settled = [job.outcome for job in jobs
                           if job.done and job.outcome is not None and job.error is None]
                if settled:
                    try:
                        on_partial(settled)
                    except Exception:  # noqa: BLE001 — harvesting must never mask the interrupt itself
                        pass
            raise

    if cutoff_kind == "cancel":
        not_started_reason = "turn cancellation requested before execution started"
    else:
        not_started_reason = "deadline elapsed before execution started"

    outcomes: list[ToolOutcome] = []
    for index, job in enumerate(jobs):
        if index in unstarted:
            outcomes.append(_cancelled(job.task, not_started_reason))
        elif index in still_running:
            outcomes.append(_indeterminate_read(
                job.task, timeout, cancelled=cutoff_kind == "cancel",
            ))
        elif index in late:
            # The deadline invalidates an otherwise ordinary late result, but it cannot erase stronger typed
            # uncertainty (or swallow an interrupt) reported by the execution boundary itself. In particular,
            # rewriting INDETERMINATE to FAILED would let a later mutation overtake an unresolved outcome.
            settled = _job_result(job)
            if settled.status is ToolStatus.INDETERMINATE:
                outcomes.append(settled.with_text(
                    f"{settled.text} (the {'delegation deadline' if cutoff_kind == 'deadline' else 'parent cancellation'} "
                    "also elapsed without a proven clean child result)"
                ))
            elif _has_committed_child(settled):
                # Canonical seal + parent-ref publication is the child commit point. The child checks its
                # cancellation lease before reaching it, and emits this effect only afterward. A cutoff that
                # races with post-commit journal cleanup/mirroring therefore loses to the settled durable fact.
                outcomes.append(settled)
            elif not job.task.timeout_safe and job.task.request_cancel is not None:
                outcomes.append(
                    _closed_lifecycle_timeout(job.task, settled, timeout)
                    if cutoff_kind == "deadline" else
                    _closed_lifecycle_cancel(job.task, settled)
                )
            else:
                outcomes.append(_late_read(
                    job.task, timeout, cancelled=cutoff_kind == "cancel",
                ))
        else:
            outcomes.append(_job_result(job))
    return outcomes


def _run_wave(
    tasks: list[ScheduledTool],
    *,
    max_workers: int,
    timeout: float | None,
    should_cancel: Callable[[], bool] | None = None,
    on_partial: Callable[[list[ToolOutcome]], None] | None = None,
) -> list[ToolOutcome]:
    if not tasks:
        return []
    # A spawned child is a cancellable lifecycle operation even when its own tool surface is writable.  Its
    # workspace purity still controls ordering (writable children remain single barriers), but it must use the
    # same daemon/cancellation lease as an explorer or an advanced child can ignore the advertised delegation
    # ceiling and freeze the parent forever. ``timeout_safe=False`` is reserved for these lifecycle handlers.
    lifecycle_wave = all(
        not task.timeout_safe and task.request_cancel is not None for task in tasks
    )
    if all(task.purity is ToolPurity.PURE_READ for task in tasks) or lifecycle_wave:
        wave_workers = max_workers
        if lifecycle_wave or any(not task.timeout_safe for task in tasks):
            wave_workers = min(wave_workers, _MAX_PARALLEL_LIFECYCLE_WAVE)
        return _run_read_wave(
            tasks, max_workers=wave_workers, timeout=timeout, should_cancel=should_cancel,
            on_partial=on_partial,
        )
    if len(tasks) != 1:
        raise RuntimeError("an effectful scheduler wave must contain exactly one barrier")
    return [_execute(tasks[0])]


def run_ordered(
    tasks: list[ScheduledTool],
    *,
    max_workers: int = 8,
    timeout: float | None = None,
    lifecycle_timeout: float | None = None,
    on_outcomes: Callable[[list[ToolOutcome]], None] | None = None,
    on_wave_ready: Callable[[list[ScheduledTool]], None] | None = None,
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
            # A lifecycle read (notably a child agent) must be allowed to seal rather than being abandoned by
            # the generic reader deadline. Keep it in a separate provider-ordered wave so its exemption cannot
            # silently disable AGENT_TOOL_TIMEOUT for an adjacent ordinary read.
            while (end < len(tasks)
                   and tasks[end].purity is ToolPurity.PURE_READ
                   and tasks[end].timeout_safe == tasks[i].timeout_safe):
                end += 1
            wave = tasks[i:end]
        else:
            end = i + 1
            wave = tasks[i:end]

        # Python threads cannot prove cancellation. Applying an outer deadline to an arbitrary
        # effectful/unknown handler would let it mutate after the turn checkpoint, so ordinary barriers must
        # use their own cancellable subprocess/protocol timeout. Spawned children are the narrow exception:
        # they carry an explicit cancellation lease and bounded close protocol even when their child tool
        # surface is writable; `_run_wave` keeps such a child serialized while enforcing lifecycle_timeout.
        # Prepare at the barrier, after the previous wave's outcomes were published. This preserves lifecycle
        # ordering and lets the narrow catastrophic safeguard inspect each physical execution before it starts.
        prepared: dict[int, ToolOutcome] = {}
        ready: list[ScheduledTool] = []
        for task in wave:
            outcome = task.prepare() if task.prepare is not None else None
            if outcome is None:
                ready.append(task)
            else:
                prepared[id(task)] = outcome
        # The admission callback also needs to observe an entirely preflight-cancelled wave so it can release
        # reservations that were proved never started before allocating later barriers.
        if on_wave_ready is not None:
            on_wave_ready(ready)
        if should_cancel is not None and should_cancel():
            cancelled_ready = {id(task): _cancelled(task, "turn cancellation requested") for task in ready}
            wave_outcomes = [prepared.get(id(task), cancelled_ready.get(id(task))) for task in wave]
            if any(outcome is None for outcome in wave_outcomes):
                raise RuntimeError("scheduler lost a cancelled tool outcome")
            if on_outcomes is not None:
                on_outcomes(wave_outcomes)
            outcomes.extend(wave_outcomes)
            tail = [_cancelled(task, "turn cancellation requested") for task in tasks[end:]]
            if on_outcomes is not None and tail:
                on_outcomes(tail)
            outcomes.extend(tail)
            break
        # Every pure-read wave carries a wall-clock ceiling so a wedged reader/child can never freeze the
        # turn. Ordinary reads use the SHORT generic reader deadline (`timeout`). A lifecycle/delegation read
        # is exempt from that short deadline — it must be allowed to SEAL its report rather than be abandoned
        # mid-write — but is still bounded by the GENEROUS `lifecycle_timeout`; without it a non-terminating
        # child hangs the parent turn forever (only Ctrl-C recovers). An effectful/unknown barrier keeps no
        # outer deadline. A writable lifecycle child remains a single ordered barrier but uses its explicit
        # cancellation/closure lease; an arbitrary effectful tool still enforces its own timeout.
        # ``timeout_safe=False`` is the scheduler's lifecycle marker for pure-read children as well as
        # writable children.  A writable lifecycle needs ``request_cancel`` before it may use the daemon
        # path (see ``_run_wave``), but a pure-read lifecycle has always been cancellable at the scheduler
        # boundary and must retain its generous lifecycle ceiling even when an older/embedder task omitted
        # the optional callback.  Requiring ``request_cancel`` here silently changed that ceiling to the
        # ordinary read timeout (or to no timeout at all) and could freeze the parent forever.
        lifecycle_ready = bool(ready) and all(not task.timeout_safe for task in ready)
        if ready and (
            all(task.purity is ToolPurity.PURE_READ for task in ready) or lifecycle_ready
        ):
            wave_timeout = lifecycle_timeout if lifecycle_ready else timeout
        else:
            wave_timeout = None
        executed: list[ToolOutcome] | None = None
        wave_outcomes: list[ToolOutcome | None] | None = None
        try:
            executed = _run_wave(
                ready, max_workers=max_workers, timeout=wave_timeout, should_cancel=should_cancel,
                # Interrupt harvest: settled real outcomes inside an interrupted wave are surfaced through the
                # ordinary publication callback before the interrupt propagates, so a finished sibling's sealed
                # work is never re-labelled indeterminate by the caller's synthesizer.
                on_partial=on_outcomes,
            )
            executed_by_task = {id(task): outcome for task, outcome in zip(ready, executed)}
            wave_outcomes = [prepared.get(id(task), executed_by_task.get(id(task))) for task in wave]
            if any(outcome is None for outcome in wave_outcomes):
                raise RuntimeError("scheduler lost a prepared tool outcome")
            # Required publication/reduction happens at every barrier. If it fails, no later mutation starts.
            if on_outcomes is not None:
                on_outcomes(wave_outcomes)  # type: ignore[arg-type]
        except KeyboardInterrupt:
            # A signal can arrive after every handler in the wave returned but before (or during) the first
            # reducer callback. At that point ``executed`` is physical truth: hand the entire provider-ordered
            # wave to the callback once more before propagating the user's interrupt. The execution loop's
            # per-invocation lifecycle acknowledgements make this at-least-once handoff idempotent when the
            # first callback had already delivered a prefix. Never let a second callback failure mask SIGINT.
            if executed is not None and on_outcomes is not None:
                if wave_outcomes is None:
                    executed_by_task = {id(task): outcome for task, outcome in zip(ready, executed)}
                    wave_outcomes = [
                        prepared.get(id(task), executed_by_task.get(id(task))) for task in wave
                    ]
                if all(outcome is not None for outcome in wave_outcomes):
                    try:
                        on_outcomes(wave_outcomes)  # type: ignore[arg-type]
                    except BaseException:  # preserve the original user interrupt after best-effort recovery
                        pass
            raise
        # ``None`` was excluded above; keep the public return and following status checks precisely typed.
        settled_wave = [outcome for outcome in (wave_outcomes or ()) if outcome is not None]
        outcomes.extend(settled_wave)
        if any(out.status is ToolStatus.INDETERMINATE for out in settled_wave):
            reason = "an earlier invocation is still running or has an unresolved outcome"
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
