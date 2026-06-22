"""Scheduled tasks (borrowed from Kimi cron).

A CronScheduler holds interval-based jobs and reports which are DUE given an injected Clock; the host runs
a due job as a NORMAL user turn (so cron sits ABOVE the kernel — it never touches the slice/loop; a fired
job just becomes another task). Jobs persist as JSON (the established per-session store pattern). Time
comes from the Clock so due-calculation is deterministically testable.

Interval-based ("every N seconds") — the common case; full cron-expression parsing is a future extension.
Experimental: wire the cron tool/firing behind flags.enabled("cron") (product breadth, off by default).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from .clock import SYSTEM_CLOCK


@dataclass
class CronJob:
    id: str
    task: str                  # the prompt/task to run when the job fires
    interval_seconds: float    # fire every N seconds
    last_run: float = 0.0      # epoch of the last fire (0 = freshly added, stamped on add)
    enabled: bool = True


class CronScheduler:
    """Holds jobs, reports which are due, marks runs, and persists. `clock` is injected (testability)."""

    def __init__(self, jobs: list[CronJob] | None = None, clock=SYSTEM_CLOCK):
        self.jobs: dict[str, CronJob] = {j.id: j for j in (jobs or [])}
        self.clock = clock

    def add(self, job: CronJob) -> CronJob:
        if not job.last_run:                       # a fresh job first fires ONE interval from now (not instantly)
            job.last_run = self.clock.now()
        self.jobs[job.id] = job
        return job

    def remove(self, job_id: str) -> bool:
        return self.jobs.pop(job_id, None) is not None

    def list(self) -> list[CronJob]:
        return list(self.jobs.values())

    def due(self) -> list[CronJob]:
        now = self.clock.now()
        return [j for j in self.jobs.values()
                if j.enabled and (now - j.last_run) >= j.interval_seconds]

    def mark_run(self, job_id: str) -> None:
        j = self.jobs.get(job_id)
        if j:
            j.last_run = self.clock.now()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(j) for j in self.jobs.values()], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, clock=SYSTEM_CLOCK) -> "CronScheduler":
        if not os.path.exists(path):
            return cls(clock=clock)
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            jobs = [CronJob(**{k: r[k] for k in ("id", "task", "interval_seconds", "last_run", "enabled")
                               if k in r}) for r in raw]
        except Exception:  # noqa: BLE001 — a corrupt cron file must not crash startup
            return cls(clock=clock)
        return cls(jobs, clock=clock)
