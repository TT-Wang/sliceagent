"""Cron scheduler + clock (Kimi cron). Deterministic due-calculation via FakeClock, add/remove/list,
mark_run, JSON persistence round-trip. No model, no pytest. Run: PYTHONPATH=src python tests/test_cron.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.clock import FakeClock  # noqa: E402
from memagent.cron import CronJob, CronScheduler  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def fresh_job_fires_after_one_interval_not_instantly():
    clk = FakeClock(0)
    sch = CronScheduler(clock=clk)
    sch.add(CronJob(id="j", task="run tests", interval_seconds=60))
    assert sch.due() == [], "a just-added job must not fire immediately"
    clk.advance(60)
    assert [j.id for j in sch.due()] == ["j"]


@check
def mark_run_resets_the_interval():
    clk = FakeClock(0)
    sch = CronScheduler(clock=clk)
    sch.add(CronJob(id="j", task="x", interval_seconds=60))
    clk.advance(60); assert sch.due()
    sch.mark_run("j")
    assert sch.due() == [], "after mark_run the job waits another interval"
    clk.advance(59); assert sch.due() == []
    clk.advance(1); assert [j.id for j in sch.due()] == ["j"]


@check
def disabled_jobs_never_fire_and_remove_works():
    clk = FakeClock(0)
    sch = CronScheduler(clock=clk)
    sch.add(CronJob(id="off", task="x", interval_seconds=10, enabled=False))
    clk.advance(1000)
    assert sch.due() == []
    assert sch.remove("off") is True and sch.remove("off") is False
    assert sch.list() == []


@check
def persistence_round_trip_preserves_due_state():
    clk = FakeClock(100)
    sch = CronScheduler(clock=clk)
    sch.add(CronJob(id="j", task="nightly", interval_seconds=3600))
    path = os.path.join(tempfile.mkdtemp(prefix="cron-"), "cron.json")
    sch.save(path)
    clk2 = FakeClock(100 + 3600)
    sch2 = CronScheduler.load(path, clock=clk2)
    assert [j.task for j in sch2.list()] == ["nightly"]
    assert [j.id for j in sch2.due()] == ["j"], "due-state survives save/load"


@check
def corrupt_or_missing_file_is_safe():
    assert CronScheduler.load("/nope/does/not/exist.json").list() == []
    p = os.path.join(tempfile.mkdtemp(prefix="cron-"), "bad.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write("not json")
    assert CronScheduler.load(p).list() == []


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
