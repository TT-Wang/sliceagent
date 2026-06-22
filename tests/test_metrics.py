"""CostMetrics — the moat-measuring observer sink (per-turn fresh-input curve, cache-hit rate, error
buckets). Pure/deterministic, no model, no pytest. Run: PYTHONPATH=src python tests/test_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import ApiRetry, SliceTightened, StepEnd, ToolResult, TurnEnd  # noqa: E402
from memagent.metrics import CostMetrics  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _usage(other, cache_read=0, output=0, cache_create=0):
    return {"input_other": other, "input_cache_read": cache_read,
            "input_cache_creation": cache_create, "output": output}


@check
def per_turn_fresh_curve_and_no_double_count():
    m = CostMetrics()
    # turn 1: two steps, fresh 100 + 50
    m(StepEnd(1, _usage(100, cache_read=900, output=20), "tool_use"))
    m(StepEnd(2, _usage(50, cache_read=950, output=10), "tool_use"))
    m(TurnEnd("end_turn", 2, {"prompt_tokens": 2000, "completion_tokens": 30}))  # cumulative total — must NOT be summed
    # turn 2: one step, fresh 60
    m(StepEnd(3, _usage(60, cache_read=1940, output=15), "stop"))
    m(TurnEnd("end_turn", 1, {"prompt_tokens": 2000, "completion_tokens": 15}))
    s = m.summary()
    assert s["per_turn_fresh"] == [150, 60], s["per_turn_fresh"]   # per-turn fresh input, the moat curve
    assert s["input_other"] == 210, s                              # 100+50+60, NOT inflated by TurnEnd totals
    assert s["turns"] == 2 and s["steps"] == 3
    assert s["avg_turn_fresh"] == 105.0 and s["peak_turn_fresh"] == 150


@check
def cache_hit_rate_reflects_cached_reads():
    m = CostMetrics()
    m(StepEnd(1, _usage(100, cache_read=900, output=10), "stop"))
    m(TurnEnd("end_turn", 1, {}))
    s = m.summary()
    # cache_read / (other + cache_read + cache_create) = 900 / 1000
    assert s["cache_hit_rate"] == 0.9, s
    assert s["input_cache_read"] == 900 and s["output"] == 10


@check
def counts_tools_retries_overflows_errors():
    m = CostMetrics()
    m(ToolResult("read_file", {}, "ok", failing=False))
    m(ToolResult("run_command", {}, "boom", failing=True))
    m(ApiRetry(attempt=1, error="429"))
    m(SliceTightened(level=1))
    m.record_error("rate_limit")
    m.record_error("rate_limit")
    m.record_error("timeout")
    s = m.summary()
    assert s["tool_calls"] == 2 and s["tool_failures"] == 1
    assert s["retries"] == 1 and s["overflows"] == 1
    assert s["errors"] == {"rate_limit": 2, "timeout": 1}, s["errors"]


@check
def empty_run_is_safe():
    s = CostMetrics().summary()
    assert s["per_turn_fresh"] == [] and s["avg_turn_fresh"] == 0.0 and s["cache_hit_rate"] == 0.0


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
