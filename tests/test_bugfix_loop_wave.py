"""Regression tests for the loop wave: #11 a truncated/filtered finish PARKS (not a clean TurnEnd),
#13 a None-args tool call serializes as "{}", #14 an empty seed doesn't crash, #56 metrics count parked
turns. No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_loop_wave.py
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.loop import run_turn, _assistant_message  # noqa: E402
from memagent.events import StepEnd, TurnEnd, TurnInterrupted  # noqa: E402
from memagent.metrics import CostMetrics  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _LLM:
    def __init__(self, resp):
        self._resp = resp
        self.is_retryable = lambda e: False
    def complete(self, messages, schemas):
        return self._resp


class _Tools:
    def schemas(self):
        return []
    def accesses(self, n, a):
        return []
    def run(self, n, a):
        return "ok"


def _run(resp, seed=None):
    events = []
    seed = seed if seed is not None else [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    r = run_turn(build_slice=lambda: list(seed), llm=_LLM(resp), tools=_Tools(), dispatch=events.append)
    return r, events


@check
def max_tokens_parks_not_clean_end():  # #11
    r, ev = _run(NS(content="partial", tool_calls=[], finish_reason="length", usage={}))
    assert r.stop_reason == "max_tokens", r.stop_reason
    assert any(isinstance(e, TurnInterrupted) for e in ev), "max_tokens must PARK"
    assert not any(isinstance(e, TurnEnd) for e in ev), "must NOT seal a truncated reply as a clean turn"


@check
def content_filter_parks():  # #11
    r, ev = _run(NS(content="", tool_calls=[], finish_reason="content_filter", usage={}))
    assert r.stop_reason == "filtered", r.stop_reason
    assert any(isinstance(e, TurnInterrupted) for e in ev)


@check
def empty_seed_does_not_crash():  # #14
    r, _ = _run(NS(content="hi", tool_calls=[], finish_reason="stop", usage={}), seed=[])
    assert r.stop_reason != "error", f"empty seed must not crash into an error park: {r.stop_reason}"


@check
def assistant_message_handles_none_args():  # #13
    msg = _assistant_message(NS(content="", tool_calls=[NS(id="1", name="x", args=None)]))
    assert msg["tool_calls"][0]["function"]["arguments"] == "{}", msg


@check
def metrics_count_parked_turns():  # #56
    m = CostMetrics()
    m(StepEnd(1, {"input_other": 100}, "max_tokens"))
    m(TurnInterrupted("max_tokens", message="cut off"))
    assert m.turns == 1, m.turns
    assert m.per_turn_fresh == [100], m.per_turn_fresh
    assert m._turn_fresh == 0, "accumulator must reset so it doesn't bleed into the next turn"
    # a second (clean) turn still tracks independently
    m(StepEnd(2, {"input_other": 50}, "end_turn"))
    m(TurnEnd("end_turn", 2, {}))
    assert m.per_turn_fresh == [100, 50], m.per_turn_fresh
    assert m.summary()["errors"].get("park:max_tokens") == 1


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
