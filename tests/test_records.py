"""Records journal + UsageRecorder. Append-only typed JSONL, robust reads,
per-turn usage journaled on TurnEnd, simple cost aggregation. No model, no pytest.
Run: PYTHONPATH=src python tests/test_records.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import StepEnd, TurnEnd  # noqa: E402
from memagent.records import Journal, UsageRecorder, total_usage  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _journal():
    return Journal("sess-1", root=tempfile.mkdtemp(prefix="rec-"))


@check
def append_and_read_typed_records():
    j = _journal()
    j.record("usage", turn=1, prompt_tokens=100)
    j.record("permission", mode="yolo")
    j.record("usage", turn=2, prompt_tokens=50)
    assert len(j.read()) == 3
    assert [r["turn"] for r in j.read("usage")] == [1, 2]
    assert j.read("permission")[0]["mode"] == "yolo"


@check
def missing_file_and_corrupt_line_are_safe():
    j = Journal("nope", root=tempfile.mkdtemp(prefix="rec-"))
    assert j.read() == []                                  # missing file → empty
    with open(j.path, "w", encoding="utf-8") as f:
        f.write('{"type":"usage","turn":1}\n')
        f.write("not json at all\n")                       # corrupt line skipped, not fatal
        f.write('{"type":"usage","turn":2}\n')
    assert [r["turn"] for r in j.read("usage")] == [1, 2]


@check
def usage_recorder_journals_on_turn_end_only():
    j = _journal()
    rec = UsageRecorder(j, model="kimi-k2.7-code")
    rec(StepEnd(1, {"prompt_tokens": 9}, "tool_use"))     # per-step → NOT journaled
    rec(TurnEnd("end_turn", 2, {"prompt_tokens": 1000, "completion_tokens": 30, "input_other": 200}))
    rec(TurnEnd("end_turn", 1, {"prompt_tokens": 500, "completion_tokens": 10}))
    recs = j.read("usage")
    assert len(recs) == 2 and [r["turn"] for r in recs] == [1, 2]
    assert recs[0]["model"] == "kimi-k2.7-code" and recs[0]["input_other"] == 200


@check
def total_usage_aggregates_per_model():
    j = _journal()
    rec = UsageRecorder(j, model="m1")
    rec(TurnEnd("end_turn", 1, {"prompt_tokens": 100, "completion_tokens": 10}))
    rec(TurnEnd("end_turn", 1, {"prompt_tokens": 200, "completion_tokens": 20}))
    tot = total_usage(j)
    assert tot["m1"]["prompt_tokens"] == 300 and tot["m1"]["completion_tokens"] == 30 and tot["m1"]["turns"] == 2


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
