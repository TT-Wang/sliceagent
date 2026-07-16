"""User-report continuity and action-epoch behavior. No model, no pytest.

Run: PYTHONPATH=src python tests/test_trajectory.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.memory import NullMemory  # noqa: E402
from sliceagent.pfc import Slice  # noqa: E402
from sliceagent.regions import capture_user_report, is_user_report  # noqa: E402
from sliceagent.seed import render_slice  # noqa: E402
from sliceagent.session import Session  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def user_report_positive_signals():
    for msg in [
        "it can't play at all",
        "it doesn't work",
        "still broken",
        "cd: no such file or directory",
        "I get a Traceback when I run it",
        "the build fails now",
        "command not found",
        "this won't run",
        "it isn't working",
    ]:
        assert is_user_report(msg), f"should be a report: {msg!r}"


@check
def user_report_negative_signals():
    for msg in [
        "add a docstring to add()",
        "now also handle the empty case",
        "how do I start it?",
        "make it faster",
        "great, thanks!",
        "refactor the parser into two functions",
    ]:
        assert not is_user_report(msg), f"should NOT be a report: {msg!r}"


@check
def capture_user_report_stores_and_bounds():
    state = Slice()
    state.reset("build the snake game")
    assert capture_user_report(state, "it can't play at all") is True
    assert state.open_report == "it can't play at all"
    assert capture_user_report(state, "thanks, looks good") is False
    assert state.open_report == "it can't play at all"
    assert capture_user_report(state, "now it crashes on launch") is True
    assert state.open_report == "now it crashes on launch"
    capture_user_report(state, "it doesn't work " + "x" * 5000)
    assert len(state.open_report) <= 280


@check
def open_user_report_renders_as_blocker():
    state = Slice()
    state.reset("snake game")
    state.findings = ["**Done — built a working snake game**"]
    state.open_report = "it can't play at all"
    out = render_slice(state, "(no files opened yet)")
    assert "OPEN USER REPORT" in out
    assert "it can't play at all" in out
    assert out.index("OPEN USER REPORT") > out.index("**Done")
    assert out.index("OPEN USER REPORT") > out.index("# YOUR NOTES")


@check
def open_user_report_survives_continue_topic():
    session = Session(NullMemory(), "s-test")
    session.new_topic("build a news aggregator")
    session.active().findings = ["built the aggregator"]
    session.continue_topic("it can not play at all")
    assert session.active().open_report == "it can not play at all"
    session.continue_topic("also add a --count flag")
    assert session.active().open_report == "it can not play at all"
    session.new_topic("unrelated: write a haiku")
    assert session.active().open_report == ""


@check
def continue_topic_demotes_epoch_not_clears():
    session = Session(NullMemory(), "s-test")
    session.new_topic("implement add()")
    state = session.active()
    state.action_log = {
        "run_command `python news.py`": {"count": 2, "failing": True, "last": "boom"},
    }
    state.since_edit = 5
    state.last_error = "boom"
    session.continue_topic("now add a docstring")
    continued = session.active()
    assert continued.action_log["run_command `python news.py`"]["count"] == 2
    assert continued.action_log["run_command `python news.py`"]["failing"] is False
    assert continued.since_edit == 0 and continued.last_error == ""


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {error!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
