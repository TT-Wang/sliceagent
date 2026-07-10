"""ask_user / confirm must PAUSE the live turn-spinner before reading input — otherwise the Rich Live
region owns the terminal and the user's typed answer is not echoed (the 'my answer is invisible' bug).
No model. Run: PYTHONPATH=src python tests/test_ask_echo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _FakeConsole:
    """Stands in for the prompt console: records that input() was called, returns a fixed answer."""
    def __init__(self, answer):
        self.answer = answer
        self.inputs = 0
    def print(self, *a, **k):
        pass
    def input(self, *a, **k):
        self.inputs += 1
        return self.answer


@check
def ask_user_pauses_the_live_spinner_before_reading():
    from rich.console import Console
    from sliceagent import tui
    from sliceagent.events import TurnStarted
    sink = tui.RichSink(Console(), {})        # registers itself as the active live sink
    sink(TurnStarted("working"))
    assert sink._status is not None, "precondition: a turn spinner is live"
    fc = _FakeConsole("blue")
    ans = tui.ask_user(fc, "which color?")
    assert ans == "blue" and fc.inputs == 1, "the answer must be read and returned"
    assert sink._status is None, "the live spinner MUST be stopped before input (else no echo)"


@check
def confirm_pauses_the_live_spinner_before_reading():
    from rich.console import Console
    from sliceagent import tui
    from sliceagent.events import TurnStarted
    sink = tui.RichSink(Console(), {})
    sink(TurnStarted("working"))
    assert sink._status is not None
    ans = tui.confirm(_FakeConsole("y"), "run_command", "rm -rf x", "danger")
    assert ans == "yes"
    assert sink._status is None, "confirm must stop the live spinner before reading input"


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
