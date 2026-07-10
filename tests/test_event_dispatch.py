"""Required state sinks and best-effort observers have different failure semantics."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import Event, make_dispatcher  # noqa: E402


def main():
    calls = []

    def required(event):
        calls.append(("required", event))

    def broken_observer(event):
        calls.append(("observer", event))
        raise RuntimeError("presentation failed")

    event = Event()
    make_dispatcher(broken_observer, required=(required,))(event)
    assert [kind for kind, _ in calls] == ["required", "observer"]

    def broken_required(_event):
        raise RuntimeError("state failed")

    try:
        make_dispatcher(lambda _e: calls.append(("should-not-run", _e)),
                        required=(broken_required,))(event)
        raise AssertionError("required sink failure must propagate")
    except RuntimeError as exc:
        assert str(exc) == "state failed"
    assert not any(kind == "should-not-run" for kind, _ in calls)
    print("PASS required sinks propagate and observers stay isolated")


if __name__ == "__main__":
    main()
