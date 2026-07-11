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

    # Host progress observations may report that a task/checkpoint is missing. They must reach UI observers
    # without the authoritative Slice reducer first trying to resolve that same missing active task.
    from sliceagent.events import TurnCommitted
    from sliceagent.memory import NullMemory
    from sliceagent.pfc import slice_sink
    from sliceagent.session import Session
    slice_sink(Session(NullMemory()))(TurnCommitted(False, "error", detail="missing task"))

    seen = []
    committed = TurnCommitted(True, "end_turn", receipt={
        "turn_status": "end_turn", "disposition": "completed",
        "counts": {"requested": 1}, "agents": {},
    })

    def mutate_completion(event):
        event.receipt["counts"]["requested"] = 999
        event.stop_reason = "error"

    def observe_completion(event):
        seen.append((event.stop_reason, event.receipt["counts"]["requested"]))

    make_dispatcher(mutate_completion, observe_completion)(committed)
    assert seen == [("end_turn", 1)], "each observer must receive an independent completion projection"
    assert committed.stop_reason == "end_turn" and committed.receipt["counts"]["requested"] == 1
    print("PASS required sinks propagate and observers stay isolated")


if __name__ == "__main__":
    main()
