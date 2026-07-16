"""The repeated-observation detector is a one-shot model advisory, never an execution gate."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import TurnEnd, TurnInterrupted  # noqa: E402
from sliceagent.hooks import Hooks  # noqa: E402
from sliceagent.loop import _OBSERVATION_REPEAT_NUDGE, run_turn  # noqa: E402
from sliceagent.registry import ToolText  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _call(name: str, call_id: str, **args):
    return NS(name=name, id=call_id, args=args)


def _tool_response(call):
    return NS(content="", tool_calls=[call], finish_reason="tool_calls", usage={})


def _done_response():
    return NS(content="finished from the evidence", tool_calls=[], finish_reason="stop", usage={})


class _ScriptLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def complete(self, messages, _schemas):
        self.seen.append([dict(message) for message in messages])
        return self.responses.pop(0)


class _Host:
    def __init__(self, output="same observation", *, failed=False):
        self.output = output
        self.failed = failed
        self.ran = []

    def schemas(self):
        return []

    def accesses(self, _name, _args):
        return []

    def run(self, name, args):
        self.ran.append((name, dict(args)))
        return ToolText(self.output, ok=not self.failed)


def _run(calls, host):
    responses = [_tool_response(call) for call in calls] + [_done_response()]
    llm = _ScriptLLM(responses)
    events = []
    outcome = run_turn(
        build_slice=lambda: [{"role": "user", "content": "inspect, then finish"}],
        llm=llm, tools=host, dispatch=events.append, hooks=Hooks(), max_steps=len(responses) + 1,
    )
    return outcome, llm, events


def _nudge_count(messages):
    return sum(message.get("content") == _OBSERVATION_REPEAT_NUDGE for message in messages)


@check
def varying_observation_calls_trigger_once_and_work_can_finish():
    names = ("read_file", "list_files", "grep", "glob", "search_history", "code_review")
    calls = [
        _call(names[index % len(names)], f"c{index}", path=f"area-{index}", query=f"q-{index}")
        for index in range(10)
    ]
    outcome, llm, events = _run(calls, _Host())

    assert outcome.stop_reason == "end_turn"
    assert len(llm.seen) == 11 and len(events) > 0
    assert all(_nudge_count(messages) == 0 for messages in llm.seen[:8])
    assert all(_nudge_count(messages) == 1 for messages in llm.seen[8:])
    assert sum(isinstance(event, TurnEnd) for event in events) == 1
    assert not any(isinstance(event, TurnInterrupted) for event in events)


@check
def exact_call_repetition_does_not_trigger():
    # Model commentary is not physical call identity; changing only ``note`` must remain one exact read.
    calls = [
        _call("read_file", f"c{index}", path="same.py", note=f"commentary-{index}")
        for index in range(10)
    ]
    outcome, llm, _events = _run(calls, _Host())
    assert outcome.stop_reason == "end_turn"
    assert _nudge_count(llm.seen[-1]) == 0


@check
def empty_observations_do_not_trigger():
    calls = [_call("read_file", f"c{index}", path=f"empty-{index}.py") for index in range(10)]
    outcome, llm, _events = _run(calls, _Host(" \n\t "))
    assert outcome.stop_reason == "end_turn"
    assert _nudge_count(llm.seen[-1]) == 0


@check
def effectful_and_control_calls_do_not_trigger():
    excluded = ("edit_file", "change_workspace", "ask_user", "spawn_agent", "spawn_explore")
    calls = [
        _call(excluded[index % len(excluded)], f"c{index}", path=f"target-{index}", task=f"task-{index}")
        for index in range(10)
    ]
    outcome, llm, _events = _run(calls, _Host("updated"))
    assert outcome.stop_reason == "end_turn"
    assert _nudge_count(llm.seen[-1]) == 0


@check
def failed_observations_do_not_trigger():
    calls = [_call("grep", f"c{index}", pattern=f"p-{index}") for index in range(10)]
    outcome, llm, _events = _run(calls, _Host("same failure", failed=True))
    assert outcome.stop_reason == "end_turn"
    assert _nudge_count(llm.seen[-1]) == 0


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
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
