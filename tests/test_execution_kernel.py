"""Typed execution-kernel invariants. No network/model dependency."""
from __future__ import annotations

import os
import signal
import shlex
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import (ApiRetry, AssistantText, ToolResult, TurnEnd,
                               TurnInterrupted)  # noqa: E402
from sliceagent.execution import (CHILD_INVOCATION_ID_ARG, CHILD_REQUEST_ORDINAL_ARG,
                                  CHILD_TOKEN_BUDGET_ARG, PreflightOverflow, ToolEffect,
                                  ToolInvocation, ToolOutcome, ToolPurity, ToolStatus, TurnOutcome,
                                  Usage, preflight_model_call, reconciliation_targets)  # noqa: E402
from sliceagent.hooks import BudgetHook, Hooks  # noqa: E402
from sliceagent.loop import (_assistant_message, _delegation_timeout, run_tool_batch,
                             run_turn)  # noqa: E402
from sliceagent.model_runner import complete_model_call  # noqa: E402
from sliceagent.registry import ToolEntry, ToolRegistry, ToolText  # noqa: E402
from sliceagent.scheduler import ScheduledTool, run_ordered  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _tc(name, args, call_id):
    return NS(name=name, args=args, id=call_id)


@check
def tool_bearing_assistant_prose_is_presentation_only():
    response = NS(
        content="I will do two waves and the report is above.",
        reasoning_content="provider replay token",
        tool_calls=[_tc("spawn_agent", {"agent": "explorer", "task": "one"}, "child-1")],
    )
    message = _assistant_message(response)
    assert message["content"] == ""
    assert message["reasoning_content"] == "provider replay token"
    assert message["tool_calls"][0]["id"] == "child-1"


@check
def settled_multi_child_batch_returns_ordered_full_reports_directly():
    class LLM:
        def __init__(self):
            self.calls = 0
            self.synthesis_messages = []

        def complete(self, messages, _schemas):
            self.calls += 1
            if self.calls == 1:
                return NS(
                    content="launching two waves; preliminary findings above",
                    tool_calls=[
                        _tc("spawn_agent", {"agent": "explorer", "task": "one"}, "child-1"),
                        _tc("spawn_agent", {"agent": "explorer", "task": "two"}, "child-2"),
                    ],
                    finish_reason="tool_calls", usage={},
                )
            self.synthesis_messages = list(messages)
            return NS(content="final synthesis", tool_calls=[], finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            from sliceagent.access import ReadAllAccess
            return [ReadAllAccess()]

        def run(self, _name, args):
            index = 1 if args["task"] == "one" else 2
            artifact_id = f"artifact-{index}"
            report = f"BEGIN CHILD REPORT {index}\n" + ("x" * 1200) + f"\nFULL CHILD {index} MIDDLE"
            return ToolText(
                report,
                effects=(ToolEffect(
                    f"{artifact_id}:outcome", "child_outcome", {
                        "artifact_id": artifact_id,
                        "operational_status": "succeeded",
                        "source_coverage_status": "source_complete",
                        "explorer_evidence_status": "content_retained",
                    },
                ),),
            )

    llm = LLM()
    events = []
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "review everything"}],
        llm=llm, tools=Host(), dispatch=events.append, hooks=Hooks(), max_steps=3,
    )
    assert result.stop_reason == "end_turn" and llm.calls == 2
    trajectory = llm.synthesis_messages[1:]
    assert [message["role"] for message in trajectory] == ["assistant", "tool", "tool"]
    assert trajectory[0]["content"] == ""
    assistant_ids = [call["id"] for call in trajectory[0]["tool_calls"]]
    assert [message["tool_call_id"] for message in trajectory[1:]] == assistant_ids
    assert "FULL CHILD 1 MIDDLE" in trajectory[1]["content"]
    assert "FULL CHILD 2 MIDDLE" in trajectory[2]["content"]
    rendered = "\n".join(str(message.get("content") or "") for message in trajectory)
    assert "HOST FAN-IN" not in rendered and "preliminary findings above" not in rendered
    finals = [event for event in events if isinstance(event, AssistantText) and event.final]
    assert len(finals) == 1 and finals[0].content == "final synthesis"


@check
def indeterminate_child_does_not_hide_settled_sibling_report():
    class LLM:
        def __init__(self):
            self.calls = 0
            self.closeout_messages = []

        def complete(self, messages, schemas):
            self.calls += 1
            if self.calls == 1:
                return NS(
                    content="",
                    tool_calls=[
                        _tc("spawn_agent", {"agent": "explorer", "task": "settled"}, "child-ok"),
                        _tc("spawn_agent", {"agent": "explorer", "task": "uncertain"}, "child-unknown"),
                    ],
                    finish_reason="tool_calls", usage={},
                )
            self.closeout_messages = list(messages)
            assert schemas == [], "an indeterminate wave permits synthesis, not another effectful tool"
            return NS(
                content="Child one found the retained issue; child two remains indeterminate.",
                tool_calls=[], finish_reason="stop", usage={},
            )

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            from sliceagent.access import ReadAllAccess
            return [ReadAllAccess()]

        def run(self, _name, args):
            if args["task"] == "settled":
                report = "BEGIN CHILD REPORT\nCONFIRMED SETTLED FINDING\nEND CHILD REPORT"
                return ToolText(report, effects=(ToolEffect(
                    "settled:outcome", "child_outcome", {
                        "status": "succeeded",
                        "operational_status": "succeeded",
                        "kind": "explorer",
                        "launch_ordinal": 1,
                        "report_completion": "complete",
                        "report_bytes": len("CONFIRMED SETTLED FINDING"),
                        "report_sha256": "a" * 64,
                        "report_handle": "artifacts/settled.md",
                    },
                ),))
            return ToolText(
                "Error: child provider state is unresolved",
                status=ToolStatus.INDETERMINATE,
                effects=(ToolEffect(
                    "unknown:outcome", "child_outcome", {
                        "status": "indeterminate",
                        "operational_status": "indeterminate",
                        "kind": "explorer",
                        "launch_ordinal": 2,
                        "report_completion": "absent",
                        "report_bytes": 0,
                    },
                ),),
            )

    llm, events = LLM(), []
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "review both areas"}],
        llm=llm, tools=Host(), dispatch=events.append, hooks=Hooks(), max_steps=3,
    )
    assert result.stop_reason == "indeterminate"
    assert llm.calls == 2, "the only follow-up is one synthesis-only closeout"
    assert any(
        "CONFIRMED SETTLED FINDING" in str(message.get("content") or "")
        for message in llm.closeout_messages
    ), "the settled sibling's full direct report must reach synthesis"
    updates = [event.content for event in events
               if isinstance(event, AssistantText) and not event.final]
    assert "Child one found the retained issue; child two remains indeterminate." in updates
    assert not any(isinstance(event, TurnEnd) for event in events)
    interrupts = [event for event in events if isinstance(event, TurnInterrupted)]
    assert len(interrupts) == 1 and "artifacts/settled.md" in (interrupts[0].message or "")


@check
def lifecycle_child_wave_caps_parallel_full_model_loops_at_four():
    lock = threading.Lock()
    release = threading.Event()
    four_running = threading.Event()
    state = {"active": 0, "maximum": 0}

    def task(index):
        invocation = ToolInvocation(f"child-{index}", "spawn_agent", {}, index)

        def run():
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
                if state["active"] >= 4:
                    four_running.set()
            assert release.wait(2)
            with lock:
                state["active"] -= 1
            return ToolOutcome(invocation, ToolStatus.SUCCEEDED, "done")

        return ScheduledTool(
            invocation, ToolPurity.PURE_READ, run, timeout_safe=False,
        )

    box = {}
    runner = threading.Thread(
        target=lambda: box.setdefault("outcomes", run_ordered([task(i) for i in range(7)])),
        daemon=True,
    )
    runner.start()
    assert four_running.wait(1)
    time.sleep(0.05)  # give an incorrectly uncapped wave ample time to launch children 5-7
    assert state["maximum"] == 4, state
    release.set()
    runner.join(2)
    assert not runner.is_alive()
    assert len(box["outcomes"]) == 7


@check
def indeterminate_lifecycle_child_cancels_only_the_unadmitted_wave_tail():
    lock = threading.Lock()
    four_running = threading.Event()
    release_started = threading.Event()
    started = []

    def task(index):
        invocation = ToolInvocation(f"uncertain-child-{index}", "spawn_agent", {}, index)

        def run():
            with lock:
                started.append(index)
                if len(started) == 4:
                    four_running.set()
            assert four_running.wait(2)
            if index == 0:
                return ToolOutcome(
                    invocation, ToolStatus.INDETERMINATE,
                    "provider watchdog expired; request may still be in flight",
                )
            assert release_started.wait(2)
            return ToolOutcome(invocation, ToolStatus.SUCCEEDED, "settled")

        return ScheduledTool(invocation, ToolPurity.PURE_READ, run, timeout_safe=False)

    box = {}
    runner = threading.Thread(
        target=lambda: box.setdefault("outcomes", run_ordered([task(i) for i in range(6)])),
        daemon=True,
    )
    runner.start()
    try:
        assert four_running.wait(1)
        time.sleep(0.12)  # allow the scheduler to observe child 0 and close the queued tail
        with lock:
            assert sorted(started) == [0, 1, 2, 3], started
        release_started.set()
        runner.join(2)
        assert not runner.is_alive()
        outcomes = box["outcomes"]
        assert [outcome.status for outcome in outcomes] == [
            ToolStatus.INDETERMINATE,
            ToolStatus.SUCCEEDED,
            ToolStatus.SUCCEEDED,
            ToolStatus.SUCCEEDED,
            ToolStatus.CANCELLED,
            ToolStatus.CANCELLED,
        ]
        expected = (
            "Not run: an earlier invocation in this wave has an unresolved outcome; "
            "queued execution was not admitted"
        )
        assert [outcome.text for outcome in outcomes[4:]] == [expected, expected]
    finally:
        release_started.set()
        runner.join(1)


@check
def failed_lifecycle_child_still_admits_queued_wave_siblings():
    started = []

    def task(index):
        invocation = ToolInvocation(f"failed-child-{index}", "spawn_agent", {}, index)

        def run():
            started.append(index)
            status = ToolStatus.FAILED if index == 0 else ToolStatus.SUCCEEDED
            return ToolOutcome(invocation, status, "settled")

        return ScheduledTool(invocation, ToolPurity.PURE_READ, run, timeout_safe=False)

    outcomes = run_ordered([task(i) for i in range(3)], max_workers=1)
    assert started == [0, 1, 2]
    assert [outcome.status for outcome in outcomes] == [
        ToolStatus.FAILED, ToolStatus.SUCCEEDED, ToolStatus.SUCCEEDED,
    ]


@check
def delegation_timeout_cannot_be_disabled_with_nonfinite_values():
    old = os.environ.get("AGENT_DELEGATION_TIMEOUT")
    try:
        for raw in ("inf", "-inf", "nan", "1e309", "0", "invalid"):
            os.environ["AGENT_DELEGATION_TIMEOUT"] = raw
            assert _delegation_timeout() == 900.0, raw
        os.environ["AGENT_DELEGATION_TIMEOUT"] = "12.5"
        assert _delegation_timeout() == 12.5
    finally:
        if old is None:
            os.environ.pop("AGENT_DELEGATION_TIMEOUT", None)
        else:
            os.environ["AGENT_DELEGATION_TIMEOUT"] = old


@check
def registered_tool_status_does_not_come_from_text():
    registry = ToolRegistry()
    registry.register(ToolEntry(
        "read_file",
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        lambda _args: "Error: this is legitimate file content",
    ))
    outcome = registry.invoke(ToolInvocation("c1", "read_file", {}, 0))
    assert outcome.status is ToolStatus.SUCCEEDED
    assert not outcome.failing
    assert outcome.effects[0].kind == "tool_outcome"
    assert outcome.effects[0].payload["status"] == "succeeded"


@check
def invalid_explicit_status_never_fabricates_success():
    value = ToolText("provider supplied an unknown status", status="definitely-not-a-status")
    assert value.status is ToolStatus.INDETERMINATE
    registry = ToolRegistry()
    registry.register(ToolEntry(
        "opaque", {"type": "function", "function": {"name": "opaque", "parameters": {}}},
        lambda _args: value,
    ))
    outcome = registry.invoke(ToolInvocation("bad-status", "opaque", {}, 0))
    assert outcome.status is ToolStatus.INDETERMINATE


@check
def registry_enforces_live_availability_and_canonical_result_coercion():
    ran = []
    registry = ToolRegistry()
    schema = lambda name: {"type": "function", "function": {"name": name, "parameters": {}}}
    registry.register(ToolEntry(
        "offline", schema("offline"), lambda _args: ran.append("offline") or "bad",
        check=lambda: False,
    ))
    assert "offline" not in registry.names()
    unavailable = registry.invoke(ToolInvocation("offline", "offline", {}, 0))
    assert unavailable.status is ToolStatus.FAILED and ran == []

    registry.register(ToolEntry(
        "bytes", schema("bytes"), lambda _args: b"\xffpayload", purity=ToolPurity.PURE_READ,
    ))
    decoded = registry.invoke(ToolInvocation("bytes", "bytes", {}, 1))
    assert decoded.status is ToolStatus.SUCCEEDED and decoded.text == "\ufffdpayload"

    class BrokenText:
        def __str__(self):
            raise RuntimeError("cannot render result")

    registry.register(ToolEntry(
        "broken", schema("broken"), lambda _args: BrokenText(),
        source="plugin:test", purity=ToolPurity.UNKNOWN,
    ))
    broken = registry.invoke(ToolInvocation("broken", "broken", {}, 2))
    assert broken.status is ToolStatus.INDETERMINATE
    assert "cannot render result" in broken.text


@check
def extension_system_exit_is_contained_but_keyboard_interrupt_still_escapes():
    schema = lambda name: {"type": "function", "function": {"name": name, "parameters": {}}}
    registry = ToolRegistry()
    registry.register(ToolEntry(
        "exit_check", schema("exit_check"), lambda _args: "unexpected",
        check=lambda: (_ for _ in ()).throw(SystemExit(9)), source="plugin:test",
    ))
    registry.register(ToolEntry(
        "exit_access", schema("exit_access"), lambda _args: "unexpected",
        accesses=lambda _args: (_ for _ in ()).throw(SystemExit(8)), source="plugin:test",
    ))
    registry.register(ToolEntry(
        "exit_handler", schema("exit_handler"),
        lambda _args: (_ for _ in ()).throw(SystemExit(7)), source="plugin:test",
    ))
    assert "exit_check" not in registry.names()
    from sliceagent.access import AllAccess
    assert isinstance(registry.accesses("exit_access", {})[0], AllAccess)
    assert registry.run("exit_handler", {}).status is ToolStatus.INDETERMINATE

    registry.register(ToolEntry(
        "exit_effect", schema("exit_effect"), lambda _args: "ran",
        source="plugin:test", effect_factory=lambda *_args: (_ for _ in ()).throw(SystemExit(6)),
    ))
    effect_outcome = registry.invoke(ToolInvocation("exit-effect", "exit_effect", {}, 0))
    assert effect_outcome.status is ToolStatus.INDETERMINATE
    assert "effect construction failed" in effect_outcome.text

    registry.register(ToolEntry(
        "interrupt", schema("interrupt"),
        lambda _args: (_ for _ in ()).throw(KeyboardInterrupt()), source="plugin:test",
    ))
    try:
        registry.run("interrupt", {})
        assert False, "KeyboardInterrupt remains user-owned and must escape"
    except KeyboardInterrupt:
        pass


@check
def malformed_tool_schema_is_rejected_atomically():
    registry = ToolRegistry()
    generation = registry.generation
    malformed = (
        {"type": "function", "function": {"name": "bad", "parameters": "not-a-schema"}},
        {"type": "function", "function": {"name": "bad", "parameters": ""}},
        {"type": "function", "function": {"name": "bad", "parameters": {"required": ""}}},
        {"type": "function", "function": {"name": "bad", "parameters": {"required": 0}}},
    )
    for schema in malformed:
        try:
            registry.register(ToolEntry("bad", schema, lambda _args: "bad"))
            assert False, "a malformed schema must not enter the shared registry"
        except ValueError:
            pass
    assert not registry.has("bad") and registry.generation == generation


@check
def registry_invoke_separates_handler_args_from_raw_effect_provenance():
    handled, constructed = [], []

    def handler(args):
        handled.append(dict(args))
        return "ok"

    def effects(invocation, status, text):
        constructed.append((dict(invocation.args), status, text))
        return (ToolEffect("custom-effect", "custom", {"path": invocation.args["path"]}),)

    registry = ToolRegistry()
    registry.register(ToolEntry(
        "edit_file", {"type": "function", "function": {
            "name": "edit_file", "parameters": {"required": ["path"]},
        }}, handler, effect_factory=effects,
    ))
    invocation = ToolInvocation(
        "edit-1", "edit_file", {"path": "a.py", "note": "raw provenance"}, 0,
    )
    outcome = registry.invoke(
        invocation, call_args={"path": "a.py"}, default_effect_id="unused-default",
    )
    assert handled == [{"path": "a.py"}]
    assert constructed == [(
        {"path": "a.py", "note": "raw provenance"}, ToolStatus.SUCCEEDED, "ok",
    )]
    assert outcome.effects == (ToolEffect("custom-effect", "custom", {"path": "a.py"}),)

    handled.clear()
    constructed.clear()

    class Host:
        def __init__(self):
            self.registry = registry

        def accesses(self, name, args):
            return registry.accesses(name, args)

        def run(self, name, args):
            return registry.run(name, args)

        def preflight_run(self, name, args):
            return registry.admit(name, args)

        def run_preflighted(self, name, args, admission):
            return registry.run_admitted(admission, args)

    _, rows = run_tool_batch(
        [_tc("edit_file", {"path": "a.py", "note": "raw provenance"}, "edit-2")],
        Host(), lambda _event: None, Hooks(), step=2, turn_id="turn-A",
    )
    assert handled == [{"path": "a.py"}], "handler must receive note-stripped call args"
    assert constructed[0][0] == {"path": "a.py", "note": "raw provenance"}
    assert rows[0]["outcome"].effects[0].id == "custom-effect"


@check
def registry_and_production_share_effect_factory_failure_semantics():
    ran = []

    def broken_effects(_invocation, _status, _text):
        raise RuntimeError("cannot construct effects")

    registry = ToolRegistry()
    def schema(name):
        return {"type": "function", "function": {"name": name, "parameters": {}}}

    registry.register(ToolEntry(
        "first_write", schema("first_write"),
        lambda _args: ran.append("first_write") or "wrote", effect_factory=broken_effects,
    ))
    registry.register(ToolEntry(
        "second_write", schema("second_write"),
        lambda _args: ran.append("second_write") or "wrote",
    ))

    direct = registry.invoke(
        ToolInvocation("direct", "first_write", {}, 0), default_effect_id="direct-default",
    )
    assert direct.status is ToolStatus.INDETERMINATE
    assert direct.effects[0].id == "direct-default"
    assert direct.effects[0].payload["status"] == "indeterminate"

    ran.clear()

    class Host:
        def __init__(self):
            self.registry = registry

        def accesses(self, name, args):
            return registry.accesses(name, args)

        def run(self, name, args):
            return registry.run(name, args)

    _, results = run_tool_batch(
        [_tc("first_write", {}, "one"), _tc("second_write", {}, "two")],
        Host(), lambda _event: None, Hooks(), step=3, turn_id="turn-A",
    )
    assert [row["status"] for row in results] == ["indeterminate", "cancelled"]
    assert ran == ["first_write"]
    effect = results[0]["outcome"].effects[0]
    assert effect.id == "turn-A:3:0:one:0"
    assert effect.payload == {"name": "first_write", "status": "indeterminate"}


@check
def preflight_stop_uses_canonical_effect_without_running_handler():
    from sliceagent.hooks import ToolPreflight

    ran = []
    registry = ToolRegistry()
    registry.register(ToolEntry(
        "edit_file", {"type": "function", "function": {
            "name": "edit_file", "parameters": {},
        }}, lambda _args: ran.append("handler") or "edited",
    ))

    class Host:
        def __init__(self):
            self.registry = registry

        def accesses(self, name, args):
            return registry.accesses(name, args)

        def run(self, name, args):
            return registry.run(name, args)

    class LifecycleStop(Hooks):
        def preflight_tool(self, _name, _args):
            return ToolPreflight(True, "cancelled for test", kind="lifecycle")

    _, rows = run_tool_batch(
        [_tc("edit_file", {"path": "a.py", "note": "keep raw"}, "blocked")],
        Host(), lambda _event: None, LifecycleStop(), step=4, turn_id="turn-P",
    )
    assert ran == []
    outcome = rows[0]["outcome"]
    assert outcome.status is ToolStatus.CANCELLED
    assert rows[0]["rejection_kind"] == "lifecycle"
    assert rows[0]["output"] == "Not run: cancelled for test"
    assert "policy" not in rows[0]["output"].casefold()
    assert outcome.effects[0] == ToolEffect(
        "turn-P:4:0:blocked:0", "tool_outcome", {"name": "edit_file", "status": "cancelled"},
    )


@check
def preflight_stop_never_invokes_execution_effect_factory():
    from sliceagent.hooks import ToolPreflight

    ran = []
    registry = ToolRegistry()

    def effects_that_must_not_run(_invocation, _status, _text):
        raise RuntimeError("execution-only effect factory was called")

    def schema(name):
        return {"type": "function", "function": {"name": name, "parameters": {}}}

    registry.register(ToolEntry(
        "blocked", schema("blocked"), lambda _args: ran.append("blocked") or "bad",
        purity=ToolPurity.EFFECTFUL, effect_factory=effects_that_must_not_run,
    ))
    registry.register(ToolEntry(
        "later", schema("later"), lambda _args: ran.append("later") or "ok",
        purity=ToolPurity.EFFECTFUL,
    ))

    class Host:
        def __init__(self):
            self.registry = registry

        def accesses(self, name, args):
            return registry.accesses(name, args)

        def run(self, name, args):
            return registry.run(name, args)

    class StopFirst(Hooks):
        def preflight_tool(self, name, _args):
            return ToolPreflight(name == "blocked", "cancelled before execution", kind="lifecycle")

    _, rows = run_tool_batch([
        _tc("blocked", {}, "blocked"), _tc("later", {}, "later"),
    ], Host(), lambda _event: None, StopFirst())
    assert [row["status"] for row in rows] == ["cancelled", "succeeded"]
    assert ran == ["later"]


@check
def registry_validation_failures_never_claim_physical_handler_start():
    from sliceagent.events import ToolExecutionStarted, ToolStarted

    ran, events = [], []
    registry = ToolRegistry()
    schema = lambda name, required=(): {"type": "function", "function": {
        "name": name,
        "parameters": {"type": "object", "required": list(required)},
    }}
    registry.register(ToolEntry(
        "offline", schema("offline"), lambda _args: ran.append("offline") or "bad",
        check=lambda: False,
    ))
    registry.register(ToolEntry(
        "needs_path", schema("needs_path", ("path",)),
        lambda _args: ran.append("needs_path") or "bad",
    ))

    class Host:
        def __init__(self):
            self.registry = registry

        def accesses(self, name, args):
            return registry.accesses(name, args)

        def run(self, name, args):
            return registry.run(name, args)

        def preflight_run(self, name, args):
            return registry.admit(name, args)

        def run_preflighted(self, name, args, admission):
            return registry.run_admitted(admission, args)

    _, rows = run_tool_batch([
        _tc("offline", {}, "offline"),
        _tc("needs_path", {}, "missing"),
        _tc("not_registered", {}, "unknown"),
    ], Host(), events.append, Hooks())
    assert [row["status"] for row in rows] == ["failed", "failed", "failed"]
    assert ran == []
    assert not any(isinstance(event, (ToolExecutionStarted, ToolStarted)) for event in events)


@check
def scheduled_registry_admission_is_one_shot_across_the_start_boundary():
    from sliceagent.events import ToolStarted

    checks, ran, events = [], [], []
    registry = ToolRegistry()

    def volatile_check():
        checks.append(len(checks) + 1)
        return len(checks) == 1

    registry.register(ToolEntry(
        "volatile", {"type": "function", "function": {
            "name": "volatile", "parameters": {},
        }}, lambda _args: ran.append("handler") or "ok", check=volatile_check,
    ))

    class Host:
        def accesses(self, name, args):
            return registry.accesses(name, args)

        def preflight_run(self, name, args):
            return registry.admit(name, args)

        def run_preflighted(self, name, args, admission):
            return registry.run_admitted(admission, args)

        def run(self, name, args):
            return registry.run(name, args)

    _, rows = run_tool_batch(
        [_tc("volatile", {}, "volatile")], Host(), events.append, Hooks(),
    )
    assert checks == [1], "availability must not be rechecked after ToolStarted"
    assert ran == ["handler"] and rows[0]["status"] == "succeeded"
    assert any(isinstance(event, ToolStarted) for event in events)


@check
def direct_registry_preflight_failure_never_invokes_execution_effect_factory():
    effects, ran = [], []
    registry = ToolRegistry()

    def factory(*_args):
        effects.append("factory")
        return ()

    registry.register(ToolEntry(
        "offline", {"type": "function", "function": {
            "name": "offline", "parameters": {},
        }}, lambda _args: ran.append("handler") or "bad", check=lambda: False,
        effect_factory=factory,
    ))
    outcome = registry.invoke(ToolInvocation("offline", "offline", {}, 0))
    assert outcome.status is ToolStatus.FAILED
    assert ran == [] and effects == []
    assert outcome.effects[0].kind == "tool_outcome"


@check
def incomplete_host_preflight_protocol_never_crosses_the_start_boundary():
    from sliceagent.events import ToolExecutionStarted, ToolStarted

    preflighted, ran, events = [], [], []

    class Host:
        def accesses(self, _name, _args):
            return []

        def preflight_run(self, name, _args):
            preflighted.append(name)
            return object(), None

        def run(self, name, _args):
            ran.append(name)
            return "bad"

    _, rows = run_tool_batch(
        [_tc("partial", {}, "partial")], Host(), events.append, Hooks(),
    )
    assert preflighted == [], "an unpaired preflight method must not claim one-shot admission"
    assert ran == [] and rows[0]["status"] == "failed"
    assert "incomplete one-shot preflight protocol" in rows[0]["output"]
    assert not any(isinstance(event, (ToolExecutionStarted, ToolStarted)) for event in events)


@check
def same_purity_registry_replacement_cannot_inherit_stale_dedup_or_effect_metadata():
    from sliceagent.events import ToolStarted

    old_effects, new_effects, ran = [], [], []
    events = []
    registry = ToolRegistry()

    def schema(name):
        return {"type": "function", "function": {"name": name, "parameters": {}}}

    def old_factory(*_args):
        old_effects.append("old")
        return (ToolEffect("old", "old", {}),)

    def new_factory(*_args):
        new_effects.append("new")
        return (ToolEffect("new", "new", {}),)

    registry.register(ToolEntry(
        "target", schema("target"), lambda _args: ran.append("old-handler") or "old",
        purity=ToolPurity.PURE_READ, deduplicable=True, effect_factory=old_factory,
    ))

    def replace(_args):
        registry.register(ToolEntry(
            "target", schema("target"), lambda _args: ran.append("new-handler") or "new",
            purity=ToolPurity.PURE_READ, deduplicable=False, effect_factory=new_factory,
        ), override=True)
        return "replaced"

    registry.register(ToolEntry(
        "replace", schema("replace"), replace, purity=ToolPurity.EFFECTFUL,
    ))

    class Host:
        def __init__(self):
            self.registry = registry

        def accesses(self, name, args):
            return registry.accesses(name, args)

        def preflight_run(self, name, args):
            return registry.admit(name, args)

        def run_preflighted(self, name, args, admission):
            return registry.run_admitted(admission, args)

        def run(self, name, args):
            return registry.run(name, args)

    _, rows = run_tool_batch([
        _tc("replace", {}, "replace"),
        _tc("target", {}, "target-1"), _tc("target", {}, "target-2"),
    ], Host(), events.append, Hooks())
    assert ran == [] and old_effects == [] and new_effects == []
    assert [row["status"] for row in rows] == ["succeeded", "failed", "failed"]
    assert "registration changed before execution" in rows[1]["output"]
    started = [event.invocation.id for event in events if isinstance(event, ToolStarted)]
    assert started == ["replace"], "neither the stale source nor its collapsed duplicate may start"


@check
def registry_replacement_cannot_run_under_stale_scheduler_purity():
    from sliceagent.access import AllAccess, ReadAllAccess
    from sliceagent.events import ToolStarted

    ran, events = [], []
    registry = ToolRegistry()

    def schema(name):
        return {"type": "function", "function": {"name": name, "parameters": {}}}

    registry.register(ToolEntry(
        "target", schema("target"), lambda _args: ran.append("old-read") or "old",
        accesses=lambda _args: [ReadAllAccess()], purity=ToolPurity.PURE_READ,
    ))

    def replace(_args):
        registry.register(ToolEntry(
            "target", schema("target"), lambda _args: ran.append("new-write") or "new",
            accesses=lambda _args: [AllAccess()], purity=ToolPurity.EFFECTFUL,
        ), override=True)
        return "replaced"

    registry.register(ToolEntry(
        "replace", schema("replace"), replace,
        accesses=lambda _args: [AllAccess()], purity=ToolPurity.EFFECTFUL,
    ))

    class Host:
        def __init__(self):
            self.registry = registry

        def accesses(self, name, args):
            return registry.accesses(name, args)

        def preflight_run(self, name, args):
            return registry.admit(name, args)

        def run_preflighted(self, name, args, admission):
            return registry.run_admitted(admission, args)

        def run(self, name, args):
            return registry.run(name, args)

    _, rows = run_tool_batch([
        _tc("replace", {}, "replace"), _tc("target", {}, "target"),
    ], Host(), events.append, Hooks())
    assert ran == [], "the replacement must not run with the old read-wave timeout/concurrency contract"
    assert rows[1]["status"] == "failed"
    assert "registration changed before execution" in rows[1]["output"]
    started = [event.invocation.id for event in events if isinstance(event, ToolStarted)]
    assert started == ["replace"]


@check
def subagent_wrapper_preserves_incomplete_inner_preflight_rejection():
    from sliceagent.events import ToolExecutionStarted, ToolStarted
    from sliceagent.subagent import SubagentHost

    preflighted, ran, events = [], [], []

    class Inner:
        def accesses(self, _name, _args):
            return []

        def preflight_run(self, name, _args):
            preflighted.append(name)
            return object(), None

        def run(self, name, _args):
            ran.append(name)
            return "bad"

    host = SubagentHost(Inner(), llm=None, retriever=None, memory=None)
    _, rows = run_tool_batch(
        [_tc("partial", {}, "partial")], host, events.append, Hooks(),
    )
    assert preflighted == [] and ran == []
    assert rows[0]["status"] == "failed"
    assert "incomplete one-shot preflight protocol" in rows[0]["output"]
    assert not any(isinstance(event, (ToolExecutionStarted, ToolStarted)) for event in events)


@check
def dynamic_host_exception_also_flows_through_canonical_default_effect():
    class Host:
        def accesses(self, _name, _args):
            return []  # no explicit pure-read contract => UNKNOWN and potentially effectful

        def run(self, _name, _args):
            raise RuntimeError("boundary failed")

    _, rows = run_tool_batch(
        [_tc("opaque_extension", {}, "opaque")], Host(), lambda _event: None, Hooks(),
        step=5, turn_id="turn-D",
    )
    outcome = rows[0]["outcome"]
    assert outcome.status is ToolStatus.INDETERMINATE
    assert outcome.effects[0] == ToolEffect(
        "turn-D:5:0:opaque:0", "tool_outcome",
        {"name": "opaque_extension", "status": "indeterminate"},
    )


@check
def provider_order_prevents_read_from_overtaking_write():
    state = {"value": "old"}

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            if name == "read_file":
                time.sleep(0.03)
                return state["value"]
            state["value"] = "new"
            return "written"

    calls = [
        _tc("read_file", {"path": "x"}, "r1"),
        _tc("edit_file", {"path": "x", "content": "new"}, "w"),
        _tc("read_file", {"path": "x"}, "r2"),
    ]
    _, results = run_tool_batch(calls, Host(), lambda _event: None, Hooks())
    assert [r["output"] for r in results] == ["old", "written", "new"]


@check
def consecutive_pure_reads_overlap():
    rendezvous = threading.Barrier(2)

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, _name, args):
            rendezvous.wait(timeout=1)
            return args["path"]

    calls = [_tc("read_file", {"path": "a"}, "a"), _tc("read_file", {"path": "b"}, "b")]
    _, results = run_tool_batch(calls, Host(), lambda _event: None, Hooks())
    assert [r["status"] for r in results] == ["succeeded", "succeeded"]


@check
def unkillable_effectful_timeout_waits_before_later_barrier():
    prior = os.environ.get("AGENT_TOOL_TIMEOUT")
    os.environ["AGENT_TOOL_TIMEOUT"] = "0.03"
    ran = []

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(name)
            if name == "unknown_mutator":
                time.sleep(0.12)
            return "ok"

    calls = [_tc("unknown_mutator", {}, "slow"), _tc("edit_file", {"path": "x"}, "later")]
    try:
        _, results = run_tool_batch(calls, Host(), lambda _event: None, Hooks())
        assert [r["status"] for r in results] == ["succeeded", "succeeded"]
        assert ran == ["unknown_mutator", "edit_file"]
    finally:
        if prior is None:
            os.environ.pop("AGENT_TOOL_TIMEOUT", None)
        else:
            os.environ["AGENT_TOOL_TIMEOUT"] = prior


@check
def local_command_timeout_is_indeterminate_and_reaps_background_tree():
    from sliceagent.tools import LocalToolHost

    with tempfile.TemporaryDirectory(prefix="command-timeout-") as root:
        target = os.path.join(root, "late.txt")
        host = LocalToolHost(root=root, timeout=1)
        command = f"(sleep 1.4; echo late > {shlex.quote(target)}) & sleep 10"
        outcome = host.run("run_command", {"command": command, "timeout": 1})
        assert outcome.status is ToolStatus.INDETERMINATE
        time.sleep(0.6)
        # POSIX killpg reaps the whole group synchronously; on Windows taskkill /T is best-effort and a
        # detached `&` subshell can escape the PID-tree walk. INDETERMINATE + the reconciliation gate is the
        # documented Windows contract (CORE-DESIGN §11.2), so only assert the strong reap off-Windows.
        if os.name != "nt":
            assert not os.path.exists(target), "a timed-out command descendant mutated after return"


@check
def execute_code_inner_timeout_is_indeterminate_and_reaps_background_tree():
    from sliceagent.tools import LocalToolHost

    with tempfile.TemporaryDirectory(prefix="execute-inner-timeout-") as root:
        target = os.path.join(root, "late.txt")
        host = LocalToolHost(root=root, timeout=8)
        command = f"(sleep 1.4; echo late > {shlex.quote(target)}) & sleep 10"
        outcome = host.run("execute_code", {"code": f"run({command!r}, timeout=1)"})
        assert outcome.status is ToolStatus.INDETERMINATE, outcome
        time.sleep(0.6)
        if os.name != "nt":   # Windows taskkill /T is best-effort on a detached `&` subshell — see above
            assert not os.path.exists(target), "the nested run() descendant mutated after timeout return"


@check
def read_only_child_is_parallelizable_but_not_abandoned_by_generic_timeout():
    from sliceagent.access import ReadAllAccess

    prior = os.environ.get("AGENT_TOOL_TIMEOUT")
    os.environ["AGENT_TOOL_TIMEOUT"] = "0.03"

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, _args):
            time.sleep(0.1)
            return "sealed child"

    started = time.monotonic()
    try:
        _, results = run_tool_batch(
            [_tc("spawn_agent", {"agent": "explorer", "task": "inspect"}, "child")],
            Host(), lambda _event: None, Hooks(),
        )
    finally:
        if prior is None:
            os.environ.pop("AGENT_TOOL_TIMEOUT", None)
        else:
            os.environ["AGENT_TOOL_TIMEOUT"] = prior
    assert time.monotonic() - started >= 0.09
    assert results[0]["status"] == "succeeded"


@check
def pure_read_timeout_returns_failure_feedback_without_reconciliation():
    prior = os.environ.get("AGENT_TOOL_TIMEOUT")
    os.environ["AGENT_TOOL_TIMEOUT"] = "0.03"

    class LLM:
        def __init__(self):
            self.calls = 0

        def complete(self, _messages, _schemas):
            self.calls += 1
            if self.calls > 1:
                return NS(content="The read timed out.", tool_calls=[], finish_reason="stop", usage={})
            return NS(content="", tool_calls=[_tc("read_file", {"path": "slow"}, "slow")],
                      finish_reason="tool_calls", usage={})

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            return []

        def run(self, _name, _args):
            time.sleep(0.06)
            return "late"

    llm, events = LLM(), []
    try:
        result = run_turn(
            build_slice=lambda: [{"role": "user", "content": "go"}],
            llm=llm, tools=Host(), dispatch=events.append, hooks=Hooks(), max_steps=4,
        )
        assert result.stop_reason == "end_turn"
        assert llm.calls == 2
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert tool_results and tool_results[0].status == "failed"
        assert not any(isinstance(e, TurnInterrupted) for e in events)
        assert any(isinstance(e, TurnEnd) for e in events)
    finally:
        if prior is None:
            os.environ.pop("AGENT_TOOL_TIMEOUT", None)
        else:
            os.environ["AGENT_TOOL_TIMEOUT"] = prior


@check
def keyboard_interrupt_after_tool_start_records_exact_reconciliation_target():
    from sliceagent.pfc import Slice, slice_sink

    state = Slice(); state.reset("edit")

    class LLM:
        def complete(self, _messages, _schemas):
            return NS(
                content="", tool_calls=[_tc("edit_file", {"path": "critical.py"}, "edit-1")],
                finish_reason="tool_calls", usage={},
            )

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            return []

        def run(self, _name, _args):
            raise KeyboardInterrupt

    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "go"}],
        llm=LLM(), tools=Host(), dispatch=slice_sink(state), hooks=Hooks(), max_steps=2,
    )
    assert result.stop_reason == "indeterminate"
    assert state.reconciliation_targets == ["path:critical.py"]
    assert "edit-1" in state.reconciliation_required


@check
def execution_uncertainty_is_advisory_on_the_next_turn():
    from sliceagent.pfc import Slice, slice_sink

    state = Slice(); state.reset("repair")
    state.reconciliation_required = "late command may still write"
    state.reconciliation_targets = ["path:a.py"]
    ran = []

    class LLM:
        def __init__(self):
            self.calls = 0

        def complete(self, _messages, _schemas):
            self.calls += 1
            usage = {"prompt_tokens": 1, "completion_tokens": 1}
            if self.calls == 1:
                return NS(content="", tool_calls=[_tc("edit_file", {"path": "a.py"}, "ordinary-edit")],
                          finish_reason="tool_calls", usage=usage)
            return NS(content="done", tool_calls=[], finish_reason="stop", usage=usage)

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(name)
            return ToolText("ok")

    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "continue safely"}],
        llm=LLM(), tools=Host(), dispatch=slice_sink(state),
        hooks=Hooks(), max_steps=4,
    )
    assert result.stop_reason == "end_turn"
    assert ran == ["edit_file"], "historical uncertainty must not become an execution blocker"
    assert state.reconciliation_required == "late command may still write", \
        "the receipt remains truthful until explicitly resolved"


@check
def incidental_arguments_cannot_narrow_an_opaque_operation():
    args = {
        "command": "curl -X POST example.test/deploy",
        "path": "README.md", "handle": "p1", "session": "main",
    }
    assert reconciliation_targets("run_command", args) == (
        "workspace:*", "opaque:run_command",
    )


@check
def unknown_mcp_uncertainty_retains_local_opaque_and_external_boundaries():
    assert reconciliation_targets("mcp__deploy__publish", {"path": "README.md"}) == (
        "workspace:*", "opaque:mcp__deploy__publish", "external:mcp__deploy__publish",
    )


@check
def local_ctrl_c_reaps_the_started_command_group():
    from sliceagent.sandbox import LocalSandbox

    if os.name == "nt":
        return
    with tempfile.TemporaryDirectory(prefix="command-interrupt-") as root:
        target = os.path.join(root, "late.txt")
        command = f"(sleep 0.8; echo late > {shlex.quote(target)}) & sleep 10"
        timer = threading.Timer(0.15, lambda: os.kill(os.getpid(), signal.SIGINT))
        timer.start()
        try:
            try:
                LocalSandbox().run(command, cwd=root, timeout=20)
                assert False, "the injected SIGINT must reach the blocking command"
            except KeyboardInterrupt:
                pass
        finally:
            timer.cancel()
        time.sleep(1.0)
        assert not os.path.exists(target), "an interrupted command descendant mutated after the turn returned"


@check
def missing_user_answer_is_a_typed_cancellation():
    from sliceagent.tools import LocalToolHost
    host = LocalToolHost(tempfile.mkdtemp(prefix="ask-user-cancel-"))
    host.on_ask_user = lambda _question, _options: "(no answer)"
    output = host.run("ask_user", {"question": "Did it settle?"})
    assert isinstance(output, ToolText) and output.status is ToolStatus.CANCELLED


@check
def read_settling_during_grace_preserves_later_barrier():
    prior = os.environ.get("AGENT_TOOL_TIMEOUT")
    os.environ["AGENT_TOOL_TIMEOUT"] = "0.03"
    ran, events = [], []

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(f"{name}:start")
            if name == "read_file":
                time.sleep(0.06)
                ran.append("read_file:end")
            return "ok"

    try:
        _, results = run_tool_batch([
            _tc("read_file", {"path": "slow"}, "slow"),
            _tc("edit_file", {"path": "later", "content": "x"}, "later"),
        ], Host(), events.append, Hooks())
        assert [result["status"] for result in results] == ["failed", "succeeded"]
        assert ran == ["read_file:start", "read_file:end", "edit_file:start"], \
            "the timeout is a normal failure, but the read must remain an ordering barrier until it exits"
        from sliceagent.events import ToolResult
        logical = [event for event in events if isinstance(event, ToolResult)]
        assert [event.invocation_id for event in logical] == ["slow", "later"]
    finally:
        if prior is None:
            os.environ.pop("AGENT_TOOL_TIMEOUT", None)
        else:
            os.environ["AGENT_TOOL_TIMEOUT"] = prior


@check
def hung_read_returns_indeterminate_cancels_tail_and_releases_fixture():
    release = threading.Event()
    finished = threading.Event()
    ran_effect = []
    read_inv = ToolInvocation("hung", "read_file", {"path": "fifo"}, 0)
    edit_inv = ToolInvocation("edit", "edit_file", {"path": "later"}, 1)

    def hung_read():
        try:
            release.wait()
            return ToolOutcome(read_inv, ToolStatus.SUCCEEDED, "late")
        finally:
            finished.set()

    def edit():
        ran_effect.append(True)
        return ToolOutcome(edit_inv, ToolStatus.SUCCEEDED, "edited")

    started = time.monotonic()
    try:
        outcomes = run_ordered([
            ScheduledTool(read_inv, ToolPurity.PURE_READ, hung_read),
            ScheduledTool(edit_inv, ToolPurity.EFFECTFUL, edit),
        ], timeout=0.03)
        elapsed = time.monotonic() - started
        assert [outcome.status for outcome in outcomes] == [
            ToolStatus.INDETERMINATE, ToolStatus.CANCELLED,
        ]
        assert not ran_effect, "a later mutation must not overtake a still-running reader"
        assert elapsed < 0.35, "the scheduler must return after deadline + bounded grace"
    finally:
        release.set()
        assert finished.wait(1), "the daemon read fixture must settle after release"


@check
def hung_read_polls_turn_cancellation_during_long_deadline():
    release = threading.Event()
    finished = threading.Event()
    cancel = threading.Event()
    inv = ToolInvocation("cancel-read", "read_file", {"path": "fifo"}, 0)

    def hung_read():
        try:
            release.wait()
            return ToolOutcome(inv, ToolStatus.SUCCEEDED, "late")
        finally:
            finished.set()

    timer = threading.Timer(0.04, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        outcomes = run_ordered(
            [ScheduledTool(inv, ToolPurity.PURE_READ, hung_read)],
            timeout=5,
            should_cancel=cancel.is_set,
        )
        assert outcomes[0].status is ToolStatus.INDETERMINATE
        assert time.monotonic() - started < 0.5, "cancellation must be polled inside the read wave"
    finally:
        timer.cancel()
        timer.join(timeout=1)
        release.set()
        assert finished.wait(1), "the cancelled daemon read fixture must settle after release"


@check
def no_timeout_parallel_reads_honor_cancellation_without_joining_workers():
    release = threading.Event()
    cancel = threading.Event()
    finished = threading.Event()
    lock = threading.Lock()
    finished_count = 0

    def task(index):
        invocation = ToolInvocation(f"no-timeout-{index}", "read_file", {"path": str(index)}, index)

        def read():
            nonlocal finished_count
            try:
                release.wait()
                return ToolOutcome(invocation, ToolStatus.SUCCEEDED, "late")
            finally:
                with lock:
                    finished_count += 1
                    if finished_count == 2:
                        finished.set()

        return ScheduledTool(invocation, ToolPurity.PURE_READ, read)

    timer = threading.Timer(0.04, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        outcomes = run_ordered([task(0), task(1)], should_cancel=cancel.is_set)
        assert all(outcome.status is ToolStatus.INDETERMINATE for outcome in outcomes), outcomes
        assert time.monotonic() - started < 0.5, "cancellation must not join no-timeout read workers"
    finally:
        timer.cancel()
        timer.join(timeout=1)
        release.set()
        assert finished.wait(1), "both abandoned read fixtures must eventually release"


@check
def sigint_does_not_freeze_on_no_timeout_parallel_reads():
    code = textwrap.dedent("""
        import os
        import threading
        from sliceagent.execution import ToolInvocation, ToolOutcome, ToolPurity, ToolStatus
        from sliceagent.scheduler import ScheduledTool, run_ordered

        gate = threading.Event()
        ready_lock = threading.Lock()
        ready_path = __import__("sys").argv[1]
        def task(index):
            invocation = ToolInvocation(str(index), "read_file", {"path": str(index)}, index)
            def read():
                with ready_lock:
                    with open(ready_path, "a", encoding="utf-8") as ready:
                        ready.write(f"{index}\\n")
                gate.wait()
                return ToolOutcome(invocation, ToolStatus.SUCCEEDED, "late")
            return ScheduledTool(invocation, ToolPurity.PURE_READ, read)
        try:
            run_ordered([task(0), task(1)])
        except KeyboardInterrupt:
            os._exit(42)
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src"))
    with tempfile.TemporaryDirectory() as directory:
        ready_path = os.path.join(directory, "ready")
        process = subprocess.Popen(
            [sys.executable, "-c", code, ready_path], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    if len(open(ready_path, encoding="utf-8").read().splitlines()) == 2:
                        break
                except OSError:
                    pass
                time.sleep(0.02)
            else:
                output, _ = process.communicate(timeout=1)
                raise AssertionError(f"parallel read workers never became ready: {output!r}")
            process.send_signal(signal.SIGINT)
            output, _ = process.communicate(timeout=1)
            assert process.returncode == 42, (process.returncode, output)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=2)
            raise AssertionError(f"SIGINT remained stuck joining read workers: {output!r}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)


@check
def completed_wave_is_retried_if_interrupt_crosses_scheduler_publication():
    invocation = ToolInvocation("completed", "read_file", {"path": "done"}, 0)
    deliveries = []

    def deliver(outcomes):
        deliveries.append(tuple((out.invocation.id, out.status) for out in outcomes))
        if len(deliveries) == 1:
            raise KeyboardInterrupt

    try:
        run_ordered([
            ScheduledTool(
                invocation, ToolPurity.PURE_READ,
                lambda: ToolOutcome(invocation, ToolStatus.SUCCEEDED, "sealed"),
            ),
        ], on_outcomes=deliver)
        assert False, "the user interrupt must remain observable after recovery delivery"
    except KeyboardInterrupt:
        pass

    assert deliveries == [
        (("completed", ToolStatus.SUCCEEDED),),
        (("completed", ToolStatus.SUCCEEDED),),
    ], "the materialized physical outcome must be handed off before SIGINT escapes"


@check
def interrupt_on_first_terminal_edge_preserves_every_completed_sibling():
    from sliceagent.events import ToolSettled

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, _name, args):
            return f"sealed:{args['path']}"

    for interrupted_type in (ToolSettled, ToolResult):
        events = []
        interrupted = [False]

        def dispatch(event):
            if isinstance(event, interrupted_type) and not interrupted[0]:
                interrupted[0] = True
                # Raise before accepting the edge. Recovery may then retry it exactly once; a dispatcher that
                # accepted and raised would still be safe because required sinks key lifecycle rows by ID.
                raise KeyboardInterrupt
            events.append(event)

        try:
            run_tool_batch([
                _tc("read_file", {"path": "a"}, "child-a"),
                _tc("read_file", {"path": "b"}, "child-b"),
            ], Host(), dispatch, Hooks())
            assert False, "the recovered batch must still propagate the user's interrupt"
        except KeyboardInterrupt:
            pass

        settled = [event for event in events if isinstance(event, ToolSettled)]
        results = [event for event in events if isinstance(event, ToolResult)]
        assert [event.outcome.invocation.id for event in settled] == ["child-a", "child-b"]
        assert [event.invocation_id for event in results] == ["child-a", "child-b"]
        assert all(event.status == "succeeded" for event in results), \
            "finished siblings must never be re-labelled indeterminate"


@check
def interrupted_rejection_settlement_and_result_edges_are_completed_in_order():
    from sliceagent.events import ToolRejected, ToolSettled
    from sliceagent.hooks import ToolPreflight

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, _name, _args):
            raise AssertionError("a preflight-stopped handler must not run")

    class Stop(Hooks):
        def preflight_tool(self, _name, _args):
            return ToolPreflight(True, "fixture stop", kind="lifecycle")

    for interrupted_type in (ToolRejected, ToolSettled, ToolResult):
        events = []
        interrupted = [False]

        def dispatch(event):
            if isinstance(event, interrupted_type) and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt
            events.append(event)

        try:
            run_tool_batch([_tc("opaque", {}, "stopped")], Host(), dispatch, Stop())
            assert False, "the recovered rejection must still propagate the user's interrupt"
        except KeyboardInterrupt:
            pass

        terminal = [event for event in events if isinstance(
            event, (ToolRejected, ToolSettled, ToolResult),
        )]
        assert [type(event) for event in terminal] == [ToolRejected, ToolSettled, ToolResult]
        assert terminal[-1].status == "cancelled"


@check
def interrupt_inside_start_publication_closes_partial_start_without_running_handler():
    from sliceagent.events import ToolExecutionStarted, ToolSettled, ToolStarted

    events = []
    handler_ran = []
    interrupted = [False]

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, _name, _args):
            handler_ran.append(True)
            return "unexpected"

    def dispatch(event):
        events.append(event)
        if isinstance(event, ToolExecutionStarted) and not interrupted[0]:
            interrupted[0] = True
            # Model a required journal that durably accepted the start row just before SIGINT.
            raise KeyboardInterrupt

    try:
        run_tool_batch([_tc("opaque", {}, "partial-start")], Host(), dispatch, Hooks())
        assert False, "the recovered partial start must still propagate the user's interrupt"
    except KeyboardInterrupt:
        pass

    assert handler_ran == []
    assert not any(isinstance(event, ToolStarted) for event in events)
    settlements = [event for event in events if isinstance(event, ToolSettled)]
    results = [event for event in events if isinstance(event, ToolResult)]
    assert len(settlements) == len(results) == 1
    assert settlements[0].outcome.status is ToolStatus.INDETERMINATE
    assert "handler did not run" in results[0].output
    assert "start record may be partial" in results[0].output


@check
def launched_but_unentered_reader_never_starts_after_deadline_settlement():
    import sliceagent.scheduler as scheduler

    original_thread = scheduler.threading.Thread
    release_entry = threading.Event()
    events = []

    class DelayedThread(original_thread):
        def run(self):
            release_entry.wait()
            super().run()

    scheduler.threading.Thread = DelayedThread
    invocation = ToolInvocation("late-entry", "read_file", {"path": "x"}, 0)
    task = ScheduledTool(
        invocation, ToolPurity.PURE_READ,
        lambda: (events.append("handler"),
                 ToolOutcome(invocation, ToolStatus.SUCCEEDED, "late"))[1],
        on_start=lambda: events.append("started"),
    )
    try:
        outcome = run_ordered([task], timeout=0.01)
        assert outcome[0].status is ToolStatus.CANCELLED, outcome
        assert events == [], "an unentered call must settle as not-started"
        release_entry.set()
        time.sleep(0.05)
        assert events == [], "a settled call must never announce/start later"
    finally:
        release_entry.set()
        scheduler.threading.Thread = original_thread


@check
def timed_read_waits_for_a_concurrent_slot_until_its_own_deadline():
    import sliceagent.scheduler as scheduler

    original_slots = scheduler._TIMEOUT_READER_SLOTS
    occupied = threading.BoundedSemaphore(1)
    assert occupied.acquire(blocking=False)
    scheduler._TIMEOUT_READER_SLOTS = occupied
    release_slot = threading.Timer(0.03, occupied.release)
    invocation = ToolInvocation("wait-slot", "read_file", {"path": "x"}, 0)
    release_slot.start()
    try:
        outcome = run_ordered([
            ScheduledTool(
                invocation, ToolPurity.PURE_READ,
                lambda: ToolOutcome(invocation, ToolStatus.SUCCEEDED, "ok"),
            ),
        ], timeout=0.2)
        assert outcome[0].status is ToolStatus.SUCCEEDED, outcome
    finally:
        release_slot.cancel()
        release_slot.join(timeout=1)
        scheduler._TIMEOUT_READER_SLOTS = original_slots


@check
def exhausted_reader_slots_settle_without_a_configured_tool_timeout():
    import sliceagent.scheduler as scheduler

    original_slots = scheduler._TIMEOUT_READER_SLOTS
    occupied = threading.BoundedSemaphore(1)
    assert occupied.acquire(blocking=False)
    scheduler._TIMEOUT_READER_SLOTS = occupied
    invocation = ToolInvocation("capacity", "read_file", {"path": "x"}, 0)
    ran = []
    started = time.monotonic()
    try:
        outcome = run_ordered([
            ScheduledTool(
                invocation, ToolPurity.PURE_READ,
                lambda: (ran.append(True), ToolOutcome(
                    invocation, ToolStatus.SUCCEEDED, "unexpected",
                ))[1],
            ),
        ])
        assert time.monotonic() - started < 0.5
        assert outcome[0].status is ToolStatus.CANCELLED
        assert "capacity" in outcome[0].text
        assert ran == []
    finally:
        scheduler._TIMEOUT_READER_SLOTS = original_slots
        occupied.release()


@check
def lifecycle_read_does_not_disable_an_adjacent_read_deadline():
    release = threading.Event()
    finished = threading.Event()
    read_inv = ToolInvocation("safe-timeout", "read_file", {"path": "fifo"}, 0)
    child_inv = ToolInvocation("child", "spawn_agent", {"task": "inspect"}, 1)
    child_ran = []

    def read():
        try:
            release.wait()
            return ToolOutcome(read_inv, ToolStatus.SUCCEEDED, "late")
        finally:
            finished.set()

    def child():
        child_ran.append(True)
        return ToolOutcome(child_inv, ToolStatus.SUCCEEDED, "child")

    started = time.monotonic()
    try:
        outcomes = run_ordered([
            ScheduledTool(read_inv, ToolPurity.PURE_READ, read, timeout_safe=True),
            ScheduledTool(child_inv, ToolPurity.PURE_READ, child, timeout_safe=False),
        ], timeout=0.02)
        assert time.monotonic() - started < 0.4
        assert [outcome.status for outcome in outcomes] == [
            ToolStatus.INDETERMINATE, ToolStatus.CANCELLED,
        ]
        assert child_ran == []
    finally:
        release.set()
        assert finished.wait(1)


@check
def late_indeterminate_read_still_closes_later_effect_barriers():
    read_inv = ToolInvocation("late-unknown", "read_file", {"path": "remote"}, 0)
    edit_inv = ToolInvocation("later-edit", "edit_file", {"path": "x"}, 1)
    edits = []

    def uncertain_read():
        time.sleep(0.04)
        return ToolOutcome(read_inv, ToolStatus.INDETERMINATE, "remote result uncertain")

    outcomes = run_ordered([
        ScheduledTool(read_inv, ToolPurity.PURE_READ, uncertain_read),
        ScheduledTool(
            edit_inv, ToolPurity.EFFECTFUL,
            lambda: (edits.append(True), ToolOutcome(
                edit_inv, ToolStatus.SUCCEEDED, "edited",
            ))[1],
        ),
    ], timeout=0.02)
    assert [outcome.status for outcome in outcomes] == [
        ToolStatus.INDETERMINATE, ToolStatus.CANCELLED,
    ]
    assert edits == []


@check
def blocking_start_publication_times_out_without_entering_handler_or_late_tool_started():
    from sliceagent.events import ToolExecutionStarted, ToolStarted

    gate = threading.Event()
    publication_entered = threading.Event()
    handler_ran = []
    events = []
    invocation = ToolInvocation("slow-start", "read_file", {"path": "x"}, 0)

    def dispatch(event):
        if isinstance(event, ToolExecutionStarted):
            publication_entered.set()
            gate.wait()
        events.append(event)

    class Host:
        def accesses(self, _name, _args):
            from sliceagent.access import ReadAllAccess
            return [ReadAllAccess()]

        def run(self, _name, _args):
            handler_ran.append(True)
            return "unexpected"

    result = []

    def invoke():
        result.extend(run_tool_batch([
            _tc("read_file", {"path": "x"}, invocation.id),
        ], Host(), dispatch, Hooks())[1])

    prior = os.environ.get("AGENT_TOOL_TIMEOUT")
    os.environ["AGENT_TOOL_TIMEOUT"] = "0.02"
    thread = threading.Thread(target=invoke, daemon=True)
    try:
        thread.start()
        assert publication_entered.wait(1)
        thread.join(0.4)
        assert not thread.is_alive(), "deadline must remain enforceable while start publication blocks"
        assert result[0]["status"] == "indeterminate"
        assert handler_ran == []
        gate.set()
        time.sleep(0.05)
        assert not any(isinstance(event, ToolStarted) for event in events), \
            "the guarded start boundary must not publish ToolStarted after settlement"
    finally:
        gate.set()
        thread.join(1)
        if prior is None:
            os.environ.pop("AGENT_TOOL_TIMEOUT", None)
        else:
            os.environ["AGENT_TOOL_TIMEOUT"] = prior


@check
def in_flight_tool_started_is_pinned_to_original_dispatch_epoch():
    from sliceagent.access import ReadAllAccess
    from sliceagent.events import ToolStarted, make_dispatcher

    started_edge = threading.Event()
    release_edge = threading.Event()
    old_events, new_events, handler_ran = [], [], []
    route = {"sink": None}

    def old_sink(event):
        if isinstance(event, ToolStarted):
            started_edge.set()
            release_edge.wait()
        old_events.append(event)

    def new_sink(event):
        new_events.append(event)

    route["sink"] = old_sink

    def router(event):
        route["sink"](event)

    def bind_router():
        return route["sink"]

    router.bind_dispatch = bind_router
    dispatch = make_dispatcher(required=(router,))

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, _args):
            handler_ran.append(True)
            return "unexpected"

    prior = os.environ.get("AGENT_TOOL_TIMEOUT")
    os.environ["AGENT_TOOL_TIMEOUT"] = "0.02"
    result = []
    thread = threading.Thread(
        target=lambda: result.extend(run_tool_batch([
            _tc("read_file", {"path": "x"}, "epoch-start"),
        ], Host(), dispatch, Hooks())[1]),
        daemon=True,
    )
    try:
        thread.start()
        assert started_edge.wait(1)
        thread.join(0.4)
        assert not thread.is_alive() and result[0]["status"] == "indeterminate"
        route["sink"] = new_sink       # simulate a new turn/workspace becoming the router target
        release_edge.set()
        time.sleep(0.05)
        assert handler_ran == []
        assert any(isinstance(event, ToolStarted) for event in old_events)
        assert not any(isinstance(event, ToolStarted) for event in new_events), \
            "an admitted edge already in flight must remain pinned to the original dispatch epoch"
    finally:
        release_edge.set()
        thread.join(1)
        if prior is None:
            os.environ.pop("AGENT_TOOL_TIMEOUT", None)
        else:
            os.environ["AGENT_TOOL_TIMEOUT"] = prior


@check
def dispatcher_detaches_nested_event_payloads_for_every_sink():
    from sliceagent.events import ToolStarted, make_dispatcher

    invocation = ToolInvocation("detach", "read_file", {"path": "truth"}, 0)
    original = ToolStarted("read_file", {"path": "truth", "nested": {"value": 1}}, invocation)
    observed = []

    def corrupt(event):
        event.args["path"] = "corrupted"
        event.args["nested"]["value"] = 99
        event.invocation.args["path"] = "also-corrupted"

    def observe(event):
        observed.append(event)

    make_dispatcher(corrupt, observe)(original)
    assert observed[0].args == {"path": "truth", "nested": {"value": 1}}
    assert dict(observed[0].invocation.args) == {"path": "truth"}
    assert original.args == {"path": "truth", "nested": {"value": 1}}
    assert dict(original.invocation.args) == {"path": "truth"}


@check
def cancelled_lifecycle_read_wave_caps_abandoned_workers_without_recursive_read_slots():
    import sliceagent.scheduler as scheduler

    original_slots = scheduler._LIFECYCLE_READER_SLOTS
    scheduler._LIFECYCLE_READER_SLOTS = threading.BoundedSemaphore(1)
    release = threading.Event()
    cancel = threading.Event()
    started = []

    def task(index):
        invocation = ToolInvocation(f"life-{index}", "spawn_agent", {"task": str(index)}, index)

        def read():
            started.append(index)
            release.wait()
            return ToolOutcome(invocation, ToolStatus.SUCCEEDED, "late")

        return ScheduledTool(invocation, ToolPurity.PURE_READ, read, timeout_safe=False)

    timer = threading.Timer(0.04, cancel.set)
    timer.start()
    try:
        outcomes = run_ordered([task(0), task(1)], should_cancel=cancel.is_set)
        assert [outcome.status for outcome in outcomes] == [
            ToolStatus.INDETERMINATE, ToolStatus.CANCELLED,
        ]
        assert started == [0], "the lifecycle cap must keep the queued child provably unstarted"
    finally:
        timer.cancel()
        timer.join(timeout=1)
        release.set()
        scheduler._LIFECYCLE_READER_SLOTS = original_slots


@check
def queued_read_never_starts_after_cancellation_cutoff():
    cancel = threading.Event()
    second_started = []
    first_inv = ToolInvocation("first", "read_file", {"path": "first"}, 0)
    second_inv = ToolInvocation("second", "read_file", {"path": "second"}, 1)

    def first():
        cancel.set()
        return ToolOutcome(first_inv, ToolStatus.SUCCEEDED, "first")

    def second():
        second_started.append(True)
        return ToolOutcome(second_inv, ToolStatus.SUCCEEDED, "second")

    outcomes = run_ordered([
        ScheduledTool(first_inv, ToolPurity.PURE_READ, first),
        ScheduledTool(second_inv, ToolPurity.PURE_READ, second),
    ], max_workers=1, timeout=5, should_cancel=cancel.is_set)
    assert [outcome.status for outcome in outcomes] == [ToolStatus.SUCCEEDED, ToolStatus.CANCELLED]
    assert not second_started, "a queued read must not start after cancellation established the cutoff"


@check
def timed_read_worker_cap_cancels_unstarted_calls_without_leaking_threads():
    import sliceagent.scheduler as scheduler

    original_slots = scheduler._TIMEOUT_READER_SLOTS
    scheduler._TIMEOUT_READER_SLOTS = threading.BoundedSemaphore(2)
    release = threading.Event()
    all_finished = threading.Event()
    lock = threading.Lock()
    counts = {"started": 0, "finished": 0}
    ran_effect = []

    def task(index):
        invocation = ToolInvocation(f"cap-{index}", "read_file", {"path": str(index)}, index)

        def read():
            with lock:
                counts["started"] += 1
            try:
                release.wait()
                return ToolOutcome(invocation, ToolStatus.SUCCEEDED, "late")
            finally:
                with lock:
                    counts["finished"] += 1
                    if counts["finished"] == 2:
                        all_finished.set()

        return ScheduledTool(invocation, ToolPurity.PURE_READ, read)

    effect_inv = ToolInvocation("after-cap", "edit_file", {"path": "later"}, 3)

    def effect():
        ran_effect.append(True)
        return ToolOutcome(effect_inv, ToolStatus.SUCCEEDED, "edited")

    try:
        outcomes = run_ordered([
            task(0), task(1), task(2),
            ScheduledTool(effect_inv, ToolPurity.EFFECTFUL, effect),
        ], max_workers=3, timeout=0.03)
        statuses = [outcome.status for outcome in outcomes]
        assert statuses[:3].count(ToolStatus.INDETERMINATE) == 2
        assert statuses[:3].count(ToolStatus.CANCELLED) == 1
        assert statuses[3] is ToolStatus.CANCELLED
        assert counts["started"] == 2, "the slot cap must prevent a third daemon reader from starting"
        assert not ran_effect
    finally:
        scheduler._TIMEOUT_READER_SLOTS = original_slots
        release.set()
        assert all_finished.wait(1), "every admitted daemon fixture must release its captured slot"


@check
def lifecycle_preflight_is_resolved_after_each_prior_barrier_settles():
    focus = {"value": "old"}
    observed, ran = [], []

    class BarrierHooks(Hooks):
        def preflight_tool(self, name, _args):
            from sliceagent.hooks import ToolPreflight
            observed.append((name, focus["value"]))
            if name == "second" and focus["value"] != "new":
                return ToolPreflight(True, "lifecycle hook observed stale focus", kind="lifecycle")
            return ToolPreflight()

        def transform_tool_result(self, name, _args, _output):
            if name == "first":
                focus["value"] = "new"

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(name)
            return "ok"

    _, results = run_tool_batch(
        [_tc("first", {}, "first"), _tc("second", {}, "second")],
        Host(), lambda _event: None, BarrierHooks(),
    )
    assert ran == ["first", "second"]
    assert observed == [("first", "old"), ("second", "new")], observed
    assert all(result["status"] == "succeeded" for result in results)


@check
def cancellation_after_model_return_prevents_returned_mutation():
    signal = threading.Event()
    ran = []

    class LLM:
        def complete(self, _messages, _schemas):
            signal.set()
            return NS(content="", tool_calls=[_tc("edit_file", {"path": "x"}, "edit")],
                      finish_reason="tool_calls", usage={"prompt_tokens": 2})

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(name)
            return "edited"

    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "go"}],
        llm=LLM(), tools=Host(), dispatch=lambda _event: None, hooks=Hooks(), signal=signal,
    )
    assert result.stop_reason == "aborted" and ran == []
    assert result.usage.prompt_tokens == 2


@check
def cancellation_during_barrier_preparation_prevents_the_effect():
    signal = threading.Event()
    ran = []
    invocation = ToolInvocation("one", "effect", {}, 0)

    def prepare():
        signal.set()
        return None

    outcome = run_ordered([
        ScheduledTool(
            invocation, ToolPurity.EFFECTFUL,
            lambda: (ran.append("effect"), ToolOutcome(invocation, ToolStatus.SUCCEEDED, "ok"))[1],
            prepare=prepare,
        ),
    ], should_cancel=signal.is_set)
    assert ran == []
    assert len(outcome) == 1 and outcome[0].status is ToolStatus.CANCELLED


@check
def cancellation_between_barriers_publishes_cancelled_tail():
    signal = threading.Event()
    ran = []

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(name)
            if name == "first_write":
                signal.set()
            return "ok"

    _, results = run_tool_batch([
        _tc("first_write", {}, "one"), _tc("second_write", {}, "two"),
    ], Host(), lambda _event: None, Hooks(), signal=signal)
    assert ran == ["first_write"]
    assert [result["status"] for result in results] == ["succeeded", "cancelled"]


@check
def typed_child_usage_is_aggregated_and_stops_before_another_parent_call():
    from sliceagent.execution import ToolEffect

    class LLM:
        def __init__(self):
            self.calls = 0

        def complete(self, _messages, _schemas):
            self.calls += 1
            if self.calls == 1:
                return NS(content="", tool_calls=[_tc("spawn_agent", {"task": "inspect"}, "child")],
                          finish_reason="tool_calls",
                          usage={"prompt_tokens": 1, "completion_tokens": 1})
            return NS(content="should not happen", tool_calls=[], finish_reason="stop",
                      usage={"prompt_tokens": 1, "completion_tokens": 1})

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            return []

        def run(self, _name, _args):
            return ToolText(
                "child report", effects=(ToolEffect(
                    "child-1:model-usage", "model_usage",
                    {"prompt_tokens": 90, "completion_tokens": 10},
                ),),
            )

    llm = LLM()
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "go"}],
        llm=llm, tools=Host(), dispatch=lambda _event: None,
        hooks=BudgetHook(10), max_steps=3,
    )
    assert result.stop_reason == "token_budget" and llm.calls == 1
    assert result.usage.prompt_tokens == 91 and result.usage.completion_tokens == 11


@check
def parallel_children_split_only_the_remaining_parent_budget():
    from sliceagent.access import ReadAllAccess

    seen = []

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, args):
            seen.append(dict(args))
            return "done"

    budget = BudgetHook(30)
    budget.reset_for_turn()
    budget.record_step_usage({"prompt_tokens": 8, "completion_tokens": 2})
    calls = [
        _tc("spawn_agent", {"agent": "explorer", "task": "one"}, "one"),
        _tc("spawn_agent", {"agent": "explorer", "task": "two"}, "two"),
    ]
    _, results = run_tool_batch(calls, Host(), lambda _event: None, budget)
    assert len(seen) == 2
    assert {args[CHILD_TOKEN_BUDGET_ARG] for args in seen} == {10}
    by_task = {args["task"]: args for args in seen}
    assert by_task["one"][CHILD_INVOCATION_ID_ARG] == "one"
    assert by_task["two"][CHILD_INVOCATION_ID_ARG] == "two"
    assert by_task["one"][CHILD_REQUEST_ORDINAL_ARG] == 1
    assert by_task["two"][CHILD_REQUEST_ORDINAL_ARG] == 2
    private = {CHILD_TOKEN_BUDGET_ARG, CHILD_INVOCATION_ID_ARG, CHILD_REQUEST_ORDINAL_ARG}
    assert all(not private.intersection(result["args"]) for result in results), \
        "scheduler metadata must not leak into the canonical invocation"


@check
def child_budget_reservation_survives_an_effect_barrier_between_explorers():
    from sliceagent.access import AllAccess, ReadAllAccess

    seen = []

    class Host:
        def accesses(self, name, _args):
            return [ReadAllAccess()] if name == "spawn_agent" else [AllAccess()]

        def run(self, name, args):
            seen.append((name, dict(args)))
            return "done"

    budget = BudgetHook(30)
    budget.reset_for_turn()
    budget.record_step_usage({"prompt_tokens": 8, "completion_tokens": 2})
    calls = [
        _tc("spawn_agent", {"agent": "explorer", "task": "one"}, "one"),
        _tc("edit_file", {"path": "a.py", "content": "x"}, "barrier"),
        _tc("spawn_agent", {"agent": "explorer", "task": "two"}, "two"),
    ]
    run_tool_batch(calls, Host(), lambda _event: None, budget)
    caps = [args[CHILD_TOKEN_BUDGET_ARG] for name, args in seen if name == "spawn_agent"]
    assert caps == [10, 10]
    assert sum(caps) == budget.remaining_token_budget(), \
        "separate child waves must share one fair parent reservation"


@check
def writable_child_barrier_waves_cannot_multiply_the_parent_budget():
    from sliceagent.access import AllAccess

    caps = []

    class Host:
        def accesses(self, _name, _args):
            return [AllAccess()]

        def run(self, _name, args):
            caps.append(args[CHILD_TOKEN_BUDGET_ARG])
            return "done"

    budget = BudgetHook(24)
    calls = [
        _tc("spawn_agent", {"agent": "general", "task": "one"}, "one"),
        _tc("spawn_agent", {"agent": "general", "task": "two"}, "two"),
        _tc("spawn_agent", {"agent": "general", "task": "three"}, "three"),
    ]
    run_tool_batch(calls, Host(), lambda _event: None, budget)
    assert caps == [8, 8, 8]
    assert sum(caps) <= budget.remaining_token_budget(), \
        "serialized child barriers must split, not multiply or starve, the remaining budget"


@check
def cancelled_child_does_not_consume_an_allowed_siblings_reservation():
    from sliceagent.access import ReadAllAccess
    from sliceagent.hooks import ToolPreflight

    seen = []

    class SelectiveBudget(BudgetHook):
        def preflight_tool(self, _name, args):
            return ToolPreflight(args.get("task") == "cancelled", "cancelled for test", kind="lifecycle")

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, args):
            seen.append(dict(args))
            return "done"

    budget = SelectiveBudget(20)
    calls = [
        _tc("spawn_agent", {"agent": "explorer", "task": "cancelled"}, "cancelled"),
        _tc("spawn_agent", {"agent": "explorer", "task": "allowed"}, "allowed"),
    ]
    _, results = run_tool_batch(calls, Host(), lambda _event: None, budget)
    assert len(seen) == 1 and seen[0][CHILD_TOKEN_BUDGET_ARG] == 20
    assert results[0]["status"] == "cancelled" and results[1]["status"] == "succeeded"


@check
def registry_rejected_child_does_not_consume_valid_child_budget_same_or_later_wave():
    from sliceagent.access import AllAccess, ReadAllAccess

    for interleaved in (False, True):
        caps = []
        registry = ToolRegistry()
        registry.register(ToolEntry(
            "spawn_agent", {"type": "function", "function": {
                "name": "spawn_agent", "parameters": {
                    "type": "object", "properties": {}, "required": ["task"],
                },
            }}, lambda args: caps.append(args[CHILD_TOKEN_BUDGET_ARG]) or "done",
            accesses=lambda _args: [ReadAllAccess()], purity=ToolPurity.PURE_READ,
        ))
        registry.register(ToolEntry(
            "barrier", {"type": "function", "function": {
                "name": "barrier", "parameters": {},
            }}, lambda _args: "done", accesses=lambda _args: [AllAccess()],
            purity=ToolPurity.EFFECTFUL,
        ))

        class Host:
            def accesses(self, name, args):
                return registry.accesses(name, args)

            def preflight_run(self, name, args):
                return registry.admit(name, args)

            def run_preflighted(self, name, args, admission):
                return registry.run_admitted(admission, args)

            def run(self, name, args):
                return registry.run(name, args)

        calls = [_tc("spawn_agent", {"agent": "explorer"}, "invalid")]
        if interleaved:
            calls.append(_tc("barrier", {}, "barrier"))
        calls.append(_tc(
            "spawn_agent", {"agent": "explorer", "task": "valid"}, "valid",
        ))
        _, rows = run_tool_batch(calls, Host(), lambda _event: None, BudgetHook(20))
        assert caps == [20], (
            "a child proven not started must not reserve budget from a valid sibling"
        )
        assert rows[0]["status"] == "failed" and rows[-1]["status"] == "succeeded"


@check
def malformed_advisory_child_budget_never_crashes_or_blocks_delegation():
    from sliceagent.access import ReadAllAccess

    seen = []

    class MalformedBudget(Hooks):
        def remaining_token_budget(self):
            return "unbounded-ish"

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, args):
            seen.append(dict(args))
            return "done"

    _, rows = run_tool_batch([
        _tc("spawn_agent", {"agent": "explorer", "task": "one"}, "one"),
        _tc("spawn_agent", {"agent": "explorer", "task": "two"}, "two"),
    ], Host(), lambda _event: None, MalformedBudget())
    assert [row["status"] for row in rows] == ["succeeded", "succeeded"]
    assert len(seen) == 2
    assert all(CHILD_TOKEN_BUDGET_ARG not in args for args in seen), \
        "malformed advisory budget means no cap opinion, not a batch crash"


@check
def preflight_counts_schemas_and_output_reserve():
    llm = NS(context_window=180, max_tokens=40)
    messages = [{"role": "user", "content": "m" * 50}]
    schemas = [{"type": "function", "function": {"name": "x", "description": "s" * 80}}]
    try:
        preflight_model_call(llm, messages, schemas, allow_unknown=False)
        assert False, "strict preflight must reject the over-capacity request"
    except PreflightOverflow as error:
        report = error.report
        assert report.schema_tokens > 0 and report.output_reserve == 40
        assert report.required_tokens > report.context_window


@check
def unknown_window_is_named_compatibility_mode():
    report = preflight_model_call(NS(max_tokens=10), [{"role": "user", "content": "x"}], [],
                                  allow_unknown=True)
    assert report.context_window == 0
    assert report.mode == "compatibility-unknown"


@check
def shared_model_runner_preflights_and_owns_retry_policy():
    import sliceagent.errors as errors

    class LLM:
        context_window = 500
        max_tokens = 20

        def __init__(self):
            self.calls = 0

        @staticmethod
        def is_retryable(_error):
            return True

        def complete(self, _messages, _schemas):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            return "ok"

    llm, events = LLM(), []
    old_sleep = errors.time.sleep
    errors.time.sleep = lambda _delay: None
    try:
        assert complete_model_call(
            llm, [{"role": "user", "content": "small"}], [], dispatch=events.append,
        ) == "ok"
    finally:
        errors.time.sleep = old_sleep
    assert llm.calls == 2
    assert len([event for event in events if isinstance(event, ApiRetry)]) == 1

    llm.context_window = 10
    try:
        complete_model_call(llm, [{"role": "user", "content": "too large"}], [], retry=False,
                            allow_unknown=False)
        assert False, "strict capacity preflight must happen before provider I/O"
    except PreflightOverflow:
        pass
    assert llm.calls == 2


@check
def turn_outcome_keeps_legacy_usage_mapping():
    result = TurnOutcome("end_turn", 2, {"prompt_tokens": 7, "completion_tokens": 3})
    assert result.stop_reason == "end_turn"
    assert isinstance(result.usage, Usage)
    assert result.usage["prompt_tokens"] == 7
    assert dict(result.usage)["completion_tokens"] == 3


@check
def required_pre_dispatch_failure_prevents_tool_execution():
    ran = []
    invocation = ToolInvocation("call-1", "edit_file", {"path": "a.py"}, 0)

    def journal_start():
        raise OSError("journal unavailable")

    task = ScheduledTool(
        invocation, ToolPurity.EFFECTFUL,
        lambda: ran.append(True), on_start=journal_start,
    )
    try:
        run_ordered([task])
        assert False, "an unjournaled effectful call must not run"
    except OSError as exc:
        assert "journal unavailable" in str(exc)
    assert ran == []


@check
def failed_required_reduction_stops_before_next_mutation_barrier():
    ran = []

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(name)
            return "ok"

    def dispatch(event):
        from sliceagent.events import ToolResult
        if isinstance(event, ToolResult):
            raise OSError("reducer unavailable")

    calls = [_tc("first_write", {}, "one"), _tc("second_write", {}, "two")]
    try:
        run_tool_batch(calls, Host(), dispatch, Hooks())
        assert False, "the failed first barrier must stop the batch"
    except OSError:
        pass
    assert ran == ["first_write"]


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
