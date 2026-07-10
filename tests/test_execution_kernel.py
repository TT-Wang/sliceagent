"""Typed execution-kernel invariants. No network/model dependency."""
from __future__ import annotations

import os
import signal
import shlex
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import ApiRetry, TurnEnd, TurnInterrupted  # noqa: E402
from sliceagent.execution import (CHILD_TOKEN_BUDGET_ARG, PreflightOverflow, ToolEffect, ToolInvocation,
                                  ToolPurity, ToolStatus, TurnOutcome, Usage,
                                  preflight_model_call, reconciliation_targets)  # noqa: E402
from sliceagent.hooks import (BudgetHook, CompositeHooks, Hooks, OracleHook,
                              ReconciliationHook)  # noqa: E402
from sliceagent.loop import run_tool_batch, run_turn  # noqa: E402
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
def production_policy_block_uses_canonical_effect_without_running_handler():
    from sliceagent.hooks import ToolDecision

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

    class Deny(Hooks):
        def authorize_tool(self, _name, _args):
            return ToolDecision(False, "denied for test")

    _, rows = run_tool_batch(
        [_tc("edit_file", {"path": "a.py", "note": "keep raw"}, "blocked")],
        Host(), lambda _event: None, Deny(), step=4, turn_id="turn-P",
    )
    assert ran == []
    outcome = rows[0]["outcome"]
    assert outcome.status is ToolStatus.FAILED
    assert outcome.effects[0] == ToolEffect(
        "turn-P:4:0:blocked:0", "tool_outcome", {"name": "edit_file", "status": "failed"},
    )


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
def indeterminate_tool_parks_without_another_model_call():
    prior = os.environ.get("AGENT_TOOL_TIMEOUT")
    os.environ["AGENT_TOOL_TIMEOUT"] = "0.03"

    class LLM:
        def __init__(self):
            self.calls = 0

        def complete(self, _messages, _schemas):
            self.calls += 1
            return NS(content="", tool_calls=[_tc("read_file", {"path": "slow"}, "slow")],
                      finish_reason="tool_calls", usage={})

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            return []

        def run(self, _name, _args):
            time.sleep(0.12)
            return "late"

    llm, events = LLM(), []
    try:
        result = run_turn(
            build_slice=lambda: [{"role": "user", "content": "go"}],
            llm=llm, tools=Host(), dispatch=events.append, hooks=Hooks(), max_steps=4,
        )
        assert result.stop_reason == "indeterminate"
        assert llm.calls == 1
        assert any(isinstance(e, TurnInterrupted) and e.reason == "indeterminate" for e in events)
        assert not any(isinstance(e, TurnEnd) for e in events)
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
def next_turn_blocks_effects_until_observation_and_explicit_reconciliation():
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
            scripts = {
                1: _tc("edit_file", {"path": "a.py"}, "early"),
                2: _tc("read_file", {"path": "unrelated.txt"}, "unrelated"),
                3: _tc("reconcile_execution", {"resolution": "unrelated exists"}, "too-early"),
                4: _tc("read_file", {"path": "a.py"}, "observe"),
                5: _tc("reconcile_execution", {"resolution": "a.py has no late write"}, "reconcile"),
                6: _tc("edit_file", {"path": "a.py"}, "after"),
            }
            if self.calls in scripts:
                return NS(content="", tool_calls=[scripts[self.calls]],
                          finish_reason="tool_calls", usage=usage)
            return NS(content="done", tool_calls=[], finish_reason="stop", usage=usage)

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(name)
            return ToolText("observed" if name == "read_file" else "ok")

    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "continue safely"}],
        llm=LLM(), tools=Host(), dispatch=slice_sink(state),
        hooks=ReconciliationHook(lambda: state), max_steps=10,
    )
    assert result.stop_reason == "end_turn"
    assert ran == ["read_file", "read_file", "reconcile_execution", "edit_file"]
    assert not state.reconciliation_required
    assert any(signal.kind == "reconciliation" for signal in state.task.progress_signals)


@check
def active_reconciliation_owns_completion_before_effectful_oracles():
    from sliceagent.pfc import Slice

    state = Slice(); state.reset("repair")
    state.reconciliation_required = "late command may still write"
    state.reconciliation_targets = ["path:a.py"]

    class Oracle:
        calls = 0

        def verify(self):
            self.calls += 1
            return True, ""

    oracle = Oracle()
    hooks = CompositeHooks(
        ReconciliationHook(lambda: state), OracleHook(oracle, lambda _out: None),
    )
    result = hooks.should_continue_after_stop("end_turn")
    assert result["continue"] and result["exclusive"]
    assert oracle.calls == 0


@check
def opaque_uncertainty_requires_workspace_inspection_and_user_confirmation():
    from sliceagent.pfc import Slice

    assert reconciliation_targets("run_command", {"command": "curl -X POST example.test"}) == (
        "workspace:*", "opaque:run_command",
    )
    state = Slice(); state.reset("repair")
    state.reconciliation_required = "a detached command may still be running"
    state.reconciliation_targets = ["workspace:*", "opaque:run_command"]
    hook = ReconciliationHook(lambda: state)
    hook.reset_for_turn()

    hook.transform_tool_result("list_files", {"path": "."}, ToolText("a.py"))
    assert not hook.authorize_tool("reconcile_execution", {"resolution": "listed files"}).allow
    hook.transform_tool_result(
        "code_review", {},
        ToolText("[workspace observation: tracked + untracked + ignored inventory complete]\nNo changes"),
    )
    assert not hook.authorize_tool("reconcile_execution", {"resolution": "reviewed files"}).allow
    hook.transform_tool_result(
        "ask_user", {"question": "Did the detached operation settle?"}, ToolText("User answered: yes"),
    )
    assert hook.authorize_tool("reconcile_execution", {"resolution": "reviewed and confirmed"}).allow


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
def observation_must_prove_settlement_not_merely_succeed():
    from sliceagent.pfc import Slice

    cases = [
        (["process:p1"], "proc_poll", {"handle": "p1"}, "running", False),
        (["process:p1"], "proc_poll", {"handle": "p1"}, "exited 0", True),
        (["process:p1"], "proc_wait", {"handle": "p1"},
         "[p1 running (still alive after 1s)]\n(no output yet)", False),
        (["process:p1"], "proc_poll", {"handle": "p1"},
         "leader exited 0; descendants running", False),
        (["terminal:main"], "terminal_read", {"session": "main"},
         "(no output; alive)", False),
        (["terminal:main"], "terminal_read", {"session": "main"},
         "(no output; leader exited 0; descendants alive)", False),
        (["terminal:main"], "terminal_read", {"session": "main"},
         "(no output; exited 0)", True),
        (["path:a.py"], "read_file", {"path": "a.py", "offset": 1, "limit": 20},
         "     1\tx\n<system>read_file a.py: lines 1-20 of 50</system>", False),
        (["path:a.py"], "read_file", {"path": "a.py", "offset": 1, "limit": 1000},
         "     1\tx\n<system>read_file a.py: lines 1-50 of 50</system>", True),
        (["path:a.py"], "read_file", {"path": "a.py"}, "     1\tx", True),
        (["opaque:run_command"], "ask_user", {}, "User answered: No, it is still running", False),
        (["opaque:run_command"], "ask_user", {}, "User answered: yes, it settled", True),
    ]
    for targets, name, args, output, expected in cases:
        state = Slice(); state.reset("repair")
        state.reconciliation_required = "uncertain"
        state.reconciliation_targets = targets
        hook = ReconciliationHook(lambda s=state: s)
        hook.reset_for_turn()
        hook.transform_tool_result(name, args, ToolText(output))
        allowed = hook.authorize_tool("reconcile_execution", {"resolution": "observed"}).allow
        assert allowed is expected, (name, output, allowed)


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
def missing_user_answer_is_not_opaque_confirmation():
    from sliceagent.pfc import Slice

    state = Slice(); state.reset("repair")
    state.reconciliation_required = "a detached command may still be running"
    state.reconciliation_targets = ["workspace:*", "opaque:run_command"]
    hook = ReconciliationHook(lambda: state)
    hook.reset_for_turn()
    hook.transform_tool_result(
        "code_review", {},
        ToolText("[workspace observation: tracked + untracked + ignored inventory complete]\nNo changes"),
    )
    hook.transform_tool_result(
        "ask_user", {"question": "Did it settle?"}, ToolText("No user answer was received.", ok=False),
    )
    assert not hook.authorize_tool("reconcile_execution", {"resolution": "no answer"}).allow

    from sliceagent.tools import LocalToolHost
    host = LocalToolHost(tempfile.mkdtemp(prefix="ask-user-cancel-"))
    host.on_ask_user = lambda _question, _options: "(no answer)"
    output = host.run("ask_user", {"question": "Did it settle?"})
    assert isinstance(output, ToolText) and output.status is ToolStatus.FAILED


@check
def indeterminate_read_publishes_cancelled_outcomes_for_later_barriers():
    prior = os.environ.get("AGENT_TOOL_TIMEOUT")
    os.environ["AGENT_TOOL_TIMEOUT"] = "0.03"
    ran, events = [], []

    class Host:
        def accesses(self, _name, _args):
            return []

        def run(self, name, _args):
            ran.append(name)
            if name == "read_file":
                time.sleep(0.1)
            return "ok"

    try:
        _, results = run_tool_batch([
            _tc("read_file", {"path": "slow"}, "slow"),
            _tc("edit_file", {"path": "later", "content": "x"}, "later"),
        ], Host(), events.append, Hooks())
        assert [result["status"] for result in results] == ["indeterminate", "cancelled"]
        assert ran == ["read_file"], "the later mutation must never start"
        from sliceagent.events import ToolResult
        logical = [event for event in events if isinstance(event, ToolResult)]
        assert [event.invocation_id for event in logical] == ["slow", "later"]
    finally:
        if prior is None:
            os.environ.pop("AGENT_TOOL_TIMEOUT", None)
        else:
            os.environ["AGENT_TOOL_TIMEOUT"] = prior


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
    assert all(CHILD_TOKEN_BUDGET_ARG not in result["args"] for result in results), \
        "scheduler metadata must not leak into the canonical invocation"


@check
def denied_child_does_not_consume_an_allowed_siblings_reservation():
    from sliceagent.access import ReadAllAccess
    from sliceagent.hooks import ToolDecision

    seen = []

    class SelectiveBudget(BudgetHook):
        def authorize_tool(self, _name, args):
            return ToolDecision(args.get("task") != "denied", "denied for test")

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, args):
            seen.append(dict(args))
            return "done"

    budget = SelectiveBudget(20)
    calls = [
        _tc("spawn_agent", {"agent": "explorer", "task": "denied"}, "denied"),
        _tc("spawn_agent", {"agent": "explorer", "task": "allowed"}, "allowed"),
    ]
    _, results = run_tool_batch(calls, Host(), lambda _event: None, budget)
    assert len(seen) == 1 and seen[0][CHILD_TOKEN_BUDGET_ARG] == 20
    assert results[0]["status"] == "failed" and results[1]["status"] == "succeeded"


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
