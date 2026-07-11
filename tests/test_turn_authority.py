"""Semantic tool effects + all-call turn-authority gate. No model/network/pytest.

Run: PYTHONPATH=src python tests/test_turn_authority.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.execution import ToolPurity  # noqa: E402
from sliceagent.events import ToolResult, TurnInterrupted  # noqa: E402
from sliceagent.hooks import (ALLOW, CompositeHooks, PermissionHook, ToolDecision,
                              TurnAuthorityHook)  # noqa: E402
from sliceagent.interfaces import AssistantMessage, ToolCall  # noqa: E402
from sliceagent.intent import analyze_turn  # noqa: E402
from sliceagent.loop import run_turn  # noqa: E402
from sliceagent.registry import (ToolEntry, ToolIntentEffect, ToolRegistry)  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _schema(name):
    return {"type": "function", "function": {
        "name": name, "parameters": {"type": "object", "properties": {}},
    }}


def _entry(name, *, source="builtin", purity=ToolPurity.UNKNOWN,
           intent_effect=ToolIntentEffect.UNKNOWN):
    return ToolEntry(name=name, schema=_schema(name), handler=lambda _args: "ok",
                     source=source, purity=purity, intent_effect=intent_effect)


def _hook(authority, registry):
    return TurnAuthorityHook(
        lambda: {"effect_authority": authority}, registry.resolve_intent_effect,
    )


@check
def builtin_metadata_is_semantic_and_run_command_is_argument_sensitive():
    with tempfile.TemporaryDirectory(prefix="turn-authority-") as root:
        host = LocalToolHost(root)
        try:
            reg = host.registry
            expected = {
                "read_file": ToolIntentEffect.OBSERVE,
                "ask_user": ToolIntentEffect.DIALOGUE,
                "update_plan": ToolIntentEffect.TASK_STATE,
                "edit_file": ToolIntentEffect.EXTERNAL,
                "change_workspace": ToolIntentEffect.EXTERNAL,
            }
            for name, effect in expected.items():
                assert reg.resolve_intent_effect(name, {}) is effect, (name, reg.resolve_intent_effect(name, {}))

            # Reuse policy.py's existing deny-by-default shell parser: only PROVEN readers become OBSERVE.
            assert reg.resolve_intent_effect("run_command", {"command": "git status --short"}) \
                is ToolIntentEffect.OBSERVE
            assert reg.resolve_intent_effect("run_command", {"command": "ls -la src"}) \
                is ToolIntentEffect.OBSERVE
            assert reg.resolve_intent_effect("run_command", {
                "command": "find . -type f -not -path './.git/*' -not -path './node_modules/*' | sort",
            }) is ToolIntentEffect.OBSERVE, "a pipeline of proven readers remains observation"
            for command in ("python -c 'print(1)'", "echo hi > out.txt", "sort -o out.txt in.txt"):
                assert reg.resolve_intent_effect("run_command", {"command": command}) \
                    is ToolIntentEffect.EXTERNAL, command
            for command in ("find . -type f | tee files.txt", "cat package.json | sh"):
                assert reg.resolve_intent_effect("run_command", {"command": command}) \
                    is ToolIntentEffect.EXTERNAL, command
        finally:
            host.cleanup()


@check
def navigation_tier_authorizes_a_reversible_switch_by_intent_not_exact_target():
    # change_workspace is reversible: navigation INTENT (a workspace.navigate grant) authorizes the switch to
    # ANY resolved path, so the spoken target ("loom") need not basename-match the real dir ("loom-app"). But
    # a turn without navigation intent still asks, and nav intent never leaks into a destructive effect.
    from sliceagent import discourse
    ext = {"change_workspace", "edit_file", "run_command"}
    resolver = lambda name, args: (ToolIntentEffect.EXTERNAL if name in ext else ToolIntentEffect.OBSERVE)

    def gate(request, pending=None):
        contract = analyze_turn(request, pending_proposal=pending)
        return TurnAuthorityHook(lambda: contract, resolver)

    def switches(g, path):
        return g.authorize_tool("change_workspace", {"path": path}).allow

    explicit = gate("go to loom project on my desktop")
    assert switches(explicit, "/Users/x/Desktop/loom-app")      # spoken "loom" ≠ dir "loom-app": still allowed
    assert switches(explicit, "/Users/x/work/frontend")         # any path — the effect is undoable
    prop = discourse.extract_pending_proposal("loom-app and loom-engine. Which one would you like to go to?")
    assert switches(gate("loom app", prop), "/Users/x/Desktop/loom-app")
    offer = discourse.extract_pending_proposal("Do you want me to switch to loom-app?")
    assert switches(gate("yes", offer), "~/Desktop/loom-app")

    # Safety: no navigation intent → still asks; nav intent does not authorize a destructive effect.
    assert not switches(gate("what folders are on my desktop?"), "/Users/x/Desktop/loom-app")
    assert not switches(gate("fix the parser bug"), "/etc")
    assert not explicit.authorize_tool("edit_file", {"path": "/Users/x/Desktop/loom-app/app.py"}).allow
    assert not explicit.authorize_tool("run_command", {"command": "rm -rf x"}).allow

    # In-turn ask_user stamp: an ambiguous opening turn is `uncertain` with no grant, so the switch is blocked
    # until the user answers the agent's menu — cli._stamp_navigation_authority then adds a workspace.navigate
    # grant, which the tier honours even under uncertain authority. Enables ONLY navigation, not edits.
    from dataclasses import replace
    from sliceagent.intent import EffectGrant
    ambiguous = analyze_turn("loom app")           # bare noun, no proposal → uncertain, no grant
    holder = type("H", (), {"c": ambiguous})()
    g = TurnAuthorityHook(lambda: holder.c, resolver)
    assert not g.authorize_tool("change_workspace", {"path": "/Users/x/Desktop/loom-app"}).allow
    holder.c = replace(ambiguous, effect_grants=(
        *ambiguous.effect_grants, EffectGrant("workspace.navigate", ("change_workspace",))))
    assert g.authorize_tool("change_workspace", {"path": "/Users/x/Desktop/loom-app"}).allow
    assert not g.authorize_tool("edit_file", {"path": "/Users/x/Desktop/loom-app/app.py"}).allow


@check
def named_workspace_resolution_probe_is_observation_not_an_external_effect():
    """Pin the exact safe path probe from the navigation failure transcript.

    This does *not* bless arbitrary shell chaining or redirection.  Both branches are allowlisted ``ls``
    readers and the only redirects discard stderr to ``/dev/null``; resolving a named workspace must remain
    available even while effect authority is uncertain.
    """
    command = (
        "ls -d /Users/tongtao/Desktop/hunter 2>/dev/null || "
        "ls -d /Users/tongtao/code/hunter 2>/dev/null"
    )
    with tempfile.TemporaryDirectory(prefix="turn-authority-nav-") as root:
        host = LocalToolHost(root)
        try:
            effect = host.registry.resolve_intent_effect("run_command", {"command": command})
            assert effect is ToolIntentEffect.OBSERVE, effect
            decision = _hook("uncertain", host.registry).authorize_tool(
                "run_command", {"command": command},
            )
            assert decision.allow, decision
            for unsafe in (
                "ls -d /tmp/hunter > discovered.txt",
                "ls -d /tmp/hunter || touch changed.txt",
                "ls -d /tmp/hunter 2>/tmp/discovery-errors.log",
            ):
                assert host.registry.resolve_intent_effect(
                    "run_command", {"command": unsafe},
                ) is not ToolIntentEffect.OBSERVE, unsafe
        finally:
            host.cleanup()


@check
def unknown_extensions_stay_unknown_even_when_scheduler_purity_says_read():
    reg = ToolRegistry()
    reg.register(_entry("mcp__db__lookup", source="mcp", purity=ToolPurity.PURE_READ))
    assert reg.resolve_intent_effect("mcp__db__lookup", {}) is ToolIntentEffect.UNKNOWN
    assert reg.resolve_intent_effect("not_registered", {}) is ToolIntentEffect.UNKNOWN

    # Extensions can make an explicit semantic declaration; it is independent of ToolPurity.
    reg.register(_entry("plugin_lookup", source="plugin:x", purity=ToolPurity.UNKNOWN,
                        intent_effect=ToolIntentEffect.OBSERVE))
    assert reg.resolve_intent_effect("plugin_lookup", {}) is ToolIntentEffect.OBSERVE


@check
def none_and_uncertain_allow_only_observation_and_dialogue():
    with tempfile.TemporaryDirectory(prefix="turn-authority-") as root:
        host = LocalToolHost(root)
        try:
            reg = host.registry
            for authority in ("none", "uncertain"):
                gate = _hook(authority, reg)
                for name in reg.names():
                    args = {"command": "ls -la"} if name == "run_command" else {}
                    effect = reg.resolve_intent_effect(name, args)
                    decision = gate.authorize_tool(name, args)
                    expected_allow = (
                        effect in (ToolIntentEffect.OBSERVE, ToolIntentEffect.DIALOGUE)
                        or name == "reconcile_execution"
                    )
                    assert decision.allow is expected_allow, (authority, name, effect, decision)
                    if not expected_allow:
                        assert decision.counts_as_stuck is False
                        assert "turn_authority_missing" in decision.reason
                        assert f"effect={effect.value}" in decision.reason

                # The same tool name can become external when its exact command is not proven read-only.
                denied = gate.authorize_tool("run_command", {"command": "touch changed.txt"})
                assert not denied.allow and "effect=EXTERNAL" in denied.reason
        finally:
            host.cleanup()


@check
def every_retry_and_sibling_effect_is_blocked_for_the_whole_turn():
    reg = ToolRegistry()
    reg.register(_entry("read_file"))
    reg.register(_entry("edit_file"))
    reg.register(_entry("update_plan"))
    gate = _hook("none", reg)

    calls = [
        ("edit_file", {"path": "a"}),
        ("edit_file", {"path": "a"}),          # exact retry
        ("update_plan", {"steps": []}),        # sibling task-state mutation
        ("edit_file", {"path": "b"}),          # different retry after other calls
    ]
    decisions = [gate.authorize_tool(name, args) for name, args in calls]
    assert all(not decision.allow for decision in decisions), decisions
    assert all(decision.counts_as_stuck is False for decision in decisions)
    assert gate.authorize_tool("read_file", {"path": "a"}).allow, "reads remain available after blocks"


class _AuthoritySpinLLM:
    def __init__(self, target):
        self.target = target
        self.calls = 0

    def complete(self, _messages, _schemas):
        self.calls += 1
        return AssistantMessage(
            content="Switching now.",
            tool_calls=[ToolCall(f"switch-{self.calls}", "change_workspace", {"path": self.target})],
            usage={"prompt_tokens": 1, "completion_tokens": 1}, finish_reason="tool_calls",
        )


@check
def repeated_identical_authority_denial_parks_before_the_step_ceiling():
    """One denial is model feedback; repeating that exact denial returns control immediately."""
    with tempfile.TemporaryDirectory(prefix="turn-authority-spin-") as root:
        target = os.path.join(root, "target")
        os.mkdir(target)
        host = LocalToolHost(root)
        host.on_workspace_switch = lambda _path: (_ for _ in ()).throw(
            AssertionError("denied navigation must never reach the workspace switch handler")
        )
        llm = _AuthoritySpinLLM(target)
        events = []
        try:
            result = run_turn(
                build_slice=lambda: [{"role": "user", "content": "yes"}],
                llm=llm, tools=host, dispatch=events.append,
                hooks=CompositeHooks(_hook("uncertain", host.registry)),
                max_steps=20,
            )
        finally:
            host.cleanup()
        denied = [
            event for event in events
            if isinstance(event, ToolResult) and "turn_authority_missing" in event.output
        ]
        interrupted = [event for event in events if isinstance(event, TurnInterrupted)]
        assert result.stop_reason == "blocked", result
        # A final ask/summary-only closeout may make one extra model call, but it cannot expose or execute the
        # denied tool again.  The execution trajectory itself ends on the first identical retry.
        assert result.steps == 2 and llm.calls <= 3, (result, llm.calls)
        assert len(denied) == 2, denied
        assert len(interrupted) == 1, interrupted
        message = interrupted[0].message or ""
        lowered = message.lower()
        # This copy is rendered directly in the terminal.  It must say what was stopped and why, without
        # leaking model-facing anti-spin instructions or names of internal control mechanisms.
        assert "change_workspace" in message, message
        assert "current request" in lowered and "does not authorize" in lowered, message
        assert "loop guard" not in lowered, message
        assert "ask_user" not in lowered, message


@check
def ask_user_and_history_reads_are_available_without_effect_authority():
    reg = ToolRegistry()
    reg.register(_entry("ask_user"))
    reg.register(_entry("search_history"))
    gate = _hook("none", reg)
    assert reg.resolve_intent_effect("ask_user", {}) is ToolIntentEffect.DIALOGUE
    assert reg.resolve_intent_effect("search_history", {}) is ToolIntentEffect.OBSERVE
    assert gate.authorize_tool("ask_user", {"question": "Proceed?"}).allow
    assert gate.authorize_tool("search_history", {"query": "finding #2"}).allow


@check
def kernel_recovery_handshake_does_not_require_new_user_effect_authority():
    reg = ToolRegistry()
    reg.register(_entry("reconcile_execution", intent_effect=ToolIntentEffect.TASK_STATE))
    gate = _hook("none", reg)
    assert reg.resolve_intent_effect("reconcile_execution", {}) is ToolIntentEffect.TASK_STATE
    assert gate.authorize_tool("reconcile_execution", {}).allow, \
        "ReconciliationHook, not conversational intent, owns proof for the recovery handshake"


@check
def untyped_continuation_fails_closed_and_typed_action_is_exact():
    reg = ToolRegistry()
    reg.register(_entry("read_file", intent_effect=ToolIntentEffect.OBSERVE))
    reg.register(_entry("change_workspace", intent_effect=ToolIntentEffect.EXTERNAL))
    untyped = TurnAuthorityHook(
        lambda: {
            "effect_authority": "continuation",
            "referents": ({"kind": "pending_proposal", "text": "Would you like me to fix it?"},),
        },
        reg.resolve_intent_effect,
    )
    assert untyped.authorize_tool("read_file", {}).allow
    denied = untyped.authorize_tool("change_workspace", {"path": "/tmp/hunter"})
    assert not denied.allow and "no typed effect grant" in denied.reason

    target = "/tmp/hunter"
    typed = analyze_turn("yes", pending_proposal={
        "action": {"tool": "change_workspace", "args": {"path": target}},
    })
    gate = TurnAuthorityHook(lambda: typed, reg.resolve_intent_effect)
    assert gate.authorize_tool("change_workspace", {"path": target}).allow
    assert not gate.authorize_tool("change_workspace", {"path": "/tmp/other"}).allow


@check
def explicit_authority_is_scoped_to_operation_target_and_related_bookkeeping():
    reg = ToolRegistry()
    for name, effect in (
        ("read_file", ToolIntentEffect.OBSERVE),
        ("edit_file", ToolIntentEffect.EXTERNAL),
        ("change_workspace", ToolIntentEffect.EXTERNAL),
        ("run_command", ToolIntentEffect.EXTERNAL),
        ("update_plan", ToolIntentEffect.TASK_STATE),
        ("world_set", ToolIntentEffect.TASK_STATE),
    ):
        reg.register(_entry(name, intent_effect=effect))
    admission = analyze_turn("edit README")
    gate = TurnAuthorityHook(lambda: admission, reg.resolve_intent_effect)
    assert gate.authorize_tool("edit_file", {"path": "README"}).allow
    assert gate.authorize_tool("edit_file", {"path": "README.md"}).allow
    assert not gate.authorize_tool("edit_file", {"path": "src/app.py"}).allow
    assert not gate.authorize_tool("change_workspace", {"path": "/tmp/other"}).allow
    assert gate.authorize_tool("update_plan", {"steps": []}).allow
    assert not gate.authorize_tool("world_set", {"key": "unrelated"}).allow
    assert gate.authorize_tool("run_command", {"command": "python -m pytest tests/test_readme.py"}).allow
    assert not gate.authorize_tool("run_command", {"command": "touch unrelated.txt"}).allow


@check
def unavailable_contract_or_effect_metadata_fails_closed_without_stuck_penalty():
    broken_contract = TurnAuthorityHook(
        lambda: (_ for _ in ()).throw(RuntimeError("gone")),
        lambda _name, _args: ToolIntentEffect.EXTERNAL,
    )
    decision = broken_contract.authorize_tool("edit_file", {})
    assert not decision.allow and "authority=uncertain" in decision.reason
    assert decision.counts_as_stuck is False

    broken_resolver = TurnAuthorityHook(
        lambda: {"effect_authority": "none"},
        lambda _name, _args: (_ for _ in ()).throw(RuntimeError("bad metadata")),
    )
    decision = broken_resolver.authorize_tool("extension", {})
    assert not decision.allow and "effect=UNKNOWN" in decision.reason
    assert decision.counts_as_stuck is False


@check
def turn_authority_and_permission_policy_remain_independent():
    reg = ToolRegistry()
    reg.register(_entry("read_file"))
    reg.register(_entry("edit_file"))

    permissive = PermissionHook(lambda _name, _args: ALLOW)
    hooks = CompositeHooks(_hook("none", reg), permissive)
    assert not hooks.authorize_tool("edit_file", {}).allow, "let-it-go cannot manufacture user authority"
    assert hooks.authorize_tool("read_file", {}).allow

    denying = PermissionHook(lambda _name, _args: ToolDecision(False, "ordinary policy denied"))
    hooks = CompositeHooks(TurnAuthorityHook(
        lambda: analyze_turn("edit README"), reg.resolve_intent_effect,
    ), denying)
    decision = hooks.authorize_tool("edit_file", {"path": "README"})
    assert not decision.allow and decision.reason == "ordinary policy denied", \
        "user authority must not bypass ordinary safety/permission policy"


@check
def sticky_per_prefix_approval_reuses_a_yes_for_the_same_operation_only():
    # Block review Move 1: 'always' remembers the OPERATION prefix, so `npm test` approved once auto-runs its
    # variants, but a different subcommand / destructive form still re-prompts (never blesses `rm -rf`).
    asks = iter(["always"])   # the user says 'always' exactly once, then no more prompts are answered
    h = PermissionHook(lambda _n, _a: ToolDecision(False, "confirm", ask=True),
                       on_ask=lambda _n, _a, _r: next(asks, "no"))

    def run(cmd):
        return h.authorize_tool("run_command", {"command": cmd}).allow
    assert run("npm test")                       # approved with 'always'
    assert run("npm test --coverage")            # same operation → sticks, no prompt
    assert run("npm test -w pkg/a")
    assert not run("npm run build")              # different subcommand → re-prompts (asks exhausted → no)
    assert not run("git push --force")           # unrelated + destructive
    assert not run("rm -rf build/")              # destructive never rides a sticky prefix


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
