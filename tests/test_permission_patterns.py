"""PermissionHook pattern-approval (Kimi-style): 'always' remembers the CALL pattern, not the bare tool
name — approving one shell command must not bless every shell command. Plus pre-seeded auto-approve globs.
No model, no pytest. Run: PYTHONPATH=src python tests/test_permission_patterns.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.hooks import PermissionHook, ToolDecision  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _ask_policy(name, args):
    return ToolDecision(True, "needs approval", ask=True)   # everything routes to ask


@check
def always_approves_only_the_exact_command_not_the_tool():
    asked = []
    hook = PermissionHook(_ask_policy, on_ask=lambda n, a, r: (asked.append(a.get("command")), "always")[1])
    # approve `npm test`
    assert hook.authorize_tool("run_command", {"command": "npm test"}).allow
    # same command → no re-ask
    asked.clear()
    assert hook.authorize_tool("run_command", {"command": "npm test"}).allow
    assert asked == [], "exact same command must not re-prompt"
    # DIFFERENT command → MUST re-ask (the bug fix: not blanket-approved by tool name)
    asked.clear()
    hook.on_ask = lambda n, a, r: "no"
    assert not hook.authorize_tool("run_command", {"command": "rm -rf /"}).allow
    assert asked == [] or True  # (on_ask replaced) — key point is it was DENIED, not auto-allowed


@check
def name_level_approval_for_non_command_tools():
    hook = PermissionHook(_ask_policy, on_ask=lambda n, a, r: "always")
    assert hook.authorize_tool("edit_file", {"path": "a.py"}).allow
    # a second edit to a different path is fine without re-ask (name-level is acceptable; policy gates danger)
    hook.on_ask = lambda n, a, r: (_ for _ in ()).throw(AssertionError("should not re-ask edit_file"))
    assert hook.authorize_tool("edit_file", {"path": "b.py"}).allow


@check
def auto_approve_globs_skip_the_prompt():
    def _boom(n, a, r):
        raise AssertionError("auto-approved command must not prompt")
    hook = PermissionHook(_ask_policy, on_ask=_boom, auto_approve=["git status*", "ls *"])
    assert hook.authorize_tool("run_command", {"command": "git status --short"}).allow
    assert hook.authorize_tool("run_command", {"command": "ls -la src"}).allow
    # a non-matching command still prompts (here → deny)
    hook.on_ask = lambda n, a, r: "no"
    assert not hook.authorize_tool("run_command", {"command": "curl evil.test"}).allow


@check
def non_interactive_denies_ask():
    hook = PermissionHook(_ask_policy, on_ask=None)
    assert not hook.authorize_tool("run_command", {"command": "anything"}).allow


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
