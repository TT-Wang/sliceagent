"""Per-turn tool-call loop guardrail — the slice's hard anti-loop floor.
No model, no pytest. Run: python tests/test_guardrails.py

Covers plan sec 5 test_guardrails.py bullets:
- note-arg stripped from signature
- exact-failure block AFTER 3 failures (not on/before the 3rd)
- success clears the exact-failure streak
- idempotent no-progress block (same read-only result N times)
- mutating-tool repeated SUCCESS never blocks (only its repeated FAILURE counts)
- reset_for_turn clears counters
- guardrail_blocked_result starts with 'Error'
- before_call is a pure read (non-mutating of counters)
- unknown tool is non-idempotent
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.guardrails import (                          # noqa: E402
    GuardrailDecision,
    IDEMPOTENT_TOOL_NAMES,
    MUTATING_TOOL_NAMES,
    ToolCallGuardrail,
    ToolCallGuardrailConfig,
    ToolCallSignature,
    canonical_tool_args,
    guardrail_blocked_result,
    is_failing_output,
)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _fail_n(g, tool, args, n):
    """Record n failures for (tool, args)."""
    for _ in range(n):
        g.after_call(tool, args, "Error: boom")


# --- signature / canonicalization ---------------------------------------------

@check
def note_arg_stripped_from_signature():
    # the 'note' findings-arg rides on every call and changes turn-to-turn; stripping it
    # is what lets the guardrail SEE a loop (otherwise every signature would be unique).
    a = canonical_tool_args({"command": "ls", "note": "trying to list"})
    b = canonical_tool_args({"command": "ls", "note": "totally different commentary"})
    assert a == b, "note must be stripped so the canonical args are identical"
    assert "note" not in a and "trying" not in a
    # and the signatures collapse too
    sa = ToolCallSignature.from_call("run_command", {"command": "ls", "note": "x"})
    sb = ToolCallSignature.from_call("run_command", {"command": "ls", "note": "y"})
    assert sa == sb


@check
def different_real_args_distinct_signatures():
    # sanity: stripping note must NOT collapse genuinely different calls
    sa = ToolCallSignature.from_call("run_command", {"command": "ls", "note": "x"})
    sb = ToolCallSignature.from_call("run_command", {"command": "pwd", "note": "x"})
    assert sa != sb


# --- exact-failure block --------------------------------------------------------

@check
def exact_failure_block_after_3():
    g = ToolCallGuardrail()
    args = {"command": "git push"}
    # first three failures must NOT block (count reaches threshold only on the 3rd)
    for i in range(3):
        assert g.before_call("run_command", args).block is False, f"blocked too early at {i}"
        g.after_call("run_command", args, "Error: rejected")
    # now the count == 3 == exact_failure_block_after → the 4th attempt is blocked
    d = g.before_call("run_command", args)
    assert d.block is True
    assert d.code == "repeated_exact_failure"
    assert d.count == 3
    assert d.tool_name == "run_command"


@check
def exact_failure_respects_custom_threshold():
    g = ToolCallGuardrail(ToolCallGuardrailConfig(exact_failure_block_after=2))
    args = {"command": "x"}
    assert g.before_call("run_command", args).block is False
    g.after_call("run_command", args, "Error: a")
    assert g.before_call("run_command", args).block is False
    g.after_call("run_command", args, "Error: b")
    assert g.before_call("run_command", args).block is True


@check
def success_clears_the_streak():
    g = ToolCallGuardrail()
    args = {"command": "make"}
    _fail_n(g, "run_command", args, 3)
    assert g.before_call("run_command", args).block is True   # at the floor
    # a SUCCESS for the same signature wipes the exact-failure streak
    g.after_call("run_command", args, "ok, built")
    d = g.before_call("run_command", args)
    assert d.block is False, "success must clear the exact-failure streak"
    assert d.code == "allow"


# --- idempotent no-progress block ----------------------------------------------

@check
def idempotent_no_progress_block():
    g = ToolCallGuardrail()
    args = {"path": "a.py"}
    # same read-only result returned 3 times → no progress
    for i in range(3):
        assert g.before_call("read_file", args).block is False, f"blocked too early at {i}"
        g.after_call("read_file", args, "same file body")
    d = g.before_call("read_file", args)
    assert d.block is True
    assert d.code == "idempotent_no_progress"
    assert d.count == 3
    assert d.tool_name == "read_file"


@check
def idempotent_changing_result_does_not_block():
    # if the read-only result CHANGES, that's progress — the repeat counter resets each time
    g = ToolCallGuardrail()
    args = {"path": "log.txt"}
    for i in range(6):
        g.after_call("read_file", args, f"body version {i}")
        assert g.before_call("read_file", args).block is False, f"changing result blocked at {i}"


@check
def idempotent_is_known_set():
    # the membership the guardrail relies on
    assert "read_file" in IDEMPOTENT_TOOL_NAMES
    assert "list_files" in IDEMPOTENT_TOOL_NAMES
    assert "recall_history" in IDEMPOTENT_TOOL_NAMES


# --- mutating tools never trip the no-progress path ----------------------------

@check
def mutating_tool_repeated_success_never_blocks_idempotent_path():
    # A mutating tool's repeated SUCCESS never trips the IDEMPOTENT no-progress path (that path is
    # for read-only tools only). I3 NOTE: it CAN now trip the tool-agnostic RESULT axis if the result
    # is byte-identical — so use realistic DISTINCT results here (a real edit changes the file and
    # returns a different summary). Distinct results = genuine progress = never blocked.
    g = ToolCallGuardrail()
    for i in range(10):
        args = {"path": f"f{i}.py", "old": "x", "new": f"y{i}"}
        assert g.before_call("edit_file", args).block is False, f"mutating success blocked at {i}"
        g.after_call("edit_file", args, f"edited f{i}.py ({i} lines changed)")
    assert "edit_file" in MUTATING_TOOL_NAMES


@check
def mutating_tool_still_blocks_on_repeated_failure():
    # mutating tools DO still get the exact-failure floor (only the no-progress path is exempt)
    g = ToolCallGuardrail()
    args = {"path": "a.py", "old": "x", "new": "y"}
    _fail_n(g, "edit_file", args, 3)
    assert g.before_call("edit_file", args).block is True


# --- unknown tool --------------------------------------------------------------

@check
def unknown_tool_non_idempotent():
    # a tool in neither set is treated as non-idempotent → never trips the IDEMPOTENT no-progress
    # path. I3 NOTE: distinct results = progress (the tool-agnostic RESULT axis keys on the output),
    # so changing-result calls never block even for an unknown tool.
    g = ToolCallGuardrail()
    assert "mystery_tool" not in IDEMPOTENT_TOOL_NAMES
    assert "mystery_tool" not in MUTATING_TOOL_NAMES
    for i in range(8):
        args = {"q": f"hi {i}"}
        assert g.before_call("mystery_tool", args).block is False, f"unknown tool blocked at {i}"
        g.after_call("mystery_tool", args, f"result {i}")
    # but unknown tools still hit the exact-failure floor
    args = {"q": "hi"}
    _fail_n(g, "mystery_tool", args, 3)
    assert g.before_call("mystery_tool", args).block is True


# --- before_call purity --------------------------------------------------------

@check
def before_call_does_not_mutate_counters():
    # before_call is a pure read of the counters — calling it repeatedly must not itself
    # advance any count (only after_call counts).
    g = ToolCallGuardrail()
    args = {"path": "a.py"}
    g.after_call("read_file", args, "body")          # one observed result, repeat == 1
    for _ in range(20):
        assert g.before_call("read_file", args).block is False
    # still only one observed repeat → not at the no-progress floor
    g.after_call("read_file", args, "body")          # repeat == 2
    g.after_call("read_file", args, "body")          # repeat == 3
    assert g.before_call("read_file", args).block is True


# --- reset_for_turn ------------------------------------------------------------

@check
def reset_for_turn_clears_counters():
    g = ToolCallGuardrail()
    args = {"command": "x"}
    _fail_n(g, "run_command", args, 3)
    assert g.before_call("run_command", args).block is True
    g.reset_for_turn()
    d = g.before_call("run_command", args)
    assert d.block is False, "reset_for_turn must drop the prior turn's loop counts"
    assert d.code == "allow"
    # also clears the no-progress streak
    for _ in range(3):
        g.after_call("read_file", {"path": "a"}, "same")
    assert g.before_call("read_file", {"path": "a"}).block is True
    g.reset_for_turn()
    assert g.before_call("read_file", {"path": "a"}).block is False


# --- blocked-result wording ----------------------------------------------------

@check
def guardrail_blocked_result_starts_with_error():
    # the synthetic tool result must start with 'Error' so slice.record_action tallies it
    # as a failure (keeps the block visible in the REPEATED/FAILING tier).
    d = GuardrailDecision(block=True, code="repeated_exact_failure", message="do not retry", count=3)
    out = guardrail_blocked_result(d)
    assert out.startswith("Error")
    assert is_failing_output(out) is True
    assert "do not retry" in out


@check
def blocked_decision_message_is_action_oriented():
    # the real decision message (the one that lands in s.last_error) should tell the model
    # what to do INSTEAD, not just that it was blocked.
    g = ToolCallGuardrail()
    args = {"command": "x"}
    _fail_n(g, "run_command", args, 3)
    d = g.before_call("run_command", args)
    assert d.block is True and d.message
    assert "Do NOT retry" in d.message or "do NOT retry" in d.message.lower()


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
