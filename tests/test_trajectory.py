"""I3 — BOUNDED TRAJECTORY CHANNEL. No model, no pytest.
Run: PYTHONPATH=src python tests/test_trajectory.py

Covers the Invariant-3 acceptance:
- RESULT-hash no-progress axis trips across ANY tools incl. shell (distinct command text, same output)
- the trajectory ring is BOUNDED (cap) — never a transcript
- the no-edit-progress axis trips after M mutating attempts with no edit landing
- op_kind is a coarse, tool-agnostic taxonomy
- OPEN USER REPORT: captured from a failure-report follow-up, rendered, and SURVIVES continue_topic
- the anti-loop epoch (action_log) is DEMOTED (failing=False, counts kept), not cleared, on a directive
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.guardrails import (                          # noqa: E402
    ToolCallGuardrail,
    ToolCallGuardrailConfig,
    op_kind,
)
from memagent.memory import NullMemory                     # noqa: E402
from memagent.session import Session                       # noqa: E402
from memagent.slice import (                               # noqa: E402
    Slice,
    capture_user_report,
    is_user_report,
    render_slice,
)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# --- RESULT-hash no-progress axis (GR1/GR3) -----------------------------------

@check
def result_hash_loop_trips_across_distinct_shell_commands():
    # the live failure: ~13 re-inspections of the same dir via DISTINCT command text — each a unique
    # arg signature at count 1, so the exact-(tool,args) axis NEVER fired. The RESULT axis keys on the
    # OUTPUT, so distinct commands returning the same listing collapse to one no-progress signature.
    g = ToolCallGuardrail()  # default result_repeat_block_after = 4
    same_output = "index.html\nstyle.css\nscript.js"
    cmds = ["ls agent_test", "ls -la agent_test", "ls -1 agent_test", "ls ./agent_test"]
    for i, c in enumerate(cmds):
        # each is a DIFFERENT (tool,args) signature → exact axis can't see the loop
        assert g.before_call("run_command", {"command": c}).block is False, f"blocked too early at {i}"
        g.after_call("run_command", {"command": c}, same_output)
    # 4 identical results across 4 distinct commands → the result axis trips the NEXT call (any tool)
    d = g.before_call("run_command", {"command": "ls -R agent_test"})
    assert d.block is True, "result-hash loop must trip even with all-distinct command text"
    assert d.code == "result_no_progress"
    assert d.count == 4
    # and it is tool-agnostic: even a different tool entirely is blocked while the loop stands
    assert g.before_call("execute_code", {"code": "import os; print(os.listdir('agent_test'))"}).block is True


@check
def result_hash_loop_spans_execute_code_and_run_command():
    # the same REAL OUTPUT produced via run_command AND execute_code is still ONE no-progress signal.
    # (Uses a non-empty output: an information-FREE "(produced no output)" sentinel is intentionally NOT a
    # loop signal — distinct successful silent commands legitimately all return it; see guardrails after_call.)
    g = ToolCallGuardrail()
    out = "total 5\n-rw-r--r-- 1 a b 0 file.txt"
    g.after_call("run_command", {"command": "ls a"}, out)
    g.after_call("execute_code", {"code": "x"}, out)
    g.after_call("run_command", {"command": "ls -l a"}, out)
    assert g.before_call("run_command", {"command": "ls a/"}).block is False  # only 3 so far
    g.after_call("run_command", {"command": "ls a/"}, out)                    # 4th identical
    assert g.before_call("execute_code", {"code": "y"}).block is True


@check
def distinct_silent_successful_commands_are_not_a_loop():
    # R12: distinct successful commands that each produce the no-output sentinel must NOT be hard-blocked
    g = ToolCallGuardrail()
    out = "(command produced no output)"
    for cmd in ("mkdir build", "touch a.txt", "cp a.txt b.txt", "git add -A", "chmod +x run.sh"):
        assert g.before_call("run_command", {"command": cmd}).block is False, f"{cmd!r} wrongly blocked"
        g.after_call("run_command", {"command": cmd}, out)


@check
def distinct_results_are_progress_never_block():
    # productive work returns CHANGING output — never a result-loop
    g = ToolCallGuardrail()
    for i in range(12):
        assert g.before_call("run_command", {"command": f"step {i}"}).block is False
        g.after_call("run_command", {"command": f"step {i}"}, f"output {i}")


@check
def result_axis_respects_custom_threshold():
    g = ToolCallGuardrail(ToolCallGuardrailConfig(result_repeat_block_after=2))
    g.after_call("run_command", {"command": "a"}, "same")
    assert g.before_call("run_command", {"command": "b"}).block is False
    g.after_call("run_command", {"command": "b"}, "same")     # 2nd identical
    assert g.before_call("run_command", {"command": "c"}).block is True


# --- bounded trajectory ring --------------------------------------------------

@check
def trajectory_ring_is_bounded():
    cap = 5
    g = ToolCallGuardrail(ToolCallGuardrailConfig(trajectory_ring_cap=cap, result_repeat_block_after=99))
    for i in range(200):
        g.after_call("run_command", {"command": f"c{i}"}, f"distinct {i}")
    assert len(g._trajectory) == cap, "ring must never exceed its cap (no transcript growth)"
    # counts derive from the LIVE ring only — an old repeated result that scrolled out stops counting
    g2 = ToolCallGuardrail(ToolCallGuardrailConfig(trajectory_ring_cap=3, result_repeat_block_after=99))
    g2.after_call("run_command", {"command": "x"}, "loop")
    g2.after_call("run_command", {"command": "y"}, "loop")
    for i in range(3):                                   # push 3 distinct → evict both "loop"s
        g2.after_call("run_command", {"command": f"z{i}"}, f"new {i}")
    _, count = g2._hottest_result()
    assert count == 1, "results scrolled out of the bounded ring must stop counting"


@check
def reset_for_turn_clears_trajectory():
    g = ToolCallGuardrail()
    for _ in range(4):
        g.after_call("run_command", {"command": "x"}, "same")
    assert g.before_call("run_command", {"command": "y"}).block is True
    g.reset_for_turn()
    assert g._trajectory == [] and g._result_counts == {}
    assert g.before_call("run_command", {"command": "y"}).block is False


# --- no-edit-progress axis ----------------------------------------------------

@check
def no_edit_mutations_axis_trips():
    # M mutating ATTEMPTS with no successful edit landing → "you are spinning" block on the next mutate
    g = ToolCallGuardrail(ToolCallGuardrailConfig(no_edit_mutations_before_warn=3,
                                                  result_repeat_block_after=99,
                                                  exact_failure_block_after=99))
    # 3 mutating attempts that don't land an edit (failing str_replace, distinct each time)
    for i in range(3):
        g.after_call("str_replace", {"path": "a.py", "old": f"x{i}"}, f"Error: no match {i}")
    d = g.before_call("str_replace", {"path": "a.py", "old": "x99"})
    assert d.block is True and d.code == "no_edit_progress"
    # a non-mutating tool is NOT blocked by this axis (read/answer always allowed)
    assert g.before_call("read_file", {"path": "a.py"}).block is False


@check
def successful_edit_resets_no_edit_streak():
    g = ToolCallGuardrail(ToolCallGuardrailConfig(no_edit_mutations_before_warn=3,
                                                  result_repeat_block_after=99,
                                                  exact_failure_block_after=99))
    g.after_call("str_replace", {"path": "a.py", "old": "x0"}, "Error: no match")
    g.after_call("str_replace", {"path": "a.py", "old": "x1"}, "Error: no match")
    g.after_call("edit_file", {"path": "a.py"}, "edited a.py (3 lines)")   # a real edit lands → reset
    assert g._mutations_since_edit == 0
    g.after_call("str_replace", {"path": "a.py", "old": "x2"}, "Error: no match")
    assert g.before_call("str_replace", {"path": "a.py", "old": "x3"}).block is False  # streak reset


# --- op_kind taxonomy ---------------------------------------------------------

@check
def op_kind_is_coarse_and_tool_agnostic():
    assert op_kind("edit_file") == op_kind("str_replace") == op_kind("append_to_file") == "edit"
    assert op_kind("run_command") == op_kind("execute_code") == "exec"
    assert op_kind("read_file") == "read"
    assert op_kind("list_files") == "list"
    assert op_kind("switch_topic") == op_kind("skill") == "meta"
    assert op_kind("some_mcp_tool") == "other"


# --- OPEN USER REPORT heuristic ----------------------------------------------

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
    # ordinary directives / conversation must NOT be captured as failure reports
    for msg in [
        "add a docstring to add()",
        "now also handle the empty case",
        "how do I start it?",
        "make it faster",
        "great, thanks!",
        "refactor the parser into two functions",
    ]:
        assert not is_user_report(msg), f"should NOT be a report: {msg!r}"


# --- OPEN USER REPORT: capture, render, and continue_topic survival ----------

@check
def capture_user_report_stores_and_bounds():
    s = Slice(); s.reset("build the snake game")
    assert capture_user_report(s, "it can't play at all") is True
    assert s.open_report == "it can't play at all"
    # a benign follow-up must NOT clear a still-open report
    assert capture_user_report(s, "thanks, looks good") is False
    assert s.open_report == "it can't play at all"
    # a NEWER report replaces the older one (most-recent wins; inherently bounded)
    assert capture_user_report(s, "now it crashes on launch") is True
    assert s.open_report == "now it crashes on launch"
    # bounded length
    long = "it doesn't work " + "x" * 5000
    capture_user_report(s, long)
    assert len(s.open_report) <= 280


@check
def open_user_report_renders_as_blocker():
    s = Slice(); s.reset("snake game")
    s.findings = ["**Done — built a working snake game**"]
    s.open_report = "it can't play at all"
    out = render_slice(s, "(no files opened yet)")
    assert "OPEN USER REPORT" in out
    assert "it can't play at all" in out
    # The blocker must OUTRANK the stale 'done' finding. Tier order is now cache-first (stable bulk
    # leads; volatile tiers in the tail), so authority is carried by RECENCY-SALIENCE: the report
    # renders AFTER the stale findings, in the most-salient tail right before NOW — so a 'done' note
    # can't outrank it. (Empirical no-regression validated by the head-to-head re-run.)
    assert out.index("OPEN USER REPORT") > out.index("**Done")
    assert out.index("OPEN USER REPORT") > out.index("# YOUR NOTES"), "report must be in the salient tail, after the notes bulk"


@check
def open_user_report_survives_continue_topic():
    # the F1 user-pushback bug: continue_topic cleared last_error so the user's "it's broken" became
    # only the goal string. Now the report rides forward as a durable blocker across the directive.
    sess = Session(NullMemory(), "s-test")
    sess.new_topic("build a news aggregator")
    sess.active().findings = ["built the aggregator"]
    sess.continue_topic("it can not play at all")          # a failure report → captured
    assert sess.active().open_report == "it can not play at all"
    # a SUBSEQUENT non-report directive does NOT retract the report (only verification / new topic does)
    sess.continue_topic("also add a --count flag")
    assert sess.active().open_report == "it can not play at all", "report must survive continue_topic"
    # a real topic reset (new_topic) DOES clear it (a genuinely fresh task)
    sess.new_topic("unrelated: write a haiku")
    assert sess.active().open_report == ""


# --- WS2: epoch DEMOTED, not cleared -----------------------------------------

@check
def continue_topic_demotes_epoch_not_clears():
    sess = Session(NullMemory(), "s-test")
    sess.new_topic("implement add()")
    s = sess.active()
    s.action_log = {"run_command `python news.py`": {"count": 2, "failing": True, "last": "boom"}}
    s.since_edit = 5; s.last_error = "boom"
    sess.continue_topic("now add a docstring")
    s2 = sess.active()
    # counts SURVIVE (a genuinely repeated command still trips REPEATED-with-no-progress)
    assert s2.action_log["run_command `python news.py`"]["count"] == 2
    # but the stale failing flag is dropped (new directive shouldn't carry a stale failure)
    assert s2.action_log["run_command `python news.py`"]["failing"] is False
    # since_edit is still a fresh epoch
    assert s2.since_edit == 0 and s2.last_error == ""


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
