"""Item 14 — typed findings + informative tool-result one-liners. No model, no pytest.
Run: python tests/test_finding_types.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.finding_types import (  # noqa: E402
    DECISION, FILE_TOUCHED, NOTE, RESOLVED, RULED_OUT, badge, classify_finding,
)
from memagent.tool_summary import summarize_tool_result  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# --- 14a: typed findings -------------------------------------------------

@check
def classify_ruled_out_beats_resolved():
    # "doesn't work" must type as RULED_OUT even though there's a fix-y context
    assert classify_finding("tried the cache approach, it doesn't work",
                            edited=True, had_error=True, resolved=True) == RULED_OUT


@check
def classify_decision():
    assert classify_finding("decided to use a bounded queue instead of a list") == DECISION


@check
def classify_resolved():
    assert classify_finding("fixed: the bug was an off-by-one") == RESOLVED
    # structural signal alone (error hit AND cleared) resolves an ambiguous note
    assert classify_finding("updated the loop", had_error=True, resolved=True) == RESOLVED


@check
def classify_file_touched_and_note_fallback():
    assert classify_finding("touched the config", edited=True) == FILE_TOUCHED
    assert classify_finding("just an observation") == NOTE


@check
def badge_blank_for_note():
    assert badge(NOTE) == "" and badge("") == ""
    assert badge(DECISION) == "[decision] "


# --- 14b: tool-result one-liners -----------------------------------------

@check
def summarize_run_command_extracts_exit():
    s = summarize_tool_result("run_command", {"command": "pytest -q"},
                              "....\nExit code: 0\n", failing=False)
    assert s.startswith("[run_command] `pytest -q` -> exit 0")
    assert "lines" in s


@check
def summarize_failing_marks_outcome():
    s = summarize_tool_result("run_command", {"command": "make"}, "boom", failing=True)
    assert "✗" in s


@check
def summarize_read_and_write():
    r = summarize_tool_result("read_file", {"path": "a.py"}, "x" * 100)
    assert r.startswith("[read_file] a.py") and "chars" in r
    w = summarize_tool_result("write_file", {"path": "b.py", "content": "1\n2\n3"}, "ok")
    assert w.startswith("[write_file] wrote b.py") and "3 lines" in w


@check
def summarize_skill_and_generic():
    assert summarize_tool_result("skill", {"name": "deploy"}, "body").startswith("[skill] loaded deploy")
    g = summarize_tool_result("weird_tool", {"foo": "bar"}, "out")
    assert g.startswith("[weird_tool]") and "foo=bar" in g


@check
def summarize_never_raises_on_bad_args():
    # args not a dict → falls through, no exception
    assert summarize_tool_result("read_file", None, "out")


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
