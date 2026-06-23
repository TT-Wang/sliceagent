"""Regression test for the slice-salience fix: the live user request must appear as a first-class tier at
the SALIENT TAIL of the user slice (not only buried in the cacheable system prefix), and the NOW footer must
be intent-aware (converse-or-act). No model, no pytest.
Run: PYTHONPATH=src python tests/test_bugfix_current_request.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.slice import Slice, render_slice  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def current_request_renders_in_salient_tail():
    s = Slice()
    s.reset("take a look at how config parsing works and tell me whether it handles bad input safely")
    out = render_slice(s, "(no files opened yet)")
    assert "# CURRENT REQUEST" in out, "the live request must be a first-class user-slice tier"
    assert "handles bad input safely" in out, "the goal text must appear in the USER slice (not just prefix)"
    # it must be in the TAIL (after OPEN FILES near the top), i.e. salient
    assert out.index("# CURRENT REQUEST") > out.index("# OPEN FILES"), "request belongs in the salient tail"


@check
def now_footer_is_intent_aware():
    s = Slice(); s.reset("explain the retry logic")
    out = render_slice(s, "(no files)")
    assert "# NOW" in out
    assert "QUESTION" in out and "answer it directly" in out, "footer must offer converse, not only act/edit"
    assert "CURRENT REQUEST" in out  # footer points back at the request


@check
def current_request_suppressed_when_no_goal():
    out = render_slice(Slice(), "(no files)")   # fresh slice, empty goal
    assert "# CURRENT REQUEST" not in out, "no goal → no request tier (no empty header)"


@check
def findings_header_not_over_hedged_but_claim_tag_kept():
    # the blanket "a note is NOT proof" distrust is gone; the per-note claim hedge (anti-ratchet) stays
    import memagent.regions as r
    s = Slice(); s.reset("x")
    s.findings = ["did the thing"]
    s.finding_source = {"did the thing": "claim"}
    body = r.render_findings(s.findings, s.finding_source)
    assert "UNVERIFIED claim" in body, "claim findings must still be marked unverified (anti-ratchet guard)"
    # an observed finding carries NO hedge
    s.finding_source = {"did the thing": "observed"}
    assert "UNVERIFIED" not in r.render_findings(s.findings, s.finding_source)


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
