"""Request salience + intent-aware footer, asserted at the BUILD level (the request and the NOW footer
render in build(), OUTSIDE the <context> fence — not inside render_slice). No model, no pytest.
Run: PYTHONPATH=src python tests/test_bugfix_current_request.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.pfc import Slice  # noqa: E402
from memagent.seed import make_build_slice  # noqa: E402
from memagent.retriever import NullRetriever         # noqa: E402
from memagent.memory import NullMemory               # noqa: E402
import memagent.regions as regions                    # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn

_ROOT = tempfile.mkdtemp(prefix="cr-root-")


class _Tools:
    def schemas(self): return []
    def accesses(self, n, a): return []
    def run(self, n, a): return ""
    def root(self): return _ROOT
    def read_text(self, p): raise FileNotFoundError(p)


def _user(goal):
    s = Slice()
    if goal:
        s.reset(goal)
    return make_build_slice(s, _Tools(), NullRetriever(), NullMemory(), goal)()[1]["content"]


@check
def request_leads_and_is_outside_the_fence():
    out = _user("take a look at how config parsing works and tell me whether it handles bad input safely")
    assert out.startswith("# CURRENT REQUEST"), "the live request leads the user message (primacy)"
    assert "handles bad input safely" in out
    # both copies live OUTSIDE the reference fence
    close = out.index("</context>")
    assert out.index("# CURRENT REQUEST") < out.index("<context>"), "primacy before the fence"
    assert out.rindex("# CURRENT REQUEST") > close, "recency after the fence"


@check
def now_footer_is_intent_aware_and_outermost():
    out = _user("explain the retry logic")
    assert "# NOW" in out and out.index("# NOW") > out.index("</context>"), "NOW is outside the fence"
    assert "QUESTION" in out and "answer it directly" in out, "footer offers converse, not only act/edit"
    assert "CURRENT REQUEST" in out, "footer points back at the request"


@check
def no_goal_suppresses_request_header():
    out = _user("")    # fresh slice, empty goal
    assert "# CURRENT REQUEST" not in out, "no goal → no request header (no empty tier)"
    assert out.startswith("<context>"), "envelope still wraps the slice"


@check
def findings_header_not_over_hedged_but_claim_tag_kept():
    # the blanket "a note is NOT proof" distrust is gone; the per-note claim hedge (anti-ratchet) stays
    src = {"did the thing": "claim"}
    assert "UNVERIFIED claim" in regions.render_findings(["did the thing"], src), "claims stay marked unverified"
    assert "UNVERIFIED" not in regions.render_findings(["did the thing"], {"did the thing": "observed"})


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
