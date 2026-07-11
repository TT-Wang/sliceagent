"""2B / SOTA within-turn transcript construction. Validates the three shipped changes at the BUILD level
(make_build_slice → [system, user]):
  (a) goal/# TASK removed from the system message → system is byte-stable regardless of goal (cache stays warm)
  (b) the verbatim request anchors the user message exactly once at recency (tail)
  (c) the slice is fenced in a <context> envelope
No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_2b.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.pfc import Slice  # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402
from sliceagent.retriever import NullRetriever         # noqa: E402
from sliceagent.memory import NullMemory               # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# A CLEAN, FIXED root so repo_map / workspace_facts are stable across builds — the byte-stable-system
# check must isolate the GOAL as the only variable (the live shared /tmp mutates under other processes).
_ROOT = tempfile.mkdtemp(prefix="2b-root-")


class _Tools:
    def schemas(self): return []
    def accesses(self, n, a): return []
    def run(self, n, a): return ""
    def root(self): return _ROOT
    def read_text(self, p): raise FileNotFoundError(p)


def _build(goal):
    s = Slice(); s.reset(goal)
    return make_build_slice(s, _Tools(), NullRetriever(), NullMemory(), s.goal)()


@check
def goal_removed_from_system_message():  # (a)
    msgs = _build("explain how config parsing handles bad input")
    system = msgs[0]["content"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "explain how config parsing handles bad input" not in system, "goal must NOT sit in the system message"
    assert "# TASK" not in system, "the dangling # TASK header should be gone"


@check
def system_is_byte_stable_across_goals():  # (a) — the cache-leak fix, the whole point
    sysA = _build("goal ALPHA one")[0]["content"]
    sysB = _build("goal BETA totally different wording")[0]["content"]
    assert sysA == sysB, "system message must be byte-identical regardless of goal (prompt-cache stays warm)"


@check
def request_anchors_once_at_recency():  # (b)
    user = _build("add a --json flag")[1]["content"]
    assert not user.startswith("# CURRENT REQUEST"), "request must not be duplicated at primacy"
    assert user.count("add a --json flag") == 1, "request must appear exactly once"


@check
def slice_fenced_in_context():  # (c)
    user = _build("do the thing")[1]["content"]
    assert "<context>" in user and "</context>" in user, "slice must be fenced"
    assert user.index("# CURRENT REQUEST") > user.index("</context>"), "request follows the fence"


@check
def request_and_now_render_OUTSIDE_the_fence():  # review fix A/C — instruction must not be 'context'
    user = _build("do the thing")[1]["content"]
    close = user.index("</context>")
    # the RECENCY request + the NOW footer come AFTER the fence closes (not inside the reference envelope)
    assert user.index("# CURRENT REQUEST") > close, "recency request must be OUTSIDE the fence"
    assert user.index("# NOW") > close, "the NOW instruction must be OUTSIDE the fence"
    assert user.rstrip().endswith("make NO tool call."), "NOW is the OUTERMOST tail"


@check
def empty_goal_suppresses_request():  # safety: a fresh slice with no goal shouldn't emit an empty header
    s = Slice()  # no reset → goal == ""
    user = make_build_slice(s, _Tools(), NullRetriever(), NullMemory(), "")()[1]["content"]
    assert "# CURRENT REQUEST" not in user, "no goal → no request header"
    assert user.startswith("<context>"), "envelope still wraps the slice"


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
