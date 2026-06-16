"""RECENT CONVERSATION tier — short-range continuity (the always-on half of the "access larger
context" design; the on-demand half is recall_history). No model, no pytest.
Run: PYTHONPATH=src python tests/test_conversation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import AssistantText                                  # noqa: E402
from memagent.slice import (MAX_CONVERSATION, Slice, record_user,          # noqa: E402
                            render_conversation, render_slice, slice_sink)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def record_user_counts_turns_and_bounds_the_ring():
    s = Slice(); s.reset("t")
    for i in range(10):
        record_user(s, f"message {i}")
    assert s.turns == 10, s.turns
    assert len(s.conversation) == MAX_CONVERSATION, len(s.conversation)
    assert s.conversation[-1]["user"] == "message 9"


@check
def sink_fills_the_assistant_side_of_the_current_exchange():
    s = Slice(); s.reset("t")
    record_user(s, "do the thing")
    sink = slice_sink(s)
    sink(AssistantText("I did the thing and it works"))
    assert s.conversation[-1]["assistant"] == "I did the thing and it works", s.conversation


@check
def render_shows_prior_exchanges_excluding_the_in_progress_one():
    s = Slice(); s.reset("t")
    record_user(s, "first request")
    slice_sink(s)(AssistantText("first reply"))
    record_user(s, "second request")          # in-progress (its user msg is the current task)
    out = render_conversation(s)
    assert "user: first request" in out and "first reply" in out, out
    assert "second request" not in out, "the in-progress exchange must be excluded"


@check
def older_turns_get_a_recall_pointer():
    s = Slice(); s.reset("t")
    for i in range(6):                         # 6 turns, ring holds last 4
        record_user(s, f"req {i}")
        slice_sink(s)(AssistantText(f"reply {i}"))
    out = render_conversation(s)
    assert "earlier turn(s)" in out and "recall_history" in out, out


@check
def fresh_slice_renders_no_conversation_tier():
    s = Slice(); s.reset("t")
    assert render_conversation(s) == ""
    assert "RECENT CONVERSATION" not in render_slice(s, "(open files)")


@check
def render_slice_includes_the_tier_when_there_is_prior_dialogue():
    s = Slice(); s.reset("t")
    record_user(s, "earlier ask")
    slice_sink(s)(AssistantText("earlier answer"))
    record_user(s, "current ask")
    assert "RECENT CONVERSATION" in render_slice(s, "(open files)")


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
