"""RECENT CONVERSATION tier — short-range continuity (the always-on half of the "access larger
context" design; the on-demand half is recall_history). No model, no pytest.
Run: PYTHONPATH=src python tests/test_conversation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import AssistantText                                  # noqa: E402
from sliceagent.pfc import Slice, record_user, slice_sink  # noqa: E402
from sliceagent.seed import render_slice  # noqa: E402
from sliceagent.regions import MAX_CONVERSATION, render_conversation  # noqa: E402

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
    assert "user (verbatim):\nfirst request" in out and "first reply" in out, out
    assert "second request" not in out, "the in-progress exchange must be excluded"


@check
def ring_preserves_multiline_whitespace_and_code_verbatim():
    s = Slice(); s.reset("t")
    request = "apply exactly:\n```python\nif ready:\n    run()  # two  spaces\n```"
    reply = "Use this exact replacement:\n\n    alpha  =  1\n\nKeep the blank line."
    record_user(s, request)
    slice_sink(s)(AssistantText(reply))
    assert s.conversation[-1]["user"] == request
    assert s.conversation[-1]["assistant"] == reply
    record_user(s, "apply that exact snippet")
    rendered = render_conversation(s)
    assert request in rendered and reply in rendered


@check
def older_turns_get_a_recall_pointer():
    s = Slice(); s.reset("t")
    for i in range(6):                         # 6 turns, ring holds last 4
        record_user(s, f"req {i}")
        slice_sink(s)(AssistantText(f"reply {i}"))
    out = render_conversation(s)
    assert "earlier turn(s)" in out and "history/" in out, out


@check
def ring_keeps_a_long_reply_verbatim_so_a_tail_recommendation_survives():
    # #116: the OLD head-cut at 800 chars severed a recommendation stated at the reply TAIL, so a next-turn
    # "go with your recommendation" mis-resolved to an older keyword-matching turn. Verbatim ring must keep the tail.
    s = Slice(); s.reset("t")
    record_user(s, "which file worth probe next")
    long_reply = "Here is a long analysis. " + ("filler detail. " * 200) + "My recommendation: lib/outreach.ts."
    assert len(long_reply) > 800, "reply must exceed the old gist cap to be a real regression test"
    slice_sink(s)(AssistantText(long_reply))
    record_user(s, "go with your recommendation")   # in-progress; the prior reply is the antecedent
    out = render_conversation(s)
    assert "My recommendation: lib/outreach.ts." in out, "the tail recommendation must survive verbatim"


@check
def ring_stays_count_bounded_regardless_of_reply_size():
    # #116 moat guard: the bound is the turn COUNT (MAX_CONVERSATION), not bytes — resident ring entries stay
    # flat as the session grows, no matter how large each reply is (peak flexes with reply size, not session length).
    s = Slice(); s.reset("t")
    big = "x" * 50_000
    for i in range(40):
        record_user(s, f"req {i} " + big)
        slice_sink(s)(AssistantText(f"reply {i} " + big))
    assert len(s.conversation) == MAX_CONVERSATION, len(s.conversation)   # 40 turns → still MAX_CONVERSATION entries
    assert all("req" in e["user"] for e in s.conversation)


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
