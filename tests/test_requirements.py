"""STANDING REQUIREMENTS contract — the model-curated replacement for the frozen task_spec. State lives
in the Slice, folded by slice_sink from require/requirement_done/drop_requirement events (the world_set
seam). EMPTY by default so a greeting never becomes a binding spec (the reported 'hi'→'who are you' bug);
carried across the seal + continue_topic; serialized in TaskState; wiped on reset.
No model, no pytest. Run: PYTHONPATH=src python tests/test_requirements.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.slice import Slice, make_build_slice, slice_sink  # noqa: E402
from memagent.events import ToolResult                          # noqa: E402
from memagent.tools import LocalToolHost                        # noqa: E402
from memagent.memory import NullMemory                          # noqa: E402
from memagent.session import Session                            # noqa: E402
from memagent.taskstate import slice_to_task_state, task_state_to_slice  # noqa: E402
from memagent.regions import MAX_REQUIREMENTS                   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _emit(sink, name, args, failing=False):
    sink(ToolResult(name=name, args=args, output="ok", failing=failing))


@check
def require_folds_and_is_idempotent():
    s = Slice(); s.reset("build a parser")
    sink = slice_sink(s)
    _emit(sink, "require", {"text": "parse_date returns ISO8601"})
    _emit(sink, "require", {"text": "parse_date returns ISO8601"})   # dup → no-op
    _emit(sink, "require", {"text": "raise on bad input"})
    assert [r["text"] for r in s.requirements] == ["parse_date returns ISO8601", "raise on bad input"], s.requirements
    assert all(r["done"] is False for r in s.requirements)


@check
def requirement_done_flips_in_place():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "handle nulls"})
    _emit(sink, "requirement_done", {"text": "handle NULLS"})        # case-insensitive match
    assert len(s.requirements) == 1 and s.requirements[0]["done"] is True, s.requirements


@check
def drop_removes_and_nomatch_is_noop():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "keep it under 100 lines"})
    _emit(sink, "drop_requirement", {"text": "something never required"})  # no match → no-op
    assert len(s.requirements) == 1
    _emit(sink, "drop_requirement", {"text": "keep it under 100 lines"})
    assert s.requirements == []


@check
def failing_require_ignored():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "x"}, failing=True)
    assert s.requirements == [], "a failing require must not mutate the contract"


@check
def renders_open_and_done():
    wd = tempfile.mkdtemp(prefix="req-")
    s = Slice(); s.reset("build a parser"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "parse_date returns ISO8601"})
    _emit(sink, "require", {"text": "raise on bad input"})
    _emit(sink, "requirement_done", {"text": "raise on bad input"})
    build = make_build_slice(s, LocalToolHost(root=wd), None, NullMemory(), s.goal)
    user = build()[1]["content"]
    assert "# STANDING REQUIREMENTS" in user
    assert "- [ ] parse_date returns ISO8601" in user
    assert "- [x] raise on bad input" in user, user


@check
def empty_contract_suppresses_region():
    # the live-bug kill: a greeting produces NO contract, so no binding region renders at all.
    wd = tempfile.mkdtemp(prefix="req-")
    s = Slice(); s.reset("hi")
    build = make_build_slice(s, LocalToolHost(root=wd), None, NullMemory(), s.goal)
    user = build()[1]["content"]
    assert "# STANDING REQUIREMENTS" not in user, "a greeting must produce NO binding contract region"


@check
def hi_then_who_are_you_has_no_stale_anchor():
    # the EXACT reported bug: "hi" then "who are you" must not re-anchor on "hi".
    sess = Session(NullMemory()); sess.new_topic("hi")
    sess.continue_topic("who are you")
    wd = tempfile.mkdtemp(prefix="req-")
    build = make_build_slice(sess, LocalToolHost(root=wd), None, NullMemory(), "who are you")
    msgs = build(); system, user = msgs[0]["content"], msgs[1]["content"]
    assert "who are you" in system, "the current directive must drive the system TASK"
    assert "STANDING REQUIREMENTS" not in user and "TASK SPEC" not in user, \
        "no stale 'hi' binding anchor may render"


@check
def carries_seal_and_continue_wipes_on_reset():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "must be thread-safe"})
    s.seal()
    assert s.requirements and s.requirements[0]["text"] == "must be thread-safe", "contract carries the seal"
    s.reset("a brand new task")
    assert s.requirements == [], "reset (new task) wipes the contract"


@check
def bounded():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    for i in range(MAX_REQUIREMENTS + 10):
        _emit(sink, "require", {"text": f"req {i}"})
    assert len(s.requirements) == MAX_REQUIREMENTS, f"contract must be bounded: {len(s.requirements)}"


@check
def serialized_roundtrip_carries_requirements_and_world():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "req A"})
    _emit(sink, "requirement_done", {"text": "req A"})
    _emit(sink, "world_set", {"key": "k", "value": "v"})
    r = task_state_to_slice(slice_to_task_state(s, "t1"))
    assert r.requirements == [{"text": "req A", "done": True}], r.requirements
    assert r.world == {"k": "v"}, "world must survive resume (was a latent serialization bug)"


@check
def tools_registered_and_confirm():
    h = LocalToolHost(root=tempfile.mkdtemp(prefix="req-"))
    names = {sc["function"]["name"] for sc in h.schemas()}
    assert {"require", "requirement_done", "drop_requirement"} <= names, names
    out = h.run("require", {"text": "x"})
    assert "REQUIREMENT" in out and not out.startswith("Error:"), out


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
