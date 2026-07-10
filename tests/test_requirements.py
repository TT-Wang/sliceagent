"""Typed standing-intent compatibility through require/requirement_done/drop_requirement. EMPTY by
default so a greeting never becomes a binding spec; carried across seal/resume and task-elastic (no
arbitrary semantic count/character cap).
No model, no pytest. Run: PYTHONPATH=src python tests/test_requirements.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.pfc import Slice, record_user, slice_sink  # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402
from sliceagent.events import ToolResult, TurnEnd, TurnInterrupted  # noqa: E402
from sliceagent.tools import LocalToolHost                        # noqa: E402
from sliceagent.memory import NullMemory                          # noqa: E402
from sliceagent.session import Session                            # noqa: E402
from sliceagent.taskstate import slice_to_task_state, task_state_to_slice  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _emit(sink, name, args, failing=False):
    sink(ToolResult(name=name, args=args, output="ok", failing=failing))


def _verified(s):
    s.runtime.recent_calls.append({"id": "verify-1", "name": "read_file", "status": "succeeded"})


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
    _verified(s)
    _emit(sink, "requirement_done", {"text": "handle NULLS"})        # case-insensitive match
    assert len(s.requirements) == 1 and s.requirements[0]["done"] is True, s.requirements
    assert s.intent.entries[0].status == "provisionally_satisfied", "model completion is provisional"


@check
def requirement_done_without_observed_evidence_does_not_retire():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "handle nulls"})
    _emit(sink, "requirement_done", {"text": "handle nulls"})
    assert s.intent.entries[0].status == "active"


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
    _verified(s)
    _emit(sink, "requirement_done", {"text": "raise on bad input"})
    build = make_build_slice(s, LocalToolHost(root=wd), None, NullMemory(), s.goal)
    user = build()[1]["content"]
    assert "# PARENT TASK CONSTRAINTS" in user
    assert "- [ ] parse_date returns ISO8601" in user
    assert "- [~] raise on bad input" in user and "not user-finalized" in user, user


@check
def empty_contract_suppresses_region():
    # the live-bug kill: a greeting produces NO contract, so no binding region renders at all.
    wd = tempfile.mkdtemp(prefix="req-")
    s = Slice(); s.reset("hi")
    build = make_build_slice(s, LocalToolHost(root=wd), None, NullMemory(), s.goal)
    user = build()[1]["content"]
    assert "# ACTIVE USER INTENT" not in user, "a greeting must produce NO binding contract region"


@check
def completed_objective_becomes_pageable_background_for_a_new_directive():
    # The production-path regression: the lexical router continues an ambiguous follow-up, but a completed
    # first objective must not remain a mandatory, unpageable instruction forever.
    first = "List the files in this repository"
    sess = Session(NullMemory()); sess.new_topic(first)
    record_user(sess.active(), first, source_artifact="turn-list-files")
    slice_sink(sess)(TurnEnd("end_turn", 1, {}))
    assert sess.active().task.objective_status == "provisionally_satisfied"
    sess.continue_topic("Who are you and what do you do?")
    wd = tempfile.mkdtemp(prefix="req-")
    build = make_build_slice(
        sess, LocalToolHost(root=wd), None, NullMemory(), "Who are you and what do you do?",
    )
    plan = build(); system, user = plan[0]["content"], plan[1]["content"]
    assert "Who are you and what do you do?" in user
    assert "Who are you and what do you do?" not in system
    objective = [block for block in plan.blocks if block.item_id == "region:task_objective"]
    assert objective and all(not block.mandatory for block in objective)
    assert {block.fidelity.value for block in objective} == {"full", "locator"}
    locator = next(block for block in objective if block.fidelity.value == "locator")
    assert "turn-list-files" in locator.content
    chosen = plan.controller.select(objective, capacity_chars=len(locator.content))
    assert chosen.blocks == (locator,), "pressure must page completed objective to its exact source"


@check
def explicit_resume_and_failure_reactivate_the_original_objective():
    for followup in ("continue", "it is still broken"):
        sess = Session(NullMemory()); sess.new_topic("Repair the parser")
        slice_sink(sess)(TurnEnd("end_turn", 1, {}))
        assert sess.active().task.objective_status == "provisionally_satisfied"
        sess.continue_topic(followup)
        assert sess.active().task.objective_status == "active"


@check
def unresolved_binding_state_keeps_objective_active_after_clean_reply():
    s = Slice(); s.reset("Implement retry scheduling and never modify config.py")
    record_user(s, s.goal, source_artifact="turn-objective")
    assert s.intent.open_entries(), "the explicit constraint must be resident"
    slice_sink(s)(TurnEnd("end_turn", 1, {}))
    assert s.task.objective_status == "active"

    planned = Slice(); planned.reset("Complete the migration")
    planned.plan = [{"step": "run compatibility tests", "status": "pending"}]
    slice_sink(planned)(TurnEnd("end_turn", 1, {}))
    assert planned.task.objective_status == "active"


@check
def interrupted_turn_reactivates_a_provisional_objective():
    s = Slice(); s.reset("Repair the parser")
    s.task.mark_objective_provisional()
    slice_sink(s)(TurnInterrupted("max_steps"))
    assert s.task.objective_status == "active"


@check
def carries_seal_and_continue_wipes_on_reset():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "must be thread-safe"})
    s.seal()
    assert s.requirements and s.requirements[0]["text"] == "must be thread-safe", "contract carries the seal"
    s.reset("a brand new task")
    assert s.requirements == [], "reset (new task) wipes the contract"


@check
def task_elastic_and_exact():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    for i in range(40):
        _emit(sink, "require", {"text": f"req {i}"})
    long_clause = "exact-signature(" + ("parameter_name, " * 80) + ")"
    _emit(sink, "require", {"text": long_clause})
    assert len(s.requirements) == 41, f"every active obligation must survive: {len(s.requirements)}"
    assert s.requirements[-1]["text"] == long_clause, "standing intent must not be truncated"


@check
def model_drop_cannot_retire_user_authored_intent():
    clause = "keep the public API stable"
    s = Slice(); s.reset(f"Refactor this, but {clause}"); sink = slice_sink(s)
    _emit(sink, "require", {"text": clause})
    assert s.intent.entries[0].authority == "user"
    _emit(sink, "drop_requirement", {"text": clause})
    assert s.intent.entries[0].status == "active" and s.requirements, \
        "a model-issued drop cannot retract the user's exact clause"


@check
def serialized_roundtrip_carries_requirements_and_world():
    s = Slice(); s.reset("t"); sink = slice_sink(s)
    _emit(sink, "require", {"text": "req A"})
    _verified(s)
    _emit(sink, "requirement_done", {"text": "req A"})
    _emit(sink, "world_set", {"key": "k", "value": "v"})
    r = task_state_to_slice(slice_to_task_state(s, "t1"))
    assert r.requirements == [{"text": "req A", "done": True}], r.requirements
    assert r.world == {"k": "v"}, "world must survive resume (was a latent serialization bug)"


@check
def tools_registered_and_confirm():
    h = LocalToolHost(root=tempfile.mkdtemp(prefix="req-"))
    names = {sc["function"]["name"] for sc in h.schemas()}
    assert {"require", "requirement_done", "drop_requirement", "supersede_requirement"} <= names, names
    out = h.run("require", {"text": "x"})
    assert "REQUIREMENT" in out and not out.startswith("Error:"), out


@check
def explicit_current_user_correction_supersedes_old_clause():
    old = "keep API v1"
    new = "use API v2 instead"
    s = Slice(); s.reset(old); sink = slice_sink(s)
    _emit(sink, "require", {"text": old})
    s.intent.begin_turn(f"Correction: {new}", source_artifact="turn-current")
    sink(ToolResult("supersede_requirement", {"old_text": old, "new_text": new}, "ok", False))
    assert s.intent.entries[0].status == "superseded"
    assert s.intent.entries[1].verbatim_clause == new and s.intent.entries[1].authority == "user"


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
