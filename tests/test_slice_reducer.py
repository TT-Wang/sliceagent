import pytest

from sliceagent.events import ToolResult, TurnStarted
from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
from sliceagent.pfc import Slice, slice_sink
from sliceagent.slice_reducer import SliceReducer


def _result(
    invocation_id: str,
    name: str,
    args,
    *,
    effects=(),
    output: str = "ok",
) -> ToolResult:
    invocation = ToolInvocation(invocation_id, name, args, 0)
    outcome = ToolOutcome(
        invocation,
        ToolStatus.SUCCEEDED,
        output,
        tuple(effects),
    )
    return ToolResult(
        name,
        args,
        output,
        False,
        status=ToolStatus.SUCCEEDED.value,
        invocation_id=invocation_id,
        outcome=outcome,
    )


def test_slice_sink_is_the_typed_reducer_and_ignores_presentation_events():
    state = Slice()
    state.reset("keep the semantic state small")

    reducer = slice_sink(state)

    assert isinstance(reducer, SliceReducer)
    reducer(TurnStarted("presentation only"))
    assert state.goal == "keep the semantic state small"
    assert state.runtime.recent_calls == []


def test_effect_replay_is_exactly_once_but_logical_calls_are_all_accounted():
    state = Slice()
    state.reset("set a world fact once")
    reducer = slice_sink(state)
    effect = ToolEffect("effect:shared", "world_observed", {})

    reducer(_result("call:1", "world_set", {"key": "branch", "value": "main"}, effects=(effect,)))
    reducer(_result("call:2", "world_set", {"key": "branch", "value": "other"}, effects=(effect,)))

    assert state.world == {"branch": "main"}
    assert [call["id"] for call in state.runtime.recent_calls] == ["call:1", "call:2"]
    assert state.runtime.applied_effect_ids == {"effect:shared"}


def test_partial_effect_replay_fails_before_a_new_runtime_row_is_created():
    state = Slice()
    state.reset("reject ambiguous replay")
    reducer = slice_sink(state)
    first = (
        ToolEffect("effect:one", "opaque", {}),
        ToolEffect("effect:two", "opaque", {}),
    )
    reducer(_result("call:1", "world_set", {"key": "x", "value": "one"}, effects=first))

    mixed = (
        ToolEffect("effect:one", "opaque", {}),
        ToolEffect("effect:three", "opaque", {}),
    )
    with pytest.raises(RuntimeError, match="partially replayed"):
        reducer(_result("call:2", "world_set", {"key": "x", "value": "two"}, effects=mixed))

    assert state.world == {"x": "one"}
    assert [call["id"] for call in state.runtime.recent_calls] == ["call:1"]


def test_child_claims_remain_delegated_testimony_with_artifact_provenance():
    state = Slice()
    state.reset("inspect CI")
    reducer = slice_sink(state)
    child = ToolEffect(
        "effect:child",
        "child_artifact",
        {
            "artifact_id": "child:sealed-report",
            "delegation_target": "inspect CI",
            "scope": [".github/workflows"],
            "claims": [{
                "text": "A CI workflow exists",
                "report_exact": "A CI workflow exists",
                "observation_refs": ["0" * 64],
            }],
        },
    )

    reducer(_result(
        "call:child",
        "spawn_agent",
        {"agent": "explorer", "task": "inspect CI"},
        effects=(child,),
        output="A CI workflow exists",
    ))

    call = state.runtime.recent_calls[-1]
    assert call["child_artifact_id"] == "child:sealed-report"
    assert call["child_scope"] == [".github/workflows"]
    assert call["child_claims"][0]["observation_refs"] == ["0" * 64]
    assert state.finding_source["A CI workflow exists"] == "delegated"


def test_malformed_third_party_args_reduce_to_an_empty_mapping():
    state = Slice()
    state.reset("survive a malformed extension event")

    slice_sink(state)(ToolResult(
        "extension_tool", ["not", "a", "mapping"], "ok", False,
        invocation_id="call:bad-args",
    ))

    assert state.runtime.recent_calls[-1]["args"] == {}


def test_failed_typed_outcome_cannot_be_laundered_by_success_compatibility_fields():
    state = Slice()
    state.reset("typed failure is authority")
    invocation = ToolInvocation("call:failed", "world_set", {"key": "k", "value": "v"}, 0)
    outcome = ToolOutcome(invocation, ToolStatus.FAILED, "failed", ())
    event = ToolResult(
        "world_set", dict(invocation.args), "failed", False,
        status="succeeded", invocation_id=invocation.id, outcome=outcome,
    )

    with pytest.raises(RuntimeError, match="status, failing"):
        slice_sink(state)(event)

    assert state.world == {}
    assert state.runtime.recent_calls == []
    assert state.runtime.applied_effect_ids == set()


@pytest.mark.parametrize("field", ["name", "args", "invocation_id", "output"])
def test_typed_outcome_identity_mismatch_is_rejected_atomically(field):
    state = Slice()
    state.reset("reject a split-brain outcome")
    invocation = ToolInvocation("call:canonical", "world_set", {"key": "k", "value": "v"}, 0)
    effect = ToolEffect("effect:canonical", "opaque", {})
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "ok", (effect,))
    fields = {
        "name": "world_set",
        "args": dict(invocation.args),
        "output": "ok",
        "failing": False,
        "status": "succeeded",
        "invocation_id": invocation.id,
        "outcome": outcome,
    }
    fields[field] = {
        "name": "world_clear",
        "args": {"key": "other"},
        "invocation_id": "call:other",
        "output": "different",
    }[field]

    with pytest.raises(RuntimeError, match=field):
        slice_sink(state)(ToolResult(**fields))

    assert state.world == {} and state.runtime.recent_calls == []
    assert state.runtime.applied_effect_ids == set()


def test_typed_outcome_fills_optional_legacy_identity_and_status_fields():
    state = Slice()
    state.reset("canonicalize an additive compatibility projection")
    invocation = ToolInvocation("call:canonical", "world_set", {"key": "k", "value": "v"}, 0)
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "ok", ())

    slice_sink(state)(ToolResult(
        "world_set", dict(invocation.args), "ok", False, outcome=outcome,
    ))

    assert state.world == {"k": "v"}
    assert state.runtime.recent_calls[-1]["id"] == invocation.id
    assert state.runtime.recent_calls[-1]["status"] == "succeeded"
