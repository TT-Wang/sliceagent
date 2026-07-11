"""Canonical completion for pure past-execution recall."""
from types import SimpleNamespace as NS

from sliceagent.hooks import (DelegatedClaimCompletionHook, DelegationCompletionHook,
                              ExecutionEvidenceCompletionHook)
from sliceagent.events import StepBegin, ToolResult, ToolStarted
from sliceagent.execution import ToolEffect, ToolInvocation, ToolOutcome, ToolStatus
from sliceagent.intent import DelegationRequirement, EvidenceQuery
from sliceagent.pfc import Slice, slice_sink


def _state(*, family="delegation", coverage="complete", mixed=False):
    query = EvidenceQuery(
        source="execution_receipt", family=family, predicate="failure_detail", scope="task",
    )
    aggregate = {
        "kind": "execution_receipt_aggregate", "operation_count": 3, "child_artifact_count": 3,
        "counts": {
            "requested": 3, "execution_started": 3, "settled": 3, "succeeded": 3,
            "rejected_before_execution": 0, "failed": 0, "cancelled": 0,
            "indeterminate": 0, "not_started": 0, "unknown": 0,
        },
    }
    contract = NS(
        evidence_query=query,
        quality_evidence_query=(NS(purpose="assess") if mixed else None),
        source_needs=("execution_receipt", "sealed_exchange") if mixed else ("execution_receipt",),
        referents=(aggregate, {"kind": "execution_receipt_coverage", "coverage": coverage}),
    )
    return NS(intent=NS(turn_contract=contract))


def test_pure_receipt_recall_is_replaced_by_the_exact_selected_family_answer():
    result = ExecutionEvidenceCompletionHook(lambda: _state()).validate_completion(
        "Probably three explorers plus one summary operation; everything worked.", "end_turn",
    )
    assert result["exclusive"]
    answer = result["replacement"]
    assert "family=delegation" in answer
    assert "3 child agents started execution" in answer
    assert "3 succeeded" in answer and "0 failed" in answer
    assert "distinct sealed child artifacts=3" in answer
    assert "summary operation" not in answer


def test_partial_receipt_recall_is_a_lower_bound_not_a_success_claim():
    answer = ExecutionEvidenceCompletionHook(
        lambda: _state(coverage="partial"),
    ).validate_completion("All succeeded.", "end_turn")["replacement"]
    assert "partial lower bound" in answer
    assert "overall outcome are unknown" in answer


def test_mixed_source_comparison_remains_model_owned():
    hook = ExecutionEvidenceCompletionHook(lambda: _state(mixed=True))
    assert hook.validate_completion("Compare my claim to the receipts.", "end_turn") is None
    assert hook.validate_completion("draft", "max_tokens") is None


def _delegation_state(calls=(), current_request=""):
    requirement = DelegationRequirement(
        agent="explorer", count=3, targets=("app.py", "auth.py", "util.py"), parallel=True,
    )
    contract = NS(delegation_requirement=requirement)
    return NS(
        intent=NS(turn_contract=contract, current_request=current_request),
        runtime=NS(recent_calls=list(calls)),
    )


def test_explicit_delegation_blocks_direct_terminal_prose_until_exact_wave_exists():
    state = _delegation_state()
    hook = DelegationCompletionHook(lambda: state)
    rejected = hook.validate_completion("I reviewed the files directly.", "end_turn")
    assert rejected["continue"] and "successful explorer children=0, expected=3" in rejected["feedback"]
    assert "Emit those calls together" in rejected["feedback"]
    state.runtime.recent_calls = [
        {
            "name": "spawn_agent", "args": {"agent": "explorer", "task": f"Review {target}"},
            "status": "succeeded", "step": 2,
        }
        for target in ("app.py", "auth.py", "util.py")
    ]
    assert hook.validate_completion("Combined child results.", "end_turn") is None


def test_irreparable_wrong_or_sequential_delegation_fails_honestly():
    calls = [
        {
            "name": "spawn_agent", "args": {"agent": "explorer", "task": f"Review {target}"},
            "status": "succeeded", "step": step,
        }
        for step, target in enumerate(("app.py", "auth.py", "other.py"), 1)
    ]
    result = DelegationCompletionHook(lambda: _delegation_state(calls)).validate_completion(
        "All done.", "end_turn",
    )
    assert "replacement" in result
    assert "won't present direct parent analysis" in result["replacement"]
    assert "missing target coverage=util.py" in result["replacement"]


def test_extra_duplicate_child_is_an_irreparable_exact_count_failure():
    calls = [
        {
            "id": f"spawn-{index}", "name": "spawn_agent",
            "args": {"agent": "explorer", "task": f"Review {target}"},
            "status": "succeeded", "step": 2,
        }
        for index, target in enumerate(("app.py", "auth.py", "util.py", "app.py"), 1)
    ]
    result = DelegationCompletionHook(lambda: _delegation_state(calls)).validate_completion(
        "Combined child results.", "end_turn",
    )
    assert "replacement" in result
    assert "completed successfully: 4" in result["replacement"]
    assert "successful explorer children=4, expected=3" in result["replacement"]


def test_delegated_consequence_gate_requires_the_condition_on_the_same_claim_line():
    state = _delegation_state([{
        "name": "spawn_agent", "args": {"agent": "explorer", "task": "Review app.py"},
        "status": "succeeded", "step": 2,
    }])
    hook = DelegatedClaimCompletionHook(lambda: state, max_revisions=1, source_review=False)
    rejected = hook.validate_completion(
        "app.py — SQL injection vulnerability from query construction.\n"
        "Elsewhere: a caller would have to execute the string.",
        "end_turn",
    )
    assert rejected["continue"]
    assert "SQL injection vulnerability" in rejected["feedback"]
    assert hook.validate_completion(
        "app.py — Potential SQL injection if a caller executes the constructed string as SQL.",
        "end_turn",
    ) is None


def test_delegated_consequence_gate_is_inactive_without_a_successful_child():
    state = _delegation_state([])
    assert DelegatedClaimCompletionHook(lambda: state, source_review=False).validate_completion(
        "The observed execute call is a SQL injection vulnerability.", "end_turn",
    ) is None


def test_delegated_consequence_gate_has_a_truthful_bounded_fallback():
    state = _delegation_state([{
        "name": "spawn_agent", "args": {"agent": "explorer", "task": "Review auth.py"},
        "status": "succeeded", "step": 2,
    }])
    hook = DelegatedClaimCompletionHook(lambda: state, max_revisions=0, source_review=False)
    result = hook.validate_completion(
        "auth.py — timing side-channel attack leaks password content.", "end_turn",
    )
    assert result["exclusive"]
    assert result["replacement"].startswith("Conditional (not established")
    assert "not established by the retained observation alone" in result["replacement"]
    assert "verify the caller, execution sink" in result["replacement"]


def test_delegated_revision_stays_private_and_non_exact_shape_survives_fallback():
    state = _delegation_state([{
        "name": "spawn_agent", "args": {"agent": "explorer", "task": "Review app.py"},
        "status": "succeeded", "step": 2,
    }], current_request="Give me a combined summary.")
    hook = DelegatedClaimCompletionHook(lambda: state, max_revisions=1, source_review=False)
    first = hook.validate_completion(
        "app.py — SQL injection vulnerability.",
        "end_turn",
    )
    assert first["continue"]
    assert "Do not call any tool" in first["feedback"]
    assert "user has not seen" in first["feedback"]
    fallback = hook.validate_completion(
        "The current TURN CONTRACT says I need to revise my original line.\n\n"
        "Here are the revised lines:\n"
        "1. app.py — Potential SQL injection if a caller executes the returned query.\n"
        "2. auth.py — unresolved get_password may fail when login is called.\n"
        "3. util.py — bare except masks transform failures.",
        "end_turn",
    )
    assert fallback["exclusive"] and "continue" not in fallback
    assert fallback["replacement"].splitlines() == [
        "1. app.py — Potential SQL injection if a caller executes the returned query.",
        "2. auth.py — unresolved get_password may fail when login is called.",
        "3. util.py — bare except masks transform failures.",
    ]
    assert "TURN CONTRACT" not in fallback["replacement"]
    assert "revised lines" not in fallback["replacement"]
    assert hook.revisions == 1


def test_delegated_fallback_rewrites_each_claim_unit_and_has_no_surviving_violation():
    state = _delegation_state([{
        "name": "spawn_agent", "args": {"agent": "explorer", "task": "Review app.py"},
        "status": "succeeded", "step": 2,
    }])
    hook = DelegatedClaimCompletionHook(lambda: state, max_revisions=0, source_review=False)
    candidate = "It may warn. App.py has SQL injection vulnerability; auth.py leaks password content."
    result = hook.validate_completion(candidate, "end_turn")
    assert result["exclusive"]
    assert not hook._violations(result["replacement"]), result["replacement"]
    assert result["replacement"].count("Conditional (not established") == 2


def test_delegated_gate_does_not_rewrite_negated_or_attributed_consequences():
    state = _delegation_state([{
        "name": "spawn_agent", "args": {"agent": "explorer", "task": "Review app.py"},
        "status": "succeeded", "step": 2,
    }])
    hook = DelegatedClaimCompletionHook(lambda: state, max_revisions=0, source_review=False)
    candidate = (
        "No SQL injection vulnerability was found.\n"
        "> The child reported SQL injection vulnerability.\n"
        "The source called it a timing side-channel attack."
    )
    assert hook.validate_completion(candidate, "end_turn") is None


def test_prior_turn_delegated_testimony_keeps_the_modality_backstop_active():
    state = _delegation_state([])
    state.evidence = NS(finding_source={"child report": "delegated"})
    hook = DelegatedClaimCompletionHook(lambda: state, max_revisions=0, source_review=False)
    result = hook.validate_completion("The project has SQL injection vulnerability.", "end_turn")
    assert result["exclusive"]
    assert result["replacement"].startswith("Conditional (not established")


def test_current_delegation_gets_one_private_tool_free_source_reconciliation_pass():
    state = _delegation_state([{
        "name": "spawn_agent", "args": {"agent": "explorer", "task": "Review app.py"},
        "status": "succeeded", "step": 2,
    }])
    hook = DelegatedClaimCompletionHook(lambda: state)
    decision = hook.validate_completion("app.py — looks wrong", "end_turn")
    assert decision["continue"] and decision["exclusive"] and decision["prose_only"]
    assert "PRIMARY OBSERVATION" in decision["feedback"]
    assert "caught exception does not crash its caller" in decision["feedback"]
    assert hook.validate_completion("app.py — undefined dependency may fail when called", "end_turn") is None


def _typed_claim_call(target, claim):
    return {
        "id": f"spawn-{target}", "name": "spawn_agent",
        "args": {"agent": "explorer", "task": f"Review {target}"},
        "status": "succeeded", "step": 2, "child_scope": [target], "child_target": target,
        "child_artifact_id": f"sealed-{target}",
        "child_claims": [{
            "v": 1, "text": claim, "report_exact": claim, "modality": "conditional",
            "observation_refs": ["a" * 64], "prerequisites": [],
        }],
    }


def test_exact_delegated_shape_is_reduced_from_verbatim_child_testimony_not_parent_prose():
    calls = [
        _typed_claim_call("app.py", "TOP CLAIM: query risk exists only if an unseen caller executes it."),
        _typed_claim_call("auth.py", "TOP CLAIM: get_password is unresolved in the inspected workspace."),
        _typed_claim_call("util.py", "TOP CLAIM: bare except catches the NameError and returns None."),
    ]
    state = _delegation_state(calls, current_request="Give me a combined 3-line summary.")
    result = DelegatedClaimCompletionHook(lambda: state).validate_completion(
        "app.py definitely executes SQL; auth.py stores plaintext; util.py crashes.", "end_turn",
    )
    assert result["exclusive"] and "typed child testimony" in result["reason"]
    lines = result["replacement"].splitlines()
    assert len(lines) == 3
    assert [target in line for target, line in zip(("app.py", "auth.py", "util.py"), lines)] == [True] * 3
    assert all("unverified delegated testimony" in line for line in lines)
    assert "only if an unseen caller executes it" in lines[0]
    assert "stores plaintext" not in result["replacement"] and "crashes" not in result["replacement"]


def test_exact_testimony_reducer_rejects_substring_and_ambiguous_scope_binding():
    calls = [
        _typed_claim_call("data.py", "TOP CLAIM: wrong target."),
        _typed_claim_call("auth.py", "TOP CLAIM: auth."),
        _typed_claim_call("util.py", "TOP CLAIM: util."),
    ]
    # The task prose mentions app.py, so a substring/text matcher could misbind it; host-minted scope cannot.
    calls[0]["args"]["task"] = "Review data.py while comparing the requested app.py name"
    state = _delegation_state(calls, current_request="Give me a combined 3-line summary.")
    hook = DelegatedClaimCompletionHook(lambda: state, source_review=False)
    assert hook._canonical_exact_testimony(state, ("app.py", "auth.py", "util.py")) is None

    calls[0]["child_target"] = "app.py"
    state.runtime.recent_calls.append(_typed_claim_call("app.py", "TOP CLAIM: ambiguous duplicate."))
    assert hook._canonical_exact_testimony(state, ("app.py", "auth.py", "util.py")) is None

    state.runtime.recent_calls.pop()
    calls_in_state = state.runtime.recent_calls
    calls_in_state[0]["child_target"] = "App.py"
    assert hook._canonical_exact_testimony(state, ("app.py", "auth.py", "util.py")) is None


def test_exact_fanout_missing_host_target_fails_closed_without_revision_or_free_synthesis():
    calls = [
        _typed_claim_call("app.py", "TOP CLAIM: app testimony."),
        _typed_claim_call("auth.py", "TOP CLAIM: auth testimony."),
        _typed_claim_call("util.py", "TOP CLAIM: util testimony."),
    ]
    calls[2]["child_target"] = ""
    state = _delegation_state(calls, current_request="Give me a combined 3-line summary.")
    candidate = (
        "1. app.py — free parent synthesis.\n"
        "2. auth.py — free parent synthesis.\n"
        "3. util.py — free parent synthesis."
    )

    hook = DelegatedClaimCompletionHook(lambda: state)
    result = hook.validate_completion(candidate, "end_turn")

    assert result["exclusive"]
    assert "continue" not in result and "prose_only" not in result
    assert result["reason"] == "exact delegated reduction lacked complete typed target/claim bindings"
    assert len(result["replacement"].splitlines()) == 1
    assert "couldn't produce the requested delegated summary" in result["replacement"]
    assert "will not substitute free parent synthesis" in result["replacement"]
    assert candidate not in result["replacement"] and "1. app.py" not in result["replacement"]
    assert hook.revisions == 0 and not hook._source_reviewed


def test_exact_fanout_malformed_typed_claim_fails_closed_without_continuation():
    calls = [
        _typed_claim_call("app.py", "TOP CLAIM: app testimony."),
        _typed_claim_call("auth.py", "TOP CLAIM: auth testimony."),
        _typed_claim_call("util.py", "TOP CLAIM: util testimony."),
    ]
    calls[1]["child_claims"] = [{
        "v": 1,
        "text": "TOP CLAIM: malformed because its exact sealed-report span is missing.",
        "modality": "inference",
        "observation_refs": [],
        "prerequisites": [],
    }]
    state = _delegation_state(calls, current_request="Give me a combined 3-line summary.")

    hook = DelegatedClaimCompletionHook(lambda: state)
    result = hook.validate_completion(
        "1. app.py — invented.\n2. auth.py — invented.\n3. util.py — invented.",
        "end_turn",
    )

    assert result["exclusive"]
    assert "continue" not in result and "feedback" not in result
    assert len(result["replacement"].splitlines()) == 1
    assert "one unambiguous host-bound target and exact TOP CLAIM" in result["replacement"]
    assert "invented" not in result["replacement"]
    assert hook.max_revisions == 0 and hook.revisions == 0


def test_pfc_captures_only_strict_typed_claim_effects_from_spawn_calls():
    state = Slice(); state.reset("task")
    claim = {
        "v": 1, "text": "TOP CLAIM: conditional result.",
        "report_exact": "TOP CLAIM: conditional result.", "modality": "conditional",
        "observation_refs": ["b" * 64], "prerequisites": [],
    }

    def deliver(call_id, name, claims):
        invocation = ToolInvocation(call_id, name, {"agent": "explorer", "task": "Review app.py"}, 0)
        outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "ok", (ToolEffect(
            f"effect-{call_id}", "child_artifact", {
                "artifact_id": "sealed-app", "scope": ["app.py"],
                "delegation_target": "app.py", "claims": claims,
            },
        ),))
        slice_sink(state)(ToolResult(
            name, invocation.args, "ok", False, invocation_id=call_id, outcome=outcome,
        ))

    deliver("good", "spawn_agent", [claim])
    good = state.runtime.recent_calls[-1]
    assert good["child_scope"] == ["app.py"] and good["child_target"] == "app.py"
    assert good["child_claims"] == [claim]

    changed = {**claim, "report_exact": "TOP CLAIM: replay tried to replace testimony.",
               "text": "TOP CLAIM: replay tried to replace testimony."}
    deliver("good", "spawn_agent", [changed])
    assert state.runtime.recent_calls[-1]["child_claims"] == [claim], \
        "a settled invocation replay cannot replace its terminal claim metadata"

    deliver("malformed", "spawn_agent", "not-a-claim-list")
    assert "child_claims" not in state.runtime.recent_calls[-1]
    missing_exact = {key: value for key, value in claim.items() if key != "report_exact"}
    deliver("missing-exact", "spawn_agent", [missing_exact])
    assert "child_claims" not in state.runtime.recent_calls[-1]
    deliver("spoof", "read_file", [claim])
    assert "child_claims" not in state.runtime.recent_calls[-1]


def test_same_effect_for_distinct_invocations_is_counted_twice_but_reduced_once():
    state = Slice()
    state.reset("task")
    sink = slice_sink(state)

    def result(call_id: str, value: str):
        invocation = ToolInvocation(call_id, "world_set", {"key": "n", "value": value}, 0)
        outcome = ToolOutcome(
            invocation, ToolStatus.SUCCEEDED, "ok",
            (ToolEffect("shared-effect", "tool_outcome", {"name": "world_set"}),),
        )
        return ToolResult(
            "world_set", {"key": "n", "value": value}, "ok", False,
            invocation_id=call_id, outcome=outcome,
        )

    sink(StepBegin(4))
    first = result("call-1", "1")
    sink(ToolStarted("world_set", {"key": "n", "value": "1"}, first.outcome.invocation))
    sink(first)
    second = result("call-2", "2")
    sink(ToolStarted("world_set", {"key": "n", "value": "2"}, second.outcome.invocation))
    sink(ToolStarted("world_set", {"key": "n", "value": "2"}, second.outcome.invocation))
    sink(second)
    assert state.world["n"] == "1", "the shared semantic effect must apply exactly once"
    assert [call["id"] for call in state.runtime.recent_calls] == ["call-1", "call-2"]
    assert all(call["status"] == "succeeded" for call in state.runtime.recent_calls)
    assert all(call["step"] == 4 for call in state.runtime.recent_calls)

    sink(first)
    assert [call["id"] for call in state.runtime.recent_calls] == ["call-1", "call-2"], (
        "replaying the same invocation ID must update its row, not create a third logical call"
    )


def test_partial_effect_replay_fails_without_mutating_logical_accounting():
    state = Slice()
    state.reset("task")
    state.runtime.applied_effect_ids.add("seen")
    invocation = ToolInvocation("call-partial", "world_set", {"key": "n", "value": "2"}, 0)
    outcome = ToolOutcome(
        invocation, ToolStatus.SUCCEEDED, "ok",
        (
            ToolEffect("seen", "tool_outcome", {"name": "world_set"}),
            ToolEffect("new", "tool_outcome", {"name": "world_set"}),
        ),
    )
    event = ToolResult(
        "world_set", {"key": "n", "value": "2"}, "ok", False,
        invocation_id=invocation.id, outcome=outcome,
    )

    import pytest
    with pytest.raises(RuntimeError, match="partially replayed"):
        slice_sink(state)(event)
    assert state.runtime.recent_calls == []
    assert "n" not in state.world
