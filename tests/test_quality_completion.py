"""Deterministic response-quality completion protocol."""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import AssistantText  # noqa: E402
from sliceagent.hooks import (CompositeHooks, Hooks, OracleHook,
                              QualityEvidenceCompletionHook)  # noqa: E402
from sliceagent.loop import run_turn  # noqa: E402


NO_ISSUE = "No supported response-quality issue is evidenced."
UNCERTAIN = "The sealed response-quality evidence is incomplete, so no observed-quality verdict is asserted."
CHECK = (
    "Quality check\n"
    "Source: artifacts/turn-1.md\n"
    "Basis: exact request/response and any sealed grounding\n"
    "Verdict: no admitted mismatch"
)


def _audited(answer: str) -> str:
    return CHECK + "\n\n" + answer


def _state(*, purpose="assess", prospective=False, coverage="complete"):
    query = NS(purpose=purpose, prospective_requested=prospective)
    referents = [{
        "kind": "execution_receipt_aggregate",
        "operation_count": 8,
        "counts": {
            "requested": 8, "execution_started": 8, "settled": 8, "succeeded": 8,
            "failed": 0, "cancelled": 0, "indeterminate": 0, "not_started": 0, "unknown": 0,
        },
        "turn_counts": {
            "completed": 6, "completed_with_warnings": 0, "paused": 0, "blocked": 0,
            "interrupted": 0, "indeterminate": 0, "unknown": 0,
        },
        "turn_warning_count": 0, "nonclean_turn_count": 0, "child_artifact_count": 3,
    }]
    if purpose == "verify_assessment":
        referents.append({
            "kind": "evidence_snapshot", "status": "frozen", "source_turn_id": "turn-assessment",
        })
    contract = NS(quality_evidence_query=query, referents=tuple(referents))
    projections = [
        {
            "kind": "quality_exchange_coverage",
            "coverage": coverage,
            "complete_exchange_pairs": 1 if coverage == "complete" else 0,
        },
        {
            "kind": "quality_exchange",
            "artifact_id": "turn-1",
            "request": "Return exactly JSON.",
            "assistant": "Plain text response.",
        },
    ]
    return NS(
        intent=NS(turn_contract=contract), runtime=NS(source_projections=projections),
        conversation=(
            {"user": "Audit yourself.", "assistant": NO_ISSUE, "artifact_id": "turn-assessment"},
            {"user": "Is that accurate?", "assistant": "", "artifact_id": "turn-current"},
        ),
    )


def test_gate_is_inactive_outside_original_self_assessment():
    ordinary = NS(intent=NS(turn_contract=NS(quality_evidence_query=None)), runtime=NS(source_projections=[]))
    assert QualityEvidenceCompletionHook(lambda: ordinary).validate_completion("anything", "end_turn") is None
    assert QualityEvidenceCompletionHook(lambda: _state()).validate_completion("anything", "max_tokens") is None


def test_verification_requires_source_exact_prior_claim_blocks():
    state = _state(purpose="verify_assessment")
    hook = QualityEvidenceCompletionHook(lambda: state)
    normalized = hook.validate_completion(_audited(NO_ISSUE), "end_turn")
    assert "replacement" in normalized
    assert "same frozen projection of 1 exact sealed request/response pair" in normalized["replacement"]
    assert "every underlying answer was universally correct" in normalized["replacement"]

    # The general source-exact path remains active for a prior assessment that did not use the canonical verdict.
    state.conversation[0]["assistant"] = "The file write failed before execution."
    hook.reset_for_turn()
    fabricated = (
        "Verification item\n"
        f"Prior claim exact: {json.dumps('I referenced 4 lifecycle operations.')}\n"
        "Verdict: supported\n"
        f"Evidence: {json.dumps('The frozen aggregate reports 4 operations.')}\n\n"
        "Overall: supported"
    )
    rejected = hook.validate_completion(fabricated, "end_turn")
    assert rejected["continue"] and "not verbatim" in rejected["feedback"]
    hook.reset_for_turn()
    state.conversation[0]["assistant"] = NO_ISSUE
    valid = _audited(NO_ISSUE)
    assert "replacement" in hook.validate_completion(valid, "end_turn")
    natural = _audited(
        f'My prior response said "{NO_ISSUE}" That exact evidence-scoped claim is supported by the frozen '
        "quality projection; it is not a universal-correctness claim."
    )
    assert "replacement" in hook.validate_completion(natural, "end_turn")


def test_unavailable_frozen_baseline_can_only_return_not_verifiable():
    state = _state(purpose="verify_assessment")
    state.intent.turn_contract.referents = tuple(
        dict(item, status="unavailable") if item.get("kind") == "evidence_snapshot" else item
        for item in state.intent.turn_contract.referents
    )
    hook = QualityEvidenceCompletionHook(lambda: state)
    result = hook.validate_completion("It was supported.", "end_turn")
    assert "replacement" in result and "cannot verify that claim now" in result["replacement"]


def test_no_issue_verdict_is_terminal_and_speculative_tail_is_removed():
    hook = QualityEvidenceCompletionHook(lambda: _state())
    normalized = hook.validate_completion(
        _audited("Receipts show three successful children.\n\n" + NO_ISSUE), "end_turn",
    )
    assert normalized["replacement"] == NO_ISSUE
    result = hook.validate_completion(
        _audited(NO_ISSUE + " That said, I should have been more proactive and less terse."), "end_turn",
    )
    assert result["replacement"] == NO_ISSUE
    assert "proactive" not in result["replacement"]

    quoted = (
        "Execution lifecycle: all three child agents succeeded.\n\n"
        "The gate says *\"No supported response-quality issue is evidenced.\"*\n\n"
        "If you disagree, point me at a turn."
    )
    sanitized = hook.validate_completion(_audited(quoted), "end_turn")["replacement"]
    assert sanitized == NO_ISSUE
    assert sanitized.count('"') == 0 and sanitized.count("*") == 0

    paraphrase = "The six exact pairs show that no four-field mismatch is supported by any pair."
    normalized = hook.validate_completion(_audited(paraphrase), "end_turn")["replacement"]
    assert normalized == NO_ISSUE


def test_host_measured_explicit_constraint_mismatch_overrides_false_clean_verdict_and_freezes():
    state = _state(prospective=True)
    state.runtime.source_projections[1]["request"] = "What else can you help with, briefly?"
    long_response = " ".join(["word"] * 161)
    state.runtime.source_projections[1]["assistant"] = long_response
    state.runtime.source_projections[1]["deterministic_mismatches"] = [{
        "kind": "deterministic_quality_mismatch",
        "constraint": "brief_response",
        "category": "violated explicit format or constraint",
        "requested_exact": state.runtime.source_projections[1]["request"],
        "produced_exact": long_response[:360],
        "measurements": {"words": 161, "brief_word_ceiling": 80},
        "explanation": "the request explicitly says 'briefly', but the response has 161 words",
    }]
    leading = QualityEvidenceCompletionHook(lambda: state).validate_completion(
        _audited(NO_ISSUE), "end_turn",
    )["replacement"]
    assert "Observed issue" in leading
    assert "violated explicit format or constraint" in leading
    assert "briefly" in leading and "161 words" in leading
    assert NO_ISSUE not in leading
    assert "honor explicit brevity requests" in leading
    assert "keep self-assessments source-scoped" not in leading

    verify = _state(purpose="verify_assessment")
    verify.runtime.source_projections = state.runtime.source_projections
    verify.conversation[0]["assistant"] = leading
    result = QualityEvidenceCompletionHook(lambda: verify).validate_completion(
        "I think it was fine.", "end_turn",
    )
    assert result["exclusive"]
    assert "supports the prior assessment" in result["replacement"]
    assert "Observed issue" in result["replacement"] and "161 words" in result["replacement"]


def test_no_issue_requires_private_source_complete_audit_certificate():
    hook = QualityEvidenceCompletionHook(lambda: _state())
    rejected = hook.validate_completion(NO_ISSUE, "end_turn")
    assert rejected["continue"] and "source-complete audit omitted" in rejected["feedback"]
    accepted = hook.validate_completion(_audited(NO_ISSUE), "end_turn")
    assert accepted["replacement"] == NO_ISSUE
    assert "Quality check" not in accepted["replacement"]
    compact = "Quality check — artifacts/turn-1.md — no admitted mismatch\n\n" + NO_ISSUE
    accepted = QualityEvidenceCompletionHook(lambda: _state()).validate_completion(compact, "end_turn")
    assert accepted["replacement"] == NO_ISSUE
    ordinal_table = (
        "| Pair | Verdict |\n|---|---|\n"
        "| turn-1 (review) | No admitted mismatch |\n\n" + NO_ISSUE
    )
    accepted = QualityEvidenceCompletionHook(lambda: _state()).validate_completion(ordinal_table, "end_turn")
    assert accepted["replacement"] == NO_ISSUE
    attested = (
        "After auditing all 1 exact request/response pairs, no admitted mismatch was found.\n\n" + NO_ISSUE
    )
    accepted = QualityEvidenceCompletionHook(lambda: _state()).validate_completion(attested, "end_turn")
    assert accepted["replacement"] == NO_ISSUE
    attestation_only = "After auditing all 1 exact request/response pairs, no admitted mismatch was found."
    accepted = QualityEvidenceCompletionHook(lambda: _state()).validate_completion(
        attestation_only, "end_turn",
    )
    assert accepted["replacement"] == NO_ISSUE
    natural_attestation = "I've audited all 1 exact request/response pairs. " + NO_ISSUE
    accepted = QualityEvidenceCompletionHook(lambda: _state()).validate_completion(
        natural_attestation, "end_turn",
    )
    assert accepted["replacement"] == NO_ISSUE
    wrong_count = attested.replace("all 1 exact", "all 2 exact")
    rejected = QualityEvidenceCompletionHook(lambda: _state()).validate_completion(wrong_count, "end_turn")
    assert rejected["continue"] and "attestation says 2 exact pair" in rejected["feedback"]


def test_universal_correctness_is_retried_then_fails_safe_to_inconclusive_verdict():
    hook = QualityEvidenceCompletionHook(lambda: _state(), max_revisions=1)
    bad = "Every explicit request was fulfilled correctly. " + NO_ISSUE
    first = hook.validate_completion(bad, "end_turn")
    assert first["continue"] and "universal correctness" in first["feedback"]
    second = hook.validate_completion(bad, "end_turn")
    assert "replacement" in second
    assert "no observed-quality verdict is asserted" in second["replacement"]
    assert "every response" not in second["replacement"].lower()
    hook.reset_for_turn()
    assert hook.validate_completion(bad, "end_turn")["continue"], "retry budget must reset per user turn"
    hook.reset_for_turn()
    followups = "All follow-up questions were answered correctly. " + NO_ISSUE
    result = hook.validate_completion(followups, "end_turn")
    assert result["continue"] and "universal correctness" in result["feedback"]
    normalized = QualityEvidenceCompletionHook(lambda: _state()).validate_completion(
        _audited("Every explicit request was fulfilled correctly. " + NO_ISSUE), "end_turn",
    )
    assert normalized["replacement"] == NO_ISSUE
    assert "Every explicit request" not in normalized["replacement"]


def test_model_lifecycle_preamble_is_replaced_by_host_canonical_projection():
    hook = QualityEvidenceCompletionHook(lambda: _state())
    bad = (
        "Execution: 8 operations started, 8 settled, 8 succeeded, 0 failed. "
        "Turn-level: 5 completed, 0 warnings. Quality: 1 exact request/response pair.\n\n" + NO_ISSUE
    )
    result = hook.validate_completion(_audited(bad), "end_turn")
    assert result["replacement"] == NO_ISSUE
    assert "5 completed" not in result["replacement"]
    hook.reset_for_turn()
    good = (
        "Execution: 8 operations started, 8 settled, 8 succeeded, 0 failed. "
        "Turn-level: 6 completed, 0 warnings. Quality: 1 exact request/response pair.\n\n" + NO_ISSUE
    )
    assert hook.validate_completion(_audited(good), "end_turn")["replacement"] == NO_ISSUE
    ordinal = _audited("Turn 1 completed the requested review.\n\n" + NO_ISSUE)
    assert QualityEvidenceCompletionHook(lambda: _state()).validate_completion(
        ordinal, "end_turn",
    )["replacement"] == NO_ISSUE


def test_prospective_advice_requires_permission_and_literal_separation():
    allowed = _state(prospective=True)
    candidate = _audited(
        NO_ISSUE + "\n\n## Prospective (not observed)\n\nIn a future run, I could summarize sooner."
    )
    normalized = QualityEvidenceCompletionHook(lambda: allowed).validate_completion(candidate, "end_turn")
    assert normalized["replacement"] == (
        NO_ISSUE + "\n\nProspective (not observed)\n\nIn a future run, I could summarize sooner."
    )

    retrospective = _audited(
        NO_ISSUE + "\n\nProspective (not observed)\n\nI should have summarized sooner."
    )
    normalized = QualityEvidenceCompletionHook(lambda: allowed).validate_completion(retrospective, "end_turn")
    assert "replacement" in normalized and "I should have" not in normalized["replacement"]

    false_past_premise = _audited(
        NO_ISSUE + "\n\nProspective (not observed)\n\nIn future I could reuse what I learned in turn 1."
    )
    normalized = QualityEvidenceCompletionHook(lambda: allowed).validate_completion(
        false_past_premise, "end_turn",
    )
    assert "replacement" in normalized and "turn 1" not in normalized["replacement"]

    disguised_retrospective = _audited(
        NO_ISSUE + "\n\nProspective (not observed)\n\n**\n\n"
        "In future, add more context. The core task was executed correctly — bugs reported accurately. "
        "No failures, incorrect answers, or violations occurred."
    )
    normalized = QualityEvidenceCompletionHook(lambda: allowed).validate_completion(
        disguised_retrospective, "end_turn",
    )
    assert "replacement" in normalized
    assert "core task was executed correctly" not in normalized["replacement"]
    assert "incorrect answers" not in normalized["replacement"]

    orphan_markdown = _audited(
        NO_ISSUE + "\n\nProspective (not observed)\n\n**\n\nIn future, summarize sooner."
    )
    normalized = QualityEvidenceCompletionHook(lambda: allowed).validate_completion(
        orphan_markdown, "end_turn",
    )
    assert normalized["replacement"].endswith("Prospective (not observed)\n\nIn future, summarize sooner.")
    assert "\n\n**\n\n" not in normalized["replacement"]

    denied = QualityEvidenceCompletionHook(lambda: _state()).validate_completion(candidate, "end_turn")
    assert denied["replacement"] == NO_ISSUE, "unrequested advice after the terminal verdict is removed"


def test_complete_clean_audit_missing_prospective_tail_is_completed_without_another_retry():
    state = _state(prospective=True)
    hook = QualityEvidenceCompletionHook(lambda: state)
    attestation_only = "After auditing all 1 exact request/response pairs, no admitted mismatch was found."
    result = hook.validate_completion(attestation_only, "end_turn")
    assert "replacement" in result and not result.get("continue")
    assert result["replacement"].startswith(NO_ISSUE)
    assert "Prospective (not observed)" in result["replacement"]
    assert "In future" in result["replacement"]

    exact_verdict = _audited(NO_ISSUE)
    result = QualityEvidenceCompletionHook(lambda: state).validate_completion(exact_verdict, "end_turn")
    assert "replacement" in result and not result.get("continue")
    assert "Prospective (not observed)" in result["replacement"]

    attestation_after_verdict = (
        NO_ISSUE + "\n\nAfter auditing all 1 exact request/response pairs, no admitted mismatch was found."
    )
    result = QualityEvidenceCompletionHook(lambda: state).validate_completion(
        attestation_after_verdict, "end_turn",
    )
    assert "replacement" in result and not result.get("continue")
    assert result["replacement"].startswith(NO_ISSUE)

    retrospective_tail = (
        attestation_after_verdict
        + "\n\nProspective (not observed)\n\nBased on the session evidence, every operation succeeded and nothing failed."
    )
    result = QualityEvidenceCompletionHook(lambda: state).validate_completion(
        retrospective_tail, "end_turn",
    )
    assert "replacement" in result
    assert "every operation succeeded" not in result["replacement"]


def test_aggregate_audit_attestation_covers_clean_remainder_but_not_an_invalid_issue():
    state = _state(prospective=True)
    state.runtime.source_projections.append({
        "kind": "quality_exchange", "artifact_id": "turn-2",
        "request": "Use YAML.", "assistant": "Plain text response.",
    })
    state.runtime.source_projections[0]["complete_exchange_pairs"] = 2
    issue = (
        "I audited all 2 exact request/response pairs.\n\n"
        "Observed issue\n"
        "Source: artifacts/turn-1.md\n"
        f"Requested exact: {json.dumps('Return exactly JSON.')}\n"
        f"Produced exact: {json.dumps('Plain text response.')}\n"
        "Mismatch: violated explicit format or constraint — the response was not JSON."
    )
    result = QualityEvidenceCompletionHook(lambda: state).validate_completion(issue, "end_turn")
    assert "replacement" in result and "Observed issue" in result["replacement"]
    assert "Prospective (not observed)" in result["replacement"]

    wrong = issue.replace("artifacts/turn-1.md", "artifacts/turn-9.md")
    rejected = QualityEvidenceCompletionHook(lambda: state).validate_completion(wrong, "end_turn")
    assert rejected["continue"] and "outside" in rejected["feedback"]

    contradictory = issue.replace(
        "I audited all 2 exact request/response pairs.",
        "After auditing all 2 exact request/response pairs, no admitted mismatch was found.",
    )
    rejected = QualityEvidenceCompletionHook(lambda: state).validate_completion(
        contradictory, "end_turn",
    )
    assert rejected["continue"] and "declares every pair clean" in rejected["feedback"]


def test_epistemic_caution_is_not_a_past_failure_without_an_exact_confidence_requirement():
    state = _state()
    state.runtime.source_projections[1]["request"] = "Report the top bug."
    state.runtime.source_projections[1]["assistant"] = (
        "Conditional (not established by this observation alone): potential SQL injection."
    )
    issue = (
        "I audited all 1 exact request/response pairs.\n\n"
        "Observed issue\n"
        "Source: artifacts/turn-1.md\n"
        f"Requested exact: {json.dumps('Report the top bug.')}\n"
        "Produced exact: "
        f"{json.dumps('Conditional (not established by this observation alone): potential SQL injection.')}\n"
        "Mismatch: contradicted explicit requirement — the conditional qualifier undercut the finding and "
        "made it sound uncertain."
    )
    rejected = QualityEvidenceCompletionHook(lambda: state).validate_completion(issue, "end_turn")
    assert rejected["continue"]
    assert "confidence level or forbid qualification" in rejected["feedback"]


def test_complete_coverage_fallback_is_protocol_disposition_and_challenge_is_host_scoped():
    state = _state(prospective=True)
    hook = QualityEvidenceCompletionHook(lambda: state, max_revisions=1)
    bad = "I made several mistakes, probably."
    assert hook.validate_completion(bad, "end_turn")["continue"]
    fallback = hook.validate_completion(bad, "end_turn")["replacement"]
    assert "The draft did not produce a protocol-valid source audit" in fallback
    assert "couldn't complete" not in fallback

    verification = _state(purpose="verify_assessment")
    verification.conversation[0]["assistant"] = fallback
    result = QualityEvidenceCompletionHook(lambda: verification).validate_completion(
        "All 999 pairs were universally correct.", "end_turn",
    )
    assert "replacement" in result and "complete and contains 1 exact sealed request/response pair" in result["replacement"]
    assert "does not verify the underlying response quality" in result["replacement"]
    assert "999" not in result["replacement"] and "universally correct" not in result["replacement"]

    no_future = _state(prospective=False)
    no_future_hook = QualityEvidenceCompletionHook(lambda: no_future, max_revisions=1)
    assert no_future_hook.validate_completion(bad, "end_turn")["continue"]
    no_future_fallback = no_future_hook.validate_completion(bad, "end_turn")["replacement"]
    verification.conversation[0]["assistant"] = no_future_fallback
    result = QualityEvidenceCompletionHook(lambda: verification).validate_completion(
        "It had a prospective section.", "end_turn",
    )
    assert "replacement" in result
    assert "prospective section" not in result["replacement"]


def test_no_issue_output_uses_host_owned_execution_preamble_with_separate_scope():
    state = _state()
    state.intent.turn_contract.evidence_query = NS(source="execution_receipt")
    result = QualityEvidenceCompletionHook(lambda: state).validate_completion(_audited(NO_ISSUE), "end_turn")
    assert result["replacement"] == (
        "Execution evidence (canonical receipts): no adverse operation or non-clean turn is recorded.\n\n"
        + NO_ISSUE
    )
    aggregate = state.intent.turn_contract.referents[0]
    aggregate["counts"]["failed"] = 1
    aggregate["turn_counts"]["completed_with_warnings"] = 1
    result = QualityEvidenceCompletionHook(lambda: state).validate_completion(_audited(NO_ISSUE), "end_turn")
    assert "failed operations=1" in result["replacement"]
    assert "completed-with-warnings turns=1" in result["replacement"]
    assert "quality" not in result["replacement"].split("\n\n", 1)[0].lower()


def test_observed_issue_requires_real_source_and_verbatim_pair_bytes():
    block = (
        "Observed issue\n"
        "Source: artifacts/turn-1.md\n"
        f"Requested exact: {json.dumps('Return exactly JSON.')}\n"
        f"Produced exact: {json.dumps('Plain text response.')}\n"
        "Mismatch: violated explicit format or constraint — the response was not JSON."
    )
    hook = QualityEvidenceCompletionHook(lambda: _state())
    assert hook.validate_completion(block, "end_turn") is None

    wrong_source = block.replace("turn-1", "turn-2")
    assert hook.validate_completion(wrong_source, "end_turn")["continue"]
    hook.reset_for_turn()
    wrong_quote = block.replace("Return exactly JSON.", "Return YAML.")
    result = hook.validate_completion(wrong_quote, "end_turn")
    assert result["continue"] and "not present" in result["feedback"]


def test_unsupported_factual_issue_requires_exact_admitted_grounding():
    state = _state()
    pair = state.runtime.source_projections[1]
    pair["grounding_artifacts"] = [{
        "artifact_id": "child-1",
        "source_text": "The actual identifier is user, not us.",
    }]
    base = (
        "Observed issue\n"
        "Source: artifacts/turn-1.md\n"
        f"Requested exact: {json.dumps('Return exactly JSON.')}\n"
        f"Produced exact: {json.dumps('Plain text response.')}\n"
    )
    hook = QualityEvidenceCompletionHook(lambda: state)
    missing = base + "Mismatch: unsupported factual claim — the identifier is wrong."
    rejected = hook.validate_completion(missing, "end_turn")
    assert rejected["continue"] and "requires Grounding source" in rejected["feedback"]
    hook.reset_for_turn()
    grounded = (
        base
        + "Grounding source: artifacts/child-1.md\n"
        + f"Grounding exact: {json.dumps('identifier is user')}\n"
        + "Mismatch: unsupported factual claim — the produced claim conflicts with the sealed child source."
    )
    assert hook.validate_completion(grounded, "end_turn") is None


def test_frozen_challenge_reaudits_pairs_and_can_contradict_prior_no_issue_verdict():
    state = _state(purpose="verify_assessment")
    issue = (
        "Observed issue\n"
        "Source: artifacts/turn-1.md\n"
        f"Requested exact: {json.dumps('Return exactly JSON.')}\n"
        f"Produced exact: {json.dumps('Plain text response.')}\n"
        "Mismatch: violated explicit format or constraint — plain text is not JSON."
    )
    result = QualityEvidenceCompletionHook(lambda: state).validate_completion(issue, "end_turn")
    assert "replacement" in result
    assert "contradicts that claim" in result["replacement"]
    assert "Observed issue" in result["replacement"]


def test_incomplete_coverage_requires_uncertainty_not_a_clean_no_issue_claim():
    state = _state(coverage="partial")
    rejected = QualityEvidenceCompletionHook(lambda: state).validate_completion(NO_ISSUE, "end_turn")
    assert rejected["continue"] and "coverage is incomplete" in rejected["feedback"]
    assert QualityEvidenceCompletionHook(lambda: state).validate_completion(UNCERTAIN, "end_turn") is None


class _Response:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []
        self.finish_reason = "stop"
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1}


class _ScriptLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.seen_schemas = []

    def complete(self, messages, schemas):
        self.seen.append([dict(message) for message in messages])
        self.seen_schemas.append(list(schemas))
        return _Response(self.replies.pop(0))


class _Tools:
    def schemas(self):
        return []


class _Oracle:
    def __init__(self):
        self.calls = 0

    def verify(self):
        self.calls += 1
        return True, ""


def test_loop_keeps_rejected_draft_private_and_runs_other_completion_hooks_only_after_acceptance():
    bad = "Every response was fully correct. " + NO_ISSUE
    llm = _ScriptLLM([bad, _audited(NO_ISSUE)])
    oracle = _Oracle()
    events = []
    hooks = CompositeHooks(
        QualityEvidenceCompletionHook(lambda: _state()),
        OracleHook(oracle, lambda _output: None),
    )
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "Audit your response quality."}],
        llm=llm, tools=_Tools(), dispatch=events.append, hooks=hooks, max_steps=5,
    )
    assert result.stop_reason == "end_turn"
    assert oracle.calls == 1, "the oracle must not run on rejected prose"
    finals = [event.content for event in events if isinstance(event, AssistantText) and event.final]
    assert finals == [NO_ISSUE]
    assert all(bad not in text for text in finals)
    assert any("host rejected the unpublished self-assessment" in str(message.get("content", "")).lower()
               for message in llm.seen[1])
    correction = next(message for message in llm.seen[1]
                      if "host rejected the unpublished self-assessment"
                      in str(message.get("content", "")).lower())
    assert correction["role"] == "system", "host correction must not masquerade as a user instruction"


def test_retry_cannot_leak_unpublished_host_feedback_as_a_user_correction():
    state = _state(prospective=True)
    hook = QualityEvidenceCompletionHook(lambda: state, max_revisions=2)
    first = hook.validate_completion("Every response was fully correct. " + NO_ISSUE, "end_turn")
    assert first["continue"]
    leaked = (
        "You're right. I converted the verdict into a blanket claim.\n\n" + NO_ISSUE
        + "\n\nProspective (not observed)\n\nIn future, I will keep quality claims source-scoped."
    )
    second = hook.validate_completion(leaked, "end_turn")
    assert second["continue"] and "user never saw" in second["feedback"]
    clean = _audited(
        NO_ISSUE + "\n\nProspective (not observed)\n\nIn future, I will keep quality claims source-scoped."
    )
    accepted = hook.validate_completion(clean, "end_turn")
    assert accepted["replacement"].startswith(NO_ISSUE)
    assert "Quality check" not in accepted["replacement"]


def test_loop_prose_only_revision_removes_tool_schemas_but_accounts_as_a_normal_step():
    class ProseOnlyHook(Hooks):
        def __init__(self):
            self.calls = 0
        def reset_for_turn(self):
            self.calls = 0
        def validate_completion(self, _candidate, _stop):
            self.calls += 1
            if self.calls == 1:
                return {
                    "continue": True, "prose_only": True, "feedback_role": "system",
                    "feedback": "Reconcile the prose from existing evidence only.",
                }
            return None

    class Toolful:
        def schemas(self):
            return [{"type": "function", "function": {"name": "danger", "parameters": {}}}]

    llm = _ScriptLLM(["draft", "reconciled answer"])
    events = []
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "answer"}],
        llm=llm, tools=Toolful(), dispatch=events.append,
        hooks=CompositeHooks(ProseOnlyHook()), max_steps=3,
    )
    assert result.stop_reason == "end_turn" and result.steps == 2
    assert llm.seen_schemas[0] and llm.seen_schemas[1] == []
    finals = [event.content for event in events if isinstance(event, AssistantText) and event.final]
    assert finals == ["reconciled answer"]


def test_frozen_quality_verification_hides_tools_before_the_first_model_call():
    class Toolful:
        def schemas(self):
            return [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

    state = _state(purpose="verify_assessment")
    hook = QualityEvidenceCompletionHook(lambda: state)
    assert hook.prepare_tool_schemas(Toolful().schemas()) == []
    assert QualityEvidenceCompletionHook(lambda: _state()).prepare_tool_schemas(
        Toolful().schemas(),
    ) is None

    llm = _ScriptLLM([_audited(NO_ISSUE)])
    result = run_turn(
        build_slice=lambda: [{"role": "user", "content": "Verify the frozen prior assessment."}],
        llm=llm, tools=Toolful(), dispatch=lambda _event: None,
        hooks=CompositeHooks(hook), max_steps=2,
    )
    assert result.stop_reason == "end_turn"
    assert llm.seen_schemas == [[]], "moving-history tools must never be offered on a frozen recheck"


def test_legacy_duck_hook_need_not_implement_new_validation_seam():
    class LegacyHook:
        pass

    assert CompositeHooks(LegacyHook()).validate_completion("answer", "end_turn") is None
    assert CompositeHooks(LegacyHook()).prepare_tool_schemas([{"name": "read"}]) is None


def main():
    checks = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    failed = 0
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {check.__name__}: {error!r}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
