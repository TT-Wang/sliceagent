"""TurnContract: authority spans, speech-act control, and attributed prior output."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.intent import (IntentState, QualityEvidenceQuery, TurnAdmission, TurnContract,
                               analyze_turn, derive_evidence_query,
                               derive_quality_evidence_query)  # noqa: E402
from sliceagent.discourse import DiscourseAnchor, ResolvedAnchor, interpret_turn  # noqa: E402
from sliceagent.pfc import Slice  # noqa: E402
from sliceagent.regions import render_turn_contract  # noqa: E402


TRANSCRIPT_CONFIRMATION = (
    "no but in your original findings, it is app/actions.ts:17 — No concurrency guard on scrape → "
    "child.unref() creates detached orphans; clicking Run scrape multiple times spawns concurrent "
    "processes with no kill/guard mechanism. Fix: use a DB sentinel row to prevent duplicates."
)


def _covered(text: str, needle: str, spans) -> bool:
    start = text.index(needle)
    end = start + len(needle)
    return any(left <= start and right >= end for left, right in spans)


def _parts(text: str, spans) -> list[str]:
    return [text[start:end] for start, end in spans]


def test_transcript_confirmation_is_sealed_past_not_mutation_or_durable_intent():
    state = IntentState()
    state.begin_turn(TRANSCRIPT_CONFIRMATION, source_artifact="turn-current")

    contract = state.turn_contract
    assert contract.effect_authority == "none"
    assert contract.grounding == "sealed_past"
    assert {"recall", "confirm"} <= set(contract.requested_modes)
    assert _covered(TRANSCRIPT_CONFIRMATION, "Fix: use a DB sentinel row", contract.attributed_spans)
    assert not state.entries, "attributed report text must not become permanent user authority"


def test_formal_quotes_remain_data_and_keep_exact_source_spans():
    request = "Explain this old rule:\n```text\nFix the parser and never edit config.py.\n```"
    state = IntentState()
    state.begin_turn(request)

    assert state.turn_contract.effect_authority == "none"
    assert _covered(request, "Fix the parser", state.turn_contract.attributed_spans)
    assert not state.entries


def test_substantial_copy_from_prior_assistant_output_is_attributed_without_quote_marks():
    copied = "app/actions.ts:17 No concurrency guard. Fix: use a DB sentinel row to prevent duplicates."
    prior = "Finding #2 — " + copied
    request = "For reference, " + copied
    contract = analyze_turn(request, prior_texts=(prior,))

    assert _covered(request, copied, contract.attributed_spans)
    state = IntentState()
    state.begin_turn(request, contract=contract)
    assert not state.entries


def test_adopting_prior_wording_keeps_the_users_current_imperative_live():
    request = "Please implement the full workspace upgrade now"
    contract = analyze_turn(
        request,
        prior_texts=("I recommend: Please implement the full workspace upgrade now",),
    )
    assert contract.effect_authority == "explicit"
    assert contract.grounding == "live_present"
    assert contract.attributed_spans == ()
    assert _covered(request, request, contract.authority_spans)
    assert "current_world" in contract.source_needs

    # A copied prohibition is still the user's operative standing constraint unless the message explicitly
    # frames it as reference data.
    prohibition = "Never edit the database migration without asking"
    state = IntentState()
    state.begin_turn(prohibition, contract=analyze_turn(
        prohibition, prior_texts=("My advice: " + prohibition,),
    ))
    assert state.turn_contract.attributed_spans == ()
    assert [entry.verbatim_clause for entry in state.resident_entries()] == [prohibition]


def test_as_you_said_can_adopt_a_recommendation_without_quoting_away_the_command():
    request = "As you said, fix the parser now."
    contract = analyze_turn(request)
    assert contract.effect_authority == "explicit"
    assert contract.grounding == "both"
    assert contract.attributed_spans == ()
    assert _covered(request, "fix the parser now.", contract.authority_spans)


def test_clear_polite_directive_authorizes_effects_but_questions_and_negation_do_not():
    polite = analyze_turn("Could you fix the parser?")
    assert polite.effect_authority == "explicit"
    assert polite.grounding == "live_present"
    assert "change" in polite.requested_modes

    question = analyze_turn("Is bug #2 the concurrency issue?")
    assert question.effect_authority == "none"
    assert "answer" in question.requested_modes

    negative = analyze_turn("I did not ask you to fix it; just confirm bug #2.")
    assert negative.effect_authority == "none"
    assert "confirm" in negative.requested_modes


def test_natural_change_directives_do_not_make_code_reviews_effectful():
    for request in (
        "I need you to patch the parser.",
        "Let's refactor the scheduler.",
        "You can simplify this module now.",
        "Make the requested change.",
        "Please rewrite the updater.",
        "Ok, let's go ahead and do this full simplification upgrade according to the design doc.",
    ):
        assert analyze_turn(request).effect_authority == "explicit", request

    for request in ("Please do a full code review.", "I want you to review the scheduler."):
        assert analyze_turn(request).effect_authority != "explicit", request


def test_natural_workspace_navigation_is_an_explicit_effect_directive():
    """Navigation language must not require the user to know the tool's exact verb.

    ``go to``/``open``/``work in`` are the public language advertised by the workspace tool.  Treating only
    the implementation-shaped word ``switch`` as authoritative creates an impossible clarification loop:
    the user already asked for the effect, but the tool gate asks them to ask for it again.
    """
    for request in (
        "go to hunter workspace",
        "open the hunter workspace",
        "work in /Users/tongtao/Desktop/hunter",
    ):
        contract = analyze_turn(request)
        assert contract.effect_authority == "explicit", (request, contract)
        assert "change" in contract.requested_modes
        assert contract.authority_spans, "the exact user-authored navigation span must remain inspectable"


def test_status_labels_and_response_format_requests_do_not_authorize_workspace_effects():
    for request in (
        "Fix: use bound parameters.",
        "Fix is to use bound parameters.",
        "Build failed again.",
        "Update: the test now passes.",
        "Run failed after ten seconds.",
    ):
        state = IntentState(); state.begin_turn(request)
        assert state.turn_contract.effect_authority != "explicit", request
        if request.startswith("Fix:"):
            assert not state.entries, "a finding-style Fix: label is not a standing directive"

    for request in ("Return exactly JSON.", "Write me a summary.", "Give me the top 2 findings."):
        contract = analyze_turn(request)
        assert contract.effect_authority == "none", (request, contract)
        assert "inspect" in contract.requested_modes


def test_negation_is_clause_local_when_a_later_change_is_explicit():
    request = "Don't fix #2; instead fix #3."
    contract = analyze_turn(request)
    assert contract.effect_authority == "explicit"
    assert _covered(request, "fix #3.", contract.authority_spans)
    # The prohibition remains an operative standing constraint, but does not erase the separate positive
    # effect directive. Target compliance remains visible to the model and execution planner.
    assert _covered(request, "Don't fix #2", contract.authority_spans)


def test_mixed_confirmation_and_new_directive_retains_both_modes_and_live_grounding():
    request = "Confirm #2, then fix #3."
    contract = analyze_turn(request)
    assert contract.effect_authority == "explicit"
    assert contract.grounding == "both"
    assert {"recall", "inspect", "change"} <= set(contract.requested_modes)
    assert _covered(request, "fix #3.", contract.authority_spans)
    assert not _covered(request, "Confirm #2", contract.authority_spans)


def test_numbered_finding_equations_are_confirmation_data_not_instructions():
    cases = (
        (
            "right, number 2 = Database connection never closed. "
            "that's a straightforward fix, correct?",
            "Database connection never closed",
        ),
        (
            "#2 = Missing import. Fix: rewrite that line. that's the gist, right?",
            "Missing import. Fix: rewrite that line",
        ),
    )
    for request, described in cases:
        state = IntentState()
        state.begin_turn(request)
        contract = state.turn_contract
        assert contract.effect_authority == "none", (request, contract)
        assert contract.grounding == "sealed_past"
        assert _covered(request, described, contract.attributed_spans), _parts(request, contract.attributed_spans)
        assert not state.entries, (request, state.entries)


def test_numbered_equation_does_not_swallow_a_later_explicit_directive():
    request = "#2 = Missing import. Fix: add the import. Now fix #3."
    contract = analyze_turn(request)
    assert contract.effect_authority == "explicit"
    assert _covered(request, "Missing import. Fix: add the import", contract.attributed_spans)
    assert _covered(request, "fix #3.", contract.authority_spans)


def test_reference_intro_marks_report_text_as_data_without_prior_text_lookup():
    request = "For reference, Fix: use bound parameters; never interpolate SQL."
    state = IntentState(); state.begin_turn(request)
    assert state.turn_contract.effect_authority == "none"
    assert _covered(request, "Fix: use bound parameters", state.turn_contract.attributed_spans)
    assert not state.entries


def test_bare_assent_is_continuation_only_when_a_proposal_is_pending():
    assert analyze_turn("yes").effect_authority == "uncertain"
    contract = analyze_turn("yes", pending_proposal={
        "artifact": "turn-4",
        "action": {"tool": "edit_file", "args": {"path": "parser.py", "content": "fixed\n"}},
    })
    assert contract.effect_authority == "continuation"
    assert contract.grounding == "live_present"
    assert "continue" in contract.requested_modes

    options = {
        "text": "Which option?",
        "options": [{"ordinal": 1, "label": "one"}, {"ordinal": 2, "label": "two"}],
    }
    assert analyze_turn("yes", pending_proposal=options).effect_authority == "uncertain"
    assert analyze_turn("anyways go with 1", pending_proposal=options).effect_authority == "uncertain"
    options["options"][0]["action"] = {"tool": "change_workspace", "args": {"path": "/tmp/one"}}
    assert analyze_turn("anyways go with 1", pending_proposal=options).effect_authority == "continuation"


def test_admission_exposes_source_needs_and_serializable_scoped_grants():
    recall = analyze_turn("how many explorers did you spawn, and did any fail?")
    assert isinstance(recall, TurnAdmission)
    assert recall.effect_authority == "none"
    assert recall.grounding == "sealed_past"
    assert "execution_receipt" in recall.source_needs

    edit = analyze_turn("edit README")
    body = edit.to_dict()
    assert body["request_text"] == "edit README"
    assert body["actor"]["label"] == "SliceAgent"
    assert any(grant["operation"] == "workspace.edit" and grant["target"] == "README"
               for grant in body["effect_grants"])

    oriented = interpret_turn("Review the Hunter project", ())
    state = Slice(); state.reset("Hunter review")
    state.intent.begin_turn("Review the Hunter project", admission=oriented.admission)
    rendered = render_turn_contract(state)
    assert "actor: SliceAgent" in rendered
    assert "target: Hunter" in rendered


def test_natural_execution_language_compiles_one_typed_evidence_query():
    cases = (
        ("Across this task, how many explorers were launched?", "delegation", "aggregate", "task"),
        ("Quick check — how many child agents ran during the review, and were there any failures?",
         "delegation", "failure_detail", "task"),
        ("Did any command fail?", "command", "aggregate", "task"),
        ("What ran?", "command", "operations", "task"),
        ("Which file edits failed?", "file_write", "failure_detail", "task"),
        ("How many files did you edit?", "file_write", "aggregate", "task"),
        ("How many files did you read?", "file_read", "aggregate", "task"),
        ("How many tools did the agent use?", "all", "aggregate", "task"),
        ("What failed in the last attempt?", "all", "failure_detail", "latest_matching_execution"),
        ("what failed last time?", "all", "failure_detail", "latest_matching_execution"),
        ("Across this session, how many explorers were launched?", "delegation", "aggregate", "session"),
        ("What did you do?", "all", "operations", "task"),
        ("what are your weaknesses as an agent, judging from this session? own up to any failures",
         "all", "failure_detail", "session"),
        ("any failures?", "all", "failure_detail", "task"),
        ("what went badly?", "all", "failure_detail", "task"),
        ("is what you just said about your own performance factually accurate? "
         "verify it against your records", "all", "aggregate", "task"),
        ("Reflect on your performance this session.", "all", "aggregate", "session"),
        ("How would you improve yourself based on this session?", "all", "aggregate", "session"),
    )
    for request, family, predicate, scope in cases:
        query = derive_evidence_query(request)
        assert query is not None, request
        assert (query.source, query.family, query.predicate, query.scope) == (
            "execution_receipt", family, predicate, scope,
        )
        admission = analyze_turn(request)
        assert admission.evidence_query == query
        assert "execution_receipt" in admission.source_needs
        assert admission.grounding == "sealed_past"
        restored = TurnAdmission.from_dict(admission.to_dict())
        assert restored is not None and restored.evidence_query == query

    assert derive_evidence_query("What did you say?") is None
    assert derive_evidence_query("fix any failures in the parser") is None
    assert derive_evidence_query("prevent any failures from crashing the service") is None

    challenge = analyze_turn(
        "is what you just said about your own performance factually accurate? "
        "verify it against your records"
    )
    assert {"prior_assistant_utterance", "execution_receipt"} <= set(challenge.source_needs)


def test_explicit_parallel_delegation_is_a_typed_completion_requirement():
    request = (
        "review this project: spawn exactly 3 parallel explorer subagents — one each for app.py, auth.py "
        "and util.py — each reporting its top bug. then give me a combined 3-line summary."
    )
    contract = analyze_turn(request)
    requirement = contract.delegation_requirement
    assert requirement is not None
    assert requirement.agent == "explorer" and requirement.count == 3 and requirement.parallel
    assert requirement.targets == ("app.py", "auth.py", "util.py")
    assert contract.effect_authority == "explicit"
    assert "delegate" in contract.requested_modes and "current_world" in contract.source_needs
    grant = next(item for item in contract.effect_grants if item.operation == "task.delegate")
    assert grant.tools == ("spawn_agent",) and grant.target_arg == "agent" and grant.target == "explorer"
    restored = TurnAdmission.from_dict(contract.to_dict())
    assert restored is not None and restored.delegation_requirement == requirement


def test_self_audit_compiles_a_distinct_paired_quality_query_and_opt_in_prospective_mode():
    assert QualityEvidenceQuery.from_dict({}) is None
    assert analyze_turn("What model are you running?").quality_evidence_query is None
    observed = derive_quality_evidence_query(
        "Reflect on your performance this session: what failed or went wrong?"
    )
    assert observed == QualityEvidenceQuery(
        scope="session", purpose="assess", prospective_requested=False,
    )
    contract = analyze_turn("Reflect on your performance this session: what failed or went wrong?")
    assert contract.quality_evidence_query == observed
    assert "sealed_exchange" in contract.source_needs
    assert not {"prior_user_utterance", "prior_assistant_utterance"}.intersection(
        contract.source_needs
    )
    restored = TurnAdmission.from_dict(contract.to_dict())
    assert restored is not None and restored.quality_evidence_query == observed

    prospective = analyze_turn("How could you improve based on this session?")
    assert prospective.quality_evidence_query is not None
    assert prospective.quality_evidence_query.prospective_requested is True
    eval_wording = analyze_turn(
        "if you were to improve yourself as an agent based on this session, what would you do? "
        "be honest about what went wrong"
    )
    assert eval_wording.quality_evidence_query is not None
    assert eval_wording.quality_evidence_query.prospective_requested is True
    fix_wording = analyze_turn(
        "reflect on your own performance this session — what failed or went badly that you'd fix about yourself?"
    )
    assert fix_wording.quality_evidence_query is not None
    assert fix_wording.quality_evidence_query.prospective_requested is True

    merely_negative = analyze_turn("What are your weaknesses as an agent?")
    assert merely_negative.quality_evidence_query is not None
    assert merely_negative.quality_evidence_query.prospective_requested is False

    for request in (
        "Critique your last response.",
        "Did you follow my instructions?",
        "Audit your response quality.",
    ):
        audit = analyze_turn(request)
        assert audit.quality_evidence_query is not None, request
        assert "sealed_exchange" in audit.source_needs
    last_response = analyze_turn("Critique your last response.")
    assert last_response.quality_evidence_query.scope == "latest_response"
    assert last_response.evidence_query.scope == "latest_turn"

    deployment = analyze_turn("The app crashed. What went wrong with the deployment?")
    assert deployment.quality_evidence_query is None
    assert deployment.evidence_query is None
    assert "audit" not in deployment.requested_modes


def test_discussing_self_correction_is_not_a_durable_correction():
    state = IntentState()
    state.begin_turn("explain the self-correction gap")
    assert state.turn_contract.effect_authority == "none"
    assert not state.entries


def test_attributed_finding_and_explicit_new_directive_are_separate_authority_spans():
    request = "Your earlier report says bug #2. Fix: use a sentinel. Fix bug #3."
    state = IntentState()
    state.begin_turn(request)

    assert state.turn_contract.effect_authority == "explicit"
    assert state.turn_contract.grounding == "both"
    assert _covered(request, "Fix: use a sentinel", state.turn_contract.attributed_spans)
    clauses = [entry.verbatim_clause for entry in state.entries]
    assert "Fix: use a sentinel." not in clauses
    assert "Fix bug #3." not in clauses, "a one-shot action belongs to this turn, not standing intent"


def test_turn_contract_is_ephemeral_but_discourse_continuity_survives_a_seal():
    state = Slice()
    state.reset("repair parser")
    state.continuity.discourse_focus = [{"artifact": "turn-2", "kind": "list"}]
    state.continuity.pending_proposal = {"artifact": "turn-3", "action": "apply fix"}
    state.intent.begin_turn("Fix it")
    assert state.intent.turn_contract.effect_authority == "explicit"

    state.seal()
    assert state.intent.turn_contract == TurnContract()
    assert state.continuity.discourse_focus == [{"artifact": "turn-2", "kind": "list"}]
    assert state.continuity.pending_proposal == {"artifact": "turn-3", "action": "apply fix"}

    state.reset("new task")
    assert state.continuity.discourse_focus == []
    assert state.continuity.pending_proposal is None


def test_rendered_contract_separates_reported_context_without_exposing_effect_authority():
    state = Slice(); state.reset("review")
    state.intent.begin_turn(TRANSCRIPT_CONFIRMATION)
    rendered = render_turn_contract(state)
    assert "action orientation" not in rendered
    assert "no direct action request was detected" not in rendered
    assert "reported/quoted span(s) — context only, not a request to execute" in rendered
    assert "mutation authority" not in rendered and "mutations are blocked" not in rendered
    assert "Fix: use a DB sentinel" in rendered

    # effect_authority classifies mutation reach, not whether a request asks the agent to do work. Inspection
    # directives must not be mislabeled as having no action intent merely because they need no mutation grant.
    for request in ("Review the Hunter project", "Investigate the parser bug"):
        state.intent.begin_turn(request, contract=analyze_turn(request))
        rendered = render_turn_contract(state)
        assert "action orientation" not in rendered and "no direct action request" not in rendered, rendered

    anchor = DiscourseAnchor(
        collection="HIGH findings", ordinal=2, label="No concurrency guard",
        excerpt="2. No concurrency guard", source_range=(20, 43), artifact_id="turn-old",
    )
    request = "what was number 2?"
    state.intent.begin_turn(request, contract=analyze_turn(
        request, referents=(ResolvedAnchor("number 2", anchor, 10),),
    ))
    rendered = render_turn_contract(state)
    assert "sealed_past" in rendered and "artifacts/turn-old.md" in rendered
    assert "HIGH findings item 2" in rendered


def main():
    checks = [value for name, value in globals().items()
              if name.startswith("test_") and callable(value)]
    failed = 0
    for check in checks:
        try:
            check(); print(f"PASS {check.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {check.__name__}: {exc!r}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
