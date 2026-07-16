"""Addressable sealed-output references. Deterministic; no model or network."""
import copy
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.discourse import (  # noqa: E402
    anchors_from_artifacts,
    extract_addressable_anchors,
    extract_pending_proposal,
    interpret_turn,
    make_evidence_snapshot,
    resolve_discourse_references,
)
from sliceagent.events import AssistantText  # noqa: E402
from sliceagent.intent import analyze_turn  # noqa: E402
from sliceagent.pfc import Slice, record_user, slice_sink  # noqa: E402
from sliceagent.regions import render_evidence_detail, render_evidence_result  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def cross_turn_nav_disambiguation_reply_authorizes_the_named_target():
    # Assistant asks "which one should I navigate to — loom-app or loom-engine?"; the turn ends; the user's
    # next-turn reply naming one option must authorize navigating there (continuation), not fail-closed.
    asst = ("There are two loom directories on your Desktop — loom-app and loom-engine. "
            "Which one should I navigate to?")
    prop = extract_pending_proposal(asst)
    assert prop is not None and prop.get("nav_targets") == ["loom-app", "loom-engine"]

    picked = analyze_turn("loom-app", pending_proposal=prop)
    assert picked.effect_authority == "continuation"
    grant = picked.effect_grants[0]
    assert grant.operation == "workspace.navigate" and grant.target == "loom-app"
    # "loom app" (spoken form) selects the same target; "loom-engine" selects the OTHER.
    assert analyze_turn("loom app", pending_proposal=prop).effect_grants[0].target == "loom-app"
    assert analyze_turn("loom-engine", pending_proposal=prop).effect_grants[0].target == "loom-engine"
    # Bare "yes" is genuinely ambiguous (which one?) → no authority; a sentence is not a bare selection.
    assert analyze_turn("yes", pending_proposal=prop).effect_authority == "uncertain"
    assert analyze_turn("the one in src please", pending_proposal=prop).effect_authority == "uncertain"
    # A non-navigation offer is never turned into nav_targets.
    other = extract_pending_proposal("Would you like me to fix the parser (config-v2 or config-v3)?")
    assert (other or {}).get("nav_targets") is None

    # "go to" phrasing (not just navigate/switch) is recognized as a nav-disambiguation question.
    goq = extract_pending_proposal(
        "I see loom-app and loom-engine on your Desktop. Which one would you like to go to?")
    assert analyze_turn("loom app", pending_proposal=goq).effect_grants[0].target == "loom-app"

    # A single-target navigation OFFER + bare "yes" continues that one navigation.
    offer = extract_pending_proposal("Do you want me to switch to loom-app?")
    assert (offer or {}).get("nav_targets") == ["loom-app"]
    yes = analyze_turn("yes", pending_proposal=offer)
    assert yes.effect_authority == "continuation" and yes.effect_grants[0].target == "loom-app"
    for variant in ("Would you like me to go to loom-app?", "Shall I cd into loom-app?"):
        assert analyze_turn("yes", pending_proposal=extract_pending_proposal(variant)) \
            .effect_grants[0].target == "loom-app"
    # But a bare "yes" to a MULTI-target offer stays ambiguous (name one to disambiguate).
    multi = extract_pending_proposal("Which do you want to switch to — loom-app or loom-engine?")
    assert analyze_turn("yes", pending_proposal=multi).effect_authority == "uncertain"


@check
def extracts_numbered_collections_with_full_item_ranges():
    text = """## HIGH findings
1. Env leak
   More detail about one.
2. No concurrency guard
   Fix: use a sentinel.

## Subagents
1. sub-1 — Scripts and config
2. sub-2 — App routes
"""
    anchors = extract_addressable_anchors(text)
    assert [(a.collection, a.ordinal) for a in anchors] == [
        ("HIGH findings", 1), ("HIGH findings", 2), ("Subagents", 1), ("Subagents", 2),
    ]
    assert "Fix: use a sentinel" in anchors[1].excerpt
    start, end = anchors[1].source_range
    assert text[start:end].strip() == anchors[1].excerpt
    assert anchors[2].stable_id == "sub-1"


@check
def nested_numbered_details_do_not_steal_top_level_ordinals():
    text = """## Issues
1. First issue
   2. nested remediation detail
2. Second issue
"""
    anchors = extract_addressable_anchors(text)
    assert [(anchor.ordinal, anchor.label) for anchor in anchors] == [
        (1, "First issue"), (2, "Second issue"),
    ]
    assert "nested remediation detail" in anchors[0].excerpt
    resolved = resolve_discourse_references("fix number 2", anchors)
    assert not resolved.ambiguous and resolved.resolved[0].anchor.label == "Second issue"

    duplicate = extract_addressable_anchors("1. first\n1. another first\n")
    ambiguous = resolve_discourse_references("number 1", duplicate)
    assert ambiguous.ambiguous and not ambiguous.resolved, \
        "same artifact/collection/ordinal at different ranges must not collapse into one identity"


def _artifact(artifact_id, timestamp, assistant, *, task_id="task"):
    return SimpleNamespace(
        id=artifact_id, kind="turn", timestamp=timestamp, task_id=task_id, summary=assistant,
        structured_body={
            "assistant": assistant,
            "anchors": [a.to_dict() for a in extract_addressable_anchors(assistant)],
        },
    )


@check
def collection_words_select_the_correct_number_two():
    artifacts = (
        _artifact("turn-review", "2026-01-01T00:00:00Z", """## HIGH severity
1. Env leak
2. No concurrency guard
3. Hardcoded PII
"""),
        _artifact("turn-agents", "2026-01-02T00:00:00Z", """## Subagent work
1. sub-1 — Scripts and config
2. sub-2 — App routes
"""),
    )
    anchors = anchors_from_artifacts(artifacts, task_id="task")
    high = resolve_discourse_references("give me number 2 high severity issue", anchors)
    assert not high.ambiguous and high.resolved
    assert "concurrency" in high.resolved[0].anchor.label.casefold()
    agents = resolve_discourse_references("summarize the second subagent", anchors)
    assert not agents.ambiguous and agents.resolved[0].anchor.stable_id == "sub-2"


@check
def top_two_is_cardinality_not_ordinal_reference():
    anchors = extract_addressable_anchors("1. one\n2. two\n3. three\n")
    result = resolve_discourse_references("give me the top 2", anchors)
    assert not result.resolved and not result.ambiguous


@check
def first_subagent_uses_launch_metadata_not_completion_order():
    def child(artifact_id, launch_ordinal, title):
        return SimpleNamespace(
            id=artifact_id, kind="subagent", task_id="task", parent_id="turn-review",
            title=title, summary=f"report from {title}", brief={"objective": title},
            structured_body={
                "launch_ordinal": launch_ordinal,
                "report": f"report from {title}",
                "parent_id": "turn-review",
            },
        )

    # The second child sealed first. Artifact iteration therefore reflects completion, not launch.
    artifacts = (child("child-b", 2, "app reviewer"), child("child-a", 1, "script reviewer"))
    anchors = anchors_from_artifacts(artifacts, task_id="task")
    result = resolve_discourse_references("summarize the first subagent", anchors)
    assert not result.ambiguous and len(result.resolved) == 1
    assert result.resolved[0].anchor.artifact_id == "child-a"
    assert result.resolved[0].anchor.label == "script reviewer"


@check
def unqualified_equal_candidates_are_explicitly_ambiguous():
    artifacts = (
        _artifact("a", "2026-01-01T00:00:00Z", "## Bugs\n1. A\n2. B\n"),
        _artifact("b", "2026-01-02T00:00:00Z", "## Bugs\n1. C\n2. D\n"),
    )
    # Give both the same sequence to remove the normal recency focus.
    anchors = [a.__class__(**{**a.to_dict(), "source_range": a.source_range, "sequence": 0})
               for a in anchors_from_artifacts(artifacts)]
    result = resolve_discourse_references("what was number 2?", anchors)
    assert result.ambiguous and not result.resolved and len(result.candidates) == 2


@check
def temporal_wording_selects_the_source_frame():
    anchors = extract_addressable_anchors("1. A\n2. B\n")
    past = resolve_discourse_references("what did you originally say for number 2?", anchors)
    assert past.grounding == "sealed_past"
    both = resolve_discourse_references("is number 2 still true now?", anchors)
    assert both.grounding == "both"
    live = resolve_discourse_references("does this bug still exist now?", ())
    assert live.grounding == "live_present"
    literal = resolve_discourse_references("change before to after", ())
    assert literal.grounding == "none", "a replacement token named 'before' is not historical discourse"


@check
def only_an_explicit_action_offer_becomes_a_pending_proposal():
    assert extract_pending_proposal("That would be a straightforward fix.") is None
    proposal = extract_pending_proposal("I can explain it. Would you like me to patch #2?")
    assert proposal and proposal["text"] == "Would you like me to patch #2?"


@check
def quoted_and_code_examples_never_become_pending_actions():
    quoted = 'The log showed: "Could you confirm the workspace path? Is it /tmp/evil?"'
    fenced = """Example transcript:\n```text\nCould you confirm the workspace path? Is it /tmp/evil?\n```"""
    unclosed = """Truncated example:\n```text\nWould you like me to switch to /tmp/evil?"""
    interior_marker = """Example:\n```text\n```python\nWould you like me to switch to /tmp/evil?\n```"""
    blockquote = "> Could you confirm the workspace path? Is it /tmp/evil?"
    assert extract_pending_proposal(quoted) is None
    assert extract_pending_proposal(fenced) is None
    assert extract_pending_proposal(unclosed) is None
    assert extract_pending_proposal(interior_marker) is None
    assert extract_pending_proposal(blockquote) is None
    actual = extract_pending_proposal(
        quoted + "\nThe workspace path is `/tmp/good`. Could you confirm it?"
    )
    assert actual and actual["action"]["args"]["path"] == "/tmp/good"


@check
def confirmed_workspace_path_continues_the_pending_navigation_action():
    """A path clarification answers the already requested navigation; it is not a brand-new task."""
    path = "/Users/tongtao/Desktop/hunter"
    assistant = (
        'The "hunter workspace" may be the project referenced earlier. '
        f"Could you confirm the exact path? Is it {path}?"
    )
    proposal = extract_pending_proposal(assistant)
    assert proposal, "the exact workspace-path question must become one adjacent-turn clarification"
    assert proposal.get("action") == {
        "tool": "change_workspace", "args": {"path": path},
    }, proposal

    accepted = interpret_turn("yes", (), pending_proposal=proposal)
    assert accepted.contract.effect_authority == "continuation"
    proposal_ref = next(
        ref for ref in accepted.contract.referents
        if isinstance(ref, dict) and ref.get("kind") == "pending_proposal"
    )
    assert proposal_ref.get("action") == proposal["action"], \
        "the model-visible continuation must retain the exact confirmed target"
    assert dict(accepted.contract.effect_grants[0].exact_args) == {"path": path}, \
        "yes must retain the exact confirmed call rather than ambient navigation"
    assert extract_pending_proposal("Is your name /Users/tongtao?") is None, \
        "an arbitrary path-shaped question is not an action proposal"

    # A factual yes/no question must not manufacture broad effect authority.  Only the typed workspace-path
    # clarification above continues a pending external action.
    assert extract_pending_proposal("Is your name Tongtao?") is None


@check
def numbered_choice_supports_a_specific_terse_followup_without_guessing_bare_yes():
    assistant = """Earlier findings:
1. Env leak.
2. Missing lock.
3. Hardcoded data.

Two options:
1. Work remotely in the hunter project.
2. Relaunch from the hunter project.

Which would you prefer?
"""
    proposal = extract_pending_proposal(assistant)
    assert proposal and [option["ordinal"] for option in proposal["options"]] == [1, 2]

    selected = interpret_turn("anyways go with 1", (), pending_proposal=proposal)
    assert selected.contract.effect_authority == "uncertain", \
        "selection continuity must not claim effect authority without a typed action"
    ref = next(ref for ref in selected.contract.referents
               if isinstance(ref, dict) and ref.get("kind") == "pending_proposal")
    assert ref["selected_option"]["ordinal"] == 1
    assert "Work remotely" in ref["selected_option"]["label"]

    ambiguous = interpret_turn("yes", (), pending_proposal=proposal)
    assert ambiguous.contract.effect_authority == "uncertain"


@check
def terminal_assistant_output_owns_one_immediate_proposal():
    state = Slice(); state.reset("review")
    record_user(state, "is #2 correct?")
    sink = slice_sink(state)
    sink(AssistantText("Would you like me to patch #2?"))
    assert state.continuity.pending_proposal
    record_user(state, "not yet")
    sink(AssistantText("Understood."))
    assert state.continuity.pending_proposal is None


@check
def one_interpretation_feeds_referents_authority_and_focus():
    artifacts = (
        _artifact("review", "2026-01-01T00:00:00Z", "## HIGH severity\n1. Env\n2. Concurrency\n"),
        _artifact("agents", "2026-01-02T00:00:00Z", "## Subagents\n1. sub-1 — Scripts\n2. sub-2 — App\n"),
    )
    result = interpret_turn(
        "give me number 2 high severity finding", artifacts, task_id="task",
    )
    assert result.contract.effect_authority == "none"
    assert result.contract.referents[0].anchor.label == "Concurrency"
    assert result.focus[0]["artifact_id"] == "review"
    assert result.referenced_artifact_ids == ("review",)

    ambiguous = interpret_turn("fix number 2", (
        _artifact("a", "2026-01-01T00:00:00Z", "## Bugs\n1. A\n2. B\n"),
        _artifact("b", "2026-01-01T00:00:00Z", "## Bugs\n1. C\n2. D\n"),
    ))
    assert ambiguous.ambiguous and ambiguous.contract.effect_authority == "uncertain"
    assert "clarify_reference" in ambiguous.contract.requested_modes

    proposal = {"text": "Would you like me to patch #2?", "source_range": [0, 31]}
    accepted = interpret_turn("yes", (), pending_proposal=proposal)
    assert accepted.contract.effect_authority == "uncertain"
    assert not accepted.contract.effect_grants


@check
def stable_ids_and_immediate_repair_questions_keep_user_visible_focus():
    artifacts = (
        _artifact("review", "2026-01-01T00:00:00Z", "## Findings\n1. Env\n2. Concurrency\n"),
        _artifact("agents", "2026-01-02T00:00:00Z", "## Subagents\n1. sub-1 — Scripts\n2. sub-2 — App\n"),
    )
    corrected = interpret_turn("no, you told me sub-1 was scripts", artifacts, task_id="task")
    assert corrected.focus[0]["stable_id"] == "sub-1"
    repair = interpret_turn(
        "why did you make that mistake again?", artifacts, task_id="task", focus=corrected.focus,
    )
    assert repair.contract.grounding == "sealed_past"
    assert repair.contract.referents[0].anchor.stable_id == "sub-1"


@check
def project_subject_is_inherited_and_explicit_i_mean_repairs_it():
    oriented = interpret_turn("Review the Hunter project", ())
    assert oriented.admission.actor.label == "SliceAgent"
    assert oriented.admission.target.label == "Hunter"
    assert oriented.focus[-1]["kind"] == "subject_focus"

    followup = interpret_turn(
        "if you were to improve, what would you do", (), focus=oriented.focus,
    )
    assert followup.admission.effect_authority == "none", "a recommendation is not change authority"
    assert "recommend" in followup.admission.requested_modes
    assert followup.admission.actor.label == "SliceAgent"
    assert followup.admission.target.label == "Hunter"
    assert followup.admission.target.source == "focus"

    repaired = interpret_turn("I mean Atlas project", (), focus=followup.focus)
    assert repaired.admission.target.label == "Atlas"
    assert repaired.admission.target.source == "repair"
    assert repaired.admission.focus_repairs[0].field == "target"
    assert repaired.focus[-1]["entity"]["label"] == "Atlas"

    meta = interpret_turn("explain the self-correction gap", (), focus=repaired.focus)
    assert not meta.admission.focus_repairs
    assert meta.focus[-1]["entity"]["label"] == "Atlas", \
        "mentioning correction as a topic must not become a focus repair"


@check
def execution_questions_refault_canonical_receipts_not_assistant_counts():
    request = "How many explorer subagents did you spawn during that review, and did any of them fail?"

    def receipt_artifact(artifact_id, timestamp, operations):
        return SimpleNamespace(
            id=artifact_id, kind="turn", timestamp=timestamp, task_id="task", summary="",
            structured_body={
                "assistant": "I may have described a different count here.",
                "turn_receipt": {
                    "turn_id": artifact_id,
                    "disposition": "completed_with_warnings",
                    "operations": operations,
                },
            },
        )

    rejected = {
        "invocation_id": "spawn-denied", "name": "spawn_agent",
        "args": {"agent": "explorer", "task": "inspect config"},
        "requested": True, "rejected_before_execution": True,
        "execution_started": False, "settled": True, "disposition": "rejected",
    }
    succeeded = {
        "invocation_id": "spawn-ok", "name": "spawn_agent",
        "args": {"agent": "explorer", "task": "inspect runtime"},
        "requested": True, "rejected_before_execution": False,
        "execution_started": True, "settled": True, "disposition": "succeeded",
    }
    unrelated = {
        "invocation_id": "read", "name": "read_file", "args": {"path": "README.md"},
        "requested": True, "rejected_before_execution": False,
        "execution_started": True, "settled": True, "disposition": "succeeded",
    }
    artifacts = (
        _artifact("prose-only", "2026-01-01T00:00:00Z", "I spawned 12 explorers and all succeeded."),
        receipt_artifact("turn-denied", "2026-01-02T00:00:00Z", [rejected]),
        receipt_artifact("turn-ok", "2026-01-03T00:00:00Z", [succeeded, unrelated]),
    )
    preview = interpret_turn(request, artifacts, task_id="task")
    assert "execution_receipt" in preview.admission.source_needs
    receipts = [ref for ref in preview.admission.referents
                if isinstance(ref, dict) and ref.get("kind") == "execution_receipt"]
    assert not receipts, "a count query projects one bounded aggregate, never one row per turn"
    aggregate = next(ref for ref in preview.admission.referents
                     if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate")
    assert aggregate["counts"]["requested"] == 2
    assert aggregate["counts"]["rejected_before_execution"] == 1
    assert aggregate["counts"]["execution_started"] == 1
    assert aggregate["counts"]["succeeded"] == 1

    state = SimpleNamespace(intent=SimpleNamespace(
        current_request=request, turn_contract=preview.admission,
    ))
    rendered = render_evidence_result(state)
    assert "rejected-before-execution=1" in rendered
    assert "execution-started=1" in rendered
    assert "I spawned 12" not in rendered
    assert "coverage: PARTIAL" in rendered and "lower bound" in rendered
    assert "verify against OPEN FILES" not in rendered, \
        "live files cannot verify canonical past lifecycle"


@check
def missing_execution_receipts_are_an_explicit_evidence_gap():
    request = "How many subagents did you spawn?"
    preview = interpret_turn(
        request,
        (_artifact("prose-only", "2026-01-01T00:00:00Z", "I spawned 12."),),
        task_id="task",
    )
    assert any(isinstance(ref, dict) and ref.get("kind") == "execution_receipt_absence"
               for ref in preview.admission.referents)
    state = SimpleNamespace(intent=SimpleNamespace(
        current_request=request, turn_contract=preview.admission,
    ))
    rendered = render_evidence_result(state)
    assert "coverage: UNAVAILABLE" in rendered
    assert "evidence gap, not evidence of success or failure" in rendered


@check
def natural_execution_recall_phrases_select_receipts_without_a_model_router():
    requests = (
        "how many tools did you use?",
        "what actually happened?",
        "did the command run?",
        "how many reports were sealed?",
        "how many children returned?",
        "what did you do?",
        "why did any explorer fail?",
        "Were any explorers successful?",
        "How many reports actually came back?",
        "Did that command ever start?",
        "What failed in the last attempt?",
        "Across this task, how many agents were launched?",
    )
    for request in requests:
        preview = interpret_turn(request, (), task_id="task")
        assert "execution_receipt" in preview.admission.source_needs, request


@check
def generic_adjacent_verification_inherits_the_exact_typed_evidence_query():
    first = interpret_turn("Which explorer failed and why?", (), task_id="task")
    original = first.admission.evidence_query
    assert original is not None
    assert (original.family, original.predicate, original.scope) == (
        "delegation", "failure_detail", "task",
    )

    followup = interpret_turn(
        "Verify that against your records.", (), task_id="task",
        previous_evidence_query=original,
    )
    assert followup.admission.evidence_query == original
    assert "execution_receipt" in followup.admission.source_needs

    natural_challenge = interpret_turn(
        "Is what you just said accurate? Verify it against your records.", (), task_id="task",
        previous_evidence_query=original,
    )
    assert natural_challenge.admission.evidence_query == original, \
        "a conversational prefix must not erase the adjacent typed family/predicate/scope"

    unanchored = interpret_turn("Verify that against your records.", (), task_id="task")
    assert unanchored.admission.evidence_query != original, \
        "without an adjacent typed selector the host must not guess which family/predicate was meant"


@check
def adjacent_verification_reuses_the_frozen_projection_instead_of_counting_its_own_answer():
    def sealed_turn(index, *, request=None, assistant=None, operations=()):
        artifact_id = f"turn-{index}"
        return SimpleNamespace(
            id=artifact_id, kind="turn", timestamp=f"2026-07-11T00:{index:02d}:00Z",
            task_id="task", session_id="session", status="end_turn", brief={}, summary="",
            structured_body={
                "request": request or f"request {index}",
                "assistant": assistant or f"answer {index}",
                "turn_receipt": {
                    "turn_id": artifact_id, "disposition": "completed", "warnings": [],
                    "operations": list(operations),
                },
            },
        )

    operation = {
        "invocation_id": "spawn-1", "name": "spawn_agent", "args": {"agent": "explorer"},
        "requested": True, "rejected_before_execution": False, "execution_started": True,
        "settled": True, "disposition": "succeeded",
    }
    prior = tuple(sealed_turn(index, operations=(operation,) if index == 1 else ())
                  for index in range(1, 7))
    leading_request = "Reflect on your own performance this session: what went wrong?"
    leading = interpret_turn(
        leading_request, prior, task_id="task", session_id="session",
    )
    before = next(ref for ref in leading.admission.referents
                  if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate")
    assert before["receipt_count"] == 6
    assert len([item for item in leading.projections if item.get("kind") == "quality_exchange"]) == 6

    leading_artifact = sealed_turn(
        7, request=leading_request,
        assistant="There were six completed prior turns and no recorded failures.",
    )
    snapshot = make_evidence_snapshot(
        leading.admission, leading.projections, leading_artifact.id,
        snapshot_basis=leading.snapshot_basis, source_generation=7,
    )
    snapshot_text = json.dumps(snapshot)
    assert "answer 1" not in snapshot_text and leading_request not in snapshot_text
    assert "execution_referents" not in snapshot and "quality_projections" not in snapshot
    challenge = interpret_turn(
        "Is what you just said accurate? Verify it against your records.",
        (*prior, leading_artifact), task_id="task", session_id="session",
        previous_evidence_snapshot=snapshot, current_generation=7,
    )
    after = next(ref for ref in challenge.admission.referents
                 if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate")
    assert challenge.admission.evidence_continuation
    assert after["receipt_count"] == 6
    assert after["projection_sha256"] == before["projection_sha256"]
    assert len([item for item in challenge.projections if item.get("kind") == "quality_exchange"]) == 6
    assert any(isinstance(ref, dict) and ref.get("kind") == "evidence_snapshot"
               and ref.get("status") == "frozen" for ref in challenge.admission.referents)
    rendered = render_evidence_result(SimpleNamespace(intent=SimpleNamespace(
        turn_contract=challenge.admission,
    )))
    assert "FROZEN at the prior response cutoff" in rendered
    assert 'read_file("artifacts/index.md")' not in rendered

    for wording in (
        "Are you sure?",
        "Is that accurate?",
        "Were there really zero failures? Check the receipts.",
        "Check your claim against the receipts.",
        "Double-check your last answer.",
    ):
        variant = interpret_turn(
            wording, (*prior, leading_artifact), task_id="task", session_id="session",
            previous_evidence_snapshot=snapshot, current_generation=7,
        )
        assert variant.admission.evidence_continuation, wording
        assert variant.admission.effect_authority == "none", wording
        assert any(isinstance(ref, dict) and ref.get("kind") == "evidence_snapshot"
                   and ref.get("status") == "frozen" for ref in variant.admission.referents), wording

    fresh = interpret_turn(
        leading_request, (*prior, leading_artifact), task_id="task", session_id="session",
    )
    fresh_aggregate = next(ref for ref in fresh.admission.referents
                           if isinstance(ref, dict)
                           and ref.get("kind") == "execution_receipt_aggregate")
    assert fresh_aggregate["receipt_count"] == 7, "only an adjacent verification is frozen"

    tampered = dict(snapshot); tampered["source_turn_id"] = "not-the-adjacent-turn"
    unavailable = interpret_turn(
        "Verify that against your records.", (*prior, leading_artifact),
        task_id="task", session_id="session", previous_evidence_snapshot=tampered,
        current_generation=7,
    )
    assert not any(isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate"
                   for ref in unavailable.admission.referents)
    absence = next(ref for ref in unavailable.admission.referents
                   if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_absence")
    assert "frozen adjacent evidence snapshot unavailable" in absence["reason"]

    forged = copy.deepcopy(snapshot)
    forged["basis"]["execution_signature"]["receipt_count"] = 999
    rejected = interpret_turn(
        "Are you sure?", (*prior, leading_artifact), task_id="task", session_id="session",
        previous_evidence_snapshot=forged, current_generation=7,
    )
    assert not any(isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate"
                   for ref in rejected.admission.referents), "snapshot payload claims are re-derived, not trusted"

    after_other_task = interpret_turn(
        "Are you sure?", (*prior, leading_artifact), task_id="task", session_id="session",
        previous_evidence_snapshot=snapshot, current_generation=8,
    )
    assert any(isinstance(ref, dict) and ref.get("kind") == "evidence_snapshot"
               and ref.get("status") == "unavailable"
               for ref in after_other_task.admission.referents), \
        "a task-local snapshot is not conversationally adjacent after another global session turn"


@check
def typed_latest_matching_execution_scope_selects_only_that_receipt():
    def failed_turn(artifact_id, timestamp, invocation_id, reason):
        return SimpleNamespace(
            id=artifact_id, kind="turn", timestamp=timestamp, task_id="task", summary="",
            structured_body={"turn_receipt": {
                "turn_id": artifact_id, "disposition": "completed_with_warnings",
                "operations": [{
                    "invocation_id": invocation_id, "name": "run_command", "args": {},
                    "requested": True, "rejected_before_execution": False,
                    "execution_started": True, "settled": True, "disposition": "failed",
                    "outcome_text": reason,
                }],
            }},
        )

    request = "What failed in the last attempt?"
    preview = interpret_turn(request, (
        failed_turn("turn-old", "2026-01-01T00:00:00Z", "old-call", "old failure"),
        failed_turn("turn-new", "2026-01-02T00:00:00Z", "new-call", "new failure"),
    ), task_id="task")
    query = preview.admission.evidence_query
    assert query is not None and query.scope == "latest_matching_execution"
    assert query.predicate == "failure_detail"
    receipts = [ref for ref in preview.admission.referents
                if isinstance(ref, dict) and ref.get("kind") == "execution_receipt"]
    assert [ref["artifact_id"] for ref in receipts] == ["turn-new"]
    state = SimpleNamespace(intent=SimpleNamespace(
        current_request=request, turn_contract=preview.admission,
    ))
    rendered = render_evidence_result(state) + "\n" + render_evidence_detail(state)
    assert "new failure" in rendered and "new-call" in rendered
    assert "old failure" not in rendered and "old-call" not in rendered


@check
def latest_execution_skips_conversational_filler_but_preserves_newer_coverage_gaps():
    command = SimpleNamespace(
        id="turn-command", kind="turn", timestamp="2026-01-01T00:00:00Z", task_id="task", summary="",
        structured_body={"turn_receipt": {
            "turn_id": "turn-command", "disposition": "completed", "warnings": [],
            "operations": [{
                "invocation_id": "cmd", "name": "run_command", "args": {}, "requested": True,
                "rejected_before_execution": False, "execution_started": True, "settled": True,
                "disposition": "succeeded",
            }],
        }},
    )
    filler = SimpleNamespace(
        id="turn-filler", kind="turn", timestamp="2026-01-02T00:00:00Z", task_id="task", summary="",
        structured_body={"turn_receipt": {
            "turn_id": "turn-filler", "disposition": "completed", "warnings": [], "operations": [],
        }},
    )
    request = "What actually ran in the last execution?"
    preview = interpret_turn(request, (command, filler), task_id="task")
    assert preview.admission.evidence_query.scope == "latest_matching_execution"
    details = [ref for ref in preview.admission.referents
               if isinstance(ref, dict) and ref.get("kind") == "execution_receipt"]
    assert [ref["artifact_id"] for ref in details] == ["turn-command"]

    newer_gap = SimpleNamespace(
        id="turn-gap", kind="turn", timestamp="2026-01-03T00:00:00Z", task_id="task", summary="",
        structured_body={},
    )
    uncertain = interpret_turn(request, (command, filler, newer_gap), task_id="task")
    coverage = next(ref for ref in uncertain.admission.referents
                    if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_coverage")
    assert coverage["coverage"] == "partial" and coverage["missing_receipt_count"] == 1


@check
def session_scope_reads_same_session_across_tasks_without_cross_session_bleed():
    def turn(identity, *, task, session):
        return SimpleNamespace(
            id=identity, kind="turn", timestamp=identity, task_id=task, session_id=session, summary="",
            structured_body={"turn_receipt": {
                "turn_id": identity, "disposition": "completed", "warnings": [],
                "operations": [{
                    "invocation_id": identity + "-spawn", "name": "spawn_agent",
                    "args": {"agent": "explorer"}, "requested": True,
                    "rejected_before_execution": False, "execution_started": True, "settled": True,
                    "disposition": "succeeded",
                }],
            }},
        )

    artifacts = (
        turn("2026-01-01", task="task-a", session="session-one"),
        turn("2026-01-02", task="task-b", session="session-one"),
        turn("2026-01-03", task="task-c", session="session-two"),
    )
    preview = interpret_turn(
        "Across this session, how many explorers were launched?", artifacts,
        task_id="task-a", session_id="session-one",
    )
    aggregate = next(ref for ref in preview.admission.referents
                     if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate")
    assert aggregate["query"]["scope"] == "session"
    assert aggregate["counts"]["requested"] == 2
    assert aggregate["counts"]["succeeded"] == 2


@check
def latest_receiptless_turn_reports_scope_specific_partial_coverage():
    receipt_turn = SimpleNamespace(
        id="turn-old", kind="turn", timestamp="2026-01-01T00:00:00Z", task_id="task", summary="",
        structured_body={"turn_receipt": {
            "turn_id": "turn-old", "disposition": "completed_with_warnings",
            "operations": [{
                "invocation_id": "old-call", "name": "run_command", "args": {},
                "requested": True, "rejected_before_execution": False,
                "execution_started": True, "settled": True, "disposition": "failed",
                "outcome_text": "old failure",
            }],
        }},
    )
    receiptless_turn = SimpleNamespace(
        id="turn-new-no-receipt", kind="turn", timestamp="2026-01-02T00:00:00Z",
        task_id="task", summary="", structured_body={"assistant": "No canonical lifecycle data."},
    )
    request = "What failed in the latest turn?"
    preview = interpret_turn(request, (receipt_turn, receiptless_turn), task_id="task")

    absence = next(
        ref for ref in preview.admission.referents
        if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_absence"
    )
    coverage = next(
        ref for ref in preview.admission.referents
        if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_coverage"
    )
    assert absence["query"]["scope"] == "latest_turn"
    assert coverage["kind"] == "execution_receipt_coverage"
    assert coverage["candidate_turn_artifacts"] == 1
    assert coverage["receipt_bearing"] == 0
    assert coverage["coverage"] == "partial"
    assert coverage["scope"] == "latest_turn"
    assert coverage["missing_receipt_count"] == 1
    assert coverage["missing_receipt_sample"] == ["turn-new-no-receipt"]
    assert coverage["source_index_handle"] == "artifacts/index.md"

    state = SimpleNamespace(intent=SimpleNamespace(
        current_request=request, turn_contract=preview.admission,
    ))
    rendered = render_evidence_result(state)
    assert "coverage: UNAVAILABLE for latest_turn" in rendered
    assert "1 missing receipt" in rendered
    assert "evidence gap, not evidence of success or failure" in rendered
    assert "old failure" not in rendered


@check
def legacy_same_second_latest_tie_is_expanded_and_marked_partial():
    def turn(artifact_id, invocation_id, *, order_ns=0):
        meta = {"order_ns": order_ns} if order_ns else {}
        return SimpleNamespace(
            id=artifact_id, kind="turn", timestamp="2026-01-01T00:00:00Z",
            task_id="task", summary="", structured_body={"meta": meta, "turn_receipt": {
                "turn_id": artifact_id, "disposition": "completed", "warnings": [],
                "operations": [{
                    "invocation_id": invocation_id, "name": "run_command", "args": {},
                    "requested": True, "rejected_before_execution": False,
                    "execution_started": True, "settled": True, "disposition": "failed",
                    "outcome_text": invocation_id,
                }],
            }},
        )

    request = "What failed in the latest turn?"
    legacy = interpret_turn(request, (turn("turn-z", "z"), turn("turn-a", "a")), task_id="task")
    coverage = next(ref for ref in legacy.admission.referents
                    if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_coverage")
    details = [ref for ref in legacy.admission.referents
               if isinstance(ref, dict) and ref.get("kind") == "execution_receipt"]
    assert {ref["artifact_id"] for ref in details} == {"turn-z", "turn-a"}
    assert coverage["coverage"] == "partial" and coverage["ambiguous_order_count"] == 2

    ordered = interpret_turn(
        request, (turn("turn-z", "z", order_ns=10), turn("turn-a", "a", order_ns=11)),
        task_id="task",
    )
    ordered_details = [ref for ref in ordered.admission.referents
                       if isinstance(ref, dict) and ref.get("kind") == "execution_receipt"]
    ordered_coverage = next(ref for ref in ordered.admission.referents
                            if isinstance(ref, dict)
                            and ref.get("kind") == "execution_receipt_coverage")
    assert [ref["artifact_id"] for ref in ordered_details] == ["turn-a"]
    assert ordered_coverage["coverage"] == "complete"


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
