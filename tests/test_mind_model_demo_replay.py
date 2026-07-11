"""Deterministic replay of the mind-model demo failures. No model, network, or pytest required.

This intentionally crosses the public seams the real host uses: discourse interpretation produces one
TurnAdmission, TurnAuthorityHook enforces it, TurnReceipt reduces canonical execution journals, and the
turn-contract renderer projects only the selected durable evidence back into the next slice.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.discourse import extract_pending_proposal, interpret_turn  # noqa: E402
from sliceagent.events import TurnEnd  # noqa: E402
from sliceagent.hooks import TurnAuthorityHook  # noqa: E402
from sliceagent.pfc import Slice, record_user, slice_sink  # noqa: E402
from sliceagent.receipts import TurnReceipt  # noqa: E402
from sliceagent.registry import ToolIntentEffect  # noqa: E402
from sliceagent.regions import render_evidence_detail, render_evidence_result  # noqa: E402
from sliceagent.session import apply_turn_continuation  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


HUNTER_PATH = "/Users/tongtao/Desktop/hunter"


def _authority_gate(admission):
    return TurnAuthorityHook(
        lambda: admission,
        lambda name, _args: (
            ToolIntentEffect.OBSERVE if name == "read_file" else ToolIntentEffect.EXTERNAL
        ),
    )


def _sliceagent_focus():
    oriented = interpret_turn("Review the Hunter project", ())
    counterfactual = interpret_turn(
        "If you were to improve it, what would you do?", (), focus=oriented.focus,
    )
    repaired = interpret_turn("I mean SliceAgent", (), focus=counterfactual.focus)
    return oriented, counterfactual, repaired


@check
def hunter_directive_and_only_the_adjacent_exact_yes_authorize_navigation():
    directive = interpret_turn("go to Hunter workspace", ())
    assert directive.admission.effect_authority == "explicit"
    assert directive.admission.target.label == "Hunter"
    directive_gate = _authority_gate(directive.admission)
    assert directive_gate.authorize_tool("change_workspace", {"path": HUNTER_PATH}).allow
    # NAVIGATION tier: a loose named directive authorizes the reversible switch to whatever path the model
    # resolves — the gate no longer second-guesses the path of an authorized navigation (it can be undone).
    assert directive_gate.authorize_tool(
        "change_workspace", {"path": "/Users/tongtao/Desktop/atlas"},
    ).allow

    assistant = (
        "The Hunter workspace appears to be outside the current project. "
        f"Could you confirm the exact path? Is it {HUNTER_PATH}?"
    )
    proposal = extract_pending_proposal(assistant)
    assert proposal and proposal["action"] == {
        "tool": "change_workspace", "args": {"path": HUNTER_PATH},
    }

    adjacent_yes = interpret_turn("yes", (), pending_proposal=proposal)
    assert adjacent_yes.admission.effect_authority == "continuation"
    adjacent_gate = _authority_gate(adjacent_yes.admission)
    assert adjacent_gate.authorize_tool("change_workspace", {"path": HUNTER_PATH}).allow
    assert not adjacent_gate.authorize_tool(
        "change_workspace", {"path": "/Users/tongtao/Desktop/atlas"},
    ).allow, "assent is an exact grant for the clarified path, not ambient navigation authority"

    stale_yes = interpret_turn("yes", ())
    assert stale_yes.admission.effect_authority == "uncertain"
    assert not _authority_gate(stale_yes.admission).authorize_tool(
        "change_workspace", {"path": HUNTER_PATH},
    ).allow, "the same word outside its adjacent proposal must not retain authority"


@check
def project_focus_flows_into_counterfactual_then_explicitly_repairs_to_sliceagent():
    oriented, counterfactual, repaired = _sliceagent_focus()
    assert oriented.admission.target.label == "Hunter"
    assert oriented.admission.target.source == "explicit"

    assert counterfactual.admission.effect_authority == "none"
    assert "recommend" in counterfactual.admission.requested_modes
    assert counterfactual.admission.target.label == "Hunter"
    assert counterfactual.admission.target.source == "focus"

    assert repaired.admission.target.label == "SliceAgent"
    assert repaired.admission.target.source == "repair"
    assert repaired.admission.focus_repairs[0].field == "target"
    assert repaired.focus[-1]["entity"]["label"] == "SliceAgent"

    after_repair = interpret_turn("What would you improve first?", (), focus=repaired.focus)
    assert after_repair.admission.target.label == "SliceAgent"
    assert after_repair.admission.target.source == "focus"


@check
def failure_questions_and_self_audits_do_not_become_open_user_reports():
    state = Slice(); state.reset("Review this project")
    record_user(state, "Review this project", source_artifact="turn-review")
    slice_sink(state)(TurnEnd("end_turn", 1, {}))
    assert state.task.objective_status == "provisionally_satisfied"

    request = "Reflect on your performance this session — what failed or went badly?"
    audit = interpret_turn(request, (), task_id="task")
    assert "audit" in audit.admission.requested_modes
    apply_turn_continuation(state, request, admission=audit.admission)
    assert not state.open_report
    assert state.task.objective_status == "provisionally_satisfied", \
        "a leading question is not evidence that the completed objective became broken"

    real_report = "The review is still broken and the command fails."
    live = interpret_turn(real_report, (), task_id="task")
    apply_turn_continuation(state, real_report, admission=live.admission)
    assert state.open_report == real_report
    assert state.task.objective_status == "active"

    state.open_report = ""
    state.task.mark_objective_provisional()
    mixed_report = "What went wrong? The app still crashes on launch."
    mixed = interpret_turn(mixed_report, (), task_id="task")
    assert "audit" not in mixed.admission.requested_modes
    assert mixed.admission.quality_evidence_query is None
    assert mixed.admission.grounding == "live_present"
    apply_turn_continuation(state, mixed_report, admission=mixed.admission)
    assert state.open_report == mixed_report
    assert state.task.objective_status == "active", \
        "a live failure assertion remains a blocker even when the same turn also asks for reflection"

    state.open_report = ""
    deployment = "The app crashed. What went wrong with the deployment?"
    diagnosed = interpret_turn(deployment, (), task_id="task")
    assert diagnosed.admission.evidence_query is None
    assert diagnosed.admission.quality_evidence_query is None
    apply_turn_continuation(state, deployment, admission=diagnosed.admission)
    assert state.open_report == deployment, "a product failure question must not be mistaken for agent self-audit"

@check
def quoted_navigation_transcript_is_data_and_cannot_authorize_effects():
    quoted = interpret_turn(
        'The transcript says: "go to Hunter workspace" and then "yes". '
        "Explain why that behavior was wrong.",
        (),
    )
    assert quoted.admission.attributed_spans, "the quoted operative language must be marked as data"
    assert quoted.admission.effect_authority in {"none", "uncertain"}
    decision = _authority_gate(quoted.admission).authorize_tool(
        "change_workspace", {"path": HUNTER_PATH},
    )
    assert not decision.allow and "turn_authority_missing" in decision.reason
    assert _authority_gate(quoted.admission).authorize_tool(
        "read_file", {"path": "README.md"},
    ).allow, "quotation blocks effects without making the turn blind"


def _event(event_type: str, payload: dict) -> dict:
    return {"type": event_type, "payload": payload}


def _spawn_receipt(count: int, *, rejected: bool, turn_id: str) -> TurnReceipt:
    events = []
    for index in range(count):
        identity = f"spawn-{'denied' if rejected else 'ok'}-{index}"
        base = {
            "invocation_id": identity,
            "name": "spawn_agent",
            "args": {
                "agent": "explorer",
                "name": f"demo-{'denied' if rejected else 'ok'}-{index}",
            },
            "provider_index": index,
        }
        events.append(_event("tool-requested", base))
        if rejected:
            reason = f"demo policy rejection {index}"
            events.append(_event("tool-rejected", {**base, "reason": reason}))
            events.append(_event("tool-settled", {
                **base,
                "outcome": {
                    "status": "failed",
                    "text": f"Error: blocked by policy: {reason}",
                    "effects": [],
                },
            }))
        else:
            events.append(_event("tool-execution-started", base))
            events.append(_event("tool-settled", {
                **base,
                "outcome": {"status": "succeeded", "text": "sealed", "effects": []},
            }))
    return TurnReceipt.from_events(events, turn_id=turn_id, turn_status="end_turn")


def _receipt_artifact(artifact_id: str, timestamp: str, receipt: TurnReceipt, lie: str):
    return SimpleNamespace(
        id=artifact_id,
        kind="turn",
        timestamp=timestamp,
        task_id="demo-task",
        summary=lie,
        structured_body={"assistant": lie, "turn_receipt": receipt.to_dict()},
    )


def _render(request: str, preview) -> str:
    state = SimpleNamespace(intent=SimpleNamespace(
        current_request=request, turn_contract=preview.admission,
    ))
    return render_evidence_result(state) + "\n" + render_evidence_detail(state)


@check
def mixed_spawn_counts_and_failure_self_reflection_come_only_from_receipts():
    denied = _spawn_receipt(13, rejected=True, turn_id="turn-denied")
    succeeded = _spawn_receipt(11, rejected=False, turn_id="turn-succeeded")
    assert denied.counts["requested"] == 13
    assert denied.counts["rejected_before_execution"] == 13
    assert denied.counts["execution_started"] == 0
    assert denied.counts["failed"] == 0, "pre-handler rejection is not physical execution failure"
    assert succeeded.counts["requested"] == 11
    assert succeeded.counts["execution_started"] == 11
    assert succeeded.counts["succeeded"] == 11

    artifacts = (
        _receipt_artifact(
            "turn-denied", "2026-07-11T00:00:00Z", denied,
            "PROSE LIE: all 13 denied agents physically started and failed.",
        ),
        _receipt_artifact(
            "turn-succeeded", "2026-07-11T00:01:00Z", succeeded,
            "PROSE LIE: no agent succeeded.",
        ),
    )
    _, _, repaired = _sliceagent_focus()

    count_request = (
        "Across this task, how many explorer agents were requested, rejected, started, and succeeded?"
    )
    count_preview = interpret_turn(
        count_request, artifacts, task_id="demo-task", focus=repaired.focus,
    )
    assert count_preview.admission.evidence_query.family == "delegation"
    assert count_preview.admission.evidence_query.predicate == "aggregate"
    aggregate = next(
        ref for ref in count_preview.admission.referents
        if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate"
    )
    expected = {
        "requested": 24, "rejected_before_execution": 13, "execution_started": 11,
        "settled": 24, "succeeded": 11, "failed": 0, "cancelled": 0,
        "indeterminate": 0, "not_started": 0, "unknown": 0,
    }
    assert {key: aggregate["counts"][key] for key in expected} == expected
    count_rendered = _render(count_request, count_preview)
    assert "requested=24" in count_rendered
    assert "rejected-before-execution=13" in count_rendered
    assert "execution-started=11" in count_rendered
    assert "succeeded=11" in count_rendered and "failed=0" in count_rendered
    assert "PROSE LIE" not in count_rendered

    reflection_request = "Own up to your failures: why were the explorer agents rejected or failed?"
    reflection = interpret_turn(
        reflection_request, artifacts, task_id="demo-task", focus=repaired.focus,
    )
    assert reflection.admission.target.label == "SliceAgent"
    assert reflection.admission.evidence_query.family == "delegation"
    assert reflection.admission.evidence_query.predicate == "failure_detail"
    detail_refs = [
        ref for ref in reflection.admission.referents
        if isinstance(ref, dict) and ref.get("kind") == "execution_receipt"
    ]
    assert [ref["artifact_id"] for ref in detail_refs] == ["turn-denied"]
    operations = detail_refs[0]["operations"]
    assert len(operations) == 13
    assert all(operation["rejected_before_execution"] for operation in operations)
    assert all(not operation["execution_started"] for operation in operations)
    assert all(operation["disposition"] == "rejected" for operation in operations)

    reflected = _render(reflection_request, reflection)
    assert "rejected-before-execution=13" in reflected
    assert "execution-started=11" in reflected and "failed=0" in reflected
    assert detail_refs[0]["counts"]["execution_started"] == 0, \
        "the rejected-turn detail remains distinct from the task-wide aggregate"
    assert "recorded reason excerpt=demo policy rejection 0" in reflected
    assert "recorded reason excerpt=demo policy rejection 12" in reflected
    assert "PROSE LIE" not in reflected


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 — standalone replay reports every contract independently
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
