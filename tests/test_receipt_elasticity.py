"""Execution-receipt projection stays exact without growing with task history. No model/network."""
from __future__ import annotations

import os
import sys
import json
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.discourse import interpret_turn  # noqa: E402
from sliceagent.regions import render_evidence_detail, render_evidence_result  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _operation(index: int, *, disposition="succeeded", reason=""):
    rejected = disposition == "rejected"
    started = not rejected
    return {
        "invocation_id": f"spawn-{index}", "name": "spawn_agent",
        "args": {"agent": "explorer", "task": f"inspect partition {index}"},
        "requested": True, "rejected_before_execution": rejected,
        "rejection_reason": reason if rejected else "",
        "execution_started": started, "settled": True, "disposition": disposition,
        "outcome_text": reason,
    }


def _artifact(turn: int, operations, *, disposition="completed", warnings=()):
    return SimpleNamespace(
        id=f"turn-{turn:04d}", kind="turn", timestamp=f"2026-07-01T00:{turn % 60:02d}:00Z",
        task_id="task", summary="", structured_body={
            "assistant": "",
            "turn_receipt": {
                "turn_id": f"turn-{turn:04d}", "disposition": disposition,
                "warnings": list(warnings),
                "operations": list(operations),
            },
        },
    )


def _render(request: str, artifacts):
    preview = interpret_turn(request, artifacts, task_id="task")
    state = SimpleNamespace(intent=SimpleNamespace(
        current_request=request, turn_contract=preview.admission,
    ))
    return preview, render_evidence_result(state) + "\n" + render_evidence_detail(state)


@check
def count_projection_is_exact_and_bounded_across_ten_thousand_operations():
    artifacts = []
    identity = 0
    for turn in range(100):
        operations = []
        for offset in range(100):
            disposition = "rejected" if offset == 0 else "succeeded"
            operations.append(_operation(identity, disposition=disposition, reason="policy denied"))
            identity += 1
        artifacts.append(_artifact(turn, operations))

    request = "Across this task, how many explorer agents were requested, started, rejected, and failed?"
    preview, rendered = _render(request, artifacts)
    assert "execution_receipt" in preview.admission.source_needs
    assert "requested=10000" in rendered
    assert "execution-started=9900" in rendered
    assert "rejected-before-execution=100" in rendered
    assert "failed=0" in rendered
    assert len(rendered) < 6000, len(rendered)
    assert "spawn-9999" not in rendered, "count questions need aggregates, not per-call narration"


@check
def failure_projection_keeps_only_recorded_failures_and_reasons():
    artifacts = []
    identity = 0
    for turn in range(80):
        operations = []
        for _offset in range(80):
            disposition = "failed" if identity == 3117 else "succeeded"
            reason = "child timed out after 30 seconds" if disposition == "failed" else ""
            operations.append(_operation(identity, disposition=disposition, reason=reason))
            identity += 1
        artifacts.append(_artifact(turn, operations))

    preview, rendered = _render("Which explorer failed and why?", artifacts)
    assert "execution_receipt" in preview.admission.source_needs
    assert "child timed out after 30 seconds" in rendered
    assert "spawn-3117" in rendered
    assert "spawn-3116" not in rendered and "spawn-3118" not in rendered
    assert len(rendered) < 8000, len(rendered)


@check
def complete_zero_partial_zero_and_unavailable_are_three_different_answers():
    request = "Own up to your failures: which explorer failed and why?"
    success = _artifact(1, [_operation(1)])

    _preview, complete = _render(request, (success,))
    assert "coverage: COMPLETE" in complete
    assert "ZERO adverse lifecycle events" in complete
    assert "selected execution failed is false" in complete

    missing = SimpleNamespace(
        id="turn-missing", kind="turn", timestamp="2026-07-01T00:02:00Z",
        task_id="task", summary="", structured_body={"assistant": "legacy turn"},
    )
    _preview, partial = _render(request, (success, missing))
    assert "coverage: PARTIAL" in partial
    assert "overall outcome is unknown" in partial
    assert "selected execution failed is false" not in partial

    _preview, unavailable = _render(request, (missing,))
    assert "coverage: UNAVAILABLE" in unavailable
    assert "evidence gap, not evidence of success or failure" in unavailable


@check
def interrupted_empty_turn_and_unknown_operation_cannot_render_as_clean_zero():
    request = "What went wrong?"
    interrupted = _artifact(2, (), disposition="interrupted", warnings=("loop guard stopped the turn",))
    _preview, rendered = _render(request, (interrupted,))
    assert "interrupted=1" in rendered and "non-clean turns=1" in rendered
    assert "recorded turn warning: loop guard stopped the turn" in rendered
    assert "ZERO adverse lifecycle events" not in rendered

    unknown = _artifact(3, [_operation(3, disposition="settled")])
    _preview, rendered = _render(request, (unknown,))
    assert "unknown=1" in rendered
    assert "disposition=unknown (recorded=settled)" in rendered
    assert "ZERO adverse lifecycle events" not in rendered


@check
def failure_premise_is_scoped_to_the_selected_operation_family():
    operations = (
        _operation(1, disposition="succeeded"),
        {
            "invocation_id": "read-failed", "name": "read_file", "args": {"path": "missing.py"},
            "requested": True, "rejected_before_execution": False, "execution_started": True,
            "settled": True, "disposition": "failed", "outcome_text": "file vanished",
        },
    )
    request = "Which explorer failed and why?"
    _preview, rendered = _render(request, (_artifact(4, operations, disposition="completed_with_warnings"),))
    assert "family=delegation" in rendered
    assert "ZERO adverse lifecycle events" in rendered
    assert "unfiltered turn context (does NOT mean the selected family failed)" in rendered
    assert "file vanished" not in rendered, "an unrelated file failure is not a delegation failure"


@check
def coverage_and_context_metadata_stay_bounded_and_digests_are_order_independent():
    request = "Which explorer failed and why?"
    missing = tuple(
        SimpleNamespace(
            id=f"missing-{index:05d}", kind="turn", timestamp=f"2026-07-01T01:{index % 60:02d}:00Z",
            task_id="task", summary="", structured_body={},
        )
        for index in range(10_000)
    )
    preview = interpret_turn(request, missing, task_id="task")
    encoded = json.dumps(preview.admission.to_dict(), sort_keys=True)
    assert len(encoded) < 8_000, len(encoded)
    coverage = next(ref for ref in preview.admission.referents
                    if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_coverage")
    assert coverage["missing_receipt_count"] == 10_000
    assert len(coverage["missing_receipt_sample"]) == 3

    left = _artifact(10, [_operation(10, disposition="failed", reason="first")])
    right = _artifact(11, [_operation(11, disposition="failed", reason="second")])
    one = interpret_turn(request, (left, right), task_id="task")
    two = interpret_turn(request, (right, left), task_id="task")
    def aggregate(result):
        return next(ref for ref in result.admission.referents
                    if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate")
    assert aggregate(one)["source_set_sha256"] == aggregate(two)["source_set_sha256"]
    assert aggregate(one)["projection_sha256"] == aggregate(two)["projection_sha256"]

    from sliceagent.memory import NullMemory
    from sliceagent.pfc import Slice
    from sliceagent.retriever import NullRetriever
    from sliceagent.seed import make_build_slice
    from sliceagent.tools import LocalToolHost

    state = Slice(); state.reset(request)
    state.intent.current_request = request
    state.intent.turn_admission = one.admission
    plan = make_build_slice(
        state, LocalToolHost(tempfile.mkdtemp(prefix="evidence-metadata-")),
        NullRetriever(), NullMemory(), request,
    )()
    evidence_blocks = [block for block in plan.blocks if block.item_id.startswith("region:evidence_")]
    assert evidence_blocks
    assert max(len(block.source_refs) for block in evidence_blocks) <= 3
    assert max(len(block.resource_refs) for block in evidence_blocks) <= 2
    assert all(any(ref.handle == "artifacts/index.md" for ref in block.resource_refs)
               for block in evidence_blocks)


@check
def long_failure_reason_is_marked_as_display_partial_with_exact_artifact_locator():
    reason = "timeout trace " + ("frame " * 100)
    preview, rendered = _render(
        "Which explorer failed and why?",
        (_artifact(12, [_operation(12, disposition="failed", reason=reason)]),),
    )
    assert preview.admission.evidence_query.predicate == "failure_detail"
    assert "recorded reason excerpt=" in rendered
    assert "DISPLAY PARTIAL ONLY" in rendered
    assert "exact reason remains in artifacts/turn-0012.md" in rendered


@check
def current_command_and_file_tool_names_are_all_visible_to_receipt_queries():
    command_names = (
        "run_command", "execute_code", "proc_start", "proc_poll", "proc_tail", "proc_wait", "proc_kill",
        "terminal_open", "terminal_send", "terminal_read", "terminal_wait", "terminal_close",
    )
    command_ops = [
        {
            **_operation(index, disposition="failed", reason="recorded failure"),
            "name": name,
        }
        for index, name in enumerate(command_names)
    ]
    preview, rendered = _render("Which command failed and why?", (_artifact(20, command_ops),))
    aggregate = next(ref for ref in preview.admission.referents
                     if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate")
    assert aggregate["operation_count"] == len(command_names)
    assert all(name in rendered for name in command_names)

    file_names = (
        "read_file", "list_files", "grep", "glob", "code_review",
        "edit_file", "append_to_file", "str_replace", "write_file",
    )
    file_ops = [
        {
            **_operation(index + 100, disposition="failed", reason="recorded failure"),
            "name": name, "args": {"path": f"file-{index}.py"},
        }
        for index, name in enumerate(file_names)
    ]
    preview, rendered = _render("Which file tools failed and why?", (_artifact(21, file_ops),))
    aggregate = next(ref for ref in preview.admission.referents
                     if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate")
    assert aggregate["operation_count"] == len(file_names)
    assert all(name in rendered for name in file_names)


@check
def file_write_counts_distinct_direct_paths_and_discloses_opaque_command_limits():
    operations = (
        {**_operation(1), "name": "edit_file", "args": {"path": "same.py"}},
        {**_operation(2), "name": "str_replace", "args": {"path": "same.py"}},
        {**_operation(3), "name": "append_to_file", "args": {"path": "other.py"}},
        {**_operation(4), "name": "execute_code", "args": {"code": "open('hidden.py','w').write('x')"}},
    )
    preview, rendered = _render("How many files did you edit?", (_artifact(22, operations),))
    assert preview.admission.evidence_query.family == "file_write"
    aggregate = next(ref for ref in preview.admission.referents
                     if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate")
    assert aggregate["counts"]["requested"] == 3
    assert aggregate["distinct_direct_file_path_count"] == 2
    assert aggregate["opaque_command_operation_count"] == 1
    assert "distinct explicitly targeted paths=2" in rendered
    assert "nested file access is not inspectable" in rendered


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
