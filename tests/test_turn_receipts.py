"""Execution lifecycle and immutable receipt regressions. No model or network."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import (  # noqa: E402
    ToolExecutionStarted,
    ToolRejected,
    ToolRequested,
    ToolResult,
    ToolSettled,
    ToolStarted,
)
from sliceagent.execution import ToolInvocation  # noqa: E402
from sliceagent.hooks import Hooks, ToolPreflight  # noqa: E402
from sliceagent.loop import run_tool_batch  # noqa: E402
from sliceagent.receipts import (TurnReceipt, compact_receipt_projection,  # noqa: E402
                                 receipt_completion_label, receipt_has_adverse_lifecycle,
                                 receipt_summary_parts)
from sliceagent.runtime_persistence import CoreArtifactFS, LocalTurnStore  # noqa: E402


class _Host:
    def __init__(self):
        self.ran = []

    def accesses(self, _name, _args):
        return []

    def run(self, name, args):
        self.ran.append((name, dict(args)))
        return "ok"


class _LifecycleStop(Hooks):
    def preflight_tool(self, name, _args):
        return ToolPreflight(name == "denied_tool", "cancelled for receipt test", kind="lifecycle")


def _call(name: str, identity: str, **args):
    return NS(name=name, id=identity, args=args)


def test_lifecycle_stop_never_announces_physical_execution():
    host = _Host()
    events = []
    _, results = run_tool_batch(
        [_call("denied_tool", "deny-1", path="a.py")], host, events.append, _LifecycleStop(),
    )

    assert host.ran == []
    assert any(isinstance(event, ToolRequested) for event in events)
    assert any(isinstance(event, ToolRejected) for event in events)
    assert any(isinstance(event, ToolSettled) for event in events)
    assert not any(isinstance(event, (ToolExecutionStarted, ToolStarted)) for event in events)
    assert results[0]["status"] == "cancelled"


def test_allowed_handler_has_distinct_requested_started_and_settled_boundaries():
    host = _Host()
    events = []
    run_tool_batch([_call("read_file", "read-1", path="a.py")], host, events.append, Hooks())

    kinds = [type(event) for event in events]
    assert kinds.index(ToolRequested) < kinds.index(ToolExecutionStarted)
    assert kinds.index(ToolExecutionStarted) < kinds.index(ToolSettled)
    assert host.ran == [("read_file", {"path": "a.py"})]


def test_receipt_is_sealed_in_existing_artifact_and_separates_turn_disposition():
    store = LocalTurnStore(
        tempfile.mkdtemp(prefix="receipt-workspace-"), "session-1",
        store_root=tempfile.mkdtemp(prefix="receipt-store-"),
    )
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="do both")
    host = _Host()

    def sink(event):
        store.observe_event(event)
        if isinstance(event, ToolResult):
            store.observe_reduction(event)

    run_tool_batch(
        [_call("denied_tool", "deny-1", path="a.py"),
         _call("read_file", "read-1", path="b.py")],
        host, sink, _LifecycleStop(), turn_id="turn-1",
    )
    store.seal(
        state={"status": "active"}, record={"meta": {"ptok": 10, "ctok": 3}}, status="end_turn",
    )
    artifact = store.coordinator.artifacts.get(active.artifact_id)
    receipt = artifact.to_dict()["structured_body"]["turn_receipt"]

    assert artifact.status == "end_turn", "an operation warning must not rewrite the terminal turn status"
    assert receipt["disposition"] == "completed"
    assert receipt["counts"]["requested"] == 2
    assert receipt["counts"]["rejected_before_execution"] == 0
    assert receipt["counts"]["cancelled"] == 1
    assert receipt["counts"]["lifecycle_not_run"] == 1
    assert receipt["counts"]["execution_started"] == 1
    assert receipt["counts"]["settled"] == 2
    assert receipt["operations"][0]["disposition"] == "cancelled"
    assert receipt["operations"][0]["preflight_kind"] == "lifecycle"
    assert receipt["operations"][0]["execution_started"] is False
    assert receipt["operations"][1]["disposition"] == "succeeded"
    assert receipt["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}
    compact = compact_receipt_projection(receipt)
    assert receipt_completion_label(compact, "end_turn") == "turn saved"
    assert not receipt_has_adverse_lifecycle(compact)
    assert any("not run" in part for part in receipt_summary_parts(compact))
    virtual = CoreArtifactFS(store.coordinator.artifacts)
    rendered = virtual.read_file(f"artifacts/{artifact.id}.md")
    assert "denied_tool · requested 1 · started 0 · rejected 0" in rendered
    assert "read_file · requested 1 · started 1 · rejected 0 · succeeded 1" in rendered
    assert "completed_with_warnings" not in virtual.index()


def test_catastrophic_preflight_remains_an_explicit_adverse_refusal():
    from sliceagent.receipts import receipt_completion_label

    invocation = ToolInvocation("stop-1", "run_command", {"command": "shutdown now"}, 0)
    events = (
        {"type": "tool-requested", "payload": {
            "invocation_id": invocation.id, "name": invocation.name,
            "args": dict(invocation.args), "provider_index": 0,
        }},
        {"type": "tool-rejected", "payload": {
            "invocation_id": invocation.id, "name": invocation.name,
            "args": dict(invocation.args), "provider_index": 0,
            "reason": "Safety stop: catastrophic fixture", "rejection_kind": "catastrophic",
        }},
        {"type": "tool-settled", "payload": {
            "invocation_id": invocation.id, "name": invocation.name,
            "outcome": {"status": "cancelled", "text": "Safety stop: catastrophic fixture"},
        }},
    )
    receipt = TurnReceipt.from_events(events, turn_id="turn-stop", turn_status="end_turn")
    record = receipt.to_dict()
    assert record["disposition"] == "completed_with_warnings"
    assert record["counts"]["rejected_before_execution"] == 1
    assert record["operations"][0]["preflight_kind"] == "catastrophic"
    assert receipt_completion_label(record, "end_turn") == "turn saved with warnings"


def test_started_without_settlement_strengthens_seal_to_indeterminate():
    store = LocalTurnStore(
        tempfile.mkdtemp(prefix="receipt-workspace-"), "session-1",
        store_root=tempfile.mkdtemp(prefix="receipt-store-"),
    )
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="deploy")
    invocation = ToolInvocation("deploy-1", "run_command", {"command": "deploy"}, 0)
    store.observe_event(ToolRequested(invocation))
    store.observe_event(ToolExecutionStarted(invocation))
    store.seal(state={"status": "active"}, record={}, status="end_turn")

    artifact = store.coordinator.artifacts.get(active.artifact_id)
    receipt = artifact.to_dict()["structured_body"]["turn_receipt"]
    assert artifact.status == "indeterminate"
    assert receipt["disposition"] == "indeterminate"
    assert receipt["operations"][0]["execution_started"] is True
    assert receipt["operations"][0]["settled"] is False


def test_receipt_projection_is_deeply_read_only_and_supports_legacy_events():
    events = (
        {"type": "tool-invocation", "payload": {
            "invocation_id": "legacy-1", "name": "read_file", "args": {"path": "a.py"},
        }},
        {"type": "tool-outcome", "payload": {
            "invocation_id": "legacy-1", "outcome": {"status": "succeeded", "text": "bytes"},
        }},
    )
    receipt = TurnReceipt.from_events(events, turn_id="turn-old")
    assert receipt.operations[0].execution_started and receipt.operations[0].disposition == "succeeded"
    try:
        receipt.operations[0].args["path"] = "changed.py"
        assert False, "receipt arguments must not be mutable"
    except TypeError:
        pass


def test_legacy_policy_denial_is_recovered_as_rejected_before_start():
    events = (
        {"type": "tool-invocation", "payload": {
            "invocation_id": "legacy-denied", "name": "run_command",
            "args": {"command": "find ."},
        }},
        {"type": "tool-outcome", "payload": {
            "invocation_id": "legacy-denied",
            "policy_denied": True,
            "rejected_before_execution": True,
            "rejection_provenance": "legacy_policy_gate",
            "outcome": {
                "status": "failed",
                "text": "Error: blocked by policy: interactive consent required",
            },
        }},
    )
    operation = TurnReceipt.from_events(events).operations[0]
    assert operation.rejected_before_execution is True
    assert operation.execution_started is False
    assert operation.disposition == "rejected"
    assert operation.rejection_reason == "interactive consent required"


def test_legacy_handler_text_cannot_erase_a_recorded_physical_start():
    events = (
        {"type": "tool-invocation", "payload": {
            "invocation_id": "legacy-plugin", "name": "plugin_gateway", "args": {},
        }},
        {"type": "tool-outcome", "payload": {
            "invocation_id": "legacy-plugin", "outcome": {
                "status": "failed",
                "text": "Error: blocked by policy: upstream service denied this request",
            },
        }},
    )
    operation = TurnReceipt.from_events(events).operations[0]
    assert operation.rejected_before_execution is False
    assert operation.execution_started is True
    assert operation.disposition == "failed"


def test_child_artifact_effect_links_exact_spawn_operation_and_compacts_for_terminal():
    effect = {
        "id": "child-1:artifact", "kind": "child_artifact",
        "payload": {
            "artifact_id": "child-1", "kind": "synthesiser", "status": "ok",
            "work_item_id": "review-parser",
            "stop_reason": "end_turn", "stop_cause": "complete",
            "recovered_from": ["provider_timeout"],
            "source_coverage_status": "source_partial",
            "required_ref_count": 2,
            "consumed_refs": ["subagents/sub-1.md"],
            "cited_refs": ["subagents/sub-1.md"],
            "covered_refs": ["subagents/sub-1.md"],
            "source_gaps": ["secret raw gap detail must stay outside compact state"],
        },
    }
    events = (
        {"type": "tool-requested", "payload": {
            "invocation_id": "spawn-1", "name": "spawn_agent", "args": {}, "provider_index": 0,
        }},
        {"type": "tool-execution-started", "payload": {
            "invocation_id": "spawn-1", "name": "spawn_agent", "args": {}, "provider_index": 0,
        }},
        {"type": "tool-settled", "payload": {
            "invocation_id": "spawn-1", "name": "spawn_agent", "outcome": {
                "status": "succeeded", "text": "done", "effects": [effect],
            },
        }},
        {"type": "tool-effect-applied", "payload": {
            "invocation_id": "spawn-1", "effect_id": effect["id"],
            "kind": effect["kind"], "payload": effect["payload"],
        }},
    )
    receipt = TurnReceipt.from_events(events, turn_id="turn-1")
    assert receipt.operations[0].artifact_refs == ("child-1",)
    assert receipt.operations[0].work_item_id == "review-parser"
    assert receipt.operations[0].child_status == "ok"
    assert receipt.operations[0].child_stop_reason == "end_turn"
    assert receipt.operations[0].child_stop_cause == "complete"
    assert receipt.operations[0].child_recovered_from == ("provider_timeout",)
    assert receipt.operations[0].child_source_coverage_status == "source_partial"
    assert receipt.operations[0].child_required_ref_count == 2
    assert receipt.operations[0].child_covered_ref_count == 1
    assert receipt.operations[0].child_source_gap_count == 1
    operation_record = receipt.operations[0].to_dict()
    assert operation_record["work_item_id"] == "review-parser"
    assert operation_record["child_recovered_from"] == ["provider_timeout"]
    assert receipt.counts["child_artifacts"] == 1
    compact = compact_receipt_projection(receipt.to_dict())
    assert compact is not None and compact["agents"]["child_artifacts"] == 1
    assert compact["agents"]["source_coverage"] == {
        "source_complete": 0, "source_partial": 1, "source_unsupported": 0, "not_assessed": 0,
        "required_refs": 2, "consumed_refs": 1, "cited_refs": 1, "covered_refs": 1,
        "source_gaps": 1,
    }
    assert "secret raw gap detail" not in json.dumps(compact)
    assert receipt_summary_parts(compact) == (
        "agents · 1/1 succeeded",
        "agents · 1 source partial · 1/2 granted reports covered",
    )


def test_mixed_agent_and_tool_summary_calls_non_agent_work_other_operations():
    compact = {
        "counts": {
            "requested": 27, "execution_started": 27, "settled": 27,
            "succeeded": 23, "failed": 4,
        },
        "agents": {
            "requested": 7, "execution_started": 7, "settled": 7,
            "succeeded": 3, "failed": 4,
        },
    }
    assert receipt_summary_parts(compact) == (
        "agents · 3/7 succeeded", "agents · 4 failed",
        "other operations · 20/20 succeeded",
    )


def test_plain_completion_uses_receipt_lifecycle_not_lossy_failure_tally():
    from contextlib import redirect_stdout
    from io import StringIO
    from sliceagent.cli import cli_sink
    from sliceagent.events import TurnCommitted

    compact = compact_receipt_projection(TurnReceipt.from_events((
        {"type": "tool-requested", "payload": {
            "invocation_id": "spawn-denied", "name": "spawn_agent", "args": {},
        }},
        {"type": "tool-rejected", "payload": {
            "invocation_id": "spawn-denied", "name": "spawn_agent", "reason": "not authorized",
        }},
        {"type": "tool-settled", "payload": {
            "invocation_id": "spawn-denied", "name": "spawn_agent",
            "outcome": {"status": "failed", "text": "blocked"},
        }},
    )).to_dict())
    output = StringIO()
    with redirect_stdout(output):
        cli_sink()(TurnCommitted(True, "end_turn", receipt=compact))
    rendered = output.getvalue()
    assert "turn saved with warnings" in rendered
    assert "agents · 0/1 succeeded" in rendered and "agents · 1 rejected before start" in rendered
    assert "agents · 1 failed" not in rendered, rendered


def test_deduplicated_logical_call_is_settled_but_not_physically_started():
    store = LocalTurnStore(
        tempfile.mkdtemp(prefix="receipt-workspace-"), "session-1",
        store_root=tempfile.mkdtemp(prefix="receipt-store-"),
    )
    active = store.begin(task_id="task-A", logical_id="turn-1", user_request="read twice")
    host = _Host()

    def sink(event):
        store.observe_event(event)
        if isinstance(event, ToolResult):
            store.observe_reduction(event)

    run_tool_batch(
        [_call("read_file", "read-1", path="a.py"), _call("read_file", "read-2", path="a.py")],
        host, sink, Hooks(), turn_id="turn-1",
    )
    store.seal(state={}, record={}, status="end_turn")
    receipt = store.coordinator.artifacts.get(active.artifact_id).to_dict()["structured_body"]["turn_receipt"]

    assert host.ran == [("read_file", {"path": "a.py"})]
    assert receipt["counts"]["requested"] == 2
    assert receipt["counts"]["execution_started"] == 1
    assert receipt["counts"]["settled"] == 2
    assert [operation["disposition"] for operation in receipt["operations"]] == ["succeeded", "succeeded"]


def main():
    checks = [value for name, value in globals().items()
              if name.startswith("test_") and callable(value)]
    failed = 0
    for fn in checks:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - standalone suite reports every focused invariant
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
