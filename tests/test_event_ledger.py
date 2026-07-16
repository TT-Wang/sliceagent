"""Application event-ledger invariants. No model, network, or real user state."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.event_ledger import (  # noqa: E402
    EventLedger,
    LedgerCorruptError,
    LedgerError,
    LedgerEvent,
    backfill_delivered_responses,
)


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _record(ledger, *, logical="task:1", segment="task:1:segment:0", epoch=0, text="hello"):
    return ledger.record(
        "user_utterance", logical_turn_id=logical, task_id="task",
        segment_id=segment, workspace_epoch=epoch, workspace_id="workspace-a",
        payload={"text": text}, identity=(logical, "user"), timestamp=1.0,
    )


@check
def exact_events_are_private_durable_and_deeply_read_only():
    root = tempfile.mkdtemp(prefix="ledger-")
    ledger = EventLedger("session", root=root)
    event = _record(ledger, text="build this")
    assert event.payload["text"] == "build this"
    try:
        event.payload["text"] = "changed"
    except TypeError:
        pass
    else:
        raise AssertionError("payload must be deeply immutable")
    # Windows does not expose POSIX owner/group mode bits through chmod/stat; the file inherits the private
    # user-profile/cache ACL there.  Assert the explicit 0600 repair on hosts where those bits are meaningful.
    if os.name != "nt":
        assert os.stat(ledger.path).st_mode & 0o777 == 0o600
    restored = EventLedger("session", root=root)
    assert restored.events()[0].to_dict() == event.to_dict()


@check
def source_lookup_uses_the_same_redacted_bytes_before_and_after_restart():
    root = tempfile.mkdtemp(prefix="ledger-")
    ledger = EventLedger("session", root=root)
    raw = "use sk-abcdefghij1234567890XYZ for this request"
    event = _record(ledger, text=raw)
    canonical = ledger.user_sources()[event.id]
    assert canonical != raw and len(canonical) == len(raw)
    assert EventLedger("session", root=root).user_sources()[event.id] == canonical


@check
def unresolved_sources_fault_in_across_normal_app_session_restarts():
    root = tempfile.mkdtemp(prefix="ledger-")
    old = EventLedger("old-app-session", root=root)
    event = _record(old, text="finish the exact earlier request")
    current = EventLedger("new-app-session", root=root)
    assert event.id not in current.user_sources()
    assert current.resolve_user_sources((event.id,)) == {
        event.id: "finish the exact earlier request",
    }
    assert current.resolve_events((event.id,))[event.id].payload["text"] == (
        "finish the exact earlier request"
    )
    # Dependency selection remains bounded: no uncited archived prompt is returned.
    other = _record(old, logical="task:2", segment="task:2:segment:0", text="unrelated")
    assert other.id not in current.resolve_user_sources((event.id,))
    assert other.id not in current.resolve_events((event.id,))


@check
def exact_event_resolution_scans_all_archives_for_conflicting_global_ids():
    root = tempfile.mkdtemp(prefix="ledger-")
    for session_id, text in (("a", "first"), ("b", "conflicting")):
        ledger = EventLedger(session_id, root=root)
        ledger.append(LedgerEvent(
            id="same-global-id", kind="user_utterance", session_id=session_id,
            logical_turn_id="logical", task_id="task", segment_id="segment",
            workspace_epoch=0, workspace_id="workspace", payload={"text": text},
            timestamp=1.0,
        ))
    current = EventLedger("z", root=root)
    try:
        current.resolve_events(("same-global-id",))
    except LedgerCorruptError:
        pass
    else:
        raise AssertionError("a later archive must not hide a conflicting global event ID")


@check
def retries_are_idempotent_but_conflicting_identity_fails():
    root = tempfile.mkdtemp(prefix="ledger-")
    ledger = EventLedger("session", root=root)
    first = _record(ledger)
    assert _record(ledger) is first
    # Timestamp is capture metadata. A deterministic retry of the same fact may
    # occur later without becoming a conflicting event.
    retry = ledger.record(
        "user_utterance", logical_turn_id="task:1", task_id="task",
        segment_id="task:1:segment:0", workspace_epoch=0, workspace_id="workspace-a",
        payload={"text": "hello"}, identity=("task:1", "user"), timestamp=99.0,
    )
    assert retry is first
    try:
        _record(ledger, text="different")
    except LedgerError:
        pass
    else:
        raise AssertionError("same event identity must not accept different content")


@check
def logical_turn_spans_workspace_segments_without_duplicate_user_event():
    root = tempfile.mkdtemp(prefix="ledger-")
    ledger = EventLedger("session", root=root)
    _record(ledger)
    ledger.record(
        "context_transition", logical_turn_id="task:1", task_id="task",
        segment_id="task:1:segment:0", workspace_epoch=0, workspace_id="workspace-a",
        payload={"source": "/a", "target": "/b", "target_epoch": 1},
        identity=("task:1", "transition", 0), timestamp=2.0,
    )
    ledger.record(
        "response_delivered", logical_turn_id="task:1", task_id="task",
        segment_id="task:1:segment:1", workspace_epoch=1, workspace_id="workspace-b",
        payload={"artifact_id": "turn-b"}, identity=("task:1", "response"), timestamp=3.0,
    )
    rows = ledger.logical_turn("task:1")
    assert [row.kind for row in rows] == [
        "user_utterance", "context_transition", "response_delivered",
    ]
    assert len(ledger.events("user_utterance")) == 1


@check
def final_torn_fragment_is_ignored_but_complete_corruption_is_fatal():
    root = tempfile.mkdtemp(prefix="ledger-")
    ledger = EventLedger("session", root=root)
    event = _record(ledger)
    with open(ledger.path, "ab") as stream:
        stream.write(b'{"partial":')
    restored = EventLedger("session", root=root)
    assert restored.events()[0].to_dict() == event.to_dict()

    # Recovery must repair the physical tail, not merely ignore it in memory.
    # A subsequent append and another restart must remain readable.
    restored.record(
        "context_transition", logical_turn_id="task:1", task_id="task",
        segment_id="task:1:segment:0", workspace_epoch=0, workspace_id="workspace-a",
        payload={"target": "/b"}, identity=("task:1", "transition", 0), timestamp=2.0,
    )
    assert [row.kind for row in EventLedger("session", root=root).events()] == [
        "user_utterance", "context_transition",
    ]

    with open(ledger.path, "ab") as stream:
        stream.write(b"}\n")
    try:
        EventLedger("session", root=root)
    except LedgerCorruptError:
        pass
    else:
        raise AssertionError("a complete corrupt record cannot be skipped")


@check
def complete_newline_less_tail_is_terminated_before_the_next_append():
    root = tempfile.mkdtemp(prefix="ledger-")
    ledger = EventLedger("session", root=root)
    _record(ledger)
    with open(ledger.path, "r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        stream.truncate()

    restored = EventLedger("session", root=root)
    restored.record(
        "response_delivered", logical_turn_id="task:1", task_id="task",
        segment_id="task:1:segment:0", workspace_epoch=0, workspace_id="workspace-a",
        payload={"artifact_id": "turn-1"}, identity=("task:1", "response"), timestamp=2.0,
    )
    assert [row.kind for row in EventLedger("session", root=root).events()] == [
        "user_utterance", "response_delivered",
    ]


@check
def duplicate_conflicts_and_cross_session_rows_are_rejected_on_restore():
    root = tempfile.mkdtemp(prefix="ledger-")
    ledger = EventLedger("session", root=root)
    original = _record(ledger)
    conflict = original.to_dict(); conflict["payload"] = {"text": "other"}
    with open(ledger.path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(conflict) + "\n")
    try:
        EventLedger("session", root=root)
    except LedgerCorruptError:
        pass
    else:
        raise AssertionError("conflicting duplicate must be visible")

    root2 = tempfile.mkdtemp(prefix="ledger-")
    other = EventLedger("session", root=root2)
    row = LedgerEvent(
        id="foreign", kind="user_utterance", session_id="other",
        logical_turn_id="t:1", task_id="t", segment_id="", workspace_epoch=0,
        workspace_id="", payload={"text": "x"}, timestamp=1.0,
    )
    with open(other.path, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(row.to_dict()) + "\n")
    try:
        EventLedger("session", root=root2)
    except LedgerCorruptError:
        pass
    else:
        raise AssertionError("foreign session row must not enter this ledger")


@check
def sealed_terminal_response_backfills_the_ledger_idempotently_after_a_crash_gap():
    root = tempfile.mkdtemp(prefix="ledger-")
    artifact = SimpleNamespace(
        id="turn-terminal", kind="turn", status="end_turn", session_id="old-session",
        task_id="task", workspace_id="workspace-b",
        structured_body={
            "assistant": "finished",
            "meta": {
                "logical_turn_id": "logical-1", "segment_id": "logical-1:segment:1",
                "segment_index": 1, "workspace_epoch": 1, "segment_outcome": "terminal",
            },
        },
    )
    assert backfill_delivered_responses((artifact,), root=root) == 1
    assert backfill_delivered_responses((artifact,), root=root) == 0
    events = EventLedger("old-session", root=root).events("response_delivered")
    assert len(events) == 1 and events[0].payload["artifact_id"] == artifact.id

    transport = SimpleNamespace(**{
        **artifact.__dict__, "id": "turn-transport",
        "structured_body": {"assistant": "switching", "meta": {
            **artifact.structured_body["meta"], "segment_outcome": "workspace_transition",
        }},
    })
    assert backfill_delivered_responses((transport,), root=root) == 0


if __name__ == "__main__":
    passed = 0
    for fn in CHECKS:
        try:
            fn()
            passed += 1
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(CHECKS)} passed")
    raise SystemExit(0 if passed == len(CHECKS) else 1)
