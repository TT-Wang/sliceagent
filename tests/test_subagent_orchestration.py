"""Adversarial coverage for retry ownership, fan-out admission, and synthesis source coverage."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.access import ReadAllAccess  # noqa: E402
from sliceagent.agents import BUILTIN_AGENTS  # noqa: E402
from sliceagent.events import (StepBegin, ToolQueued, ToolResult, ToolStarted,  # noqa: E402
                               TurnStarted)
from sliceagent.hooks import Hooks  # noqa: E402
from sliceagent.execution import (CHILD_CANCEL_SIGNAL_ARG, ToolEffect, ToolInvocation,  # noqa: E402
                                  ToolOutcome, ToolPurity, ToolStatus)
from sliceagent.loop import run_tool_batch  # noqa: E402
from sliceagent.progress import TurnProgress  # noqa: E402
from sliceagent.registry import ToolText  # noqa: E402
from sliceagent import scheduler  # noqa: E402
from sliceagent.scheduler import ScheduledTool, run_ordered  # noqa: E402
from sliceagent.subagent import (_GrantConsumptionSink, _assess_synthesis_source_coverage,  # noqa: E402
                                 _report_cites_ref)
from sliceagent.subagent_contract import ChildOutcome, SubagentArtifact, SubagentBrief  # noqa: E402


def _brief() -> SubagentBrief:
    return SubagentBrief.create(
        "merge the two reports",
        canonical_refs=("subagents/sub-1.md", "subagents/sub-2.md"),
    )


def _successful_read(sink: _GrantConsumptionSink, ref: str, output: str = "sealed report") -> None:
    from sliceagent.fan_in import artifact_read_coverage

    args = {"path": ref}
    invocation = ToolInvocation(f"read-{abs(hash((ref, output)))}", "read_file", args, 0)
    kind = "artifact" if ref.lstrip("./").startswith("artifacts/") else "subagent"
    effect = ToolEffect("resource", "resource_observed", {
        "resource_kind": kind, "handle": ref.lstrip("./"),
        "read_coverage": artifact_read_coverage(args, output),
    })
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, output, (effect,))
    sink(ToolResult(
        "read_file", args, output, False, status="succeeded",
        invocation_id=invocation.id, outcome=outcome,
    ))


def test_synthesis_citations_without_reads_are_explicitly_unsupported():
    brief = _brief()
    sink = _GrantConsumptionSink(brief.canonical_refs)
    report = "Finding A (subagents/sub-1.md); finding B (subagents/sub-2.md)."

    status, consumed, cited, covered, gaps = _assess_synthesis_source_coverage(
        BUILTIN_AGENTS["synthesiser"], brief, report, sink,
    )

    assert status == "source_unsupported"
    assert consumed == () and covered == ()
    assert cited == brief.canonical_refs
    assert all("did not read" in gap for gap in gaps if "granted report" in gap)


def test_reader_slot_is_rolled_back_when_thread_handoff_is_interrupted():
    invocation = ToolInvocation("read-handoff", "read_file", {"path": "a.py"}, 0)
    task = ScheduledTool(
        invocation, ToolPurity.PURE_READ,
        lambda: ToolOutcome(invocation, ToolStatus.SUCCEEDED, "ok"),
    )
    slot = threading.BoundedSemaphore(1)
    original_slot = scheduler._TIMEOUT_READER_SLOTS
    original_thread = scheduler.threading.Thread

    def interrupted_thread(*_args, **_kwargs):
        raise KeyboardInterrupt("thread construction interrupted")

    scheduler._TIMEOUT_READER_SLOTS = slot
    scheduler.threading.Thread = interrupted_thread
    try:
        try:
            run_ordered([task], max_workers=1)
            assert False, "thread handoff interruption must surface"
        except KeyboardInterrupt:
            pass
        assert slot.acquire(blocking=False), "the scheduler leaked its physical reader slot"
        slot.release()
    finally:
        scheduler.threading.Thread = original_thread
        scheduler._TIMEOUT_READER_SLOTS = original_slot


def test_synthesis_partial_read_cannot_launder_full_fan_in_success():
    brief = _brief()
    sink = _GrantConsumptionSink(brief.canonical_refs)
    _successful_read(sink, "./subagents/sub-1.md")
    report = "Only the first input supports this finding (subagents/sub-1.md)."

    status, consumed, cited, covered, gaps = _assess_synthesis_source_coverage(
        BUILTIN_AGENTS["synthesiser"], brief, report, sink,
    )

    assert status == "source_partial"
    assert consumed == cited == covered == ("subagents/sub-1.md",)
    assert any("did not read granted report subagents/sub-2.md" in gap for gap in gaps)
    assert any("did not cite subagents/sub-2.md" in gap for gap in gaps)


def test_synthesis_is_source_complete_only_after_every_complete_read_and_citation():
    brief = _brief()
    sink = _GrantConsumptionSink(brief.canonical_refs)
    for ref in brief.canonical_refs:
        _successful_read(sink, ref)
    report = "A (subagents/sub-1.md). B (subagents/sub-2.md)."

    status, consumed, cited, covered, gaps = _assess_synthesis_source_coverage(
        BUILTIN_AGENTS["synthesiser"], brief, report, sink,
    )

    assert status == "source_complete"
    assert consumed == cited == covered == brief.canonical_refs
    assert gaps == ()


def test_contradictory_report_can_be_source_complete_without_claim_quality_language():
    brief = _brief()
    sink = _GrantConsumptionSink(brief.canonical_refs)
    for ref in brief.canonical_refs:
        _successful_read(sink, ref)
    report = (
        "Report one says the flag is enabled (subagents/sub-1.md). "
        "Report two says the flag is disabled (subagents/sub-2.md)."
    )

    status, consumed, cited, covered, gaps = _assess_synthesis_source_coverage(
        BUILTIN_AGENTS["synthesiser"], brief, report, sink,
    )

    assert status == "source_complete"
    assert consumed == cited == covered == brief.canonical_refs
    assert gaps == ()
    assert "ground" not in status and "verif" not in status


def test_citation_boundary_accepts_sentence_punctuation_but_rejects_path_collisions():
    assert _report_cites_ref("See subagents/sub-1.md.", "subagents/sub-1.md")
    assert _report_cites_ref("See (subagents/sub-1.md), then continue", "subagents/sub-1.md")
    assert not _report_cites_ref("See subagents/sub-1.md.bak", "subagents/sub-1.md")
    assert not _report_cites_ref("See subagents/sub-10.md", "subagents/sub-1.md")


def test_truncated_grant_read_is_consumed_but_not_source_covered():
    brief = _brief()
    sink = _GrantConsumptionSink(brief.canonical_refs)
    _successful_read(sink, "subagents/sub-1.md", "prefix\n[truncated; use offset=20]\nsuffix")
    report = "A (subagents/sub-1.md)."

    status, consumed, cited, covered, gaps = _assess_synthesis_source_coverage(
        BUILTIN_AGENTS["synthesiser"], brief, report, sink,
    )

    assert status == "source_partial"
    assert consumed == cited == ("subagents/sub-1.md",)
    assert covered == ()
    assert any("complete origin-to-end coverage" in gap for gap in gaps)


def test_tail_page_alone_never_claims_complete_grant_coverage():
    brief = _brief()
    sink = _GrantConsumptionSink(brief.canonical_refs)
    args = {"path": "subagents/sub-1.md", "offset": 20000}
    invocation = ToolInvocation("tail-read", "read_file", args, 0)
    effect = ToolEffect("tail-resource", "resource_observed", {
        "resource_kind": "subagent", "handle": "subagents/sub-1.md",
        "read_coverage": "partial",
    })
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "clean final page", (effect,))
    sink(ToolResult(
        "read_file", args, outcome.text, False, status="succeeded",
        invocation_id=invocation.id, outcome=outcome,
    ))
    report = "A (subagents/sub-1.md)."

    status, consumed, cited, covered, gaps = _assess_synthesis_source_coverage(
        BUILTIN_AGENTS["synthesiser"], brief, report, sink,
    )

    assert status == "source_partial" and consumed == cited == ("subagents/sub-1.md",)
    assert covered == ()
    assert any("complete origin-to-end coverage" in gap for gap in gaps)


def test_workspace_shadow_or_missing_virtual_report_cannot_count_as_grant_consumption():
    ref = "artifacts/subagent-one.md"
    brief = SubagentBrief.create("merge", canonical_refs=(ref,))
    report = f"Claim ({ref})."

    for kind, text in (
        ("workspace_file", "convincing project-owned shadow bytes"),
        ("artifact", f"{ref}: no such retained artifact; read artifacts/index.md"),
    ):
        sink = _GrantConsumptionSink(brief.canonical_refs)
        invocation = ToolInvocation(f"read-{kind}", "read_file", {"path": ref}, 0)
        effect = ToolEffect(f"effect-{kind}", "resource_observed", {
            "resource_kind": kind, "handle": ref, "read_coverage": "partial",
        })
        outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, text, (effect,))
        sink(ToolResult(
            "read_file", dict(invocation.args), text, False, status="succeeded",
            invocation_id=invocation.id, outcome=outcome,
        ))
        status, consumed, _cited, covered, _gaps = _assess_synthesis_source_coverage(
            BUILTIN_AGENTS["synthesiser"], brief, report, sink,
        )
        assert status == "source_unsupported" and consumed == covered == ()


def test_grant_consumption_joins_absolute_argument_to_canonical_resource_receipt():
    ref = "artifacts/subagent-one.md"
    brief = SubagentBrief.create("merge", canonical_refs=(ref,))
    sink = _GrantConsumptionSink(brief.canonical_refs)
    args = {"path": f"/tmp/project/{ref}"}
    invocation = ToolInvocation("read-absolute", "read_file", args, 0)
    effect = ToolEffect("effect-absolute", "resource_observed", {
        "resource_kind": "artifact", "handle": ref, "read_coverage": "complete",
    })
    outcome = ToolOutcome(invocation, ToolStatus.SUCCEEDED, "sealed report", (effect,))
    sink(ToolResult(
        "read_file", args, outcome.text, False, status="succeeded",
        invocation_id=invocation.id, outcome=outcome,
    ))
    status, consumed, cited, covered, gaps = _assess_synthesis_source_coverage(
        BUILTIN_AGENTS["synthesiser"], brief, f"Claim ({ref}).", sink,
    )
    assert status == "source_complete"
    assert consumed == cited == covered == (ref,)
    assert gaps == ()


def test_source_coverage_round_trips_without_rewriting_operational_status():
    brief = _brief()
    with tempfile.TemporaryDirectory() as root:
        artifact = SubagentArtifact.create(
            kind="synthesiser", name="", workspace_id=root, session_id="s", task_id="t",
            parent_id="p", brief=brief, status="ok", coverage="two reports", report="merged",
            source_coverage_status="source_partial", consumed_refs=("subagents/sub-1.md",),
            cited_refs=("subagents/sub-1.md",), covered_refs=("subagents/sub-1.md",),
            source_gaps=("second report absent",),
            workspace_root=root,
        )
    restored = SubagentArtifact.from_record(artifact.to_record())
    assert restored.status == "ok", "operational completion remains a separate fact"
    assert restored.source_coverage_status == "source_partial"
    assert restored.covered_refs == ("subagents/sub-1.md",)
    assert restored.source_gaps == ("second report absent",)


def test_legacy_v1_artifact_maps_to_source_coverage_without_re_emitting_old_keys():
    brief = _brief()
    with tempfile.TemporaryDirectory() as root:
        current = SubagentArtifact.create(
            kind="synthesiser", name="", workspace_id=root, session_id="s", task_id="t",
            parent_id="p", brief=brief, status="ok", coverage="one report", report="merged",
            source_coverage_status="not_assessed", workspace_root=root,
        ).to_record()
    current.pop("source_coverage_status")
    current.pop("covered_refs")
    current.pop("source_gaps")
    current.update({
        "consumed_refs": list(brief.canonical_refs),
        "cited_refs": list(brief.canonical_refs),
        "grounding_refs": list(brief.canonical_refs),
    })

    for legacy, expected in (
        ("grounded", "source_complete"),
        ("partial", "source_partial"),
        ("unsupported", "source_unsupported"),
    ):
        current["epistemic_status"] = legacy
        restored = SubagentArtifact.from_record(current)
        emitted = restored.to_record()
        assert restored.source_coverage_status == expected
        assert restored.covered_refs == brief.canonical_refs
        assert emitted["source_coverage_status"] == expected
        assert "epistemic_status" not in emitted and "grounding_refs" not in emitted


def test_fifth_lifecycle_child_is_announced_as_queued_before_it_starts():
    release = threading.Event()
    events = []
    error = []

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, _args):
            release.wait(2)
            return "done"

    class Call:
        def __init__(self, index: int):
            self.id = f"child-{index}"
            self.name = "spawn_agent"
            self.args = {"agent": "explorer", "task": f"inspect area {index}"}

    def invoke():
        try:
            run_tool_batch([Call(index) for index in range(5)], Host(), events.append, Hooks())
        except BaseException as exc:  # preserve the worker failure for the assertion thread
            error.append(exc)

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        queued = [event for event in events if isinstance(event, ToolQueued)]
        started = [event for event in events if isinstance(event, ToolStarted)]
        if queued and len(started) == 4:
            break
        time.sleep(0.01)
    queued = [event for event in events if isinstance(event, ToolQueued)]
    started = [event for event in events if isinstance(event, ToolStarted)]
    try:
        assert len(queued) == 1
        assert queued[0].invocation.provider_index == 4
        assert queued[0].invocation_id == "child-4"
        assert queued[0].request_ordinal == 5
        assert queued[0].reason == "waiting for agent slot"
        assert {event.invocation.provider_index for event in started} == {0, 1, 2, 3}
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert error == []
    assert len([event for event in events if isinstance(event, ToolStarted)]) == 5


def test_lifecycle_wave_ramps_provider_launches_without_serialising_the_batch():
    release = threading.Event()
    starts = []
    errors = []

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, args):
            starts.append((args["task"], time.monotonic()))
            release.wait(2)
            return "done"

    class Call:
        def __init__(self, index: int):
            self.id = f"ramp-child-{index}"
            self.name = "spawn_agent"
            self.args = {"agent": "explorer", "task": f"area-{index}"}

    original_burst = scheduler._LIFECYCLE_INITIAL_BURST
    original_stagger = scheduler._LIFECYCLE_LAUNCH_STAGGER_SECONDS
    scheduler._LIFECYCLE_INITIAL_BURST = 1
    scheduler._LIFECYCLE_LAUNCH_STAGGER_SECONDS = 0.08

    def invoke():
        try:
            run_tool_batch([Call(index) for index in range(3)], Host(), lambda _event: None, Hooks())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    deadline = time.monotonic() + 1.5
    while len(starts) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    try:
        assert len(starts) == 3, starts
        gaps = [starts[index][1] - starts[index - 1][1] for index in (1, 2)]
        assert all(gap >= 0.05 for gap in gaps), gaps
        assert all(gap < 0.5 for gap in gaps), "ramping must not serialize on child completion"
    finally:
        release.set()
        worker.join(3)
        scheduler._LIFECYCLE_INITIAL_BURST = original_burst
        scheduler._LIFECYCLE_LAUNCH_STAGGER_SECONDS = original_stagger
    assert not worker.is_alive() and errors == []


def test_global_capacity_cancellation_announces_every_child_once_without_start_accounting():
    events = []

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, _args):
            raise AssertionError("a globally capacity-blocked child must not start")

    class Call:
        def __init__(self, index: int):
            self.id = f"capacity-child-{index}"
            self.name = "spawn_agent"
            self.args = {"agent": "explorer", "task": f"inspect area {index}"}

    original_slots = scheduler._LIFECYCLE_READER_SLOTS
    scheduler._LIFECYCLE_READER_SLOTS = threading.BoundedSemaphore(0)
    try:
        _, results = run_tool_batch(
            [Call(index) for index in range(3)], Host(), events.append, Hooks(),
            turn_id="capacity-turn",
        )
    finally:
        scheduler._LIFECYCLE_READER_SLOTS = original_slots

    queued = [event for event in events if isinstance(event, ToolQueued)]
    started = [event for event in events if isinstance(event, ToolStarted)]
    terminal = [event for event in events if isinstance(event, ToolResult)]
    assert [event.invocation_id for event in queued] == [
        "capacity-child-0", "capacity-child-1", "capacity-child-2",
    ]
    assert len({event.invocation_id for event in queued}) == 3
    assert all(event.reason == "waiting for global agent capacity" for event in queued)
    assert started == []
    assert [event.status for event in terminal] == ["cancelled", "cancelled", "cancelled"]
    assert [row["status"] for row in results] == ["cancelled", "cancelled", "cancelled"]
    assert all("agent capacity remained unavailable" in row["output"] for row in results)
    assert all("reader capacity" not in row["output"] for row in results)

    progress = TurnProgress()
    progress.reduce(TurnStarted("inspect three areas", turn_id="capacity-turn"))
    progress.reduce(StepBegin(1))
    for event in events:
        progress.reduce(event)
    snapshot = progress.snapshot()
    assert snapshot.counts.get("agent", 0) == 0, "queued/not-started children are not operations"
    assert len(snapshot.subagents) == 3
    assert all(child.phase == "cancelled" and child.started_at is None
               for child in snapshot.subagents)


def test_delegation_deadline_cancels_child_closes_slot_and_reports_timed_out():
    entered = threading.Event()
    exited = threading.Event()
    events = []
    observed = {}

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, args):
            token = args[CHILD_CANCEL_SIGNAL_ARG]
            observed["token"] = token
            entered.set()
            assert token.wait(1), "delegation deadline must reach the physical child"
            observed["reason"] = token.reason
            exited.set()
            return ToolText("child transport closed", status=ToolStatus.CANCELLED)

    class Call:
        id = "deadline-child"
        name = "spawn_agent"
        args = {"agent": "explorer", "task": "inspect deadline path"}

    old_timeout = os.environ.get("AGENT_DELEGATION_TIMEOUT")
    old_grace = os.environ.get("LLM_STREAM_CLOSE_GRACE_SEC")
    old_slots = scheduler._LIFECYCLE_READER_SLOTS
    os.environ["AGENT_DELEGATION_TIMEOUT"] = "0.04"
    os.environ["LLM_STREAM_CLOSE_GRACE_SEC"] = "0.02"
    scheduler._LIFECYCLE_READER_SLOTS = threading.BoundedSemaphore(1)
    started = time.monotonic()
    try:
        _, rows = run_tool_batch([Call()], Host(), events.append, Hooks(), turn_id="deadline-turn")
        elapsed = time.monotonic() - started
        assert entered.is_set() and exited.is_set()
        assert observed["reason"] == "deadline"
        assert elapsed < 0.8, elapsed
        outcome = rows[0]["outcome"]
        assert outcome.status is ToolStatus.FAILED
        effect = next(item for item in outcome.effects if item.kind == "child_outcome")
        assert effect.payload["stop_cause"] == "delegation_timeout"
        assert effect.payload["operational_status"] == "failed"
        assert effect.payload["artifact_id"] == "", \
            "timeout truth must not pretend an optional archive was stored"
        assert scheduler._LIFECYCLE_READER_SLOTS.acquire(blocking=False), \
            "a proven closed child must release global lifecycle capacity before return"
        scheduler._LIFECYCLE_READER_SLOTS.release()
        time.sleep(0.03)
        assert exited.is_set(), "no late child continuation may survive the settled deadline"

        progress = TurnProgress()
        progress.reduce(TurnStarted("delegate", turn_id="deadline-turn"))
        progress.reduce(StepBegin(1))
        for event in events:
            progress.reduce(event)
        snapshot = progress.snapshot()
        assert len(snapshot.subagents) == 1
        assert snapshot.subagents[0].phase == "timed_out"
    finally:
        scheduler._LIFECYCLE_READER_SLOTS = old_slots
        if old_timeout is None:
            os.environ.pop("AGENT_DELEGATION_TIMEOUT", None)
        else:
            os.environ["AGENT_DELEGATION_TIMEOUT"] = old_timeout
        if old_grace is None:
            os.environ.pop("LLM_STREAM_CLOSE_GRACE_SEC", None)
        else:
            os.environ["LLM_STREAM_CLOSE_GRACE_SEC"] = old_grace


def test_deadline_grace_preserves_direct_full_report_without_artifact():
    """Optional persistence must not decide whether a safely returned report reaches the parent."""
    cancelled = threading.Event()
    full_report = "FULL HEADLESS REPORT\n" + ("grounded detail\n" * 200)
    child = ChildOutcome(
        status="succeeded", report=full_report, kind="explorer", launch_ordinal=1,
        report_completion="complete", stop_reason="end_turn", stop_cause="complete",
    )
    invocation = ToolInvocation("headless-late-child", "spawn_agent", {
        "agent": "explorer", "task": "return a direct report without persistence",
    }, 0)

    def run():
        assert cancelled.wait(1), "deadline never reached the child"
        return ToolOutcome(
            invocation, ToolStatus.SUCCEEDED, child.render(),
            (ToolEffect("headless-late-child:outcome", "child_outcome", child.to_effect()),),
        )

    outcome = run_ordered([
        ScheduledTool(
            invocation, ToolPurity.PURE_READ, run, timeout_safe=False,
            request_cancel=lambda reason: cancelled.set() if reason == "deadline" else None,
            cancel_grace=0.08,
        ),
    ], lifecycle_timeout=0.03)[0]

    assert outcome.status is ToolStatus.FAILED
    assert full_report in outcome.text
    assert "Lifecycle warning:" in outcome.text
    effect = next(item for item in outcome.effects if item.kind == "child_outcome")
    assert effect.payload["operational_status"] == "failed"
    assert effect.payload["stop_cause"] == "delegation_timeout"
    assert effect.payload["report_bytes"] == len(full_report.encode("utf-8"))
    assert not any(item.kind == "child_artifact" for item in outcome.effects)


def test_parent_cancellation_composes_into_each_child_and_releases_its_slot():
    parent = threading.Event()
    entered = threading.Event()
    closed = threading.Event()
    box = {}

    class Host:
        def accesses(self, _name, _args):
            return [ReadAllAccess()]

        def run(self, _name, args):
            token = args[CHILD_CANCEL_SIGNAL_ARG]
            box["token"] = token
            entered.set()
            assert token.wait(1)
            box["reason"] = token.reason
            closed.set()
            return ToolText("parent cancellation closed child", status=ToolStatus.CANCELLED)

    class Call:
        id = "parent-cancel-child"
        name = "spawn_agent"
        args = {"agent": "explorer", "task": "inspect cancellation path"}

    old_slots = scheduler._LIFECYCLE_READER_SLOTS
    scheduler._LIFECYCLE_READER_SLOTS = threading.BoundedSemaphore(1)
    thread = threading.Thread(
        target=lambda: box.setdefault(
            "rows", run_tool_batch([Call()], Host(), lambda _event: None, Hooks(), signal=parent)[1],
        ),
        daemon=True,
    )
    try:
        thread.start()
        assert entered.wait(1)
        parent.set()
        thread.join(1)
        assert not thread.is_alive() and closed.is_set()
        assert box["reason"] in {"parent", "cancel"}
        assert box["rows"][0]["outcome"].status is ToolStatus.CANCELLED
        assert scheduler._LIFECYCLE_READER_SLOTS.acquire(blocking=False)
        scheduler._LIFECYCLE_READER_SLOTS.release()
    finally:
        parent.set()
        thread.join(1)
        scheduler._LIFECYCLE_READER_SLOTS = old_slots


def test_keyboard_interrupt_signals_entered_lifecycle_child_before_propagating():
    import _thread

    entered = threading.Event()
    cancelled = threading.Event()
    closed = threading.Event()
    reasons = []
    invocation = ToolInvocation("interrupt-child", "spawn_agent", {
        "agent": "explorer", "task": "wait for direct interrupt",
    }, 0)

    def run():
        entered.set()
        assert cancelled.wait(2), "KeyboardInterrupt never reached the entered child"
        closed.set()
        return ToolOutcome(invocation, ToolStatus.CANCELLED, "child closed after interrupt")

    def request_cancel(reason):
        reasons.append(reason)
        cancelled.set()

    def interrupt_when_entered():
        assert entered.wait(1)
        _thread.interrupt_main()

    old_slots = scheduler._LIFECYCLE_READER_SLOTS
    scheduler._LIFECYCLE_READER_SLOTS = threading.BoundedSemaphore(1)
    interrupter = threading.Thread(target=interrupt_when_entered, daemon=True)
    try:
        interrupter.start()
        try:
            run_ordered([
                ScheduledTool(
                    invocation, ToolPurity.PURE_READ, run, timeout_safe=False,
                    request_cancel=request_cancel, cancel_grace=0.1,
                ),
            ], lifecycle_timeout=5)
            assert False, "direct KeyboardInterrupt must propagate"
        except KeyboardInterrupt:
            pass
        assert reasons == ["interrupt"] and cancelled.is_set()
        deadline = time.monotonic() + 1
        while not closed.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert closed.is_set()
        while time.monotonic() < deadline:
            if scheduler._LIFECYCLE_READER_SLOTS.acquire(blocking=False):
                scheduler._LIFECYCLE_READER_SLOTS.release()
                break
            time.sleep(0.01)
        else:
            raise AssertionError("interrupt-signalled child did not retire its lifecycle slot")
    finally:
        cancelled.set()
        interrupter.join(1)
        scheduler._LIFECYCLE_READER_SLOTS = old_slots


def test_unresolved_child_after_close_grace_stays_indeterminate_and_keeps_capacity():
    entered = threading.Event()
    release = threading.Event()
    cancel = threading.Event()
    reason = []
    invocation = ToolInvocation("unresolved-child", "spawn_agent", {
        "agent": "explorer", "task": "ignore cancellation",
    }, 0)

    def run():
        entered.set()
        release.wait(2)  # adversarial provider/tool ignores the cancellation lease
        return ToolOutcome(invocation, ToolStatus.SUCCEEDED, "late result")

    def request_cancel(why):
        reason.append(why)
        cancel.set()

    old_slots = scheduler._LIFECYCLE_READER_SLOTS
    scheduler._LIFECYCLE_READER_SLOTS = threading.BoundedSemaphore(1)
    try:
        outcome = run_ordered([
            ScheduledTool(
                invocation, ToolPurity.PURE_READ, run,
                timeout_safe=False, request_cancel=request_cancel, cancel_grace=0.03,
            ),
        ], lifecycle_timeout=0.03)[0]
        assert entered.is_set() and cancel.is_set() and reason == ["deadline"]
        assert outcome.status is ToolStatus.INDETERMINATE
        assert not scheduler._LIFECYCLE_READER_SLOTS.acquire(blocking=False), \
            "unresolved physical work must continue owning its global slot"
        release.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if scheduler._LIFECYCLE_READER_SLOTS.acquire(blocking=False):
                scheduler._LIFECYCLE_READER_SLOTS.release()
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the lifecycle slot did not release after physical closure")
    finally:
        release.set()
        scheduler._LIFECYCLE_READER_SLOTS = old_slots


def test_committed_child_artifact_wins_scheduler_cutoff_race():
    cancel = threading.Event()
    invocation = ToolInvocation("committed-child", "spawn_agent", {
        "agent": "explorer", "task": "seal before optional mirrors",
    }, 0)
    effect = ToolEffect("committed-child:artifact", "child_artifact", {
        "artifact_id": "subagent-committed-child",
        "kind": "explorer",
        "status": "ok",
        "operational_status": "ok",
        "stop_reason": "end_turn",
        "stop_cause": "complete",
        "partial": False,
    })

    def run():
        assert cancel.wait(1)
        # run_subagent emits this effect only after canonical store + parent-ref publication. The remaining
        # physical time represents post-commit journal cleanup or optional mirror suppression.
        return ToolOutcome(invocation, ToolStatus.SUCCEEDED, "committed report", (effect,))

    outcome = run_ordered([
        ScheduledTool(
            invocation, ToolPurity.PURE_READ, run, timeout_safe=False,
            request_cancel=lambda _reason: cancel.set(), cancel_grace=0.08,
        ),
    ], lifecycle_timeout=0.03)[0]

    assert outcome.status is ToolStatus.SUCCEEDED
    assert outcome.text == "committed report"
    assert outcome.effects == (effect,)
    assert outcome.effects[0].payload["status"] == "ok", \
        "scheduler cancellation cannot contradict the already-published store/ref truth"


def test_writable_lifecycle_child_obeys_deadline_and_preserves_effect_uncertainty():
    """Purity orders a writable child as a barrier; it must not disable its lifecycle cancellation lease."""
    entered = threading.Event()
    cancelled = threading.Event()
    closed = threading.Event()
    invocation = ToolInvocation("writable-child", "spawn_agent", {
        "agent": "general", "task": "perform one isolated implementation",
    }, 0)

    def run():
        entered.set()
        assert cancelled.wait(1), "delegation deadline must reach a writable child too"
        closed.set()
        return ToolOutcome(invocation, ToolStatus.CANCELLED, "child unwound")

    started = time.monotonic()
    outcome = run_ordered([
        ScheduledTool(
            invocation, ToolPurity.EFFECTFUL, run, timeout_safe=False,
            request_cancel=lambda reason: cancelled.set() if reason == "deadline" else None,
            cancel_grace=0.08,
        ),
    ], lifecycle_timeout=0.03)[0]
    elapsed = time.monotonic() - started

    assert entered.is_set() and cancelled.is_set() and closed.is_set()
    assert elapsed < 0.5, elapsed
    assert outcome.status is ToolStatus.INDETERMINATE
    assert "may have applied workspace effects" in outcome.text
    effect = next(item for item in outcome.effects if item.kind == "child_outcome")
    assert effect.payload["stop_cause"] == "delegation_timeout"
    assert effect.payload["operational_status"] == "indeterminate"


def main() -> int:
    checks = (
        test_synthesis_citations_without_reads_are_explicitly_unsupported,
        test_synthesis_partial_read_cannot_launder_full_fan_in_success,
        test_synthesis_is_source_complete_only_after_every_complete_read_and_citation,
        test_contradictory_report_can_be_source_complete_without_claim_quality_language,
        test_citation_boundary_accepts_sentence_punctuation_but_rejects_path_collisions,
        test_truncated_grant_read_is_consumed_but_not_source_covered,
        test_tail_page_alone_never_claims_complete_grant_coverage,
        test_workspace_shadow_or_missing_virtual_report_cannot_count_as_grant_consumption,
        test_grant_consumption_joins_absolute_argument_to_canonical_resource_receipt,
        test_source_coverage_round_trips_without_rewriting_operational_status,
        test_legacy_v1_artifact_maps_to_source_coverage_without_re_emitting_old_keys,
        test_fifth_lifecycle_child_is_announced_as_queued_before_it_starts,
        test_lifecycle_wave_ramps_provider_launches_without_serialising_the_batch,
        test_global_capacity_cancellation_announces_every_child_once_without_start_accounting,
        test_delegation_deadline_cancels_child_closes_slot_and_reports_timed_out,
        test_deadline_grace_preserves_direct_full_report_without_artifact,
        test_parent_cancellation_composes_into_each_child_and_releases_its_slot,
        test_keyboard_interrupt_signals_entered_lifecycle_child_before_propagating,
        test_unresolved_child_after_close_grace_stays_indeterminate_and_keeps_capacity,
        test_committed_child_artifact_wins_scheduler_cutoff_race,
        test_writable_lifecycle_child_obeys_deadline_and_preserves_effect_uncertainty,
    )
    failed = 0
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as exc:  # noqa: BLE001 - standalone CI runner reports every adversarial case
            failed += 1
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
