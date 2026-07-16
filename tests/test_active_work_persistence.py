from __future__ import annotations

from sliceagent.active_work import ResourceRef, WorkDelta, WorkItem
from sliceagent.memory import _now_iso, _parse_task_md, _render_task_md
from sliceagent.pfc import Slice, record_user
from sliceagent.taskstate import slice_to_task_state, task_state_to_slice


def test_active_work_survives_checkpoint_markdown_and_slice_rehydration(tmp_path):
    state = Slice(); state.reset("task")
    record_user(
        state, "canonical request", source_artifact="local-turn",
        source_event_id="event-global", logical_id="logical-1", workspace_epoch=3,
    )
    root = state.active_work.request_roots[0]
    child = WorkItem(
        id="inspect", root_id=root.id, source_refs=root.source_refs,
        description="Inspect target", status="ready",
        resource_refs=(ResourceRef("workspace_file", "src/app.py", workspace_epoch=3),),
    )
    state.active_work = state.active_work.apply(WorkDelta(
        expected_revision=state.active_work.revision, creates=(child,),
    ))
    task = slice_to_task_state(
        state, "task-1", session_id="session-1", workspace_epoch=3,
    )
    path = tmp_path / "task.md"
    path.write_text(
        _render_task_md(task, created=_now_iso(), updated=_now_iso()), encoding="utf-8",
    )

    restored_task = _parse_task_md(str(path))
    restored = task_state_to_slice(restored_task)
    assert restored_task.workspace_epoch == 3
    assert restored.active_work == state.active_work
    assert restored.active_work.digest == state.active_work.digest
    assert restored.active_work.get("inspect").resource_refs[0].workspace_epoch == 3
    restored.active_work.validate_sources({"event-global": "canonical request"})


def test_task_state_roundtrip_does_not_share_mutable_active_work_wire_records():
    state = Slice(); state.reset("task")
    record_user(state, "request", source_event_id="event", logical_id="logical")
    task = slice_to_task_state(state, "task")
    task.active_work[1]["status"] = "in_progress"
    assert state.active_work.request_roots[0].status == "open"
    assert task_state_to_slice(task).active_work.request_roots[0].status == "in_progress"
