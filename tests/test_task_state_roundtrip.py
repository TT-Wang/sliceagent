"""Offline tests for task-state round-trip (MEMORY-SPEC step 2). No model, no pytest.
Run: python tests/test_task_state_roundtrip.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.interfaces import TaskState   # noqa: E402
from sliceagent.memory import (_now_iso, _parse_session_index, _parse_task_md,   # noqa: E402
                             _render_task_md, _upsert_session_index)
from sliceagent.pfc import Slice  # noqa: E402
from sliceagent.seed import build_artifacts  # noqa: E402
from sliceagent.regions import render_convergence  # noqa: E402
from sliceagent.taskstate import slice_to_task_state, task_state_to_slice   # noqa: E402
from sliceagent.tools import LocalToolHost   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def make_slice():
    s = Slice(); s.reset("fix the widget")
    s.task.add_progress("edit", "a.py")
    s.findings = ["root cause: X", "ruled out: Y", "fix: Z"]
    s.active_files = ["a.py", "b.py"]
    s.edited_files = {"a.py"}
    s.edit_anchor = {"a.py": "foo :: bar :: baz"}        # anchor containing ' :: '
    s.last_error = "Traceback:\n  line 1\n  line 2\nValueError: bad"
    s.reconciliation_required = "command call-7 may still be running"
    s.since_edit = 2
    return s

def _write(ts, tmp):
    path = os.path.join(tmp, f"{ts.task_id}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render_task_md(ts, created=_now_iso(), updated=_now_iso()))
    return path


@check
def roundtrip():
    s = make_slice()
    s.task.mark_objective_provisional()
    ts = slice_to_task_state(s, "t1", session_id="s1")
    ts2 = _parse_task_md(_write(ts, tempfile.mkdtemp()))
    assert ts2 is not None
    assert ts2.goal == s.goal
    assert ts2.findings == s.findings
    assert ts2.active_files == s.active_files
    assert set(ts2.edited_files) == s.edited_files
    assert ts2.edit_anchor == s.edit_anchor              # ' :: ' survived (TAB delimiter)
    assert ts2.last_error == s.last_error                # multi-line verbatim
    assert ts2.reconciliation_required == s.reconciliation_required
    assert ts2.objective_status == "provisionally_satisfied"
    assert ts2.since_edit == 2                           # record preserved it
    assert ts2.progress_signals == [{"kind": "edit", "detail": "a.py", "count": 1}]

@check
def resume_reconstructs_slice():
    s = make_slice()
    s.task.mark_objective_provisional()
    r = task_state_to_slice(slice_to_task_state(s, "t1"))
    assert r.goal == s.goal and r.findings == s.findings and r.edited_files == s.edited_files
    assert r.edit_anchor == s.edit_anchor
    assert r.reconciliation_required == s.reconciliation_required
    assert r.task.objective_status == "provisionally_satisfied"
    assert r.since_edit == 0                             # resume = fresh action epoch
    assert r.action_log == {} and r.active_skills == []   # transient cleared
    assert r.task.progress_signals and r.task.progress_signals[0].detail == "a.py"

@check
def empty_error_not_none_string():
    s = Slice(); s.reset("g"); s.last_error = ""
    ts2 = _parse_task_md(_write(slice_to_task_state(s, "t2"), tempfile.mkdtemp()))
    assert ts2.last_error == ""                          # not "None"

@check
def resume_emits_no_convergence_nudge():
    r = task_state_to_slice(slice_to_task_state(make_slice(), "t3"))
    r.last_error = ""                                    # edited non-empty + no error...
    assert render_convergence(r) == ""                  # ...but since_edit=0 → no spurious STOP

@check
def resume_uses_live_ground_truth():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "a.py"), "w") as f:
        f.write("v1\n")
    s = Slice(); s.reset("g"); s.active_files = ["a.py"]; s.edited_files = {"a.py"}
    ts = slice_to_task_state(s, "t4")
    with open(os.path.join(tmp, "a.py"), "w") as f:      # modify AFTER checkpoint
        f.write("v2-modified\n")
    art = build_artifacts(task_state_to_slice(ts), LocalToolHost(tmp))
    assert "v2-modified" in art and "v1\n" not in art    # re-read live, contents not stored

@check
def session_index_is_bounded_per_session():
    tmp = tempfile.mkdtemp()
    for tid, sess, st in [("ta1", "A", "parked"), ("ta2", "A", "active"), ("tb1", "B", "done")]:
        _upsert_session_index(tmp, TaskState(task_id=tid, session_id=sess,
                                             title=f"task {tid}", status=st), _now_iso())
    refs = _parse_session_index(os.path.join(tmp, "sessions", "A.md"))
    assert sorted(r.task_id for r in refs) == ["ta1", "ta2"]   # only session A
    by = {r.task_id: r for r in refs}
    assert by["ta1"].status == "parked" and by["ta2"].status == "active"
    assert by["ta1"].title == "task ta1"

@check
def memem_disk_roundtrip_if_available():
    try:
        from sliceagent.memory import MememMemory
        m = MememMemory()
    except Exception:
        print("  (skip: memem not importable)")
        return
    tmp = tempfile.mkdtemp()
    m._vault = tmp
    assert m.load_task("nope") is None
    ts = slice_to_task_state(make_slice(), "tz", session_id="sz")
    m.checkpoint_task(ts)
    got = m.load_task("tz")
    assert got is not None and got.findings == ts.findings and got.edit_anchor == ts.edit_anchor
    assert any(r.task_id == "tz" for r in m.list_session_tasks("sz"))
    m.append_episode("sz", "tz", 1,
                     {"steps": [{"slice": "S", "action": [], "observation": ["o"]}],
                      "note": "", "meta": {}})
    import json
    line = json.loads(open(os.path.join(tmp, "episodic", "sz.jsonl")).read().strip())
    assert line["record"]["steps"][0]["observation"] == ["o"]


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
