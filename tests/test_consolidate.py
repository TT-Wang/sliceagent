"""Offline tests for consolidation (MEMORY-SPEC step 4 — cache→memory). No model, no pytest.
Run: python tests/test_consolidate.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.consolidate import promote_episodes   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def rec(task, turn, obs, note="", failing=False, stop="tool_use", files=None):
    return {"task_id": task, "turn": turn, "record": {
        "steps": [{"slice": "", "action": [], "observation": obs}],
        "note": note, "meta": {"failing": failing, "stop_reason": stop, "files": files or []}}}


@check
def corrective_episode_promotes():
    recs = [rec("t1", 1, ["Error: boom"], failing=True, files=["a.py"]),
            rec("t1", 2, ["ok"], note="fixed by using to_native_string", stop="end_turn", files=["a.py"])]
    lessons = promote_episodes(recs)
    assert len(lessons) == 1
    c = lessons[0]["content"]
    assert "boom" in c and "to_native_string" in c and "a.py" in c
    assert "python" in lessons[0]["tags"]


@check
def no_error_no_lesson():
    assert promote_episodes([rec("t1", 1, ["ok"], note="done", stop="end_turn", files=["a.py"])]) == []


@check
def unresolved_no_lesson():
    recs = [rec("t1", 1, ["Error: boom"], failing=True),
            rec("t1", 2, ["Exit code 1"], failing=True, stop="max_steps")]   # never ended clean
    assert promote_episodes(recs) == []


@check
def dedupe_same_pitfall():
    recs = [rec("t1", 1, ["Error: same boom"], failing=True), rec("t1", 2, ["ok"], stop="end_turn"),
            rec("t2", 1, ["Error: same boom"], failing=True), rec("t2", 2, ["ok"], stop="end_turn")]
    assert len(promote_episodes(recs)) == 1


@check
def secret_excluded():
    recs = [rec("t1", 1, ["Error: api_key=sk-abc123 rejected"], failing=True),
            rec("t1", 2, ["ok"], stop="end_turn")]
    assert promote_episodes(recs) == []


@check
def consolidate_reads_cache_if_memem():
    try:
        from memagent.memory import MememMemory
        m = MememMemory()
    except Exception:
        print("  (skip: memem not importable)")
        return
    m._vault = tempfile.mkdtemp()
    captured = []
    m.remember = lambda content, *, title="", scope="default", tags="": captured.append((title, content, tags))
    m.append_episode("s1", "t1", 1, {"steps": [{"slice": "", "action": [], "observation": ["Error: boom"]}],
                                     "note": "", "meta": {"failing": True, "stop_reason": "tool_use", "files": ["a.py"]}})
    m.append_episode("s1", "t1", 2, {"steps": [{"slice": "", "action": [], "observation": ["ok"]}],
                                     "note": "fixed it", "meta": {"failing": False, "stop_reason": "end_turn", "files": ["a.py"]}})
    m.consolidate("s1")
    assert len(captured) == 1 and "boom" in captured[0][1] and "fixed it" in captured[0][1]
    captured.clear()
    m.consolidate("s-none")                      # no cache file → no-op, no crash
    assert captured == []


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
