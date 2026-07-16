"""Regression test for the redaction sibling found by the class-completeness sweep:
checkpoint_task task-state + session index were written unredacted.
No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_selfreview_siblings.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.memory import MememMemory  # noqa: E402
from sliceagent.interfaces import TaskState  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn

SECRET = "sk-" + "Z" * 40


def _bare(vault):
    m = MememMemory.__new__(MememMemory)
    m._vault = vault
    m._idx = None
    return m


@check
def checkpoint_task_redacts_state_to_disk():  # Class 2 / task-state + session index
    vault = tempfile.mkdtemp(prefix="vault-")
    m = _bare(vault)
    ts = TaskState(task_id="t1", session_id="s1",
                   title=f"work on {SECRET}", goal=f"use {SECRET}",
                   findings=[f"the key is {SECRET}"], last_error=f"failed with {SECRET}")
    m.checkpoint_task(ts)
    task_md = open(os.path.join(vault, "tasks", "t1.md")).read()
    assert SECRET not in task_md, "task-state markdown must be redacted on disk (Class 2 sibling)"
    sess_md = open(os.path.join(vault, "sessions", "s1.md")).read()
    assert SECRET not in sess_md, "session index (title) must be redacted on disk"


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
