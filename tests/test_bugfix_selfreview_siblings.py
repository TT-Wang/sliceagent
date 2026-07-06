"""Regression tests for the SIBLINGS the class-completeness sweep found (same 3 classes, other call sites):
  Class 2 (redaction): checkpoint_task task-state + session index were written unredacted
  Class 3 (name-based): readonly/ask policy let unknown + non-file-mutating builtins bypass; guardrail
                        loop-tracking ignored unknown/non-file mutators
No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_selfreview_siblings.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.policy import make_policy  # noqa: E402
from sliceagent.guardrails import _NON_MUTATORS  # noqa: E402
from sliceagent.memory import MememMemory  # noqa: E402
from sliceagent.interfaces import TaskState  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn

SECRET = "sk-" + "Z" * 40


@check
def readonly_and_ask_are_deny_by_default():  # Class 3 / policy
    ro = make_policy("readonly")
    assert ro("read_file", {}).allow, "a reader is allowed in readonly"
    assert ro("grep", {}).allow
    for mut in ("edit_file", "terminal_open", "proc_start", "world_set", "update_plan", "some_unknown_mcp_tool"):
        d = ro(mut, {})
        assert not d.allow, f"readonly must DENY {mut} (incl. unknown / non-file builtins)"
    ask = make_policy("ask")
    assert ask("read_file", {}).allow, "a reader needs no confirmation"
    for mut in ("edit_file", "world_set", "some_unknown_mcp_tool"):
        d = ask(mut, {})
        assert d.ask and not d.allow, f"ask mode must CONFIRM {mut}"


@check
def guardrail_treats_unknown_and_nonfile_builtins_as_mutators():  # Class 3 / guardrails
    for reader in ("read_file", "list_files", "search_history", "grep", "glob", "ask_user"):
        assert reader in _NON_MUTATORS, f"{reader} should be a known non-mutator"
    for mut in ("edit_file", "world_set", "terminal_open", "proc_start", "update_plan",
                "some_unknown_mcp_tool"):
        assert mut not in _NON_MUTATORS, f"{mut} must be treated as a mutator for loop tracking"


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
