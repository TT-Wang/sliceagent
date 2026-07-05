"""Regression tests for the follow-up audit:
  M1 — skill placeholder expansion (the sequential-replace bug: $10 corruption / re-expansion / leak)
  H1 — task-state survives a real DISK round-trip (checkpoint_task → load_task), not just the in-memory
       _render/_parse helpers (the gap the reviewer flagged).
No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_skills_h1.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.skills import expand_skill_args  # noqa: E402
from sliceagent.memory import MememMemory  # noqa: E402
from sliceagent.interfaces import TaskState  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def skill_expand_no_dollar10_corruption():  # M1
    # 10 positional args; $10 must map to the 10th token, NOT "<tok1>0"
    args = "a b c d e f g h i j"
    out = expand_skill_args("first=$1 tenth=$10", args)
    assert out == "first=a tenth=j", out


@check
def skill_expand_no_reexpansion():  # M1
    # $1 resolves to a token that itself contains "$2" — it must stay literal, not get re-substituted
    out = expand_skill_args("[$1] [$2]", '"$2" world')
    assert out == "[$2] [world]", out


@check
def skill_expand_unfilled_does_not_leak():  # M1
    out = expand_skill_args("a=$1 c=$3", "only")   # only 1 arg → $3 unfilled
    assert out == "a=only c=", out                  # blank, not literal "$3"
    assert "$3" not in out


@check
def skill_expand_arguments_and_noop():  # M1 (regression-safety)
    assert expand_skill_args("run $ARGUMENTS now", "x y") == "run x y now"
    assert expand_skill_args("no placeholders here", "ignored") == "no placeholders here"


@check
def taskstate_survives_disk_roundtrip():  # H1 — the real checkpoint→file→load path
    m = MememMemory.__new__(MememMemory)        # bypass memem import
    m._vault = tempfile.mkdtemp(prefix="vault-")
    m._idx = None
    ts = TaskState(
        task_id="t-h1", session_id="s1", title="add json export", goal="add a --json flag",
        requirements=[{"text": "keep the public API stable", "done": False}],
        plan=[{"step": "write the flag", "status": "done"}, {"step": "test", "status": "in_progress"}],
        world={"port": "8137", "entry": "export.py:main"},
        findings=["the flag is parsed in export.py"],
    )
    m.checkpoint_task(ts)                        # writes <vault>/tasks/t-h1.md to DISK
    # confirm it actually hit disk
    assert os.path.exists(os.path.join(m._vault, "tasks", "t-h1.md"))
    back = m.load_task("t-h1")                   # reads it back from DISK
    assert back is not None, "load_task returned None"
    assert back.requirements == [{"text": "keep the public API stable", "done": False}], back.requirements
    assert [p["step"] for p in back.plan] == ["write the flag", "test"], back.plan
    assert back.plan[1]["status"] == "in_progress"
    assert back.world == {"port": "8137", "entry": "export.py:main"}, back.world
    assert back.findings == ["the flag is parsed in export.py"], back.findings


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
