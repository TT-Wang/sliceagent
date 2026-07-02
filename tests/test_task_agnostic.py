"""Task-agnostic framing — the system prompt frames the agent for code AND general terminal/system
tasks, and defines 'done' as the task's real END-STATE (a passing test is just one instance), without
stripping the coding guidance or breaking system-message byte-stability (the prompt-cache moat).
Deterministic, no model, no pytest. Run: PYTHONPATH=src python tests/test_task_agnostic.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.pfc import Slice  # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402
from sliceagent.tools import LocalToolHost            # noqa: E402
from sliceagent.memory import NullMemory              # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _build(goal="do a task"):
    wd = tempfile.mkdtemp(prefix="ta-")
    s = Slice(); s.reset(goal)
    tools = LocalToolHost(root=wd)
    return make_build_slice(s, tools, None, NullMemory(), goal)


@check
def framing_is_task_general():
    sysmsg = _build()()[0]["content"].lower()
    assert "terminal/system tasks" in sysmsg or "engineering agent" in sysmsg, \
        "opener still frames the agent as coding-only"
    assert "end-state" in sysmsg, "verification lacks a general real-end-state definition of 'done'"
    # the general end-state must name non-code cases, not just tests
    assert any(w in sysmsg for w in ("solved puzzle", "service", "extracted answer", "configured system")), \
        "end-state framing doesn't cover non-code tasks"


@check
def coding_guidance_retained():
    sysmsg = _build()()[0]["content"]
    assert "OPEN FILES" in sysmsg and "cheapest sufficient check" in sysmsg.lower(), \
        "coding guidance was stripped — must keep the code path strong"


@check
def system_message_byte_stable():
    build = _build()
    s1 = build()[0]["content"]
    s2 = build()[0]["content"]
    assert s1 == s2, "system message must stay byte-stable across builds (prompt cache)"


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
