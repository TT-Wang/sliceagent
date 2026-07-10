"""Adversarial runtime-boundary regressions. No model, network, or pytest dependency.

Run: PYTHONPATH=src python tests/test_runtime_boundaries.py
"""
from __future__ import annotations

import os
import shlex
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.cli import _cwd_message  # noqa: E402
from sliceagent.execution import ToolPurity, ToolStatus  # noqa: E402
from sliceagent.hooks import Hooks  # noqa: E402
from sliceagent.loop import run_tool_batch  # noqa: E402
from sliceagent.platform_compat import ProcessGroupTerminationError  # noqa: E402
from sliceagent.procman import ProcManager  # noqa: E402
from sliceagent.registry import ToolEntry  # noqa: E402
from sliceagent.terminal import SessionManager  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _schema(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "parameters": {"type": "object", "properties": {}},
    }}


def _call(name: str, call_id: str) -> NS:
    return NS(name=name, args={}, id=call_id)


def _wait_pid_file(path: str, timeout: float = 3.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw = open(path, encoding="utf-8").read().strip()
        except OSError:
            raw = ""
        if raw:
            return int(raw)
        time.sleep(0.025)
    raise AssertionError(f"child pid was not written to {path}")


def _pid_is_gone(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.025)
    return False


@check
def cwd_is_a_process_boundary_not_a_partial_reroot():
    current = tempfile.mkdtemp(prefix="cwd-current-")
    target = tempfile.mkdtemp(prefix="cwd-target-")
    assert _cwd_message(current) == f"workspace: {os.path.realpath(current)}"
    msg = _cwd_message(current, target)
    assert "fixed for this SliceAgent process" in msg and os.path.realpath(target) in msg, msg
    assert os.path.realpath(current) != os.path.realpath(target), "test premise broken"


@check
def mutating_extension_exception_is_indeterminate_and_stops_later_barriers():
    root = tempfile.mkdtemp(prefix="extension-boundary-")
    host = LocalToolHost(root)
    ran = []

    def mutate_then_raise(_args):
        ran.append("uncertain")
        with open(os.path.join(root, "maybe.txt"), "w", encoding="utf-8") as f:
            f.write("mutation landed")
        raise RuntimeError("handler crashed after mutation")

    def later(_args):
        ran.append("later")
        return "later ran"

    host.registry.register(ToolEntry(
        "uncertain_extension", _schema("uncertain_extension"), mutate_then_raise,
        source="plugin:adversarial",
    ))
    host.registry.register(ToolEntry(
        "later_extension", _schema("later_extension"), later,
        source="plugin:adversarial",
    ))
    _, results = run_tool_batch(
        [_call("uncertain_extension", "first"), _call("later_extension", "second")],
        host, lambda _event: None, Hooks(),
    )
    assert os.path.exists(os.path.join(root, "maybe.txt")), "test premise: mutation must land first"
    assert [row["status"] for row in results] == ["indeterminate", "cancelled"], results
    assert ran == ["uncertain"], "a later barrier ran past unresolved extension side effects"


@check
def declared_pure_read_extension_exception_is_failed_not_indeterminate():
    root = tempfile.mkdtemp(prefix="extension-read-")
    host = LocalToolHost(root)
    ran = []

    def bad_read(_args):
        raise RuntimeError("read failed")

    def later(_args):
        ran.append("later")
        return "settled"

    host.registry.register(ToolEntry(
        "bad_read_extension", _schema("bad_read_extension"), bad_read,
        source="plugin:adversarial", purity=ToolPurity.PURE_READ,
    ))
    host.registry.register(ToolEntry(
        "settled_extension", _schema("settled_extension"), later,
        source="plugin:adversarial", purity=ToolPurity.EFFECTFUL,
    ))
    _, results = run_tool_batch(
        [_call("bad_read_extension", "first"), _call("settled_extension", "second")],
        host, lambda _event: None, Hooks(),
    )
    assert [row["status"] for row in results] == ["failed", "succeeded"], results
    assert ran == ["later"]


@check
def opaque_host_exception_is_indeterminate_at_the_scheduler_boundary():
    ran = []

    class OpaqueExtensionHost:
        def accesses(self, _name, _args):
            return []  # no explicit PURE_READ metadata: conservatively UNKNOWN

        def run(self, name, _args):
            ran.append(name)
            if name == "opaque_extension":
                raise RuntimeError("opaque host boundary failed")
            return "later ran"

    _, results = run_tool_batch(
        [_call("opaque_extension", "first"), _call("later_barrier", "second")],
        OpaqueExtensionHost(), lambda _event: None, Hooks(),
    )
    assert [row["status"] for row in results] == ["indeterminate", "cancelled"], results
    assert ran == ["opaque_extension"]


@check
def proc_kill_escalates_and_proves_orphan_group_extinction():
    if os.name != "posix":
        return
    marker = os.path.join(tempfile.mkdtemp(prefix="proc-extinct-"), "child.pid")
    pm = ProcManager(term_grace=0.1, kill_grace=2.0)
    # The descendant inherits SIGTERM ignored, while its shell leader exits immediately. A leader-only wait
    # would falsely report success; the group contract must notice it, escalate, and prove the group gone.
    command = f"trap '' TERM HUP; sleep 120 & echo $! > {shlex.quote(marker)}; exit 0"
    handle = pm.start(command, cwd="/tmp")
    child_pid = _wait_pid_file(marker)
    os.kill(child_pid, 0)
    result = pm.kill(handle)
    try:
        assert "killed" in result and _pid_is_gone(child_pid), result
    finally:
        pm.cleanup()


@check
def terminal_close_reaches_descendant_after_leader_exit():
    if os.name != "posix":
        return
    marker = os.path.join(tempfile.mkdtemp(prefix="terminal-extinct-"), "child.pid")
    sessions = SessionManager(term_grace=0.1, kill_grace=2.0)
    command = f"trap '' HUP; sleep 120 & echo $! > {shlex.quote(marker)}; exit 0"
    sessions.open("orphan", cwd="/tmp", command=command)
    child_pid = _wait_pid_file(marker)
    os.kill(child_pid, 0)
    # Let Popen observe the leader exit so this specifically pins the old leader-gated teardown bug.
    deadline = time.monotonic() + 2.0
    while sessions._s["orphan"].popen.poll() is None and time.monotonic() < deadline:
        time.sleep(0.025)
    assert sessions._s["orphan"].popen.poll() is not None, "terminal leader did not exit"
    assert "descendants alive" in sessions._status(sessions._s["orphan"]), (
        "terminal status must not equate leader exit with whole-group settlement"
    )
    result = sessions.close("orphan")
    try:
        assert result == "closed orphan" and _pid_is_gone(child_pid), result
    finally:
        sessions.cleanup()


@check
def unproved_process_and_terminal_teardown_are_typed_indeterminate():
    host = LocalToolHost(tempfile.mkdtemp(prefix="typed-teardown-"))

    def no_proof(_handle):
        raise ProcessGroupTerminationError("group still observable")

    host.procs.kill = no_proof
    proc = host.run("proc_kill", {"handle": "p1"})
    assert proc.status is ToolStatus.INDETERMINATE, proc

    def no_terminal_proof(_name):
        raise ProcessGroupTerminationError("terminal group still observable")

    host.terminals.close = no_terminal_proof
    terminal = host.run("terminal_close", {"session": "main"})
    assert terminal.status is ToolStatus.INDETERMINATE, terminal


@check
def windows_dead_leader_is_not_fabricated_as_tree_extinction():
    from sliceagent import platform_compat as compat

    class SettledLeader:
        def poll(self):
            return 0

    prior = compat.IS_WINDOWS
    try:
        compat.IS_WINDOWS = True
        assert not compat.terminate_process_group(None, SettledLeader()), \
            "without a live PID for taskkill /T, descendant extinction is unprovable"
    finally:
        compat.IS_WINDOWS = prior


def main() -> None:
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
