"""Truthful block rendering: benign steering is neither success nor failure."""
from __future__ import annotations

import os
import tempfile

from sliceagent.execution import ToolInvocation, ToolPurity, ToolStatus
from sliceagent.cli import _classify_workspace_schedule
from sliceagent.reach import ReachSteer
from sliceagent.registry import ToolEntry, ToolRegistry
from sliceagent.tools import LocalToolHost, TOOL_SCHEMAS
from sliceagent.workspace_handoff import WorkspaceScheduleDecision


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "fixture",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_reach_steer_preserves_permission_recovery_family() -> None:
    root = tempfile.mkdtemp(prefix="render-reach-")
    host = LocalToolHost(root)

    try:
        host._resolve("../outside")
    except ReachSteer as error:
        assert isinstance(error, PermissionError)
        assert "change_workspace" in str(error)
    else:
        raise AssertionError("relative boundary traversal must steer")

    outcome = host.registry.invoke(ToolInvocation(
        "reach", "read_file", {"path": "../outside"}, 0,
    ))
    assert outcome.status is ToolStatus.STEERED
    assert not outcome.failing


def test_effectful_extension_cannot_downgrade_uncertainty_with_reach_steer() -> None:
    registry = ToolRegistry()

    def handler(_args):
        raise ReachSteer("extension claimed a benign boundary")

    registry.register(ToolEntry(
        "extension", _schema("extension"), handler,
        source="plugin", purity=ToolPurity.EFFECTFUL,
    ))
    result = registry.run("extension", {})
    assert result.status is ToolStatus.INDETERMINATE
    assert "may have applied side effects" in result


def test_virtual_archive_and_clean_str_replace_rejections_are_steered() -> None:
    root = tempfile.mkdtemp(prefix="render-edit-")
    host = LocalToolHost(root)
    path = os.path.join(root, "sample.txt")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("same\nsame\n")

    ambiguous = host.run("str_replace", {
        "path": "sample.txt", "old_string": "same", "new_string": "changed",
    })
    missing = host.run("str_replace", {
        "path": "sample.txt", "old_string": "absent", "new_string": "changed",
    })
    assert ambiguous.status is ToolStatus.STEERED
    assert missing.status is ToolStatus.STEERED
    assert open(path, encoding="utf-8").read() == "same\nsame\n"


def test_no_user_answer_remains_a_real_cancellation() -> None:
    host = LocalToolHost(tempfile.mkdtemp(prefix="render-ask-"))
    host.on_ask_user = lambda _question, _options: "(cancelled)"
    result = host.run("ask_user", {"question": "Proceed?"})
    assert result.status is ToolStatus.CANCELLED


def test_workspace_callback_projects_typed_steer_failure_and_success() -> None:
    root = tempfile.mkdtemp(prefix="render-workspace-")
    target = tempfile.mkdtemp(prefix="render-workspace-target-")
    host = LocalToolHost(root)

    host.on_workspace_switch = lambda _path: WorkspaceScheduleDecision.steered("navigation loop")
    assert host.run("change_workspace", {"path": target}).status is ToolStatus.STEERED

    host.on_workspace_switch = lambda _path: WorkspaceScheduleDecision.failed("transition unavailable")
    assert host.run("change_workspace", {"path": target}).status is ToolStatus.FAILED

    host.on_workspace_switch = lambda _path: WorkspaceScheduleDecision.scheduled()
    assert host.run("change_workspace", {"path": target}).status is ToolStatus.SUCCEEDED


def test_repeated_workspace_edge_is_a_steer_not_false_idempotent_success() -> None:
    root = tempfile.mkdtemp(prefix="render-edge-a-")
    target = tempfile.mkdtemp(prefix="render-edge-b-")
    decision = _classify_workspace_schedule(
        root, target,
        workspace_edges={(os.path.realpath(root), os.path.realpath(target))},
        max_transitions=4,
    )
    assert decision.status is ToolStatus.STEERED
    assert not decision.accepted

    budget = _classify_workspace_schedule(
        root, target, workspace_switches=4, max_transitions=4,
    )
    assert budget.status is ToolStatus.FAILED


def test_terminal_duplicate_is_steered_before_spawn() -> None:
    host = LocalToolHost(tempfile.mkdtemp(prefix="render-terminal-"))
    host.terminals._s["main"] = object()
    result = host.run("terminal_open", {"session": "main"})
    assert result.status is ToolStatus.STEERED
    assert "already open" in result


def test_update_work_schema_names_host_owned_root_exclusion() -> None:
    schema = next(row for row in TOOL_SCHEMAS if row["function"]["name"] == "update_work")
    description = schema["function"]["parameters"]["properties"]["changes"]["items"][
        "properties"
    ]["id"]["description"]
    assert "CHILD" in description
    assert "never" in description and "request-root" in description


if __name__ == "__main__":
    checks = tuple(
        value for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    passed = 0
    for check in checks:
        try:
            check()
            passed += 1
            print(f"PASS {check.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {check.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(checks)} passed")
    raise SystemExit(0 if passed == len(checks) else 1)
