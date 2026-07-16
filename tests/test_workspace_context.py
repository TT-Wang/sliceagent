"""Focused workspace-context extraction and unpublished-build rollback checks."""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.cli import _WorkspaceBuildCleanup, _prepare_workspace_resources  # noqa: E402
from sliceagent.config import Config  # noqa: E402
from sliceagent.contextfs import ContextFS  # noqa: E402
from sliceagent.memory import NullMemory  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402
from sliceagent.workspace_context import configure_workspace_contextfs  # noqa: E402


class _EmptyArtifacts:
    def read_file(self, _path):
        raise KeyError(_path)

    def _artifacts(self):
        return ()

    def _get(self, identity):
        raise KeyError(identity)


class _Session:
    session_id = "context-session"
    active_id = None
    logical_turn = None

    def active(self):
        raise KeyError("no active task")


class _ReportingMemory:
    def knowledge_records(self, **_kwargs):
        return []

    def knowledge_counts(self):
        return {"unique": 2, "user": 1, "project": 2, "craft": 0}

    def knowledge_health(self):
        return {
            "native": {"active": True, "backend": "sqlite-lexical"},
            "memem": {"active": False, "state": "disabled", "detail": "not connected"},
        }

    def memory_status(self):
        return {
            "compatibility_transition": {"state": "retained", "detail": "layout retained"},
            "compatibility_health": {
                "state": "degraded", "detail": "one mirror failed",
                "channels": {
                    "episodic_mirror": {"attempts": 2, "failed": 1, "state": "degraded"},
                },
            },
            "retirement_gate": {
                "state": "blocked", "ready": False, "detail": "proof incomplete",
                "gates": {"compatibility_writes": "failed"},
            },
            "last_consolidation": {"state": "not_recorded", "detail": "no run"},
        }


def test_extracted_context_mounts_and_status_match_the_workspace_contract():
    base_tools = SimpleNamespace(
        _contextfs=ContextFS(), _artifacts=_EmptyArtifacts(), _roster=None,
    )
    configure_workspace_contextfs(
        base_tools=base_tools,
        session=_Session(),
        memory=_ReportingMemory(),
        project_identity=SimpleNamespace(label="context-project"),
        root="/workspace/context-project",
    )

    assert base_tools._contextfs.provider_mounts == (
        "evidence", "evidence/events", "evidence/turns", "evidence/children",
        "evidence/receipts", "history", "work", "memory",
    )
    manifest = base_tools._contextfs.read_file("@sliceagent/index.md")
    assert "project: context-project" in manifest
    assert "workspace: /workspace/context-project" in manifest
    assert "history: available" in manifest and "work: available" in manifest
    assert "native index: healthy — sqlite-lexical" in manifest
    assert base_tools._contextfs.read_file("@sliceagent/work/active.md").endswith(
        "(no active request)"
    )

    status = base_tools._contextfs.read_file("@sliceagent/memory/status.md")
    assert "compatibility mirror writes (current process): degraded — one mirror failed" in status
    assert "compatibility retirement gate: blocked — proof incomplete" in status
    diagnostics = base_tools._contextfs.read_file("@sliceagent/memory/diagnostics.md")
    assert "episodic_mirror: attempts=2, failed=1, state=degraded" in diagnostics
    assert "compatibility_writes: failed" in diagnostics


def test_unpublished_cleanup_is_dependency_ordered_and_idempotent():
    calls = []

    class Reviewer:
        def join(self, timeout=None):
            calls.append(("reviewer", timeout))

    class Tools:
        def cleanup(self):
            calls.append("tools")

    class Mcp:
        def shutdown(self):
            calls.append("mcp")

    class Writer:
        def close(self):
            calls.append("monitor")

    class Store:
        def close(self):
            calls.append("store")

    cleanup = _WorkspaceBuildCleanup()
    cleanup.reviewer = Reviewer()
    cleanup.base_tools = Tools()
    cleanup.mcp_runtime = Mcp()
    cleanup.monitor_sink = SimpleNamespace(writer=Writer())
    cleanup.store = Store()
    cleanup.close()
    cleanup.close()
    assert calls == [("reviewer", 2), "tools", "mcp", "monitor", "store"]

    transferred = _WorkspaceBuildCleanup()
    transferred.store = Store()
    transferred.release()
    transferred.close()
    assert calls == [("reviewer", 2), "tools", "mcp", "monitor", "store"]


def test_workspace_lease_precedes_plugins_and_plugin_failure_rolls_back_once():
    events = []
    warnings = []

    class Store:
        def __init__(self, _root, session_id, *, exclusive):
            assert exclusive
            self.session_id = session_id
            events.append("lease")

        def recover_pending(self):
            return ()

        def checkpoints(self):
            return ()

        def close(self):
            events.append("store")

    class Tools(LocalToolHost):
        def cleanup(self):
            super().cleanup()
            events.append("tools")
            raise RuntimeError("secondary cleanup failure")

    def fail_plugins(*_args, **_kwargs):
        events.append("plugins")
        raise RuntimeError("primary plugin failure")

    root = tempfile.mkdtemp(prefix="workspace-context-order-")
    with (
        mock.patch("sliceagent.runtime_persistence.LocalTurnStore", Store),
        mock.patch("sliceagent.tools.LocalToolHost", Tools),
        mock.patch("sliceagent.plugins.load_plugins", fail_plugins),
    ):
        try:
            _prepare_workspace_resources(
                root, cfg=Config({}), llm=object(), memory=NullMemory(),
                schedule_workspace=lambda _path: "", notify_subagent=lambda _message: None,
                session_id="context-order-session", on_log=warnings.append,
            )
        except RuntimeError as exc:
            assert str(exc) == "primary plugin failure"
        else:
            raise AssertionError("plugin failure should abort workspace preparation")

    assert events == ["lease", "plugins", "tools", "store"], events
    assert len(warnings) == 1 and "secondary cleanup failure" in warnings[0], warnings


if __name__ == "__main__":
    tests = (
        test_extracted_context_mounts_and_status_match_the_workspace_contract,
        test_unpublished_cleanup_is_dependency_ordered_and_idempotent,
        test_workspace_lease_precedes_plugins_and_plugin_failure_rolls_back_once,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
