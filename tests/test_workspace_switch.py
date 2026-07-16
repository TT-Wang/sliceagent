"""Workspace handoff regressions: prepare B transactionally, then switch in the same process.

No model, network, process restart, or real MCP. Run: PYTHONPATH=src python tests/test_workspace_switch.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import contextlib
import io
import json
import threading
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.agents import SUBAGENT_EXCLUDED_TOOLS  # noqa: E402
from sliceagent.cli import (  # noqa: E402
    _WorkspaceHandoffHook,
    _is_workspace_transport_completion,
    _prepare_workspace_resources,
    _resolve_workspace_target,
    _workspace_presentation_sink,
    WorkspaceManager,
    WorkspaceResources,
)
from sliceagent.config import Config, load_config  # noqa: E402
from sliceagent.hooks import Hooks  # noqa: E402
from sliceagent.events import AssistantText, TurnCommitted, TurnEnd  # noqa: E402
from sliceagent.interfaces import AssistantMessage, ToolCall  # noqa: E402
from sliceagent.loop import run_tool_batch  # noqa: E402
from sliceagent.memory import LocalMemory, NullMemory  # noqa: E402
from sliceagent.pfc import Slice, record_user  # noqa: E402
from sliceagent.runtime_persistence import (  # noqa: E402
    LocalTurnStore, WorkspaceTransitionStore,
)
from sliceagent.session import (  # noqa: E402
    Session, SessionBinding, rebase_session_for_workspace, route_topic_lexical,
)
from sliceagent.tools import LocalToolHost  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def target_resolution_is_pure_canonical_and_fail_closed():
    root = tempfile.mkdtemp(prefix="switch-root-")
    target = tempfile.mkdtemp(prefix="switch-target-", dir=os.path.dirname(root))
    home_target = tempfile.mkdtemp(prefix="switch-home-", dir=os.path.expanduser("~"))
    old_cwd = os.getcwd()
    relative = os.path.relpath(target, root)
    resolved, error = _resolve_workspace_target(root, relative)
    assert not error and resolved == os.path.realpath(target)
    for quoted in (f'"{target}"', f"'{target}'"):
        resolved, error = _resolve_workspace_target(root, quoted)
        assert not error and resolved == os.path.realpath(target), (quoted, resolved, error)
    assert os.getcwd() == old_cwd, "resolving must not partially mutate the live process"
    assert _resolve_workspace_target(root, "\x00bad")[0] is None
    assert _resolve_workspace_target(root, os.path.join(root, "missing"))[0] is None
    assert _resolve_workspace_target(root, root)[0] is None
    try:
        home_spelling = os.path.join("~", os.path.basename(home_target))
        assert _resolve_workspace_target(root, home_spelling)[0] == os.path.realpath(home_target)
    finally:
        os.rmdir(home_target)

    if os.name != "nt":
        link = target + "-link"
        try:
            os.symlink(target, link)
            assert _resolve_workspace_target(root, link)[0] == os.path.realpath(target)
        finally:
            try:
                os.unlink(link)
            except OSError:
                pass


class _Bundle:
    def __init__(self, root, events):
        self.root = os.path.realpath(root)
        self.events = events
        self.closed = 0

    def close(self):
        self.closed += 1
        self.events.append(("close", self.root))


@check
def workspace_manager_switches_in_process_and_preserves_app_identity():
    current_root = tempfile.mkdtemp(prefix="switch-current-")
    target_root = tempfile.mkdtemp(prefix="switch-target-")
    events = []
    current = _Bundle(current_root, events)
    candidate = _Bundle(target_root, events)
    llm, ui, session = object(), object(), object()
    stable_ids = (id(llm), id(ui), id(session))
    cwd_before = os.getcwd()

    def prepare(target):
        events.append(("prepare", os.path.realpath(target)))
        return candidate

    manager = WorkspaceManager(current, prepare)

    def activate(bundle):
        # The candidate is authoritative before workspace-derived UI bindings refresh. The provider,
        # terminal application, and conversation session are application owners and must not be recreated.
        assert manager.current is bundle
        assert (id(llm), id(ui), id(session)) == stable_ids
        events.append(("activate", bundle.root))

    result = manager.switch(target_root, activate=activate)
    assert manager.current is candidate
    assert result is candidate or getattr(result, "ok", True), result
    assert events == [
        ("prepare", os.path.realpath(target_root)),
        ("activate", os.path.realpath(target_root)),
        ("close", os.path.realpath(current_root)),
    ], events
    assert current.closed == 1 and candidate.closed == 0
    assert os.getcwd() == cwd_before, "an in-process switch must not chdir or replace the UI process"


@check
def failed_workspace_prepare_rolls_back_without_closing_current():
    current_root = tempfile.mkdtemp(prefix="switch-rollback-current-")
    target_root = tempfile.mkdtemp(prefix="switch-rollback-target-")
    events = []
    current = _Bundle(current_root, events)

    def fail_prepare(target):
        events.append(("prepare", os.path.realpath(target)))
        raise RuntimeError("target lease unavailable")

    manager = WorkspaceManager(current, fail_prepare)
    try:
        manager.switch(target_root)
    except RuntimeError as exc:
        assert "lease" in str(exc)
    assert manager.current is current, "failed preparation must leave A authoritative"
    assert current.closed == 0, "rollback must not close the still-active workspace"
    assert events == [("prepare", os.path.realpath(target_root))]


@check
def failed_workspace_activation_closes_candidate_and_restores_current():
    current_root = tempfile.mkdtemp(prefix="switch-activate-current-")
    target_root = tempfile.mkdtemp(prefix="switch-activate-target-")
    events = []
    current = _Bundle(current_root, events)
    candidate = _Bundle(target_root, events)
    manager = WorkspaceManager(current, lambda _target: candidate)

    try:
        manager.switch(target_root, activate=lambda _bundle: (_ for _ in ()).throw(
            RuntimeError("UI refresh failed")
        ))
    except RuntimeError as exc:
        assert "refresh" in str(exc)
    assert manager.current is current
    assert current.closed == 0 and candidate.closed == 1, \
        "a failed publication must retire B without damaging A"


@check
def memory_scope_switch_replaces_project_root_and_revision_as_one_binding():
    state = tempfile.mkdtemp(prefix="switch-memory-state-")
    root_a = tempfile.mkdtemp(prefix="switch-memory-a-")
    root_b = tempfile.mkdtemp(prefix="switch-memory-b-")
    keys = ("SLICEAGENT_CACHE_DIR", "SLICEAGENT_KNOWLEDGE_DB")
    before = {key: os.environ.get(key) for key in keys}
    memory = None
    try:
        os.environ["SLICEAGENT_CACHE_DIR"] = state
        os.environ["SLICEAGENT_KNOWLEDGE_DB"] = os.path.join(state, "knowledge.db")
        memory = LocalMemory(prefer_memem=False)
        memory.set_scope(
            project_id="project-a", workspace_id="workspace-a", label="A",
            workspace_root=root_a, resource_revision="revision-a",
        )
        old_binding = memory._scope_binding
        memory.set_scope(
            project_id="project-b", workspace_id="workspace-b", label="B",
            workspace_root=root_b, resource_revision="revision-b",
        )
        assert (
            memory._project_id, memory._workspace_id, memory._workspace_root,
            memory._resource_revision, memory._scope,
        ) == (
            "project-b", "workspace-b", os.path.realpath(root_b), "revision-b", "B",
        )
        assert old_binding.project_id == "project-a"
        assert old_binding.workspace_root == os.path.realpath(root_a)

        # Normalization failure cannot publish a half-B/half-A scope.
        stable = memory._scope_binding
        with mock.patch("sliceagent.memory.os.path.realpath", side_effect=OSError("bad root")):
            try:
                memory.set_scope(
                    project_id="project-c", workspace_id="workspace-c", label="C",
                    workspace_root="/bad", resource_revision="revision-c",
                )
            except OSError:
                pass
        assert memory._scope_binding is stable
    finally:
        if memory is not None:
            memory.close()
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(state, ignore_errors=True)
        shutil.rmtree(root_a, ignore_errors=True)
        shutil.rmtree(root_b, ignore_errors=True)


@check
def workspace_resource_close_retires_each_old_owner_once():
    calls = []

    class Tools:
        def cleanup(self):
            calls.append("tools")

    class Store:
        def close(self):
            calls.append("store")

    class Mcp:
        def shutdown(self):
            calls.append("mcp")

    class Reviewer:
        _thread = None

        def join(self, timeout=None):
            calls.append(("reviewer", timeout))

    root = tempfile.mkdtemp(prefix="switch-owned-")
    base_tools, store = Tools(), Store()
    resources = WorkspaceResources(
        root=root, config=None, session=None, store=store, sandbox=None, retriever=None,
        base_tools=base_tools, tools=base_tools, skills=None, mcp_runtime=Mcp(), reviewer=Reviewer(),
    )
    resources.close()
    resources.close()  # idempotent: repeated shutdown/finally paths must not double-close owners
    assert calls == [("reviewer", 10), "tools", "mcp", "store"], calls


@check
def teardown_diagnostic_failure_never_escapes_workspace_close():
    class BrokenTools:
        def cleanup(self):
            raise RuntimeError("cleanup failed")

    class Store:
        def close(self):
            pass

    resources = WorkspaceResources(
        root=tempfile.mkdtemp(prefix="switch-close-"), config=None, session=None,
        store=Store(), sandbox=None, retriever=None, base_tools=BrokenTools(),
        tools=None, skills=None, _on_log=lambda _message: (_ for _ in ()).throw(
            BrokenPipeError("logger closed")
        ),
    )
    resources.close()
    assert resources._closed


@check
def post_commit_old_retirement_failure_does_not_falsify_switch_success():
    class Old(_Bundle):
        def close(self):
            raise RuntimeError("retirement failed")

    old = Old(tempfile.mkdtemp(prefix="switch-old-"), [])
    candidate = _Bundle(tempfile.mkdtemp(prefix="switch-new-"), [])
    candidate._on_log = lambda _message: None
    manager = WorkspaceManager(old, lambda _target: candidate)
    result = manager.switch(candidate.root)
    assert result is candidate and manager.current is candidate


@check
def target_config_loads_by_explicit_root_without_process_chdir():
    current_root = tempfile.mkdtemp(prefix="switch-config-current-")
    target_root = tempfile.mkdtemp(prefix="switch-config-target-")
    with open(os.path.join(current_root, "sliceagent.toml"), "w", encoding="utf-8") as f:
        f.write('[oracle]\nverify_cmd = "CURRENT_ONLY"\n')
    with open(os.path.join(target_root, "sliceagent.toml"), "w", encoding="utf-8") as f:
        f.write('[oracle]\nverify_cmd = "TARGET_ONLY"\n')
    cwd_before = os.getcwd()
    cfg = load_config(target_root)
    assert cfg.verify_cmd == "TARGET_ONLY", "workspace configuration must follow the selected root"
    assert os.getcwd() == cwd_before, "loading target config must not mutate process cwd"


@check
def staged_workspace_resources_all_point_at_the_explicit_target():
    target = tempfile.mkdtemp(prefix="switch-runtime-target-")
    cache = tempfile.mkdtemp(prefix="switch-runtime-cache-")
    keys = ("SLICEAGENT_CACHE_DIR", "AGENT_SUBAGENT_DEPTH", "AGENT_WEB",
            "AGENT_MONITOR", "AGENT_ALLOW_PLUGINS")
    before = {key: os.environ.get(key) for key in keys}
    resources = None
    try:
        os.environ.update({
            "SLICEAGENT_CACHE_DIR": cache,
            "AGENT_SUBAGENT_DEPTH": "0",
            "AGENT_WEB": "0",
            "AGENT_ALLOW_PLUGINS": "0",
        })
        os.environ.pop("AGENT_MONITOR", None)
        resources = _prepare_workspace_resources(
            target, cfg=Config({}), llm=object(), memory=NullMemory(),
            schedule_workspace=lambda _path: "",
            notify_subagent=lambda _message: None,
            session_id="stable-app-session",
        )
        canonical = os.path.realpath(target)
        assert resources.root == canonical
        assert resources.base_tools.root() == canonical
        assert resources.store.workspace_root == canonical
        assert resources.store.workspace_id
        assert isinstance(resources.session, SessionBinding)
        assert resources.session.session_id == resources.store.session_id == "stable-app-session"
        assert resources.tools is resources.base_tools, "subagents=0 should expose the target host directly"
        if hasattr(resources.retriever, "root"):
            assert resources.retriever.root == canonical
        assert resources.base_tools._artifacts is not None
        assert resources.project_identity is not None
        assert resources.base_tools._contextfs.provider_mounts == (
            "evidence", "evidence/events", "evidence/turns", "evidence/children",
            "evidence/receipts", "history", "work",
        )
        manifest = resources.base_tools.run("read_file", {"path": "@sliceagent/index.md"})
        assert manifest.ok and f"workspace: {canonical}" in str(manifest)
        assert "native index: unavailable" in str(manifest), \
            "a host without a knowledge repository must not report a degraded native index"
        # Once a task exists, live status must remain available. This catches accidental method calls on
        # immutable WorkGraph projection properties during workspace preparation.
        task_id = resources.session.new_topic("continue the workspace-spanning request")
        live_manifest = resources.base_tools.run("read_file", {"path": "@sliceagent/index.md"})
        assert live_manifest.ok and "live status: unavailable" not in str(live_manifest)
        assert "logical request: continue the workspace-spanning request" in str(live_manifest)
        exact_request = "continue here\nand preserve this exact second line"
        resources.session.start_logical_turn(
            logical_id="logical-contextfs", task_id=task_id, request=exact_request,
            source_artifact_id="turn-contextfs", source_event_id="event-contextfs",
            source_workspace=canonical,
        )
        record_user(
            resources.session.active(), exact_request, source_artifact="turn-contextfs",
            source_event_id="event-contextfs", logical_id="logical-contextfs",
            source_text=exact_request,
        )
        active_work = resources.base_tools.run(
            "read_file", {"path": "@sliceagent/work/active.md"},
        )
        assert active_work.ok and "CURRENT REQUEST (verbatim user source)" in str(active_work)
        assert exact_request in str(active_work), \
            "the standalone Active Work surface must fulfill its CURRENT REQUEST-below locator"
        assert any(path.startswith(canonical + os.sep) for path in resources.skills.roots), \
            "default project skills must follow B rather than process cwd"
    finally:
        if resources is not None:
            resources.close()
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(target, ignore_errors=True)


@check
def native_knowledge_failure_isolated_from_canonical_history_and_work():
    target = tempfile.mkdtemp(prefix="switch-runtime-degraded-knowledge-")
    cache = tempfile.mkdtemp(prefix="switch-runtime-degraded-cache-")
    before = os.environ.get("SLICEAGENT_CACHE_DIR")

    class BrokenKnowledge(NullMemory):
        def knowledge_records(self, **_kwargs):
            raise OSError("knowledge unavailable")

        def knowledge_counts(self):
            raise OSError("knowledge unavailable")

        def knowledge_health(self):
            return {
                # Index liveness cannot override an observed repository/query failure.
                "native": {"active": True, "backend": "native-fts5"},
                "memem": {"active": False, "state": "disabled", "detail": "not installed"},
            }

    resources = None
    try:
        os.environ["SLICEAGENT_CACHE_DIR"] = cache
        resources = _prepare_workspace_resources(
            target, cfg=Config({}), llm=object(), memory=BrokenKnowledge(),
            schedule_workspace=lambda _path: "", notify_subagent=lambda _message: None,
            session_id="degraded-knowledge-session",
        )
        manifest = str(resources.base_tools.run("read_file", {"path": "@sliceagent/index.md"}))
        assert "history: available" in manifest and "work: available" in manifest
        assert "memory: degraded" in manifest and "OSError" in manifest
        memory_index = str(resources.base_tools.run(
            "read_file", {"path": "@sliceagent/memory/index.md"},
        ))
        assert "degraded" in memory_index and "OSError" in memory_index
    finally:
        if resources is not None:
            resources.close()
        if before is None:
            os.environ.pop("SLICEAGENT_CACHE_DIR", None)
        else:
            os.environ["SLICEAGENT_CACHE_DIR"] = before
        shutil.rmtree(target, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


@check
def memory_status_uses_bounded_host_telemetry_without_private_store_discovery():
    target = tempfile.mkdtemp(prefix="switch-runtime-memory-status-")
    cache = tempfile.mkdtemp(prefix="switch-runtime-memory-status-cache-")
    before = os.environ.get("SLICEAGENT_CACHE_DIR")

    class ReportingMemory(NullMemory):
        def knowledge_records(self, **_kwargs):
            return []

        def knowledge_counts(self):
            return {"unique": 6, "user": 3, "project": 5, "craft": 2}

        def knowledge_health(self):
            return {
                "native": {
                    "active": True, "backend": "sqlite-lexical", "error": "",
                    "warning": "FTS5 unavailable",
                },
                "memem": {
                    "active": True, "state": "healthy", "detail": "latest retrieval completed",
                },
            }

        def memory_status(self):
            return {
                "legacy_inventory": {
                    "episodic_session_files": 147,
                    "task_projection_files": 126,
                    "session_projection_files": 126,
                    "subagent_archive_files": None,
                },
                "legacy_inventory_scope": "global compatibility store; not typed project knowledge",
                "compatibility_transition": {
                    "state": "retained", "detail": "legacy compatibility layout retained",
                },
                "last_consolidation": {
                    "state": "not_recorded", "detail": "no run metadata reported by host",
                },
            }

    resources = None
    try:
        os.environ["SLICEAGENT_CACHE_DIR"] = cache
        resources = _prepare_workspace_resources(
            target, cfg=Config({}), llm=object(), memory=ReportingMemory(),
            schedule_workspace=lambda _path: "", notify_subagent=lambda _message: None,
            session_id="memory-status-session",
        )
        status = str(resources.base_tools.run(
            "read_file", {"path": "@sliceagent/memory/status.md"},
        ))
        assert "Exactly three memory layers" in status
        assert "unique active current-scope records: 6" in status
        assert "USER scope memberships: 3" in status
        assert "PROJECT scope memberships: 5" in status
        assert "compatibility layout (global): retained — legacy compatibility layout retained" in status
        assert "selective knowledge consolidation (current project): not-recorded" in status
        assert "native index: healthy — sqlite-lexical" in status
        assert "query unavailable" not in status
        assert "Memem: available — last scoped operation succeeded; no continuous health probe" in status
        assert "episodic session files" not in status
        assert ".sliceagent" not in status and cache not in status
        diagnostics = str(resources.base_tools.run(
            "read_file", {"path": "@sliceagent/memory/diagnostics.md"},
        ))
        assert "episodic session files: 147" in diagnostics
        assert "session projection files: 126" in diagnostics
        assert "inventory status: degraded" in diagnostics
        assert "subagent archive files: (unknown)" in diagnostics
        assert "scope: global compatibility store; not typed project knowledge" in diagnostics
        assert ".sliceagent" not in diagnostics and cache not in diagnostics
    finally:
        if resources is not None:
            resources.close()
        if before is None:
            os.environ.pop("SLICEAGENT_CACHE_DIR", None)
        else:
            os.environ["SLICEAGENT_CACHE_DIR"] = before
        shutil.rmtree(target, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


@check
def unavailable_memory_status_does_not_claim_consolidation_never_ran():
    target = tempfile.mkdtemp(prefix="switch-runtime-memory-status-error-")
    cache = tempfile.mkdtemp(prefix="switch-runtime-memory-status-error-cache-")
    before = os.environ.get("SLICEAGENT_CACHE_DIR")

    class UnavailableStatusMemory(NullMemory):
        def memory_status(self):
            raise OSError("status store unavailable")

    resources = None
    try:
        os.environ["SLICEAGENT_CACHE_DIR"] = cache
        resources = _prepare_workspace_resources(
            target, cfg=Config({}), llm=object(), memory=UnavailableStatusMemory(),
            schedule_workspace=lambda _path: "", notify_subagent=lambda _message: None,
            session_id="memory-status-error-session",
        )
        status = str(resources.base_tools.run(
            "read_file", {"path": "@sliceagent/memory/status.md"},
        ))
        assert "selective knowledge consolidation (current project): unknown — status unavailable (OSError)" in status
        assert "selective knowledge consolidation (current project): not-recorded" not in status
    finally:
        if resources is not None:
            resources.close()
        if before is None:
            os.environ.pop("SLICEAGENT_CACHE_DIR", None)
        else:
            os.environ["SLICEAGENT_CACHE_DIR"] = before
        shutil.rmtree(target, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


@check
def model_tool_only_schedules_and_never_partially_reroots():
    root = tempfile.mkdtemp(prefix="switch-host-")
    target = tempfile.mkdtemp(prefix="switch-project-")
    host = LocalToolHost(root)
    scheduled = []
    host.on_workspace_switch = lambda path: scheduled.append(path) or ""
    cwd_before = os.getcwd()
    result = host.run("change_workspace", {"path": target})
    assert result.ok and "scheduled" in str(result).lower(), result
    assert "restart" not in str(result).lower(), \
        "the model-facing handoff must not promise a process reconnect"
    assert scheduled == [os.path.realpath(target)]
    assert host.root() == os.path.realpath(root) and os.getcwd() == cwd_before
    assert "workspace_handoff" in host.registry.entry("change_workspace").capabilities


@check
def handoff_is_a_terminal_barrier_inside_one_tool_batch():
    root = tempfile.mkdtemp(prefix="switch-batch-")
    target = tempfile.mkdtemp(prefix="switch-batch-target-")
    marker = os.path.join(root, "later.txt")
    host = LocalToolHost(root)
    host.on_workspace_switch = lambda _path: ""
    blocked, results = run_tool_batch(
        [
            ToolCall("switch", "change_workspace", {"path": target}),
            ToolCall("later", "edit_file", {"path": "later.txt", "content": "must not run"}),
        ],
        host, lambda _event: None, Hooks(),
    )
    assert blocked == 0, "terminal-barrier blocks are not model-stuck loops"
    assert results[0]["failing"] is False and results[1]["failing"] is True
    assert "earlier tool" in results[1]["output"] and not os.path.exists(marker)


@check
def pending_handoff_cancels_later_model_steps_as_lifecycle_work():
    hook = _WorkspaceHandoffHook({"target": "/tmp/next", "ready": False})
    preflight = hook.preflight_tool("read_file", {"path": "x.py"})
    assert preflight.stop and preflight.kind == "lifecycle"
    assert hook.should_continue_after_stop("end_turn") == {"exclusive": True}, \
        "old-workspace completion hooks must not run after navigation succeeds"


@check
def source_transport_completion_is_not_published_as_a_second_final_answer():
    state = {"target": "/tmp/next", "ready": False}
    seen = []
    sink = _workspace_presentation_sink(state, seen.append)
    update = AssistantText("I found the target.", final=False)
    final = AssistantText("Switching now.", final=True)
    for event in (update, final, TurnEnd("end_turn", 2, {}), TurnCommitted(True, "end_turn")):
        sink(event)
    assert seen == [update], seen
    assert _is_workspace_transport_completion(state, final)
    state["target"] = ""
    sink(AssistantText("Target answer.", final=True))
    assert seen[-1].content == "Target answer."


@check
def subagents_cannot_move_their_parent_workspace():
    assert "change_workspace" in SUBAGENT_EXCLUDED_TOOLS, "a child must not move its parent's workspace"


@check
def an_old_greeting_task_does_not_own_the_next_real_request():
    session = Session(NullMemory(), "switch-greeting")
    session.new_topic("hi how are you")
    assert route_topic_lexical("review the parser", session) == ("new", "")


@check
def workspace_round_trip_preserves_app_continuity_but_not_old_root_projection():
    current_a = Session(NullMemory(), "stable-app-session")
    task_id = current_a.new_topic("review the project")
    state_a = current_a.active()
    state_a.conversation = [{
        "user": "review it", "assistant": "I found the parser issue", "artifact_id": "turn-a",
    }]
    state_a.continuity.pending_proposal = {"question": "Shall I fix it?", "source_artifact": "turn-a"}
    state_a.continuity.previous_evidence_snapshot = {"source_turn_id": "turn-a"}
    state_a.continuity.discourse_focus = [
        {"kind": "subject_focus", "entity": {"label": "Hunter", "kind": "project"}},
        {"collection": "findings", "ordinal": 2, "artifact_id": "turn-a"},
    ]
    state_a.active_files = ["a.py"]
    state_a.findings = ["A parser fact"]
    state_a.finding_source = {"A parser fact": "observed"}
    state_a.intent.add_exact(
        "Never remove compatibility mode", source_artifact="turn-a", authority="user",
    )

    blank_b = Session(NullMemory(), "stable-app-session")
    merged_b = rebase_session_for_workspace(current_a, blank_b)
    assert merged_b.active_id == task_id
    state_b = merged_b.active()
    assert state_b.conversation[0]["assistant"] == "I found the parser issue"
    assert state_b.conversation[0]["artifact_id"] == ""
    assert state_b.continuity.pending_proposal["question"] == "Shall I fix it?"
    assert not state_b.active_files and not state_b.findings
    assert state_b.intent.entries[0].source_artifact is None
    assert state_b.intent.entries[0].source_range is None
    assert state_b.continuity.previous_evidence_snapshot is None
    assert state_b.continuity.discourse_focus == [
        {"kind": "subject_focus", "entity": {"label": "Hunter", "kind": "project"}},
    ]

    # Simulate work and conversation in B, then return to an A checkpoint with the same active task ID.
    state_b.conversation.append({
        "user": "also keep the CLI", "assistant": "Understood", "artifact_id": "turn-b",
    })
    state_b.intent.add_exact("Keep the CLI", source_artifact="turn-b", authority="user")
    state_b.active_files = ["b.py"]
    other_id = merged_b.new_topic("secondary topic")
    merged_b.active().conversation = [{
        "user": "remember secondary", "assistant": "secondary B-era reply", "artifact_id": "turn-b2",
    }]
    merged_b.switch_topic(task_id)
    restored_a = Session(NullMemory(), "stable-app-session")
    restored_a.tasks[task_id] = Slice(); restored_a.tasks[task_id].reset("old A checkpoint")
    restored_a.tasks[task_id].active_files = ["a.py"]
    restored_a.tasks[other_id] = Slice(); restored_a.tasks[other_id].reset("stale secondary A")
    restored_a.active_id = task_id

    merged_a = rebase_session_for_workspace(merged_b, restored_a)
    round_trip = merged_a.active()
    assert [row["assistant"] for row in round_trip.conversation] == [
        "I found the parser issue", "Understood",
    ]
    assert round_trip.continuity.pending_proposal["question"] == "Shall I fix it?"
    assert {entry.verbatim_clause for entry in round_trip.intent.entries} == {
        "Never remove compatibility mode", "Keep the CLI",
    }
    assert all(entry.source_artifact is None for entry in round_trip.intent.entries)
    assert all(entry.source_range is None for entry in round_trip.intent.entries)
    assert round_trip.active_files == ["a.py"] and "b.py" not in round_trip.active_files
    assert merged_a.tasks[other_id].conversation[0]["assistant"] == "secondary B-era reply"

    app_binding = SessionBinding(current_a)
    candidate_binding = SessionBinding(blank_b)
    stable_identity = id(app_binding)
    app_binding.bind(merged_b); candidate_binding.bind(merged_b)
    assert id(app_binding) == stable_identity and app_binding.target is candidate_binding.target


@check
def logical_turn_can_span_bounded_distinct_workspace_edges_without_synthetic_messages():
    from sliceagent.discourse import interpret_turn

    request = "now go to target workspace\nthen explain the end-game design exactly"
    current = Session(NullMemory(), "logical-workspace-session")
    task_id = current.new_topic(request)
    admission = interpret_turn(request, (), task_id=task_id).admission
    record_user(current.active(), request, source_artifact="source-turn", contract=admission)
    current.start_logical_turn(
        logical_id="logical-1", task_id=task_id, request=request,
        source_artifact_id="source-turn", admission=admission, source_workspace="/tmp/source",
    )
    original_turns = current.active().turns
    original_rows = len(current.active().conversation)

    target = Session(NullMemory(), "logical-workspace-session")
    merged = rebase_session_for_workspace(current, target)
    assert merged.workspace_epoch == 1 and merged.logical_turn.id == "logical-1"
    merged.begin_workspace_segment(
        source_artifact_id="target-turn", admission=admission, workspace_path="/tmp/target",
    )
    logical = merged.logical_turn
    assert logical.segment_index == logical.workspace_switches == 1
    assert logical.workspace_epoch == merged.workspace_epoch == 1
    assert merged.active_id == task_id and logical.task_id == task_id
    assert merged.active().intent.current_request == request
    assert merged.active().turns == original_turns
    assert len(merged.active().conversation) == original_rows
    assert merged.active().conversation[-1]["user"] == request
    merged.workspace_epoch += 1
    merged.begin_workspace_segment(
        source_artifact_id="segment-2", admission=admission, workspace_path="/tmp/source",
    )
    merged.workspace_epoch += 1
    try:
        merged.begin_workspace_segment(
            source_artifact_id="repeat-a-to-b", admission=admission,
            workspace_path="/tmp/target",
        )
        raise AssertionError("an already-traversed directed edge must be rejected")
    except RuntimeError as exc:
        assert "repeat transition" in str(exc)
    assert logical.workspace_switches == 2
    for index, path in enumerate(("/tmp/third", "/tmp/fourth"), 3):
        merged.begin_workspace_segment(
            source_artifact_id=f"segment-{index}", admission=admission, workspace_path=path,
        )
        merged.workspace_epoch += 1
    assert logical.workspace_switches == logical.segment_index == 4
    assert len(logical.workspace_edges) == 4
    merged.workspace_epoch += 1
    try:
        merged.begin_workspace_segment(
            source_artifact_id="over-budget", admission=admission, workspace_path="/tmp/sixth",
        )
        raise AssertionError("one logical request must not spin through unbounded workspaces")
    except RuntimeError as exc:
        assert "at most 4" in str(exc)


@check
def durable_logical_ids_do_not_collide_when_a_task_is_resumed_in_a_new_app_session():
    from sliceagent.taskstate import slice_to_task_state, task_state_to_slice

    state = Slice(); state.reset("long-running task")
    record_user(
        state, "first request", source_artifact="artifact-one", source_event_id="event-one",
        logical_id="session-one:1:task", workspace_epoch=0,
    )
    restored = task_state_to_slice(slice_to_task_state(state, "task"))
    record_user(
        restored, "second request", source_artifact="artifact-two", source_event_id="event-two",
        logical_id="session-two:1:task", workspace_epoch=0,
    )
    assert [root.logical_id for root in restored.active_work.request_roots] == [
        "session-one:1:task", "session-two:1:task",
    ]


@check
def segment_and_cross_workspace_transition_identity_are_crash_visible():
    workspace = tempfile.mkdtemp(prefix="segment-store-workspace-")
    store_root = tempfile.mkdtemp(prefix="segment-store-core-")
    transition_root = tempfile.mkdtemp(prefix="segment-transition-")
    target = tempfile.mkdtemp(prefix="segment-target-")
    local = LocalTurnStore(workspace, "segment-session", store_root=store_root)
    active = local.begin(
        task_id="task-1", logical_id="logical-1", user_request="switch and inspect",
        segment_index=0, workspace_epoch=3,
    )
    event = active.journal.snapshot().event("logical-segment")
    assert event["payload"] == {
        "logical_turn_id": "logical-1", "segment_id": "logical-1:segment:0",
        "segment_index": 0, "workspace_epoch": 3,
        "workspace_root": os.path.realpath(workspace),
    }
    local.close()

    transitions = WorkspaceTransitionStore(transition_root)
    row = transitions.prepare(
        session_id="segment-session", logical_turn_id="logical-1", task_id="task-1",
        request="switch and inspect", source_root=workspace, target_root=target,
        source_artifact_id=active.artifact_id, source_segment_index=0,
        source_workspace_epoch=3,
    )
    assert transitions.pending(workspace_root=workspace) == (row,)
    row = transitions.mark_activated(row)
    row = transitions.mark_continuing(row, target_artifact_id="target-artifact")
    # A fresh object simulates restart: the atomic control record remains sufficient to locate both epochs.
    recovered = WorkspaceTransitionStore(transition_root).pending(workspace_root=target)
    assert len(recovered) == 1 and recovered[0].status == "continuing"
    assert recovered[0].target_workspace_epoch == 4
    assert recovered[0].target_artifact_id == "target-artifact"
    transitions.clear(row)
    assert not WorkspaceTransitionStore(transition_root).pending()


@check
def case_insensitive_workspace_aliases_reuse_one_transition_identity():
    source = tempfile.mkdtemp(prefix="transition-case-source-")
    target = tempfile.mkdtemp(prefix="transition-case-target-")
    root = tempfile.mkdtemp(prefix="transition-case-store-")

    def case_insensitive(path):
        return os.path.realpath(path).casefold()

    # Deterministically emulate ntpath.normcase while running this suite on a case-sensitive POSIX host.
    with mock.patch.object(WorkspaceTransitionStore, "_root_identity", side_effect=case_insensitive):
        store = WorkspaceTransitionStore(root)
        first = store.prepare(
            session_id="case-session", logical_turn_id="case-logical", task_id="case-task",
            request="switch once", source_root=source, target_root=target,
            source_artifact_id="case-source-artifact", source_segment_index=0,
            source_workspace_epoch=0,
        )
        retry = store.prepare(
            session_id="case-session", logical_turn_id="case-logical", task_id="case-task",
            request="switch once", source_root=source.swapcase(), target_root=target.swapcase(),
            source_artifact_id="case-source-artifact", source_segment_index=0,
            source_workspace_epoch=0,
        )
        assert retry == first
        assert store.pending(workspace_root=source.swapcase()) == (first,)


@check
def crashed_target_segment_recovers_the_same_source_linked_work_root():
    from dataclasses import replace
    from sliceagent.active_work import WorkGraph
    from sliceagent.discourse import interpret_turn
    from sliceagent.taskstate import task_state_from_checkpoint

    workspace = tempfile.mkdtemp(prefix="segment-recovery-workspace-")
    store_root = tempfile.mkdtemp(prefix="segment-recovery-core-")
    request = "switch here and inspect the architecture"
    graph = WorkGraph().open_request(
        "user-event-global", request, logical_id="logical-cross-workspace", workspace_epoch=0,
    ).seal_current("end_turn", transitioned=True, logical_id="logical-cross-workspace")
    first = LocalTurnStore(workspace, "segment-recovery-session", store_root=store_root)
    active = first.begin(
        task_id="task-cross", logical_id="logical-cross-workspace", user_request=request,
        segment_index=1, workspace_epoch=1,
    )
    admission = replace(
        interpret_turn(request, (), task_id="task-cross").admission,
        request_source=active.artifact_id,
    )
    first.record_admission({
        "action": "workspace_continue", "task_id": "task-cross",
        "logical_turn_id": "logical-cross-workspace", "source_event_id": "user-event-global",
        "segment_index": 1, "workspace_epoch": 1,
        "source_artifact_id": "source-workspace-artifact",
        "active_work": graph.to_records(), "admission": admission.to_dict(),
    })
    first.close()  # crash model: target journal exists, but no target artifact/checkpoint seal

    reopened = LocalTurnStore(workspace, "new-process-session", store_root=store_root)
    recovered = reopened.recover_pending()
    assert len(recovered) == 1 and recovered[0].status == "attached", recovered
    checkpoint = reopened.coordinator.checkpoints.load(reopened.workspace_id, "task-cross")
    state = task_state_from_checkpoint(checkpoint)
    restored = WorkGraph.from_records(state.active_work)
    assert restored == graph
    assert restored.request_roots[0].logical_id == "logical-cross-workspace"
    assert restored.request_roots[0].source_refs[0].event_id == "user-event-global"
    reopened.close()


class _CrossWorkspaceLLM:
    instances = []
    target = ""
    second_target = ""
    second_switch = False

    def __init__(self, model=None, **_kwargs):
        self.model = model or "workspace-test-model"
        self.reasoning = "full"
        self.proxy_used = "direct"
        self._base_url = ""
        self.max_tokens = 100_000
        self.client = SimpleNamespace(api_key="test-key")
        self.calls = []
        self.cache_keys = []
        self.delta_sink = None
        type(self).instances.append(self)

    def set_cache_key(self, key):
        self.cache_keys.append(key)

    def set_delta_sink(self, sink):
        self.delta_sink = sink

    def is_retryable(self, _exc):
        return False

    def complete(self, messages, _tools):
        self.calls.append(json.loads(json.dumps(messages)))
        index = len(self.calls)
        usage = {"prompt_tokens": 8, "completion_tokens": 2}
        if index == 1:
            return AssistantMessage(
                content="", tool_calls=[ToolCall(
                    "switch-call", "change_workspace", {"path": self.target},
                )], usage=usage, finish_reason="tool_calls",
            )
        if index == 2:
            return AssistantMessage(
                content="TRANSPORT ONLY", tool_calls=[], usage=usage, finish_reason="stop",
            )
        if index == 3 and self.second_switch:
            return AssistantMessage(
                content="", tool_calls=[ToolCall(
                    "second-switch-call", "change_workspace", {"path": self.second_target},
                )], usage=usage, finish_reason="tool_calls",
            )
        return AssistantMessage(
            content="TARGET FINAL", tool_calls=[], usage=usage, finish_reason="stop",
        )


def _run_cli_workspace_continuation(
    *, live: bool, fail_target_prepare: bool = False, cancel_after_switch_tool: bool = False,
    second_switch: bool = False,
):
    from sliceagent import cli as cli_mod
    from sliceagent import llm as llm_mod
    from sliceagent import tui as tui_mod

    source = tempfile.mkdtemp(prefix="cli-segment-source-")
    target = tempfile.mkdtemp(prefix="cli-segment-target-")
    second_target = tempfile.mkdtemp(prefix="cli-segment-second-target-")
    cache = tempfile.mkdtemp(prefix="cli-segment-cache-")
    home = tempfile.mkdtemp(prefix="cli-segment-home-")
    request = "go to target workspace and tell me its end-game architecture"
    _CrossWorkspaceLLM.instances = []
    _CrossWorkspaceLLM.target = target
    _CrossWorkspaceLLM.second_target = second_target
    _CrossWorkspaceLLM.second_switch = second_switch
    env = {
        "HOME": home, "SLICEAGENT_CACHE_DIR": cache,
        "LLM_API_KEY": "test-key", "OPENAI_API_KEY": "test-key",
        "AGENT_MODEL": "workspace-test-model", "AGENT_PROXY": "off",
        "AGENT_TUI": "live" if live else "off", "AGENT_SUBAGENT_DEPTH": "0",
        "AGENT_WEB": "0", "AGENT_MONITOR": "", "AGENT_ALLOW_PLUGINS": "0",
        "AGENT_BACKGROUND_REVIEW": "0",
    }
    stdout = io.StringIO()
    live_events = []
    workspace_updates = []

    class _Console:
        def print(self, *values, **_kwargs):
            print(*values, file=stdout)

    class _Sink:
        def __init__(self, signal=None):
            self.signal = signal

        def __call__(self, event):
            live_events.append(event)
            if cancel_after_switch_tool and getattr(event, "name", "") == "change_workspace":
                self.signal.set()

        def on_delta(self, _kind, _text):
            pass

        def subagent_notify(self, _text):
            pass

    class _Input:
        def __init__(self, *_args, **_kwargs):
            pass

        def set_workspace(self, value):
            workspace_updates.append(value)

    def _run_live(**kwargs):
        kwargs["on_ready"](workspace_updates.append, lambda _q, _o: "")
        signal = threading.Event()
        kwargs["run_one_turn"](request, _Sink(signal), signal)

    real_prepare = cli_mod._prepare_workspace_resources
    prepare_calls = {"count": 0}

    def _prepare(*args, **kwargs):
        prepare_calls["count"] += 1
        if fail_target_prepare and prepare_calls["count"] > 1:
            raise RuntimeError("target preparation failed")
        return real_prepare(*args, **kwargs)

    old_cwd = os.getcwd()
    patches = [
        mock.patch.dict(os.environ, env),
        mock.patch.object(sys, "argv", ["sliceagent"]),
        mock.patch.object(sys, "stdin", io.StringIO(request + "\n" if not live else "")),
        mock.patch.object(llm_mod, "OpenAILLM", _CrossWorkspaceLLM),
        mock.patch.object(cli_mod, "_prepare_workspace_resources", _prepare),
    ]
    if live:
        patches.extend([
            mock.patch.object(tui_mod, "tui_enabled", lambda: True),
            mock.patch.object(tui_mod, "make_console", lambda: _Console()),
            mock.patch.object(tui_mod, "make_rich_sink", lambda *_a, **_k: _Sink()),
            mock.patch.object(tui_mod, "TuiInput", _Input),
            mock.patch.object(tui_mod, "run_live", _run_live),
        ])
    try:
        os.chdir(source)
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(stdout):
            for patcher in patches:
                stack.enter_context(patcher)
            cli_mod.main()
    finally:
        os.chdir(old_cwd)
    ledger_dir = os.path.join(cache, "event-ledger")
    ledger_path = os.path.join(ledger_dir, os.listdir(ledger_dir)[0])
    with open(ledger_path, encoding="utf-8") as stream:
        ledger = [json.loads(line) for line in stream if line.strip()]
    from sliceagent.active_work import WorkGraph
    from sliceagent.recovery import root_key

    def _graph(workspace):
        core = os.path.join(cache, "core", root_key(workspace))
        if not os.path.isdir(core):
            return None
        reader = LocalTurnStore(workspace, "inspection", store_root=core, exclusive=False)
        try:
            checkpoints = reader.checkpoints()
            if not checkpoints:
                return None
            return WorkGraph.from_records(checkpoints[-1].thawed_state().get("active_work") or ())
        finally:
            reader.close()
    return {
        "request": request, "source": source, "target": target, "cache": cache,
        "output": stdout.getvalue(), "events": live_events,
        "calls": _CrossWorkspaceLLM.instances[0].calls, "ledger": ledger,
        "source_graph": _graph(source), "target_graph": _graph(target),
    }


@check
def inline_flow_automatically_continues_the_exact_request_and_delivers_once():
    result = _run_cli_workspace_continuation(live=False)
    assert len(result["calls"]) == 3
    target_prompt = json.dumps(result["calls"][2], ensure_ascii=False)
    assert result["request"] in target_prompt
    # json.dumps escapes Windows backslashes; compare against the target's JSON spelling rather than a raw
    # host path substring so this still proves the exact target entered the provider prompt.
    assert json.dumps(result["target"], ensure_ascii=False)[1:-1] in target_prompt
    assert "TRANSPORT ONLY" not in target_prompt, "source transport prose must not become target continuity"
    assert "TARGET FINAL" in result["output"]
    assert "TRANSPORT ONLY" not in result["output"]
    kinds = [row["kind"] for row in result["ledger"]]
    assert kinds == ["user_utterance", "context_transition", "response_delivered"], kinds
    assert len({row["logical_turn_id"] for row in result["ledger"]}) == 1
    assert [row["workspace_epoch"] for row in result["ledger"]] == [0, 1, 1]
    source_root = result["source_graph"].request_roots[0]
    target_root = result["target_graph"].request_roots[0]
    assert source_root.logical_id == target_root.logical_id
    assert source_root.status == "in_progress" and not source_root.output_refs
    assert target_root.status == "delivered"
    assert target_root.output_refs[0].kind == "turn_artifact"


@check
def live_flow_uses_the_same_continuation_protocol_without_a_second_user_echo():
    result = _run_cli_workspace_continuation(live=True)
    finals = [event.content for event in result["events"]
              if isinstance(event, AssistantText) and event.final]
    starts = [event for event in result["events"] if event.__class__.__name__ == "TurnStarted"]
    assert finals == ["TARGET FINAL"], finals
    assert len(starts) == 2 and all(event.request == result["request"] for event in starts)
    assert len(result["calls"]) == 3
    assert [row["kind"] for row in result["ledger"]] == [
        "user_utterance", "context_transition", "response_delivered",
    ]


@check
def failed_target_preparation_rolls_back_without_claiming_transition_or_delivery():
    result = _run_cli_workspace_continuation(live=False, fail_target_prepare=True)
    assert "workspace unchanged" in result["output"] and "target preparation failed" in result["output"]
    assert "TRANSPORT ONLY" not in result["output"] and "TARGET FINAL" not in result["output"]
    assert [row["kind"] for row in result["ledger"]] == ["user_utterance"]
    transition_dir = os.path.join(result["cache"], "workspace-transitions")
    assert not [name for name in os.listdir(transition_dir) if name.endswith(".json")]


@check
def cancellation_after_switch_tool_seals_source_but_never_publishes_target():
    result = _run_cli_workspace_continuation(live=True, cancel_after_switch_tool=True)
    assert len(result["calls"]) == 1, "cancellation must win before a transport-final or target model call"
    assert [row["kind"] for row in result["ledger"]] == ["user_utterance"]
    assert not any(row["kind"] == "context_transition" for row in result["ledger"])
    assert not any(isinstance(event, AssistantText) and event.final for event in result["events"])


@check
def target_model_can_continue_the_same_request_through_a_second_distinct_workspace():
    result = _run_cli_workspace_continuation(live=False, second_switch=True)
    assert len(result["calls"]) == 5
    assert result["output"].count("switching workspace") == 2
    assert result["output"].count("TARGET FINAL") == 1
    assert [row["kind"] for row in result["ledger"]].count("context_transition") == 2


@check
def restart_in_either_endpoint_reuses_the_transition_event_ledger_namespace():
    from sliceagent import cli as cli_mod
    from sliceagent import llm as llm_mod

    source = tempfile.mkdtemp(prefix="boot-transition-source-")
    target = tempfile.mkdtemp(prefix="boot-transition-target-")
    cache = tempfile.mkdtemp(prefix="boot-transition-cache-")
    home = tempfile.mkdtemp(prefix="boot-transition-home-")
    store = WorkspaceTransitionStore(os.path.join(cache, "workspace-transitions"))
    pending = store.prepare(
        session_id="recovered-app-session", logical_turn_id="logical-boot", task_id="task-boot",
        request="switch and continue", source_root=source, target_root=target,
        source_artifact_id="source-artifact", source_segment_index=0, source_workspace_epoch=0,
    )
    pending = store.mark_activated(pending)
    store.mark_continuing(pending, target_artifact_id="target-artifact")
    from sliceagent.persistence import Artifact, ArtifactStore
    from sliceagent.recovery import root_key
    ArtifactStore(os.path.join(cache, "core", root_key(target))).put(Artifact(
        id="target-artifact", kind="turn", workspace_id=root_key(target),
        session_id="recovered-app-session", task_id="task-boot", status="end_turn",
        brief={"request": "switch and continue"},
        structured_body={
            "assistant": "target result",
            "meta": {
                "logical_turn_id": "logical-boot", "segment_id": "logical-boot:segment:1",
                "segment_index": 1, "workspace_epoch": 1, "segment_outcome": "terminal",
                "stop_reason": "end_turn",
            },
        },
    ))
    env = {
        "HOME": home, "SLICEAGENT_CACHE_DIR": cache,
        "LLM_API_KEY": "test-key", "OPENAI_API_KEY": "test-key",
        "AGENT_MODEL": "workspace-test-model", "AGENT_PROXY": "off", "AGENT_TUI": "off",
        "AGENT_SUBAGENT_DEPTH": "0", "AGENT_WEB": "0", "AGENT_MONITOR": "",
        "AGENT_ALLOW_PLUGINS": "0", "AGENT_BACKGROUND_REVIEW": "0",
    }
    _CrossWorkspaceLLM.instances = []
    stdout = io.StringIO()
    old_cwd = os.getcwd()
    try:
        # Relaunching from the source endpoint must repair the target-qualified transition too; requiring the
        # user to guess the already-activated endpoint would leave an immutable ledger gap.
        os.chdir(source)
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(sys, "argv", ["sliceagent"]), \
             mock.patch.object(sys, "stdin", io.StringIO("")), \
             mock.patch.object(llm_mod, "OpenAILLM", _CrossWorkspaceLLM), \
             contextlib.redirect_stdout(stdout):
            cli_mod.main()
    finally:
        os.chdir(old_cwd)
    assert _CrossWorkspaceLLM.instances[0].cache_keys[0] == "recovered-app-session"
    output = stdout.getvalue()
    assert "recovered interrupted workspace continuation (continuing)" in output
    assert source in output and target in output
    ledger_path = os.path.join(cache, "event-ledger", "recovered-app-session.jsonl")
    with open(ledger_path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert [row["kind"] for row in rows] == ["context_transition", "response_delivered"]
    assert rows[0]["payload"]["target_artifact_id"] == "target-artifact"
    assert rows[0]["workspace_id"] == root_key(target)
    assert rows[1]["payload"]["artifact_id"] == "target-artifact"


@check
def same_session_restart_mints_a_distinct_logical_turn_and_retires_the_old_ticket():
    from sliceagent.cli import _mint_logical_turn_id, _retire_recovered_transition

    first = _mint_logical_turn_id("stable-session", 1, "task", nonce="before-restart")
    second = _mint_logical_turn_id("stable-session", 1, "task", nonce="after-restart")
    assert first != second, "a reset live generation must not rebind an earlier request identity"

    source = tempfile.mkdtemp(prefix="retire-source-")
    target = tempfile.mkdtemp(prefix="retire-target-")
    root = tempfile.mkdtemp(prefix="retire-store-")
    store = WorkspaceTransitionStore(root)
    pending = store.prepare(
        session_id="stable-session", logical_turn_id=first, task_id="task",
        request="switch and inspect", source_root=source, target_root=target,
        source_artifact_id="artifact", source_segment_index=0, source_workspace_epoch=0,
    )
    warnings = []
    assert _retire_recovered_transition(store, pending, warnings.append) is None
    assert store.pending() == () and warnings == []


@check
def failed_session_publication_leaves_old_binding_and_task_untouched():
    old = Session(NullMemory(), "stable-app-session")
    old_task = old.new_topic("old live task")
    binding = SessionBinding(old)
    candidate = Session(NullMemory(), "stable-app-session")
    manager = WorkspaceManager(_Bundle(tempfile.mkdtemp(prefix="old-"), []),
                               lambda _target: _Bundle(tempfile.mkdtemp(prefix="new-"), []))

    def fail_before_binding(_bundle):
        rebase_session_for_workspace(binding.target, candidate)
        raise RuntimeError("derived hook construction failed")

    try:
        manager.switch(tempfile.mkdtemp(prefix="target-"), activate=fail_before_binding)
    except RuntimeError:
        pass
    assert binding.target is old and binding.active_id == old_task


if __name__ == "__main__":
    passed = 0
    for fn in CHECKS:
        try:
            fn()
            passed += 1
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(CHECKS)} passed")
    raise SystemExit(0 if passed == len(CHECKS) else 1)
