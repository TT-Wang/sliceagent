"""Workspace handoff regressions: prepare B transactionally, then switch in the same process.

No model, network, process restart, or real MCP. Run: PYTHONPATH=src python tests/test_workspace_switch.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.agents import SUBAGENT_EXCLUDED_TOOLS  # noqa: E402
from sliceagent.cli import (  # noqa: E402
    _WorkspaceHandoffHook,
    _prepare_workspace_resources,
    _resolve_workspace_target,
    WorkspaceManager,
    WorkspaceResources,
)
from sliceagent.config import Config, load_config  # noqa: E402
from sliceagent.hooks import Hooks  # noqa: E402
from sliceagent.interfaces import ToolCall  # noqa: E402
from sliceagent.loop import run_tool_batch  # noqa: E402
from sliceagent.memory import NullMemory  # noqa: E402
from sliceagent.policy import make_policy  # noqa: E402
from sliceagent.session import Session, route_topic_lexical  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def spoken_nav_shorthand_authorizes_disambiguated_sibling_dir():
    # "switch to loom" grants target "loom"; the user disambiguates to loom-app/loom-engine. Navigation
    # (basename_anywhere) accepts a stem→dir at a component boundary; a bare edit target still needs exact.
    from sliceagent.hooks import TurnAuthorityHook as H
    assert H._same_path("loom", "/Users/x/Desktop/loom-app", basename_anywhere=True)
    assert H._same_path("loom", "/Users/x/Desktop/loom-engine", basename_anywhere=True)
    assert H._same_path("loom", "/Users/x/Desktop/loom", basename_anywhere=True)
    assert not H._same_path("loom", "/Users/x/Desktop/looming", basename_anywhere=True)  # no boundary
    assert not H._same_path("loom", "/Users/x/Desktop/app-loom", basename_anywhere=True)
    assert not H._same_path("loom", "/Users/x/loom-app", basename_anywhere=False)  # edit target stays exact


@check
def target_resolution_is_pure_canonical_and_fail_closed():
    root = tempfile.mkdtemp(prefix="switch-root-")
    target = tempfile.mkdtemp(prefix="switch-target-", dir=os.path.dirname(root))
    home_target = tempfile.mkdtemp(prefix="switch-home-", dir=os.path.expanduser("~"))
    old_cwd = os.getcwd()
    relative = os.path.relpath(target, root)
    resolved, error = _resolve_workspace_target(root, relative)
    assert not error and resolved == os.path.realpath(target)
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
            policy=make_policy("letitgo"), schedule_workspace=lambda _path: "",
            notify_subagent=lambda _message: None,
        )
        canonical = os.path.realpath(target)
        assert resources.root == canonical
        assert resources.base_tools.root() == canonical
        assert resources.store.workspace_root == canonical
        assert resources.store.workspace_id
        assert resources.tools is resources.base_tools, "subagents=0 should expose the target host directly"
        if hasattr(resources.retriever, "root"):
            assert resources.retriever.root == canonical
        assert resources.base_tools._artifacts is not None
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
def pending_handoff_blocks_later_model_steps_without_counting_as_stuck():
    hook = _WorkspaceHandoffHook({"target": "/tmp/next", "ready": False})
    decision = hook.authorize_tool("read_file", {"path": "x.py"})
    assert not decision.allow and decision.counts_as_stuck is False
    assert hook.should_continue_after_stop("end_turn") == {"exclusive": True}, \
        "old-workspace completion hooks must not run after navigation succeeds"


@check
def navigation_policy_and_subagents_have_the_right_boundary():
    teen = make_policy("teenager")("change_workspace", {"path": "/tmp/project"})
    baby = make_policy("baby-sitter")("change_workspace", {"path": "/tmp/project"})
    assert teen.allow and not teen.ask, "explicit navigation should not look like shell execution"
    assert baby.ask, "baby-sitter still confirms every control-plane mutation"
    assert "change_workspace" in SUBAGENT_EXCLUDED_TOOLS, "a child must not move its parent's workspace"


@check
def an_old_greeting_task_does_not_own_the_next_real_request():
    session = Session(NullMemory(), "switch-greeting")
    session.new_topic("hi how are you")
    assert route_topic_lexical("review the parser", session) == ("new", "")


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
