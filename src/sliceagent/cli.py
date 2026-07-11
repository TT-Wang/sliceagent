"""sliceagent CLI — a thin event-sink host over the stateless slice core.

The loop only dispatches events; this host wires the sinks (slice-updater, durable
log, terminal output) and the policy hooks (permission gate, optional Oracle/budget).
Other surfaces (TUI, SDK, channels) are just different sinks over the same core.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .events import (
    ApiRetry,
    AssistantText,
    Event,
    LessonSaved,
    SliceBuilt,
    ToolResult,
    TurnCommitted,
    TurnInterrupted,
    TurnPhaseChanged,
    TurnStarted,
    make_dispatcher,
)
from .hooks import (BudgetHook, CompositeHooks, DelegatedClaimCompletionHook,
                    DelegationCompletionHook, ExecutionEvidenceCompletionHook,
                    FrozenEvidenceCutoffHook, GuardrailHook,
                    Hooks, OracleHook, PermissionHook, QualityEvidenceCompletionHook,
                    ReconciliationHook, ToolDecision, TurnAuthorityHook)
from .receipts import (compact_receipt_projection, receipt_completion_label,
                       receipt_summary_parts)


def _load_env(path: str = ".env") -> dict[str, str]:
    """Load unset values and return exactly the repository overlay introduced by this call."""
    applied: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                        v = v[1:-1]   # drop surrounding quotes (common .env convention) so the key isn't literal-quoted
                    key = k.strip()
                    if key not in os.environ:
                        os.environ[key] = v
                        applied[key] = v
    except FileNotFoundError:
        pass
    return applied


LOG_MAX_BYTES = 5 * 1024 * 1024   # rotate the debug log past this (keep one prior)

# Host commands that can write files/preferences, perform network setup, or abandon the owning task. They
# remain closed while an earlier effect is indeterminate, just like model tools. `/cwd` is included because
# its path form publishes a new workspace runtime; the no-argument query stays usable.
_RECONCILIATION_BLOCKED_SLASH = frozenset({
    "/config", "/model", "/mode", "/reasoning", "/undo", "/switch", "/resume", "/learn", "/cwd",
})


def _slash_blocked_by_reconciliation(command: str, argument: str = "") -> bool:
    """Queries remain available; any slash operation that changes state stays behind the gate."""
    return command in _RECONCILIATION_BLOCKED_SLASH and not (
        command == "/cwd" and not (argument or "").strip()
    )


def _resolve_workspace_target(workspace_root: str, path: str) -> tuple[str | None, str]:
    """Resolve one requested workspace without mutating cwd or any live runtime owner."""
    root = os.path.realpath(workspace_root)
    raw = (path or "").strip()
    if not raw:
        return None, f"workspace: {root}"
    if "\x00" in raw:
        return None, "not a directory: path contains a NUL byte"
    try:
        expanded = os.path.expanduser(raw)
        target = os.path.realpath(expanded if os.path.isabs(expanded) else os.path.join(root, expanded))
        if not os.path.isdir(target):
            return None, f"not a directory: {path}"
    except (OSError, ValueError):
        return None, f"not a directory: {path}"
    if target == root:
        return None, f"workspace already active: {root}"
    return target, ""


def _cwd_message(workspace_root: str, path: str = "") -> str:
    """Pure `/cwd` projection used before the host stages a workspace-runtime handoff."""
    target, message = _resolve_workspace_target(workspace_root, path)
    return message or f"workspace switch ready: {target}"


def _fold_chitchat_continuity(state, user_text: str, assistant_text: str) -> None:
    """Keep one bounded exact social adjacency while consuming stale action/evidence continuity."""
    state.continuity.pending_proposal = None
    state.continuity.previous_evidence_snapshot = None
    state.conversation.append({
        "user": str(user_text), "assistant": str(assistant_text), "artifact_id": "",
    })
    state.conversation = state.conversation[-4:]


class _WorkspaceHandoffHook(Hooks):
    """Once a handoff is pending, the model may only finish the current turn."""

    def __init__(self, state: dict):
        self.state = state

    def authorize_tool(self, name, args):
        if self.state.get("target"):
            return ToolDecision(
                False,
                "workspace switch is already scheduled; finish this turn without more tool calls",
                counts_as_stuck=False,
            )
        return ToolDecision(True)

    def should_continue_after_stop(self, stop_reason):
        # A workspace-navigation turn must not run the old workspace's Oracle/plugin completion hooks after
        # the control tool succeeds. The host still requires an ordinary clean stop + durable seal below.
        return {"exclusive": True} if self.state.get("target") else None


@dataclass
class WorkspaceResources:
    """Everything whose truth is tied to one workspace.

    The terminal surface and LLM deliberately do not live here: they are process-owned and survive a
    workspace change.  Keeping workspace owners in one object makes the handoff an atomic pointer swap
    instead of a collection of partially-rebound globals/closures.
    """

    root: str
    config: Any
    session: Any
    store: Any
    sandbox: Any
    retriever: Any
    base_tools: Any
    tools: Any
    skills: Any
    plugin_hooks: tuple = ()
    mcp_runtime: Any = None
    reviewer: Any = None
    episodic: Any = None
    monitor_sink: Any = None
    recovery_results: tuple = ()
    mcp_tool_count: int = 0
    plugin_tool_count: int = 0
    mine_mode: str = "deterministic"
    subagent_depth: int = 0
    _closed: bool = field(default=False, init=False, repr=False)
    _on_log: Callable[[str], None] = field(default=lambda _message: None, repr=False)

    def close(self) -> None:
        """Retire workspace-owned activity without touching the shared UI, LLM, or memory facade."""
        if self._closed:
            return
        self._closed = True

        def safe(label: str, fn) -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — teardown is best-effort, but never silent
                self._on_log(f"warning: {label} failed ({type(exc).__name__}: {exc})")

        if self.reviewer is not None:
            def _join_reviewer() -> None:
                self.reviewer.join(timeout=10)
                worker = getattr(self.reviewer, "_thread", None)
                if worker is not None and worker.is_alive():
                    raise TimeoutError("background review did not stop within 10 seconds")
            safe("background-review join", _join_reviewer)
        safe("tool cleanup", self.base_tools.cleanup)
        if self.mcp_runtime is not None:
            done = threading.Event()

            def _shutdown_mcp() -> None:
                try:
                    safe("MCP shutdown", self.mcp_runtime.shutdown)
                finally:
                    done.set()

            threading.Thread(target=_shutdown_mcp, daemon=True).start()
            if not done.wait(8):
                self._on_log("warning: MCP shutdown timed out after 8s")
        writer = getattr(self.monitor_sink, "writer", None)
        if writer is not None:
            safe("monitor writer", writer.close)
        safe("local state lease", self.store.close)


class WorkspaceManager:
    """Prepare-then-publish workspace resources while preserving process-owned identities.

    Preparation failures leave ``current`` untouched.  ``activate`` is intentionally a tiny publication
    callback (CLI variable delegates + TUI labels/completion); if it fails, the candidate is retired and
    the old pointer remains current.  Old resources are closed only after successful publication.
    """

    def __init__(self, current, prepare: Callable[[str], Any]):
        self.current = current
        self._prepare = prepare
        self._lock = threading.RLock()

    def switch(self, target: str, activate: Callable[[Any], None] | None = None):
        target = os.path.realpath(target)
        if not os.path.isdir(target):
            raise NotADirectoryError(target)
        with self._lock:
            old = self.current
            candidate = self._prepare(target)
            try:
                self.current = candidate
                if activate is not None:
                    activate(candidate)
            except BaseException:
                self.current = old
                try:
                    candidate.close()
                finally:
                    raise
            old.close()
            return candidate


def _workspace_paths(root: str, configured: list[str] | None, *defaults: str) -> list[str]:
    """Resolve project-relative extension paths against an explicit workspace, never process cwd."""
    paths = list(configured) if configured is not None else list(defaults)
    resolved = []
    for path in paths:
        expanded = os.path.expanduser(path)
        resolved.append(os.path.realpath(expanded if os.path.isabs(expanded)
                                         else os.path.join(root, expanded)))
    return list(dict.fromkeys(resolved))


def _workspace_mcp_config(root: str, servers: dict) -> dict:
    """Give every target MCP process an explicit target cwd without mutating global cwd."""
    rooted = {}
    for name, value in (servers or {}).items():
        if not isinstance(value, dict):
            rooted[name] = value
            continue
        conf = dict(value)
        raw_cwd = conf.get("cwd")
        if not raw_cwd:
            conf["cwd"] = root
        else:
            expanded = os.path.expanduser(str(raw_cwd))
            conf["cwd"] = os.path.realpath(expanded if os.path.isabs(expanded)
                                             else os.path.join(root, expanded))
        rooted[name] = conf
    return rooted


def _hydrate_workspace_tasks(store, session, on_log: Callable[[str], None]) -> None:
    """Restore only the selected workspace's checkpoints into its fresh session."""
    from .taskstate import task_state_from_checkpoint, task_state_to_slice

    restored = []
    for checkpoint in store.checkpoints():
        try:
            task_state = task_state_from_checkpoint(checkpoint)
            session.tasks[checkpoint.task_id] = task_state_to_slice(task_state)
            restored.append((checkpoint.updated_at, checkpoint.task_id, task_state.status))
        except Exception as exc:  # noqa: BLE001 — one incompatible task must not hide the others
            on_log(f"local task {checkpoint.task_id} could not be restored "
                   f"({type(exc).__name__}: {exc})")
    candidates = [row for row in restored if row[2] == "indeterminate"] or [
        row for row in restored if row[2] in ("active", "parked")
    ]
    if candidates:
        session.active_id = max(candidates)[1]


def _prepare_workspace_resources(
    root: str,
    *,
    cfg,
    llm,
    memory,
    policy,
    schedule_workspace,
    notify_subagent,
    ask_user=None,
    on_log: Callable[[str], None] = lambda _message: None,
) -> WorkspaceResources:
    """Stage one complete workspace runtime before publishing it.

    Lease acquisition, journal recovery, and checkpoint validation happen before plugin code or MCP
    subprocesses.  Any failure retires the partial candidate and leaves the caller's current runtime intact.
    """
    from .background_review import make_background_reviewer
    from .code_grep import make_glob_tool, make_grep_tool
    from .code_index import make_code_index
    from .config import load_config
    from .hippocampus import make_episode_sink
    from .mcp_client import connect_mcp_servers
    from .memory import make_write_skill_tool
    from .plugins import load_plugins
    from .runtime_persistence import CoreArtifactFS, LocalTurnStore
    from .sandbox import make_sandbox
    from .session import Session, make_topic_tools
    from .skills import make_skill_manager, make_skill_tool
    from .subagent import SubagentHost
    from .text_utils import one_line
    from .tools import LocalToolHost

    root = os.path.realpath(root)
    cfg = cfg or load_config(root)
    store = base_tools = mcp_runtime = reviewer = monitor_sink = None
    try:
        session = Session(memory)
        # The lease/recovery boundary precedes every executable extension surface.
        store = LocalTurnStore(root, session.session_id, exclusive=True)
        recovery_results = tuple(store.recover_pending())
        conflicts = tuple(result for result in recovery_results if result.status == "conflict")
        if conflicts:
            detail = "; ".join(f"{item.artifact_id}: {item.detail}" for item in conflicts)
            raise RuntimeError(f"local recovery found an ambiguous journal ({detail})")
        store.checkpoints()  # validate checkpoint bytes + artifact dependencies before startup effects

        retriever = make_code_index(root)
        sandbox = make_sandbox(
            cfg.sandbox_backend, image=cfg.sandbox_image, network=cfg.sandbox_network,
        )
        base_tools = LocalToolHost(root, sandbox=sandbox)
        base_tools.on_workspace_switch = schedule_workspace
        if ask_user is not None:
            base_tools.on_ask_user = ask_user
        if os.environ.get("AGENT_ADVANCED_TOOLS", "").strip().lower() not in (
            "1", "on", "true", "yes",
        ):
            for name in tuple(base_tools.registry._tools):
                if name.startswith(("proc_", "terminal_")):
                    base_tools.registry.deregister(name)
        for extra in os.environ.get("AGENT_ROOT", "").split(os.pathsep):
            if extra.strip():
                expanded = os.path.expanduser(extra.strip())
                base_tools.add_root(expanded if os.path.isabs(expanded)
                                    else os.path.join(root, expanded))

        skill_roots = _workspace_paths(
            root, cfg.skills_roots,
            os.path.join(root, ".sliceagent", "skills"),
            os.path.join(os.path.expanduser("~"), ".sliceagent", "skills"),
        )
        skills = make_skill_manager(skill_roots)
        plugin_dirs = _workspace_paths(root, cfg.plugin_dirs) if cfg.plugin_dirs else []
        plugin_mcp, plugin_hooks = load_plugins(
            base_tools.registry, skills, plugin_dirs, root=root, config=cfg, on_log=on_log,
        )
        skill_tool = make_skill_tool(skills)
        if skill_tool is not None:
            base_tools.registry.register(skill_tool)
        base_tools.registry.register(make_grep_tool(base_tools))
        base_tools.registry.register(make_glob_tool(base_tools))
        if os.environ.get("AGENT_WEB", "1").strip().lower() not in ("0", "off", "false", "no"):
            from .web import make_web_tools
            for web_tool in make_web_tools(base_tools):
                base_tools.registry.register(web_tool)
        base_tools.registry.register(make_write_skill_tool())

        all_mcp = {**cfg.mcp_servers, **plugin_mcp}
        _connected, mcp_runtime = connect_mcp_servers(
            base_tools.registry, _workspace_mcp_config(root, all_mcp), on_log=on_log,
            page_out=base_tools._page_out,
        )
        mcp_tool_count = sum(
            1 for entry in base_tools.registry._tools.values() if entry.source == "mcp"
        )
        plugin_tool_count = sum(
            1 for entry in base_tools.registry._tools.values()
            if entry.source.startswith("plugin:")
        )

        base_tools._artifacts = CoreArtifactFS(store.coordinator.artifacts)
        _hydrate_workspace_tasks(store, session, on_log)
        tools = base_tools
        sub_depth = cfg.subagent_depth
        if sub_depth > 0:
            from .agents import BUILTIN_AGENTS, load_agents
            advanced_agents = os.environ.get("AGENT_ADVANCED_AGENTS", "").strip().lower() in (
                "1", "on", "true", "yes",
            )
            agent_roots = skill_roots + [root, os.path.join(root, ".sliceagent")]
            agents = (load_agents(agent_roots) if advanced_agents
                      else {"explorer": BUILTIN_AGENTS["explorer"]})
            tools = SubagentHost(
                base_tools, llm=llm, retriever=retriever, memory=memory, policy=policy,
                max_depth=sub_depth if advanced_agents else 1, notify=notify_subagent,
                agents=agents, session_id=session.session_id,
                intent_provider=lambda _task, s=session: s.active().intent,
                task_id_fn=lambda s=session: s.active_id or "t-none",
                parent_id_fn=lambda st=store: (
                    st.active.artifact_id if st.active is not None else ""
                ),
                workspace_id=store.workspace_id,
                artifact_store=store.coordinator.artifacts,
                artifact_ref_sink=lambda artifact_id, st=store: st.record_artifact_ref(artifact_id),
                core_mode=not advanced_agents,
            )
        if os.environ.get("AGENT_TOPIC_TOOLS", "").strip().lower() in ("1", "on", "true", "yes"):
            for topic_tool in make_topic_tools(session):
                base_tools.registry.register(topic_tool)
        if getattr(memory, "is_durable", False):
            from .hippocampus import HistoryFS, RosterFS, SubagentFS, make_search_history_tool
            base_tools._history = HistoryFS(memory, session.session_id)
            base_tools._subagents = SubagentFS(memory, session.session_id)
            base_tools._roster = RosterFS(memory)
            base_tools.registry.register(make_search_history_tool(memory, session.session_id))

        reviewer = make_background_reviewer(
            memory, scope=os.path.basename(root) or "default", on_log=on_log,
        )
        episodic = make_episode_sink(
            None,  # collect first; publish to optional semantic history only after the core seal commits
            session_id=session.session_id,
            task_id_fn=lambda s=session: s.active_id or "t-none",
            title_fn=lambda s=session: one_line(s.active().goal, 80) if s.active_id else "",
            outcome_fn=lambda s=session: {"requirements_open": sum(
                1 for requirement in s.active().requirements
                if isinstance(requirement, dict) and not requirement.get("done")
            )} if s.active_id else {},
            collect=True,
        )
        if os.environ.get("AGENT_MONITOR"):
            from .monitor import make_file_monitor_sink
            monitor_sink = make_file_monitor_sink(
                session.session_id,
                context_fn=lambda s=session: {
                    "goal": s.active().goal if s.active_id else "",
                    "topic": s.active_id or "",
                },
            )

        return WorkspaceResources(
            root=root, config=cfg, session=session, store=store, sandbox=sandbox,
            retriever=retriever, base_tools=base_tools, tools=tools, skills=skills,
            plugin_hooks=tuple(plugin_hooks), mcp_runtime=mcp_runtime, reviewer=reviewer,
            episodic=episodic, monitor_sink=monitor_sink, recovery_results=recovery_results,
            mcp_tool_count=mcp_tool_count, plugin_tool_count=plugin_tool_count,
            mine_mode=cfg.mine, subagent_depth=sub_depth, _on_log=on_log,
        )
    except BaseException:
        if reviewer is not None:
            try:
                reviewer.join(timeout=2)
            except Exception:
                pass
        if base_tools is not None:
            try:
                base_tools.cleanup()
            except Exception:
                pass
        if mcp_runtime is not None:
            try:
                mcp_runtime.shutdown()
            except Exception:
                pass
        writer = getattr(monitor_sink, "writer", None)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if store is not None:
            store.close()
        raise


def log_sink(root: str = ".", path: str | None = None):
    from .recovery import root_key, state_dir
    from .safety import redact_text   # strip secrets before they hit the on-disk debug log (off the moat)
    # the debug log lives in the sliceagent STATE dir (~/.sliceagent/logs/<workspace-key>/), NOT scratch/ in the
    # user's workspace — a coding agent must not litter the repo it's working on. `path` overrides (tests).
    path = path or os.path.join(state_dir("logs", root_key(root)), "durable-log.jsonl")

    def _scrub_args(args: dict) -> dict:          # redact string values (edit_file content, inline tokens)
        def _rec(v):                              # RECURSE — a secret nested in a dict/list arg (e.g. an
            if isinstance(v, str):                # MCP tool's {config:{api_key:…}} / {headers:{Authorization:…}})
                return redact_text(v)             # must not reach the on-disk log in plaintext (top-level-only
            if isinstance(v, dict):               # redaction leaked it; sibling hippocampus._clamp already recurses)
                return {k: _rec(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_rec(x) for x in v]
            return v
        return _rec(args or {})

    def sink(e: Event) -> None:
        rec = None
        # REDACT each string field BEFORE serializing (redacting the JSON line itself can corrupt
        # quotes/escapes). A .env read or a token in a command must not land in the log in plaintext
        # — reuses the same safety.redact_text the episodic-persist path uses, so the stores agree.
        if isinstance(e, AssistantText):
            rec = {"role": "assistant", "content": redact_text(e.content)}
        elif isinstance(e, ToolResult):
            rec = {"role": "tool", "name": e.name, "args": _scrub_args(e.args), "full": redact_text(e.output)}
        elif isinstance(e, LessonSaved):
            rec = {"role": "lesson", "title": redact_text(e.title), "content": redact_text(e.content)}
        if rec is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            try:                                  # rotate so the debug log can't grow unbounded
                if os.path.getsize(path) > LOG_MAX_BYTES:
                    os.replace(path, path + ".1")
            except OSError:
                pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return sink


def _plain_arg(args: dict) -> str:
    """The one informative arg for a plain-mode tool line (path/command/pattern/…), whitespace-collapsed."""
    if not isinstance(args, dict):
        return ""
    for k in ("path", "command", "pattern", "name", "ref", "goal", "task"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())[:60]
    v = next((x for x in args.values() if isinstance(x, str) and x.strip()), "")
    return " ".join(v.split())[:60]


def cli_sink(show_slice: bool = False):
    """Plain stdout sink (AGENT_TUI=off / no tui extra / pipes / CI): no color, no spinner — one readable
    line per tool action (✓/✗ + name + primary arg, output for commands/failures) and a delimited answer."""
    _quiet = {"read_file", "list_files"}   # header says enough; their content is noise in a log

    def sink(e: Event) -> None:
        if isinstance(e, SliceBuilt) and show_slice:
            print("\n  ┌─ slice ─────────────")
            print("\n".join("  │ " + ln for ln in e.rendered.splitlines()))
            print("  └─────────────")
        elif isinstance(e, ToolResult):
            mark = "✓" if not e.failing else "✗"
            head = f"  {mark} {e.name} {_plain_arg(e.args)}".rstrip()
            out = " ".join((e.output or "").split())[:140]
            if out and (e.failing or e.name not in _quiet):
                head += f"  — {out}"
            print(head)
        elif isinstance(e, AssistantText):
            if (e.content or "").strip():
                label = "[assistant update]\n" if not e.final else ""
                print(f"\n{label}{e.content}\n")
        elif isinstance(e, ApiRetry):
            print(f"  …retry #{e.attempt} ({e.error})")
        elif isinstance(e, TurnInterrupted):
            print(f"\n[interrupted: {e.reason}]")
        elif isinstance(e, LessonSaved):
            print(f"  💡 learned: {e.title}")
        elif isinstance(e, TurnCommitted):
            if e.ok:
                label = receipt_completion_label(e.receipt, e.stop_reason)
                detail = receipt_summary_parts(e.receipt)
                print(f"  [{label}" + (" · " + " · ".join(detail) if detail else "") + "]")
            else:
                print(f"  [save failed: {e.detail or e.stop_reason}]")
    return sink


def _reasoning_note(llm) -> str:
    """One-line hint after a /model or /reasoning switch, so it's never a silent no-op. Checked in order of
    how badly it fails: (1) the model's name implies a DIFFERENT provider than the current endpoint —
    /model only switches the model STRING, never the endpoint (that's `config --use`), so 'gpt-5.5' while
    still connected to DeepSeek used to 'succeed' silently and fail opaquely on the very next real turn (a
    404 or a 400, always caught by the generic handler as an unhelpful 'internal error'); (2) the requested
    reasoning effort has no route on this (model, endpoint) pair; (3) high/max needs /v1/responses."""
    from .model_catalog import capability, likely_endpoint_mismatch
    base = getattr(llm, "_base_url", "")
    home = likely_endpoint_mismatch(llm.model, base)
    if home:
        label = {"openai": "an OpenAI", "deepseek": "a DeepSeek",
                 "moonshot": "a Moonshot", "anthropic": "an Anthropic"}.get(home, f"a {home}")
        return (f"note: {llm.model} looks like {label} model, but you're connected to {base or '(default)'}"
                f" — it will likely fail on your next message. Add {label} provider with /config, then pick "
                f"its model from the /model menu (the menu switches the endpoint too; typed /model doesn't).")
    eff = (getattr(llm, "reasoning", "full") or "full").lower()
    if eff == "full":
        return ""
    if not capability(llm.model, base).supports_reasoning_effort:
        return f"note: {llm.model} has no reasoning-effort knob — it runs at the provider default."
    return "high/max run WITH tools via /v1/responses." if eff in ("high", "max") else ""


def _ws_name(path: str) -> str:
    """Short display name for the workspace shown in the status bar — the folder's basename, or '~' when the
    workspace is the home dir (so a home-launched session doesn't read as the username)."""
    p = os.path.realpath(path or ".")
    return "~" if p == os.path.realpath(os.path.expanduser("~")) else (os.path.basename(p) or p)


def _env_from_config(c, pid: str | None = None) -> None:
    """Populate LLM_API_KEY/LLM_BASE_URL from config (ENV still wins). A prefs-pinned provider `pid`
    binds AS A UNIT: its key with ITS OWN endpoint — an absent base_url means the SDK default (OpenAI),
    NOT the default provider's base_url (that fallback cross-wired an openai key onto the deepseek
    endpoint after `/model`-hop + restart when default_provider pointed elsewhere)."""
    tbl = (c.providers() or {}).get(pid) if pid else None
    if isinstance(tbl, dict) and tbl.get("api_key"):
        key, base = tbl["api_key"], tbl.get("base_url") or ""
    else:                                    # no/unusable pin → the default provider's own pairing
        key, base = c.api_key, c.base_url
    for _env, _val in (("LLM_API_KEY", key), ("LLM_BASE_URL", base)):
        if not os.environ.get(_env) and _val:
            os.environ[_env] = _val


def main() -> None:
    # Host subcommands are handled BEFORE .env, the key gate, workspace leases, plugins, or MCP. In particular,
    # `sliceagent update` must never ingest repository configuration or accidentally boot an agent session.
    _argv = sys.argv[1:]
    if _argv and _argv[0] in (
        "init", "config", "update", "upgrade", "help", "--help", "-h", "version", "--version", "-V",
    ):
        from .onboarding import dispatch as _dispatch
        sys.exit(_dispatch(_argv))

    _project_env_overlay = _load_env()
    # config-persisted key/endpoint (written by `sliceagent init`) populate the env BEFORE the gate, so a
    # configured user never has to export anything; ENV still wins for one-off overrides.
    from .config import load_config, load_prefs
    _boot_prefs = load_prefs()   # the last /model choice may pin a PROVIDER (endpoint+key), not just a model

    def _key_present() -> bool:
        return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
                    or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MOONSHOT_API_KEY"))

    def _first_run_setup(reason: str) -> bool:
        """First-run UX: a bare interactive `sliceagent` with nothing configured drops STRAIGHT into the
        init wizard instead of bouncing the user to a separate command. Non-interactive (piped/CI) keeps
        the print-and-exit gate — never prompt into a pipe. Returns True when the wizard completed."""
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
        print(f"  {reason} — starting guided setup.\n")
        from .onboarding import run_init
        return run_init() == 0

    cfg = load_config()
    # A prefs `provider` pin whose [providers.<id>] table was since removed from config.toml is STALE:
    # its key/endpoint already fall back to the default provider (see _env_from_config), but its pinned
    # MODEL would still win at the resolution below and get sent to the wrong endpoint (model_not_found
    # every turn). Drop the whole stale pin so model/endpoint resolve together from the live config.
    if _boot_prefs.get("provider") and _boot_prefs["provider"] not in (cfg.providers() or {}):
        from .config import save_prefs as _sp
        _sp({"provider": None, "model": None})       # None DELETES (see save_prefs)
        _boot_prefs.pop("provider", None); _boot_prefs.pop("model", None)
    _env_from_config(cfg, _boot_prefs.get("provider"))
    if not _key_present() and _first_run_setup("Welcome! No API key configured yet"):
        cfg = load_config()          # the wizard just wrote ~/.sliceagent/config.toml — pick it up
        _env_from_config(cfg)
    if not _key_present():
        print("No API key found. Run `sliceagent init` for guided setup, or set LLM_API_KEY (e.g. in a .env file).")
        sys.exit(1)
    # validate enum env vars (warn + use default; never crash) — a typo'd AGENT_POLICY is now visible.
    from .envspec import validate_env
    for _w in validate_env():
        print(f"  · config warning: {_w}")

    from .llm import OpenAILLM
    from .loop import run_turn
    from .memory import make_memory
    from .oracle import CommandOracle
    from .policy import CONFIRMS, legacy_warning, make_policy, policy_label, resolve_policy_mode
    from .session import route, route_topic_lexical
    from .pfc import consolidate_checkpoint, record_user, slice_sink
    from .seed import make_build_slice
    from .text_utils import one_line

    # cfg already loaded above (for the config→env key population + the gate)
    root = os.getcwd()
    _workspace_handoff = {"target": "", "ready": False}

    def _schedule_workspace(target: str) -> str:
        existing = str(_workspace_handoff.get("target") or "")
        if existing and existing != target:
            return f"workspace switch already scheduled for {existing}"
        _workspace_handoff.update(target=target, ready=False)
        return ""

    from .config import load_prefs, save_prefs
    _prefs = load_prefs()
    # mode resolution: explicit env wins, then the saved /mode choice, then config (default teenager).
    _raw_policy = os.environ.get("AGENT_POLICY") or _prefs.get("policy") or cfg.policy
    canonical = resolve_policy_mode(_raw_policy) or "teenager"
    _pol_warn = legacy_warning(_raw_policy)   # loud note if a legacy name (e.g. guard) was used — no silent downgrade
    if _pol_warn:
        print(f"  · {_pol_warn}")
    mine_mode = cfg.mine           # deterministic | llm | off
    sub_depth = cfg.subagent_depth  # 0 disables delegation
    # SUBAGENT/base policy never prompts (no human in a spawned turn) → a confirm-mode runs as let-it-go for
    # them (still blocks catastrophic). The MAIN agent's confirming policy is built at the hook below.
    policy = make_policy("letitgo" if CONFIRMS.get(canonical) else canonical)
    # model + reasoning resolution: explicit env wins, then the saved /model choice (prefs), then config.
    _model = os.environ.get("AGENT_MODEL") or _prefs.get("model") or cfg.model
    if not _model and _first_run_setup("No model configured yet"):
        cfg = load_config()          # wizard picks provider+model → re-resolve from the fresh config
        _env_from_config(cfg)
        _model = os.environ.get("AGENT_MODEL") or _prefs.get("model") or cfg.model
    if not _model:   # no built-in default — the user picks the model (parallels the API-key gate above)
        print("No model configured. Run `sliceagent init` to pick a provider + model, "
              "or set AGENT_MODEL to your model name.")
        sys.exit(1)
    llm = OpenAILLM(model=_model)
    if _prefs.get("reasoning") and not (os.environ.get("AGENT_REASONING") or os.environ.get("AGENT_THINKING")):
        llm.reasoning = str(_prefs["reasoning"]).lower()   # apply the saved /reasoning choice
    memory = make_memory()  # memem if available + a vault is configured, else NullMemory
    # Process-owned bridges stay stable while their workspace targets are replaced.
    _sub_render: dict = {"fn": None}
    def _notify_subagent(text):
        fn = _sub_render["fn"]
        if fn is not None:
            fn(text)
    _ask_user_bridge: dict = {"fn": None}

    def _stamp_navigation_authority() -> None:
        """The user just answered the agent's own in-turn question — fresh, explicit consent. For the
        reversible NAVIGATION tier that is enough to authorize a workspace switch this turn: a turn contract
        frozen as `uncertain` from an ambiguous opening request ("loom app") would otherwise block
        change_workspace even after the user clearly disambiguated. Only navigation is enabled — the grant's
        tool list is change_workspace alone, so destructive effects still need their own grant."""
        try:
            from dataclasses import replace
            from .intent import EffectGrant
            intent = session.active().intent
            admission = intent.turn_admission
            if any(str(getattr(g, "operation", "")) == "workspace.navigate" for g in admission.effect_grants):
                return
            intent.turn_admission = replace(admission, effect_grants=(
                *admission.effect_grants, EffectGrant("workspace.navigate", ("change_workspace",))))
        except Exception:  # best-effort; never break the ask_user round-trip
            pass

    def _workspace_ask_user(question, options):
        fn = _ask_user_bridge.get("fn")
        if fn is None:
            return "(no interactive answer; proceed with a stated assumption)"
        answer = fn(question, options)
        if answer and answer != "(no answer)":
            _stamp_navigation_authority()
        return answer

    def _workspace_log(message: str) -> None:
        print(f"  · {message}")

    def _prepare_workspace(target: str, initial_cfg=None) -> WorkspaceResources:
        from .config import load_config as _load_workspace_config
        if initial_cfg is not None:
            return _prepare_workspace_resources(
                target, cfg=initial_cfg, llm=llm, memory=memory, policy=policy,
                schedule_workspace=_schedule_workspace, notify_subagent=_notify_subagent,
                ask_user=_workspace_ask_user, on_log=_workspace_log,
            )
        # Repository .env is a launch overlay, not process identity. Stage the target with A's injected values
        # temporarily absent, then restore A until atomic publication. We intentionally do not auto-load B's
        # .env during navigation; the live model binding stays process-owned and no new repo code/config gains
        # ambient authority merely because the user moved the workspace frame.
        saved_overlay = {
            key: os.environ.pop(key) for key in tuple(_project_env_overlay) if key in os.environ
        }
        try:
            target_cfg = _load_workspace_config(target)
            return _prepare_workspace_resources(
                target, cfg=target_cfg, llm=llm, memory=memory, policy=policy,
                schedule_workspace=_schedule_workspace, notify_subagent=_notify_subagent,
                ask_user=_workspace_ask_user, on_log=_workspace_log,
            )
        finally:
            os.environ.update(saved_overlay)

    try:
        workspace = _prepare_workspace(root, cfg)
    except Exception as exc:  # noqa: BLE001 — no partial workspace runtime may reach the prompt
        print(f"Workspace could not start safely: {type(exc).__name__}: {exc}")
        try:
            memory.close()
        except Exception:
            pass
        sys.exit(2)
    workspace_manager = WorkspaceManager(workspace, _prepare_workspace)
    session, local_store = workspace.session, workspace.store
    sandbox, retriever = workspace.sandbox, workspace.retriever
    base_tools, tools, skills = workspace.base_tools, workspace.tools, workspace.skills
    plugin_hooks, mcp_runtime = list(workspace.plugin_hooks), workspace.mcp_runtime
    reviewer, episodic, monitor_sink = workspace.reviewer, workspace.episodic, workspace.monitor_sink
    mine_mode, sub_depth = workspace.mine_mode, workspace.subagent_depth
    mcp_tool_count, plugin_tool_count = workspace.mcp_tool_count, workspace.plugin_tool_count
    llm.set_cache_key(session.session_id)
    for _recovered in workspace.recovery_results:
        print(f"  · recovered local artifact {_recovered.artifact_id} ({_recovered.status})")
    _pending_seal_records: dict[str, tuple[int, dict]] = {}

    def _preview_turn_admission(text: str, state, task_id: str):
        """Purely orient one request against the prospective task.

        No focus, proposal, intent, or active-task field changes here.  The resulting AdmissionPreview is
        journaled before it is installed, so every model/tool consumer observes the same immutable reading.
        """
        from .discourse import interpret_turn
        from .intent import EntityRef

        focus = list(state.continuity.discourse_focus)
        # A new task that explicitly discusses this project needs a stable subject even before any numbered
        # assistant output exists.  The workspace default is a source-linked candidate, not guessed user text.
        if not any(isinstance(item, dict) and item.get("kind") == "subject_focus" for item in focus) \
                and re.search(r"\b(?:this|the|current)\s+(?:project|repo(?:sitory)?|codebase|workspace)\b",
                              text, re.IGNORECASE):
            try:
                project_label = os.path.basename(os.path.realpath(tools.root())) or "current project"
            except Exception:
                project_label = "current project"
            focus.append({
                "kind": "subject_focus",
                "entity": EntityRef(
                    project_label, kind="project", source="workspace_default",
                ).to_dict(),
            })
        return interpret_turn(
            text,
            local_store.coordinator.artifacts.list_all(),
            task_id=task_id,
            session_id=session.session_id,
            recent_assistant=(
                str(exchange.get("assistant") or "")
                for exchange in state.conversation if exchange.get("assistant")
            ),
            focus=focus,
            pending_proposal=state.continuity.pending_proposal,
            previous_evidence_snapshot=state.continuity.previous_evidence_snapshot,
            current_generation=session.turn_generation,
        )

    def _begin_local_turn(text: str, preview, *, action: str, task_id: str, state):
        """Journal one admission, then publish its task/focus/intent changes exactly once."""
        logical_id = f"{task_id}:{state.turns + 1}"
        active = local_store.begin(
            task_id=task_id or "t-none", logical_id=logical_id, user_request=text,
        )
        session.turn_generation += 1
        from dataclasses import replace
        admission = replace(preview.admission, request_source=active.artifact_id)
        local_store.record_admission({
            "action": action,
            "task_id": task_id,
            "admission": admission.to_dict(),
            "focus": [dict(item) for item in preview.focus],
            "consume_pending_proposal": bool(preview.consume_pending_proposal),
        })
        if action in ("new", "resume"):
            session.activate_prepared_topic(task_id, state)
        session.turn_task_id = task_id
        if action != "new":
            session.continue_topic(
                text, resume=(action == "resume"), admission=admission, install_intent=False,
            )
        record_user(state, text, source_artifact=active.artifact_id, contract=admission)
        # Focus and proposal consumption are part of this admitted turn, never preview-time side effects.
        state.continuity.discourse_focus = list(preview.focus)
        state.runtime.source_projections = tuple(dict(item) for item in preview.projections)
        from .discourse import make_evidence_snapshot
        state.continuity.previous_evidence_snapshot = make_evidence_snapshot(
            admission, state.runtime.source_projections, active.artifact_id,
            snapshot_basis=preview.snapshot_basis,
            source_generation=session.turn_generation,
        )
        if preview.consume_pending_proposal:
            state.continuity.pending_proposal = None
        # A refaulted discourse item becomes a real immutable dependency of this turn/checkpoint. This keeps
        # the exact source alive without copying it into the active slice or transcript.
        referenced = list(preview.referenced_artifact_ids)
        for referent in getattr(admission, "referents", ()):
            artifact_id = str(
                getattr(getattr(referent, "anchor", None), "artifact_id", "")
                or (referent.get("artifact_id") if isinstance(referent, dict) else "")
                or ""
            )
            if artifact_id:
                referenced.append(artifact_id)
        for artifact_id in dict.fromkeys(referenced):
            if artifact_id and artifact_id != active.artifact_id \
                    and local_store.coordinator.artifacts.exists(artifact_id):
                local_store.record_artifact_ref(artifact_id)
        return admission

    def _admit_routed_turn(text: str, action: str, task_id: str = ""):
        """Prepare a prospective task, preview intent, then cross the durable admission boundary."""
        if action == "new" or session.active_id is None:
            task_id, state = session.prepare_new_topic(text)
            action = "new"
        elif action == "resume":
            task_id, state = session.prepare_switch_topic(task_id)
        else:
            task_id, state = session.active_id or "t-none", session.active()
            action = "continue"
        preview = _preview_turn_admission(text, state, task_id)
        return _begin_local_turn(text, preview, action=action, task_id=task_id, state=state)

    def _seal_local_turn(stop_reason: str, event_dispatch=None) -> bool:
        """Required artifact-first seal; emit completion only after durable activation succeeds."""
        emit = event_dispatch or dispatch
        emit(TurnPhaseChanged(
            "saving",
            "saving checkpoint" if stop_reason == "end_turn" else f"saving {stop_reason} state",
        ))
        active = local_store.active

        def _failed(detail: str) -> bool:
            emit(TurnCommitted(ok=False, stop_reason=stop_reason, detail=one_line(detail, 180)))
            return False

        if active is None:
            return _failed("no active local turn to save")
        artifact_id = active.artifact_id
        target = session.tasks.get(active.task_id)
        if target is None:
            return _failed(f"starting task {active.task_id!r} no longer exists")
        closed = episodic.take_last_record() if episodic is not None else None
        if closed is not None:
            _pending_seal_records[active.artifact_id] = closed
        pending_record = _pending_seal_records.get(active.artifact_id)
        history_turn = pending_record[0] if pending_record is not None else max(1, target.turns)
        record = pending_record[1] if pending_record is not None else {
            "title": one_line(target.goal, 80), "steps": [], "note": "",
            "markdown": "", "meta": {"stop_reason": stop_reason, "files": []},
        }
        # Prepare the next state on a copy. The live task is published only after artifact+checkpoint commit,
        # so a storage failure cannot half-seal working memory and then let a newer turn overtake it.
        import copy
        sealed_target = copy.deepcopy(target)
        sealed_target.seal()
        from dataclasses import asdict
        from .taskstate import slice_to_task_state, task_state_from_checkpoint
        task_status = ("indeterminate" if sealed_target.reconciliation_required else
                       ("active" if stop_reason == "end_turn" else "parked"))
        task_state = slice_to_task_state(
            sealed_target, active.task_id, session_id=session.session_id,
            # A clean model turn does not close the user's task. Only explicit task-boundary operations do.
            status=task_status,
        )
        workspace_versions = {}
        try:
            from .workspace_revision import WorkspaceRevision
            paths = sorted(set(sealed_target.active_files) | set(sealed_target.edited_files))
            workspace_versions = WorkspaceRevision.capture(root, paths).as_dict() if paths else {}
        except Exception:  # noqa: BLE001 — an out-of-root/vanished path is re-observed next turn
            workspace_versions = {}
        try:
            source_refs = tuple(dict.fromkeys(
                source for source in (
                    sealed_target.task.goal_source,
                    *(entry.source_artifact for entry in sealed_target.intent.entries),
                )
                if source and source != active.artifact_id
                and local_store.coordinator.artifacts.exists(source)
            ))
            local_store.seal(
                state=asdict(task_state), record=record, status=stop_reason,
                title=record.get("title", ""), summary=record.get("note", ""),
                files=tuple((record.get("meta") or {}).get("files") or ()),
                refs=source_refs,
                error="" if stop_reason == "end_turn" else stop_reason,
                workspace_versions=workspace_versions,
            )
        except Exception as _e:  # noqa: BLE001 — keep both journals for replay; never claim durability
            try:
                recovered = local_store.recover_active_seal()
            except Exception:  # noqa: BLE001 — persistent storage failure remains an honest hard stop
                recovered = None
            if recovered is None or recovered.status not in ("replayed", "attached", "cleaned"):
                return _failed(
                    f"local seal incomplete ({type(_e).__name__}: {_e}); recovery journal retained"
                )
        # The journal-completeness safeguard may strengthen an apparently ordinary stop to INDETERMINATE
        # while committing (for example, ToolStarted was durable but Ctrl-C prevented ToolResult). Merge the
        # committed safety gate back into the live copy, or this process could immediately bypass a gate that
        # rehydration correctly enforces. Keep the live copy's richer within-session continuity/working-set fields,
        # which TaskState intentionally does not serialize; the optional mirror receives committed TaskState.
        try:
            committed = local_store.coordinator.checkpoints.load(local_store.workspace_id, active.task_id)
            if committed is None:
                raise RuntimeError("committed checkpoint is missing after local seal")
            task_state = task_state_from_checkpoint(committed)
            sealed_target.reconciliation_required = task_state.reconciliation_required
            sealed_target.reconciliation_targets = list(task_state.reconciliation_targets)
            sealed_artifact = local_store.coordinator.artifacts.get(artifact_id)
            sealed_body = sealed_artifact.to_dict().get("structured_body")
            sealed_body = sealed_body if isinstance(sealed_body, dict) else {}
            receipt_projection = compact_receipt_projection(sealed_body.get("turn_receipt"))
        except Exception as exc:  # noqa: BLE001 — do not continue from a state weaker than durable truth
            return _failed(f"committed local state could not be activated ({type(exc).__name__}: {exc})")
        session.tasks[active.task_id] = sealed_target
        session.turn_task_id = None
        _pending_seal_records.pop(active.artifact_id, None)
        if getattr(memory, "is_durable", False):
            try:
                memory.append_episode(
                    session.session_id, active.task_id, history_turn,
                    sealed_artifact.to_dict()["structured_body"],
                )
            except Exception as exc:  # noqa: BLE001 — optional history cannot invalidate the core seal
                print(f"  · legacy history mirror failed ({type(exc).__name__}: {exc})")
            try:
                memory.checkpoint_task(task_state)  # derived compatibility view, after core commit
            except Exception as exc:  # noqa: BLE001 — optional memory cannot invalidate the core seal
                print(f"  · legacy task mirror failed ({type(exc).__name__}: {exc})")
        emit(TurnCommitted(
            ok=True,
            stop_reason=stop_reason,
            artifact_id=artifact_id,
            detail="checkpoint saved" if stop_reason == "end_turn" else f"{stop_reason} state saved",
            receipt=receipt_projection,
        ))
        return True

    class _DurabilityStop(RuntimeError):
        stop_session = True

    # optional rich TUI (the `tui` extra). Output via Rich, input via prompt_toolkit — temporally
    # separate from the synchronous run_turn, so no patch_stdout/threading. Off when piped (eval).
    _tui = None
    _stats = {"model": llm.model, "policy": policy_label(canonical), "topic": "",
              "workspace": _ws_name(root), "tokens": 0}
    try:
        from . import tui as _tuimod
        if _tuimod.tui_enabled():
            _tui = _tuimod
    except Exception:
        _tui = None
    _console = _tui.make_console() if _tui else None   # themed: no black-bg highlight on inline `code`/paths

    # DEFAULT UI = the inline rich+prompt_toolkit REPL: it stays in the NORMAL terminal buffer, so native
    # copy / paste / scrollback work on ANY terminal (incl. macOS Terminal.app), with a pinned composer
    # (patch_stdout, which provides a pinned static region above a live-updating composer) and
    # streaming replies. AGENT_TUI=off → plain stdout (handled in tui_enabled). AGENT_TUI=live → the
    # always-pinned live composer: the bordered box stays at the bottom EVEN WHILE the agent streams (output
    # prints above it). Opt-in/experimental; the default REPL (box between turns) is the proven path, and
    # live falls back to it if it can't start.
    tui_env = os.environ.get("AGENT_TUI", "").strip().lower()
    use_live = (_tui is not None and tui_env == "live")

    # wire the ask_user capability to a real prompt when interactive — NOT in live mode, where a worker-
    # thread console.input() would contend with the pinned prompt_toolkit app for stdin and HANG.
    if not use_live and (_tui or sys.stdin.isatty()):
        # wire the ask_user capability to a real prompt when interactive (TUI rich prompt, or plain
        # input); headless/eval — AND live mode, where a worker-thread console.input() would contend with
        # the pinned prompt_toolkit app for stdin and HANG — keep the non-interactive default.
        def _ask_user(question, options):
            if _tui:
                return _tui.ask_user(_console, question, options)
            print(f"\n  ❓ {question}")
            for i, o in enumerate(options or [], 1):
                print(f"     {i}. {o}")
            try:
                a = input("  your answer ▸ ").strip()
            except (EOFError, KeyboardInterrupt):
                return "(no answer)"
            if options and a.isdigit() and 1 <= int(a) <= len(options):
                return options[int(a) - 1]
            return a or "(no answer)"
        _ask_user_bridge["fn"] = _ask_user

    # sinks: update the active slice from tool results, cache the turn, persist, print (no per-turn
    # miner — distillation is cache-only at session end via memory.consolidate).
    reducer = slice_sink(session)

    # Keep the store lookup behind tiny routers so required journaling and reduction always dereference the
    # currently-published workspace. The delegates themselves stay stable across in-process handoffs.
    def _journal_event(event):
        local_store.observe_event(event)

    def _reduce_event(event):
        """Transactionally publish in-memory reduction with its durable applied-effect IDs."""
        if not isinstance(event, ToolResult) or event.outcome is None:
            reducer(event)
            return
        import copy
        task_id = session.active_id
        before = copy.deepcopy(session.tasks.get(task_id)) if task_id in session.tasks else None
        try:
            reducer(event)
            local_store.observe_reduction(event)
        except Exception:
            if before is not None:
                session.tasks[task_id] = before
            raise

    # optional: the moat-MEASURING cost sink (AGENT_METRICS=1). Accumulates the per-turn FRESH-input
    # curve (should stay flat as the conversation grows) + cache-hit rate + reliability counters; the
    # summary prints at session end. Pure observer — eval/default path untouched.
    metrics = None
    if os.environ.get("AGENT_METRICS"):
        from .metrics import make_metrics_sink
        metrics = make_metrics_sink()
    # EXACTLY ONE renderer is wired: the rich+prompt_toolkit sink (TUI), OR the plain stdout sink
    # (headless/eval). Never two.
    if _tui:
        _rich = _tui.make_rich_sink(_console, _stats, await_commit=True)
        _presentation_sink = _rich
        llm.set_delta_sink(_rich.on_delta)   # STREAM completions live into the rich TUI spinner
        # child agent activity → one dynamic spinner line. NOT in live mode: a rich console Status would
        # fight the pinned prompt_toolkit Application for the screen (garbled output) — let the spawn tool's
        # result line carry the child summary instead, as the plain/headless path does.
        _sub_render["fn"] = None if use_live else _rich.subagent_notify
    else:
        _presentation_sink = cli_sink(cfg.show_slice)
    # optional: feed the live web monitor (AGENT_MONITOR=1) — eval path untouched. Writes per-step
    # snapshots to the shared monitor dir; view them in the STANDING server (python -m sliceagent.monitor),
    # which stays up across sessions and goes idle when none is running.
    if monitor_sink is not None:
        from .monitor import _monitor_dir
        print(f"  · slice monitor: writing to {_monitor_dir()} — view at the persistent server "
              "(run: python -m sliceagent.monitor)")

    def _make_workspace_dispatch():
        sinks = []
        if episodic is not None:
            sinks.append(episodic)
        sinks.append(log_sink(root))
        if metrics is not None:
            sinks.append(metrics)
        sinks.append(_presentation_sink)
        if monitor_sink is not None:
            sinks.append(monitor_sink)
        # Execution truth is journaled before reduction; applied transition IDs are journaled only after the
        # reducer succeeds. Presentation/metrics remain isolated observers.
        return make_dispatcher(*sinks, required=(_journal_event, _reduce_event))

    dispatch = _make_workspace_dispatch()
    # Social fast-path messages are deliberately outside the task/turn lifecycle. Sending them through the
    # required dispatcher would overwrite the previous conversation exchange and bleed into the next
    # episode journal even though no local turn was opened.
    chitchat_dispatch = make_dispatcher(_presentation_sink)

    # policy hooks (the seam is always wired; default 'guard' blocks catastrophic commands)
    def _ask(name, args, reason):  # interactive resolver for AGENT_POLICY=ask
        detail = args.get("command") or args.get("path") or args.get("code", "")
        if use_live:  # the pinned prompt_toolkit app owns stdin → a Rich confirm would hang; deny (safe default)
            return "no"
        if _tui:  # synchronous mid-run (no pt app live) → a Rich confirm is safe
            return _tui.confirm(_console, name, str(detail), reason)
        if not sys.stdin.isatty():
            return "no"
        ans = input(f"  ⚠ allow {name} {str(detail)[:60]!r}? ({reason}) [y]es/[n]o/[a]lways: ").strip().lower()
        return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(ans, "no")

    def _live_pid():
        """The configured provider the LIVE llm is actually bound to (endpoint+key match), or None.
        Prefs must pin what's IN EFFECT, not the last menu pick — a typed `/model <name>` keeps the
        endpoint, so blindly re-saving the old pin resurrected a dead endpoint+model pairing at boot."""
        lbase = getattr(llm, "_base_url", "") or ""
        lkey = getattr(llm.client, "api_key", None)
        for p, t in (cfg.providers() or {}).items():
            if isinstance(t, dict) and t.get("api_key") and t["api_key"] == lkey \
                    and (t.get("base_url") or "") == lbase:
                return p
        return None

    def _handle_slash(line):  # TUI navigation palette — wired to existing session ops
        nonlocal cfg          # /config hot-reloads the config after the in-session wizard
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0], (parts[1].strip() if len(parts) > 1 else "")
        if (session.active_id is not None and session.active().reconciliation_required
                and _slash_blocked_by_reconciliation(cmd, arg)):
            _console.print(
                f"  {cmd} blocked: reconcile the prior indeterminate operation first", markup=False,
            )
            return True
        if cmd == "/help":
            _console.print("commands: /config · /model · /mode · /cwd · /learn · /plan · /cost · /update · /threads · "
                           "/plugins · /mcp · /help · /exit\n  (type / for the menu · /config adds LLM "
                           "providers · /cwd shows the workspace; /cwd <path> switches it safely · "
                           "Esc = undo last turn · "
                           "say \"review my changes\" for code_review · @path pins a file)")
        elif cmd == "/config":
            # THE clear user journey: providers are managed INSIDE sliceagent — same wizard as first-run
            # onboarding (provider → model → key → live test), then the new provider shows up in /model.
            if sys.stdin.isatty() and not use_live:
                from .onboarding import run_init
                try:
                    rc = run_init()
                except (EOFError, KeyboardInterrupt):
                    rc = 1
                if rc == 0:
                    from .config import load_config as _reload
                    cfg = _reload(root)                  # hot-reload the currently selected workspace layer
                    provs = cfg.providers()
                    _console.print("  providers configured: "
                                   + (", ".join(f"[bold]{k}[/] ({v.get('model', '?')})"
                                                for k, v in provs.items()) or "(none)"))
                    _console.print("  switch with [bold]/model[/] — it lists your configured providers' models.")
            else:
                _console.print("  /config needs an interactive terminal — run `sliceagent init` instead.")
        elif cmd == "/plan":
            s = session.active() if session.active_id else None
            plan = getattr(s, "plan", None) if s else None
            if not plan:
                _console.print("  (no active plan — the agent sets one with update_plan on multi-step tasks)")
            else:
                mark = {"done": "✓", "in_progress": "▶", "pending": "○"}
                for it in plan:
                    _console.print(f"  {mark.get(it.get('status'), '○')} {it.get('step', '')}", markup=False)
        elif cmd == "/cost":
            from .tui import _saved_dollars
            saved = _saved_dollars(_stats)
            spent = _stats.get("cost", 0.0)
            head = (f"  💰 saved ${saved:.4f} vs full-history (@ {_stats.get('model','?')}, cache-aware)"
                    f"  ·  spent ${spent:.4f}" if saved is not None
                    else f"  💰 {_stats.get('saved_cached_tok', 0):,} tokens saved vs full-history (model price unknown)")
            _console.print(head)
            if metrics is None:
                _console.print("  (per-turn curve off — start with AGENT_METRICS=1 to track it)")
            else:
                s = metrics.summary()
                _console.print(f"  per_turn_fresh={s['per_turn_fresh']} avg={s['avg_turn_fresh']} "
                               f"cache_hit={s['cache_hit_rate']} tools={s['tool_calls']} "
                               f"out={s['output']} retries={s['retries']} overflows={s['overflows']}")
        elif cmd == "/update":
            _console.print("  Updating replaces the running environment. Exit, then run:  sliceagent update")
        elif cmd == "/threads":
            ts = session.open_threads(include_active=True)
            # markup=False: task_id/title are DATA — `[{t.task_id}]` renders as a Rich tag → MarkupError crash.
            _console.print(("  (no topics yet)" if not ts else
                            "\n".join(f"  [{t.task_id}] {t.title} ({t.status})" for t in ts)), markup=False)
        elif cmd in ("/switch", "/resume"):
            if not arg:
                _console.print(f"  usage: {cmd} <task_id>")
            else:
                try:
                    session.switch_topic(arg)
                    _stats["topic"] = one_line(session.active().goal, 40)
                    _console.print(f"  switched to {arg}", markup=False)
                except Exception as exc:  # noqa: BLE001 - show reconciliation gate as well as bad ids
                    _console.print(f"  could not switch: {exc}", markup=False)
        elif cmd == "/undo":
            # markup=False: undo_last() embeds the edited file PATH; a Next.js-style '[id]'/'[...slug]'
            # segment is parsed as a Rich tag → corrupted output or a MarkupError crash.
            if session.active_id is not None and session.active().reconciliation_required:
                _console.print("  undo blocked: reconcile the prior indeterminate operation first")
            else:
                _console.print("  " + base_tools.undo_last(), markup=False)  # revert the last file edit
        elif cmd == "/plugins":
            tools = sorted(e.name for e in base_tools.registry._tools.values()
                           if getattr(e, "source", "") == "plugin")
            # markup=False: plugin dirs are filesystem PATHS that may contain '[...]' (same Rich-tag hazard)
            _console.print(f"  plugin dirs: {', '.join(cfg.plugin_dirs) or '(none configured)'}", markup=False)
            _console.print(f"  plugin tools ({len(tools)}): {', '.join(tools) or '(none loaded)'}", markup=False)
        elif cmd == "/mcp":
            configured = list(cfg.mcp_servers.keys())
            mtools = sorted(e.name for e in base_tools.registry._tools.values()
                            if getattr(e, "source", "") == "mcp")
            if not configured and not mtools:
                _console.print("  no MCP servers configured — add [mcp_servers.<name>] to ~/.sliceagent/config.toml")
            else:
                _console.print(f"  configured servers: {', '.join(configured) or '(none)'}")
                _console.print(f"  connected tools ({len(mtools)}): {', '.join(mtools) or '(none — check startup logs)'}")
        elif cmd == "/mode":
            _menu_ok = sys.stdin.isatty() and not use_live
            chosen = None
            if not arg and _menu_ok:                       # no arg + interactive → open the picker menu
                from .tui import run_selector
                order = ["babysitter", "teenager", "letitgo"]
                rows = [("baby-sitter", "confirm every edit + command"),
                        ("teenager", "auto edits, confirm commands"),
                        ("let-it-go", "auto-run — still blocks catastrophic moves")]
                cur = next((i for i, k in enumerate(order) if policy_label(k) == _stats.get("policy")), -1)
                pick = run_selector("Permission mode", rows, current=cur)
                chosen = order[pick] if pick is not None else None
            elif arg:                                      # typed: /mode teenager
                chosen = resolve_policy_mode(arg)
                if chosen not in ("babysitter", "teenager", "letitgo"):
                    _console.print("  unknown mode — use: baby-sitter | teenager | let-it-go"); return True
            else:                                          # no arg, non-interactive → just show current
                _console.print(f"  mode: [bold]{_stats.get('policy', '?')}[/]   options: baby-sitter · teenager · let-it-go")
                _console.print("    baby-sitter = confirm every edit + command · teenager = auto edits, confirm "
                               "commands · let-it-go = auto (blocks catastrophic)")
            if chosen:
                eff = chosen if (sys.stdin.isatty() and not use_live) or not CONFIRMS.get(chosen) else "letitgo"
                perm_hook.policy = make_policy(eff)
                perm_hook.on_ask = _ask if CONFIRMS.get(eff) else None
                _stats["policy"] = policy_label(eff)
                save_prefs({"policy": eff})   # remember the choice for next launch
                _console.print(f"  → [bold]{policy_label(eff)}[/]"
                               + ("" if eff == chosen else "  (no interactive prompt here → running as let-it-go)"))
        elif cmd == "/model":
            if not arg and sys.stdin.isatty() and not use_live:   # open the two-tier model→reasoning menu
                from .tui import select_model_reasoning
                choice = select_model_reasoning(llm, cfg)
                if choice:
                    _model, _reasoning, _pid = choice
                    _kw = {}
                    if _pid:                              # a CONFIGURED provider → switch endpoint+key too
                        _tbl = (cfg.providers() or {}).get(_pid) or {}
                        if _tbl.get("api_key"):
                            _kw = {"base_url": _tbl.get("base_url") or "", "api_key": _tbl["api_key"]}
                    llm.switch(model=_model, reasoning=_reasoning, **_kw)
                    _stats["model"] = llm.model
                    # pin the provider ACTUALLY in effect (None DELETES a stale pin — see save_prefs)
                    save_prefs({"model": llm.model, "reasoning": llm.reasoning,
                                "provider": _pid or _live_pid()})
                    note = _reasoning_note(llm)
                    _console.print(f"  ✓ model → [bold]{llm.model}[/]"
                                   + (f" @ [bold]{_pid}[/]" if _pid else "")
                                   + f" · reasoning [bold]{llm.reasoning}[/] (saved)"
                                   + (f"\n  {note}" if note else ""))
            elif not arg:
                _console.print(f"  model: [bold]{llm.model}[/]  ·  reasoning: [bold]{llm.reasoning}[/]"
                               f"  ·  net: {getattr(llm, 'proxy_used', 'direct')}")
                _console.print("  switch:  /model <name> [fast|full|high|max]  (same endpoint)")
                provs = cfg.providers()
                if provs:
                    _console.print("  configured providers (pick via the /model menu to switch endpoint too): "
                                   + ", ".join(f"{k}={v.get('model', '?')}" for k, v in provs.items()))
                else:
                    _console.print("  no providers configured yet — add one with [bold]/config[/]")
            else:
                name, *rest = arg.split()
                eff = rest[0].lower() if rest else None
                if eff and eff not in ("fast", "full", "high", "max"):
                    _console.print("  effort must be one of: fast | full | high | max"); return True
                llm.switch(model=name, reasoning=eff)
                _stats["model"] = llm.model
                # typed /model keeps the endpoint — re-resolve the pin against the LIVE binding so a
                # stale provider pin can't resurrect the old endpoint under this model at next boot
                save_prefs({"model": llm.model, "reasoning": llm.reasoning, "provider": _live_pid()})
                note = _reasoning_note(llm)
                _console.print(f"  ✓ model → [bold]{llm.model}[/]"
                               + (f" · reasoning [bold]{llm.reasoning}[/]" if eff else "")
                               + " (saved)" + (f"\n  {note}" if note else ""))
        elif cmd == "/reasoning":
            if arg.lower() not in ("fast", "full", "high", "max"):
                _console.print("  usage: /reasoning <fast|full|high|max>"
                               "   (full = provider default; high/max use /v1/responses for gpt-5)")
            else:
                llm.switch(reasoning=arg)
                save_prefs({"reasoning": llm.reasoning})
                note = _reasoning_note(llm)
                _console.print(f"  ✓ reasoning → [bold]{llm.reasoning}[/] (saved)" + (f"\n  {note}" if note else ""))
        elif cmd == "/cwd":
            target, message = _resolve_workspace_target(base_tools.root(), arg)
            if target is None:
                _console.print("  " + message, markup=False)
            else:
                problem = _schedule_workspace(target)
                if problem:
                    _console.print("  " + problem, markup=False)
                else:
                    _workspace_handoff["ready"] = True  # slash commands run between already-sealed turns
                    _switch_workspace(target)
        else:
            _console.print(f"  unknown command {cmd} (/help)", markup=False)
        return True

    _IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

    def _expand_mentions(text):
        """@path mentions: pin each EXISTING workspace file referenced as @path into
        the slice's OPEN FILES; an @image is ATTACHED as a vision content part for the next turn when the model
        supports vision (else skipped with a hint). Best-effort; leaves the text intact."""
        if "@" not in text or session.active_id is None:
            return
        import re as _re
        from .model_catalog import capability
        from .pfc import touch_file
        vision = capability(llm.model, getattr(llm, "_base_url", "")).supports_vision
        pinned, images, skipped = [], [], []
        for m in _re.findall(r"@([\w./\-]+)", text):
            rel = m.lstrip("/")
            if ".." in rel:               # never let @../ reach outside the workspace (defense-in-depth)
                continue
            if not (rel and os.path.isfile(os.path.join(root, rel))):
                continue
            if rel.lower().endswith(_IMG_EXT):
                if vision:
                    res = base_tools.attach_image(rel)   # only claim "attached" if it actually worked
                    (skipped if isinstance(res, str) and res.startswith("Error") else images).append(rel)
                else:
                    skipped.append(rel)
            else:
                touch_file(session.active(), rel); pinned.append(rel)
        if _tui and _console is not None:
            if pinned:   # paths like app/jobs/[id]/page.tsx contain [ ] → markup=False or Rich crashes
                _console.print(f"  📎 pinned: {', '.join(pinned)}", markup=False)
            if images:
                _console.print(f"  🖼  attached image: {', '.join(images)}", markup=False)
            if skipped:
                _console.print(f"  🖼  skipped (needs a vision-capable AGENT_MODEL): {', '.join(skipped)}", markup=False)

    # AGENT_AUTO_APPROVE: comma-separated fnmatch globs over the command, pre-approved so safe read-only
    # commands never prompt (e.g. AGENT_AUTO_APPROVE="git status*,git diff*,ls *,cat *").
    _auto = [r.strip() for r in (os.environ.get("AGENT_AUTO_APPROVE") or "").split(",") if r.strip()]
    # MAIN agent's policy: the canonical mode, but a confirm-mode needs a human — with no interactive prompt
    # (headless/piped, or the live composer that owns stdin) fall back to let-it-go (auto, still catastrophic-gated).
    _can_confirm = sys.stdin.isatty() and not use_live
    _eff_mode = canonical if (_can_confirm or not CONFIRMS.get(canonical)) else "letitgo"
    _stats["policy"] = policy_label(_eff_mode)
    if CONFIRMS.get(canonical) and _eff_mode != canonical:   # confirm-mode with no way to prompt → say so
        print(f"  · non-interactive shell: '{policy_label(canonical)}' can't ask for confirmation here → "
              f"running as '{policy_label(_eff_mode)}' (auto-run, still blocks catastrophic commands).")
    perm_hook = PermissionHook(make_policy(_eff_mode),
                               on_ask=_ask if CONFIRMS.get(_eff_mode) else None, auto_approve=_auto)
    # The fail-closed turn-authority gate is OFF by default (cfg.intent_gate="essential"): it over-blocks
    # ordinary local work — read-only git during a review, edits, a workspace switch — and its errors mislead.
    # The ESSENTIAL protections are unaffected: perm_hook still blocks catastrophic commands and, in a confirm
    # policy, asks before a command runs; GuardrailHook still stops runaway loops. AGENT_INTENT_GATE=strict
    # (or [policy] intent_gate="strict") restores the full v2 gate.
    _intent_gate_strict = cfg.intent_gate == "strict"
    if not _intent_gate_strict:
        print("  · intent gate: essential (catastrophic-command + confirm-mode protections only; "
              "set AGENT_INTENT_GATE=strict for the full turn-authority gate)")
    def _make_workspace_hooks():
        hook_list = [
            _WorkspaceHandoffHook(_workspace_handoff),
            ReconciliationHook(lambda: session.active()),
            FrozenEvidenceCutoffHook(
                lambda: session.active(),
                lambda name, args: tools.resolve_intent_effect(name, args),
                lambda path: base_tools.resource_ref(path),
            ),
            DelegationCompletionHook(lambda: session.active()),
            DelegatedClaimCompletionHook(lambda: session.active()),
            ExecutionEvidenceCompletionHook(lambda: session.active()),
            QualityEvidenceCompletionHook(lambda: session.active()),
        ]
        if _intent_gate_strict:
            hook_list.append(TurnAuthorityHook(
                lambda: session.active().intent.turn_contract,
                lambda name, args: tools.resolve_intent_effect(name, args),
            ))
        hook_list += [
            perm_hook,
            GuardrailHook(),  # cross-step loop guard (per-turn counters, reset each task)
        ]
        if cfg.verify_cmd:
            oracle = CommandOracle(cfg.verify_cmd, root=root)
            hook_list.append(OracleHook(
                oracle,
                lambda out: setattr(
                    session.active(), "last_error", f"Verification failed:\n{out[:600]}",
                ),
            ))
        if cfg.max_tokens:
            hook_list.append(BudgetHook(cfg.max_tokens))
        hook_list.extend(plugin_hooks)
        return CompositeHooks(*hook_list)

    hooks = _make_workspace_hooks()

    def _workspace_info() -> str:
        return (f"model={llm.model} · net={getattr(llm, 'proxy_used', 'direct')} · "
                f"policy={policy_label(_eff_mode)} · sandbox={cfg.sandbox_backend} · "
                f"code={type(retriever).__name__} · memory={type(memory).__name__} · "
                f"episodic={'on' if episodic is not None else 'off'} · "
                f"mine={mine_mode} · subagents={'on' if sub_depth > 0 else 'off'} · "
                f"skills={len(skills.names())} · mcp_tools={mcp_tool_count} · "
                f"plugin_tools={plugin_tool_count}")

    info = _workspace_info()
    # ── choose UI: the always-pinned live composer (AGENT_TUI=live), else the rich+prompt_toolkit REPL ──
    _input = _tui.TuiInput(_stats, root=root) if _tui else None
    _live_workspace_setter: dict[str, Any] = {"fn": None}
    _retired_sessions: list[tuple[str, str]] = []

    def _publish_workspace(candidate: WorkspaceResources) -> None:
        """Atomically redirect every workspace-facing delegate; process-owned UI/LLM objects stay intact."""
        nonlocal workspace, root, cfg, session, local_store, sandbox, retriever
        nonlocal base_tools, tools, skills, plugin_hooks, mcp_runtime, reviewer, episodic, monitor_sink
        nonlocal mine_mode, sub_depth, mcp_tool_count, plugin_tool_count, reducer, hooks, dispatch, info
        nonlocal _project_env_overlay

        previous_session = session.session_id
        previous_mine_mode = mine_mode
        for key in tuple(_project_env_overlay):
            os.environ.pop(key, None)
        _project_env_overlay = {}
        workspace = candidate
        root, cfg = candidate.root, candidate.config
        session, local_store = candidate.session, candidate.store
        sandbox, retriever = candidate.sandbox, candidate.retriever
        base_tools, tools, skills = candidate.base_tools, candidate.tools, candidate.skills
        plugin_hooks, mcp_runtime = list(candidate.plugin_hooks), candidate.mcp_runtime
        reviewer, episodic, monitor_sink = candidate.reviewer, candidate.episodic, candidate.monitor_sink
        mine_mode, sub_depth = candidate.mine_mode, candidate.subagent_depth
        mcp_tool_count, plugin_tool_count = candidate.mcp_tool_count, candidate.plugin_tool_count
        reducer = slice_sink(session)
        hooks = _make_workspace_hooks()
        dispatch = _make_workspace_dispatch()
        info = _workspace_info()
        _pending_seal_records.clear()
        _retired_sessions.append((previous_session, previous_mine_mode))

        # set_cache_key changes request/cache identity only; it does not reconstruct the provider client.
        try:
            llm.set_cache_key(session.session_id)
        except Exception as exc:  # noqa: BLE001 — a cache hint cannot invalidate a valid workspace swap
            _workspace_log(f"cache key refresh failed ({type(exc).__name__}: {exc})")
        if hasattr(memory, "_scope"):
            try:
                memory._scope = os.path.basename(root) or "default"
            except Exception:
                pass
        _stats["workspace"] = _ws_name(root)
        _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
        if _input is not None:
            try:
                _input.set_workspace(root)
            except Exception as exc:  # noqa: BLE001 — completion refresh is cosmetic
                _workspace_log(f"file completion refresh failed ({type(exc).__name__}: {exc})")
        live_setter = _live_workspace_setter.get("fn")
        if live_setter is not None:
            try:
                live_setter(root)
            except Exception as exc:  # noqa: BLE001 — keep the running composer even if completion fails
                _workspace_log(f"live completion refresh failed ({type(exc).__name__}: {exc})")
        for recovered in candidate.recovery_results:
            _workspace_log(f"recovered local artifact {recovered.artifact_id} ({recovered.status})")

    def _switch_workspace(target: str) -> bool:
        """Switch only workspace-owned state; never exit the app or reconnect the model client."""
        target = os.path.realpath(target)
        emit = (lambda message: _console.print(message, markup=False)) if _console is not None else print
        emit(f"  ↪ switching workspace → {target}")
        try:
            workspace_manager.switch(target, activate=_publish_workspace)
        except Exception as exc:  # noqa: BLE001 — staged failure rolls back to the still-live old workspace
            emit(f"  ✗ workspace unchanged: {type(exc).__name__}: {exc}")
            return False
        finally:
            _workspace_handoff.update(target="", ready=False)
        emit(f"  ✓ workspace → {target} · interface and model connection kept")
        return True

    from .text_utils import is_chitchat

    def _chitchat_reply(text, _dispatch):
        """A pure greeting/social message → cheap reply: MINIMAL prompt, NO tools, NO slice. Skips the full
        per-turn token cost (the 12k system prompt + tool schemas + tiers) for a 'hi'/'thanks'. (item D)"""
        from .model_runner import complete_model_call
        from .text_utils import CHITCHAT_PROMPT
        active_state = session.active() if session.active_id is not None else None
        msgs = [{"role": "system", "content": CHITCHAT_PROMPT}, {"role": "user", "content": text}]
        try:
            am = complete_model_call(llm, msgs, [])     # NO tools
            if _tui is not None and am.usage:
                _tui._record_usage(_stats, am.usage)
            reply = (am.content or "").strip() or "Hi! What would you like to work on?"
        except Exception:  # noqa: BLE001 — a chitchat reply must never crash the session
            reply = "Hi! What would you like to work on?"
        if active_state is not None:
            # Keep exactly enough ephemeral adjacency for "what did you just say?" without opening a task
            # artifact or turning social text into durable evidence. The next normal record_user call appends
            # its in-progress row, so render_conversation naturally treats this pair as the immediate prior turn.
            _fold_chitchat_continuity(active_state, text, reply)
        _dispatch(AssistantText(reply))

    def _activate_workspace_handoff(stop_reason: str) -> bool:
        """Publish a requested workspace only after the old turn is clean, reconciled, and durable."""
        target = str(_workspace_handoff.get("target") or "")
        if not target:
            return False
        reconciled = not (session.active_id is not None and session.active().reconciliation_required)
        if stop_reason == "end_turn" and reconciled:
            _workspace_handoff["ready"] = True
            return _switch_workspace(target)
        _workspace_handoff.update(target="", ready=False)
        message = f"  workspace switch cancelled because the turn stopped as {stop_reason!r}"
        (_console.print(message, markup=False) if _console is not None else print(message))
        return False

    def _run_one_turn(text, sink, signal):
        """One turn for the LIVE composer: route (lexical) → build slice → run_turn with a per-turn dispatch
        that feeds the LiveSink. Runs in run_live's worker thread, so the pinned box stays responsive."""
        if is_chitchat(text):                       # 'hi'/'thanks' → cheap reply, no slice/tools (item D)
            _chitchat_reply(text, make_dispatcher(sink))
            return
        if session.active_id is None:
            action, tid = "new", ""
        elif session.active().reconciliation_required:
            action, _tid = route_topic_lexical(text, session)
            if action in ("new", "resume"):
                make_dispatcher(sink)(AssistantText(
                    "Cannot change tasks while an earlier operation is indeterminate. Continue the active "
                    "task to re-observe live state and reconcile it first."
                ))
                return
            action, tid = "continue", ""
        else:
            action, tid = route(llm, text, session)
        _admit_routed_turn(text, action, tid)
        _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
        # Build the live-mode dispatcher before slice construction so the same shared progress state sees
        # the host lifecycle from its real beginning.  Required sinks are safe once _begin_local_turn runs.
        _live_sinks = []
        if episodic is not None:
            _live_sinks.append(episodic)
        _live_sinks.append(log_sink(root))
        if metrics is not None:
            _live_sinks.append(metrics)
        if monitor_sink is not None:
            _live_sinks.append(monitor_sink)
        _live_sinks.append(sink)
        live_dispatch = make_dispatcher(
            *_live_sinks,
            required=(_journal_event, _reduce_event),
        )
        live_dispatch(TurnStarted(
            request=text,
            task_title=_stats["topic"],
            task_id=session.active_id or "",
            plan=list(session.active().plan or ()),
        ))
        try:
            _expand_mentions(text)        # @path → pin the file into OPEN FILES
            build = make_build_slice(
                session, tools, retriever, memory, text, session.session_id, model_id=llm.model,
            )
        except KeyboardInterrupt:
            live_dispatch(TurnInterrupted("aborted", "cancelled during context preparation"))
            if _seal_local_turn("aborted", live_dispatch):
                from . import recovery as _rec
                _rec.clear(root)
                return
            raise _DurabilityStop("required local seal failed after cancellation")
        except Exception as exc:  # noqa: BLE001 — preparation is inside the required durability lifecycle
            message = f"context preparation failed ({type(exc).__name__}: {exc})"
            live_dispatch(TurnInterrupted("error", message))
            if _seal_local_turn("error", live_dispatch):
                from . import recovery as _rec
                _rec.clear(root)
                return
            raise _DurabilityStop("required local seal failed after a context-preparation error")
        llm.set_delta_sink(sink.on_delta)
        import time as _t
        _t0 = _t.monotonic()
        from . import recovery as _rec
        result = run_turn(build_slice=build, llm=llm, tools=tools, dispatch=live_dispatch,
                          hooks=hooks, signal=signal, max_steps=cfg.max_steps,
                          consolidate=lambda: consolidate_checkpoint(session.active(), compact=False),
                          checkpoint=lambda m, s, _g=text: _rec.record(root, goal=_g, messages=m, step=s),
                          turn_id=local_store.active.artifact_id if local_store.active else "")
        if _seal_local_turn(result.stop_reason, live_dispatch):
            _rec.clear(root)                               # clear legacy WAL only after required local commit
        else:
            raise _DurabilityStop(
                "required local seal could not complete; the recovery journal is retained. "
                "Restart SliceAgent after fixing storage so recovery can finish before new work.")
        _stats["last_turn_s"] = _t.monotonic() - _t0
        switched = _activate_workspace_handoff(result.stop_reason)
        if not switched and reviewer is not None:
            reviewer.review(session.session_id)
        return None

    def _try_live() -> bool:
        """Run the always-pinned live composer; return True if it ran (REPL below is then skipped), False to
        fall back to the REPL on any startup failure — input is never left broken."""
        try:
            _tui.run_live(console=_console, stats=_stats, banner_info=info, root=root,
                          run_one_turn=_run_one_turn, handle_slash=_handle_slash,
                          on_ready=lambda setter: _live_workspace_setter.__setitem__("fn", setter))
            return True
        except Exception as _e:  # noqa: BLE001
            print(f"\n  live UI failed ({type(_e).__name__}: {_e}); using the inline REPL instead.")
            return False

    # crash recovery: a leftover WAL means the last turn in this workspace never reached a clean/parked
    # exit (a hard crash). Surface what was in flight, then clear it (auto-loop-resume is a future step).
    from . import recovery
    _pend = recovery.pending(root)
    if _pend:
        _rg = one_line(_pend.get("goal", ""), 60)
        _rla = one_line(recovery.last_assistant(_pend), 160)
        _note = (f"  ⚠ recovered an interrupted turn (step {_pend.get('step', '?')}): {_rg}"
                 + (f"\n    last said: {_rla}" if _rla else "")
                 + "\n    its progress wasn't saved cleanly — re-send the request to continue.")
        # markup=False: _note holds RECOVERED agent/user text (paths like app/jobs/[id]/page.tsx, code, or a
        # stray `[/learn]`) — parsing it as Rich markup crashes startup with a MarkupError. Style, don't parse.
        (_console.print(_note, style="yellow", markup=False) if _console is not None else print(_note))
        recovery.clear(root)

    if use_live and _try_live():
        pass                              # the live composer ran the whole session (until ctrl-d/exit)
    else:
        if _tui:
            _tui.banner(_console, info)
        else:
            print("sliceagent · slice core (run_turn) · " + info)
            print('type a task, or "exit" to quit\n')
        from .sensory_cortex import project_root as _project_root
        if _project_root(root) is None:        # launched outside a project → tell the user how to pick one
            _hint = ("  · no project here — use /cwd <path>, or ask me to find and switch to one")
            (_console.print(f"[grey50]{_hint}[/]") if _console is not None else print(_hint))
        while True:
            if _input is not None:
                line = _input.prompt()
                if line is None:                               # ctrl-d / EOF
                    break
                line = line.strip()
            else:
                try:
                    line = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            if line in ("exit", "quit", "/exit"):
                break
            if not line:
                continue
            if line == "/learn" or line.startswith("/learn "):  # transcript → reusable skill (runs as a turn)
                from .neocortex import build_learn_prompt
                line = build_learn_prompt(line[len("/learn"):].strip())
            elif _tui and line.startswith("/"):                # navigation palette (no turn)
                _handle_slash(line)
                continue
            # INVARIANT: echo the user's line BEFORE any blocking work (esp. route_topic's LLM round-trip),
            # so the message paints the instant Enter is pressed — not ~0.5-2s later.
            if _tui:                              # anchor the user turn with spacing (fixes cramped layout)
                _tui.user_echo(_console, line)
            if is_chitchat(line):                # 'hi'/'thanks' → cheap reply, skip routing/slice/run_turn (item D)
                _chitchat_reply(line, chitchat_dispatch)
                continue
            if session.active_id is None:                      # first message bootstraps the first topic
                action, tid = "new", ""
            elif session.active().reconciliation_required:
                action, tid = route_topic_lexical(line, session)
                if action in ("new", "resume"):
                    print("  · task change blocked: reconcile the prior indeterminate operation first")
                    continue
                action, tid = "continue", ""
            else:                                              # route: continue / new / resume (no junk topic)
                # route() is lexical by default (instant, zero round-trips); AGENT_ROUTER=llm restores the
                # classifier (a provider round-trip). Cover it with a 'routing…' spinner; the shared turn
                # progress state begins immediately after routing and remains live through slice construction.
                if _tui:
                    with _console.status("[grey50]routing…[/]", spinner="dots"):
                        action, tid = route(llm, line, session)
                else:
                    action, tid = route(llm, line, session)
                if not _tui:                                   # TUI shows the topic in the status bar, not as noise
                    print(f"  · topic: {action}{(' ' + tid) if tid else ''}")
            _admit_routed_turn(line, action, tid)
            _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
            dispatch(TurnStarted(
                request=line,
                task_title=_stats["topic"],
                task_id=session.active_id or "",
                plan=list(session.active().plan or ()),
            ))
            try:
                # slice-build phase happens BEFORE run_turn's own KeyboardInterrupt handling — a ctrl-c
                # here (e.g. during the one-time repo-map build) must cancel the turn, not crash the REPL.
                _expand_mentions(line)           # @path → pin the file into OPEN FILES
                build = make_build_slice(session, tools, retriever, memory, line, session.session_id, model_id=llm.model)
            except KeyboardInterrupt:
                dispatch(TurnInterrupted("aborted", "cancelled during context preparation"))
                if _seal_local_turn("aborted", dispatch):
                    recovery.clear(root)
                    print("\n  · cancelled")
                    continue
                print("\n  · required local seal failed; stopping before accepting another request")
                break
            except Exception as exc:  # noqa: BLE001 — preparation belongs to the required turn lifecycle
                message = f"context preparation failed ({type(exc).__name__}: {exc})"
                dispatch(TurnInterrupted("error", message))
                if _seal_local_turn("error", dispatch):
                    recovery.clear(root)
                    print(f"\n  · {message}")
                    continue
                print("\n  · required local seal failed; stopping before accepting another request")
                break
            if os.environ.get("AGENT_TIMING"):   # per-turn latency breakdown (build vs model) → find the hang
                import time as _tt
                _b = build
                def build(_b=_b):
                    _s = _tt.monotonic()
                    r = _b()
                    print(f"  ⏱ slice build {(_tt.monotonic() - _s) * 1000:.0f} ms (progress was already "
                          "visible; the remaining wait is the model's first token)", flush=True)
                    return r
            # ctrl-c OR esc during the turn (incl. while the LLM is thinking) raises KeyboardInterrupt, which
            # run_turn catches → aborts the turn cleanly and returns here to the prompt (then ctrl-d quits).
            # Esc is translated to a real SIGINT by a narrow background sentinel (a no-op on non-tty/eval —
            # start() gates itself exactly like _arrow_select does), so it reaches the SAME KeyboardInterrupt
            # path ctrl-c already uses instead of a separate abort mechanism.
            _esc = _tui.make_esc_sentinel() if _tui else None
            if _esc is not None:
                _esc.start()
            import time as _time
            _t0 = _time.monotonic()
            try:
                result = run_turn(build_slice=build, llm=llm, tools=tools, dispatch=dispatch, hooks=hooks,
                                  max_steps=cfg.max_steps,
                                  consolidate=lambda: consolidate_checkpoint(session.active(), compact=False),
                                  checkpoint=lambda m, s, _g=line: recovery.record(root, goal=_g, messages=m, step=s),
                                  turn_id=local_store.active.artifact_id if local_store.active else "")
            finally:
                if _esc is not None:
                    _esc.stop()
            if _seal_local_turn(result.stop_reason, dispatch):
                recovery.clear(root)                           # clear legacy WAL only after core commit
            else:
                print("  · required local seal failed; stopping before accepting another request")
                break
            _stats["last_turn_s"] = _time.monotonic() - _t0   # shown as ⏲ in the status bar
            switched = _activate_workspace_handoff(result.stop_reason)
            if not switched and reviewer is not None:           # OPT-IN: critique the turn off-thread
                reviewer.review(session.session_id)

    # Session end: retire the current workspace once, consolidate every session visited by this long-lived
    # process, then close the process-owned memory facade. A workspace change already retired its old bundle.
    def _safe(label, fn):
        try:
            return fn()
        except Exception as _e:  # noqa: BLE001
            print(f"  · warning: {label} failed ({type(_e).__name__}: {_e})")
            return None

    # A ctrl-c during this shutdown sequence means "just quit" — _safe only catches Exception, but ctrl-c
    # raises KeyboardInterrupt (a BaseException), so without this outer guard a ctrl-c landing mid-step (esp.
    # the slow consolidation LLM call) would escape every _safe and dump a raw traceback. Catch it once here.
    try:
        _safe("workspace cleanup", workspace.close)
        # consolidate the episodic cache into long-term memory (the cache→memory loop). mine_mode gates it:
        # off → skip; deterministic → recorded skills; llm → render_skill_llm generalizes (scan-first).
        if getattr(memory, "is_durable", False):
            sessions_to_consolidate = list(dict.fromkeys(
                [*_retired_sessions, (session.session_id, mine_mode)]
            ))
            for session_id, mode in sessions_to_consolidate:
                if mode in ("0", "off", "none"):
                    continue
                st = _safe(
                    "memory consolidation",
                    lambda sid=session_id, selected=mode: memory.consolidate(
                        sid, llm=llm, mode=selected,
                    ),
                ) or {}
                if st.get("lessons") or st.get("skills"):
                    print(f"  · consolidated: {st.get('lessons', 0)} lesson(s), "
                          f"{st.get('skills', 0)} skill(s)"
                          + (f", {st['skills_rejected']} rejected" if st.get("skills_rejected") else "")
                          + (f", {st['errors']} error(s)" if st.get("errors") else ""))
                elif st.get("skills_rejected") or st.get("errors"):
                    print(f"  · consolidation: {st.get('skills_rejected', 0)} rejected, "
                          f"{st.get('errors', 0)} error(s)")
        _safe("memory close", getattr(memory, "close", lambda: None))  # FTS5 WAL checkpoint
        if metrics is not None:                                 # the moat number: per-turn fresh-input curve
            s = metrics.summary()
            print(f"  · metrics: per_turn_fresh={s['per_turn_fresh']} avg={s['avg_turn_fresh']} "
                  f"cache_hit={s['cache_hit_rate']} tools={s['tool_calls']}({s['tool_failures']} fail) "
                  f"retries={s['retries']} overflows={s['overflows']} errors={s['errors']}")
    except KeyboardInterrupt:
        _workspace_handoff["ready"] = False
        print("\n  · exiting (skipped remaining cleanup)")


if __name__ == "__main__":
    main()
