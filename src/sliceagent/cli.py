"""sliceagent CLI — a thin event-sink host over the stateless slice core.

The loop only dispatches events; this host wires the sinks (slice-updater, durable
log, terminal output) and the narrow runtime safeguards (catastrophic floor,
optional Oracle/budget).
Other surfaces (TUI, SDK, channels) are just different sinks over the same core.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid
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
    TurnEnd,
    TurnInterrupted,
    TurnPhaseChanged,
    TurnStarted,
    make_dispatcher,
)
from .hooks import (BudgetHook, CatastrophicSafeguardHook, CompositeHooks, Hooks,
                    OracleHook, ToolPreflight)
from .execution import ToolStatus
from .mentions import workspace_mentions
from .receipts import (compact_receipt_projection, receipt_completion_label,
                       receipt_summary_parts)
from .slash import SUPPORTED_SLASH_COMMANDS, slash_help_line
from .tui_projection import normalized_tool_status, safe_terminal_text
from .workspace_handoff import WorkspaceScheduleDecision


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


def _mint_logical_turn_id(session_id: str, generation: int, task_id: str, *, nonce: str = "") -> str:
    """Mint a request identity that cannot collide when a durable app session restarts."""
    token = str(nonce or uuid.uuid4().hex[:16])
    return f"{session_id}:{int(generation)}:{token}:{task_id}"


def _retire_recovered_transition(store, transition, on_log: Callable[[str], None]) -> object | None:
    """Retire an old crash ticket once a fresh explicit user turn has taken ownership."""
    if transition is None:
        return None
    try:
        store.clear(transition)
    except Exception as exc:  # noqa: BLE001 — admission remains durable; keep the ticket for the next repair
        on_log(f"recovered workspace continuation could not be retired "
               f"({type(exc).__name__}: {exc})")
        return transition
    return None

def _resolve_workspace_target(workspace_root: str, path: str) -> tuple[str | None, str]:
    """Resolve one requested workspace without mutating cwd or any live runtime owner."""
    root = os.path.realpath(workspace_root)
    raw = (path or "").strip()
    if not raw:
        return None, f"workspace: {root}"
    # `/cwd` consumes the complete remainder of the line, so raw spaced paths already work. Also accept the
    # conventional shell spelling with one matching outer quote pair; those quotes are syntax, not filename
    # bytes. Do not invoke shlex or expand anything else.
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1]
        if not raw:
            return None, f"not a directory: {path}"
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


def _classify_workspace_schedule(
    root: str,
    target: str,
    *,
    pending_target: str = "",
    workspace_switches: int = 0,
    workspace_edges=(),
    max_transitions: int,
) -> WorkspaceScheduleDecision:
    """Pure truth classification for one proposed workspace transition."""
    if pending_target and pending_target != target:
        return WorkspaceScheduleDecision.failed(
            f"workspace switch already scheduled for {pending_target}",
        )
    if workspace_switches >= max_transitions:
        return WorkspaceScheduleDecision.failed(
            f"this request reached its {max_transitions}-workspace transition budget; "
            "finish here or ask the user for a new navigation request",
        )
    edge = (os.path.realpath(root), os.path.realpath(target))
    if edge in workspace_edges:
        return WorkspaceScheduleDecision.steered(
            "this request already traversed that workspace transition; refusing a navigation loop",
        )
    return WorkspaceScheduleDecision.scheduled()


def _discovery_skill_lines(skills) -> list[str]:
    """Safe, compact projection for ``/skills`` (also a pure test seam)."""
    try:
        catalog = list(skills.catalog())
    except Exception:  # noqa: BLE001 — discovery output must not destabilize the session
        catalog = []
    if not catalog:
        return ["  no skills loaded"]
    lines = [f"  available skills ({len(catalog)}):"]
    for name, description in catalog:
        desc = " ".join(str(description or "").split())
        lines.append(f"  {name}" + (f" — {desc}" if desc else ""))
    return lines


def _rich_escape(value: object) -> str:
    """Escape user/config/provider data before interpolating it into intentional Rich markup."""
    try:
        from rich.markup import escape
        return escape(str(value or ""))
    except ImportError:  # Rich is optional; this helper is normally reached only by the TUI
        return str(value or "")


def _discovery_tool_lines(tools, registry) -> list[str]:
    """Project the tool schemas the model can actually see, grouped by provenance."""
    try:
        schemas = list(tools.schemas())
    except Exception:  # noqa: BLE001
        schemas = []
    grouped: dict[str, list[str]] = {}
    seen: set[str] = set()
    for schema in schemas:
        name = str((schema.get("function") or {}).get("name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            entry = registry.entry(name)
        except Exception:  # noqa: BLE001
            entry = None
        source = str(getattr(entry, "source", "") or
                     ("subagent" if name == "spawn_agent" else "runtime"))
        grouped.setdefault(source, []).append(name)
    if not seen:
        return ["  no tools available"]
    lines = [f"  available tools ({len(seen)}):"]
    for source in sorted(grouped):
        lines.append(f"  {source}: {', '.join(sorted(grouped[source]))}")
    return lines


def _discovery_agent_lines(tools) -> list[str]:
    """Project only subagent kinds reachable from the host's current schema/depth."""
    try:
        schema_names = {
            str((schema.get("function") or {}).get("name") or "")
            for schema in tools.schemas()
        }
    except Exception:  # noqa: BLE001
        schema_names = set()
    if "spawn_agent" not in schema_names:
        return ["  subagents are disabled at the current depth"]
    available = dict(getattr(tools, "agents", {}) or {})
    if getattr(tools, "core_mode", False):
        available = ({"explorer": available["explorer"]}
                     if "explorer" in available else {})
    if not available:
        return ["  no subagent profiles configured"]
    lines = [f"  available agents ({len(available)}):"]
    for name in sorted(available):
        spec = available[name]
        access = "read-only" if getattr(spec, "read_only", False) else "writable"
        desc = " ".join(str(getattr(spec, "description", "") or "").split())
        lines.append(f"  {name} [{access}]" + (f" — {desc}" if desc else ""))
    return lines


def _fold_chitchat_continuity(state, user_text: str, assistant_text: str) -> None:
    """Keep one bounded exact social adjacency while consuming stale action/evidence continuity."""
    state.continuity.pending_proposal = None
    state.continuity.previous_evidence_snapshot = None
    state.conversation.append({
        "user": str(user_text), "assistant": str(assistant_text), "artifact_id": "",
    })
    state.conversation = state.conversation[-4:]


def _use_chitchat_fast_path(text: str, state=None) -> bool:
    """Route only genuinely standalone social messages through the cheap path.

    ``ok``/``okay``/``sounds good`` are social in isolation, but they are also the shared intent grammar's
    bare assents.  When the immediately preceding assistant response carries a proposal, let normal turn
    admission interpret that adjacency instead of clearing it as chitchat before interpretation begins.
    """
    from .intent import is_bare_assent
    from .text_utils import is_chitchat

    if not is_chitchat(text):
        return False
    pending = getattr(getattr(state, "continuity", None), "pending_proposal", None)
    conversation = tuple(getattr(state, "conversation", ()) or ())
    paired_adjacency = bool(
        conversation
        and str(conversation[-1].get("user") or "").strip()
        and str(conversation[-1].get("assistant") or "").strip()
    )
    # Restart hydration deliberately reconstructs the one exact pair, not host-interpreted proposal state.
    # A bare assent after that pair still belongs on the normal model path, where the model can resolve it from
    # the paired adjacency; the social fast path must not consume it first.
    return not (is_bare_assent(text) and (pending or paired_adjacency))


def _cost_lines(stats: dict, metrics=None) -> list[str]:
    """Stable `/cost` projection: session totals always, optional per-turn diagnostics when enabled."""
    from .tui import _saved_dollars

    def count(key: str) -> int:
        try:
            return max(0, int(stats.get(key, 0) or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    saved = _saved_dollars(stats)
    try:
        spent = float(stats.get("cost", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        spent = 0.0
    saved_tokens = count("saved_cached_tok")
    head = (
        f"  💰 saved ${saved:.4f} vs full-history (@ {stats.get('model', '?')}, cache-aware)"
        f"  ·  spent ${spent:.4f}"
        if saved is not None else
        f"  💰 {saved_tokens:,} tokens saved vs full-history (model price unknown)"
    )
    lines = [
        head,
        f"  tokens: {count('tokens'):,} total · {count('fresh'):,} fresh · "
        f"{saved_tokens:,} cached-history saved",
    ]
    if metrics is None:
        lines.append("  (per-turn curve off — start with AGENT_METRICS=1 to track it)")
        return lines
    try:
        summary = metrics.summary()
        lines.append(
            f"  per_turn_fresh={summary['per_turn_fresh']} avg={summary['avg_turn_fresh']} "
            f"cache_hit={summary['cache_hit_rate']} tools={summary['tool_calls']} "
            f"out={summary['output']} retries={summary['retries']} overflows={summary['overflows']}"
        )
    except Exception as exc:  # noqa: BLE001 — a metrics observer must not break a slash command
        lines.append(f"  (per-turn metrics unavailable: {type(exc).__name__})")
    return lines


class _WorkspaceHandoffHook(Hooks):
    """Once a handoff is pending, the model may only finish the current turn."""

    def __init__(self, state: dict):
        self.state = state

    def preflight_tool(self, name, args):
        if self.state.get("target"):
            return ToolPreflight(
                True,
                "workspace switch is already scheduled; finish this turn without more tool calls",
                kind="lifecycle",
            )
        return ToolPreflight()

    def should_continue_after_stop(self, stop_reason):
        # A workspace-navigation turn must not run the old workspace's Oracle/plugin completion hooks after
        # the control tool succeeds. The host still requires an ordinary clean stop + durable seal below.
        return {"exclusive": True} if self.state.get("target") else None


def _is_workspace_transport_completion(state: dict, event: Event) -> bool:
    """Whether ``event`` closes only the source segment, not the user's logical turn.

    The source model normally emits a final sentence such as "switching now" after the control tool.  It is a
    transport acknowledgement, not the answer to the compound request.  Keep it in execution logs, but do not
    publish it as a final answer or fold it into conversational continuity before the target segment runs.
    """
    if not state.get("target"):
        return False
    return (
        isinstance(event, TurnEnd)
        or isinstance(event, TurnCommitted)
        or (isinstance(event, AssistantText) and event.final)
    )


def _workspace_presentation_sink(state: dict, sink):
    """Suppress a source-segment terminal projection while forwarding all real progress and target output."""
    def present(event):
        if not _is_workspace_transport_completion(state, event):
            sink(event)
    return present


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
    project_identity: Any = None
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

        def warn(message: str) -> None:
            try:
                self._on_log(message)
            except Exception:
                pass  # teardown diagnostics can never turn successful retirement into a failed switch

        def safe(label: str, fn) -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — teardown is best-effort, but never silent
                warn(f"warning: {label} failed ({type(exc).__name__}: {exc})")

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
                warn("warning: MCP shutdown timed out after 8s")
        writer = getattr(self.monitor_sink, "writer", None)
        if writer is not None:
            safe("monitor writer", writer.close)
        safe("local state lease", self.store.close)


class _WorkspaceBuildCleanup:
    """Idempotently retire owners acquired by an unpublished workspace build.

    Ownership transfers to :class:`WorkspaceResources` only after the complete candidate
    has been assembled. Until then this helper preserves the dependency-aware rollback
    order and ensures a cleanup failure cannot replace the original preparation error.
    """

    def __init__(self, on_log: Callable[[str], None] = lambda _message: None):
        self.store = None
        self.base_tools = None
        self.mcp_runtime = None
        self.reviewer = None
        self.monitor_sink = None
        self._on_log = on_log
        self._lock = threading.Lock()
        self._closed = False
        self._released = False

    def release(self) -> None:
        """Transfer all tracked owners to the completed resource bundle."""
        with self._lock:
            self._released = True

    def close(self) -> None:
        with self._lock:
            if self._closed or self._released:
                return
            self._closed = True
            reviewer, base_tools = self.reviewer, self.base_tools
            mcp_runtime, monitor_sink, store = self.mcp_runtime, self.monitor_sink, self.store

        def warn(message: str) -> None:
            try:
                self._on_log(message)
            except Exception:
                pass

        def safe(label: str, fn) -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — preserve the primary preparation failure
                warn(f"warning: workspace build {label} failed ({type(exc).__name__}: {exc})")

        if reviewer is not None:
            safe("background-review rollback", lambda: reviewer.join(timeout=2))
        if base_tools is not None:
            safe("tool rollback", base_tools.cleanup)
        if mcp_runtime is not None:
            safe("MCP rollback", mcp_runtime.shutdown)
        writer = getattr(monitor_sink, "writer", None)
        if writer is not None:
            safe("monitor rollback", writer.close)
        if store is not None:
            safe("local-state rollback", store.close)


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
            try:
                old.close()
            except Exception as exc:  # noqa: BLE001 — publication already committed; retirement is advisory
                logger = getattr(candidate, "_on_log", None)
                if callable(logger):
                    try:
                        logger(f"warning: prior workspace retirement failed "
                               f"({type(exc).__name__}: {exc})")
                    except Exception:
                        pass
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
    from .persistence import artifact_order_key
    from .receipts import compact_receipt_projection
    from .runtime_persistence import recoverable_child_report_count
    from .taskstate import task_state_from_checkpoint, task_state_to_slice

    restored = []
    for checkpoint in store.checkpoints():
        try:
            task_state = task_state_from_checkpoint(checkpoint)
            session.tasks[checkpoint.task_id] = task_state_to_slice(task_state)
            restored.append((checkpoint.order_ns, checkpoint.updated_at,
                             checkpoint.task_id, task_state.status,
                             max(0, int(getattr(task_state, "workspace_epoch", 0) or 0))))
        except Exception as exc:  # noqa: BLE001 — one incompatible task must not hide the others
            on_log(f"local task {checkpoint.task_id} could not be restored "
                   f"({type(exc).__name__}: {exc})")
    # Reconstruct the standing constant-size receipt view from immutable artifacts rather than duplicating it
    # into the semantic task checkpoint.  This keeps it available after restart without making it another
    # writable work-state owner.
    latest_receipts = {}
    latest_turns = {}
    for artifact in store.coordinator.artifacts.list_all():
        if artifact.kind != "turn" or artifact.task_id not in session.tasks:
            continue
        key = artifact_order_key(artifact)
        previous_turn = latest_turns.get(artifact.task_id)
        if previous_turn is None or key > previous_turn[0]:
            latest_turns[artifact.task_id] = (key, artifact)
        body = artifact.structured_body if isinstance(artifact.structured_body, dict) else dict(
            artifact.structured_body,
        )
        projection = compact_receipt_projection(body.get("turn_receipt"))
        if projection is None:
            continue
        previous = latest_receipts.get(artifact.task_id)
        if previous is None or key > previous[0]:
            latest_receipts[artifact.task_id] = (key, artifact.id, projection)
    for task_id, (_key, artifact_id, projection) in latest_receipts.items():
        state = session.tasks[task_id]
        state.continuity.last_receipt = dict(projection)
        state.continuity.last_receipt_artifact_id = artifact_id
    # TaskState intentionally excludes transcript residue. Rehydrate only the latest sealed, user-visible pair
    # from its immutable turn artifact so a normal restart retains the one adjacency needed by deictic follow-ups
    # without reviving a conversation ring. A hidden workspace-transport response or an internal/partial note is
    # not a prior assistant answer and must not become one after restart.
    for task_id, (_key, artifact) in latest_turns.items():
        body = artifact.structured_body if isinstance(artifact.structured_body, dict) else dict(
            artifact.structured_body,
        )
        recovered_child_reports = recoverable_child_report_count(artifact)
        if recovered_child_reports:
            # The report text itself remains canonical only in the immutable interrupted-turn artifact.
            # Advertise that exact readable source for one resumed turn; do not copy it into TaskState,
            # Active Work, findings, or every later seed.
            continuity = session.tasks[task_id].continuity
            continuity.recovery_child_artifact_id = artifact.id
            continuity.recovery_child_report_count = recovered_child_reports
        meta = body.get("meta")
        meta = meta if isinstance(meta, dict) else dict(meta or {})
        provenance = str(body.get("assistant_provenance") or "")
        request = str((artifact.brief.get("request") if artifact.brief else "") or "")
        assistant = str(body.get("assistant") or "")
        visible_final = provenance in {"", "final_response"}
        if request.strip() and assistant.strip() and visible_final \
                and meta.get("segment_outcome") != "workspace_transition":
            session.tasks[task_id].conversation = [{
                "user": request, "assistant": assistant, "artifact_id": artifact.id,
            }]
    # Uncertain receipts remain visible on their own task, but they do not seize the workspace from a newer
    # active/parked task. Pick the newest resumable checkpoint uniformly.
    candidates = [row for row in restored if row[3] in ("active", "parked", "indeterminate")]
    if candidates:
        latest = max(candidates, key=lambda row: (row[0], row[1], row[2]))
        selected = None
        if latest[0] > 0:
            selected = latest
        else:
            # Old checkpoints have only second-resolution timestamps. A tie has no truthful total order;
            # prefer one uniquely active record, otherwise leave selection explicit instead of task-id guessing.
            tied = [row for row in candidates if row[1] == latest[1]]
            active = [row for row in tied if row[3] == "active"]
            if len(tied) == 1 or len(active) == 1:
                selected = active[0] if len(active) == 1 else tied[0]
            else:
                on_log("multiple legacy tasks share the latest checkpoint timestamp; "
                       "choose one explicitly with /threads then /resume <task-id>")
        if selected is not None:
            session.active_id = selected[2]
            session.workspace_epoch = selected[4]


def _prepare_workspace_resources(
    root: str,
    *,
    cfg,
    llm,
    memory,
    schedule_workspace,
    notify_subagent,
    ask_user=None,
    session_id: str | None = None,
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
    from .identity import resolve_project_identity
    from .mcp_client import connect_mcp_servers
    from .memory import make_write_skill_tool
    from .plugins import load_plugins
    from .runtime_persistence import CoreArtifactFS, LocalTurnStore
    from .sandbox import make_sandbox
    from .session import Session, SessionBinding, make_topic_tools
    from .skills import make_skill_manager, make_skill_tool
    from .subagent import SubagentHost
    from .text_utils import one_line
    from .tools import LocalToolHost

    root = os.path.realpath(root)
    project_identity = resolve_project_identity(root)
    cfg = cfg or load_config(root)
    store = base_tools = mcp_runtime = reviewer = monitor_sink = None
    cleanup = _WorkspaceBuildCleanup(on_log)
    try:
        concrete_session = Session(memory, session_id=session_id)
        session = SessionBinding(concrete_session)
        # The lease/recovery boundary precedes every executable extension surface.
        store = LocalTurnStore(root, session.session_id, exclusive=True)
        cleanup.store = store
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
        cleanup.base_tools = base_tools
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
        plugin_mcp = load_plugins(
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
        cleanup.mcp_runtime = mcp_runtime
        mcp_tool_count = sum(
            1 for entry in base_tools.registry._tools.values() if entry.source == "mcp"
        )
        plugin_tool_count = sum(
            1 for entry in base_tools.registry._tools.values()
            if entry.source.startswith("plugin:")
        )

        base_tools._artifacts = CoreArtifactFS(
            store.coordinator.artifacts, archive_root=os.path.dirname(store.store_root),
        )
        _hydrate_workspace_tasks(store, concrete_session, on_log)
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
                base_tools, llm=llm, retriever=retriever, memory=memory,
                max_depth=sub_depth if advanced_agents else 1, notify=notify_subagent,
                agents=agents, session_id=session.session_id,
                intent_provider=lambda _task, s=session: s.active().intent,
                task_id_fn=lambda s=session: s.active_id or "t-none",
                parent_id_fn=lambda st=store: (
                    st.active.artifact_id if st.active is not None else ""
                ),
                workspace_id=store.workspace_id,
                artifact_store=store.coordinator.artifacts,
                artifact_ref_sink=store.record_artifact_ref,
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

        from .workspace_context import configure_workspace_contextfs

        configure_workspace_contextfs(
            base_tools=base_tools,
            session=session,
            memory=memory,
            project_identity=project_identity,
            root=root,
        )

        reviewer = make_background_reviewer(
            memory, scope=os.path.basename(root) or "default",
            project_id=project_identity.project_id, on_log=on_log,
        )
        cleanup.reviewer = reviewer
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
            cleanup.monitor_sink = monitor_sink

        resources = WorkspaceResources(
            root=root, config=cfg, session=session, store=store, sandbox=sandbox,
            retriever=retriever, base_tools=base_tools, tools=tools, skills=skills,
            project_identity=project_identity,
            mcp_runtime=mcp_runtime, reviewer=reviewer,
            episodic=episodic, monitor_sink=monitor_sink, recovery_results=recovery_results,
            mcp_tool_count=mcp_tool_count, plugin_tool_count=plugin_tool_count,
            mine_mode=cfg.mine, subagent_depth=sub_depth, _on_log=on_log,
        )
        cleanup.release()
        return resources
    except BaseException:
        cleanup.close()
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
                    rotated = path + ".1"
                    os.replace(path, rotated)
                    try:
                        os.chmod(rotated, 0o600)
                    except OSError:
                        pass
            except OSError:
                pass
            # This log contains assistant text and full tool results. ``open(..., 'a')`` would create it 0644
            # under a normal umask, exposing private task data to other local users. Create/repair it as 0600.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                try:
                    os.fchmod(fd, 0o600)
                except (AttributeError, OSError):
                    pass
                with os.fdopen(fd, "a", encoding="utf-8") as f:
                    fd = -1
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            finally:
                if fd >= 0:
                    os.close(fd)
    return sink


def _plain_arg(args: dict) -> str:
    """The one informative arg for a plain-mode tool line (path/command/pattern/…), whitespace-collapsed."""
    if not isinstance(args, dict):
        return ""
    for k in ("path", "command", "pattern", "name", "ref", "goal", "task"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return " ".join(safe_terminal_text(v, multiline=False).split())[:60]
    v = next((x for x in args.values() if isinstance(x, str) and x.strip()), "")
    return " ".join(safe_terminal_text(v, multiline=False).split())[:60]


def cli_sink(show_slice: bool = False):
    """Plain stdout sink (AGENT_TUI=off / no tui extra / pipes / CI): no color, no spinner — one readable
    line per tool action (✓/✗ + name + primary arg, output for commands/failures) and a delimited answer."""
    _quiet = {"read_file", "list_files"}   # header says enough; their content is noise in a log

    def sink(e: Event) -> None:
        if isinstance(e, SliceBuilt) and show_slice:
            print("\n  ┌─ slice ─────────────")
            rendered = safe_terminal_text(e.rendered, multiline=True)
            print("\n".join("  │ " + ln for ln in rendered.splitlines()))
            print("  └─────────────")
        elif isinstance(e, ToolResult):
            status = normalized_tool_status(e)
            mark = {
                "succeeded": "✓", "steered": "↷", "cancelled": "↷",
                "failed": "✗", "indeterminate": "!",
            }[status]
            head = f"  {mark} {e.name} {_plain_arg(e.args)}".rstrip()
            out = " ".join(safe_terminal_text(e.output, multiline=True).split())[:140]
            if out and (status != "succeeded" or e.name not in _quiet):
                head += f"  — {out}"
            print(head)
        elif isinstance(e, AssistantText):
            if (e.content or "").strip():
                label = "[assistant update]\n" if not e.final else ""
                print(f"\n{label}{safe_terminal_text(e.content, multiline=True)}\n")
        elif isinstance(e, ApiRetry):
            print(f"  …retry #{e.attempt} ({safe_terminal_text(e.error, multiline=False)})")
        elif isinstance(e, TurnInterrupted):
            print(f"\n[interrupted: {safe_terminal_text(e.reason, multiline=False)}]")
        elif isinstance(e, LessonSaved):
            print(f"  💡 learned: {safe_terminal_text(e.title, multiline=False)}")
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


def _plugin_tool_names(registry) -> list[str]:
    """Names contributed by plugins (whose provenance is namespaced as ``plugin:<manifest-name>``)."""
    return sorted(
        entry.name for entry in registry._tools.values()
        if str(getattr(entry, "source", "")).startswith("plugin:")
    )


def _restore_inline_after_live_failure(*, llm, rich_sink, live_runtime: dict,
                                       workspace_setter: dict, ask_bridge: dict,
                                       ask_user, subagent_bridge: dict,
                                       interactive: bool) -> None:
    """Retire every live-Application callback and restore the inline renderer's process bridges."""
    live_runtime["active"] = False
    workspace_setter["fn"] = None
    ask_bridge["fn"] = ask_user if interactive else None
    if rich_sink is not None:
        llm.set_delta_sink(rich_sink.on_delta)
        subagent_bridge["fn"] = rich_sink.subagent_notify


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
    # Validate documented enum env vars (warn + use default; never crash).
    from .envspec import validate_env
    for _w in validate_env():
        print(f"  · config warning: {_w}")

    from .llm import OpenAILLM
    from .loop import run_turn
    from .memory import make_memory
    from .oracle import CommandOracle
    from .session import MAX_WORKSPACE_TRANSITIONS, route
    from .pfc import consolidate_checkpoint, record_user, slice_sink
    from .seed import make_build_slice
    from .text_utils import one_line

    # cfg already loaded above (for the config→env key population + the gate)
    root = os.getcwd()
    _workspace_handoff = {"target": "", "ready": False}
    from .runtime_persistence import WorkspaceTransitionStore
    _workspace_transitions = WorkspaceTransitionStore()

    def _schedule_workspace(target: str) -> WorkspaceScheduleDecision:
        existing = str(_workspace_handoff.get("target") or "")
        try:
            logical = session.logical_turn
        except (NameError, AttributeError):
            logical = None
        decision = _classify_workspace_schedule(
            root, target,
            pending_target=existing,
            workspace_switches=int(getattr(logical, "workspace_switches", 0) or 0),
            workspace_edges=getattr(logical, "workspace_edges", ()),
            max_transitions=MAX_WORKSPACE_TRANSITIONS,
        )
        if not decision.accepted:
            return decision
        _workspace_handoff.update(target=target, ready=False)
        return decision

    from .config import load_prefs, save_prefs
    _prefs = load_prefs()
    mine_mode = cfg.mine           # deterministic | llm | off
    sub_depth = cfg.subagent_depth  # 0 disables delegation
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
    memory = make_memory()  # native L0/L2 always; optional Memem adapter never controls durability
    # Process-owned bridges stay stable while their workspace targets are replaced.
    _sub_render: dict = {"fn": None}
    def _notify_subagent(update):
        fn = _sub_render["fn"]
        if fn is not None:
            fn(update)
    _ask_user_bridge: dict = {"fn": None}

    def _workspace_ask_user(question, options):
        fn = _ask_user_bridge.get("fn")
        if fn is None:
            return "(no interactive answer; proceed with a stated assumption)"
        return fn(question, options)

    def _workspace_log(message: str) -> None:
        print(f"  · {message}")

    # A crash between workspace segments leaves one application-level transition record spanning both local
    # stores. Reuse its session namespace when SliceAgent restarts in either endpoint so the recovered Active
    # Work SourceRefs still resolve against the same event ledger. Ambiguous stale records are reported below
    # rather than choosing one by filename.
    _boot_workspace_transitions = _workspace_transitions.pending(workspace_root=root)
    _app_session_id = (
        _boot_workspace_transitions[0].session_id if len(_boot_workspace_transitions) == 1 else ""
    )

    def _prepare_workspace(target: str, initial_cfg=None) -> WorkspaceResources:
        from .config import load_config as _load_workspace_config
        if initial_cfg is not None:
            return _prepare_workspace_resources(
                target, cfg=initial_cfg, llm=llm, memory=memory,
                schedule_workspace=_schedule_workspace, notify_subagent=_notify_subagent,
                ask_user=_workspace_ask_user, on_log=_workspace_log,
                session_id=_app_session_id or None,
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
                target, cfg=target_cfg, llm=llm, memory=memory,
                schedule_workspace=_schedule_workspace, notify_subagent=_notify_subagent,
                ask_user=_workspace_ask_user, on_log=_workspace_log,
                session_id=_app_session_id or None,
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
    _set_memory_scope = getattr(memory, "set_scope", None)
    if callable(_set_memory_scope) and workspace.project_identity is not None:
        _set_memory_scope(
            project_id=workspace.project_identity.project_id,
            workspace_id=workspace.store.workspace_id,
            label=workspace.project_identity.label,
            workspace_root=root,
        )
    workspace_manager = WorkspaceManager(workspace, _prepare_workspace)
    session, local_store = workspace.session, workspace.store
    _app_session_id = session.session_id
    from .event_ledger import EventLedger, backfill_delivered_responses
    from .recovery import state_dir as _state_dir
    _ledger_root = _state_dir("event-ledger")
    _event_ledger = EventLedger(_app_session_id, root=_ledger_root)
    _boot_transition = None
    if len(_boot_workspace_transitions) == 1:
        pending = _boot_workspace_transitions[0]
        _boot_transition = pending
        if pending.status in {"activated", "continuing"}:
            from .recovery import root_key as _root_key
            _event_ledger.record(
                "context_transition", logical_turn_id=pending.logical_turn_id,
                task_id=pending.task_id,
                segment_id=f"{pending.logical_turn_id}:segment:{pending.target_segment_index}",
                workspace_epoch=pending.target_workspace_epoch,
                workspace_id=_root_key(pending.target_root),
                payload={
                    "source_root": pending.source_root, "target_root": pending.target_root,
                    "source_artifact_id": pending.source_artifact_id,
                    "target_artifact_id": pending.target_artifact_id,
                    "source_segment_index": pending.source_segment_index,
                    "target_segment_index": pending.target_segment_index,
                    "source_workspace_epoch": pending.source_workspace_epoch,
                    "target_workspace_epoch": pending.target_workspace_epoch,
                },
                identity=(pending.logical_turn_id, pending.source_segment_index,
                          pending.target_segment_index),
            )
        print(f"  · recovered interrupted workspace continuation ({pending.status}): "
              f"{pending.source_root} → {pending.target_root}; its request remains active")
    elif len(_boot_workspace_transitions) > 1:
        print(f"  · warning: {len(_boot_workspace_transitions)} unfinished workspace continuations refer to "
              "this path; none was selected as the application session identity")
    # The sealed artifact/checkpoint is the response commit point. Repair a crash gap after recording any
    # preceding workspace transition, and inspect the pending target store as well when startup occurs at A.
    _repair_artifacts = list(local_store.coordinator.artifacts.list_all())
    if _boot_transition is not None:
        from .persistence import ArtifactStore as _ArtifactStore
        from .recovery import root_key as _root_key
        _target_store_root = _state_dir("core", _root_key(_boot_transition.target_root))
        if os.path.realpath(_target_store_root) != os.path.realpath(local_store.store_root):
            _repair_artifacts.extend(_ArtifactStore(_target_store_root).list_all())
    _repaired_responses = backfill_delivered_responses(_repair_artifacts, root=_ledger_root)
    if _repaired_responses:
        # Refresh the in-memory index when the repaired event belongs to this intentionally reused app session.
        _event_ledger = EventLedger(_app_session_id, root=_ledger_root)
        print(f"  · repaired {_repaired_responses} delivered-response ledger projection(s)")
    # The application session can intentionally survive a workspace-boundary crash. Reconstruct the admitted
    # generation from its immutable source events; in-memory Session counters alone restart at zero.
    session.turn_generation = max(
        session.turn_generation, len(_event_ledger.events("user_utterance")),
    )
    sandbox, retriever = workspace.sandbox, workspace.retriever
    base_tools, tools, skills = workspace.base_tools, workspace.tools, workspace.skills
    base_tools._event_ledger = _event_ledger
    mcp_runtime = workspace.mcp_runtime
    reviewer, episodic, monitor_sink = workspace.reviewer, workspace.episodic, workspace.monitor_sink
    mine_mode, sub_depth = workspace.mine_mode, workspace.subagent_depth
    mcp_tool_count, plugin_tool_count = workspace.mcp_tool_count, workspace.plugin_tool_count
    def _bind_active_work_host(host) -> None:
        binder = getattr(host, "bind_active_work", None)
        if not callable(binder):
            return

        def snapshot():
            if session.active_id is None:
                raise ValueError("no active task")
            logical = session.logical_turn
            return (
                session.active().active_work,
                str(getattr(logical, "id", "") or ""),
                session.workspace_epoch,
            )

        binder(snapshot)

    _bind_active_work_host(base_tools)
    llm.set_cache_key(session.session_id)
    for _recovered in workspace.recovery_results:
        print(f"  · recovered local artifact {_recovered.artifact_id} ({_recovered.status})")
    _pending_seal_records: dict[str, tuple[int, dict]] = {}

    def _preview_turn_admission(text: str, state, task_id: str):
        """Create the versioned logistics envelope without interpreting the user's language.

        Active Work owns semantic continuity; the exact current request and one prior-assistant adjacency let
        the model resolve meaning.  The host retains focus only as an opaque compatibility locator and performs
        no proposal/assent/effect/evidence grammar or artifact scan on the production path.
        """
        from .discourse import AdmissionPreview
        from .intent import TurnAdmission

        return AdmissionPreview(
            admission=TurnAdmission(request_text=str(text)),
            focus=tuple(dict(item) for item in state.continuity.discourse_focus),
            consume_pending_proposal=True,
        )

    def _begin_local_turn(text: str, preview, *, action: str, task_id: str, state):
        """Journal one admission, then publish its task/focus/intent changes exactly once."""
        # ``Slice.turns`` and the live Session counter cannot identify a durable request root by themselves.
        # A persisted generation keeps useful ordering while the nonce prevents a crash gap or same-session
        # restart from ever rebinding an older logical/event identity.
        logical_id = _mint_logical_turn_id(
            session.session_id, session.turn_generation + 1, task_id,
        )
        active = local_store.begin(
            task_id=task_id or "t-none", logical_id=logical_id, user_request=text,
            segment_index=0, workspace_epoch=session.workspace_epoch,
        )
        session.turn_generation += 1
        from dataclasses import replace
        admission = replace(preview.admission, request_source=active.artifact_id)
        utterance_event = _event_ledger.record(
            "user_utterance", logical_turn_id=logical_id, task_id=task_id,
            segment_id=active.segment_id, workspace_epoch=session.workspace_epoch,
            workspace_id=local_store.workspace_id,
            payload={"text": text, "source_artifact_id": active.artifact_id},
            identity=(logical_id,),
        )
        local_store.record_admission({
            "action": action,
            "task_id": task_id,
            "logical_turn_id": logical_id,
            "source_event_id": utterance_event.id,
            # Deterministic crash-replay mirror of the event ledger's canonical persisted bytes.
            "source_event_text": str(utterance_event.payload.get("text") or ""),
            "segment_index": active.segment_index,
            "workspace_epoch": active.workspace_epoch,
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
        record_user(
            state, text, source_artifact=active.artifact_id,
            source_event_id=utterance_event.id, logical_id=logical_id,
            workspace_epoch=session.workspace_epoch,
            # SourceRef binds to the ledger's canonical persisted bytes.  The live current request above
            # remains verbatim; a persistence-redacted secret must not make restart validation fail.
            source_text=str(utterance_event.payload.get("text") or ""), contract=admission,
        )
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
        session.start_logical_turn(
            logical_id=logical_id, task_id=task_id, request=text,
            source_artifact_id=active.artifact_id, source_event_id=utterance_event.id,
            admission=admission, source_workspace=root,
        )
        nonlocal _boot_transition
        _boot_transition = _retire_recovered_transition(
            _workspace_transitions, _boot_transition, _workspace_log,
        )
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

    def _begin_workspace_continuation(transition):
        """Open the target-local segment journal and reinstall the same exact request without re-admission."""
        logical = session.logical_turn
        if logical is None or logical.id != transition.logical_turn_id:
            raise _DurabilityStop("workspace continuation lost its logical-turn identity")
        if session.active_id != transition.task_id or logical.task_id != transition.task_id:
            raise _DurabilityStop("workspace continuation lost its task identity")
        if logical.workspace_switches >= MAX_WORKSPACE_TRANSITIONS:
            raise _DurabilityStop("workspace continuation exhausted its transition budget")
        from dataclasses import replace
        next_index = logical.segment_index + 1
        active = local_store.begin(
            task_id=logical.task_id, logical_id=logical.id, user_request=logical.request,
            segment_index=next_index, workspace_epoch=session.workspace_epoch,
        )
        admission = replace(logical.admission, request_source=active.artifact_id)
        local_store.record_admission({
            "action": "workspace_continue", "task_id": logical.task_id,
            "logical_turn_id": logical.id, "segment_index": next_index,
            "workspace_epoch": session.workspace_epoch,
            "source_artifact_id": transition.source_artifact_id,
            "source_event_id": logical.source_event_id,
            "active_work": session.active().active_work.to_records(),
            "admission": admission.to_dict(), "focus": [],
            "consume_pending_proposal": False,
        })
        continued = session.begin_workspace_segment(
            source_artifact_id=active.artifact_id, admission=admission,
            workspace_path=transition.target_root,
        )
        if continued.segment_index != next_index or continued.workspace_epoch != session.workspace_epoch:
            raise _DurabilityStop("workspace continuation published inconsistent segment identity")
        transition = _workspace_transitions.mark_continuing(
            transition, target_artifact_id=active.artifact_id,
        )
        _event_ledger.record(
            "context_transition", logical_turn_id=logical.id, task_id=logical.task_id,
            segment_id=continued.segment_id, workspace_epoch=continued.workspace_epoch,
            workspace_id=local_store.workspace_id,
            payload={
                "source_root": transition.source_root, "target_root": transition.target_root,
                "source_artifact_id": transition.source_artifact_id,
                "target_artifact_id": active.artifact_id,
                "source_segment_index": transition.source_segment_index,
                "target_segment_index": transition.target_segment_index,
                "source_workspace_epoch": transition.source_workspace_epoch,
                "target_workspace_epoch": transition.target_workspace_epoch,
            },
            identity=(logical.id, transition.source_segment_index, transition.target_segment_index),
        )
        return transition

    def _record_response_delivered(artifact_id: str, stop_reason: str) -> None:
        """Record delivery only for the terminal segment whose final answer reached the user."""
        logical = session.logical_turn
        if logical is None or stop_reason != "end_turn" or not artifact_id:
            return
        _event_ledger.record(
            "response_delivered", logical_turn_id=logical.id, task_id=logical.task_id,
            segment_id=logical.segment_id, workspace_epoch=logical.workspace_epoch,
            workspace_id=local_store.workspace_id,
            payload={"artifact_id": artifact_id, "stop_reason": stop_reason},
            identity=(logical.id, artifact_id),
        )

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
        # A source workspace seal is a runtime-segment boundary, not delivery of the compound user request.
        # Mark it explicitly so the Active Work seal below keeps the same request root in progress.
        record = dict(record)
        record_meta = dict(record.get("meta") or {})
        if _workspace_handoff.get("target") and stop_reason == "end_turn":
            record_meta.update({
                "segment_outcome": "workspace_transition",
                "continuation_target": str(_workspace_handoff.get("target") or ""),
            })
        else:
            record_meta.setdefault("segment_outcome", "terminal")
        record["meta"] = record_meta
        # Prepare the next state on a copy. The live task is published only after artifact+checkpoint commit,
        # so a storage failure cannot half-seal working memory and then let a newer turn overtake it.
        import copy
        sealed_target = copy.deepcopy(target)
        sealed_target.seal()
        # Active Work owns semantic request lifecycle; receipts own only execution. A source workspace seal
        # keeps the same root in progress, while the one user-visible terminal response cites the artifact this
        # seal is about to publish. The graph and artifact enter the checkpoint atomically below.
        from .active_work import OutputRef
        transitioned = bool(_workspace_handoff.get("target") and stop_reason == "end_turn")
        response_ref = (
            OutputRef("turn_artifact", artifact_id)
            if stop_reason == "end_turn" and not transitioned else None
        )
        sealed_target.active_work = sealed_target.active_work.seal_current(
            stop_reason, response_ref=response_ref, transitioned=transitioned,
            logical_id=active.logical_id,
        )
        from dataclasses import asdict
        from .taskstate import slice_to_task_state, task_state_from_checkpoint
        task_status = ("indeterminate" if sealed_target.reconciliation_required else
                       ("active" if stop_reason == "end_turn" else "parked"))
        task_state = slice_to_task_state(
            sealed_target, active.task_id, session_id=session.session_id,
            # A clean model turn does not close the user's task. Only explicit task-boundary operations do.
            status=task_status,
            workspace_epoch=active.workspace_epoch,
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
        # Journal completeness may strengthen an apparently ordinary stop to INDETERMINATE while committing
        # (for example, ToolStarted was durable but Ctrl-C prevented ToolResult). Merge that advisory truth
        # back into the live copy so later claims see the same evidence as rehydration. Keep the live copy's
        # richer within-session continuity/working-set fields,
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
        sealed_target.continuity.last_receipt = (
            copy.deepcopy(receipt_projection) if isinstance(receipt_projection, dict) else None
        )
        sealed_target.continuity.last_receipt_artifact_id = artifact_id
        session.tasks[active.task_id] = sealed_target
        session.turn_task_id = None
        _pending_seal_records.pop(active.artifact_id, None)
        if getattr(memory, "is_durable", False):
            try:
                legacy_episode = copy.deepcopy(sealed_artifact.to_dict()["structured_body"])
                legacy_meta = legacy_episode.setdefault("meta", {})
                if isinstance(legacy_meta, dict):
                    identity = workspace.project_identity
                    legacy_meta["project_id"] = (
                        identity.project_id if identity is not None else "project-unscoped"
                    )
                    legacy_meta["workspace_id"] = local_store.workspace_id
                memory.append_episode(
                    session.session_id, active.task_id, history_turn,
                    legacy_episode,
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
    _stats = {"model": llm.model, "topic": "", "workspace": _ws_name(root), "tokens": 0}
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
    # always-pinned live composer: the bordered box stays at the bottom while its worker publishes lifecycle/
    # tool progress above it. Provider completion itself uses the off-main blocking watchdog, so response text
    # arrives assembled there rather than pretending an unfenced background SSE is safe. Opt-in/experimental;
    # the default REPL (box between turns) is the proven path, and live falls back to it if it can't start.
    tui_env = os.environ.get("AGENT_TUI", "").strip().lower()
    use_live = (_tui is not None and tui_env == "live")
    _live_runtime = {"active": use_live}

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

    # Inline UI owns stdin synchronously. Live mode installs its composer bridge in _live_ready below.
    if not _live_runtime["active"] and (_tui or sys.stdin.isatty()):
        _ask_user_bridge["fn"] = _ask_user

    # sinks: update the active slice from tool results, cache the turn, persist, print (no per-turn
    # miner — distillation is cache-only at session end via memory.consolidate).
    reducer = slice_sink(session)

    # Keep the store lookup behind tiny routers so required journaling and reduction always dereference the
    # currently-published workspace. The delegates themselves stay stable across in-process handoffs.
    def _journal_event(event):
        local_store.observe_event(event)

    def _record_application_effects(event, owner, store) -> None:
        if not isinstance(event, ToolResult) or event.outcome is None:
            return
        logical = getattr(owner, "logical_turn", None)
        if logical is None:
            return
        for effect in getattr(event.outcome, "effects", ()) or ():
            kind = getattr(effect, "kind", "")
            if kind not in {"work_delta", "child_artifact"}:
                continue
            payload = getattr(effect, "payload", {}) or {}
            if kind == "child_artifact":
                _event_ledger.record(
                    "child_artifact", logical_turn_id=logical.id, task_id=logical.task_id,
                    segment_id=logical.segment_id, workspace_epoch=logical.workspace_epoch,
                    workspace_id=store.workspace_id,
                    payload={"effect_id": effect.id, **dict(payload)},
                    identity=(logical.id, effect.id),
                )
                continue
            _event_ledger.record(
                "work_delta", logical_turn_id=logical.id, task_id=logical.task_id,
                segment_id=logical.segment_id, workspace_epoch=logical.workspace_epoch,
                workspace_id=store.workspace_id,
                payload={
                    "effect_id": effect.id,
                    "invocation_id": event.invocation_id,
                    "delta": payload.get("delta") or {},
                },
                identity=(logical.id, effect.id),
            )

    def _reduce_event(event):
        """Transactionally publish in-memory reduction with its durable applied-effect IDs."""
        if _is_workspace_transport_completion(_workspace_handoff, event):
            return
        if not isinstance(event, ToolResult) or event.outcome is None:
            reducer(event)
            return
        import copy
        task_id = session.active_id
        before = copy.deepcopy(session.tasks.get(task_id)) if task_id in session.tasks else None
        locally_committed = False
        try:
            reducer(event)
            local_store.observe_reduction(event)
            locally_committed = True
            # The application ledger is a replay/audit projection. Publish it only after the workspace-local
            # semantic transition is durable, so it can never claim an effect that rollback rejected.
            _record_application_effects(event, session, local_store)
        except Exception:
            if before is not None and not locally_committed:
                session.tasks[task_id] = before
            raise

    def _bind_journal_event():
        bound_store = local_store
        bound_active = bound_store.active

        def sink(event):
            # Identity, not merely artifact text: a recovered/new writer must never receive a callback from
            # the retired epoch. The scheduler already publishes an indeterminate settlement before sealing.
            if bound_store.active is bound_active:
                bound_store.observe_event(event)

        return sink

    def _bind_reduce_event():
        import copy

        bound_store = local_store
        bound_active = bound_store.active
        bound_session = session.target
        bound_reducer = slice_sink(bound_session)

        def sink(event):
            if bound_store.active is not bound_active:
                return
            if _is_workspace_transport_completion(_workspace_handoff, event):
                return
            if not isinstance(event, ToolResult) or event.outcome is None:
                bound_reducer(event)
                return
            task_id = bound_session.active_id
            before = copy.deepcopy(
                bound_session.tasks.get(task_id) if task_id in bound_session.tasks else None
            )
            try:
                bound_reducer(event)
                _record_application_effects(event, bound_session, bound_store)
                bound_store.observe_reduction(event)
            except Exception:
                if before is not None:
                    bound_session.tasks[task_id] = before
                raise

        return sink

    _journal_event.bind_dispatch = _bind_journal_event
    _reduce_event.bind_dispatch = _bind_reduce_event

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

    def _make_workspace_dispatch_for(dispatch_root, dispatch_episodic, dispatch_monitor):
        sinks = []
        if dispatch_episodic is not None:
            sinks.append(dispatch_episodic)
        sinks.append(log_sink(dispatch_root))
        if metrics is not None:
            sinks.append(metrics)
        sinks.append(_workspace_presentation_sink(_workspace_handoff, _presentation_sink))
        if dispatch_monitor is not None:
            sinks.append(dispatch_monitor)
        # Execution truth is journaled before reduction; applied transition IDs are journaled only after the
        # reducer succeeds. Presentation/metrics remain isolated observers.
        return make_dispatcher(*sinks, required=(_journal_event, _reduce_event))

    def _make_workspace_dispatch():
        return _make_workspace_dispatch_for(root, episodic, monitor_sink)

    dispatch = _make_workspace_dispatch()
    # Social fast-path messages are deliberately outside the task/turn lifecycle. Sending them through the
    # required dispatcher would overwrite the previous conversation exchange and bleed into the next
    # episode journal even though no local turn was opened.
    chitchat_dispatch = make_dispatcher(_presentation_sink)

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

    def _handle_slash(line, *, modal_safe: bool = False):  # TUI navigation palette — existing session ops
        nonlocal cfg
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0], (parts[1].strip() if len(parts) > 1 else "")
        if cmd not in SUPPORTED_SLASH_COMMANDS:
            _console.print(f"  unknown command {cmd} (/help)", markup=False)
            return True
        if cmd == "/help":
            _console.print(slash_help_line() + "\n  (type / for the menu · /config adds LLM "
                           "providers · /cwd shows the workspace; /cwd <path> switches it safely · "
                           "Esc = undo last turn · "
                           "say \"review my changes\" for code_review · @path pins a file · "
                           "quote paths with spaces as @\"docs/my guide.md\")")
        elif cmd == "/config":
            # THE clear user journey: providers are managed INSIDE sliceagent — same wizard as first-run
            # onboarding (provider → model → key → live test), then the new provider shows up in /model.
            if sys.stdin.isatty() and (modal_safe or not _live_runtime["active"]):
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
                                   + (", ".join(f"[bold]{_rich_escape(k)}[/] "
                                                f"({_rich_escape(v.get('model', '?'))})"
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
            for cost_line in _cost_lines(_stats, metrics):
                _console.print(cost_line, markup=False)
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
                except Exception as exc:  # noqa: BLE001 - report invalid/unavailable ids without crashing the UI
                    _console.print(f"  could not switch: {exc}", markup=False)
        elif cmd == "/undo":
            # markup=False: undo_last() embeds the edited file PATH; a Next.js-style '[id]'/'[...slug]'
            # segment is parsed as a Rich tag → corrupted output or a MarkupError crash.
            _console.print("  " + base_tools.undo_last(), markup=False)  # revert the last file edit
        elif cmd == "/plugins":
            plugin_names = _plugin_tool_names(base_tools.registry)
            # markup=False: plugin dirs are filesystem PATHS that may contain '[...]' (same Rich-tag hazard)
            _console.print(f"  plugin dirs: {', '.join(cfg.plugin_dirs) or '(none configured)'}", markup=False)
            _console.print(
                f"  plugin tools ({len(plugin_names)}): {', '.join(plugin_names) or '(none loaded)'}",
                markup=False,
            )
        elif cmd == "/mcp":
            configured = list(cfg.mcp_servers.keys())
            mtools = sorted(e.name for e in base_tools.registry._tools.values()
                            if getattr(e, "source", "") == "mcp")
            if not configured and not mtools:
                _console.print("  no MCP servers configured — add [mcp_servers.<name>] to ~/.sliceagent/config.toml")
            else:
                _console.print(f"  configured servers: {', '.join(configured) or '(none)'}", markup=False)
                _console.print(
                    f"  connected tools ({len(mtools)}): "
                    f"{', '.join(mtools) or '(none — check startup logs)'}",
                    markup=False,
                )
        elif cmd == "/skills":
            _console.print("\n".join(_discovery_skill_lines(skills)), markup=False)
        elif cmd == "/tools":
            _console.print("\n".join(_discovery_tool_lines(tools, base_tools.registry)), markup=False)
        elif cmd == "/agents":
            _console.print("\n".join(_discovery_agent_lines(tools)), markup=False)
        elif cmd == "/model":
            if not arg and sys.stdin.isatty() and (modal_safe or not _live_runtime["active"]):
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
                    _console.print(f"  ✓ model → [bold]{_rich_escape(llm.model)}[/]"
                                   + (f" @ [bold]{_rich_escape(_pid)}[/]" if _pid else "")
                                   + f" · reasoning [bold]{_rich_escape(llm.reasoning)}[/] (saved)"
                                   + (f"\n  {_rich_escape(note)}" if note else ""))
            elif not arg:
                _console.print(f"  model: [bold]{_rich_escape(llm.model)}[/]  ·  reasoning: "
                               f"[bold]{_rich_escape(llm.reasoning)}[/]"
                               f"  ·  net: {_rich_escape(getattr(llm, 'proxy_used', 'direct'))}")
                _console.print("  switch:  /model <name> [fast|full|high|max]  (same endpoint)")
                provs = cfg.providers()
                if provs:
                    _console.print(
                        "  configured providers (pick via the /model menu to switch endpoint too): "
                        + ", ".join(f"{k}={v.get('model', '?')}" for k, v in provs.items()),
                        markup=False,
                    )
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
                _console.print(f"  ✓ model → [bold]{_rich_escape(llm.model)}[/]"
                               + (f" · reasoning [bold]{_rich_escape(llm.reasoning)}[/]" if eff else "")
                               + " (saved)" + (f"\n  {_rich_escape(note)}" if note else ""))
        elif cmd == "/reasoning":
            if arg.lower() not in ("fast", "full", "high", "max"):
                _console.print("  usage: /reasoning <fast|full|high|max>"
                               "   (full = provider default; high/max use /v1/responses for gpt-5)")
            else:
                llm.switch(reasoning=arg)
                save_prefs({"reasoning": llm.reasoning})
                note = _reasoning_note(llm)
                _console.print(
                    f"  ✓ reasoning → [bold]{_rich_escape(llm.reasoning)}[/] (saved)"
                    + (f"\n  {_rich_escape(note)}" if note else ""),
                )
        elif cmd == "/cwd":
            target, message = _resolve_workspace_target(base_tools.root(), arg)
            if target is None:
                _console.print("  " + message, markup=False)
            else:
                decision = _schedule_workspace(target)
                if not decision.accepted:
                    glyph = "↷" if decision.status is ToolStatus.STEERED else "✗"
                    _console.print(f"  {glyph} {decision.message}", markup=False)
                else:
                    _workspace_handoff["ready"] = True  # slash commands run between already-sealed turns
                    _switch_workspace(target)
        else:
            _console.print(f"  unknown command {cmd} (/help)", markup=False)
        return True

    _IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

    def _expand_mentions(text):
        """@path mentions: pin each EXISTING, workspace-confined file into
        the slice's OPEN FILES; an @image is ATTACHED as a vision content part for the next turn when the model
        supports vision (else skipped with a hint). Whitespace paths use @"quoted path". Best-effort; leaves
        the text intact."""
        if "@" not in text or session.active_id is None:
            return
        from .model_catalog import capability
        from .pfc import touch_file
        vision = capability(llm.model, getattr(llm, "_base_url", "")).supports_vision
        pinned, images, skipped = [], [], []
        for rel in workspace_mentions(text, base_tools.root()):
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

    def _make_workspace_hooks_for(hook_root, hook_cfg, hook_session):
        # Smart-kernel boundary: the model owns semantic judgment, ordinary action choice, and response
        # completion. The host keeps workspace lifecycle integrity, a narrow catastrophic-command floor, and
        # explicit resource controls. Active Work and deliverable metadata are context—not publication gates.
        hook_list = [
            _WorkspaceHandoffHook(_workspace_handoff),
            CatastrophicSafeguardHook(),
        ]
        if hook_cfg.verify_cmd:
            oracle = CommandOracle(hook_cfg.verify_cmd, root=hook_root)
            hook_list.append(OracleHook(
                oracle,
                lambda out: setattr(
                    hook_session.active(), "last_error", f"Verification failed:\n{out[:600]}",
                ),
            ))
        if hook_cfg.max_tokens:
            hook_list.append(BudgetHook(hook_cfg.max_tokens))
        return CompositeHooks(*hook_list)

    def _make_workspace_hooks():
        return _make_workspace_hooks_for(root, cfg, session)

    hooks = _make_workspace_hooks()

    def _workspace_info_for(info_cfg, info_retriever, info_episodic, info_mine,
                            info_sub_depth, info_skills, info_mcp_count, info_plugin_count) -> str:
        return (f"model={llm.model} · net={getattr(llm, 'proxy_used', 'direct')} · "
                f"sandbox={info_cfg.sandbox_backend} · code={type(info_retriever).__name__} · "
                f"memory={type(memory).__name__} · "
                f"episodic={'on' if info_episodic is not None else 'off'} · "
                f"mine={info_mine} · subagents={'on' if info_sub_depth > 0 else 'off'} · "
                f"skills={len(info_skills.names())} · mcp_tools={info_mcp_count} · "
                f"plugin_tools={info_plugin_count}")

    def _workspace_info() -> str:
        return _workspace_info_for(
            cfg, retriever, episodic, mine_mode, sub_depth, skills,
            mcp_tool_count, plugin_tool_count,
        )

    info = _workspace_info()
    # ── choose UI: the always-pinned live composer (AGENT_TUI=live), else the rich+prompt_toolkit REPL ──
    _input = _tui.TuiInput(_stats, root=root) if _tui else None
    _live_workspace_setter: dict[str, Any] = {"fn": None}
    _retired_sessions: list[tuple[str, str, str, str]] = []

    def _workspace_notice(message: str) -> None:
        try:
            _workspace_log(message)
        except Exception:
            pass

    def _publish_workspace(candidate: WorkspaceResources) -> None:
        """Atomically redirect every workspace-facing delegate; process-owned UI/LLM objects stay intact."""
        nonlocal workspace, root, cfg, session, local_store, sandbox, retriever
        nonlocal base_tools, tools, skills, mcp_runtime, reviewer, episodic, monitor_sink
        nonlocal mine_mode, sub_depth, mcp_tool_count, plugin_tool_count, reducer, hooks, dispatch, info
        nonlocal _project_env_overlay

        from .session import SessionBinding, rebase_session_for_workspace

        previous_session = session.session_id
        previous_mine_mode = mine_mode
        previous_identity = workspace.project_identity
        previous_workspace_id = workspace.store.workspace_id
        previous_root = root
        if not isinstance(session, SessionBinding) or not isinstance(candidate.session, SessionBinding):
            raise TypeError("workspace resources must use an application SessionBinding")
        candidate_binding = candidate.session
        merged_session = rebase_session_for_workspace(session.target, candidate_binding.target)
        # Everything that can construct/validate derived runtime objects happens before either live binding
        # moves. If one of these raises, WorkspaceManager can close B and A's session/task identity is untouched.
        next_reducer = slice_sink(candidate_binding)
        next_hooks = _make_workspace_hooks_for(candidate.root, candidate.config, candidate_binding)
        next_dispatch = _make_workspace_dispatch_for(
            candidate.root, candidate.episodic, candidate.monitor_sink,
        )
        next_info = _workspace_info_for(
            candidate.config, candidate.retriever, candidate.episodic,
            candidate.mine_mode, candidate.subagent_depth, candidate.skills,
            candidate.mcp_tool_count, candidate.plugin_tool_count,
        )
        # Scope the shared knowledge facade before target tools/ContextFS become
        # reachable. A failed hard-scope transition aborts the staged handoff;
        # serving workspace B through a facade still querying A is never allowed.
        _set_scope = getattr(memory, "set_scope", None)
        if callable(_set_scope) and candidate.project_identity is not None:
            try:
                _set_scope(
                    project_id=candidate.project_identity.project_id,
                    workspace_id=candidate.store.workspace_id,
                    label=candidate.project_identity.label,
                    workspace_root=candidate.root,
                )
            except Exception as exc:
                if previous_identity is not None:
                    try:
                        _set_scope(
                            project_id=previous_identity.project_id,
                            workspace_id=previous_workspace_id,
                            label=previous_identity.label,
                            workspace_root=previous_root,
                        )
                    except Exception:
                        pass
                raise RuntimeError(
                    f"memory scope could not bind target workspace ({type(exc).__name__}: {exc})"
                ) from exc
        for key in tuple(_project_env_overlay):
            os.environ.pop(key, None)
        _project_env_overlay = {}
        workspace = candidate
        root, cfg = candidate.root, candidate.config
        # Keep the application-owned binding object stable. Every candidate-owned closure captured its own
        # binding during staging; point both bindings at the same merged view before exposing target tools.
        session.bind(merged_session)
        candidate_binding.bind(merged_session)
        candidate.session = session
        local_store = candidate.store
        sandbox, retriever = candidate.sandbox, candidate.retriever
        base_tools, tools, skills = candidate.base_tools, candidate.tools, candidate.skills
        base_tools._event_ledger = _event_ledger
        _bind_active_work_host(base_tools)
        mcp_runtime = candidate.mcp_runtime
        reviewer, episodic, monitor_sink = candidate.reviewer, candidate.episodic, candidate.monitor_sink
        mine_mode, sub_depth = candidate.mine_mode, candidate.subagent_depth
        mcp_tool_count, plugin_tool_count = candidate.mcp_tool_count, candidate.plugin_tool_count
        reducer, hooks, dispatch, info = next_reducer, next_hooks, next_dispatch, next_info
        _pending_seal_records.clear()
        _retired_sessions.append((
            previous_session,
            previous_mine_mode,
            str(getattr(previous_identity, "project_id", "") or "project-unscoped"),
            str(previous_workspace_id or ""),
        ))

        # set_cache_key changes request/cache identity only; it does not reconstruct the provider client.
        try:
            llm.set_cache_key(session.session_id)
        except Exception as exc:  # noqa: BLE001 — a cache hint cannot invalidate a valid workspace swap
            _workspace_notice(f"cache key refresh failed ({type(exc).__name__}: {exc})")
        _stats["workspace"] = _ws_name(root)
        _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
        if _input is not None:
            try:
                _input.set_workspace(root)
            except Exception as exc:  # noqa: BLE001 — completion refresh is cosmetic
                _workspace_notice(f"file completion refresh failed ({type(exc).__name__}: {exc})")
        live_setter = _live_workspace_setter.get("fn")
        if live_setter is not None:
            try:
                live_setter(root)
            except Exception as exc:  # noqa: BLE001 — keep the running composer even if completion fails
                _workspace_notice(f"live completion refresh failed ({type(exc).__name__}: {exc})")
        for recovered in candidate.recovery_results:
            _workspace_notice(f"recovered local artifact {recovered.artifact_id} ({recovered.status})")

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

    def _activate_workspace_handoff(stop_reason: str):
        """Publish a requested workspace and open the next segment of the *same* logical turn.

        Returns the crash-visible transition while target work is active, otherwise ``None``.  The source
        segment has already sealed before this function is called; target preparation never races its writer.
        """
        target = str(_workspace_handoff.get("target") or "")
        if not target:
            return None
        if stop_reason != "end_turn":
            _workspace_handoff.update(target="", ready=False)
            glyph = "↷" if stop_reason in {"aborted", "cancelled"} else "!"
            message = f"  {glyph} workspace switch cancelled because the turn stopped as {stop_reason!r}"
            (_console.print(message, markup=False) if _console is not None else print(message))
            return None
        logical = session.logical_turn
        # The local active writer is None after a successful source seal. The sealed source identity remains on
        # the logical turn and is also deterministic from this workspace's segment journal.
        if logical is None or logical.workspace_switches >= MAX_WORKSPACE_TRANSITIONS:
            _workspace_handoff.update(target="", ready=False)
            message = "  ✗ workspace switch cancelled because its logical-turn identity is unavailable"
            (_console.print(message, markup=False) if _console is not None else print(message))
            return None
        try:
            transition = _workspace_transitions.prepare(
                session_id=session.session_id, logical_turn_id=logical.id, task_id=logical.task_id,
                request=logical.request, source_root=root, target_root=target,
                source_artifact_id=logical.source_artifact_id,
                source_segment_index=logical.segment_index,
                source_workspace_epoch=logical.workspace_epoch,
            )
        except Exception as exc:  # noqa: BLE001 — do not cross a boundary without a durable continuation ticket
            _workspace_handoff.update(target="", ready=False)
            message = (f"  ✗ workspace switch not started: transition could not be saved "
                       f"({type(exc).__name__}: {exc})")
            (_console.print(message, markup=False) if _console is not None else print(message))
            return None
        _workspace_handoff["ready"] = True
        if not _switch_workspace(target):
            _workspace_transitions.clear(transition)
            return None
        # From this point on, a failure is a durability stop rather than a recoverable navigation error: the
        # process has published B, so the transition record must remain available for restart diagnostics.
        transition = _workspace_transitions.mark_activated(transition)
        return _begin_workspace_continuation(transition)

    def _run_one_turn(text, sink, signal):
        """One turn for the LIVE composer: route (lexical) → build slice → run_turn with a per-turn dispatch
        that feeds the LiveSink. Runs in run_live's worker thread, so the pinned box stays responsive."""
        active_state = session.active() if session.active_id is not None else None
        if _use_chitchat_fast_path(text, active_state):  # pure social message → cheap reply, no slice/tools
            _chitchat_reply(text, make_dispatcher(sink))
            return
        if session.active_id is None:
            action, tid = "new", ""
        else:
            action, tid = route(llm, text, session)
        _admit_routed_turn(text, action, tid)
        import time as _t
        _t0 = _t.monotonic()
        from . import recovery as _rec
        transition = None
        while True:
            _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
            # Rebuild every workspace-owned sink after a handoff. The LiveSink itself is process-owned and is
            # wrapped only to hide the source segment's transport acknowledgement/final commit.
            _live_sinks = []
            if episodic is not None:
                _live_sinks.append(episodic)
            _live_sinks.append(log_sink(root))
            if metrics is not None:
                _live_sinks.append(metrics)
            if monitor_sink is not None:
                _live_sinks.append(monitor_sink)
            _live_sinks.append(_workspace_presentation_sink(_workspace_handoff, sink))
            live_dispatch = make_dispatcher(
                *_live_sinks, required=(_journal_event, _reduce_event),
            )
            live_dispatch(TurnStarted(
                request=text, task_title=_stats["topic"], task_id=session.active_id or "",
                plan=list(session.active().plan or ()),
                turn_id=local_store.active.artifact_id if local_store.active else "",
            ))
            try:
                _expand_mentions(text)        # @path → pin the file into OPEN FILES
                build = make_build_slice(
                    session, tools, retriever, memory, text, session.session_id, model_id=llm.model,
                    event_ledger=_event_ledger,
                )
            except KeyboardInterrupt:
                live_dispatch(TurnInterrupted("aborted", "cancelled during context preparation"))
                if _seal_local_turn("aborted", live_dispatch):
                    _rec.clear(root)
                    if transition is not None:
                        _workspace_transitions.clear(transition)
                    session.finish_logical_turn()
                    return
                raise _DurabilityStop("required local seal failed after cancellation")
            except Exception as exc:  # noqa: BLE001 — preparation is inside the required durability lifecycle
                message = f"context preparation failed ({type(exc).__name__}: {exc})"
                live_dispatch(TurnInterrupted("error", message))
                if _seal_local_turn("error", live_dispatch):
                    _rec.clear(root)
                    if transition is not None:
                        _workspace_transitions.clear(transition)
                    session.finish_logical_turn()
                    return
                raise _DurabilityStop("required local seal failed after a context-preparation error")
            llm.set_delta_sink(sink.on_delta)
            segment_artifact_id = local_store.active.artifact_id if local_store.active else ""
            result = run_turn(
                build_slice=build, llm=llm, tools=tools, dispatch=live_dispatch,
                hooks=hooks, signal=signal, max_steps=cfg.max_steps,
                consolidate=lambda: consolidate_checkpoint(session.active(), compact=False),
                checkpoint=lambda m, s, _g=text: _rec.record(root, goal=_g, messages=m, step=s),
                turn_id=local_store.active.artifact_id if local_store.active else "",
            )
            if _seal_local_turn(result.stop_reason, live_dispatch):
                _rec.clear(root)                           # clear WAL only after this segment's required commit
            else:
                raise _DurabilityStop(
                    "required local seal could not complete; the recovery journal is retained. "
                    "Restart SliceAgent after fixing storage so recovery can finish before new work.")
            settled_transition, transition = transition, None
            handoff_requested = bool(_workspace_handoff.get("target"))
            # A continuing ticket has completed once its target segment seals. Retire it before preparing a
            # further boundary so a crash can never leave two live transition heads for one logical request.
            if handoff_requested and settled_transition is not None:
                _workspace_transitions.clear(settled_transition)
                settled_transition = None
            transition = _activate_workspace_handoff(result.stop_reason)
            if transition is not None:
                continue                                  # same exact request, target workspace, no user echo/route
            _stats["last_turn_s"] = _t.monotonic() - _t0
            if not handoff_requested:
                _record_response_delivered(segment_artifact_id, result.stop_reason)
            if settled_transition is not None:
                _workspace_transitions.clear(settled_transition)
            session.finish_logical_turn()
            if reviewer is not None:
                reviewer.review(session.session_id)
            return None

    def _try_live() -> bool:
        """Run the always-pinned live composer; return True if it ran (REPL below is then skipped), False to
        fall back to the REPL on any startup failure — input is never left broken."""
        try:
            def _live_ready(setter, requester):
                _live_workspace_setter["fn"] = setter
                _ask_user_bridge["fn"] = requester

            def _live_turn(text, sink, signal):
                # The process-owned child bridge follows the one active LiveSink and is
                # retired at the same boundary.  Child workers never write to the terminal.
                previous = _sub_render.get("fn")
                _sub_render["fn"] = sink.subagent_notify
                # Bind before routing/context preparation.  An LLM router may stream;
                # leaving the previous terminal sink installed lets a late delta clear
                # the new turn's Preparing state.
                llm.set_delta_sink(sink.on_delta)
                try:
                    return _run_one_turn(text, sink, signal)
                finally:
                    if _sub_render.get("fn") == sink.subagent_notify:
                        _sub_render["fn"] = previous

            _tui.run_live(console=_console, stats=_stats, banner_info=info, root=root,
                          run_one_turn=_live_turn, handle_slash=_handle_slash,
                          handle_modal_slash=lambda line: _handle_slash(line, modal_safe=True),
                          on_ready=_live_ready)
            return True
        except _tui.LiveWorkerRetirementError:
            # The renderer is gone but a turn thread still owns session/workspace state. Starting the inline
            # REPL would create a second concurrent owner, so fail the session boundary instead of falling back.
            _live_runtime["active"] = False
            _live_workspace_setter["fn"] = None
            _ask_user_bridge["fn"] = None
            raise
        except Exception as _e:  # noqa: BLE001
            # A live turn installs its own delta sink. If the Application later fails, the inline REPL uses
            # the process-wide Rich sink again; restore that stream target explicitly rather than continuing
            # to update the retired live status object. Child activity follows the same ownership transfer.
            _restore_inline_after_live_failure(
                llm=llm, rich_sink=_rich, live_runtime=_live_runtime,
                workspace_setter=_live_workspace_setter, ask_bridge=_ask_user_bridge,
                ask_user=_ask_user, subagent_bridge=_sub_render,
                interactive=bool(_tui or sys.stdin.isatty()),
            )
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

    _live_ran = False
    if use_live:
        try:
            _live_ran = _try_live()
        except _tui.LiveWorkerRetirementError as exc:
            print(f"\n  fatal: {exc}. Inline fallback was not started because the prior turn still owns "
                  "runtime state; restart SliceAgent after the operation settles.")
            raise SystemExit(2) from None
    if _live_ran:
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
            active_state = session.active() if session.active_id is not None else None
            if _use_chitchat_fast_path(line, active_state):  # pure social message → cheap reply
                _chitchat_reply(line, chitchat_dispatch)
                continue
            if session.active_id is None:                      # first message bootstraps the first topic
                action, tid = "new", ""
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
            import time as _time
            _t0 = _time.monotonic()
            transition = None
            stop_inline_session = False
            while True:
                _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
                dispatch(TurnStarted(
                    request=line, task_title=_stats["topic"], task_id=session.active_id or "",
                    plan=list(session.active().plan or ()),
                    turn_id=local_store.active.artifact_id if local_store.active else "",
                ))
                try:
                    # Slice construction belongs to each workspace segment. The exact request is reused, but is
                    # neither echoed nor admitted again after a switch.
                    _expand_mentions(line)
                    build = make_build_slice(
                        session, tools, retriever, memory, line, session.session_id, model_id=llm.model,
                        event_ledger=_event_ledger,
                    )
                except KeyboardInterrupt:
                    dispatch(TurnInterrupted("aborted", "cancelled during context preparation"))
                    if _seal_local_turn("aborted", dispatch):
                        recovery.clear(root)
                        if transition is not None:
                            _workspace_transitions.clear(transition)
                        session.finish_logical_turn()
                        print("\n  · cancelled")
                    else:
                        print("\n  · required local seal failed; stopping before accepting another request")
                        stop_inline_session = True
                    break
                except Exception as exc:  # noqa: BLE001 — preparation belongs to required durability
                    message = f"context preparation failed ({type(exc).__name__}: {exc})"
                    dispatch(TurnInterrupted("error", message))
                    if _seal_local_turn("error", dispatch):
                        recovery.clear(root)
                        if transition is not None:
                            _workspace_transitions.clear(transition)
                        session.finish_logical_turn()
                        print(f"\n  · {message}")
                    else:
                        print("\n  · required local seal failed; stopping before accepting another request")
                        stop_inline_session = True
                    break
                if os.environ.get("AGENT_TIMING"):
                    import time as _tt
                    _b = build
                    def build(_b=_b):
                        _s = _tt.monotonic()
                        r = _b()
                        print(f"  ⏱ slice build {(_tt.monotonic() - _s) * 1000:.0f} ms (progress was already "
                              "visible; the remaining wait is the model's first token)", flush=True)
                        return r
                # Esc and ctrl-c share the same run_turn cancellation path for every segment.
                _esc = _tui.make_esc_sentinel() if _tui else None
                if _esc is not None:
                    _esc.start()
                try:
                    segment_artifact_id = local_store.active.artifact_id if local_store.active else ""
                    result = run_turn(
                        build_slice=build, llm=llm, tools=tools, dispatch=dispatch, hooks=hooks,
                        max_steps=cfg.max_steps,
                        consolidate=lambda: consolidate_checkpoint(session.active(), compact=False),
                        checkpoint=lambda m, s, _g=line: recovery.record(
                            root, goal=_g, messages=m, step=s,
                        ),
                        turn_id=local_store.active.artifact_id if local_store.active else "",
                    )
                finally:
                    if _esc is not None:
                        _esc.stop()
                if not _seal_local_turn(result.stop_reason, dispatch):
                    print("  · required local seal failed; stopping before accepting another request")
                    stop_inline_session = True
                    break
                recovery.clear(root)
                settled_transition, transition = transition, None
                handoff_requested = bool(_workspace_handoff.get("target"))
                if handoff_requested and settled_transition is not None:
                    _workspace_transitions.clear(settled_transition)
                    settled_transition = None
                transition = _activate_workspace_handoff(result.stop_reason)
                if transition is not None:
                    continue                              # automatic target segment; no synthetic `go`
                _stats["last_turn_s"] = _time.monotonic() - _t0
                if not handoff_requested:
                    _record_response_delivered(segment_artifact_id, result.stop_reason)
                if settled_transition is not None:
                    _workspace_transitions.clear(settled_transition)
                session.finish_logical_turn()
                if reviewer is not None:
                    reviewer.review(session.session_id)
                break
            if stop_inline_session:
                break

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
        # Derive typed L2 records (with L0 provenance) and adjacent skills from the legacy episodic mirror.
        # mine_mode gates it: off → skip; deterministic → recorded skills; llm → generalized skills.
        if getattr(memory, "is_durable", False):
            current_identity = workspace.project_identity
            sessions_to_consolidate = list(dict.fromkeys(
                [*_retired_sessions, (
                    session.session_id,
                    mine_mode,
                    str(getattr(current_identity, "project_id", "") or "project-unscoped"),
                    str(workspace.store.workspace_id or ""),
                )]
            ))
            for session_id, mode, scoped_project_id, scoped_workspace_id in sessions_to_consolidate:
                if mode in ("0", "off", "none"):
                    continue
                consolidate_bound = getattr(memory, "consolidate_for_project", None)
                st = _safe(
                    "memory consolidation",
                    (
                        lambda sid=session_id, selected=mode, pid=scoped_project_id,
                               wid=scoped_workspace_id: consolidate_bound(
                                   sid, project_id=pid, workspace_id=wid,
                                   llm=llm, mode=selected,
                               )
                    ) if callable(consolidate_bound) else (
                        lambda sid=session_id, selected=mode: memory.consolidate(
                            sid, llm=llm, mode=selected,
                        )
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
