"""The contracts the core depends on — never the implementations.

The moat (loop + tiers) talks only to these. Everything commodity (LLM I/O,
retrieval, tool execution/sandbox, verification) lives behind them and is swappable.
Policy (Oracle/permissions/budget) is supplied via hooks.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class AssistantMessage:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None            # {"prompt_tokens": int, "completion_tokens": int}
    finish_reason: str | None = None     # provider's raw finish reason → normalized by the loop


@dataclass
class Snippet:
    path: str
    text: str
    score: float = 0.0


@dataclass
class TaskRef:
    """A bounded index row for the OTHER OPEN THREADS tier (Step 3)."""
    task_id: str
    title: str
    status: str            # active | parked | done | abandoned
    updated: str = ""


@dataclass
class TaskState:
    """Resumable, distilled state for one task = the serializable Slice fields. Stores REFS
    (file paths + anchors), never file contents — ground truth is re-read from disk on resume.
    Transient tiers (recent, action_log, active_skills) are intentionally NOT serialized."""
    task_id: str
    session_id: str = ""
    title: str = ""
    status: str = "active"
    goal: str = ""
    findings: list[str] = field(default_factory=list)
    active_files: list[str] = field(default_factory=list)
    edited_files: list[str] = field(default_factory=list)   # list on the wire; a set in the Slice
    edit_anchor: dict[str, str] = field(default_factory=dict)
    last_error: str = ""
    since_edit: int = 0
    links: list[str] = field(default_factory=list)          # task-graph edges (Step 3)
    tags: str = ""                                          # comma-joined (matches remember()/_tags)
    resolution: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic completion + tool-calling. (borrow: official SDKs / LiteLLM)
    May optionally expose `is_retryable(error) -> bool` for the retry policy."""
    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantMessage: ...


@runtime_checkable
class ToolHost(Protocol):
    """Executes tools, ideally behind a sandbox. (borrow: container / MCP / OpenHands runtime)"""
    def schemas(self) -> list[dict]: ...
    def run(self, name: str, args: dict) -> str: ...
    def read_text(self, path: str) -> str: ...   # reconstruct the artifacts tier (raises if missing)
    def accesses(self, name: str, args: dict) -> list: ...  # resource accesses for the scheduler


@runtime_checkable
class Retriever(Protocol):
    """Code discovery for the RELATED CODE tier (repo search). (build: ripgrep + tree-sitter)"""
    def retrieve(self, query: str, k: int = 6) -> list[Snippet]: ...


@runtime_checkable
class Memory(Protocol):
    """Cross-session memory + the durable STATE VAULT (episodic cache, task-state, lessons).
    Distinct from Retriever (memem indexes a curated vault, NOT source code). `is_durable` is the
    structural no-op marker: NullMemory sets it False so hosts skip cache/checkpoint wiring (keeps
    evals deterministic). The full surface is frozen here; implementations land incrementally.
    NOTE: @runtime_checkable isinstance() verifies method-NAME presence only — not signatures or
    return types; behavioral fidelity is enforced by the round-trip tests."""
    is_durable: bool
    # --- long-term lessons (exists) ---
    def recall(self, query: str, k: int = 6) -> list[Snippet]: ...
    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "") -> None: ...
    # --- episodic cache (lossless; never recalled into the LLM context) ---
    def append_episode(self, session_id: str, task_id: str, turn: int, record: dict) -> None: ...
    # read side: the model's on-demand valve into the cold cache (recall_history tool). Returns
    # raw line dicts ({v,session_id,task_id,turn,ts,record}); the host renders/bounds them.
    def read_episodes(self, session_id: str, *, limit: int | None = None) -> list[dict]: ...
    # --- task state / resume ---
    def checkpoint_task(self, task: TaskState) -> None: ...
    def load_task(self, task_id: str) -> TaskState | None: ...
    def list_session_tasks(self, session_id: str) -> list[TaskRef]: ...
    # --- consolidation / retrieval-feedback (declared now; implemented in later steps) ---
    def mark_used(self, memory_id: str) -> None: ...
    def consolidate(self, session_id: str) -> None: ...


@runtime_checkable
class Oracle(Protocol):
    """Ground-truth verification independent of retrieval. (borrow: the project's test/lint runners)"""
    def verify(self) -> tuple[bool, str]: ...
