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
    """Per-turn retrieval (+ cross-session memory). (plug: memem)"""
    def retrieve(self, query: str, k: int = 6) -> list[Snippet]: ...


@runtime_checkable
class Oracle(Protocol):
    """Ground-truth verification independent of retrieval. (borrow: the project's test/lint runners)"""
    def verify(self) -> tuple[bool, str]: ...
