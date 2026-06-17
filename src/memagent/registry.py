"""ToolRegistry — one registry, many sources (builtin / MCP / plugin / skill).

Borrowed from Hermes' tools/registry.py (the `generation` counter + `check` availability
gate) and Kimi's three-source single-registry projection. The ToolHost projects
schemas()/run()/accesses() from here, so every tool — wherever it comes from —
satisfies one contract and appears in one list. This is the keystone of Step ③:
MCP, plugins, and skills all register into the SAME registry the loop already drives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .access import AllAccess

Handler = Callable[[dict], str]      # (args) -> result string
AccessFn = Callable[[dict], list]    # (args) -> list[Access] for the scheduler/permissions


def _all_access(_args: dict) -> list:
    return [AllAccess()]


@dataclass
class ToolEntry:
    name: str
    schema: dict                              # {"type":"function","function":{name,description,parameters}}
    handler: Handler
    accesses: AccessFn = _all_access
    check: Optional[Callable[[], bool]] = None  # availability gate (None = always available)
    source: str = "builtin"                  # builtin | mcp | plugin | skill


class ToolRegistry:
    """A name->ToolEntry map with a generation counter (for downstream schema caching)
    and a per-tool availability gate. Robust by construction: a flaky check or handler
    hides/erros the one tool, never the whole registry."""

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}
        self.generation = 0

    def register(self, entry: ToolEntry, *, override: bool = False) -> None:
        if entry.name in self._tools and not override:
            raise ValueError(f"tool {entry.name!r} already registered (pass override=True to replace)")
        self._tools[entry.name] = entry
        self.generation += 1

    def deregister(self, name: str) -> None:
        if self._tools.pop(name, None) is not None:
            self.generation += 1

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return [e.name for e in self._available()]

    def _available(self) -> list[ToolEntry]:
        out = []
        for e in self._tools.values():
            try:
                if e.check is None or e.check():
                    out.append(e)
            except Exception:
                pass  # a flaky availability check hides that tool, never crashes the registry
        return out

    def schemas(self) -> list[dict]:
        return [e.schema for e in self._available()]

    def accesses(self, name: str, args: dict) -> list:
        e = self._tools.get(name)
        if e is None:
            return [AllAccess()]
        try:
            return e.accesses(args)
        except Exception:
            return [AllAccess()]

    def run(self, name: str, args: dict) -> str:
        e = self._tools.get(name)
        if e is None:
            return f'Error: unknown tool "{name}"'
        try:
            return str(e.handler(args))
        except Exception as ex:  # errors come back as strings so the model can react
            return f"Error: {ex}"
