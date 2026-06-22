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


class ToolText(str):
    """A tool result that carries an EXPLICIT success flag (.ok). It IS a str — every existing caller
    that concatenates / slices / .startswith() keeps working — but the loop reads `.ok` instead of
    re-inferring failure from prose (`startswith("Error")`), which false-flagged legitimate output that
    merely begins with "Error"/"Exit code" (a grep hit, a log line, a docstring). A handler that fails
    WITHOUT raising returns ToolText(msg, ok=False); the registry sets ok=True for any normal return and
    ok=False for a raised exception. See run()."""
    __slots__ = ("_ok",)

    def __new__(cls, value: str = "", ok: bool = True):
        obj = super().__new__(cls, value)
        obj._ok = ok  # type: ignore[attr-defined]
        return obj

    @property
    def ok(self) -> bool:
        return getattr(self, "_ok", True)


def _all_access(_args: dict) -> list:
    return [AllAccess()]


def _missing_required(schema: dict, args: dict) -> list:
    """Required parameters the tool schema declares that are absent (or None) in the call. Present-but-
    empty (e.g. content="") counts as supplied; only truly-missing args are flagged."""
    params = (schema.get("function") or {}).get("parameters") or {}
    a = args or {}
    return [r for r in (params.get("required") or []) if a.get(r) is None]


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

    def run(self, name: str, args: dict) -> ToolText:
        """The single tool choke point. Returns ToolText (a str carrying .ok) so the loop reads an
        EXPLICIT success flag rather than re-inferring failure from prose. ok=False ⟺ a genuine failure:
        an unknown tool, a raised handler, or a handler that returned ToolText(ok=False) itself (e.g. a
        nonzero exit code, a not-unique str_replace). A handler that returns a plain string is SUCCESS —
        even if that string happens to begin with "Error" (a grep hit, a log line)."""
        e = self._tools.get(name)
        if e is None:
            return ToolText(f'Error: unknown tool "{name}"', ok=False)
        # Validate the call against the tool's declared required args (Kimi AJV-style) — a clear
        # "missing required argument" lets a no-transcript model self-correct, vs an opaque KeyError.
        missing = _missing_required(e.schema, args)
        if missing:
            return ToolText(f'Error: {name} missing required argument(s): {", ".join(missing)}', ok=False)
        try:
            out = e.handler(args)
        except Exception as ex:  # a raised handler is a genuine failure → ok=False, surfaced for the model
            return ToolText(f"Error: {ex}", ok=False)
        if isinstance(out, ToolText):
            return out  # handler already declared ok/not-ok (e.g. a nonzero exit code)
        return ToolText("" if out is None else str(out), ok=True)  # normal return = success
