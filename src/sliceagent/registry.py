"""ToolRegistry — one registry, many sources (builtin / MCP / plugin / skill).

A `generation` counter plus a `check` availability gate project the three sources into
one registry. The ToolHost projects schemas()/run()/accesses() from here, so every
tool — wherever it comes from — satisfies one contract and appears in one list. This is
the keystone of Step ③: MCP, plugins, and skills all register into the SAME registry the
loop already drives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .access import AllAccess
from .execution import (ToolEffect, ToolInvocation, ToolOutcome, ToolPurity,
                        ToolStatus, coerce_tool_status)

Handler = Callable[[dict], str]      # (args) -> result string
AccessFn = Callable[[dict], list]    # (args) -> list[Access] for the scheduler/permissions


class ToolText(str):
    """A tool result that carries an EXPLICIT success flag (.ok). It IS a str — every existing caller
    that concatenates / slices / .startswith() keeps working — but the loop reads `.ok` instead of
    re-inferring failure from prose (`startswith("Error")`), which false-flagged legitimate output that
    merely begins with "Error"/"Exit code" (a grep hit, a log line, a docstring). A handler that fails
    WITHOUT raising returns ToolText(msg, ok=False); the registry sets ok=True for any normal return and
    ok=False for a raised exception. See run()."""
    __slots__ = ("_status", "_effects")

    def __new__(cls, value: str = "", ok: bool = True, *, status: ToolStatus | str | None = None,
                effects: tuple[ToolEffect, ...] = ()):
        obj = super().__new__(cls, value)
        obj._status = coerce_tool_status(status if status is not None else ok)  # type: ignore[attr-defined]
        obj._effects = tuple(effects or ())  # type: ignore[attr-defined]
        return obj

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.SUCCEEDED

    @property
    def status(self) -> ToolStatus:
        return getattr(self, "_status", ToolStatus.SUCCEEDED)

    @property
    def effects(self) -> tuple[ToolEffect, ...]:
        return getattr(self, "_effects", ())


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
    purity: ToolPurity = ToolPurity.UNKNOWN
    deduplicable: bool = False
    capabilities: frozenset[str] = frozenset()
    effect_factory: Optional[
        Callable[[ToolInvocation, ToolStatus, str], tuple[ToolEffect, ...]]
    ] = None


def tool_result_text(value) -> str:
    """Canonical presentation coercion for handler results.

    Preserve ``ToolText`` as text, keep ``None`` empty, and decode byte results rather than leaking Python's
    ``b'...'`` representation into the model transcript.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def finalize_tool_outcome(
    invocation: ToolInvocation,
    result,
    *,
    entry: ToolEntry | None = None,
    default_effect_id: str | None = None,
) -> ToolOutcome:
    """Build the one canonical typed outcome from a completed/blocked handler result.

    Execution remains host-owned: wrappers such as ``SubagentHost`` must enforce their restrictions before
    this boundary. This function exclusively owns status projection, effect construction, effect-factory
    failure semantics, and the default audit effect used when a tool declares no semantic effects.
    """
    explicit = getattr(result, "status", None)
    if explicit is not None:
        status = coerce_tool_status(explicit)
    else:
        ok = getattr(result, "ok", None)
        status = (coerce_tool_status(bool(ok)) if ok is not None else
                  coerce_tool_status(None, legacy_text=tool_result_text(result)))
    text = tool_result_text(result)
    effects = tuple(getattr(result, "effects", ()) or ())
    factory = getattr(entry, "effect_factory", None)
    if factory is not None:
        try:
            effects = tuple(factory(invocation, status, text) or ())
        except Exception as error:  # tool may have run, but its required semantic effects are now unknown
            status = ToolStatus.INDETERMINATE
            text = f"Error: tool effect construction failed ({type(error).__name__}: {error})"
            effects = ()
    if not effects:
        effect_id = default_effect_id or f"invoke:{invocation.provider_index}:{invocation.id}:0"
        effects = (ToolEffect(
            effect_id, "tool_outcome", {"name": invocation.name, "status": status.value},
        ),)
    return ToolOutcome(invocation=invocation, status=status, text=text, effects=effects)


# Compatibility metadata for built-ins registered before ToolEntry carried execution
# properties. New/plugin/MCP tools remain UNKNOWN unless they declare otherwise.
_PURE_READ_BUILTINS = frozenset({
    "read_file", "list_files", "grep", "glob", "search_history", "code_review",
})
_DEDUPLICABLE_BUILTINS = frozenset({"read_file", "list_files", "grep", "glob", "search_history"})


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
        if entry.source == "builtin" and entry.purity is ToolPurity.UNKNOWN:
            entry.purity = (ToolPurity.PURE_READ if entry.name in _PURE_READ_BUILTINS
                            else ToolPurity.EFFECTFUL)
        if entry.source == "builtin" and entry.name in _DEDUPLICABLE_BUILTINS:
            entry.deduplicable = True
        self._tools[entry.name] = entry
        self.generation += 1

    def deregister(self, name: str) -> None:
        if self._tools.pop(name, None) is not None:
            self.generation += 1

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return [e.name for e in self._available()]

    def entry(self, name: str) -> ToolEntry | None:
        """Canonical metadata lookup. Unknown tools stay conservative in callers."""
        return self._tools.get(name)

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
        EXPLICIT success flag rather than re-inferring failure from prose. ok=False means a non-success:
        an unknown tool, a raised handler (FAILED or INDETERMINATE by contract), or a handler that returned
        ToolText(ok=False) itself (e.g. a nonzero exit code, a not-unique str_replace). A handler that returns
        a plain string is SUCCESS —
        even if that string happens to begin with "Error" (a grep hit, a log line).

        An extension handler may mutate before it raises. For UNKNOWN/EFFECTFUL plugin, MCP, or skill
        entries, a raised exception therefore means INDETERMINATE rather than FAILED; the ordered scheduler
        will stop every later barrier until live reconciliation. A declared PURE_READ extension has no side
        effects to leave unresolved, so its exception remains a normal failure.
        """
        e = self._tools.get(name)
        if e is None:
            return ToolText(f'Error: unknown tool "{name}"', ok=False)
        # Validate the call against the tool's declared required args (JSON-schema-style) — a clear
        # "missing required argument" lets a no-transcript model self-correct, vs an opaque KeyError.
        missing = _missing_required(e.schema, args)
        if missing:
            return ToolText(f'Error: {name} missing required argument(s): {", ".join(missing)}', ok=False)
        try:
            out = e.handler(args)
        except Exception as ex:
            uncertain_extension = (e.source != "builtin" and e.purity is not ToolPurity.PURE_READ)
            status = ToolStatus.INDETERMINATE if uncertain_extension else ToolStatus.FAILED
            suffix = (" (the extension may have applied side effects before raising)"
                      if uncertain_extension else "")
            return ToolText(f"Error: {ex}{suffix}", status=status)
        if isinstance(out, ToolText):
            return out  # handler already declared ok/not-ok (e.g. a nonzero exit code)
        return ToolText("" if out is None else str(out), ok=True)  # normal return = success

    def invoke(self, invocation: ToolInvocation, *, call_args: dict | None = None,
               default_effect_id: str | None = None) -> ToolOutcome:
        """Execute through the registry, then use the canonical typed-outcome boundary.

        ``invocation.args`` remains the raw provider/audit record supplied to effect factories; ``call_args``
        optionally supplies the sanitized handler view. Production wrappers execute themselves and call the
        same :func:`finalize_tool_outcome` helper so wrapper-level restrictions are never bypassed.
        """
        args = dict(invocation.args) if call_args is None else dict(call_args)
        out = self.run(invocation.name, args)
        return finalize_tool_outcome(
            invocation, out, entry=self._tools.get(invocation.name),
            default_effect_id=default_effect_id,
        )
