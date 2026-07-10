"""Typed execution contracts shared by the loop, registry, and scheduler.

The provider wire format and the legacy event/result dictionaries are projections of
these values.  In particular, tool status is data: output wording never decides whether
an invocation succeeded once it has crossed the typed registry boundary.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterator

from .context_overflow import ContextOverflow


# Private scheduler→delegation metadata. It is never part of the provider invocation,
# policy decision, journal, or public tool schema; SubagentHost consumes it before
# validating the child's public arguments.
CHILD_TOKEN_BUDGET_ARG = "__sliceagent_token_budget"


class ToolStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INDETERMINATE = "indeterminate"


class ToolPurity(str, Enum):
    """Ordering class, deliberately separate from deduplication/idempotence."""

    PURE_READ = "pure_read"
    EFFECTFUL = "effectful"
    UNKNOWN = "unknown"


def _scope_path(value: object) -> str:
    path = str(value or "").replace("\\", "/").strip()
    if not path:
        return "."
    normalized = posixpath.normpath(path)
    return normalized[2:] if normalized.startswith("./") else normalized


def reconciliation_targets(name: str, args: Mapping[str, object] | None) -> tuple[str, ...]:
    """Conservative affected-resource identities for an operation that did not settle conclusively."""
    name = str(name or "")
    args = args if isinstance(args, Mapping) else {}
    if name.startswith("mcp__"):
        # MCP methods have no trustworthy common effect schema. A nominal database/network method may also
        # write files through its server process (or vice versa), so an interrupted call must retain every
        # relevant boundary until the user and live workspace state have both been re-observed.
        return ("workspace:*", f"opaque:{name}", f"external:{name}")
    if name in {"fetch_url", "web_search", "ask_user"}:
        return (f"external:{name}",)
    if name in {"proc_poll", "proc_tail", "proc_wait", "proc_kill"} and args.get("handle") is not None:
        return (f"process:{args.get('handle')}",)
    if name in {"terminal_send", "terminal_read", "terminal_wait", "terminal_close"}:
        return (f"terminal:{args.get('session') or 'main'}",)
    if name in {"read_file", "list_files", "edit_file", "append_to_file", "str_replace", "grep", "glob"}:
        return (f"path:{_scope_path(args.get('path') or '.')}",)
    # A shell/code/plugin/child call without a declared resource can affect both local and non-local state.
    # Tool semantics deliberately precede incidental argument names: run_command(..., path="README.md") and
    # an unknown plugin with a `path` field remain opaque. Do not let provider-added keys narrow real reach.
    return ("workspace:*", f"opaque:{name or 'unknown'}")


def observation_targets(
    name: str,
    args: Mapping[str, object] | None,
    output: object = "",
) -> tuple[str, ...]:
    """Resource identities conclusively observed by one successful read-only call.

    A typed ``SUCCEEDED`` status says the observation tool itself ran; it does not say the affected operation
    has settled.  This function therefore checks the observation's semantics too (complete file bytes,
    exited processes/sessions, an affirmative human confirmation, and the code-review inventory marker).
    """
    name = str(name or "")
    args = args if isinstance(args, Mapping) else {}
    text = str(output or "")
    folded = text.casefold()
    if name == "code_review":
        # Emitted only after tracked/untracked status, recursive ignored-file inventory, and the safe diff.
        # The handler observes the complete status/diff before presentation paging, so a large inline view
        # remains conclusive even though the model may page its durable detail blob back for analysis.
        if folded.startswith("[workspace observation: tracked + untracked + ignored inventory complete]"):
            return ("workspace:*",)
        return ()
    if name == "ask_user":
        if not folded.startswith("user answered:"):
            return ()
        answer = folded.split(":", 1)[1].strip()
        # Negative, ongoing, or hedged answers are evidence that uncertainty remains, not confirmation.
        if (not answer or re.search(
                r"\b(no|not|never mind|unsure|uncertain|unknown|maybe|perhaps|running|alive|pending|"
                r"in[ -]?progress|still|don't know|do not know|can't tell|cannot tell)\b", answer)):
            return ()
        if re.search(r"\b(yes|confirmed|settled|finished|stopped|exited|completed|complete|done|failed)\b", answer):
            return ("external:*", "opaque:*")
        return ()
    if name == "read_file":
        # The footer reports exact line coverage for both implicit capping and explicit windows. A caller can
        # deliberately request offset=1 with a large limit to prove a large file in one complete observation;
        # partial pages never count. Binary previews are hexdump heads, not complete bytes.
        if "hexdump (first " in folded:
            return ()
        footer = re.search(r"<system>read_file .*?: lines (\d+)-(\d+) of (\d+)", text)
        if footer is not None:
            start, end, total = (int(value) for value in footer.groups())
            if start != 1 or end != total:
                return ()
        elif args.get("offset") is not None or args.get("limit") is not None:
            return ()
        return (f"path:{_scope_path(args.get('path'))}",)
    if name in {"list_files", "grep", "glob"}:
        return (f"tree:{_scope_path(args.get('path') or '.')}",)
    if name in {"proc_poll", "proc_tail", "proc_wait"} and args.get("handle") is not None:
        first_line = folded.splitlines()[0].strip() if folded.splitlines() else ""
        handle = re.escape(str(args.get("handle")))
        settled = (bool(re.fullmatch(r"exited\s+-?\d+", first_line)) if name == "proc_poll" else
                   bool(re.match(rf"^\[{handle}\s+exited\s+-?\d+\]$", first_line)))
        if settled:
            return (f"process:{args.get('handle')}",)
        return ()
    if name in {"terminal_read", "terminal_wait"}:
        first_line = folded.splitlines()[0].strip() if folded.splitlines() else ""
        if ((name == "terminal_read" and re.fullmatch(r"\(no output; exited\s+-?\d+\)", first_line))
                or (name == "terminal_wait" and re.match(
                    r"^\[[^\]\n]+;\s*exited\s+-?\d+\]$", first_line))):
            return (f"terminal:{args.get('session') or 'main'}",)
        return ()
    return ()


def reconciliation_covered(required: tuple[str, ...], observed: set[str]) -> bool:
    """True only when every uncertain resource has a matching live observation."""
    if not required:
        required = ("workspace:*",)
    for target in required:
        if target in observed:
            continue
        if target.startswith("external:") and "external:*" in observed:
            continue
        if target.startswith("opaque:") and "opaque:*" in observed:
            continue
        if target.startswith("path:"):
            # A tree listing proves only presence, not the bytes an uncertain edit may have written.
            # Exact read_file evidence or a workspace-wide code review is required.
            if "workspace:*" in observed:
                continue
        return False
    return True


@dataclass(frozen=True)
class ToolInvocation:
    id: str
    name: str
    args: Mapping[str, object]
    provider_index: int


@dataclass(frozen=True)
class ToolEffect:
    id: str
    kind: str
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutcome:
    invocation: ToolInvocation
    status: ToolStatus
    text: str
    effects: tuple[ToolEffect, ...] = ()

    @property
    def failing(self) -> bool:
        return self.status is not ToolStatus.SUCCEEDED

    def with_text(self, text: object) -> "ToolOutcome":
        """Change presentation only; status/effects remain authoritative."""
        return replace(self, text="" if text is None else str(text))

    def as_legacy(self) -> dict:
        """Compatibility projection consumed by the existing provider/event path."""
        return {
            "id": self.invocation.id,
            "name": self.invocation.name,
            "args": dict(self.invocation.args),
            "output": self.text,
            "failing": self.failing,
            "status": self.status.value,
            "outcome": self,
        }


class TurnStatus(str, Enum):
    END_TURN = "end_turn"
    ABORTED = "aborted"
    MAX_STEPS = "max_steps"
    TOKEN_BUDGET = "token_budget"
    BLOCKED = "blocked"
    STUCK = "stuck"
    OVERFLOW = "overflow"
    MAX_TOKENS = "max_tokens"
    FILTERED = "filtered"
    ERROR = "error"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class Usage(Mapping[str, int | float]):
    """Provider-neutral token accounting with a legacy mapping view."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_other: int = 0
    input_cache_read: int = 0
    input_cache_creation: int = 0
    output: int = 0
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        typed_input = self.input_other + self.input_cache_read + self.input_cache_creation
        if self.prompt_tokens == 0 and typed_input:
            object.__setattr__(self, "prompt_tokens", typed_input)
        if self.completion_tokens == 0 and self.output:
            object.__setattr__(self, "completion_tokens", self.output)
        if self.output == 0 and self.completion_tokens:
            object.__setattr__(self, "output", self.completion_tokens)

    @classmethod
    def from_value(cls, value: "Usage | Mapping[str, object] | None") -> "Usage":
        if isinstance(value, cls):
            return value
        data = value or {}

        def integer(key: str) -> int:
            try:
                return int(data.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        cost = data.get("cost_usd")
        return cls(
            prompt_tokens=integer("prompt_tokens"),
            completion_tokens=integer("completion_tokens"),
            input_other=integer("input_other"),
            input_cache_read=integer("input_cache_read"),
            input_cache_creation=integer("input_cache_creation"),
            output=integer("output"),
            cost_usd=float(cost) if isinstance(cost, (int, float)) and cost >= 0 else None,
        )

    def as_dict(self) -> dict[str, int | float]:
        out: dict[str, int | float] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "input_other": self.input_other,
            "input_cache_read": self.input_cache_read,
            "input_cache_creation": self.input_cache_creation,
            "output": self.output,
        }
        if self.input_cache_read:
            out["cached_tokens"] = self.input_cache_read
        if self.cost_usd is not None:
            out["cost_usd"] = self.cost_usd
        return out

    def __getitem__(self, key: str) -> int | float:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def __add__(self, other: "Usage | Mapping[str, object]") -> "Usage":
        rhs = Usage.from_value(other)
        cost = None if self.cost_usd is None and rhs.cost_usd is None else (self.cost_usd or 0) + (rhs.cost_usd or 0)
        return Usage(
            prompt_tokens=self.prompt_tokens + rhs.prompt_tokens,
            completion_tokens=self.completion_tokens + rhs.completion_tokens,
            input_other=self.input_other + rhs.input_other,
            input_cache_read=self.input_cache_read + rhs.input_cache_read,
            input_cache_creation=self.input_cache_creation + rhs.input_cache_creation,
            output=self.output + rhs.output,
            cost_usd=cost,
        )


@dataclass(frozen=True)
class TurnOutcome:
    status: TurnStatus | str
    steps: int
    usage: Usage | Mapping[str, object]
    message: str | None = None

    def __post_init__(self) -> None:
        try:
            status = self.status if isinstance(self.status, TurnStatus) else TurnStatus(str(self.status))
        except ValueError:
            status = str(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "usage", Usage.from_value(self.usage))

    @property
    def stop_reason(self) -> str:
        return self.status.value if isinstance(self.status, TurnStatus) else str(self.status)


@dataclass(frozen=True)
class PreflightReport:
    estimated_input_tokens: int
    schema_tokens: int
    output_reserve: int
    context_window: int
    mode: str                         # strict | compatibility-unknown

    @property
    def required_tokens(self) -> int:
        return self.estimated_input_tokens + self.schema_tokens + self.output_reserve


class PreflightOverflow(ContextOverflow):
    """A local capacity rejection shaped like provider overflow for existing recovery."""

    def __init__(self, report: PreflightReport):
        self.report = report
        super().__init__(ValueError(
            f"model context window preflight failed: need {report.required_tokens} tokens "
            f"including output reserve, window={report.context_window}"))


class UnknownContextWindow(RuntimeError):
    pass


def _positive_int(value: object) -> int:
    try:
        result = int(value or 0)
        return result if result > 0 else 0
    except (TypeError, ValueError):
        return 0


def _context_window(llm) -> int:
    """Explicit runtime configuration wins; catalog values are used when genuinely known."""
    configured = _positive_int(os.environ.get("AGENT_CONTEXT_WINDOW"))
    if configured:
        return configured
    direct = _positive_int(getattr(llm, "context_window", 0))
    if direct:
        return direct
    try:
        from .model_catalog import capability
        cap = capability(getattr(llm, "model", ""), getattr(llm, "_base_url", ""))
        return _positive_int(cap.context_window)
    except Exception:  # noqa: BLE001 - preflight discovery itself must not break compatibility mode
        return 0


def model_context_window(llm) -> int:
    """Public capacity lookup shared by the seed projector and strict preflight."""
    return _context_window(llm)


def _byte_upper_bound(value: object) -> int:
    """Conservative tokenizer-independent upper bound for text/JSON request material."""
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return len(body.encode("utf-8", "replace"))


def preflight_model_call(
    llm,
    messages: list[dict],
    tools: list[dict],
    *,
    allow_unknown: bool,
    estimator: Callable[[object], int] = _byte_upper_bound,
) -> PreflightReport:
    """Check the exact per-call messages, schemas, and configured output reserve.

    Unknown model windows remain an explicit compatibility mode during migration.  A
    configured or catalogued positive window is always enforced before provider I/O.
    """
    report = estimate_model_call(llm, messages, tools, estimator=estimator)
    window = report.context_window
    try:
        llm._last_preflight = report
    except Exception:  # noqa: BLE001 - diagnostic only
        pass
    if not window:
        if not allow_unknown:
            raise UnknownContextWindow(
                "model context window is unknown; set AGENT_CONTEXT_WINDOW or opt into compatibility mode")
        return report
    if report.required_tokens > window:
        raise PreflightOverflow(report)
    return report


def estimate_model_call(
    llm,
    messages: list[dict],
    tools: list[dict],
    *,
    estimator: Callable[[object], int] = _byte_upper_bound,
) -> PreflightReport:
    """Return the canonical physical-cost estimate without accepting/rejecting the call."""
    message_estimate = estimator(messages) + 12 * len(messages) + 16
    schema_estimate = estimator(tools) + 24 * len(tools) if tools else 0
    output_reserve = _positive_int(getattr(llm, "max_tokens", 0))
    window = _context_window(llm)
    return PreflightReport(
        estimated_input_tokens=message_estimate,
        schema_tokens=schema_estimate,
        output_reserve=output_reserve,
        context_window=window,
        mode="strict" if window else "compatibility-unknown",
    )


def available_content_capacity(llm, fixed_messages: list[dict], tools: list[dict]) -> int | None:
    """Conservative request units left for one user-content projection.

    ``fixed_messages`` should contain the system message, an empty user placeholder, and the current
    trajectory. Unknown windows return ``None`` so compatibility mode preserves the roomy projection.
    Final strict preflight still checks the exact rendered JSON and corrects escaping/Unicode overhead.
    """
    report = estimate_model_call(llm, fixed_messages, tools)
    if not report.context_window:
        return None
    return max(0, report.context_window - report.required_tokens)


def coerce_tool_status(value: object, *, legacy_text: str | None = None) -> ToolStatus:
    """Normalize an explicit status; prose is used only by a named legacy adapter."""
    if isinstance(value, ToolStatus):
        return value
    if isinstance(value, str):
        try:
            return ToolStatus(value.lower())
        except ValueError:
            pass
    if isinstance(value, bool):
        return ToolStatus.SUCCEEDED if value else ToolStatus.FAILED
    if legacy_text is not None:
        return (ToolStatus.FAILED if legacy_text.startswith(("Error", "Exit code"))
                else ToolStatus.SUCCEEDED)
    return ToolStatus.SUCCEEDED


__all__ = [
    "PreflightOverflow", "PreflightReport", "ToolEffect", "ToolInvocation", "ToolOutcome",
    "ToolPurity", "ToolStatus", "TurnOutcome", "TurnStatus", "UnknownContextWindow", "Usage",
    "available_content_capacity", "coerce_tool_status", "estimate_model_call", "model_context_window",
    "preflight_model_call",
]
