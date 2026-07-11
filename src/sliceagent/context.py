"""Canonical context planning and elasticity primitives.

The Active Slice is logical task state.  This module owns the separate, provider-facing
representation decision: regions offer graded alternatives for one semantic item and the
controller selects at most one alternative per item under the next request's capacity.

The controller deliberately works in characters.  Token estimation belongs to the model runner,
which can translate a provider window into a conservative character budget.  Keeping selection
independent from any tokenizer makes the state/representation boundary deterministic and easy to
test.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable


class InstructionClass(str, Enum):
    """Whether text can direct behavior, independently from factual freshness."""

    SYSTEM = "system"
    USER = "user"
    TASK_STATE = "task_state"
    DATA = "data"


class FreshnessClass(str, Enum):
    """Epistemic freshness, independently from instruction authority and residency value."""

    LIVE = "live"
    REVISION_BOUND = "revision_bound"
    DERIVED = "derived"
    HISTORICAL = "historical"


class EpistemicRole(str, Enum):
    """What a block can establish; independent from freshness and instruction authority."""

    DIRECTIVE = "directive"
    OBSERVATION = "observation"
    CLAIM = "claim"
    PROCEDURE = "procedure"
    CONTROL_STATE = "control_state"
    LOCATOR = "locator"


class ResourceKind(str, Enum):
    """Host resource namespaces must not collapse into ordinary workspace paths."""

    WORKSPACE_FILE = "workspace_file"
    ARTIFACT = "artifact"
    HISTORY = "history"
    SUBAGENT = "subagent"
    ROSTER = "roster"
    SKILL = "skill"


_VIRTUAL_MOUNTS = {
    "artifacts": ResourceKind.ARTIFACT,
    "history": ResourceKind.HISTORY,
    "subagents": ResourceKind.SUBAGENT,
    "roster": ResourceKind.ROSTER,
}


@dataclass(frozen=True)
class ResourceRef:
    kind: ResourceKind
    handle: str

    @property
    def virtual(self) -> bool:
        return self.kind is not ResourceKind.WORKSPACE_FILE


@dataclass(frozen=True)
class SourceRef:
    """A compact provenance pointer. Content remains in its owning store."""

    kind: str
    handle: str
    revision: str = ""

    def __post_init__(self) -> None:
        if not self.kind or not self.handle:
            raise ValueError("source reference kind and handle must be non-empty")


def reserved_resource_ref(path: str) -> ResourceRef:
    """Classify a model-visible handle without touching the filesystem.

    The live host may override this classification when a real project path shadows a reserved mount.
    """
    normalized = str(path or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.rstrip("/")
    mount = normalized.split("/", 1)[0] if normalized else ""
    return ResourceRef(_VIRTUAL_MOUNTS.get(mount, ResourceKind.WORKSPACE_FILE), normalized or ".")


class Fidelity(str, Enum):
    FULL = "full"
    EXCERPT = "excerpt"
    DIGEST = "digest"
    LOCATOR = "locator"


class RepresentationLoss(str, Enum):
    NONE = "none"
    SELECTION = "selection"
    SUMMARY = "summary"
    POINTER_ONLY = "pointer_only"


class PressureLevel(str, Enum):
    ROOMY = "roomy"
    ELEVATED = "elevated"
    TIGHT = "tight"
    CRITICAL = "critical"
    UNFIT = "unfit"


_FIDELITY_RANK = {
    Fidelity.FULL: 4,
    Fidelity.EXCERPT: 3,
    Fidelity.DIGEST: 2,
    Fidelity.LOCATOR: 1,
}


@dataclass(frozen=True)
class ContextBlock:
    """One representation alternative for one semantic context item.

    ``item_id`` is semantic identity. ``alternative_group`` groups mutually exclusive
    representations of that item. An incomplete representation is legal only when omitted detail
    is already durable behind ``handles`` or deterministically re-observable.
    """

    block_id: str
    item_id: str
    alternative_group: str
    priority: int
    instruction_class: InstructionClass
    freshness: FreshnessClass
    fidelity: Fidelity
    representation_loss: RepresentationLoss
    content: str
    handles: tuple[str, ...] = ()
    mandatory: bool = False
    reobservable: bool = False
    order: int = 0
    slot: int = 0
    epistemic_role: EpistemicRole = EpistemicRole.CLAIM
    scope: tuple[str, ...] = ()
    source_refs: tuple[SourceRef, ...] = ()
    resource_refs: tuple[ResourceRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.block_id or not self.item_id or not self.alternative_group:
            raise ValueError("context block identity fields must be non-empty")
        if self.representation_loss is not RepresentationLoss.NONE and not (self.handles or self.reobservable):
            raise ValueError(
                f"incomplete context block {self.block_id!r} has no recovery handle or re-observation path"
            )
        if self.mandatory and self.representation_loss is not RepresentationLoss.NONE:
            raise ValueError("mandatory meaning cannot be represented by a lossy alternative")
        if any(not str(scope).strip() for scope in self.scope):
            raise ValueError("context block scopes must be non-empty")


@dataclass(frozen=True)
class ContextSelection:
    blocks: tuple[ContextBlock, ...]
    pressure: PressureLevel
    used_chars: int
    capacity_chars: int | None

    def by_slot(self) -> dict[int, tuple[ContextBlock, ...]]:
        slots: dict[int, list[ContextBlock]] = {}
        for block in self.blocks:
            slots.setdefault(block.slot, []).append(block)
        return {slot: tuple(sorted(items, key=lambda b: (b.order, b.block_id)))
                for slot, items in slots.items()}


class ContextUnfitError(ValueError):
    """Mandatory meaning cannot fit in the requested physical capacity."""

    def __init__(self, required_chars: int, capacity_chars: int, mandatory_items: tuple[str, ...]):
        self.required_chars = required_chars
        self.capacity_chars = capacity_chars
        self.mandatory_items = mandatory_items
        super().__init__(
            f"mandatory context needs {required_chars} chars but capacity is {capacity_chars}; "
            f"items={', '.join(mandatory_items) or '(none)'}"
        )


def _pressure(used: int, capacity: int | None) -> PressureLevel:
    if capacity is None or capacity <= 0:
        return PressureLevel.ROOMY
    ratio = used / capacity
    if ratio <= 0.55:
        return PressureLevel.ROOMY
    if ratio <= 0.75:
        return PressureLevel.ELEVATED
    if ratio <= 0.90:
        return PressureLevel.TIGHT
    if ratio <= 1.0:
        return PressureLevel.CRITICAL
    return PressureLevel.UNFIT


class ElasticityController:
    """Select one graded alternative per semantic item under a global capacity.

    Selection begins at the highest fidelity. Under pressure the lowest-priority degradable item
    steps down one representation at a time. This centralizes pressure so individual regions cannot
    privately truncate binding state or consume the whole request window.
    """

    def select(self, blocks: Iterable[ContextBlock], *, capacity_chars: int | None = None) -> ContextSelection:
        groups: dict[str, list[ContextBlock]] = {}
        seen_ids: set[str] = set()
        for block in blocks:
            if block.block_id in seen_ids:
                raise ValueError(f"duplicate context block id {block.block_id!r}")
            seen_ids.add(block.block_id)
            groups.setdefault(block.alternative_group, []).append(block)

        ranked: dict[str, list[ContextBlock]] = {}
        selected_index: dict[str, int] = {}
        for group, alternatives in groups.items():
            item_ids = {b.item_id for b in alternatives}
            if len(item_ids) != 1:
                raise ValueError(f"alternative group {group!r} spans multiple semantic items")
            ordered = sorted(
                alternatives,
                key=lambda b: (_FIDELITY_RANK[b.fidelity],
                               int(b.representation_loss is RepresentationLoss.NONE),
                               len(b.content)),
                reverse=True,
            )
            # Mandatory groups may expose presentation alternatives, but every selectable alternative
            # must preserve exact meaning.
            if any(b.mandatory for b in ordered):
                ordered = [b for b in ordered if b.representation_loss is RepresentationLoss.NONE]
                if not ordered:
                    raise ValueError(f"mandatory group {group!r} has no lossless representation")
            ranked[group] = ordered
            selected_index[group] = 0

        def chosen() -> list[ContextBlock]:
            return [ranked[g][selected_index[g]] for g in ranked]

        def size() -> int:
            return sum(len(b.content) for b in chosen())

        if capacity_chars is not None and capacity_chars < 0:
            raise ValueError("capacity_chars must be non-negative or None")

        while capacity_chars is not None and size() > capacity_chars:
            candidates = []
            for group, alternatives in ranked.items():
                i = selected_index[group]
                if i + 1 >= len(alternatives):
                    continue
                cur, nxt = alternatives[i], alternatives[i + 1]
                savings = len(cur.content) - len(nxt.content)
                if savings <= 0:
                    continue
                candidates.append((cur.priority, -savings, cur.order, group))
            if not candidates:
                picked = chosen()
                mandatory = tuple(sorted({b.item_id for b in picked if b.mandatory}))
                raise ContextUnfitError(size(), capacity_chars, mandatory)
            _, _, _, group = min(candidates)
            selected_index[group] += 1

        picked = tuple(sorted(chosen(), key=lambda b: (b.order, b.block_id)))
        used = sum(len(b.content) for b in picked)
        return ContextSelection(picked, _pressure(used, capacity_chars), used, capacity_chars)


class SeedPlan(list):
    """A list-compatible seed plus its graded, re-renderable context plan.

    Existing callers can keep treating the result of ``build_slice()`` as ``[system, user]``. The model
    runner recognizes this richer value and projects it again before every provider call as trajectory
    pressure changes. The logical slice is not rebuilt; only its physical representation changes.
    """

    def __init__(
        self,
        *,
        system: str,
        blocks: Iterable[ContextBlock],
        render_blocks: Callable[[ContextSelection], str],
        request_block: str,
        now_block: str,
        media_parts: Iterable[dict] = (),
        controller: ElasticityController | None = None,
    ):
        self.system = str(system)
        self.blocks = tuple(blocks)
        self.render_blocks = render_blocks
        self.request_block = str(request_block)
        self.now_block = str(now_block)
        self.media_parts = tuple(dict(part) for part in media_parts)
        self.controller = controller or ElasticityController()
        self.last_selection: ContextSelection | None = None
        self.last_request_copies = 1
        list.__init__(self, self.project())

    def _fixed_user_chars(self, copies: int = 1) -> int:
        """Physical envelope cost for the one exact recency request presentation.

        ``copies=2`` remains accepted for older/custom plans, but new projections deliberately use one copy.
        Repeating a leading premise at both ends gives it accidental evidentiary weight. The request itself is
        never shortened or summarized.
        """
        if copies not in (1, 2):
            raise ValueError("request copies must be one or two")
        primacy = self.request_block if copies == 2 else ""
        return len(primacy + "<context>\n" + "\n</context>\n\n"
                   + self.request_block + self.now_block)

    def project(self, capacity_chars: int | None = None) -> list[dict]:
        if capacity_chars is not None and capacity_chars < 0:
            raise ValueError("capacity_chars must be non-negative or None")
        body_capacity = None
        copies = 1
        if capacity_chars is not None:
            fixed = self._fixed_user_chars(copies)
            if fixed > capacity_chars:
                raise ContextUnfitError(
                    fixed, capacity_chars,
                    ("current_request",) if self.request_block.strip() else ("request_envelope",),
                )
            body_capacity = capacity_chars - fixed
        selection = self.controller.select(self.blocks, capacity_chars=body_capacity)
        self.last_selection = selection
        self.last_request_copies = copies
        body = self.render_blocks(selection)
        primacy = self.request_block if copies == 2 else ""
        user_text = (
            f"{primacy}<context>\n{body}\n</context>\n\n"
            f"{self.request_block}{self.now_block}"
        )
        content: str | list[dict]
        if self.media_parts:
            content = [{"type": "text", "text": user_text}, *[dict(part) for part in self.media_parts]]
        else:
            content = user_text
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": content},
        ]

    def next_tighter_capacity(self) -> int | None:
        """Capacity that forces at least one fidelity step below the current selection.

        Used only after an unknown-window provider reports a real overflow. It derives pressure from the
        representation actually sent rather than inventing a model context size.
        """
        if self.last_selection is None:
            self.project()
        used = int(getattr(self.last_selection, "used_chars", 0) or 0)
        return self._fixed_user_chars(1) + used - 1 if used > 0 else None


__all__ = [
    "ContextBlock", "ContextSelection", "ContextUnfitError", "ElasticityController", "Fidelity",
    "FreshnessClass", "InstructionClass", "PressureLevel", "RepresentationLoss", "SeedPlan",
]
