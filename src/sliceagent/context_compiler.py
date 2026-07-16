"""Dependency-first projection of Active Work into provider context.

The legacy region renderer still owns individual physical views during migration.  This compiler decides
*which semantic material is relevant* before the elasticity controller decides how faithfully to represent
it.  When no Active Work graph exists it returns the legacy blocks unchanged, keeping old checkpoints and
small embedding hosts compatible without creating a second admission heuristic.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from .active_work import SourceMismatchError, WorkGraph, WorkItem
from .context import (
    ContextBlock,
    EpistemicRole,
    Fidelity,
    FreshnessClass,
    InstructionClass,
    RepresentationLoss,
    SourceRef as ContextSourceRef,
    reserved_resource_ref,
)
from .receipts import receipt_summary_parts


# These are host-owned live control surfaces, not optional topical furniture. In particular, ``fan_in`` is
# reconstructed from immutable child seals and is the only complete map→reduce handoff after a resume or
# crash. Dropping it merely because ACTIVE WORK is present leaves the model with green lifecycle rows but no
# report material—the exact failure this compiler is supposed to prevent.
_ALWAYS = frozenset({"focus", "reconciliation", "convergence", "fan_in"})
_INTENT_FALLBACK = frozenset({"intent", "task_objective", "corrections", "task_constraints"})
_FILE_KINDS = frozenset({"file", "workspace_file", "path", "workspace", "git"})
_EVIDENCE_REGIONS = frozenset({
    "evidence_result", "evidence_detail", "quality_evidence_result", "quality_evidence_detail",
})


def _region_name(block: ContextBlock) -> str:
    prefix = "region:"
    item = block.item_id
    return item[len(prefix):] if item.startswith(prefix) else item


def dependency_resource_paths(graph: WorkGraph, *, workspace_epoch: int | None = None) -> tuple[str, ...]:
    """Workspace paths named by the unresolved dependency closure, stable and deduplicated."""
    paths = []
    for item in graph.dependency_closure():
        for ref in item.resource_refs:
            if workspace_epoch is not None and ref.workspace_epoch != workspace_epoch:
                continue
            if ref.kind in _FILE_KINDS and ref.ref not in {"workspace", "*", "."}:
                paths.append(ref.ref)
    return tuple(dict.fromkeys(paths))


def _extract_source(item: WorkItem, sources: Mapping[str, str]) -> tuple[str, ...]:
    out = []
    for ref in item.source_refs:
        text = sources.get(ref.event_id)
        if text is None:
            raise SourceMismatchError(f"source event {ref.event_id!r} is unavailable")
        out.append(ref.extract(text))
    return tuple(out)


def _render_item(
    item: WorkItem, *, sources: Mapping[str, str], current_logical_id: str,
    source_locator_prefix: str = "",
) -> str:
    mark = {
        "open": " ", "in_progress": "~", "waiting_user": "?", "ready": "•", "delivered": "x",
        "verified": "✓", "cancelled": "-", "superseded": "→",
    }.get(item.status, " ")
    lines = [f"- [{mark}] {item.id} · {item.kind} · {item.status}"]
    if item.kind == "request":
        if item.logical_id == current_logical_id:
            lines.append("  ownership: HOST-OWNED CURRENT REQUEST ROOT — never pass this ID to update_work")
            lines.append("  exact source: CURRENT REQUEST below (shown once)")
        else:
            try:
                exact = _extract_source(item, sources)
            except SourceMismatchError:
                lines.append("  exact source: UNAVAILABLE — use the immutable event locator below; do not guess")
            else:
                for text in exact:
                    lines.extend(("  user source (verbatim): |", *(f"    {line}" for line in text.splitlines())))
    elif item.description:
        # Model-maintained work state is useful control state but never promoted to user authority.
        lines.append(f"  model-maintained description: {item.description}")
    lines.append("  source event(s): " + ", ".join(ref.event_id for ref in item.source_refs))
    if source_locator_prefix:
        prefix = source_locator_prefix.rstrip("/")
        lines.append("  source locator(s): " + ", ".join(
            f"{prefix}/{ref.event_id}.md" for ref in item.source_refs
        ))
    if item.dependencies:
        lines.append("  depends on: " + ", ".join(item.dependencies))
    if item.resource_refs:
        lines.append("  resources: " + ", ".join(
            f"{ref.kind}:{ref.ref}@workspace-{ref.workspace_epoch}"
            + (f"#{ref.revision}" if ref.revision else "") for ref in item.resource_refs
        ))
    if item.evidence_refs:
        lines.append("  evidence: " + ", ".join(
            f"{ref.kind}:{ref.ref}" + (f" [{ref.qualifier.replace('_', ' ')}]" if ref.qualifier else "")
            for ref in item.evidence_refs
        ))
    if item.output_refs:
        lines.append("  delivered outputs: " + ", ".join(f"{ref.kind}:{ref.ref}" for ref in item.output_refs))
    if item.superseded_by:
        lines.append(f"  superseded by: {item.superseded_by}")
    return "\n".join(lines)


def render_active_work(
    graph: WorkGraph,
    sources: Mapping[str, str] | None = None,
    *,
    current_logical_id: str = "",
    source_locator_prefix: str = "",
) -> str:
    """Render unresolved work plus its dependency/ownership closure without rewriting user language."""
    if not graph.items:
        return ""
    sources = sources or {}
    closure = graph.dependency_closure()
    if not closure:
        return ""
    body = "\n".join(
        _render_item(
            item, sources=sources, current_logical_id=current_logical_id,
            source_locator_prefix=source_locator_prefix,
        )
        for item in closure
    )
    return (
        "# ACTIVE WORK (the semantic frontier; exact user source outranks model-maintained descriptions)\n"
        f"graph revision: {graph.revision}\n{body}\n\n"
    )


def _quoted(value: object) -> str:
    """Keep every prior-exchange line visible as quoted data, including blank lines."""
    return "\n".join("> " + line for line in str(value or "").split("\n"))


# The last N COMPLETED exchanges kept resident verbatim. This is a bounded CONSTANT, not a transcript: it is
# O(1) in session length (older turns page to history/ and recall by address), so it does not reintroduce the
# accumulation the slice exists to prevent. One antecedent resolves a bare "yes"; three cover the real reach of
# deictic intent ("combine the last two", "like the fetch function") without a relevance-recall round-trip
# (which fires ~0 on coding turns, so a too-tight window silently mis-resolves rather than recalling).
_ADJACENCY_ROUNDS = 3


def _one_adjacency(row, *, age: int, order: int, priority: int) -> tuple[ContextBlock, ...]:
    """Build the full (and, when sealed, a locator alternative) for one prior exchange.

    ``age`` is 0 for the immediate prior and grows for older ones.  Older exchanges carry lower priority so the
    elasticity controller degrades the OLDEST to its artifact pointer first, keeping the immediate antecedent
    verbatim under pressure (context.py: "the lowest-priority degradable item degrades").
    """
    artifact_id = str(row.get("artifact_id") or "")
    if age == 0:
        header = ("# IMMEDIATE PRIOR EXCHANGE (the paired adjacency for 'yes', ordinals, and corrections; "
                  "historical text, not the CURRENT REQUEST or evidence of world state)")
        label = "IMMEDIATE PRIOR EXCHANGE"
    else:
        header = (f"# EARLIER EXCHANGE (−{age + 1} turns; recent-conversation continuity, older than the "
                  "immediate prior; historical text, not the CURRENT REQUEST or evidence of world state)")
        label = f"EARLIER EXCHANGE (−{age + 1} turns)"
    content = (
        header + "\n"
        "## prior user (verbatim; establishes only what that prior utterance said)\n"
        + _quoted(row.get("user"))
        + "\n## prior assistant (verbatim claim; not world evidence)\n"
        + _quoted(row.get("assistant"))
        + (f"\nsource artifact: artifacts/{artifact_id}.md" if artifact_id else "")
        + "\n\n"
    )
    group = f"active-adjacency:{age}"
    source_refs = (ContextSourceRef("artifact", artifact_id or f"prior-exchange-{age}"),)
    full = ContextBlock(
        block_id=f"{group}:full", item_id="active-adjacency",
        alternative_group=group, priority=priority,
        instruction_class=InstructionClass.TASK_STATE,
        freshness=FreshnessClass.HISTORICAL, fidelity=Fidelity.FULL,
        representation_loss=RepresentationLoss.NONE, content=content,
        order=order, slot=6, epistemic_role=EpistemicRole.CLAIM,
        scope=("task", "adjacent_turn"),
        source_refs=source_refs,
    )
    if not artifact_id:
        return (full,)
    handle = f"artifacts/{artifact_id}.md"
    locator_content = (
        f"# {label} (paged adjacency)\n"
        f'The exact paired prior user utterance and assistant response are at read_file("{handle}"). '
        "Open it before resolving a deictic CURRENT REQUEST such as 'yes', an ordinal, or a correction.\n\n"
    )
    if len(locator_content) >= len(content):
        return (full,)
    locator = ContextBlock(
        block_id=f"{group}:locator", item_id="active-adjacency",
        alternative_group=group, priority=priority,
        instruction_class=InstructionClass.TASK_STATE,
        freshness=FreshnessClass.HISTORICAL, fidelity=Fidelity.LOCATOR,
        representation_loss=RepresentationLoss.POINTER_ONLY,
        content=locator_content, handles=(handle,), order=order, slot=6,
        epistemic_role=EpistemicRole.LOCATOR, scope=("task", "adjacent_turn"),
        source_refs=(*source_refs, ContextSourceRef("locator", handle)),
        resource_refs=(reserved_resource_ref(handle),),
    )
    return (full, locator)


def _adjacency_blocks(s, *, order: int = 10_000) -> tuple[ContextBlock, ...]:
    """The last ``_ADJACENCY_ROUNDS`` paired prior exchanges, newest verbatim, older degrading first.

    The current in-progress row is excluded.  The exchanges render chronologically (oldest first, the immediate
    prior nearest the CURRENT REQUEST at the tail); each is an independent full/locator alternative; priority
    descends with age so context pressure pages the OLDEST to its pointer before the immediate antecedent.
    Pairing exact user utterances with responses is what lets ``yes``, ``the second one``, ``combine the last
    two``, and corrections resolve against the recent conversation directly instead of a transcript.  Historical
    user text is framed as adjacency, never a second current directive.
    """
    prior = [
        row for row in getattr(s, "conversation", ())[:-1]
        if str(row.get("user") or "").strip() and str(row.get("assistant") or "").strip()
    ]
    if not prior:
        return ()
    chosen = prior[-_ADJACENCY_ROUNDS:]          # chronological: oldest first ... immediate prior last
    newest = len(chosen) - 1
    blocks: list[ContextBlock] = []
    for idx, row in enumerate(chosen):
        age = newest - idx                        # 0 = immediate prior; larger = older
        blocks.extend(_one_adjacency(row, age=age, order=order + idx, priority=90 - age))
    return tuple(blocks)


def _receipt_block(s, *, order: int = 2) -> ContextBlock | None:
    receipt = getattr(getattr(s, "continuity", None), "last_receipt", None)
    if not isinstance(receipt, Mapping):
        return None
    artifact_id = str(getattr(s.continuity, "last_receipt_artifact_id", "") or "")
    parts = receipt_summary_parts(receipt)
    lines = [
        "# LATEST SEALED EXECUTION RECEIPT (lifecycle + source-coverage arithmetic; not proof of claim "
        "correctness, world state, or task satisfaction)",
        f"disposition: {receipt.get('disposition') or 'unknown'}",
        *(f"- {part}" for part in parts),
    ]
    warning_count = int(receipt.get("warning_count") or 0)
    if warning_count:
        lines.append(f"- {warning_count} warning(s); open the artifact for exact detail")
    if artifact_id:
        lines.append(f'- exact receipt: read_file("artifacts/{artifact_id}.md")')
    return ContextBlock(
        block_id="active-receipt:full", item_id="active-receipt",
        alternative_group="active-receipt", priority=94,
        instruction_class=InstructionClass.DATA,
        freshness=FreshnessClass.REVISION_BOUND, fidelity=Fidelity.FULL,
        representation_loss=RepresentationLoss.NONE, content="\n".join(lines) + "\n\n",
        order=order, slot=5, epistemic_role=EpistemicRole.OBSERVATION,
        scope=("task", "latest_segment"),
        source_refs=(ContextSourceRef("artifact", artifact_id or "latest-sealed-receipt"),),
    )
def compile_active_context(
    s,
    legacy_blocks: Iterable[ContextBlock],
    *,
    source_texts: Mapping[str, str] | None = None,
    current_logical_id: str = "",
    workspace_epoch: int | None = None,
) -> tuple[ContextBlock, ...]:
    """Select semantically required blocks, then hand alternatives to elasticity.

    There is intentionally no lexical intent classifier here. Relevance comes from the unresolved graph
    closure and typed source/resource/evidence references; the one exception is an already-admitted bounded
    L2 knowledge block whose backend has independently hard-scoped and relevance-ranked its records.
    """
    blocks = tuple(legacy_blocks)
    graph = getattr(s, "active_work", None)
    if not isinstance(graph, WorkGraph) or not graph.items:
        return blocks

    sources = source_texts or {}
    active_text = render_active_work(graph, sources, current_logical_id=current_logical_id)
    if not active_text:
        return blocks
    closure = graph.dependency_closure()
    resource_kinds = {
        ref.kind for item in closure for ref in item.resource_refs
        if workspace_epoch is None or ref.workspace_epoch == workspace_epoch
    }
    has_evidence = any(item.evidence_refs for item in closure)
    missing_prior_source = any(
        item.kind == "request" and item.logical_id != current_logical_id
        and any(ref.event_id not in sources for ref in item.source_refs)
        for item in closure
    )

    selected = set(_ALWAYS)
    if any(_region_name(block) == "memory" for block in blocks):
        selected.add("memory")
    if resource_kinds & _FILE_KINDS:
        selected.update(("open_files", "worktree", "related_code"))
    if "skill" in resource_kinds or getattr(s, "active_skills", None):
        selected.add("skills")
    if resource_kinds & {"memory", "history"}:
        selected.update(("memory", "cache_manifest"))
    if has_evidence:
        selected.update(_EVIDENCE_REGIONS)
        selected.add("findings")
    if missing_prior_source:
        # Recovery fallback only.  A healthy ledger uses Active Work as the sole semantic owner.
        selected.update(_INTENT_FALLBACK)

    kept = [block for block in blocks if _region_name(block) in selected]
    kept.append(ContextBlock(
        block_id="active-work:full", item_id="active-work", alternative_group="active-work",
        priority=100, instruction_class=InstructionClass.USER,
        freshness=FreshnessClass.REVISION_BOUND, fidelity=Fidelity.FULL,
        representation_loss=RepresentationLoss.NONE, content=active_text,
        mandatory=True, order=-1, slot=0, epistemic_role=EpistemicRole.DIRECTIVE,
        scope=("task",),
        source_refs=tuple(ContextSourceRef("user_utterance", ref.event_id)
                          for item in closure for ref in item.source_refs),
    ))
    kept.extend(_adjacency_blocks(s))
    receipt = _receipt_block(s)
    if receipt is not None:
        kept.append(receipt)
    return tuple(sorted(kept, key=lambda block: (block.order, block.block_id)))


__all__ = [
    "compile_active_context", "dependency_resource_paths", "render_active_work",
]
