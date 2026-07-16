"""Workspace-bound ContextFS configuration.

This module owns the read-only virtual context projections that are staged with one
workspace runtime.  It intentionally has no process-environment or publication logic:
the CLI acquires the workspace lease and loads executable extensions before calling
this function, then atomically publishes the completed resource bundle.
"""
from __future__ import annotations

import json
from collections.abc import Mapping


def configure_workspace_contextfs(
    *,
    base_tools,
    session,
    memory,
    project_identity,
    root: str,
) -> None:
    """Mount canonical context providers and bind their live status projection."""
    # Permanent cognitive address space. It is bound to truthful live projections, not to physical vault
    # paths, and remains present when any optional provider is missing.
    from .context_compiler import render_active_work
    from .contextfs import (ArtifactContextProvider, ArtifactHistoryProvider, CapabilityStatus,
                            LedgerContextProvider, LegacyMountProvider, MappingContextProvider)

    base_tools._contextfs.mount("evidence", MappingContextProvider({
        "index.md": (
            "# CANONICAL EVIDENCE\n"
            "- application events: @sliceagent/evidence/events/\n"
            "- sealed turns: @sliceagent/evidence/turns/\n"
            "- sealed child reports and page-backed observations: @sliceagent/evidence/children/\n"
            "- execution receipts (embedded in their turn seals): @sliceagent/evidence/receipts/\n"
            "These views share immutable source records; a receipt establishes execution lifecycle only."
        ),
    }))
    # The application ledger is bound after the first workspace yields the application session ID. This
    # provider remains stable and faults exact cited events across archived ledgers without listing them.
    base_tools._event_ledger = None
    base_tools._contextfs.mount(
        "evidence/events", LedgerContextProvider(lambda: base_tools._event_ledger),
    )
    base_tools._contextfs.mount(
        "evidence/turns", ArtifactContextProvider(
            base_tools._artifacts, kinds=("turn",),
            canonical_mount="@sliceagent/evidence/turns", title="SEALED TURN EVIDENCE",
        ),
    )
    base_tools._contextfs.mount(
        "evidence/children", ArtifactContextProvider(
            base_tools._artifacts, kinds=("subagent",),
            canonical_mount="@sliceagent/evidence/children", title="SEALED CHILD EVIDENCE",
        ),
    )
    base_tools._contextfs.mount(
        "evidence/receipts", ArtifactContextProvider(
            base_tools._artifacts, kinds=("turn",),
            canonical_mount="@sliceagent/evidence/receipts", title="CANONICAL EXECUTION RECEIPTS",
        ),
    )
    base_tools._contextfs.mount(
        "history", ArtifactHistoryProvider(
            base_tools._artifacts, current_session=session.session_id,
        ),
    )
    if base_tools._roster is not None:
        base_tools._contextfs.mount(
            "roster", LegacyMountProvider(
                base_tools._roster, "roster", canonical_mount="@sliceagent/roster",
            ),
        )

    def _active_state():
        try:
            return session.active() if session.active_id else None
        except Exception:
            return None

    def _work_documents():
        state = _active_state()
        if state is None:
            return {
                "active.md": "# ACTIVE WORK\n(no active request)",
                "plan.md": "# PLAN\n(no active plan)",
                "dependencies.md": "# DEPENDENCIES\n(none)",
                "receipts.md": "# LATEST RECEIPT\n(none)",
            }
        logical = session.logical_turn
        graph = state.active_work
        source_ids = tuple(dict.fromkeys(
            ref.event_id for item in graph.items for ref in item.source_refs
        ))
        sources = {}
        source_error = ""
        ledger = getattr(base_tools, "_event_ledger", None)
        if ledger is not None and source_ids:
            try:
                sources = ledger.resolve_user_sources(source_ids)
            except Exception as exc:  # keep the graph readable while making the evidence gap explicit
                source_error = type(exc).__name__
        active = render_active_work(
            graph, sources,
            current_logical_id=str(getattr(logical, "id", "") or ""),
            source_locator_prefix="@sliceagent/evidence/events",
        ) or "# ACTIVE WORK\n(no open graph items)"
        if source_error:
            active = active.rstrip() + f"\n\n- source resolution: degraded ({source_error})"
        current_request = str(getattr(logical, "request", "") or "")
        if current_request:
            # render_active_work deliberately points the current root at "CURRENT REQUEST below" to avoid
            # duplicating it in the normal seed. This mounted document is a standalone read surface, so
            # carry that exact user source here or the pointer is false when the model opens Active Work.
            active = (
                active.rstrip()
                + "\n\n# CURRENT REQUEST (verbatim user source)\n"
                + current_request
            )
        plan = getattr(state, "plan", None) or []
        plan_lines = ["# PLAN"] + [
            f"- {item.get('status', 'pending')}: {item.get('step', '')}"
            if isinstance(item, dict) else f"- {item}" for item in plan
        ]
        closure = graph.dependency_closure() if getattr(graph, "items", ()) else ()
        dependency_lines = ["# DEPENDENCIES"]
        for item in closure:
            if item.dependencies:
                dependency_lines.append(f"- {item.id}: " + ", ".join(item.dependencies))
            for ref in item.resource_refs:
                dependency_lines.append(
                    f"- {item.id}: {ref.kind}:{ref.ref}@workspace-{ref.workspace_epoch}"
                )
        receipt = getattr(getattr(state, "continuity", None), "last_receipt", None)
        receipt_text = (
            "# LATEST RECEIPT (execution lifecycle only)\n"
            + json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            if isinstance(receipt, dict) else "# LATEST RECEIPT\n(none)"
        )
        return {
            "active.md": active.rstrip(),
            "plan.md": "\n".join(plan_lines) if len(plan_lines) > 1 else "# PLAN\n(no active plan)",
            "dependencies.md": ("\n".join(dependency_lines)
                                 if len(dependency_lines) > 1 else "# DEPENDENCIES\n(none)"),
            "receipts.md": receipt_text,
        }

    base_tools._contextfs.mount("work", MappingContextProvider(_work_documents))

    def _knowledge_documents():
        getter = getattr(memory, "knowledge_records", None)
        # ContextFS is the inspection surface, so include inert candidates/legacy records as labelled
        # lifecycle states. The seed push still queries ACTIVE records only.
        records = getter(include_candidates=True, limit=100) if callable(getter) else []
        docs: dict[str, str] = {}

        def index_for(title: str, selected) -> str:
            lines = [f"# {title}"]
            for record in selected:
                record_title = getattr(memory, "_record_title", lambda value: value.id)(record)
                lines.append(
                    f'- {record.id} · {record.kind.value} · {record.status.value} · {record_title} '
                    f'→ read_file("@sliceagent/memory/records/{record.id}.md")'
                )
            return "\n".join(lines + (["(none)"] if len(lines) == 1 else []))

        docs["index.md"] = index_for("KNOWLEDGE", records)
        docs["user/index.md"] = index_for(
            "USER KNOWLEDGE", [record for record in records if record.scopes.user_id is not None],
        )
        docs["project/index.md"] = index_for(
            "PROJECT KNOWLEDGE", [record for record in records if record.scopes.project_id is not None],
        )
        docs["craft/index.md"] = index_for(
            "CRAFT KNOWLEDGE", [record for record in records if record.scopes.agent_id is not None],
        )
        for record in records:
            lines = [
                f"# {getattr(memory, '_record_title', lambda value: value.id)(record)}",
                f"- id: {record.id}", f"- kind: {record.kind.value}",
                f"- status: {record.status.value}", f"- authority: {record.authority}",
                f"- proof family: {record.proof_family}",
                f"- freshness: {record.freshness.value}",
                f"- user scope: {record.scopes.user_id or '(none)'}",
                f"- project scope: {record.scopes.project_id or '(none)'}",
                f"- agent scope: {record.scopes.agent_id or '(none)'}",
                "", "## Content", record.content, "", "## Canonical sources",
            ]
            lines.extend(
                f"- {ref.namespace}:{ref.record_id} · sha256:{ref.digest[:16]}…"
                for ref in record.source_refs
            )
            if not record.source_refs:
                lines.append("- (no source on file; candidate only, never automatic authority)")
            docs[f"records/{record.id}.md"] = "\n".join(lines)
        return docs

    if callable(getattr(memory, "knowledge_records", None)):
        base_tools._contextfs.mount("memory", MappingContextProvider(_knowledge_documents))

    def _context_status():
        state = _active_state()
        graph = getattr(state, "active_work", None)
        # ``unresolved_roots`` is an immutable WorkGraph projection, not a query method. Calling it
        # made every live manifest fall back to ``status unavailable (TypeError)`` once a task existed.
        open_count = len(graph.unresolved_roots) if graph is not None else 0
        counts_fn = getattr(memory, "knowledge_counts", None)
        counts = {}
        counts_error = ""
        if callable(counts_fn):
            try:
                counts = counts_fn()
            except Exception as exc:
                counts_error = type(exc).__name__
        health_fn = getattr(memory, "knowledge_health", None)
        health_error = ""
        try:
            health = health_fn() if callable(health_fn) else {}
        except Exception as exc:  # diagnostic failure cannot hide L0/L1
            health = {}
            health_error = type(exc).__name__
        # Memory owns its compatibility telemetry.  ContextFS consumes only this bounded public report;
        # it never discovers counts by walking ~/.sliceagent or another private physical store.  Older
        # memory implementations simply produce explicit unknown/not-recorded fields until upgraded.
        memory_status_fn = getattr(memory, "memory_status", None)
        memory_status = {}
        memory_status_error = ""
        if callable(memory_status_fn):
            try:
                reported = memory_status_fn()
                memory_status = reported if isinstance(reported, Mapping) else {}
                if reported is not None and not isinstance(reported, Mapping):
                    memory_status_error = "TypeError"
            except Exception as exc:
                memory_status_error = type(exc).__name__
        inventory_fn = getattr(memory, "memory_inventory", None)
        inventory = {}
        inventory_error = ""
        inventory_reported = False
        if callable(inventory_fn):
            try:
                reported = inventory_fn()
                if isinstance(reported, Mapping):
                    nested = reported.get("legacy_inventory", reported.get("inventory", reported))
                    inventory = nested if isinstance(nested, Mapping) else {}
                    inventory_reported = isinstance(nested, Mapping)
                elif reported is not None:
                    inventory_error = "TypeError"
            except Exception as exc:
                inventory_error = type(exc).__name__
        if not inventory_reported:
            nested = memory_status.get("legacy_inventory", memory_status.get("inventory", {}))
            inventory = nested if isinstance(nested, Mapping) else {}
            inventory_reported = (
                "legacy_inventory" in memory_status or "inventory" in memory_status
            ) and isinstance(nested, Mapping)
            if inventory_reported:
                inventory_error = ""

        inventory_status = memory_status.get("legacy_inventory_status")
        if inventory_status is None:
            inventory_partial = inventory_reported and any(
                value is None for value in inventory.values()
            )
            inventory_status = {
                "state": (
                    "degraded" if inventory_error or inventory_partial else
                    "available" if inventory_reported else
                    "unknown"
                ),
                "detail": (
                    f"inventory unavailable ({inventory_error})" if inventory_error else
                    "host supplied inventory with unavailable aggregates" if inventory_partial else
                        "host-counted aggregate snapshot; values may change during measurement"
                        if inventory_reported else
                    "not reported by host"
                ),
            }

        transition = memory_status.get(
            "compatibility_transition",
            memory_status.get("migration", memory_status.get("migration_state")),
        )
        if transition is None:
            transition = {
                "state": "unknown",
                "detail": (
                    f"status unavailable ({memory_status_error})"
                    if memory_status_error else "not reported by host"
                ),
            }
        compatibility_health = memory_status.get("compatibility_health")
        if compatibility_health is None:
            compatibility_health = {
                "state": "unknown",
                "detail": (
                    f"status unavailable ({memory_status_error})"
                    if memory_status_error else "not reported by host"
                ),
            }
        retirement_gate = memory_status.get("retirement_gate")
        if retirement_gate is None:
            retirement_gate = {
                "state": "unknown",
                "detail": (
                    f"status unavailable ({memory_status_error})"
                    if memory_status_error else "not reported by host"
                ),
            }
        last_consolidation = memory_status.get(
            "last_consolidation", memory_status.get("consolidation"),
        )
        if last_consolidation is None:
            last_consolidation = {
                "state": "unknown",
                "detail": (
                    f"status unavailable ({memory_status_error})"
                    if memory_status_error else "not reported by host"
                ),
            }
        native = health.get("native", {}) if isinstance(health, dict) else {}
        memem = health.get("memem", {}) if isinstance(health, dict) else {}
        has_knowledge = callable(getattr(memory, "knowledge_records", None))
        native_active = bool(native.get("active", native.get("fts5", False)))
        memem_active = bool(memem.get("active", False))
        memem_reported_state = str(memem.get("state", memem.get("status", "")) or "").lower()
        if memem_reported_state not in {
            "available", "healthy", "degraded", "unavailable", "disabled", "unknown",
        }:
            # The current compatibility adapter reports import availability as ``active``. Import success
            # is not evidence that an external semantic backend is connected or healthy.
            memem_reported_state = "available" if memem_active else "disabled"
        memem_context_state = (
            "available" if memem_reported_state == "healthy" else memem_reported_state
        )
        logical = session.logical_turn
        return {
            # The private stable UUID is a scoping key, not useful model context. The exact workspace
            # path remains visible separately, while this field names the human project.
            "current_project": project_identity.label,
            "current_workspace": root,
            "logical_request": getattr(logical, "request", "") or getattr(state, "goal", ""),
            "regions": {
                "evidence": CapabilityStatus("available", "canonical event and artifact views mounted"),
                "history": CapabilityStatus(
                    "available", "canonical artifact-history view mounted",
                ),
                "work": CapabilityStatus("available", "live Active Work projection mounted"),
                "memory": CapabilityStatus(
                    (
                        "degraded" if has_knowledge and (
                            counts_error or health_error or native.get("error")
                        ) else
                        "available" if native_active else
                        "unavailable"
                    ),
                    (
                        f"native knowledge query unavailable "
                        f"({counts_error or health_error or native.get('error')})"
                        if has_knowledge and (
                            counts_error or health_error or native.get("error")
                        ) else
                        "typed native knowledge" if native_active else
                        "no knowledge repository"
                    ),
                ),
                "roster": CapabilityStatus(
                    "available" if base_tools._roster is not None else "unavailable",
                    "standing-agent view mounted" if base_tools._roster is not None else "no roster provider",
                ),
            },
            "open_active_work_count": open_count,
            "knowledge_counts": counts,
            "legacy_inventory": inventory,
            "legacy_inventory_scope": memory_status.get("legacy_inventory_scope"),
            "legacy_inventory_status": inventory_status,
            "compatibility_transition": transition,
            "compatibility_health": compatibility_health,
            "retirement_gate": retirement_gate,
            "last_consolidation": last_consolidation,
            "native_index": {
                "state": (
                    "degraded" if has_knowledge and (
                        counts_error or health_error or native.get("error")
                    ) else
                    "healthy" if native_active else
                    "unavailable"
                ),
                "detail": (
                    f"query unavailable ({counts_error or health_error or native.get('error')})"
                    if has_knowledge and (
                        counts_error or health_error or native.get("error")
                    ) else native.get("backend", "native status unavailable") if has_knowledge
                    else "no knowledge repository"
                ),
            },
            "memem": {
                "state": memem_context_state,
                "detail": (
                    "last scoped operation succeeded; no continuous health probe"
                    if memem_reported_state == "healthy" else
                    str(memem.get("detail") or "")
                    or ("optional adapter available; backend health unverified"
                        if memem_active else "not connected")
                ),
            },
            "cross_project_search_policy": (
                "typed knowledge is stable-project-scoped; canonical history lists the current workspace's "
                "sealed turns; legacy search_history is an explicit compatibility search"
            ),
        }

    base_tools._contextfs.set_status_provider(_context_status)


__all__ = ["configure_workspace_contextfs"]
