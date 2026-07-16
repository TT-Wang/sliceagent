from __future__ import annotations

from dataclasses import replace

from sliceagent.active_work import EvidenceRef, OutputRef, ResourceRef, WorkDelta, WorkGraph, WorkItem
from sliceagent.context import (
    ContextBlock,
    EpistemicRole,
    Fidelity,
    FreshnessClass,
    InstructionClass,
    RepresentationLoss,
    ElasticityController,
)
from sliceagent.context_compiler import (
    compile_active_context,
    dependency_resource_paths,
    render_active_work,
)
from sliceagent.pfc import Slice, record_user
from sliceagent.regions import render_context_selection


def block(name: str, content: str | None = None) -> ContextBlock:
    return ContextBlock(
        block_id=f"region:{name}:full", item_id=f"region:{name}",
        alternative_group=f"region:{name}", priority=50,
        instruction_class=InstructionClass.TASK_STATE,
        freshness=FreshnessClass.DERIVED, fidelity=Fidelity.FULL,
        representation_loss=RepresentationLoss.NONE,
        content=content or f"#{name}\n", order=20, slot=3,
        epistemic_role=EpistemicRole.CONTROL_STATE,
    )


def graph_with_current_dependency() -> tuple[WorkGraph, dict[str, str]]:
    graph = WorkGraph().open_request("event-prior", "first exact request", logical_id="prior")
    graph = graph.open_request("event-current", "current exact request", logical_id="current")
    _prior, current = graph.request_roots
    child = WorkItem(
        id="inspect-file", root_id=current.id, source_refs=current.source_refs,
        description="Inspect the implementation", status="in_progress",
        resource_refs=(ResourceRef("workspace_file", "src/app.py", workspace_epoch=1),),
        evidence_refs=(EvidenceRef("tool_receipt", "invocation:read-1"),),
    )
    graph = graph.apply(WorkDelta(expected_revision=2, creates=(child,)))
    return graph, {"event-prior": "first exact request", "event-current": "current exact request"}


def test_active_work_renders_prior_source_but_current_request_only_by_reference():
    graph, sources = graph_with_current_dependency()
    text = render_active_work(graph, sources, current_logical_id="current")
    assert "first exact request" in text
    assert "current exact request" not in text
    assert "CURRENT REQUEST below (shown once)" in text
    assert "HOST-OWNED CURRENT REQUEST ROOT" in text
    assert "never pass this ID to update_work" in text
    assert "model-maintained description: Inspect the implementation" in text
    assert "workspace_file:src/app.py@workspace-1" in text

    mounted = render_active_work(
        graph, sources, current_logical_id="current",
        source_locator_prefix="@sliceagent/evidence/events",
    )
    assert "@sliceagent/evidence/events/event-prior.md" in mounted
    assert "@sliceagent/evidence/events/event-current.md" in mounted


def test_dependency_compiler_removes_global_furniture_before_elasticity():
    graph, sources = graph_with_current_dependency()
    s = Slice(active_work=graph)
    s.conversation = [
        {"user": "old", "assistant": "old assistant", "artifact_id": "old"},
        {"user": "prior", "assistant": "immediate assistant offer", "artifact_id": "prior"},
        {"user": "current", "assistant": "", "artifact_id": "current"},
    ]
    s.continuity.last_receipt = {
        "disposition": "completed_with_warnings", "warning_count": 1,
        "counts": {"requested": 2, "execution_started": 2, "succeeded": 1, "failed": 1},
        "agents": {},
    }
    s.continuity.last_receipt_artifact_id = "turn-prior"
    names = (
        "intent", "task_objective", "corrections", "task_constraints", "open_files",
        "related_code", "skills", "memory", "conversation", "findings", "plan", "world",
        "threads", "cache_manifest", "roster", "action_header", "action_history",
        "evidence_result", "evidence_detail", "focus", "worktree", "reconciliation", "error",
        "convergence",
    )
    compiled = compile_active_context(
        s, [block(name) for name in names], source_texts=sources, current_logical_id="current",
    )
    kept = {item.item_id for item in compiled}
    assert {"active-work", "active-adjacency", "active-receipt", "region:memory"} <= kept
    assert {"region:open_files", "region:related_code", "region:worktree"} <= kept
    assert {"region:evidence_result", "region:evidence_detail", "region:findings"} <= kept
    assert not ({"region:plan", "region:world", "region:threads", "region:roster",
                 "region:conversation", "region:action_history"} & kept)
    adjacency = "\n".join(
        item.content for item in compiled
        if item.item_id == "active-adjacency" and item.fidelity is Fidelity.FULL
    )
    # Both completed priors are within the last _ADJACENCY_ROUNDS, so both are resident (the in-progress
    # "current" row is still excluded); the immediate prior is labelled as the primary antecedent.
    assert "> prior" in adjacency and "immediate assistant offer" in adjacency
    assert "> old" in adjacency and "old assistant" in adjacency
    assert "IMMEDIATE PRIOR EXCHANGE" in adjacency and "EARLIER EXCHANGE" in adjacency
    receipt = next(item.content for item in compiled if item.item_id == "active-receipt")
    assert "operations · 1/2 succeeded" in receipt and "operations · 1 failed" in receipt
    assert 'read_file("artifacts/turn-prior.md")' in receipt
    rendered = render_context_selection(ElasticityController().select(compiled))
    assert rendered.rfind("immediate assistant offer") > rendered.rfind("LATEST SEALED EXECUTION RECEIPT")


def test_recent_paired_adjacencies_are_retained_for_deictic_resolution():
    graph = WorkGraph().open_request("event-current", "yes, the second one", logical_id="current")
    s = Slice(active_work=graph)
    s.conversation = [
        {"user": "earlier question", "assistant": "earlier answer", "artifact_id": "old"},
        {
            "user": "Should we use SQLite or Postgres?",
            "assistant": "1. SQLite for local use\n2. Postgres for shared use",
            "artifact_id": "turn-prior",
        },
        {"user": "yes, the second one", "assistant": "", "artifact_id": "current"},
    ]
    compiled = compile_active_context(
        s, (), source_texts={"event-current": "yes, the second one"},
        current_logical_id="current",
    )
    fulls = sorted(
        (item for item in compiled
         if item.item_id == "active-adjacency" and item.fidelity is Fidelity.FULL),
        key=lambda b: b.order,
    )
    joined = "\n".join(item.content for item in fulls)
    # Both completed exchanges are within the last _ADJACENCY_ROUNDS and stay resident (in-progress excluded).
    assert "> Should we use SQLite or Postgres?" in joined and "> 2. Postgres for shared use" in joined
    assert "> earlier question" in joined and "> earlier answer" in joined
    assert joined.count("## prior user") == joined.count("## prior assistant") == 2
    # Chronological: the older exchange renders first; the immediate prior sits nearest the tail, is labelled
    # the primary antecedent, and carries the higher priority so it survives context pressure last.
    assert "earlier question" in fulls[0].content
    assert "Should we use SQLite" in fulls[-1].content and "IMMEDIATE PRIOR EXCHANGE" in fulls[-1].content
    assert fulls[-1].priority > fulls[0].priority


def test_oldest_adjacency_pages_out_before_the_immediate_prior_under_pressure():
    graph = WorkGraph().open_request("event-current", "use that", logical_id="current")
    s = Slice(active_work=graph)
    s.conversation = [
        {"user": "older q", "assistant": "older " * 2_000, "artifact_id": "turn-older"},
        {"user": "immediate q", "assistant": "immediate " * 2_000, "artifact_id": "turn-immediate"},
        {"user": "use that", "assistant": "", "artifact_id": "current"},
    ]
    compiled = compile_active_context(
        s, (), source_texts={"event-current": "use that"}, current_logical_id="current",
    )

    def alt(group, fidelity):
        return next(b for b in compiled if b.alternative_group == group and b.fidelity is fidelity)

    old_full, old_loc = alt("active-adjacency:1", Fidelity.FULL), alt("active-adjacency:1", Fidelity.LOCATOR)
    roomy = ElasticityController().select(compiled)
    roomy_chars = sum(len(item.content) for item in roomy.blocks)
    # Shave exactly enough to force one adjacency to page out; the OLDEST (lowest priority) goes first.
    selected = ElasticityController().select(
        compiled, capacity_chars=roomy_chars - len(old_full.content) + len(old_loc.content),
    )
    picked = {b.alternative_group: b.fidelity for b in selected.blocks if b.item_id == "active-adjacency"}
    assert picked["active-adjacency:1"] is Fidelity.LOCATOR   # oldest paged to its pointer
    assert picked["active-adjacency:0"] is Fidelity.FULL      # immediate prior stays verbatim


def test_large_paired_adjacency_degrades_to_its_artifact_pointer_under_pressure():
    graph = WorkGraph().open_request("event-current", "use that", logical_id="current")
    s = Slice(active_work=graph)
    s.conversation = [
        {
            "user": "Compare the two designs",
            "assistant": "long response " * 2_000,
            "artifact_id": "turn-prior",
        },
        {"user": "use that", "assistant": "", "artifact_id": "current"},
    ]
    compiled = compile_active_context(
        s, (), source_texts={"event-current": "use that"}, current_logical_id="current",
    )
    alternatives = [item for item in compiled if item.item_id == "active-adjacency"]
    full = next(item for item in alternatives if item.fidelity is Fidelity.FULL)
    locator = next(item for item in alternatives if item.fidelity is Fidelity.LOCATOR)
    roomy = ElasticityController().select(compiled)
    roomy_chars = sum(len(item.content) for item in roomy.blocks)
    selected = ElasticityController().select(
        compiled, capacity_chars=roomy_chars - len(full.content) + len(locator.content),
    )
    adjacency = next(item for item in selected.blocks if item.item_id == "active-adjacency")
    assert adjacency.fidelity is Fidelity.LOCATOR
    assert 'read_file("artifacts/turn-prior.md")' in adjacency.content
    assert "long response" not in adjacency.content


def test_dependency_paths_are_selected_from_the_active_closure_only():
    graph, _sources = graph_with_current_dependency()
    delivered = graph.open_request("event-done", "done", logical_id="done")
    done_root = delivered.request_roots[-1]
    delivered = delivered.transition(
        done_root.id, "delivered",
        output_refs=(OutputRef("response", "done-response"),),
    )
    # A terminal root's stale resource never enters the unresolved closure.
    stale = replace(
        delivered.get(done_root.id),
        resource_refs=(ResourceRef("workspace_file", "stale.py", workspace_epoch=0),),
    )
    delivered = delivered.upsert(stale)
    assert dependency_resource_paths(delivered) == ("src/app.py",)
    assert dependency_resource_paths(delivered, workspace_epoch=0) == ()
    assert dependency_resource_paths(delivered, workspace_epoch=1) == ("src/app.py",)


def test_missing_prior_event_keeps_legacy_exact_intent_as_recovery_fallback():
    graph, sources = graph_with_current_dependency()
    sources.pop("event-prior")
    s = Slice(active_work=graph)
    compiled = compile_active_context(
        s,
        [block("intent"), block("task_objective"), block("world")],
        source_texts=sources,
        current_logical_id="current",
    )
    kept = {item.item_id for item in compiled}
    assert {"region:intent", "region:task_objective"} <= kept
    assert "region:world" not in kept


def test_record_user_opens_graph_only_at_explicit_application_ledger_seam():
    legacy = Slice(); legacy.reset("task")
    record_user(legacy, "hello", source_artifact="local-artifact")
    assert legacy.active_work == WorkGraph()

    active = Slice(); active.reset("task")
    record_user(
        active, "raw token=secret", source_artifact="local-artifact",
        source_event_id="event-1", source_text="raw token=[REDACT]",
        logical_id="logical-1", workspace_epoch=4,
    )
    root = active.active_work.request_roots[0]
    assert root.source_refs[0].extract("raw token=[REDACT]") == "raw token=[REDACT]"
    assert active.intent.current_request == "raw token=secret"
    assert root.workspace_epoch == 4
