"""Typed native L2 records, canonical persistence, scope safety, and retrieval."""
from __future__ import annotations

import dataclasses
import builtins
import json
import os
import sqlite3

import pytest

from sliceagent.active_work import SourceRef as ActiveWorkSourceRef
from sliceagent.knowledge import (
    FeedbackEvent,
    FeedbackKind,
    KnowledgeConflictError,
    KnowledgeKind,
    KnowledgeQuery,
    KnowledgeRecord,
    KnowledgeRepository,
    KnowledgeScope,
    KnowledgeSensitivity,
    KnowledgeSourceRef,
    KnowledgeStatus,
    KnowledgeValidationError,
)
from sliceagent.knowledge_index import KnowledgeIndex, NativeKnowledgeIndex, NullKnowledgeIndex
from sliceagent.interfaces import Memory, TaskState
from sliceagent.memory import LocalMemory


NOW = "2026-07-12T08:00:00Z"


def source(record_id: str = "event-1", text: str = "canonical evidence") -> KnowledgeSourceRef:
    return KnowledgeSourceRef.bind_text(
        "application-events",
        record_id,
        text,
        observer="sliceagent-host",
        observed_at=NOW,
        project_id="hunter",
        workspace_id="hunter-main",
    )


def record(
    record_id: str,
    content: str,
    *,
    user_id: str = "user-1",
    project_id: str | None = "hunter",
    agent_id: str | None = None,
    kind: KnowledgeKind | str = KnowledgeKind.FACT,
    status: KnowledgeStatus | str = KnowledgeStatus.ACTIVE,
    sensitivity: KnowledgeSensitivity | str = KnowledgeSensitivity.PRIVATE,
) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=record_id,
        kind=kind,
        scopes=KnowledgeScope(user_id=user_id, project_id=project_id, agent_id=agent_id),
        content=content,
        applicability="coding work",
        source_refs=(source(f"event-{record_id}", content),),
        authority="observed",
        proof_family="resource_observation",
        created_at=NOW,
        observed_at=NOW,
        freshness="current",
        status=status,
        sensitivity=sensitivity,
        metadata={"tags": ["architecture", "memory"]},
    )


def test_knowledge_source_ref_is_distinct_digest_bound_and_json_round_trips():
    text = "prefix 世界 suffix"
    ref = KnowledgeSourceRef.bind_text(
        "evidence/turns",
        "turn-7",
        text,
        observer="event-ledger",
        observed_at=NOW,
        byte_start=7,
        byte_end=13,
        project_id="hunter",
        resource_revision="git:abc123",
    )

    assert KnowledgeSourceRef is not ActiveWorkSourceRef
    assert len(ref.digest) == 64
    assert KnowledgeSourceRef.from_dict(json.loads(json.dumps(ref.to_dict()))) == ref
    with pytest.raises(KnowledgeValidationError, match="either a byte range or a field"):
        dataclasses.replace(ref, field="payload.text")
    with pytest.raises(KnowledgeValidationError, match="within"):
        KnowledgeSourceRef.bind_text(
            "events", "bad", text, observer="host", observed_at=NOW,
            byte_start=0, byte_end=999,
        )


def test_record_is_typed_immutable_provenance_required_and_wire_stable():
    item = record("fact-1", "Hunter stores exact history separately from learned facts")
    restored = KnowledgeRecord.from_dict(json.loads(json.dumps(item.to_dict())))

    assert restored == item
    assert restored.digest == item.digest
    assert restored.kind is KnowledgeKind.FACT
    assert restored.status is KnowledgeStatus.ACTIVE
    assert restored.metadata["tags"] == ("architecture", "memory")
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.content = "changed"
    with pytest.raises(TypeError):
        item.metadata["new"] = True
    with pytest.raises(KnowledgeValidationError, match="at least one knowledge scope"):
        KnowledgeScope()
    with pytest.raises(KnowledgeValidationError, match="requires at least one source ref"):
        KnowledgeRecord(
            id="unsourced-active",
            kind="fact",
            scopes=KnowledgeScope(project_id="hunter"),
            content="unsupported claim",
            status="active",
        )
    for unsafe_id in ("../escape", "nested/record", "row\n- injected", "[fake](locator)"):
        with pytest.raises(KnowledgeValidationError, match="path-safe ASCII identifier|CR or LF"):
            dataclasses.replace(item, id=unsafe_id)


def test_repository_roundtrip_is_private_and_persistent(tmp_path):
    path = tmp_path / "knowledge" / "knowledge.db"
    item = record("persistent", "Hunter uses a permanent internal context namespace")
    with KnowledgeRepository(str(path)) as repository:
        assert repository.put(item) == item
        assert repository.put(item) == item  # identical upsert is idempotent
        assert repository.get(item.id) == item

    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0
    with KnowledgeRepository(str(path)) as reopened:
        assert reopened.get(item.id) == item
        assert reopened.indexed_count() == 1


def test_repository_runtime_metadata_is_durable_but_not_a_knowledge_record(tmp_path):
    path = tmp_path / "knowledge" / "knowledge.db"
    with KnowledgeRepository(str(path)) as repository:
        repository.set_runtime_metadata(
            "consolidation:project-a",
            {"state": "completed", "lessons": 2},
            updated_at=NOW,
        )
        assert repository.get_runtime_metadata("consolidation:project-a") == {
            "state": "completed", "lessons": 2,
        }
        assert repository.indexed_count() == 0
    with KnowledgeRepository(str(path)) as reopened:
        assert reopened.get_runtime_metadata("consolidation:project-a")["lessons"] == 2


def test_auto_recall_admission_uses_bounded_primary_index_and_cues(tmp_path, monkeypatch):
    """A Memem-ranked hit must survive the host gate that validates the same representation."""
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path / "sliceagent"))
    memory = LocalMemory(prefer_memem=False)
    try:
        memory.set_scope(project_id="hunter", workspace_id="hunter-main")
        item = dataclasses.replace(
            record(
                "cue-relevant",
                "Hunter parser validates request boundaries before decoding.",
                user_id="user-1",
            ),
            metadata={
                "title": "Parser boundary",
                "primary_index": "Parser request boundary",
                "cues": ["bounds validation"],
                "paths": ["src/parser.py"],
            },
        )
        memory._put_knowledge(item)

        assert memory._record_relevant_to_query(item, ["parser", "bounds", "validation"])
    finally:
        memory.close()


def test_repository_axis_counts_are_not_limited_to_one_query_page(tmp_path):
    with KnowledgeRepository(str(tmp_path / "counts.db")) as repository:
        for index in range(101):
            repository.put(record(f"count-{index}", f"fact {index}"))
        counts = repository.count_by_axis(KnowledgeQuery(
            user_id="user-1", project_id="hunter", statuses=(KnowledgeStatus.ACTIVE,),
        ))
        assert counts == {"unique": 101, "user": 101, "project": 101, "craft": 0}
        assert repository.count_by_source_namespace(
            KnowledgeQuery(user_id="user-1", project_id="hunter"), "application-events",
        ) == 101

def test_fts_projection_recovers_records_written_under_fallback(tmp_path):
    path = tmp_path / "knowledge.db"
    item = record("fallback-first", "ContextFS is always addressable")
    with KnowledgeRepository(str(path), use_fts=False) as fallback:
        fallback.put(item)

    with KnowledgeRepository(str(path), use_fts=True) as upgraded:
        hits = upgraded.search(KnowledgeQuery(
            text="ContextFS", user_id="user-1", project_id="hunter",
        ))
        assert [hit.record.id for hit in hits] == [item.id]


@pytest.mark.parametrize("use_fts", [False, True])
def test_search_hard_filters_scope_before_native_or_fallback_ranking(tmp_path, use_fts):
    repository = KnowledgeRepository(str(tmp_path / f"knowledge-{use_fts}.db"), use_fts=use_fts)
    try:
        repository.put(record(
            "user-wide", "Prefer concise implementation notes about memory architecture",
            project_id=None, kind="preference",
        ))
        repository.put(record("hunter", "Hunter memory uses a stable ContextFS namespace"))
        repository.put(record(
            "other-project", "Another project has many repeated memory memory memory terms",
            project_id="another-project",
        ))
        repository.put(record(
            "other-user", "Memory architecture for someone else",
            user_id="user-2",
        ))
        repository.put(record(
            "secret", "Hunter memory credential material",
            sensitivity="secret",
        ))

        hits = repository.search(KnowledgeQuery(
            text="memory architecture ContextFS",
            user_id="user-1",
            project_id="hunter",
            limit=10,
        ))
        ids = {hit.record.id for hit in hits}
        assert "hunter" in ids
        assert "user-wide" in ids
        assert "other-project" not in ids
        assert "other-user" not in ids
        assert "secret" not in ids

        project_records = repository.query(KnowledgeQuery(
            user_id="user-1", project_id="hunter", limit=20,
        ))
        assert {item.id for item in project_records} == {"user-wide", "hunter"}
        assert repository.search(KnowledgeQuery(
            text="ContextFS", user_id="user-1", project_id="another-project",
        )) == []
    finally:
        repository.close()


def test_empty_or_missing_scope_never_turns_into_cross_project_admin_search(tmp_path):
    repository = KnowledgeRepository(str(tmp_path / "knowledge.db"), use_fts=False)
    repository.put(record("hunter", "Hunter local fact", project_id="hunter"))
    repository.put(record("other", "Other local fact", project_id="other"))

    assert repository.query(KnowledgeQuery()) == []
    assert [item.id for item in repository.query(KnowledgeQuery(
        user_id="user-1", project_id="hunter",
    ))] == ["hunter"]


def test_supersession_retraction_and_feedback_have_separate_typed_lifecycles(tmp_path):
    repository = KnowledgeRepository(str(tmp_path / "knowledge.db"), use_fts=False)
    old = record("old", "Hunter uses the old memory route")
    replacement = record("new", "Hunter uses ContextFS for memory routing")
    repository.put(old)
    repository.put(replacement)

    superseded, linked = repository.supersede(old.id, replacement.id)
    assert superseded.status is KnowledgeStatus.SUPERSEDED
    assert old.id in linked.supersedes
    assert repository.replacement_for(old.id) == replacement.id
    assert repository.supersede(old.id, replacement.id) == (superseded, linked)
    with pytest.raises(KnowledgeConflictError, match="must be active"):
        repository.supersede(replacement.id, old.id)

    event = FeedbackEvent(
        id="feedback-1",
        record_id=replacement.id,
        kind=FeedbackKind.SERVED,
        created_at=NOW,
        metadata={"surface": "context-compiler"},
    )
    assert repository.feedback(event) == event
    assert repository.feedback(event) == event
    assert repository.list_feedback(replacement.id) == [event]
    # Merely serving a record neither validates it nor changes its lifecycle/ranking authority.
    assert repository.get(replacement.id).status is KnowledgeStatus.ACTIVE

    retracted = repository.retract(replacement.id)
    assert retracted.status is KnowledgeStatus.RETRACTED
    assert repository.query(KnowledgeQuery(user_id="user-1", project_id="hunter")) == []
    assert repository.query(KnowledgeQuery(
        user_id="user-1", project_id="hunter", statuses=(KnowledgeStatus.RETRACTED,),
    )) == [retracted]


def test_active_meaning_cannot_be_mutated_in_place(tmp_path):
    repository = KnowledgeRepository(str(tmp_path / "knowledge.db"))
    item = record("immutable", "Original observed fact")
    repository.put(item)
    with pytest.raises(KnowledgeConflictError, match="meaning is immutable"):
        repository.put(dataclasses.replace(item, content="Rewritten claim"))
    with pytest.raises(KnowledgeConflictError, match="use supersede"):
        repository.put(dataclasses.replace(item, status=KnowledgeStatus.SUPERSEDED))
    with pytest.raises(KnowledgeConflictError, match="cannot claim supersession"):
        repository.put(dataclasses.replace(
            record("replacement-claim", "Replacement claim"), supersedes=(item.id,),
        ))
    with pytest.raises(KnowledgeConflictError, match="cannot claim supersession"):
        repository.put(dataclasses.replace(
            record("already-retired", "Claim with no replacement"), status=KnowledgeStatus.SUPERSEDED,
        ))


def test_native_index_remove_rebuild_health_and_protocol(tmp_path):
    repository = KnowledgeRepository(str(tmp_path / "knowledge.db"), use_fts=False)
    item = record("indexed", "Stable native lexical recall without Memem")
    repository.put(item)
    index = NativeKnowledgeIndex(repository)

    assert isinstance(index, KnowledgeIndex)
    assert index.is_active
    assert index.health() == {
        "active": True,
        "backend": "sqlite-lexical",
        "fts5": False,
        "indexed_records": 1,
        "error": "",
        "warning": "",
    }
    repository.fts_error = "simulated FTS5 setup failure"
    fallback_health = index.health()
    assert fallback_health["active"] is True
    assert fallback_health["backend"] == "sqlite-lexical"
    assert fallback_health["error"] == ""
    assert fallback_health["warning"] == "simulated FTS5 setup failure"
    query = KnowledgeQuery(text="lexical recall", user_id="user-1", project_id="hunter")
    assert [hit.record.id for hit in index.search(query)] == [item.id]

    index.remove([item.id])
    assert repository.get(item.id) == item  # index removal never removes canonical meaning
    assert index.search(query) == []
    index.rebuild(repository)
    assert [hit.record.id for hit in index.search(query)] == [item.id]
    assert index.health()["indexed_records"] == 1

    null = NullKnowledgeIndex()
    assert isinstance(null, KnowledgeIndex)
    assert not null.is_active
    assert null.search(query) == []


def test_tombstone_removes_content_from_all_search_paths(tmp_path):
    repository = KnowledgeRepository(str(tmp_path / "knowledge.db"), use_fts=False)
    item = record("forget-me", "uniqueterm private learned content")
    repository.put(item)
    tombstone = repository.tombstone(item.id)

    assert tombstone.status is KnowledgeStatus.TOMBSTONED
    assert tombstone.content == "[tombstoned]"
    assert tombstone.source_refs == ()
    assert repository.search(KnowledgeQuery(
        text="uniqueterm", user_id="user-1", project_id="hunter", statuses=(),
    )) == []


def test_local_memory_does_not_append_unprovenanced_legacy_memem_tail(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="project-hunter", workspace_id="hunter-main", label="shared-name")
    try:
        # Legacy Markdown without canonical typed ids/provenance is no longer
        # a second recall tail. First-class Memem hits enter through the
        # KnowledgeIndex protocol and resolve back to canonical records.
        assert memory.recall("memory route", k=5) == []
        assert not hasattr(memory, "_memem_retrieve")
    finally:
        memory.close()


def test_native_seed_push_keeps_user_preferences_and_relevance_filters_project_facts(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="hunter", workspace_id="hunter-main")
    for item in (
        KnowledgeRecord(
            id="user-pref", kind=KnowledgeKind.PREFERENCE,
            scopes=KnowledgeScope(user_id="local-user"), content="Use concise final reports",
            source_refs=(source("user-pref-source", "Use concise final reports"),), status="active",
        ),
        KnowledgeRecord(
            id="parser-fact", kind=KnowledgeKind.FACT,
            scopes=KnowledgeScope(project_id="hunter"), content="The parser entry point is src/parser.py",
            source_refs=(source("parser-source", "The parser entry point is src/parser.py"),), status="active",
        ),
        KnowledgeRecord(
            id="billing-fact", kind=KnowledgeKind.FACT,
            scopes=KnowledgeScope(project_id="hunter"),
            content="The billing entry service uses a separate webhook",
            source_refs=(source("billing-source", "The billing entry service uses a separate webhook"),),
            status="active",
        ),
    ):
        memory.knowledge_repository.put(item)
    try:
        hits = memory.recall("fix the parser entry point", k=6)
        assert [hit.path for hit in hits] == [
            "@sliceagent/memory/records/user-pref.md",
            "@sliceagent/memory/records/parser-fact.md",
        ]
        assert hits[0].text.startswith("[USER knowledge preference")
        assert hits[1].text.startswith("[PROJECT knowledge")
    finally:
        memory.close()


def test_native_seed_relevance_gate_preserves_multilingual_queries(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="hunter")
    for item in (
        KnowledgeRecord(
            id="parser-cn", kind="fact", scopes=KnowledgeScope(project_id="hunter"),
            content="解析器 入口 位于 src/parser.py",
            source_refs=(source("parser-cn-source", "解析器 入口 位于 src/parser.py"),), status="active",
        ),
        KnowledgeRecord(
            id="billing-cn", kind="fact", scopes=KnowledgeScope(project_id="hunter"),
            content="计费 入口 由 webhook 服务处理",
            source_refs=(source("billing-cn-source", "计费 入口 由 webhook 服务处理"),), status="active",
        ),
    ):
        memory.knowledge_repository.put(item)
    try:
        assert [hit.path for hit in memory.seed_recall("修复 解析器 入口")] == [
            "@sliceagent/memory/records/parser-cn.md",
        ]
    finally:
        memory.close()


def test_standing_user_preference_is_not_crowded_out_by_project_fact_page(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="hunter")
    preference = KnowledgeRecord(
        id="standing-pref", kind="preference", scopes=KnowledgeScope(user_id="local-user"),
        content="Use concise final reports",
        source_refs=(source("standing-pref-source", "Use concise final reports"),), status="active",
    )
    memory.knowledge_repository.put(preference)
    for index in range(101):
        fact = f"Unrelated project fact {index}"
        memory.knowledge_repository.put(KnowledgeRecord(
            id=f"seed-fact-{index}", kind="fact", scopes=KnowledgeScope(project_id="hunter"),
            content=fact, source_refs=(source(f"seed-source-{index}", fact),), status="active",
        ))
    try:
        assert memory.seed_recall("no lexical project match", k=2)[0].path == (
            "@sliceagent/memory/records/standing-pref.md"
        )
    finally:
        memory.close()


def test_project_issue_lifecycle_prevents_hunter_report_noise_but_keeps_open_lead_and_explicit_search(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="hunter")
    for index in range(20):
        text = f"Hunter code review parser bug report {index}: fixed after validation."
        memory.knowledge_repository.put(KnowledgeRecord(
            id=f"resolved-report-{index}", kind="fact",
            scopes=KnowledgeScope(project_id="hunter"), content=text,
            applicability="corrective engineering work",
            source_refs=(source(f"resolved-source-{index}", text),),
            freshness="current", status="active",
            # Legacy consolidated reports lack issue_state. Their source arc is
            # failure→fix, so they are explicit-pull evidence, not open issues.
            metadata={"title": f"Hunter parser bug {index}"},
        ))
    open_text = "Hunter parser request boundary remains unresolved in the current validation path."
    memory.knowledge_repository.put(KnowledgeRecord(
        id="open-parser-boundary", kind="fact", scopes=KnowledgeScope(project_id="hunter"),
        content=open_text, applicability="current diagnostic work",
        source_refs=(source("open-parser-source", open_text),), freshness="current", status="active",
        metadata={
            "title": "Hunter parser request boundary",
            "memory_role": "diagnostic_issue", "issue_state": "open",
        },
    ))
    stale_text = "Hunter parser stale diagnostic remains unresolved."
    memory.knowledge_repository.put(KnowledgeRecord(
        id="stale-parser", kind="fact", scopes=KnowledgeScope(project_id="hunter"),
        content=stale_text, source_refs=(source("stale-parser-source", stale_text),),
        freshness="stale", status="active",
        metadata={"memory_role": "diagnostic_issue", "issue_state": "open"},
    ))
    try:
        pushed = memory.recall("review Hunter parser request boundary", k=8)
        assert [hit.path for hit in pushed] == [
            "@sliceagent/memory/records/open-parser-boundary.md",
        ]
        assert memory.recall("update the unrelated billing dashboard", k=8) == []

        # Auto-admission is the noise boundary; explicit discovery keeps the
        # historical reports available for a user who actually asks for them.
        explicit = memory.knowledge_repository.search(memory._query(
            "Hunter parser bug report", limit=100,
        ))
        assert {hit.record.id for hit in explicit}.issuperset({
            "resolved-report-0", "resolved-report-19",
        })
    finally:
        memory.close()


def test_revision_drift_moves_project_diagnostic_from_auto_push_to_explicit_pull(monkeypatch, tmp_path):
    from sliceagent.workspace_revision import WorkspaceRevision

    workspace = tmp_path / "hunter"
    target = workspace / "src" / "parser.py"
    target.parent.mkdir(parents=True)
    target.write_text("old parser\n", encoding="utf-8")
    observed = WorkspaceRevision.capture(str(workspace), ["src/parser.py"])

    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "state" / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="hunter", workspace_root=str(workspace))
    text = "Parser boundary remains unresolved at the observed dependency revision."
    item = KnowledgeRecord(
        id="revision-bound-parser", kind="fact", scopes=KnowledgeScope(project_id="hunter"),
        content=text, source_refs=(source("revision-bound-source", text),),
        freshness="current", status="active",
        metadata={
            "title": "Parser boundary issue", "memory_role": "diagnostic_issue",
            "issue_state": "open", "workspace_revision": observed.as_dict(),
        },
    )
    memory.knowledge_repository.put(item)
    try:
        assert [hit.path for hit in memory.recall("parser boundary unresolved")] == [
            "@sliceagent/memory/records/revision-bound-parser.md",
        ]
        target.write_text("new parser\n", encoding="utf-8")
        assert memory.recall("parser boundary unresolved") == []
        assert [hit.record.id for hit in memory.knowledge_repository.search(memory._query(
            "parser boundary unresolved",
        ))] == ["revision-bound-parser"]
    finally:
        memory.close()


def test_local_native_consolidation_is_idempotent_without_memem(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="project-hunter", workspace_id="hunter-main")
    memory.append_episode("session-1", "task-1", 1, {
        "title": "fix parser",
        "steps": [{"slice": "", "action": [{"name": "run_command", "args": {}, "failing": True}],
                   "observation": ["Error: parser exploded"]}],
        "note": "", "meta": {"failing": True, "stop_reason": "tool_use", "files": ["parser.py"]},
    })
    memory.append_episode("session-1", "task-1", 2, {
        "title": "fix parser",
        "steps": [{"slice": "", "action": [], "observation": ["ok"]}],
        "note": "fixed parser bounds",
        "meta": {"failing": False, "stop_reason": "end_turn", "files": ["parser.py"]},
    })
    try:
        assert memory.consolidate("session-1") == {
            "lessons": 1, "skills": 0, "skills_rejected": 0, "errors": 0,
        }
        assert memory.consolidate("session-1") == {
            "lessons": 0, "skills": 0, "skills_rejected": 0, "errors": 0,
        }
        records = memory.knowledge_records()
        assert len(records) == 1
        assert records[0].source_refs[0].namespace == "legacy-episodic-session"
        assert records[0].status is KnowledgeStatus.ACTIVE
    finally:
        memory.close()


def test_app_wide_episode_session_consolidates_each_project_under_its_captured_scope(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="project-b", workspace_id="workspace-b")
    for offset, (project_id, task_id, filename) in enumerate((
        ("project-a", "task-a", "alpha.py"),
        ("project-b", "task-b", "beta.py"),
    )):
        memory.append_episode("shared-app-session", task_id, offset * 2 + 1, {
            "title": f"fix {filename}",
            "steps": [{
                "slice": "",
                "action": [{"name": "run_command", "args": {}, "failing": True}],
                "observation": [f"Error: {filename} failed"],
            }],
            "note": "",
            "meta": {
                "project_id": project_id, "workspace_id": f"workspace-{offset}",
                "failing": True, "stop_reason": "tool_use", "files": [filename],
            },
        })
        memory.append_episode("shared-app-session", task_id, offset * 2 + 2, {
            "title": f"fix {filename}",
            "steps": [{"slice": "", "action": [], "observation": ["ok"]}],
            "note": f"fixed {filename} bounds",
            "meta": {
                "project_id": project_id, "workspace_id": f"workspace-{offset}",
                "failing": False, "stop_reason": "end_turn", "files": [filename],
            },
        })
    try:
        a_rows = memory.read_project_episodes(
            "shared-app-session", project_id="project-a",
        )
        assert {row["task_id"] for row in a_rows} == {"task-a"}
        assert memory.consolidate_for_project(
            "shared-app-session", project_id="project-a", workspace_id="workspace-a",
        )["lessons"] == 1
        assert memory.consolidate_for_project(
            "shared-app-session", project_id="project-b", workspace_id="workspace-b",
        )["lessons"] == 1
        for project_id, filename in (("project-a", "alpha.py"), ("project-b", "beta.py")):
            rows = memory.knowledge_repository.query(KnowledgeQuery(
                project_id=project_id, agent_id="sliceagent",
                statuses=(KnowledgeStatus.ACTIVE,),
            ))
            assert len(rows) == 1 and filename in rows[0].content
            assert rows[0].source_refs[0].project_id == project_id
    finally:
        memory.close()


def test_local_memory_without_memem_is_durable_scoped_and_closes_idempotently(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    evidence = source("native-source", "native evidence")

    first = LocalMemory(prefer_memem=False)
    assert first.knowledge_health()["memem"]["state"] == "disabled"
    assert isinstance(first, Memory)
    first.set_scope(project_id="project-hunter", workspace_id="hunter-main")
    first.append_episode("session-1", "task-1", 1, {
        "title": "native evidence", "steps": [], "meta": {},
    })
    first.knowledge_repository.put(KnowledgeRecord(
        id="hunter-fact", kind="fact", scopes=KnowledgeScope(project_id="project-hunter"),
        content="Hunter ContextFS memory route", source_refs=(evidence,), status="active",
    ))
    first.knowledge_repository.put(KnowledgeRecord(
        id="other-fact", kind="fact", scopes=KnowledgeScope(project_id="project-other"),
        content="Other ContextFS memory route", source_refs=(evidence,), status="active",
    ))
    assert [hit.path for hit in first.recall("ContextFS")] == [
        "@sliceagent/memory/records/hunter-fact.md",
    ]
    first.close()
    first.close()

    reopened = LocalMemory(prefer_memem=False)
    reopened.set_scope(project_id="project-hunter", workspace_id="hunter-main")
    try:
        assert reopened.read_episodes("session-1")[0]["task_id"] == "task-1"
        assert [hit.path for hit in reopened.recall("ContextFS")] == [
            "@sliceagent/memory/records/hunter-fact.md",
        ]
    finally:
        reopened.close()


def test_memory_status_reports_exact_legacy_inventory_and_persisted_consolidation(
    monkeypatch, tmp_path,
):
    vault = tmp_path / "vault"
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_VAULT", str(vault))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    for directory, names in {
        "tasks": ("a.md", "b.md"),
        "sessions": ("s.md",),
        "subagents": ("sub.jsonl",),
    }.items():
        target = vault / directory
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            (target / name).write_text("compatibility\n", encoding="utf-8")
    profile = vault / "roster" / "reviewer"
    profile.mkdir(parents=True)
    (profile / "profile.json").write_text("{}", encoding="utf-8")

    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="project-a", workspace_id="workspace-a")
    memory.append_episode("session-a", "task-a", 1, {
        "title": "ordinary turn", "steps": [], "note": "", "meta": {
            "project_id": "project-a", "failing": False, "stop_reason": "end_turn",
        },
    })
    try:
        initial = memory.memory_status()
        assert initial["legacy_inventory"] == {
            "task_projection_files": 2,
            "session_projection_files": 1,
            "episodic_session_files": 1,
            "subagent_archive_files": 1,
            "roster_profile_files": 1,
            "legacy_search_rows": 1,
        }
        assert initial["compatibility_transition"]["state"] == "retained"
        assert initial["last_consolidation"]["state"] == "not_recorded"
        memory.consolidate_for_project(
            "session-a", project_id="project-a", workspace_id="workspace-a",
        )
        latest = memory.memory_status()["last_consolidation"]
        assert latest["state"] == "no_eligible_output"
        assert latest["source_episode_count"] == 1
        assert memory.memory_status()["compatibility_transition"]["state"] == "retained"
    finally:
        memory.close()

    reopened = LocalMemory(prefer_memem=False)
    reopened.set_scope(project_id="project-a", workspace_id="workspace-a")
    try:
        assert reopened.memory_status()["last_consolidation"]["state"] == "no_eligible_output"
    finally:
        reopened.close()


def test_memory_status_never_calls_an_unreadable_legacy_inventory_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="project-a")
    monkeypatch.setattr(memory, "_legacy_inventory", lambda: {
        "episodic_session_files": None,
        "legacy_search_rows": None,
    })
    try:
        status = memory.memory_status()
        assert status["compatibility_transition"] == {
            "state": "unknown",
            "detail": "legacy compatibility inventory is incomplete",
        }
    finally:
        memory.close()


def test_memory_status_recognizes_historical_consolidation_output_without_run_metadata(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="project-a", workspace_id="workspace-a")
    ref = KnowledgeSourceRef.bind_text(
        "legacy-episodic-session", "session-before-status", "sealed legacy mirror",
        observer="sliceagent-host", observed_at=NOW, project_id="project-a",
        workspace_id="workspace-a",
    )
    memory.knowledge_repository.put(KnowledgeRecord(
        id="historical-derived", kind="fact",
        scopes=KnowledgeScope(project_id="project-a", agent_id="sliceagent"),
        content="A derived lesson predates consolidation status tracking.",
        source_refs=(ref,), authority="derived", proof_family="execution_outcome",
        observed_at=NOW, status="active",
    ))
    try:
        consolidation = memory.memory_status()["last_consolidation"]
        assert consolidation["state"] == "historical_output_present"
        assert "1 current-scope typed record(s)" in consolidation["detail"]
        assert "predates status tracking" in consolidation["detail"]
    finally:
        memory.close()


def test_native_knowledge_failure_is_visible_not_false_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.close()
    with pytest.raises(KnowledgeConflictError, match="closed"):
        memory.knowledge_records()
    assert memory.knowledge_health()["native"]["active"] is False


def test_native_seed_search_failure_marks_health_degraded_until_recovery(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    memory.set_scope(project_id="hunter")

    def broken(_query):
        raise OSError("simulated search failure")

    monkeypatch.setattr(memory._knowledge_index, "search", broken)
    try:
        assert memory.recall("parser") == []
        health = memory.knowledge_health()["native"]
        assert health["active"] is False
        assert health["state"] == "degraded"
        assert health["error"] == "OSError"

        monkeypatch.setattr(memory._knowledge_index, "search", memory.knowledge_repository.search)
        assert memory.recall("parser") == []
        assert memory.knowledge_health()["native"]["active"] is True
    finally:
        memory.close()


def test_native_repository_open_failure_does_not_disable_hippocampal_compatibility(monkeypatch, tmp_path):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_VAULT", str(tmp_path / "vault"))

    class BrokenRepository:
        def __init__(self, _path):
            raise OSError("simulated unavailable knowledge database")

    monkeypatch.setattr("sliceagent.memory.KnowledgeRepository", BrokenRepository)
    memory = LocalMemory(prefer_memem=False)
    try:
        assert memory.knowledge_health()["native"] == {
            "active": False, "backend": "native-unavailable", "error": "OSError",
        }
        with pytest.raises(KnowledgeConflictError, match=r"unavailable \(OSError\)"):
            memory.knowledge_records()
        memory.append_episode("session-1", "task-1", 1, {
            "title": "still durable", "steps": [], "meta": {},
        })
        assert memory.read_episodes("session-1")[0]["task_id"] == "task-1"
    finally:
        memory.close()


def test_compatibility_writer_failures_are_structured_and_block_retirement(monkeypatch, tmp_path):
    from sliceagent.runtime_persistence import LocalTurnStore

    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    core = LocalTurnStore(
        str(tmp_path / "workspace"), "session", store_root=str(tmp_path / "core"),
    )
    active = core.begin(task_id="task", logical_id="turn-1", user_request="check memory")
    core.seal(state={"status": "active"}, record={}, status="end_turn")
    original_open = builtins.open

    def failing_episode_open(path, mode="r", *args, **kwargs):
        if str(path).endswith(".jsonl") and "a" in mode:
            raise OSError("episode mirror unavailable")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_episode_open)
    memory.append_episode("session", "task", 1, {"title": "turn", "steps": []})

    def failing_atomic(*_args, **_kwargs):
        raise PermissionError("task projection unavailable")

    monkeypatch.setattr("sliceagent.memory._write_atomic", failing_atomic)
    memory.checkpoint_task(TaskState(task_id="task", session_id="session", title="work"))
    status = memory.memory_status()
    assert status["compatibility_health"]["state"] == "degraded"
    assert status["compatibility_health"]["channels"]["episodic_mirror"] == {
        "attempts": 1, "succeeded": 0, "failed": 1,
        "last_error": "OSError", "state": "degraded",
    }
    assert status["compatibility_health"]["channels"]["task_projection"]["last_error"] == (
        "PermissionError"
    )
    assert status["retirement_gate"]["ready"] is False
    assert status["retirement_gate"]["gates"]["compatibility_writes"] == "failed"
    assert core.coordinator.artifacts.get(active.artifact_id).status == "end_turn"
    core.close()
    memory.close()


def test_legacy_fts_failure_is_independent_from_episode_mirror_and_canonical_seal(monkeypatch, tmp_path):
    from sliceagent.runtime_persistence import LocalTurnStore

    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    core = LocalTurnStore(
        str(tmp_path / "workspace"), "session", store_root=str(tmp_path / "core"),
    )
    active = core.begin(task_id="task", logical_id="turn-1", user_request="check memory")
    core.seal(state={"status": "active"}, record={}, status="end_turn")

    class BrokenIndex:
        is_active = True

        def index_episode(self, **_kwargs):
            raise sqlite3.OperationalError("fts unavailable")

        def close(self):
            return None

    memory._idx = BrokenIndex()
    memory.append_episode("session", "task", 1, {"title": "turn", "steps": []})
    health = memory.memory_status()["compatibility_health"]
    assert health["channels"]["episodic_mirror"]["state"] == "healthy"
    assert health["channels"]["legacy_fts"] == {
        "attempts": 1, "succeeded": 0, "failed": 1,
        "last_error": "OperationalError", "state": "degraded",
    }
    assert memory.read_episodes("session")[0]["task_id"] == "task"
    assert core.coordinator.artifacts.get(active.artifact_id).status == "end_turn"
    core.close()
    memory.close()


def test_compatibility_retirement_requires_explicit_equivalence_and_never_deletes(tmp_path, monkeypatch):
    monkeypatch.setenv("SLICEAGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SLICEAGENT_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    memory = LocalMemory(prefer_memem=False)
    proof = {
        "canonical_l0_equivalence": "passed",
        "canonical_l1_equivalence": "passed",
        "canonical_l2_equivalence": "passed",
        "legacy_read_fallback": "passed",
        "compatibility_writes": "passed",
    }
    memory.knowledge_repository.set_runtime_metadata(
        "compatibility-retirement:global", proof, updated_at=NOW,
    )
    gate = memory.memory_status()["retirement_gate"]
    assert gate["state"] == "ready" and gate["ready"] is True
    assert gate["automatic_deletion"] is False
    assert set(gate["gates"].values()) == {"passed"}
    memory.close()
